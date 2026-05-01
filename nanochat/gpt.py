"""
GPT model in JAX/Flax NNX (rewrite from PyTorch, keeping same architecture)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Splash Attention on TPU via jax.nn.dot_product_attention
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.common import print0, COMPUTE_DTYPE


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def rms_norm(x):
    """RMSNorm without learnable params — runs in compute dtype."""
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)


class Linear(nnx.Module):
    """Linear layer without bias. Weights stored in fp32 for optimizer precision,
    cast to compute dtype in forward pass."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        # Initialize with zeros — actual init done by GPT.init_weights()
        self.kernel = nnx.Param(
            jnp.zeros((out_features, in_features), dtype=jnp.float32)
        )
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        w = self.kernel.value.astype(x.dtype)
        return x @ w.T


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def precompute_rotary_embeddings(seq_len, head_dim, base=100000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(COMPUTE_DTYPE)
    sin = jnp.sin(freqs).astype(COMPUTE_DTYPE)
    # Add batch and head dims for broadcasting: (1, seq_len, 1, head_dim/2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=3)


def _compute_window_sizes(config):
    """
    Compute per-layer window sizes for sliding window attention.
    Returns list of ints: -1 means full context, positive int means window size.
    """
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), (
        f"Invalid window_pattern: {pattern}. Use only S and L."
    )
    long_window = config.sequence_len
    short_window = -(-long_window // 4 // 128) * 128  # ceil to tile size
    char_to_window = {
        "L": -1,  # full context
        "S": short_window,
    }
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    # Final layer always gets full context
    window_sizes[-1] = -1
    return window_sizes


def _make_sliding_window_mask(T, window_size):
    """Create a (T, T) causal mask with optional sliding window.
    True = attend, False = mask out.
    If window_size < 0, returns a simple causal mask (full context).
    """
    row_idx = jnp.arange(T)[:, None]
    col_idx = jnp.arange(T)[None, :]
    # Causal: can only attend to current and past positions
    mask = col_idx <= row_idx
    # Sliding window: also restrict how far back we attend
    if window_size > 0:
        mask = mask & ((row_idx - col_idx) <= window_size)
    return mask


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, rngs=rngs)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, rngs=rngs)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, rngs=rngs)
        self.c_proj = Linear(self.n_embd, self.n_embd, rngs=rngs)
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, rngs=rngs)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def __call__(self, x, ve, cos_sin, window_size):
        B, T, C = x.shape

        # Project to get queries, keys, values: (B, T, H, D)
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * jax.nn.sigmoid(
                self.ve_gate(x[..., : self.ve_gate_channels])
            )  # (B, T, n_kv_head)
            v = v + gate[..., None] * ve

        # Apply Rotary Embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)  # QK norm
        q = q * 1.2  # sharper attention
        k = k * 1.2

        # Handle GQA: repeat k,v heads to match q heads
        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Build attention mask & calculate attention
        # Note: q, k, v remain in (B, T, H, D) format natively for JAX
        if window_size < 0:
            # Full causal attention
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        else:
            # Sliding window: need explicit mask
            mask = _make_sliding_window_mask(T, window_size)
            # Explicitly shape bias to (1, 1, T, T) for broadcasting with (B, H, T, T) scores
            bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min)
            bias = jnp.expand_dims(bias, (0, 1))
            y = jax.nn.dot_product_attention(q, k, v, bias=bias)

        # y is returned as (B, T, H, D) — just reshape to (B, T, C)
        y = y.reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jnp.square(jax.nn.relu(x))  # relu^2
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x, ve, cos_sin, window_size):
        x = x + self.attn(rms_norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nnx.Module):
    def __init__(
        self, config: GPTConfig, *, rngs: nnx.Rngs, pad_vocab_size_to: int = 64
    ):
        self.config = config

        # Compute per-layer window sizes
        self.window_sizes = _compute_window_sizes(config)

        # Pad vocab for efficiency (tensor cores / MXU alignment)
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.padded_vocab_size = padded_vocab_size

        # Transformer blocks
        self.wte = nnx.Embed(
            num_embeddings=padded_vocab_size, features=config.n_embd, rngs=rngs
        )
        self.blocks = [Block(config, i, rngs=rngs) for i in range(config.n_layer)]
        self.lm_head = Linear(config.n_embd, padded_vocab_size, rngs=rngs)

        # Per-layer learnable scalars
        self.resid_lambdas = nnx.Param(jnp.ones(config.n_layer))
        self.x0_lambdas = nnx.Param(jnp.zeros(config.n_layer))

        # Smear: mix previous token's embedding into current token
        self.smear_gate = Linear(24, 1, rngs=rngs)
        self.smear_lambda = nnx.Param(jnp.zeros(1))

        # Backout: subtract cached mid-layer residual before final norm
        self.backout_lambda = nnx.Param(0.2 * jnp.ones(1))

        # Value embeddings (ResFormer-style)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nnx.Embed(
                num_embeddings=padded_vocab_size, features=kv_dim, rngs=rngs
            )
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        }

        # Precompute rotary embeddings (large enough for 10X the sequence length)
        rotary_seq_len = config.sequence_len * 10
        cos, sin = precompute_rotary_embeddings(rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

        # Now init all the weights properly
        self.init_weights()

    def init_weights(self):
        """
        Initialize all model weights.

        wte (embedding):     normal, std=0.8
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd) * 0.4
            mlp.c_proj:      zeros
        """
        key = jax.random.key(42)

        # Embedding and unembedding
        key, k1, k2 = jax.random.split(key, 3)
        self.wte.embedding.value = (
            jax.random.normal(k1, self.wte.embedding.value.shape) * 0.8
        )
        self.lm_head.kernel.value = (
            jax.random.normal(k2, self.lm_head.kernel.value.shape) * 0.001
        )

        # Transformer blocks
        n_embd = self.config.n_embd
        s = (
            3**0.5 * n_embd**-0.5
        )  # sqrt(3) * 1/sqrt(n_embd) for Uniform matching Normal std

        for block in self.blocks:
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            block.attn.c_q.kernel.value = jax.random.uniform(
                k1, block.attn.c_q.kernel.value.shape, minval=-s, maxval=s
            )
            block.attn.c_k.kernel.value = jax.random.uniform(
                k2, block.attn.c_k.kernel.value.shape, minval=-s, maxval=s
            )
            block.attn.c_v.kernel.value = jax.random.uniform(
                k3, block.attn.c_v.kernel.value.shape, minval=-s, maxval=s
            )
            block.attn.c_proj.kernel.value = jnp.zeros_like(
                block.attn.c_proj.kernel.value
            )
            block.mlp.c_fc.kernel.value = jax.random.uniform(
                k4, block.mlp.c_fc.kernel.value.shape, minval=-s * 0.4, maxval=s * 0.4
            )
            block.mlp.c_proj.kernel.value = jnp.zeros_like(
                block.mlp.c_proj.kernel.value
            )

        # Per-layer scalars
        n_layer = self.config.n_layer
        resid = jnp.array(
            [1.15 - (0.10 * i / max(n_layer - 1, 1)) for i in range(n_layer)]
        )
        x0 = jnp.array(
            [0.20 - (0.15 * i / max(n_layer - 1, 1)) for i in range(n_layer)]
        )
        self.resid_lambdas.value = resid
        self.x0_lambdas.value = x0

        # Smear/backout
        key, k1 = jax.random.split(key)
        self.smear_lambda.value = jnp.zeros(1)
        self.backout_lambda.value = 0.2 * jnp.ones(1)
        self.smear_gate.kernel.value = jax.random.uniform(
            k1, self.smear_gate.kernel.value.shape, minval=0.0, maxval=0.02
        )

        # Value embeddings
        for ve in self.value_embeds.values():
            key, k1 = jax.random.split(key)
            ve.embedding.value = jax.random.uniform(
                k1, ve.embedding.value.shape, minval=-s, maxval=s
            )

        # Gate weights
        for block in self.blocks:
            if block.attn.ve_gate is not None:
                key, k1 = jax.random.split(key)
                block.attn.ve_gate.kernel.value = jax.random.uniform(
                    k1, block.attn.ve_gate.kernel.value.shape, minval=0.0, maxval=0.02
                )

        # Cast embeddings to COMPUTE_DTYPE to save memory (same as PyTorch version)
        self.wte.embedding.value = self.wte.embedding.value.astype(COMPUTE_DTYPE)
        for ve in self.value_embeds.values():
            ve.embedding.value = ve.embedding.value.astype(COMPUTE_DTYPE)

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 6 FLOPs per token.
        Attention adds 12 * h * q * effective_seq_len per layer.
        """
        # Count all kernel params
        nparams = 0
        for block in self.blocks:
            for linear in [
                block.attn.c_q,
                block.attn.c_k,
                block.attn.c_v,
                block.attn.c_proj,
                block.mlp.c_fc,
                block.mlp.c_proj,
            ]:
                nparams += linear.kernel.value.size
            if block.attn.ve_gate is not None:
                nparams += block.attn.ve_gate.kernel.value.size
        nparams += self.lm_head.kernel.value.size
        # Add smear gate
        nparams += self.smear_gate.kernel.value.size

        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        attn_flops = 0
        for window_size in self.window_sizes:
            effective_seq = t if window_size < 0 else min(window_size, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * nparams + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte_count = self.wte.embedding.value.size
        ve_count = sum(ve.embedding.value.size for ve in self.value_embeds.values())
        lm_head_count = self.lm_head.kernel.value.size
        transformer_count = sum(
            sum(
                l.kernel.value.size
                for l in [
                    b.attn.c_q,
                    b.attn.c_k,
                    b.attn.c_v,
                    b.attn.c_proj,
                    b.mlp.c_fc,
                    b.mlp.c_proj,
                ]
            )
            + (b.attn.ve_gate.kernel.value.size if b.attn.ve_gate is not None else 0)
            for b in self.blocks
        )
        scalars_count = (
            self.resid_lambdas.value.size
            + self.x0_lambdas.value.size
            + self.smear_gate.kernel.value.size
            + self.smear_lambda.value.size
            + self.backout_lambda.value.size
        )
        total = wte_count + ve_count + lm_head_count + transformer_count + scalars_count
        return {
            "wte": wte_count,
            "value_embeds": ve_count,
            "lm_head": lm_head_count,
            "transformer_matrices": transformer_count,
            "scalars": scalars_count,
            "total": total,
        }

    def __call__(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.shape

        # Grab rotary embeddings for current sequence length
        assert T <= self.cos.shape[1], (
            f"Sequence length {T} exceeds rotary cache {self.cos.shape[1]}"
        )
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embed the tokens
        x = self.wte(idx)
        x = x.astype(COMPUTE_DTYPE)
        x = rms_norm(x)

        # Smear: mix previous token's embedding into current position
        assert T > 1, "Forward pass should have T > 1"
        gate = self.smear_lambda.value.astype(x.dtype) * jax.nn.sigmoid(
            self.smear_gate(x[:, 1:, :24])
        )
        x = jnp.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)

        # Forward the transformer blocks
        x0 = x  # save initial embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas.value[i] * x + self.x0_lambdas.value[i] * x0
            ve = (
                self.value_embeds[str(i)](idx).astype(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x

        # Subtract mid-layer residual
        if x_backout is not None:
            x = x - self.backout_lambda.value.astype(x.dtype) * x_backout
        x = rms_norm(x)

        # Compute logits
        softcap = 15  # smoothly cap logits to [-softcap, softcap]
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]  # remove padding
        logits = logits.astype(jnp.float32)  # switch to fp32 for softcap and loss
        logits = softcap * jnp.tanh(logits / softcap)

        if targets is not None:
            # Training: compute cross-entropy loss
            # one-hot targets for cross-entropy
            one_hot = jax.nn.one_hot(targets, self.config.vocab_size)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

            if loss_reduction == "mean":
                # Mask out ignore_index (-1) targets
                valid = targets >= 0
                loss_per_token = -jnp.sum(one_hot * log_probs, axis=-1)
                loss = jnp.sum(loss_per_token * valid) / jnp.maximum(jnp.sum(valid), 1)
                return loss
            else:  # 'none'
                loss_per_token = -jnp.sum(one_hot * log_probs, axis=-1)
                return loss_per_token
        else:
            return logits

    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        Assumes batch size 1, tokens is a Python list of ints.
        Returns a generator yielding one token at a time.
        """
        assert isinstance(tokens, list)
        key = jax.random.key(seed)
        ids = jnp.array([tokens], dtype=jnp.int32)

        for _ in range(max_tokens):
            logits = self(ids)  # (1, T, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size)
            if top_k is not None and top_k > 0:
                top_vals, top_idx = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                # Mask out everything below top_k
                logits = jnp.full_like(logits, -jnp.inf)
                logits = logits.at[0, top_idx[0]].set(top_vals[0])
            if temperature > 0:
                logits = logits / temperature
                key, subkey = jax.random.split(key)
                next_ids = jax.random.categorical(subkey, logits, axis=-1)  # (1,)
                next_ids = next_ids[:, None]  # (1, 1)
            else:
                next_ids = jnp.argmax(logits, axis=-1, keepdims=True)  # (1, 1)
            ids = jnp.concatenate([ids, next_ids], axis=1)
            token = int(next_ids[0, 0])
            yield token
