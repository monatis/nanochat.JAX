"""
Combined MuonAdamW optimizer for JAX, using optax.

Implements the same optimization strategy as the PyTorch version:
- AdamW for embeddings, lm_head, and scalar parameters
- Muon (with Polar Express orthogonalization + NorMuon variance reduction) for 2D matrix params

The JAX version is much simpler than the PyTorch version because:
1. jax.jit replaces torch.compile — no fused kernel boilerplate
2. JAX SPMD replaces manual DDP — no reduce_scatter/all_gather code
3. optax provides a clean GradientTransformation interface

Adapted from: https://github.com/KellerJordan/modded-nanogpt
References:
- Polar Express: https://arxiv.org/pdf/2505.16932
- NorMuon: https://arxiv.org/pdf/2510.05491
"""

import jax
import jax.numpy as jnp
import optax

from nanochat.common import COMPUTE_DTYPE

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _polar_express(X, ns_steps=5):
    """
    Polar Express orthogonalization: approximate the unitary polar factor of X.
    Works on 2D matrices (or batched 2D matrices via vmap).
    """
    # Normalize
    norm = jnp.sqrt(jnp.sum(jnp.square(X), axis=(-2, -1), keepdims=True))
    X = X / (norm * 1.01 + 1e-6)

    tall = X.shape[-2] >= X.shape[-1]

    def _step_tall(X, coeffs):
        a, b, c = coeffs
        A = X.swapaxes(-2, -1) @ X
        B = b * A + c * (A @ A)
        return a * X + X @ B

    def _step_wide(X, coeffs):
        a, b, c = coeffs
        A = X @ X.swapaxes(-2, -1)
        B = b * A + c * (A @ A)
        return a * X + B @ X

    coeffs_to_use = POLAR_EXPRESS_COEFFS[:ns_steps]
    for coeffs in coeffs_to_use:
        X = jax.lax.cond(tall, _step_tall, _step_wide, X, jnp.array(coeffs))

    return X


def _normuon_variance_reduction(g, second_moment, beta2):
    """
    NorMuon variance reduction: per-neuron/column adaptive learning rate
    that normalizes update scales after orthogonalization.
    """
    # Determine reduction dimension: reduce over the smaller dimension
    tall = g.shape[-2] >= g.shape[-1]
    red_dim = -1 if tall else -2

    # Compute per-neuron variance
    v_mean = jnp.mean(jnp.square(g.astype(jnp.float32)), axis=red_dim, keepdims=True)
    red_dim_size = g.shape[red_dim]
    v_norm_sq = jnp.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
    v_norm = jnp.sqrt(v_norm_sq)

    # EMA update of second moment
    new_second_moment = second_moment * beta2 + v_mean.astype(second_moment.dtype) * (
        1 - beta2
    )

    # Compute per-neuron step size
    step_size = jax.lax.rsqrt(jnp.maximum(new_second_moment, 1e-10))

    # Rescale to preserve overall norm
    scaled_sq_sum = (v_mean * red_dim_size) * jnp.square(step_size.astype(jnp.float32))
    v_norm_new = jnp.sqrt(jnp.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))

    g = g * final_scale.astype(g.dtype)
    return g, new_second_moment


def muon(
    learning_rate: float = 0.02,
    momentum: float = 0.95,
    ns_steps: int = 5,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """
    Muon optimizer as an optax GradientTransformation.

    Implements: Nesterov momentum -> Polar Express orthogonalization ->
    NorMuon variance reduction -> Cautious weight decay + update.

    This is a custom implementation matching the nanochat PyTorch version exactly,
    including Polar Express coefficients and NorMuon variance reduction.
    """

    def init_fn(params):
        # Momentum buffer and factored second moment per parameter
        momentum_buf = jax.tree.map(jnp.zeros_like, params)

        def _make_second_moment(p):
            # Factored: per-row if tall, per-column if wide
            if p.ndim < 2:
                return jnp.zeros((1,), dtype=p.dtype)
            if p.shape[-2] >= p.shape[-1]:
                shape = p.shape[:-1] + (1,)
            else:
                shape = p.shape[:-2] + (1, p.shape[-1])
            return jnp.zeros(shape, dtype=p.dtype)

        second_moment = jax.tree.map(_make_second_moment, params)
        return (momentum_buf, second_moment)

    def update_fn(updates, state, params=None):
        momentum_buf, second_moment = state

        def _muon_update(grad, mom_buf, sec_mom, param):
            if grad.ndim < 2:
                # Fallback: for non-2D params, just return gradient as-is
                return grad, mom_buf, sec_mom

            # Nesterov momentum
            new_mom = mom_buf * momentum + grad * (1 - momentum)
            g = grad * (1 - momentum) + new_mom * momentum  # Nesterov lookahead

            # Polar Express orthogonalization (cast to bf16 for speed if available)
            g_orth = g.astype(COMPUTE_DTYPE) if COMPUTE_DTYPE == jnp.bfloat16 else g
            g_orth = _polar_express(g_orth, ns_steps)

            # NorMuon variance reduction
            g_orth, new_sec_mom = _normuon_variance_reduction(g_orth, sec_mom, beta2)

            # Scale lr by aspect ratio
            aspect_scale = jnp.maximum(1.0, (grad.shape[-2] / grad.shape[-1]) ** 0.5)
            lr_scaled = learning_rate * aspect_scale

            # Cautious weight decay: only decay weights aligned with the update direction
            if param is not None and weight_decay > 0:
                mask = (g_orth * param) >= 0
                wd_term = lr_scaled * weight_decay * param * mask
            else:
                wd_term = jnp.zeros_like(g_orth)

            update = -lr_scaled * g_orth - wd_term
            return update, new_mom, new_sec_mom

        # 1. Map over the tree. This returns a single PyTree where every leaf
        # is a 3-element tuple: (update, new_mom, new_sec_mom)
        results_tree = jax.tree.map(
            _muon_update,
            updates,
            momentum_buf,
            second_moment,
            params if params is not None else jax.tree.map(jnp.zeros_like, updates),
        )

        # 2. Tell JAX not to traverse inside our 3-element result tuples
        is_result_tuple = lambda x: isinstance(x, tuple) and len(x) == 3

        # 3. Unzip the tree of tuples into three separate PyTrees
        new_updates = jax.tree.map(
            lambda x: x[0], results_tree, is_leaf=is_result_tuple
        )
        new_mom_buf = jax.tree.map(
            lambda x: x[1], results_tree, is_leaf=is_result_tuple
        )
        new_sec_mom = jax.tree.map(
            lambda x: x[2], results_tree, is_leaf=is_result_tuple
        )

        return new_updates, (new_mom_buf, new_sec_mom)

    return optax.GradientTransformation(init_fn, update_fn)


# --- Parameter grouping utilities ---


def classify_param(path: str, param) -> str:
    """
    Classify a parameter into an optimizer group based on its path in the model.
    Returns one of: 'muon', 'embed', 'lm_head', 'scalars'
    """
    # Embedding layers -> adamw
    if "wte" in path or "value_embeds" in path:
        return "embed"
    # LM head -> adamw
    if "lm_head" in path:
        return "lm_head"
    # 2D weight matrices in transformer blocks -> muon
    if param.ndim == 2 and "blocks" in path:
        return "muon"
    # Everything else (scalars, 1D, gates, lambdas) -> adamw/scalars
    return "scalars"


def build_optimizer(model, config):
    """
    Build a combined Muon+AdamW optimizer for the GPT model.

    Args:
        model: GPT model (nnx.Module)
        config: dict with optimizer hyperparams:
            - muon_lr, muon_momentum, muon_ns_steps, muon_beta2, muon_wd
            - adamw_embed_lr, adamw_embed_betas, adamw_embed_eps, adamw_embed_wd
            - adamw_lm_head_lr, adamw_lm_head_betas, adamw_lm_head_eps, adamw_lm_head_wd
            - adamw_scalars_lr, adamw_scalars_betas, adamw_scalars_eps, adamw_scalars_wd
    """
    import jax
    import optax
    from flax import nnx

    # Assuming muon and classify_param are available in your namespace
    # from nanochat.optim import muon, classify_param

    # Default hyperparams (matching PyTorch nanochat defaults)
    c = {
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "muon_beta2": 0.999,
        "muon_wd": 0.0,
        "adamw_embed_lr": 0.3,
        "adamw_embed_betas": (0.8, 0.95),
        "adamw_embed_eps": 1e-10,
        "adamw_embed_wd": 0.0,
        "adamw_lm_head_lr": 0.008,
        "adamw_lm_head_betas": (0.8, 0.95),
        "adamw_lm_head_eps": 1e-10,
        "adamw_lm_head_wd": 0.0,
        "adamw_scalars_lr": 0.5,
        "adamw_scalars_betas": (0.8, 0.95),
        "adamw_scalars_eps": 1e-10,
        "adamw_scalars_wd": 0.0,
    }
    c.update(config)

    # Build the multi-transform optimizer
    transforms = {
        "muon": muon(
            learning_rate=c["muon_lr"],
            momentum=c["muon_momentum"],
            ns_steps=c["muon_ns_steps"],
            beta2=c["muon_beta2"],
            weight_decay=c["muon_wd"],
        ),
        "embed": optax.adamw(
            learning_rate=c["adamw_embed_lr"],
            b1=c["adamw_embed_betas"][0],
            b2=c["adamw_embed_betas"][1],
            eps=c["adamw_embed_eps"],
            weight_decay=c["adamw_embed_wd"],
        ),
        "lm_head": optax.adamw(
            learning_rate=c["adamw_lm_head_lr"],
            b1=c["adamw_lm_head_betas"][0],
            b2=c["adamw_lm_head_betas"][1],
            eps=c["adamw_lm_head_eps"],
            weight_decay=c["adamw_lm_head_wd"],
        ),
        "scalars": optax.adamw(
            learning_rate=c["adamw_scalars_lr"],
            b1=c["adamw_scalars_betas"][0],
            b2=c["adamw_scalars_betas"][1],
            eps=c["adamw_scalars_eps"],
            weight_decay=c["adamw_scalars_wd"],
        ),
    }

    # Build the label function that maps the ENTIRE params tree to a tree of labels
    def map_labels(params_tree):

        def get_label_for_leaf(path, leaf):
            # JAX paths are tuples of keys. We extract the string name from each key
            # to reconstruct the dotted path string (e.g., "blocks.0.attn.c_q.kernel")
            path_str = ".".join(
                str(p.key) if hasattr(p, "key") else str(p) for p in path
            )
            return classify_param(path_str, leaf)

        # Apply our leaf logic over the PyTree
        return jax.tree_util.tree_map_with_path(get_label_for_leaf, params_tree)

    tx = optax.multi_transform(transforms, map_labels)
    return tx
