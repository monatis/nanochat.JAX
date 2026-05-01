"""
Test attention implementations (JAX edition).

Validates that the attention primitives used in nanochat's GPT model
produce correct outputs: causal masking, sliding window, GQA, and gradients.

Run: python -m pytest tests/test_attention_fallback.py -v -s
"""
import jax
import jax.numpy as jnp
import pytest

from nanochat.gpt import (
    _make_sliding_window_mask,
    apply_rotary_emb,
    precompute_rotary_embeddings,
    rms_norm,
)


def assert_close(a, b, name, atol=1e-2, rtol=1e-2):
    """Assert two JAX arrays are close, with a helpful error message."""
    diff = jnp.abs(a - b)
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert jnp.allclose(a, b, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# Sliding Window Mask tests
# =============================================================================
class TestSlidingWindowMask:
    """Test the sliding window mask construction."""

    def test_causal_mask_basic(self):
        """Full-context causal mask: lower-triangular True."""
        T = 8
        mask = _make_sliding_window_mask(T, window_size=-1)
        # Should be lower-triangular
        expected = jnp.tril(jnp.ones((T, T), dtype=bool))
        assert jnp.array_equal(mask, expected), "Causal mask should be lower triangular"

    def test_sliding_window_restricts_attention(self):
        """Sliding window should restrict how far back a token can attend."""
        T = 16
        window = 4
        mask = _make_sliding_window_mask(T, window_size=window)

        # Token at position 10 should attend to positions 6..10 (window=4)
        for row in range(T):
            for col in range(T):
                if col <= row and (row - col) <= window:
                    assert mask[row, col], f"mask[{row},{col}] should be True"
                else:
                    assert not mask[row, col], f"mask[{row},{col}] should be False"

    def test_window_size_1_is_diagonal(self):
        """Window size 0 means only attend to self (current position)."""
        T = 8
        mask = _make_sliding_window_mask(T, window_size=0)
        expected = jnp.eye(T, dtype=bool)
        assert jnp.array_equal(mask, expected), "Window=0 should be identity (self-attention only)"

    def test_large_window_equals_causal(self):
        """Window >= T should be equivalent to full causal mask."""
        T = 16
        mask_windowed = _make_sliding_window_mask(T, window_size=T)
        mask_causal = _make_sliding_window_mask(T, window_size=-1)
        assert jnp.array_equal(mask_windowed, mask_causal), \
            "Window >= T should produce the same mask as full causal"


# =============================================================================
# dot_product_attention tests
# =============================================================================
class TestDotProductAttention:
    """Test jax.nn.dot_product_attention used in CausalSelfAttention."""

    DTYPE = jnp.bfloat16

    def test_causal_basic(self):
        """Basic causal attention produces valid output shape."""
        B, H, T, D = 2, 4, 64, 32
        key = jax.random.key(0)
        q = jax.random.normal(jax.random.fold_in(key, 0), (B, H, T, D), dtype=self.DTYPE)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D), dtype=self.DTYPE)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D), dtype=self.DTYPE)

        y = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        assert y.shape == (B, H, T, D), f"Expected shape {(B, H, T, D)}, got {y.shape}"
        assert not jnp.any(jnp.isnan(y)), "Output contains NaN"

    def test_sliding_window_with_bias(self):
        """Sliding window via bias mask produces valid output."""
        B, H, T, D = 2, 4, 64, 32
        window = 16
        key = jax.random.key(1)
        q = jax.random.normal(jax.random.fold_in(key, 0), (B, H, T, D), dtype=self.DTYPE)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D), dtype=self.DTYPE)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D), dtype=self.DTYPE)

        mask = _make_sliding_window_mask(T, window)
        bias = jnp.where(mask[None, None, :, :], 0.0, jnp.finfo(self.DTYPE).min)
        y = jax.nn.dot_product_attention(q, k, v, bias=bias)

        assert y.shape == (B, H, T, D)
        assert not jnp.any(jnp.isnan(y)), "Output contains NaN"

    def test_sliding_window_differs_from_full_context(self):
        """Sliding window attention should produce different results than full context."""
        B, H, T, D = 1, 2, 32, 16
        window = 8
        key = jax.random.key(2)
        q = jax.random.normal(jax.random.fold_in(key, 0), (B, H, T, D), dtype=jnp.float32)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D), dtype=jnp.float32)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D), dtype=jnp.float32)

        # Full causal
        y_full = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        # Sliding window
        mask = _make_sliding_window_mask(T, window)
        bias = jnp.where(mask[None, None, :, :], 0.0, jnp.finfo(jnp.float32).min)
        y_window = jax.nn.dot_product_attention(q, k, v, bias=bias)

        # They should differ (except for early tokens within window)
        # At position >= window, the outputs should diverge
        late_full = y_full[:, :, window+1:, :]
        late_window = y_window[:, :, window+1:, :]
        assert not jnp.allclose(late_full, late_window, atol=1e-3), \
            "Sliding window should differ from full causal for positions beyond the window"

    def test_gqa_head_expansion(self):
        """GQA: fewer KV heads than Q heads via jnp.repeat."""
        B, T, D = 2, 32, 16
        n_heads = 8
        n_kv_heads = 2
        key = jax.random.key(3)

        q = jax.random.normal(jax.random.fold_in(key, 0), (B, n_heads, T, D), dtype=self.DTYPE)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, n_kv_heads, T, D), dtype=self.DTYPE)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, n_kv_heads, T, D), dtype=self.DTYPE)

        # Repeat KV heads to match Q heads (how GPT model does it)
        repeats = n_heads // n_kv_heads
        k_expanded = jnp.repeat(k, repeats, axis=1)
        v_expanded = jnp.repeat(v, repeats, axis=1)

        assert k_expanded.shape == (B, n_heads, T, D)
        y = jax.nn.dot_product_attention(q, k_expanded, v_expanded, is_causal=True)

        assert y.shape == (B, n_heads, T, D)
        assert not jnp.any(jnp.isnan(y)), "GQA output contains NaN"

    def test_gradients_flow(self):
        """Verify gradients flow through attention."""
        B, H, T, D = 2, 4, 32, 16
        key = jax.random.key(4)
        q = jax.random.normal(jax.random.fold_in(key, 0), (B, H, T, D), dtype=jnp.float32)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D), dtype=jnp.float32)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D), dtype=jnp.float32)

        def attn_loss(q, k, v):
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
            return jnp.sum(y)

        grads = jax.grad(attn_loss, argnums=(0, 1, 2))(q, k, v)
        q_grad, k_grad, v_grad = grads

        assert q_grad.shape == q.shape, "q gradient shape mismatch"
        assert k_grad.shape == k.shape, "k gradient shape mismatch"
        assert v_grad.shape == v.shape, "v gradient shape mismatch"
        assert not jnp.any(jnp.isnan(q_grad)), "NaN in q gradient"
        assert not jnp.any(jnp.isnan(k_grad)), "NaN in k gradient"
        assert not jnp.any(jnp.isnan(v_grad)), "NaN in v gradient"

    def test_causal_masking_correctness(self):
        """Verify causal masking: future tokens should not influence past outputs."""
        B, H, T, D = 1, 1, 8, 4
        key = jax.random.key(5)
        q = jax.random.normal(jax.random.fold_in(key, 0), (B, H, T, D), dtype=jnp.float32)
        k = jax.random.normal(jax.random.fold_in(key, 1), (B, H, T, D), dtype=jnp.float32)
        v = jax.random.normal(jax.random.fold_in(key, 2), (B, H, T, D), dtype=jnp.float32)

        y_full = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        # Now zero out future K/V (positions > 4) — output at positions 0..4 should be unchanged
        k_truncated = k.at[:, :, 5:, :].set(0.0)
        v_truncated = v.at[:, :, 5:, :].set(0.0)
        y_truncated = jax.nn.dot_product_attention(q, k_truncated, v_truncated, is_causal=True)

        # Positions 0..4 should be identical (causal = can't see positions 5+)
        assert_close(
            y_full[:, :, :5, :], y_truncated[:, :, :5, :],
            "causal_mask_correctness", atol=1e-5, rtol=1e-5
        )


# =============================================================================
# Rotary Embedding tests
# =============================================================================
class TestRotaryEmbeddings:
    """Test rotary position embedding implementation."""

    def test_shapes(self):
        """Rotary embeddings have correct output shapes."""
        seq_len = 128
        head_dim = 64
        cos, sin = precompute_rotary_embeddings(seq_len, head_dim)
        assert cos.shape == (1, seq_len, 1, head_dim // 2)
        assert sin.shape == (1, seq_len, 1, head_dim // 2)

    def test_apply_preserves_shape(self):
        """apply_rotary_emb preserves input shape."""
        B, T, H, D = 2, 32, 4, 64
        cos, sin = precompute_rotary_embeddings(T, D)
        x = jax.random.normal(jax.random.key(0), (B, T, H, D))
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape

    def test_different_positions_get_different_embeddings(self):
        """Different sequence positions should produce different rotary outputs."""
        B, T, H, D = 1, 16, 1, 32
        cos, sin = precompute_rotary_embeddings(T, D)
        x = jnp.ones((B, T, H, D))  # Same input at every position
        y = apply_rotary_emb(x, cos, sin)
        # Output at position 0 should differ from position 8
        assert not jnp.allclose(y[0, 0], y[0, 8], atol=1e-3), \
            "Rotary embeddings should produce different outputs at different positions"

    def test_orthogonality(self):
        """Rotary embedding should approximately preserve vector norms."""
        B, T, H, D = 1, 32, 1, 64
        cos, sin = precompute_rotary_embeddings(T, D)
        x = jax.random.normal(jax.random.key(0), (B, T, H, D))
        y = apply_rotary_emb(x, cos, sin)
        # Norms should be approximately preserved (rotary is a rotation)
        x_norms = jnp.linalg.norm(x, axis=-1)
        y_norms = jnp.linalg.norm(y, axis=-1)
        assert_close(x_norms, y_norms, "rotary_norm_preservation", atol=1e-5, rtol=1e-5)


# =============================================================================
# RMSNorm tests
# =============================================================================
class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_output_scale(self):
        """RMSNorm should normalize the RMS to approximately 1."""
        x = jax.random.normal(jax.random.key(0), (2, 32, 256))
        y = rms_norm(x)
        # After RMSNorm, RMS of each vector should be ~1
        rms = jnp.sqrt(jnp.mean(jnp.square(y), axis=-1))
        assert_close(rms, jnp.ones_like(rms), "rms_norm_output", atol=1e-3, rtol=1e-3)

    def test_gradient_flows(self):
        """Gradients should flow through RMSNorm."""
        x = jax.random.normal(jax.random.key(0), (2, 32, 64))
        grad_fn = jax.grad(lambda x: jnp.sum(rms_norm(x)))
        g = grad_fn(x)
        assert g.shape == x.shape
        assert not jnp.any(jnp.isnan(g)), "NaN in RMSNorm gradient"


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    pytest.main([__file__, "-v", "-s"])
