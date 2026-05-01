"""
Functions for evaluating a base model (JAX/TPU edition).
"""
import math
import jax
import jax.numpy as jnp


def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Compute bits-per-byte (bpb) — a tokenization-independent metric.

    Instead of average loss, we compute sum(loss * token_bytes) / sum(token_bytes)
    which normalizes by the byte length of each token. Special tokens (byte count = 0)
    are excluded from the metric.

    Args:
        model: GPT model
        batches: iterator yielding (x, y) batches
        steps: number of evaluation steps
        token_bytes: 1D array of shape (vocab_size,) with byte count per token id
    """
    total_nats = 0.0
    total_bytes = 0

    token_bytes = jnp.array(token_bytes, dtype=jnp.int32)
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        x = jnp.array(x, dtype=jnp.int32)
        y = jnp.array(y, dtype=jnp.int32)

        # Get per-token loss
        loss2d = model(x, y, loss_reduction='none')  # (B, T)
        loss2d = loss2d.reshape(-1)
        y_flat = y.reshape(-1)

        # Handle ignore_index (< 0)
        valid = y_flat >= 0
        y_safe = jnp.where(valid, y_flat, jnp.zeros_like(y_flat))
        num_bytes2d = jnp.where(valid, token_bytes[y_safe], jnp.zeros_like(y_flat, dtype=jnp.int32))

        total_nats += float(jnp.sum(loss2d * (num_bytes2d > 0)))
        total_bytes += int(jnp.sum(num_bytes2d))

    if total_bytes == 0:
        return float('inf')

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
