"""
Attention is now handled directly via jax.nn.dot_product_attention in gpt.py.
On TPU, this automatically dispatches to Splash Attention (the TPU equivalent of Flash Attention).
On GPU, it uses cuDNN flash attention.
On CPU, it falls back to the standard dot-product implementation.

This file is kept as a stub for backward compatibility of imports.
"""
