"""
FP8 training is not supported on TPU.
TPU v4+ uses bfloat16 natively on the MXU (Matrix Multiply Unit) which is already
the primary high-throughput compute path. No FP8 replacement is needed.

This file is kept as a stub for backward compatibility of imports.
"""
