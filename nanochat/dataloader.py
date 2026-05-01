"""
A tokenizing, distributed data loader for nanochat (JAX/TPU edition).

How it works:
1. Read rows from parquet files (using pyarrow)
2. Tokenize them (using our tokenizer)
3. Pack tokenized sequences into fixed-length chunks, aligned on BOS tokens

Original design philosophy:
- Does not waste any data (every token is used)
- Uses best-fit packing to minimize padding waste
- Supports distributed training via process-based sharding

JAX adaptation:
- Returns jax.numpy arrays instead of torch tensors
- Uses jax.process_index/process_count instead of DDP rank/world_size
- Data is returned as host-side numpy arrays; caller handles jax.device_put with sharding
"""

import os
import glob
import logging
import numpy as np
import jax
import jax.numpy as jnp

from nanochat.common import get_base_dir, setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if jax.process_index() == 0:
        logger.info(message)

def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    data_dir, tokenizer, batch_size, sequence_len,
    initial_file_idx=0, initial_row_idx=0,
    single_shard: str = None, repeat: bool = False,
):
    """
    Generator that yields (x, y, state) batches of tokenized data.

    Changes from PyTorch version:
    - Returns numpy arrays instead of torch tensors
    - Caller is responsible for jax.device_put with appropriate sharding
    - Uses jax.process_index/process_count for distributed sharding
    """
    import pyarrow.parquet as pq

    # Distributed info
    process_idx = jax.process_index()
    process_count = jax.process_count()

    # Find and sort shard files
    if single_shard is not None:
        shard_files = [os.path.join(data_dir, single_shard)]
    else:
        shard_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    assert len(shard_files) > 0, f"No parquet files found in {data_dir}"

    bos_token_id = tokenizer.get_bos_token_id()
    buf = []  # token buffer

    file_idx = initial_file_idx
    row_idx = initial_row_idx

    while True:
        # Read from parquet files
        shard_path = shard_files[file_idx % len(shard_files)]
        log0(f"Loading shard: {os.path.basename(shard_path)} (file_idx={file_idx}, row_idx={row_idx})")
        table = pq.read_table(shard_path)
        texts = table.column("text").to_pylist()

        # Each process takes every Nth row, offset by process index
        for i in range(row_idx, len(texts)):
            if i % process_count != process_idx:
                continue

            text = texts[i]
            tokens = tokenizer.encode(text, prepend=bos_token_id)
            buf.extend(tokens)

            # Yield batches when we have enough tokens
            while len(buf) >= batch_size * (sequence_len + 1):
                batch_tokens = np.array(buf[:batch_size * (sequence_len + 1)], dtype=np.int32)
                batch_tokens = batch_tokens.reshape(batch_size, sequence_len + 1)
                x = batch_tokens[:, :-1]  # inputs
                y = batch_tokens[:, 1:]   # targets
                state = {'file_idx': file_idx, 'row_idx': i}
                yield x, y, state
                buf = buf[batch_size * (sequence_len + 1):]

        # Move to next file
        row_idx = 0
        file_idx += 1

        # If we've gone through all files
        if file_idx >= len(shard_files):
            if repeat:
                file_idx = 0
                log0("Looping back to the beginning of the dataset")
            else:
                log0("Reached end of dataset")
                break
