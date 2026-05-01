"""
Common utilities for nanochat (JAX/TPU edition).
"""

import os
import re
import logging
import urllib.request
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from filelock import FileLock

# The dtype used for compute (matmuls, activations).
# On TPU, bfloat16 is the native high-throughput dtype for the MXU.
# On CPU/GPU fallback, we use float32.
COMPUTE_DTYPE = jnp.bfloat16 if jax.default_backend() == 'tpu' else jnp.float32
COMPUTE_DTYPE_REASON = f"auto-detected: {jax.default_backend()} backend"

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    if jax.process_index() == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

# --- Distributed / Mesh utilities ---

def get_dist_info():
    """
    Returns (num_devices, process_index, process_count).
    In JAX SPMD, there is no "DDP" flag — all devices are always visible.
    """
    return jax.device_count(), jax.process_index(), jax.process_count()

def create_mesh(mesh_shape=None, axis_names=('data',)):
    """
    Create a device mesh for SPMD parallelism.
    By default, creates a 1D mesh with all devices on the 'data' axis (pure data parallelism).
    """
    if mesh_shape is None:
        mesh_shape = (jax.device_count(),)
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices, axis_names=axis_names)

def compute_init():
    """Basic initialization for JAX distributed training."""
    # Initialize multi-host JAX if needed (multi-host TPU pods)
    if not jax.distributed.is_initialized():
        jax.distributed.initialize()

    num_devices, proc_idx, proc_count = get_dist_info()

    if proc_idx == 0:
        logger.info(f"JAX backend: {jax.default_backend()}")
        logger.info(f"Devices: {num_devices} (across {proc_count} hosts)")
        logger.info(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    return num_devices, proc_idx, proc_count

def compute_cleanup():
    """Companion function to compute_init — no-op for JAX (no process groups to destroy)."""
    pass

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for various accelerators
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # Google TPU
        (["tpu v6e"], 918e12),      # TPU v6e: 918 bf16 TFLOPS
        (["tpu v5p"], 918e12),      # TPU v5p: 918 bf16 TFLOPS
        (["tpu v5e"], 197e12),      # TPU v5e (v5 litepod): 197 bf16 TFLOPS
        (["tpu v4"], 275e12),       # TPU v4: 275 bf16 TFLOPS
        (["tpu v3"], 123e12),       # TPU v3: 123 bf16 TFLOPS
        # NVIDIA (kept for GPU fallback)
        (["h200"], 989e12),
        (["h100"], 989e12),
        (["a100"], 312e12),
        (["4090"], 165.2e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    # Unknown device - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

def get_device_name() -> str:
    """Get a human-readable device name for the current accelerator."""
    backend = jax.default_backend()
    if backend == 'tpu':
        # JAX exposes TPU type info
        device = jax.devices()[0]
        return f"TPU {device.device_kind}"
    elif backend == 'gpu':
        device = jax.devices()[0]
        return str(device.device_kind)
    else:
        return backend.upper()
