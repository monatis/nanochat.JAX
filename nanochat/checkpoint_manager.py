"""
Utilities for saving and loading model/optim/state checkpoints (JAX/Orbax edition).

Key differences from the PyTorch version:
- Uses Orbax for async, distributed-safe checkpointing
- Supports GCS paths natively (critical for TPU VMs)
- No per-rank sharding of optimizer state — JAX SPMD handles this automatically
- Checkpoints are directories (not single .pt files)
"""
import os
import re
import glob
import json
import logging
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from nanochat.common import get_base_dir, setup_default_logging
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if jax.process_index() == 0:
        logger.info(message)


def _get_checkpoint_manager(checkpoint_dir, max_to_keep=3):
    """Create an Orbax CheckpointManager for the given directory."""
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
    return ocp.CheckpointManager(checkpoint_dir, options=options)


def save_checkpoint(checkpoint_dir, step, model, opt_state, meta_data):
    """
    Save a checkpoint using Orbax.
    - model: the nnx.Module (we extract its state)
    - opt_state: the optax optimizer state pytree
    - meta_data: dict of training metadata (saved as JSON separately for human readability)
    """
    if jax.process_index() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Extract model state as a pytree
    model_state = nnx.state(model)

    # Use Orbax for the heavy lifting (model + optimizer)
    mgr = _get_checkpoint_manager(checkpoint_dir)
    mgr.save(step, args=ocp.args.StandardSave({
        'model': model_state,
        'opt_state': opt_state,
    }))
    mgr.wait_until_finished()

    # Save metadata as JSON (human-readable, rank 0 only)
    if jax.process_index() == 0:
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def load_checkpoint(checkpoint_dir, step, model, opt_state=None):
    """
    Load a checkpoint using Orbax.
    Returns (model_state, opt_state, meta_data).
    If opt_state template is None, only model state is restored.
    """
    mgr = _get_checkpoint_manager(checkpoint_dir)

    # Build restore target
    model_state = nnx.state(model)
    target = {'model': model_state}
    if opt_state is not None:
        target['opt_state'] = opt_state

    restored = mgr.restore(step, args=ocp.args.StandardRestore(target))

    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    return restored.get('model'), restored.get('opt_state'), meta_data


def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")


def build_model(checkpoint_dir, step, phase):
    """
    Build a model from a given checkpoint.
    Returns: (model, tokenizer, meta_data)
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"

    # Load metadata to get config
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)

    # Create model
    model = GPT(model_config, rngs=nnx.Rngs(0))

    # Restore model state from checkpoint
    model_state, _, _ = load_checkpoint(checkpoint_dir, step, model)
    if model_state is not None:
        nnx.update(model, model_state)

    # Load the Tokenizer
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"

    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    """Attempt to guess the model tag: take the biggest model available."""
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # Try d<number> format first
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # Fall back to most recently updated
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """Find the latest checkpoint step in a directory."""
    # Look for meta_*.json files (since Orbax checkpoint dirs are numbered)
    meta_files = glob.glob(os.path.join(checkpoint_dir, "meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in meta_files))
    return last_step


# --- Convenience functions ---

def load_model_from_dir(checkpoints_dir, phase, model_tag=None, step=None):
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, phase)
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
