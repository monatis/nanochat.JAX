"""
Train base model (JAX/TPU edition). From root directory:

    python -m scripts.base_train

For multi-host TPU pods:

    python -m scripts.base_train --coordinator-address=<host0>:1234

For CPU testing (small model):
    python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
import gc
import json
import time
import math
import argparse
import functools
from dataclasses import asdict

import wandb
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import nnx
import optax

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    get_peak_flops,
    get_device_name,
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    create_mesh,
)
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.optim import build_optimizer, classify_param

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model (JAX/TPU)")
# Logging
parser.add_argument(
    "--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)"
)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="depth of the Transformer model"
)
parser.add_argument(
    "--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio"
)
parser.add_argument(
    "--head-dim", type=int, default=128, help="target head dimension for attention"
)
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument(
    "--window-pattern",
    type=str,
    default="SSSL",
    help="sliding window pattern: L=full, S=quarter context",
)
# Training horizon
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="explicit number of steps (-1 = auto)",
)
parser.add_argument(
    "--target-flops", type=float, default=-1.0, help="target total FLOPs (-1 = disable)"
)
parser.add_argument(
    "--target-param-data-ratio",
    type=float,
    default=12,
    help="data:param ratio (Chinchilla=20)",
)
# Optimization
parser.add_argument(
    "--device-batch-size", type=int, default=32, help="per-device batch size"
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=-1,
    help="total batch size in tokens (-1 = auto)",
)
parser.add_argument(
    "--embedding-lr", type=float, default=0.3, help="LR for embedding (AdamW)"
)
parser.add_argument(
    "--unembedding-lr", type=float, default=0.008, help="LR for lm_head (AdamW)"
)
parser.add_argument(
    "--weight-decay", type=float, default=0.28, help="weight decay for Muon"
)
parser.add_argument(
    "--matrix-lr", type=float, default=0.02, help="LR for matrix params (Muon)"
)
parser.add_argument(
    "--scalar-lr", type=float, default=0.5, help="LR for scalars (AdamW)"
)
parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps")
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.65,
    help="fraction of iterations for LR warmdown",
)
parser.add_argument(
    "--final-lr-frac", type=float, default=0.05, help="final LR as fraction of peak"
)
parser.add_argument(
    "--resume-from-step", type=int, default=-1, help="resume from checkpoint step"
)
# Evaluation
parser.add_argument(
    "--eval-every",
    type=int,
    default=250,
    help="evaluate val loss every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval-tokens", type=int, default=80 * 524288, help="tokens for val evaluation"
)
parser.add_argument(
    "--core-metric-every",
    type=int,
    default=2000,
    help="evaluate CORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core-metric-max-per-task",
    type=int,
    default=500,
    help="examples per task for CORE",
)
parser.add_argument(
    "--sample-every",
    type=int,
    default=2000,
    help="sample from model every N steps (-1 = disable)",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=-1,
    help="checkpoint every N steps (-1 = only at end)",
)
# Output
parser.add_argument(
    "--model-tag", type=str, default=None, help="override checkpoint directory name"
)
# Distributed (for multi-host TPU pods)
parser.add_argument(
    "--coordinator-address",
    type=str,
    default=None,
    help="coordinator address for multi-host",
)
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Initialize JAX distributed + device mesh

num_devices, proc_idx, proc_count = compute_init()
master_process = proc_idx == 0

# Create device mesh for data parallelism
mesh = create_mesh()  # 1D mesh: all devices on 'data' axis
data_sharding = NamedSharding(mesh, P("data", None))
replicated_sharding = NamedSharding(mesh, P())

# Device info
device_name = get_device_name()
peak_flops = get_peak_flops(device_name)
print0(f"Device: {device_name} | Peak FLOPS (BF16): {peak_flops:.2e}")
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model


def build_model(depth):
    """Build a GPT model for a given depth."""
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    model = GPT(config, rngs=nnx.Rngs(0))
    return model


model = build_model(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

# Checkpoint directory
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

# -----------------------------------------------------------------------------
# Scaling laws to determine optimal training horizon, batch size, learning rates

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")


def get_scaling_params(m):
    pc = m.num_scaling_params()
    return pc["transformer_matrices"] + pc["lm_head"]


num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

# Reference model for muP-style hyperparameter transfer
d12_ref = build_model(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

# Optimal batch size (Power Lines paper: Bopt ∝ D^0.383)
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio**0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# LR scaling for batch size
batch_lr_scale = (total_batch_size / B_REF) ** 0.5
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

# Weight decay scaling
weight_decay_scaled = (
    args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
)
if weight_decay_scaled != args.weight_decay:
    print0(
        f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}"
    )

# -----------------------------------------------------------------------------
# Initialize Optimizer & NNX State

optim_config = {
    "muon_lr": args.matrix_lr * batch_lr_scale,
    "muon_wd": weight_decay_scaled,
    "adamw_embed_lr": args.embedding_lr * batch_lr_scale,
    "adamw_lm_head_lr": args.unembedding_lr * batch_lr_scale,
    "adamw_scalars_lr": args.scalar_lr * batch_lr_scale,
}

tx = build_optimizer(model, optim_config)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
opt_state = tx.init(params)

meta_data = None
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    loaded_params, opt_state_restored, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, params, opt_state
    )
    if loaded_params is not None:
        params = loaded_params
    if opt_state_restored is not None:
        opt_state = opt_state_restored

# -----------------------------------------------------------------------------
# DataLoader

data_dir = os.path.join(base_dir, "data")
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    data_dir,
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    repeat=True,
)

# -----------------------------------------------------------------------------
# Determine training iterations

assert (
    args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
)
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated iterations from data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(
    f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}"
)

# Gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * num_devices
assert total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({total_batch_size}) must be divisible by world_tokens_per_fwdbwd ({world_tokens_per_fwdbwd})"
)
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")


# LR schedule
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# Momentum schedule for Muon
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


# Weight decay schedule (cosine)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))


# -----------------------------------------------------------------------------
# JIT-compiled train step


@jax.jit
def train_step(p, r, opt, x, y):
    """Single training step: forward + backward + optimizer update."""

    def loss_fn(p_inner):
        # Merge params and rest state back into model graph
        m = nnx.merge(graphdef, p_inner, r)
        loss = m(x, targets=y, loss_reduction="mean")
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(p)

    # Apply optimizer
    updates, new_opt = tx.update(grads, opt, p)
    new_params = optax.apply_updates(p, updates)

    return loss, new_params, new_opt


# -----------------------------------------------------------------------------
# Training loop

step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0

if resuming and meta_data is not None:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

print0(f"\n{'=' * 80}")
print0(f"Starting training: {num_iterations} iterations, {total_tokens:,} tokens")
print0(f"{'=' * 80}\n")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: sample from the model (only on master process)
    if (
        args.sample_every > 0
        and master_process
        and (last_step or (step > 0 and step % args.sample_every == 0))
    ):
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "The planets of the solar system are:",
        ]
        # Update model from current state for generation
        nnx.update(model, params, rest)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            text_out = []
            for token in model.generate(tokens, max_tokens=16, temperature=0):
                text_out.append(token)
            print0(tokenizer.decode(tokens + text_out))

    # Save checkpoint
    if last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    ):
        dataloader_state = {}  # TODO: implement dataloader state saving
        save_checkpoint(
            checkpoint_dir,
            step,
            params,
            opt_state,
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step (with gradient accumulation)
    t0 = time.time()
    accumulated_loss = 0.0

    for micro_step in range(grad_accum_steps):
        x_np, y_np, dataloader_state_dict = next(train_loader)
        # Move to devices with data-parallel sharding
        x = jax.device_put(jnp.array(x_np, dtype=jnp.int32), data_sharding)
        y = jax.device_put(jnp.array(y_np, dtype=jnp.int32), data_sharding)

        loss, params, opt_state = train_step(params, rest, opt_state, x, y)
        accumulated_loss += float(loss)

    train_loss_f = accumulated_loss / grad_accum_steps

    # Block until computation is complete (for accurate timing)
    jax.block_until_ready(params)
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (peak_flops * num_devices)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        eta_seconds = (num_iterations - step) * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""

    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}% | total time: {total_training_time / 60:.2f}m{eta_str}"
    )

    if step % 100 == 0:
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            }
        )

    step += 1

# Final stats
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

wandb_run.finish()
compute_cleanup()
