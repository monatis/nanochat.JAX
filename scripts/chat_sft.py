"""
Supervised fine-tuning (SFT) the model (JAX/TPU edition).
Run as:

python -m scripts.chat_sft

Or with options:

python -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
import time
import numpy as np

import wandb
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx

from nanochat.common import (compute_init, compute_cleanup, print0, DummyWandb,
                              get_base_dir, get_peak_flops, get_device_name,
                              COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, create_mesh)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.optim import build_optimizer
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) (JAX/TPU)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=None, help="LR for embeddings (default: inherit)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="LR for lm_head (default: inherit)")
parser.add_argument("--matrix-lr", type=float, default=None, help="LR for matrix params (default: inherit)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="epochs of MMLU in training mixture")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="epochs of GSM8K in training mixture")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
num_devices, proc_idx, proc_count = compute_init()
master_process = proc_idx == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# Device mesh
mesh = create_mesh()
data_sharding = NamedSharding(mesh, P('data', None))

device_name = get_device_name()
peak_flops = get_peak_flops(device_name)
print0(f"Device: {device_name} | Peak FLOPS (BF16): {peak_flops:.2e}")

# wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")

depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * num_devices
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes()

# Initialize Optimizer (SFT continues with zero weight decay)
optim_config = {
    'muon_lr': args.matrix_lr * args.init_lr_frac,
    'muon_wd': 0.0,
    'adamw_embed_lr': args.embedding_lr * args.init_lr_frac,
    'adamw_lm_head_lr': args.unembedding_lr * args.init_lr_frac,
    'adamw_scalars_lr': 0.5 * args.init_lr_frac,
}
tx = build_optimizer(model, optim_config)
params = nnx.state(model, nnx.Param)
opt_state = tx.init(params)

# SFT data mixture
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [
    SmolTalk(split="train"),
    CustomJSON(filepath=identity_conversations_filepath),
    CustomJSON(filepath=identity_conversations_filepath),
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    SimpleSpelling(size=200000, split="train"),
    SpellingBee(size=80000, split="train"),
]
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows")
val_dataset = TaskMixture([
    SmolTalk(split="test"),
    MMLU(subset="all", split="test", stop=5200),
    GSM8K(subset="main", split="test", stop=420),
])

# Global state for data generator
last_step = False
approx_progress = 0.0
current_epoch = 1

def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """BOS-aligned dataloader for SFT with bestfit-pad packing."""
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = proc_idx
    consumed = proc_idx
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += num_devices
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += num_devices
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            if consumed >= dataset_size:
                last_step = True

        # Build numpy arrays
        batch = np.array(rows, dtype=np.int32)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        mask_arr = np.array(mask_rows, dtype=np.int8)
        mask_targets = mask_arr[:, 1:]
        targets = np.where(mask_targets == 0, -1, targets)

        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0

# LR schedule
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# JIT-compiled train step
@jax.jit
def train_step(model_state, opt_state, x, y):
    graphdef = nnx.graphdef(model)
    def loss_fn(params):
        m = nnx.merge(graphdef, params)
        return m(x, y, loss_reduction='mean')
    loss, grads = jax.value_and_grad(loss_fn)(model_state)
    updates, new_opt_state = tx.update(grads, opt_state, model_state)
    import optax
    new_model_state = optax.apply_updates(model_state, updates)
    return loss, new_model_state, new_opt_state

# -----------------------------------------------------------------------------
# Training loop
model_state = nnx.state(model, nnx.Param)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0

while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Save checkpoint at end
    if last_step:
        nnx.update(model, model_state)
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir, step, model, opt_state,
            {
                "step": step,
                "val_bpb": min_val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step
    t0 = time.time()
    accumulated_loss = 0.0

    for micro_step in range(grad_accum_steps):
        x_np, y_np = next(train_loader)
        x = jax.device_put(jnp.array(x_np, dtype=jnp.int32), data_sharding)
        y = jax.device_put(jnp.array(y_np, dtype=jnp.int32), data_sharding)
        loss, model_state, opt_state = train_step(model_state, opt_state, x, y)
        accumulated_loss += float(loss)
        progress = max(progress, approx_progress)

    train_loss_f = accumulated_loss / grad_accum_steps

    jax.block_until_ready(model_state)
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    step += 1

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (peak_flops * num_devices)
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}% | epoch: {current_epoch}")

    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "train/loss": debiased_smooth_loss,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# Final stats
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config,
    {"Number of iterations": step, "Devices": num_devices},
    {"Minimum validation bpb": min_val_bpb},
])

wandb_run.finish()
compute_cleanup()
