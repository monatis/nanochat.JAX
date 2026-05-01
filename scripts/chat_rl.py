"""
Reinforcement learning on GSM8K via simplified REINFORCE (JAX/TPU edition).

python -m scripts.chat_rl
"""

import argparse
import os
import itertools
import numpy as np

import wandb
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, create_mesh, get_dist_info
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from nanochat.optim import build_optimizer
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K (JAX/TPU)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--num-epochs", type=int, default=1, help="epochs over GSM8K")
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="examples per optimization step")
parser.add_argument("--num-samples", type=int, default=16, help="samples per example/question")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embeddings")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LR for lm_head")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR fraction")
parser.add_argument("--eval-every", type=int, default=60, help="evaluate every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="examples for eval")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

num_devices, proc_idx, proc_count = compute_init()
master_process = proc_idx == 0
mesh = create_mesh()

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

model, tokenizer, meta = load_model("sft", phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

# Training data
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

# Optimizer
optim_config = {
    'muon_lr': args.matrix_lr * args.init_lr_frac,
    'muon_wd': args.weight_decay,
    'adamw_embed_lr': args.embedding_lr * args.init_lr_frac,
    'adamw_lm_head_lr': args.unembedding_lr * args.init_lr_frac,
    'adamw_scalars_lr': 0.5 * args.init_lr_frac,
}
tx = build_optimizer(model, optim_config)
params = nnx.state(model, nnx.Param)
opt_state = tx.init(params)

# LR schedule: rampdown to zero
def get_lr_multiplier(it):
    return 1.0 - it / num_steps

assert args.examples_per_step % num_devices == 0
examples_per_rank = args.examples_per_step // num_devices
print0(f"Examples per rank: {examples_per_rank}")

# Rollout generator
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(proc_idx, len(train_task), num_devices)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            seqs_batch, masks_batch = engine.generate_batch(
                tokens, num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, seed=seed,
            )
            generated_token_sequences.extend(seqs_batch)
            masks.extend(masks_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_seqs = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        ids = np.array(padded_seqs, dtype=np.int32)
        mask_ids = np.array(padded_masks, dtype=np.int32)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].copy()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards_arr = np.array(rewards, dtype=np.float32)
        advantages = rewards_arr - rewards_arr.mean()
        yield generated_token_sequences, inputs, targets, rewards_arr, advantages

# Training loop
model_state = nnx.state(model, nnx.Param)
batch_iterator = get_batch()

for step in range(num_steps):

    # Evaluate
    if step % args.eval_every == 0:
        # Simple pass@1 evaluation
        num_correct, total = 0, 0
        for idx in range(proc_idx, min(args.eval_examples, len(val_task)), num_devices):
            conversation = val_task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            seqs, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=args.max_new_tokens,
                temperature=0.0, top_k=args.top_k,
            )
            generated_text = tokenizer.decode(seqs[0][prefix_length:])
            if val_task.evaluate(conversation, generated_text):
                num_correct += 1
            total += 1
        accuracy = num_correct / max(total, 1)
        print0(f"Step {step} | Pass@1: {accuracy:.4f}")
        wandb_run.log({"step": step, "pass@1": accuracy})

    # Forward/Backward on rollouts
    rewards_list = []
    sequence_lengths = []

    # RL loss computation (not JIT-compiled because inputs vary in shape)
    graphdef = nnx.graphdef(model)

    for example_step in range(examples_per_rank):
        sequences_all, inputs_np, targets_np, rewards_np, advantages_np = next(batch_iterator)

        inputs_all = jnp.array(inputs_np)
        targets_all = jnp.array(targets_np)
        advantages_all = jnp.array(advantages_np)

        num_passes = inputs_all.shape[0] // args.device_batch_size
        for pass_idx in range(num_passes):
            b0 = pass_idx * args.device_batch_size
            b1 = (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            advantages = advantages_all[b0:b1]

            def rl_loss_fn(params):
                m = nnx.merge(graphdef, params)
                logp = -m(inputs, targets, loss_reduction='none')  # (B, T)
                pg_obj = jnp.sum(logp * advantages[:, None])
                num_valid = jnp.maximum(jnp.sum(targets >= 0), 1)
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
                return -pg_obj

            loss, grads = jax.value_and_grad(rl_loss_fn)(model_state)
            updates, opt_state_new = tx.update(grads, opt_state, model_state)
            model_state = optax.apply_updates(model_state, updates)
            opt_state = opt_state_new

            print0(f"Step {step}/{num_steps} | Example {example_step} | Pass {pass_idx} | loss: {float(loss):.6f}")

        rewards_list.append(float(rewards_np.mean()))
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    print0(f"Step {step}/{num_steps} | Avg reward: {mean_reward:.4f} | Avg seq len: {mean_seq_len:.1f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_seq_len})

    # Save checkpoint
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        nnx.update(model, model_state)
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        from dataclasses import asdict
        save_checkpoint(checkpoint_dir, step, model, None, {"model_config": asdict(model.config)})
        print(f"✅ Saved checkpoint to {checkpoint_dir}")

from nanochat.report import get_report
get_report().log(section="Chat RL", data=[user_config])

wandb_run.finish()
compute_cleanup()
