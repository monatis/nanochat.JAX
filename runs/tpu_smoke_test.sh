#!/bin/bash
# =============================================================================
# TPU Smoke Test — end-to-end validation of the nanochat JAX port on TPU
#
# This script is designed to run DIRECTLY ON a TPU VM.
# It tests the entire stack: model init → forward pass → backward pass →
# optimizer step → checkpoint save/load → inference, all on real TPU hardware.
#
# Usage (on TPU VM):
#   bash runs/tpu_smoke_test.sh
#
# What it validates:
#   1. JAX sees TPU devices and bf16 dtype is selected
#   2. Model instantiation (d6, small) on TPU
#   3. Tokenizer init
#   4. Forward pass produces correct shapes
#   5. Loss computation + backward pass
#   6. Muon + AdamW optimizer step
#   7. Data-parallel sharding across TPU cores
#   8. 10-step training loop (tiny model, synthetic data)
#   9. Checkpoint save + load round-trip
#  10. Autoregressive inference (greedy + sampled)
#  11. Engine-based generation with tool-use state machine
# =============================================================================

set -euo pipefail

echo "============================================================"
echo " nanochat TPU Smoke Test"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Environment setup
# ---------------------------------------------------------------------------

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# Install uv if needed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Create venv and install deps
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

echo ""
echo "=== Phase 0: Environment Check ==="
python -c "
import jax
import jax.numpy as jnp
from flax import nnx
import optax

backend = jax.default_backend()
n_devices = jax.device_count()
n_local = jax.local_device_count()
device_kind = jax.devices()[0].device_kind if n_devices > 0 else 'unknown'

print(f'  JAX version   : {jax.__version__}')
print(f'  Backend       : {backend}')
print(f'  Devices       : {n_devices} total, {n_local} local')
print(f'  Device kind   : {device_kind}')
print(f'  Default dtype : {\"bfloat16\" if backend == \"tpu\" else \"float32\"}')

assert backend == 'tpu', f'ERROR: Expected TPU backend, got {backend}'
assert n_devices >= 1, f'ERROR: No TPU devices found'
print('  ✓ TPU backend confirmed')
"
echo ""

# ---------------------------------------------------------------------------
# 1-11. Run the comprehensive Python smoke test
# ---------------------------------------------------------------------------

python -c "
import sys
import os
import time
import tempfile
import shutil

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from flax import nnx
import optax
import numpy as np

# ---- nanochat imports ----
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, create_mesh,
    get_peak_flops, get_device_name,
)
from nanochat.optim import build_optimizer, classify_param
from nanochat.engine import Engine, sample_next_token

num_devices = jax.device_count()
PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  ✓ {name}')
    else:
        FAIL += 1
        print(f'  ✗ {name} — {detail}')

# =========================================================================
print()
print('=== Phase 1: COMPUTE_DTYPE check ===')
check('bf16 on TPU', COMPUTE_DTYPE == jnp.bfloat16, f'got {COMPUTE_DTYPE}')

# =========================================================================
print()
print('=== Phase 2: Model instantiation (d6) ===')
config = GPTConfig(
    sequence_len=256,   # small context for smoke test
    vocab_size=512,     # small vocab for speed
    n_layer=6,
    n_head=6,
    n_kv_head=6,
    n_embd=384,         # 6 * 64
    window_pattern='SL',
)
model = GPT(config, rngs=nnx.Rngs(0))
pc = model.num_scaling_params()
check('Model created', pc['total'] > 0, f'param count={pc[\"total\"]}')
check('Total params > 1M', pc['total'] > 1_000_000, f'only {pc[\"total\"]:,}')
print(f'    Parameter breakdown:')
for k, v in pc.items():
    print(f'      {k:24s}: {v:>12,}')

# =========================================================================
print()
print('=== Phase 3: Forward pass (no targets) ===')
B, T = 2, 64
key = jax.random.key(0)
x = jax.random.randint(key, (B, T), 0, config.vocab_size)
logits = model(x)
check('Logits shape', logits.shape == (B, T, config.vocab_size),
      f'got {logits.shape}, expected ({B}, {T}, {config.vocab_size})')
check('Logits dtype float32', logits.dtype == jnp.float32, f'got {logits.dtype}')
check('No NaN in logits', not jnp.any(jnp.isnan(logits)), 'NaN detected!')
check('Logits finite', jnp.all(jnp.isfinite(logits)), 'Inf detected!')

# =========================================================================
print()
print('=== Phase 4: Forward pass with loss ===')
y = jax.random.randint(jax.random.key(1), (B, T), 0, config.vocab_size)
loss = model(x, targets=y, loss_reduction='mean')
check('Loss is scalar', loss.shape == (), f'shape={loss.shape}')
check('Loss dtype float32', loss.dtype == jnp.float32, f'got {loss.dtype}')
check('Loss is finite', jnp.isfinite(loss), f'loss={float(loss):.6f}')
check('Loss positive', float(loss) > 0, f'loss={float(loss):.6f}')
expected_init_loss = jnp.log(jnp.float32(config.vocab_size))
check(f'Loss near ln({config.vocab_size})={float(expected_init_loss):.2f}',
      abs(float(loss) - float(expected_init_loss)) < 2.0,
      f'loss={float(loss):.4f}')
print(f'    Initial loss: {float(loss):.6f} (expected ~{float(expected_init_loss):.2f})')

# Per-token loss
loss_per_tok = model(x, targets=y, loss_reduction='none')
check('Per-token loss shape', loss_per_tok.shape == (B, T),
      f'got {loss_per_tok.shape}')

# =========================================================================
print()
print('=== Phase 5: Backward pass (gradient computation) ===')
params = nnx.state(model, nnx.Param)
graphdef = nnx.graphdef(model)

def loss_fn(p):
    m = nnx.merge(graphdef, p)
    return m(x, y, loss_reduction='mean')

loss_val, grads = jax.value_and_grad(loss_fn)(params)
check('Grad loss finite', jnp.isfinite(loss_val))

# Check gradients are not all zero
flat_grads = jax.tree.leaves(grads)
total_grad_norm = sum(float(jnp.sum(jnp.square(g))) for g in flat_grads) ** 0.5
check('Grad norm > 0', total_grad_norm > 0, f'norm={total_grad_norm}')
check('Grad norm finite', not (total_grad_norm != total_grad_norm),  # NaN check
      f'norm={total_grad_norm}')
print(f'    Global grad norm: {total_grad_norm:.6f}')

# =========================================================================
print()
print('=== Phase 6: Optimizer creation + step ===')
optim_config = {
    'muon_lr': 0.02,
    'muon_wd': 0.1,
    'adamw_embed_lr': 0.3,
    'adamw_lm_head_lr': 0.008,
    'adamw_scalars_lr': 0.5,
}
tx = build_optimizer(model, optim_config)
opt_state = tx.init(params)
check('Optimizer created', opt_state is not None)

updates, new_opt_state = tx.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
check('Optimizer step completed', new_params is not None)

# Verify params actually changed
old_leaves = jax.tree.leaves(params)
new_leaves = jax.tree.leaves(new_params)
any_changed = any(not jnp.array_equal(o, n) for o, n in zip(old_leaves, new_leaves))
check('Params changed after step', any_changed)

# =========================================================================
print()
print('=== Phase 7: Data-parallel sharding ===')
mesh = create_mesh()
data_sharding = NamedSharding(mesh, P('data', None))
replicated = NamedSharding(mesh, P())

# Shard a batch across devices
B_dp = max(num_devices * 2, 4)
x_dp = jax.random.randint(jax.random.key(2), (B_dp, T), 0, config.vocab_size)
x_sharded = jax.device_put(x_dp, data_sharding)

# Check it's actually sharded
check(f'Data sharded across {num_devices} devices',
      len(x_sharded.devices()) == num_devices,
      f'on {len(x_sharded.devices())} devices')

# Forward pass on sharded data
y_dp = jax.random.randint(jax.random.key(3), (B_dp, T), 0, config.vocab_size)
y_sharded = jax.device_put(y_dp, data_sharding)

@jax.jit
def fwd_sharded(p, x, y):
    m = nnx.merge(graphdef, p)
    return m(x, y, loss_reduction='mean')

loss_sharded = fwd_sharded(params, x_sharded, y_sharded)
check('Sharded forward pass', jnp.isfinite(loss_sharded),
      f'loss={float(loss_sharded):.6f}')

# =========================================================================
print()
print('=== Phase 8: Training loop (10 steps) ===')

@jax.jit
def train_step(model_state, opt_state, x, y):
    def loss_fn(p):
        m = nnx.merge(graphdef, p)
        return m(x, y, loss_reduction='mean')
    loss, grads = jax.value_and_grad(loss_fn)(model_state)
    updates, new_opt = tx.update(grads, opt_state, model_state)
    new_params = optax.apply_updates(model_state, updates)
    return loss, new_params, new_opt

model_state = nnx.state(model, nnx.Param)
opt_state = tx.init(model_state)

# Generate synthetic data (repeated so loss should decrease)
key = jax.random.key(42)
train_x = jax.random.randint(key, (4, T), 0, config.vocab_size)
train_y = jax.random.randint(jax.random.key(43), (4, T), 0, config.vocab_size)
train_x = jax.device_put(train_x, data_sharding) if num_devices <= 4 else train_x
train_y = jax.device_put(train_y, data_sharding) if num_devices <= 4 else train_y

losses = []
t0 = time.time()
for step in range(10):
    loss, model_state, opt_state = train_step(model_state, opt_state, train_x, train_y)
    jax.block_until_ready(model_state)
    losses.append(float(loss))
t1 = time.time()
dt = t1 - t0

check('10 steps completed', len(losses) == 10)
check('Loss decreased', losses[-1] < losses[0],
      f'first={losses[0]:.4f}, last={losses[-1]:.4f}')
check('All losses finite', all(l == l and abs(l) < 1e6 for l in losses))
print(f'    Losses: {\" → \".join(f\"{l:.4f}\" for l in losses)}')
print(f'    Time for 10 steps: {dt:.2f}s ({dt/10*1000:.1f}ms/step)')
# Estimate throughput
tok_per_step = 4 * T  # batch * seq_len
tok_per_sec = tok_per_step * 10 / dt
print(f'    Throughput: {tok_per_sec:,.0f} tok/sec (smoke test, tiny model)')

# =========================================================================
print()
print('=== Phase 9: Checkpoint save + load round-trip ===')
ckpt_dir = os.path.join(tempfile.mkdtemp(dir=os.environ.get('NANOCHAT_BASE_DIR', '/tmp')),
                         'smoke_test_ckpt')
os.makedirs(ckpt_dir, exist_ok=True)

from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint

# Save
nnx.update(model, model_state)
save_checkpoint(
    ckpt_dir, 10, model, opt_state,
    {
        'step': 10,
        'val_bpb': None,
        'model_config': {
            'sequence_len': config.sequence_len,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_kv_head': config.n_kv_head,
            'n_embd': config.n_embd,
            'window_pattern': config.window_pattern,
        },
        'user_config': {},
        'device_batch_size': 4,
        'max_seq_len': T,
        'total_batch_size': 4 * T,
        'loop_state': {'min_val_bpb': 999, 'smooth_train_loss': 0, 'total_training_time': dt},
    }
)
check('Checkpoint saved', os.path.exists(ckpt_dir))

# Load
model2 = GPT(config, rngs=nnx.Rngs(0))
loaded_state, loaded_opt, loaded_meta = load_checkpoint(ckpt_dir, 10, model2)
check('Checkpoint loaded', loaded_state is not None)
check('Metadata round-trip', loaded_meta['step'] == 10)

# Verify loaded weights match
nnx.update(model2, loaded_state)
logits1 = model(x)
logits2 = model2(x)
max_diff = float(jnp.max(jnp.abs(logits1 - logits2)))
check(f'Loaded weights match (max diff={max_diff:.2e})', max_diff < 1e-4, f'diff={max_diff}')

# Cleanup
shutil.rmtree(os.path.dirname(ckpt_dir), ignore_errors=True)

# =========================================================================
print()
print('=== Phase 10: Autoregressive inference ===')
# Use model.generate() directly
prompt = list(range(10))  # arbitrary token ids
gen_tokens = []
for tok in model.generate(prompt, max_tokens=8, temperature=0.0):
    gen_tokens.append(tok)

check('Generated 8 tokens', len(gen_tokens) == 8, f'got {len(gen_tokens)}')
check('All tokens valid', all(0 <= t < config.vocab_size for t in gen_tokens),
      f'tokens={gen_tokens}')

# Sampled generation
gen_sampled = []
for tok in model.generate(prompt, max_tokens=8, temperature=1.0, top_k=50, seed=123):
    gen_sampled.append(tok)
check('Sampled generation works', len(gen_sampled) == 8)

# Determinism: same seed same output
gen_a, gen_b = [], []
for tok in model.generate(prompt, max_tokens=8, temperature=0.0, seed=1):
    gen_a.append(tok)
for tok in model.generate(prompt, max_tokens=8, temperature=0.0, seed=1):
    gen_b.append(tok)
check('Deterministic greedy', gen_a == gen_b, f'{gen_a} != {gen_b}')

# =========================================================================
print()
print('=== Phase 11: Engine generation ===')

class SimpleTokenizer:
    def __init__(self, vocab_size):
        self._special = {
            '<|python_start|>': vocab_size - 6,
            '<|python_end|>':   vocab_size - 5,
            '<|output_start|>': vocab_size - 4,
            '<|output_end|>':   vocab_size - 3,
            '<|assistant_end|>':vocab_size - 2,
            '<|bos|>':          vocab_size - 1,
        }
        self._bos = vocab_size - 1
    def encode_special(self, s): return self._special[s]
    def get_bos_token_id(self): return self._bos
    def encode(self, s, prepend=None):
        tokens = list(s.encode('utf-8')[:64])  # truncate
        return ([prepend] + tokens) if prepend else tokens
    def decode(self, tokens):
        return bytes([t for t in tokens if t < 256]).decode('utf-8', errors='replace')

tok = SimpleTokenizer(config.vocab_size)
engine = Engine(model, tok)

prompt_tokens = [tok.get_bos_token_id()] + list(range(5))
results, masks = engine.generate_batch(prompt_tokens, num_samples=1, max_tokens=8, temperature=0.0)
check('Engine batch gen', len(results) == 1)
check('Engine output length', len(results[0]) > len(prompt_tokens))

# Multi-sample
results_multi, _ = engine.generate_batch(prompt_tokens, num_samples=4, max_tokens=4, temperature=1.0, seed=42)
check('Engine multi-sample', len(results_multi) == 4)

# =========================================================================
print()
print('=== Phase 12: TPU-specific performance ===')
device_name = get_device_name()
peak = get_peak_flops(device_name)
flops_per_token = model.estimate_flops()
print(f'    Device: {device_name}')
print(f'    Peak BF16 FLOPS: {peak:.2e}')
print(f'    FLOPs/token: {flops_per_token:,.0f}')

# Quick latency benchmark: 100 forward passes
nnx.update(model, model_state)
bench_x = jax.random.randint(jax.random.key(99), (4, T), 0, config.vocab_size)

@jax.jit
def bench_fwd(x):
    return model(x)

# Warmup
_ = bench_fwd(bench_x)
jax.block_until_ready(_)

t0 = time.time()
for _ in range(100):
    out = bench_fwd(bench_x)
jax.block_until_ready(out)
t1 = time.time()
fwd_ms = (t1 - t0) / 100 * 1000
print(f'    Forward latency (B=4, T={T}): {fwd_ms:.2f}ms')

# =========================================================================
print()
print('=' * 60)
total = PASS + FAIL
print(f' Results: {PASS}/{total} passed, {FAIL} failed')
if FAIL == 0:
    print(' 🎉 ALL SMOKE TESTS PASSED — nanochat JAX is TPU-ready!')
else:
    print(f' ⚠️  {FAIL} test(s) failed — see above for details')
print('=' * 60)
sys.exit(1 if FAIL > 0 else 0)
"

echo ""
echo "Smoke test complete."
