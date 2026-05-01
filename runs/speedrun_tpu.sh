#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a TPU v4-8 or v5e-4 node and takes approximately 3 hours to complete.
#
# On TPU VM:
#   bash runs/speedrun_tpu.sh
#
# With wandb logging:
#   WANDB_RUN=speedrun bash runs/speedrun_tpu.sh

# Default intermediate artifacts directory
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies (including JAX TPU)
uv sync
# activate venv
source .venv/bin/activate

# Verify JAX sees TPU devices
python -c "import jax; print(f'JAX backend: {jax.default_backend()}, devices: {jax.device_count()}')"

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer

# Download first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8
# Download more shards in background
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# Train the tokenizer
python -m scripts.tok_train
# Evaluate the tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model on TPU
# No --fp8 flag (TPU uses bf16 natively)
# No torchrun (JAX handles distribution automatically)
python -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --run=$WANDB_RUN

# Evaluate the model
python -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate
