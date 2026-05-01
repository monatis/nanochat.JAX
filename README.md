# nanochat (JAX Edition)

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. Originally written in PyTorch, this version is a **100% JAX/Flax NNX** implementation designed to run with maximum efficiency on **Google Cloud TPUs** (and GPUs). It is minimal, hackable, and covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI.

For example, you can train your own GPT-2 capability LLM for only a few dollars on a TPU VM and then talk to it in a familiar ChatGPT-like web UI. nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model. All other hyperparameters are calculated automatically in an optimal way.

## Motivation

Even though there exist multiple attempts to port NanoChat to JAX, they are either not kept up-to-date with the latest JAX/Flax NNX API changes, or they are . incomplete in the sense that they do not cover the entire LLM lifecycle (tokenization, pretraining, finetuning, evaluation, inference, and a chat UI). By porting the original NanoChat repo to JAX while keeping the code minimal and readable, I aim to to about provide the same experience of training LLMs as the original NanoChat repo does, but with the performance benefits of JAX.

## Getting started

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install for JAX/TPU:

```bash
uv sync                # Installs JAX with TPU support by default
source .venv/bin/activate
```

For development (adds pytest, transformers, etc.):

```bash
uv sync --group dev
```

### Reproduce and talk to GPT-2

The entire pipeline is designed for TPU VMs. Boot up a TPU node (e.g., v5p-8) and run:

```bash
# Provision a TPU (see tpu-builders-guide.md for details)
# Once on the TPU VM:
bash runs/tpu_smoke_test.sh   # Verify everything is working
bash runs/speedrun.sh         # Kick off the GPT-2 speedrun
```

Once training is done, you can serve the chat UI:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Talk to your LLM as you'd normally talk to ChatGPT!

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

## Research

nanochat is written using **Flax NNX**, the next-generation module system for JAX. It leverages JAX's powerful transformation system:
- `jax.jit` for fused, high-performance execution.
- `jax.sharding` for seamless data-parallel training across TPU cores.
- **Splash Attention**: TPU-native optimized attention via `jax.nn.dot_product_attention`.

To run a research experiment (e.g., a d12 model):

```bash
python -m scripts.base_train \
    --depth=12 \
    --run="d12_experiment" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows an example of running on CPU or Apple Silicon. It shrinks the model to fit into a few minutes of training. Note that JAX on MPS is still evolving and may have different performance characteristics than CUDA/TPU.

## Precision / dtype

nanochat manages precision explicitly via `COMPUTE_DTYPE` (defined in `nanochat/common.py`).
- **TPU**: Defaults to `bfloat16` for native MXU performance.
- **CPU/MPS**: Defaults to `float32`.

Model weights are generally stored in `float32` for optimizer precision but cast to `COMPUTE_DTYPE` during the forward pass. This "manual mixed precision" approach provides full control over the numerical behavior of the model.

## File structure

```
.
├── LICENSE
├── README.md
├── nanochat
│   ├── gpt.py                  # GPT model in JAX/Flax NNX
│   ├── optim.py                # Muon + AdamW optimizers (JAX-native)
│   ├── checkpoint_manager.py   # Save/Load via Orbax
│   ├── engine.py               # Optimized JAX inference with KV Cache
│   ├── common.py               # Sharding and TPU utilities
│   ├── dataloader.py           # Tokenizing Distributed Data Loader
│   ├── tokenizer.py            # BPE Tokenizer (rustbpe backend)
│   └── ui.html                 # Chat frontend
├── runs
│   ├── tpu_smoke_test.sh       # Comprehensive TPU validation
│   ├── speedrun.sh             # TPU-optimized training script
│   └── runcpu.sh               # CPU/MPS example
├── scripts
│   ├── base_train.py           # JAX training entry point
│   ├── base_eval.py            # JAX evaluation entry point
│   ├── chat_web.py             # Chat Web UI
│   └── ...
└── tests
    └── test_engine.py          # JAX-native tests
```

## Contributing

The goal of nanochat remains the same: a single, cohesive, minimal, readable, and hackable "strong baseline" for micro-LLMs. The shift to JAX enables us to leverage TPU hardware while keeping the code clean and expressive.

Currently, the most interesting part is establishing new SOTA training speeds on TPU. If you're a JAX wizard, we welcome PRs that improve TPU utilization (`mfu`) or training convergence.

## Acknowledgements

- JAX and Flax teams at Google for the amazing ecosystem.
- TPU Builders Program for the compute.
- Original [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt).

## Cite

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy (JAX/TPU edition)},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
