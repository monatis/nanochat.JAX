"""
Unified evaluation script for base models (JAX/TPU edition).

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:
    # Evaluate a nanochat model
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""

import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse

import jax
import jax.numpy as jnp

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    download_file_with_lock,
)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.loss_eval import evaluate_bpb

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ",
            end="",
        )

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (
            1.0 - 0.01 * random_baseline
        )
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
    return out


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation (JAX/TPU)")
    parser.add_argument(
        "--eval",
        type=str,
        default="core,bpb,sample",
        help="Comma-separated evaluations: core,bpb,sample",
    )
    parser.add_argument(
        "--model-tag", type=str, default=None, help="nanochat model tag"
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Model step to load (default = last)"
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Max examples per CORE task (-1 = all)",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=32,
        help="Per-device batch size for BPB evaluation",
    )
    parser.add_argument(
        "--split-tokens", type=int, default=40 * 524288, help="Tokens per split for BPB"
    )
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(","))
    valid_modes = {"core", "bpb", "sample"}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Init
    num_devices, proc_idx, proc_count = compute_init()
    master_process = proc_idx == 0

    # Load model and tokenizer
    model, tokenizer, meta = load_model(
        "base", phase="eval", model_tag=args.model_tag, step=args.step
    )
    sequence_len = meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes()
    model_name = f"base_model (step {meta['step']})"
    model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if "sample" in eval_modes:
        print0("\n" + "=" * 80)
        print0("Model Samples")
        print0("=" * 80)
        if master_process:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                text_out = []
                for token in model.generate(tokens, max_tokens=16, temperature=0.0):
                    text_out.append(token)
                sample_str = tokenizer.decode(tokens + text_out)
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            for i in range(8):
                tokens = tokenizer("", prepend="<|bos|>")
                text_out = []
                for token in model.generate(
                    tokens, max_tokens=128, temperature=1.0, seed=123 + i
                ):
                    text_out.append(token)
                sample_str = tokenizer.decode(tokens + text_out)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)

    # --- BPB evaluation ---
    if "bpb" in eval_modes:
        print0("\n" + "=" * 80)
        print0("BPB Evaluation")
        print0("=" * 80)
        from nanochat.dataloader import (
            tokenizing_distributed_data_loader_with_state_bos_bestfit,
        )

        tokens_per_step = args.device_batch_size * sequence_len * num_devices
        if args.split_tokens % tokens_per_step != 0:
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens}")
        steps = args.split_tokens // tokens_per_step

        base_dir = get_base_dir()
        for split_name in ["train", "val"]:
            data_dir = os.path.join(base_dir, "data", split_name)
            loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                data_dir,
                tokenizer,
                args.device_batch_size,
                sequence_len,
            )
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if "core" in eval_modes:
        print0("\n" + "=" * 80)
        print0("CORE Evaluation")
        print0("=" * 80)
        core_results = evaluate_core(model, tokenizer, max_per_task=args.max_per_task)

        if master_process:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(
                    f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n"
                )
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    from nanochat.report import get_report

    report_data = [{"model": model_name}]
    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])
    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")
    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append(
            {f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)}
        )
    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
