"""
Evaluate the Chat model (JAX/TPU edition).

Example runs:
    python -m scripts.chat_eval -a ARC-Easy
    python -m scripts.chat_eval -i sft
"""

import argparse
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0
from nanochat.checkpoint_manager import load_model

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Generative evaluation loop


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):

    num_devices, proc_idx, proc_count = get_dist_info()

    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )

    num_passed, total = 0, 0
    for i in range(proc_idx, num_problems, num_devices):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        completions = []
        for s in range(num_samples):
            gen_tokens = []
            seed = 123 + i * num_samples + s
            for tok in model.generate(
                encoded_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            ):
                gen_tokens.append(tok)
            completions.append(tokenizer.decode(gen_tokens))

        outcomes = [
            task_object.evaluate(conversation, completion) for completion in completions
        ]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)
        print(
            f"\r\033[KProcess {proc_idx} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    print()

    # Aggregate across hosts (if multi-host)
    if proc_count > 1:
        num_passed_arr = jnp.array([num_passed], dtype=jnp.int32)
        total_arr = jnp.array([total], dtype=jnp.int32)
        num_passed = int(jax.lax.psum(num_passed_arr, axis_name="i")[0])
        total = int(jax.lax.psum(total_arr, axis_name="i")[0])

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    num_devices, proc_idx, proc_count = get_dist_info()
    bos = tokenizer.get_bos_token_id()

    num_problems = (
        len(task_object)
        if max_problems is None
        else min(len(task_object), max_problems)
    )
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Extract state once for jitted forward pass
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    @jax.jit
    def get_logits(p, r, x):
        m = nnx.merge(graphdef, p, r)
        return m(x)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    for i in range(proc_idx, num_batches, num_devices):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [
            tokenizer.render_for_completion(conversation)
            for conversation in conversations
        ]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [
            ids + [bos] * (max_length - len(ids)) for ids in prompt_ids
        ]
        input_ids = jnp.array(padded_prompt_ids, dtype=jnp.int32)

        # Forward pass via jitted function
        logits = get_logits(params, rest, input_ids)  # (B, T, V)

        for idx, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, jnp.array(letter_ids)]
            argmax_letter_id = int(jnp.argmax(focus_logits))
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate across hosts
    if proc_count > 1:
        num_passed_arr = jnp.array([num_passed], dtype=jnp.int32)
        total_arr = jnp.array([total], dtype=jnp.int32)
        num_passed = int(jax.lax.psum(num_passed_arr, axis_name="i")[0])
        total = int(jax.lax.psum(total_arr, axis_name="i")[0])

    average = num_passed / total if total > 0 else 0
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------


def run_chat_eval(
    task_name,
    model,
    tokenizer,
    batch_size=1,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
):
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--source", type=str, required=True, help="Source: sft|rl"
    )
    parser.add_argument(
        "-a",
        "--task-name",
        type=str,
        default=None,
        help="Task name (default=all, use | to split)",
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for categorical eval",
    )
    parser.add_argument(
        "-g", "--model-tag", type=str, default=None, help="Model tag to load"
    )
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument(
        "-x", "--max-problems", type=int, default=None, help="Max problems to evaluate"
    )
    args = parser.parse_args()

    num_devices, proc_idx, proc_count = compute_init()

    model, tokenizer, meta = load_model(
        args.source, phase="eval", model_tag=args.model_tag, step=args.step
    )

    all_tasks = [
        "ARC-Easy",
        "ARC-Challenge",
        "MMLU",
        "GSM8K",
        "HumanEval",
        "SpellingBee",
    ]
    baseline_accuracies = {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "HumanEval": 0.0,
        "SpellingBee": 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model,
            tokenizer,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    from nanochat.report import get_report

    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    get_report().log(
        section="Chat evaluation " + args.source,
        data=[
            vars(args),
            results,
            chatcore_metric_dict,
        ],
    )

    compute_cleanup()
