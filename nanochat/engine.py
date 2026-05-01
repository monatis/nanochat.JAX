"""
Engine for efficient inference of nanochat models (JAX edition).

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.
- For now, this is a simple autoregressive generation without KV cache.
- TODO: implement KV cache for efficient inference on TPU.
"""

import jax
import jax.numpy as jnp
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """Evaluate a Python expression safely."""
    expr = expr.replace(",", "")

    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
def sample_next_token(logits, key, temperature=1.0, top_k=None):
    """Sample a single next token from logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1, keepdims=True), key
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals, idx = jax.lax.top_k(logits, k)
        vals = vals / temperature
        key, subkey = jax.random.split(key)
        choice = jax.random.categorical(subkey, vals, axis=-1)  # (B,)
        return idx[jnp.arange(idx.shape[0]), choice][:, None], key
    else:
        logits = logits / temperature
        key, subkey = jax.random.split(key)
        sampled = jax.random.categorical(subkey, logits, axis=-1)
        return sampled[:, None], key

# -----------------------------------------------------------------------------
class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Autoregressive generation with tool use support."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        key = jax.random.key(seed)

        # Get special tokens for tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Prefill: run the prompt through the model
        ids = jnp.array([tokens], dtype=jnp.int32)
        logits = self.model(ids)
        logits = logits[:, -1, :]
        if num_samples > 1:
            logits = jnp.tile(logits, (num_samples, 1))

        # Initialize row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        all_ids = jnp.tile(ids, (num_samples, 1)) if num_samples > 1 else ids

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            # Sample
            next_ids, key = sample_next_token(logits, key, temperature, top_k)
            sampled_tokens = [int(next_ids[i, 0]) for i in range(num_samples)]

            # Process each row
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            # Forward next token through model (naive, no KV cache)
            new_col = jnp.array(token_column, dtype=jnp.int32)[:, None]
            all_ids = jnp.concatenate([all_ids, new_col], axis=1)
            logits = self.model(all_ids)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
