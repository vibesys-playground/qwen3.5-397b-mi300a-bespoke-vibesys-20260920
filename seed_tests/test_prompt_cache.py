"""Hermetic tests for incremental prompt tokenization.

The property that matters is exactness: encoding a prompt from a cached prefix plus the
new tail must give the ids a full tokenization gives. A real BPE tokenizer is trained
here so the test covers merges, which is where a naive text split would be wrong.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompt_cache import PromptCache  # noqa: E402

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CORPUS = [
    "the server schedules the batch and then the batch decodes another token",
    "prefix reuse keeps the history so the next turn only tokenizes the new tail",
    "assistant replies are appended to the conversation before the next user message",
]


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.BPE(unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.train_from_iterator(
        CORPUS * 4,
        trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",  # noqa: S106
        eos_token="<|im_end|>",  # noqa: S106
        chat_template=TEMPLATE,
    )
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    return fast


def conversation(tok, turns: int) -> str:  # noqa: ANN001
    messages = []
    for i in range(turns):
        messages.append({"role": "user", "content": CORPUS[i % len(CORPUS)]})
        if i < turns - 1:
            messages.append({"role": "assistant", "content": CORPUS[(i + 1) % len(CORPUS)]})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_bpe_merges_do_not_break_incremental_encoding(tok) -> None:  # noqa: ANN001
    cache = PromptCache(tok)
    assert cache.selfcheck([conversation(tok, k) for k in (1, 2, 3)])
    for turns in (1, 2, 3, 4, 5):
        text = conversation(tok, turns)
        assert cache.encode(text) == cache.tokenize(text), f"{turns} turns"


def test_only_the_new_tail_is_tokenized(tok) -> None:  # noqa: ANN001
    class Counting:
        def __init__(self, inner) -> None:  # noqa: ANN001
            self.inner, self.chars = inner, 0
            self.all_special_tokens = inner.all_special_tokens

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:  # noqa: FBT001, FBT002
            self.chars += len(text)
            return self.inner(text, add_special_tokens=add_special_tokens)

    counting = Counting(tok)
    cache = PromptCache(counting)
    first, second = conversation(tok, 4), conversation(tok, 5)
    cache.encode(first)
    expected = cache.tokenize(second)
    counting.chars = 0

    assert cache.encode(second) == expected
    # The new tail is tokenized twice (once for the prompt, once to extend the cached
    # prefix) and the shared history not at all, so the work is O(tail), not O(prompt).
    tail = len(second) - len(first)
    assert counting.chars <= 2 * tail + 64
    assert counting.chars < len(second)


def test_lru_bound_is_respected(tok) -> None:  # noqa: ANN001
    cache = PromptCache(tok, capacity=2)
    for turns in (1, 2, 3, 4):
        text = conversation(tok, turns)
        assert cache.encode(text) == cache.tokenize(text)
    assert len(cache.entries) <= 2


class _Splittable:
    """Tokenizer whose encoding is prefix-decomposable, so the cache is safe."""

    all_special_tokens = ["<|s|>"]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:  # noqa: FBT001, FBT002, ARG002
        return {"input_ids": [ord(c) for c in text]}


class _Merging(_Splittable):
    """Tokenizer whose encoding depends on the whole string, so the cache must stay off."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:  # noqa: FBT001, FBT002, ARG002
        return {"input_ids": [len(text), *[ord(c) for c in text]]}


def test_selfcheck_accepts_a_decomposable_tokenizer() -> None:
    cache = PromptCache(_Splittable())
    assert cache.selfcheck(["<|s|>one", "<|s|>one<|s|>two"])
    assert cache.encode("<|s|>one<|s|>two<|s|>three") == cache.tokenize(
        "<|s|>one<|s|>two<|s|>three"
    )


def test_selfcheck_disables_the_cache_when_splitting_would_change_ids() -> None:
    cache = PromptCache(_Merging())
    assert not cache.selfcheck(["<|s|>one", "<|s|>one<|s|>two"])
    text = "<|s|>one<|s|>two<|s|>three"
    assert cache.encode(text) == cache.tokenize(text)  # falls back to full tokenization
    assert cache.entries == {}


def test_selfcheck_fails_when_there_is_nothing_to_check() -> None:
    cache = PromptCache(_Splittable())
    assert not cache.selfcheck([])  # no chat template, so no probes were rendered


def test_cache_is_off_without_special_tokens() -> None:
    class NoMarks(_Splittable):
        all_special_tokens: list[str] = []

    cache = PromptCache(NoMarks())
    assert not cache.enabled
    assert cache.encode("abc") == [97, 98, 99]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
