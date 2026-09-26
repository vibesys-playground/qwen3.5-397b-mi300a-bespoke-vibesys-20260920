"""Hermetic CPU tests for `Detok` and `DetokWorker` in server.py.

Two properties matter:

- Exactness: the windowed incremental decoder must emit exactly what a from-scratch
  `tok.decode(ids_so_far)` on every token would emit (the old implementation), for
  streamed deltas and for the final concatenation, across text that forces UTF-8
  multi-byte splits and BPE merges across token boundaries.
- Amortized cost: total characters passed to `tok.decode` across a whole reply must be
  linear in the reply length, not quadratic (the old implementation's O(reply length)
  per token).

`DetokWorker` moves that decode work onto a per-request thread; the concurrency tests
check its output is identical to the single-threaded reference under many simultaneous,
jittered streams: nothing dropped, duplicated, or reordered.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_detok.py
"""

import asyncio
import random
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import Detok, DetokWorker  # noqa: E402

TEXTS = [
    "hello world, this is plain ascii streamed one token at a time",
    "café résumé naïve — accented Latin, a curly quote, and an em dash",
    "Ω≈ç√∫˜µ≤≥÷ math and Greek symbols mixed with punctuation",
    "你好，世界！这是中文文本，包含多字节字符。",
    "こんにちは世界。ストリーミングのテストです。",
    "emoji stress test: 🎉🔥👍🏽🧑‍💻🇺🇸 family: 👨‍👩‍👧‍👦 flags and skin tones",
    "mixed: hello 世界 🎉 café Ω done",
    "",
]


@pytest.fixture(scope="module")
def tok():
    """A real byte-level BPE tokenizer, trained small on purpose.

    `pre_tokenizers.ByteLevel` is the same family Qwen's tokenizer uses: every string
    encodes to *some* id sequence (unseen text falls back to single-byte tokens), which
    is the worst case for incremental decode since it forces every multi-byte codepoint
    to split across as many tokens as it has bytes.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.BPE(unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    corpus = [t for t in TEXTS if t] * 4 + ["the quick brown fox jumps over the lazy dog"] * 4
    core.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=800,
            special_tokens=["<unk>", "<|im_end|>"],
            # Full byte coverage, like a real byte-level BPE tokenizer: any input text
            # round-trips through encode/decode, even bytes absent from this tiny corpus.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",
        eos_token="<|im_end|>",  # noqa: S106
    )


def naive_deltas(tok, ids: list[int]) -> list[str]:  # noqa: ANN001
    """What the pre-fix `Detok.push` produced: a full redecode of `ids_so_far` each step."""
    out, sent, seen = [], "", []
    for token in ids:
        seen.append(token)
        text = tok.decode(seen, skip_special_tokens=True)
        if text.endswith("�"):
            out.append("")
            continue
        out.append(text[len(sent) :])
        sent = text
    return out


def incremental_deltas(tok, ids: list[int]) -> list[str]:  # noqa: ANN001
    detok = Detok(tok)
    return [detok.push(t) for t in ids]


@pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
def test_incremental_matches_full_redecode(tok, text: str) -> None:  # noqa: ANN001
    ids = tok(text, add_special_tokens=False)["input_ids"]
    expected = naive_deltas(tok, ids)
    got = incremental_deltas(tok, ids)
    assert got == expected
    assert "".join(got) == tok.decode(ids, skip_special_tokens=True) == text


def test_incremental_handles_codepoints_split_across_several_tokens(tok) -> None:  # noqa: ANN001
    """Worst case for a windowed decoder: a codepoint whose bytes span multiple token ids."""
    text = "🎉🔥你好€§±"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    # With an 800-token vocab trained on a small English/emoji corpus, most of these
    # codepoints have no single learned token, so encoding needs more ids than
    # characters -- confirming this text actually exercises multi-token splits.
    assert len(ids) > len(text)
    expected = naive_deltas(tok, ids)
    got = incremental_deltas(tok, ids)
    assert got == expected
    assert "".join(got) == tok.decode(ids, skip_special_tokens=True) == text


def test_window_does_not_grow_with_reply_length(tok) -> None:  # noqa: ANN001
    """After a resolved push, the undecoded window is small and reset, not the whole reply."""
    ids = tok("the quick brown fox jumps over the lazy dog " * 20, add_special_tokens=False)[
        "input_ids"
    ]
    detok = Detok(tok)
    max_window = 0
    for token in ids:
        detok.push(token)
        max_window = max(max_window, len(detok.ids) - detok.start)
    assert max_window <= 8, f"window grew to {max_window} ids over a {len(ids)}-token reply"


def test_decode_work_is_linear_not_quadratic_in_reply_length(tok) -> None:  # noqa: ANN001
    """The old bug: redecoding the whole reply each token makes total decode work O(n^2)."""

    class CountingTok:
        def __init__(self, inner) -> None:  # noqa: ANN001
            self.inner, self.ids_seen = inner, 0

        def decode(self, ids, **kw):  # noqa: ANN001, ANN003, ANN201
            self.ids_seen += len(ids)
            return self.inner.decode(ids, **kw)

    counting = CountingTok(tok)
    reply = "the quick brown fox jumps over the lazy dog " * 20
    ids = tok(reply, add_special_tokens=False)["input_ids"]
    detok = Detok(counting)
    for token in ids:
        detok.push(token)
    # Linear (each id decoded a small constant number of times), not quadratic
    # (O(n) work on the n-th token would sum to O(n^2) over the whole reply).
    assert counting.ids_seen <= 4 * len(ids), (
        f"decoded {counting.ids_seen} ids total over a {len(ids)}-token reply: not O(n)"
    )


# -- DetokWorker: off the scheduler thread, preserves order under concurrency -----------


async def _drive_one(tok, loop, ids: list[int]) -> list[tuple]:  # noqa: ANN001
    """Simulate one request's scheduler-thread emits landing on a `DetokWorker`."""
    out: asyncio.Queue = asyncio.Queue()
    worker = DetokWorker(tok, loop, out)

    def feed() -> None:
        for token in ids:
            worker.submit(("tok", token))
            if random.random() < 0.3:  # jitter to encourage interleaving across requests
                time.sleep(0.0005)
        worker.submit(("end", ("stop", len(ids), 0)))

    thread = threading.Thread(target=feed)
    thread.start()
    events = []
    while True:
        item = await out.get()
        events.append(item)
        if item[0] != "tok":
            break
    thread.join(timeout=5)
    assert not thread.is_alive()
    return events


async def _drive_many(tok, token_lists: list[list[int]]) -> list[list[tuple]]:  # noqa: ANN001
    loop = asyncio.get_running_loop()
    return await asyncio.gather(*(_drive_one(tok, loop, ids) for ids in token_lists))


def test_worker_output_matches_serial_detok_under_concurrency(tok) -> None:  # noqa: ANN001
    random.seed(0)
    token_lists = [
        tok(text, add_special_tokens=False)["input_ids"]
        for text in TEXTS * 3
        if text  # empty text has nothing to stream
    ]
    results = asyncio.run(_drive_many(tok, token_lists))

    for ids, events in zip(token_lists, results, strict=True):
        tok_events = [v for k, v in events[:-1]]
        assert all(k == "tok" for k, _v in events[:-1]), events
        assert len(tok_events) == len(ids), "a token was dropped or duplicated"
        assert tok_events == incremental_deltas(tok, ids), "output reordered vs. serial decode"
        assert "".join(tok_events) == tok.decode(ids, skip_special_tokens=True)
        assert events[-1] == ("end", ("stop", len(ids), 0))


def test_worker_thread_terminates_after_terminal_event(tok) -> None:  # noqa: ANN001
    loop = asyncio.new_event_loop()
    out: asyncio.Queue = asyncio.Queue()

    async def run() -> None:
        worker = DetokWorker(tok, loop, out)
        worker.submit(("tok", tok("hi", add_special_tokens=False)["input_ids"][0]))
        worker.submit(("error", "boom"))
        await out.get()
        last = await out.get()
        assert last == ("error", "boom")

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


def test_submit_does_not_block_on_slow_decode(tok) -> None:  # noqa: ANN001
    """The scheduler-thread call (`submit`) must stay cheap even if decode is slow.

    This is the latency-overlap property: `submit` only queues, so many scheduler
    iterations can proceed while a slow decode is still catching up on another thread.
    """

    class SlowTok:
        def decode(self, ids, **kw):  # noqa: ANN001, ANN003, ANN201
            time.sleep(0.02)
            return tok.decode(ids, **kw)

    loop = asyncio.new_event_loop()
    out: asyncio.Queue = asyncio.Queue()

    async def run() -> None:
        worker = DetokWorker(SlowTok(), loop, out)
        ids = tok("the quick brown fox", add_special_tokens=False)["input_ids"]
        t0 = time.monotonic()
        for token in ids:
            worker.submit(("tok", token))
        submit_elapsed = time.monotonic() - t0
        worker.submit(("end", ("stop", len(ids), 0)))
        while (await out.get())[0] != "end":
            pass
        total_elapsed = time.monotonic() - t0
        # Submitting all tokens is near-instant; only draining them is slow.
        assert submit_elapsed < 0.01, f"submit blocked on decode: {submit_elapsed:.3f}s"
        assert total_elapsed >= 0.02 * len(ids)

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
