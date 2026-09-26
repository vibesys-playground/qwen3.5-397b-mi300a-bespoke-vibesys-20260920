"""End-to-end CPU tests for the batching server: HTTP contract, concurrency, cache transparency.

Runs the real `server.py` against the tiny random checkpoint, so the tokenizer, the prompt
cache, the scheduler and the model all take part. The benchmark's rules are the assertions:
every turn completes, none is dropped, and each returns exactly its `max_tokens` budget.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_server_concurrency.py
"""

import http.client
import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_seed_parity import VOCAB, build_hf, post, wait_healthy, write_checkpoint  # noqa: E402

WORDS = ("hello", "world", "think")
MAX_TOKENS = 5


def build_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    names = ["<unk>", "<|im_end|>", "<|im_start|>", "user", "assistant", "system", *WORDS]
    vocab = {w: i for i, w in enumerate(names + [f"w{i}" for i in range(VOCAB - len(names))])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))  # noqa: S106
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    template = (
        "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="<unk>",  # noqa: S106
        eos_token="<|im_end|>",  # noqa: S106
        chat_template=template,
    ).save_pretrained(path)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("srv")
    write_checkpoint(build_hf(), out, mxfp4=False)
    build_tokenizer(out)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, str(ROOT / "server.py"), "--model-path", str(out),
           "--host", "127.0.0.1", "--port", str(port), "--tp", "1", "--devices", "cpu",
           "--dtype", "float32", "--max-seq-len", "96", "--max-batch", "3"]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT)  # noqa: S603
    try:
        wait_healthy(port)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def body(messages: list[dict], max_tokens: int = MAX_TOKENS) -> dict:
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def stream(port: int, messages: list[dict], max_tokens: int = MAX_TOKENS) -> tuple[str, dict]:
    """Send one streamed turn; returns (text, usage), asserting the SSE contract."""
    status, data = post(port, body(messages, max_tokens))
    assert status == 200, data[:300]
    events = [line[6:] for line in data.decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(
        c["choices"][0]["delta"].get("content") or ""
        for c in chunks
        if c.get("choices") and "delta" in c["choices"][0]
    )
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    return text, chunks[-1]["usage"]


def turn(user: str, prior: list[dict] | None = None) -> list[dict]:
    return [*(prior or []), {"role": "user", "content": user}]


def flush(port: int) -> None:
    """Clear every cached prefix so the next turn is a guaranteed cold start.

    Stage 1's cache lived one-per-slot, so cycling unrelated conversations through every slot
    (`--max-batch 3`) was enough to push out whatever a slot held before. Stage 2 decouples the
    prefix trie from batch lanes (`session_cache.SessionCache`): occupying every lane briefly
    says nothing about the trie, which keeps unrelated entries (from this module's own earlier
    tests, sharing this module-scoped server) until *cache* pressure, not *lane* pressure, evicts
    them -- and a handful of unrelated turns is not guaranteed to walk every stale leaf out of a
    trie an earlier test may have left several nodes deep. Use the admin endpoint instead
    (`server.py`'s `/reset_prefix_cache`, `Scheduler.reset_prefix_cache` ->
    `SessionCache.evict_all`) for a deterministic clear, exactly as a real benchmark sweep would
    between points -- see that endpoint's own docstring.
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/reset_prefix_cache")
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 200


def test_concurrent_turns_all_complete_their_budget(server: int) -> None:
    """More requests in flight than slots: none dropped, none short of its budget."""
    prompts = [turn(" ".join(WORDS[: 1 + i % 3])) for i in range(6)]
    budgets = [3, 4, 5, 6, 4, 3]
    with ThreadPoolExecutor(len(prompts)) as pool:
        got = list(pool.map(lambda a: stream(server, *a), zip(prompts, budgets, strict=True)))
    for (text, use), budget in zip(got, budgets, strict=True):
        assert use["completion_tokens"] == budget
        assert text != ""


def test_same_prompt_is_stable_under_concurrency(server: int) -> None:
    """Sharing a decode step must not make a turn depend on who else was in the batch."""
    messages = turn("hello world")
    alone, _ = stream(server, messages)
    with ThreadPoolExecutor(3) as pool:
        together = list(
            pool.map(
                lambda m: stream(server, m)[0],
                [messages, turn("world think"), turn("think hello world")],
            )
        )
    assert together[0] == alone


def test_second_turn_served_from_the_cache_matches_a_cold_turn(server: int) -> None:
    """The cross-turn property: reuse changes latency, never the reply."""
    first = turn("hello world")
    reply, _ = stream(server, first)
    second = turn("world think", [*first, {"role": "assistant", "content": reply}])

    flush(server)
    cold, cold_use = stream(server, second)
    assert cold_use["prompt_tokens_details"]["cached_tokens"] == 0, "meant to be a full recompute"

    flush(server)
    warm, _ = stream(server, first)  # re-seat only the first turn's prefix
    assert warm == reply
    again, again_use = stream(server, second)
    assert again_use["prompt_tokens_details"]["cached_tokens"] > 0, "prefix was not reused"
    assert again == cold
    assert again_use["completion_tokens"] == cold_use["completion_tokens"] == MAX_TOKENS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
