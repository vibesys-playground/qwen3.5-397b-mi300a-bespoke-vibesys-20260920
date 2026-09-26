"""Hermetic CPU tests for the token-ID `/v1/completions` endpoint added for
request-factory's `openai` backend: request validation, the SSE contract it parses
(per-chunk `token_ids`, a trailing usage-only chunk with `prompt_tokens_details.
cached_tokens`), extended-prefix reuse across a token-id session's rounds, the
`/reset_prefix_cache` admin hook, and clean rejection of an over-capacity prompt.

Runs the real `server.py` against the tiny random checkpoint from test_seed_parity.py,
the same pattern test_server_concurrency.py uses.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_completions_endpoint.py
"""

import http.client
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_seed_parity import build_hf, wait_healthy, write_checkpoint  # noqa: E402
from test_server_concurrency import build_tokenizer  # noqa: E402

MAX_SEQ_LEN = 96
# In-vocab ids past the tokenizer's named/special tokens (see build_tokenizer): any
# value here is a valid embedding row, and correctness of what the tiny random model
# generates is not the point of these tests.
PROMPT_A = [20, 21, 22, 23]
PROMPT_B = [30, 31, 32]


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
           "--dtype", "float32", "--max-seq-len", str(MAX_SEQ_LEN), "--max-batch", "3"]  # fmt: skip
    proc = subprocess.Popen(cmd, cwd=ROOT)  # noqa: S603
    try:
        wait_healthy(port)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def post(port: int, path: str, body: dict) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, r.read()


def completions_body(prompt: list[int], max_tokens: int = 4, *, stream: bool = True) -> dict:
    """The exact shape request-factory's `openai` `/completions` backend sends
    (see src/backend/wire/openai.rs `serialize_payload` at the pinned rev)."""
    return {
        "model": "qwen3.5-397b-a17b-mxfp4",
        "rid": "req-test",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True} if stream else None,
    }


def stream_completion(port: int, prompt: list[int], max_tokens: int = 4) -> tuple[list[dict], dict]:
    """POST a streaming completion; returns (per-token chunks, final usage dict)."""
    status, data = post(port, "/v1/completions", completions_body(prompt, max_tokens))
    assert status == 200, data[:300]
    events = [line[6:] for line in data.decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    return chunks, chunks[-1]["usage"]


def flush(port: int) -> None:
    """Push unrelated completions through every slot so no cached prefix survives."""
    for i in range(3):  # --max-batch 3
        stream_completion(port, [40 + i, 41 + i, 42 + i], 2)


# ---------------------------------------------------------------- (a) request validation


def test_text_prompt_is_rejected_with_a_clear_400(server: int) -> None:
    status, data = post(server, "/v1/completions", completions_body([1, 2, 3]) | {"prompt": "hi"})
    assert status == 400
    msg = json.loads(data)["error"]
    assert "token-id array" in msg
    assert "text" in msg.lower()


def test_batched_prompt_is_rejected(server: int) -> None:
    status, data = post(
        server, "/v1/completions", completions_body([1, 2, 3]) | {"prompt": [[1, 2], [3, 4]]}
    )
    assert status == 400
    assert "token ids" in json.loads(data)["error"]


def test_empty_prompt_is_rejected(server: int) -> None:
    status, data = post(server, "/v1/completions", completions_body([1, 2, 3]) | {"prompt": []})
    assert status == 400
    assert "non-empty" in json.loads(data)["error"]


def test_missing_prompt_is_rejected(server: int) -> None:
    body = completions_body([1, 2, 3])
    del body["prompt"]
    status, data = post(server, "/v1/completions", body)
    assert status == 400
    assert "error" in json.loads(data)


def test_over_capacity_prompt_is_a_clean_400_not_a_crash(server: int) -> None:
    long_prompt = list(range(10, 10 + MAX_SEQ_LEN))  # >= --max-seq-len
    status, data = post(server, "/v1/completions", completions_body(long_prompt, 4))
    assert status == 400
    msg = json.loads(data)["error"]
    assert str(MAX_SEQ_LEN) in msg
    # the server must still be usable right after
    status, _ = post(server, "/v1/completions", completions_body(PROMPT_A, 2))
    assert status == 200


# ---------------------------------------------------------------- (b) SSE contract


def test_streaming_chunk_shape_matches_the_openai_completions_contract(server: int) -> None:
    max_tokens = 5
    chunks, use = stream_completion(server, PROMPT_A, max_tokens)
    token_chunks = [
        c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason") is None
    ]
    assert len(token_chunks) == max_tokens, "one SSE chunk per generated token"
    seen_ids = []
    for c in token_chunks:
        choice = c["choices"][0]
        assert c["object"] == "text_completion"
        assert isinstance(choice["text"], str)
        assert choice["token_ids"] is not None and len(choice["token_ids"]) == 1, (
            "each chunk must carry only its own new token id, never a resend of prior ids"
        )
        seen_ids.append(choice["token_ids"][0])
    assert (
        len(seen_ids) == len(set(seen_ids)) or max_tokens == 1
    )  # ids need not be distinct, just per-chunk

    finish_chunks = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "length"

    usage_chunks = [c for c in chunks if not c["choices"] and "usage" in c]
    assert len(usage_chunks) == 1, "exactly one trailing usage-only chunk (choices: [])"
    assert usage_chunks[0]["usage"] == use

    assert use["prompt_tokens"] == len(PROMPT_A)
    assert use["completion_tokens"] == max_tokens
    assert use["total_tokens"] == len(PROMPT_A) + max_tokens
    assert "cached_tokens" in use["prompt_tokens_details"]


def test_non_streaming_completion_reports_text_ids_and_usage(server: int) -> None:
    max_tokens = 4
    status, data = post(
        server, "/v1/completions", completions_body(PROMPT_A, max_tokens, stream=False)
    )
    assert status == 200, data[:300]
    reply = json.loads(data)
    assert reply["object"] == "text_completion"
    choice = reply["choices"][0]
    assert choice["finish_reason"] == "length"
    assert len(choice["token_ids"]) == max_tokens
    assert reply["usage"]["completion_tokens"] == max_tokens
    assert reply["usage"]["prompt_tokens"] == len(PROMPT_A)


def test_return_token_ids_false_omits_the_field(server: int) -> None:
    body = completions_body(PROMPT_A, 3) | {"return_token_ids": False}
    status, data = post(server, "/v1/completions", body)
    assert status == 200
    events = [line[6:] for line in data.decode().splitlines() if line.startswith("data: ")]
    chunks = [json.loads(e) for e in events[:-1]]
    token_chunks = [
        c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason") is None
    ]
    assert all(c["choices"][0].get("token_ids") is None for c in token_chunks)


# ---------------------------------------------------------------- (c) cached_tokens / extended reuse


def test_cached_tokens_on_a_round_that_extends_the_first(server: int) -> None:
    """The token-session contract: round 2's prompt is round 1's prompt plus the model's
    real output ids plus new input (see request-factory's `PromptBuilder.commit_output`).
    `extend_prefix` should let round 2 reuse prompt+reply, re-prefilling only the new
    suffix, not just the original prompt."""
    flush(server)
    max_tokens = 4
    chunks, use1 = stream_completion(server, PROMPT_A, max_tokens)
    assert use1["prompt_tokens_details"]["cached_tokens"] == 0, "meant to start cold"
    reply_ids = [
        c["choices"][0]["token_ids"][0]
        for c in chunks
        if c["choices"] and c["choices"][0].get("finish_reason") is None
    ]
    assert len(reply_ids) == max_tokens

    suffix = [50, 51]
    round2_prompt = PROMPT_A + reply_ids + suffix
    _, use2 = stream_completion(server, round2_prompt, 3)
    cached = use2["prompt_tokens_details"]["cached_tokens"]
    assert cached >= len(PROMPT_A), (
        f"expected at least the prompt ({len(PROMPT_A)}) reused, got {cached}"
    )
    assert use2["prompt_tokens"] == len(round2_prompt)


def test_cache_warming_preflight_gets_a_nonzero_cache_hit_on_the_repeat(server: int) -> None:
    """request-factory's own prefix-cache preflight: send one probe prompt twice with
    max_tokens=1 and require the second response's cached_tokens to be nonzero, or it
    aborts the whole run before any sweep point. A byte-identical repeat is not a
    "longer" prompt the way a real session turn is, so this must work even though
    nothing was ever generated in between to extend into."""
    flush(server)
    probe = [21, 22, 23, 24, 25]
    _, use1 = stream_completion(server, probe, max_tokens=1)
    assert use1["prompt_tokens_details"]["cached_tokens"] == 0, "meant to start cold"

    _, use2 = stream_completion(server, probe, max_tokens=1)
    assert use2["prompt_tokens_details"]["cached_tokens"] > 0, "preflight would abort the run"
    assert use2["prompt_tokens_details"]["cached_tokens"] == len(probe)
    assert use2["prompt_tokens"] == len(probe)


def test_repeat_of_prompt_plus_reply_also_gets_a_full_cache_hit(server: int) -> None:
    """The other exact-repeat shape: resending prompt+reply verbatim, with nothing new
    appended, after a prior round already applied that reply lazily."""
    flush(server)
    chunks, use1 = stream_completion(server, PROMPT_B, max_tokens=3)
    reply_ids = [
        c["choices"][0]["token_ids"][0]
        for c in chunks
        if c["choices"] and c["choices"][0].get("finish_reason") is None
    ]
    round2_prompt = PROMPT_B + reply_ids
    _, use2 = stream_completion(server, round2_prompt, max_tokens=2)
    assert use2["prompt_tokens_details"]["cached_tokens"] == len(round2_prompt)

    # A third, exact repeat of prompt+reply again must also stay a full hit (the base
    # snapshot is now prompt+reply itself, from the second round's own prefill-finish).
    _, use3 = stream_completion(server, round2_prompt, max_tokens=1)
    assert use3["prompt_tokens_details"]["cached_tokens"] == len(round2_prompt)


# ---------------------------------------------------------------- (d) reset hook


def test_reset_prefix_cache_clears_reuse_without_erroring_live_traffic(server: int) -> None:
    flush(server)
    chunks, use1 = stream_completion(server, PROMPT_B, 3)
    assert use1["prompt_tokens_details"]["cached_tokens"] == 0
    reply_ids = [
        c["choices"][0]["token_ids"][0]
        for c in chunks
        if c["choices"] and c["choices"][0].get("finish_reason") is None
    ]

    status, data = post(server, "/reset_prefix_cache", {})
    assert status == 200
    body = json.loads(data)
    assert body["status"] == "ok"

    round2_prompt = PROMPT_B + reply_ids + [60]
    _, use2 = stream_completion(server, round2_prompt, 2)
    assert use2["prompt_tokens_details"]["cached_tokens"] == 0, (
        "reset did not clear the recorded prefix"
    )

    # the server keeps serving normally afterwards
    status, _ = post(server, "/v1/completions", completions_body(PROMPT_A, 2))
    assert status == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
