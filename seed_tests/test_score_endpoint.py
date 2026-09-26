"""Hermetic CPU tests for `/v1/score`'s teacher-forced scoring: `Model.forward`'s
`logits_from` slicing, request validation, and the scoring math, against the tiny random
checkpoint from `test_seed_parity.py`. No HTTP server is booted (see `server.score_continuation`
and `server.parse_score_request`, exercised directly): this is the request-parsing/
response-building layer server.py's other endpoint tests already cover at the HTTP level for
their own contracts. Scoring runs through `Scheduler.score` (inline here, since `run()` is never
started), so nothing here needs a live server.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_score_endpoint.py
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_batched_decode import build, prompt_of  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

import server  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def test_logits_from_tail_matches_all_logits_tail(checkpoint: Path) -> None:
    """`logits_from=k` must return exactly the last `T - k` rows `all_logits=True` would."""
    m = build(checkpoint, max_batch=1)
    prompt = prompt_of(seed=3, length=9)
    m.begin(0)
    full = m.forward(torch.tensor([prompt]), 0, all_logits=True)
    m.begin(0)
    tail = m.forward(torch.tensor([prompt]), 0, logits_from=4)
    assert tail.shape[0] == len(prompt) - 4
    assert torch.equal(full[4:], tail)


def scheduler_for(m) -> Scheduler:  # noqa: ANN001
    return Scheduler(m, SessionCache(m.block_allocator, m.block_size, m.num_snapshots))


def test_score_continuation_logprob_matches_manual_log_softmax(checkpoint: Path) -> None:
    m = build(checkpoint, max_batch=1)
    prompt = prompt_of(seed=3, length=9)
    continuation_start = 5
    scores = server.score_continuation(scheduler_for(m), prompt, continuation_start, top_k=3)
    assert len(scores) == len(prompt) - continuation_start

    m.begin(0)
    full = m.forward(torch.tensor([prompt]), 0, all_logits=True)
    for offset, entry in enumerate(scores):
        want_row = torch.log_softmax(full[continuation_start - 1 + offset], dim=-1)
        token_id = prompt[continuation_start + offset]
        assert entry["token_id"] == token_id
        assert want_row[token_id].item() == pytest.approx(entry["logprob"], abs=1e-5)
        top_val, top_id = want_row.topk(3)
        assert [t["token_id"] for t in entry["top_logprobs"]] == top_id.tolist()
        for got, want in zip(entry["top_logprobs"], top_val.tolist(), strict=True):
            assert got["logprob"] == pytest.approx(want, abs=1e-5)


@pytest.mark.parametrize(
    ("body", "message_fragment"),
    [
        ({"prompt": [1, 2, 3], "continuation_start": 0}, "0 < continuation_start"),
        ({"prompt": [1, 2, 3], "continuation_start": 3}, "0 < continuation_start"),
        ({"prompt": [1, 2, 3], "continuation_start": 1, "top_k": 0}, "top_k must be"),
        ({"prompt": [1, 2, 3], "continuation_start": 1, "top_k": 1.5}, "top_k must be"),
        ({"prompt": "abc", "continuation_start": 1}, "token-id array"),
        ({"continuation_start": 1}, "non-empty array"),
    ],
)
def test_parse_score_request_rejects_invalid_bodies(body: dict, message_fragment: str) -> None:
    with pytest.raises(ValueError, match=message_fragment):
        server.parse_score_request(body)


def test_parse_score_request_accepts_valid_body() -> None:
    prompt, continuation_start, top_k = server.parse_score_request(
        {"prompt": [1, 2, 3, 4], "continuation_start": 2}
    )
    assert (prompt, continuation_start, top_k) == ([1, 2, 3, 4], 2, 5)


def test_score_leaves_a_live_lane_and_the_block_pool_untouched(checkpoint: Path) -> None:
    """`/v1/score` used to `begin(0)` on the model directly: it clobbered a live lane 0,
    leaked lane 0's blocks, and raced the scheduler thread. Now it takes a free lane through
    the scheduler and releases its blocks."""
    m = build(checkpoint, max_batch=2)
    sched = scheduler_for(m)
    busy = prompt_of(seed=5, length=6)
    want = list(build(checkpoint, max_batch=1).generate(busy, 6, 0.0, frozenset()))
    got: list[int] = []
    sched.submit(
        Request(busy, 6, 0.0, frozenset(), lambda e: got.append(e[1]) if e[0] == "tok" else None)
    )
    for _ in range(3):
        sched.step()
    assert sched.decoding, "setup: a request must be mid-decode on lane 0"
    free = m.block_allocator.free_count
    server.score_continuation(sched, prompt_of(seed=3, length=9), 5, top_k=2)
    assert m.block_allocator.free_count == free
    while sched.step():
        pass
    assert got == want


def test_score_route_is_registered_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class Args:
        pass

    monkeypatch.setattr(server, "Engine", lambda args, workers: object())
    monkeypatch.delenv(server.SCORE_ENDPOINT_ENV, raising=False)
    paths = {r.resource.canonical for r in server.make_app(Args()).router.routes()}
    assert "/v1/score" not in paths
    monkeypatch.setenv(server.SCORE_ENDPOINT_ENV, "1")
    paths = {r.resource.canonical for r in server.make_app(Args()).router.routes()}
    assert "/v1/score" in paths
