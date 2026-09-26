"""Chat requests (`Request.suffix_len > 0`, `SEED_FOLD_TURN_SUFFIX`) served under MTP must
stream the plain greedy continuation of the whole prompt: the folded suffix is input, never
output. Round 15 served runs emitted the chat suffix tail (`</think>\\n\\n`) as generated text
under MTP (graphpf3/graphpf4 legacy greedy prompt 4, token 0).

    /tmp/torchenv/bin/python -m pytest seed_tests/test_mtp_fold_suffix.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_mtp  # noqa: E402
import scheduler as sched_mod  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402
from session_cache import SessionCache  # noqa: E402
from test_graph_capture import EagerBackend  # noqa: E402
from test_mtp import prompt_of  # noqa: E402
from test_mtp_serve import (  # noqa: E402
    K,
    Oracle,
    Sink,
    mtp_model,
    patch_eager_draft,
    patch_graph_draft,
    plain_model,
)
from tp_driver import pool_handshake  # noqa: E402

# (prompt, suffix_len, max_new): the suffix is the prompt's last `suffix_len` ids.
CASES = [
    (prompt_of(21, 9), 4, 7),
    (prompt_of(22, 6), 3, 6),
    (prompt_of(23, 11), 2, 5),
    (prompt_of(24, 5), 1, 6),
]


@pytest.fixture(autouse=True)
def roomy_kv_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import model as seed_model

    monkeypatch.setattr(seed_model, "KV_POOL_MIN_CAPACITY_FACTOR", 16.0)
    monkeypatch.setattr(seed_model, "SNAPSHOT_POOL_MIN_CAPACITY_FACTOR", 16.0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from test_mtp import write_mtp_shard
    from test_seed_parity import build_hf, tiny_config, write_checkpoint

    out = tmp_path_factory.mktemp("tiny-mtp-fold")
    cfg = tiny_config()
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


def reference(checkpoint: Path) -> tuple[list[list[int]], Oracle]:
    ref = plain_model(checkpoint, 1)
    oracle = Oracle(K)
    want = []
    for prompt, _, max_new in CASES:
        full = list(ref.generate(list(prompt), max_new + K, 0.0, frozenset()))
        oracle.add(list(prompt) + full, len(prompt) - 1)
        want.append(full[:max_new])
    return want, oracle


def serve(runner, model, concurrent: bool, served: bool) -> list[list[int]]:  # noqa: ANN001
    pool_handshake(model)
    cache = SessionCache(model.block_allocator, model.block_size, model.num_snapshots)
    extra = {}
    if served:  # the served flag set (mtp_serve_flags.sh), bar the overlap scheduler
        clock = [0.0]
        extra = dict(
            batched_prefill=True,
            defer_turn_close=True,
            prefill_fit_graph=True,
            prefill_accum=True,
            prefill_accum_min_decode=1,
            prefill_accum_max_wait_ms=0.0,
            clock=lambda: clock[0],
        )
    sched = Scheduler(
        runner, cache, prefill_chunk=4, spec_decode=True, fold_turn_suffix=True, **extra
    )
    sinks = []
    for prompt, suffix_len, max_new in CASES:
        sink = Sink()
        sched.submit(Request(list(prompt), max_new, 0.0, frozenset(), sink, suffix_len))
        sinks.append(sink)
        if not concurrent:
            while sched.step():
                pass
    while sched.step():
        pass
    assert all(s.end is not None and s.end[0] == "end" for s in sinks)
    return [s.tokens for s in sinks]


@pytest.mark.parametrize("prefill_graphs", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("served", [False, True])
def test_folded_suffix_is_not_output_under_mtp(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, prefill_graphs: bool, concurrent: bool, served: bool
) -> None:
    """Forced drafts need the captured wide verify (GPU); on CPU the suffix drains through
    plain decode ticks, then MTP rounds take over (graphpf4's configuration)."""
    monkeypatch.setenv("SEED_PREFILL_GRAPH_SHAPES", "1x4,2x4,3x8")
    monkeypatch.setattr(sched_mod, "MTP_FORCED_DRAFTS", False)
    want, oracle = reference(checkpoint)
    patch_eager_draft(monkeypatch, oracle)
    model = mtp_model(checkpoint, 4)
    runner = graph_mtp.GraphMTPRunner(model, backend=EagerBackend())
    runner.want_prefill_graphs = prefill_graphs
    runner.prepare()
    assert runner.mtp_runner.enabled
    patch_graph_draft(runner.mtp_runner, oracle)
    got = serve(runner, model, concurrent, served)
    assert got == want
