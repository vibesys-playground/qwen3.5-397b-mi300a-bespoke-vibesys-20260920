"""GPU validation + benchmark for bucketed decode-graph capture (`graph_decode.py`) and MTP
wide-verify graph capture (`graph_mtp.py`).

No GPU is available in this environment (cluster maintenance) -- everything above this script
was validated on CPU (numerics via the eager reference, structure via `seed_tests/
test_graph_capture.py` / `test_graph_capture_tp.py` / `test_graph_mtp.py`). This is the script
those CPU checks predict against once a GPU is available: it captures every decode bucket and
(if `SEED_MTP_SERVE=1`) every MTP draft+verify bucket, confirms each replay agrees with the eager step
(`GraphDecodeRunner.validate`/`MTPVerifyRunner._validate`, already run inside `prepare()`), and
times each bucket size, so the printed `RESULT` line can be compared against this campaign's
predicted numbers (see the final report for the derivation).

**SPMD, no broadcast protocol.** Every rank runs this exact script with the same CLI
arguments, so unlike `scheduler.py` (which makes runtime, data-dependent decisions rank 0
alone knows and has to broadcast via `tp_driver.Broadcaster`), every rank here reaches the
same calls in the same order on its own -- unbroadcast, the same "this file is both the test
and the worker" pattern `seed_tests/test_graph_capture_tp.py` already uses under gloo. Each
call below (`model.decode`, `graph_runner.decode`, `mtp_runner.verify_and_commit`, eager
`mtp.verify_and_commit`) issues this rank's share of the layer stack's all-reduces, so every
rank must reach it, which SPMD determinism guarantees without a broadcast.

MTP (`SEED_MTP_SERVE=1 SEED_MTP_K=<k>`): per bucket, `graph_ms` times the verify graph alone
(forced drafts) and `round_graph_ms` the full served round (draft graph + verify graph), the
number to compare with the same bucket's decode `graph_ms`. The server path is
`graph_mtp.GraphMTPRunner`; this script drives `MTPVerifyRunner` directly, SPMD.

    srun --jobid=<id> --overlap --environment=/path/to/scratch/runtime.toml \\
      timeout 600 python3 -u graph_bucket_bench.py --tp 4

Add `SEED_STEP_TIMING=1 SEED_STEP_TIMING_EVERY=1` to the environment to also get a
`[step-timing]` line on *every* decode replay with `fill_host_ms=...` from inside
`GraphDecodeRunner.decode` itself -- the direct measurement of what `LaneBlockTables`'
incremental device mirror is claimed to have cut from `fill`'s host time (see that class's
docstring: ~4.6 ms/step measured at b48/max_seq 16384 before this change). This script's own
`fill_host_ms` field (measured with a plain `time.perf_counter()` pair around one `fill` call
directly) should agree with it.

Set `SEED_MTP=1` (and optionally `SEED_MTP_K=1|2|3`, default 3) to also capture and benchmark
the MTP verify path; leave it unset to benchmark decode buckets only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tp
from graph_decode import GraphDecodeRunner
from graph_mtp import MTPVerifyRunner
from model import Model, load_cfg

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
PROMPT_LEN = int(os.environ.get("BENCH_PROMPT_LEN", "32"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
STEPS = int(os.environ.get("BENCH_STEPS", "20"))


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    return Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)


def time_steps(step: Callable[[], torch.Tensor], warmup: int, steps: int) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3, out


def bench_decode_buckets(model: Model, graph_runner: GraphDecodeRunner) -> list[dict]:
    """Per-bucket eager vs. replay wall time at that bucket's exact batch size (real rows,
    no padding beyond what the bucket itself already adds). Toggles `graph_runner.enabled`
    around each phase instead of calling `model.decode` and `graph_runner.decode` as two
    unrelated code paths, so both phases go through the identical dispatcher production uses
    (`GraphDecodeRunner.decode`'s own eager/graph branch) and every rank -- which reaches this
    function deterministically, see module docstring -- flips the same flag at the same point.

    The correctness check and the timing loop are deliberately separate calls, not the same
    one: `prepare()`'s own `validate()` already proved bit-exactness bucket by bucket, bracketed
    by `reset_slots()` on both sides so eager and replay start from the identical DeltaNet
    state; that bracket is repeated here, once per bucket, purely so this script's own `RESULT`
    line carries the same confirmation without a human needing to go find `prepare()`'s log
    line. The *timing* loop below is a separate, un-reset run of `WARMUP + STEPS` calls in a
    row (both phases keep advancing the same live DeltaNet state, eager into graph), which is
    the right thing for a steady-state per-step cost but not a correctness comparison -- so it
    reports timing only, and does not compare its own two final logits tensors.
    """
    prompt = list(range(10, 10 + PROMPT_LEN))
    for s in range(model.max_batch):
        model.begin(s)
        model.prefill(s, prompt, 0)

    was_enabled = graph_runner.enabled
    results = []
    for capacity in graph_runner.buckets:
        slots = list(range(capacity))
        tokens = [(7 * i + 3) % model.cfg.vocab for i in slots]
        positions = [0] * capacity

        graph_runner.reset_slots()
        graph_runner.enabled = False
        check_eager = graph_runner.decode(slots, tokens, positions).clone()
        graph_runner.reset_slots()
        graph_runner.enabled = was_enabled
        check_graph = graph_runner.decode(slots, tokens, positions)
        argmax_agree = bool(torch.equal(check_eager.argmax(-1), check_graph.argmax(-1)))
        graph_runner.reset_slots()

        timed_tokens = [1000 + s for s in slots]
        timed_positions = [PROMPT_LEN] * capacity
        graph_runner.enabled = False
        eager_ms, _ = time_steps(
            lambda: graph_runner.decode(slots, timed_tokens, timed_positions), WARMUP, STEPS
        )
        graph_runner.enabled = was_enabled
        graph_ms, _ = time_steps(
            lambda: graph_runner.decode(slots, timed_tokens, timed_positions), WARMUP, STEPS
        )

        t0 = time.perf_counter()
        graph_runner.fill(
            graph_runner.graphs[capacity].buffers, capacity, slots, timed_tokens, timed_positions
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fill_ms = (time.perf_counter() - t0) * 1e3

        results.append(
            {
                "bucket": capacity,
                "batch": capacity,
                "eager_ms": round(eager_ms, 3),
                "graph_ms": round(graph_ms, 3),
                "speedup": round(eager_ms / graph_ms, 2) if graph_ms else None,
                "fill_host_ms": round(fill_ms, 3),
                "argmax_agree": argmax_agree,
            }
        )
    graph_runner.enabled = was_enabled
    graph_runner.reset_slots()
    return results


def bench_mtp_buckets(model: Model, mtp_runner: MTPVerifyRunner) -> list[dict]:
    """Same split as `bench_decode_buckets`: a bracketed, reset-between-phases correctness
    check per bucket (mirrors `MTPVerifyRunner._validate`'s own snapshot/restore recipe, so
    eager and replay start from the identical DeltaNet state), separate from an un-reset
    timing loop that measures steady-state cost, not correctness.
    """
    k = model.mtp.k
    prompt = list(range(10, 10 + PROMPT_LEN))
    for s in range(model.max_batch):
        model.begin(s)
        model.prefill(s, prompt, 0)

    was_enabled = mtp_runner.enabled
    results = []
    for capacity in mtp_runner.buckets:
        slots = list(range(capacity))
        base_tokens = [(7 * i + 3) % model.cfg.vocab for i in slots]
        draft_tokens = [[(11 * i + 5 * s + 1) % model.cfg.vocab for s in range(k)] for i in slots]
        positions = [1] * capacity

        for slot in slots:
            model.begin(slot)
            model.hidden_scratch[slot].zero_()
        snap_conv = [pool["conv"].clone() for pool in model.pool if "conv" in pool]
        snap_rec = [pool["rec"].clone() for pool in model.pool if "rec" in pool]
        mtp_runner.enabled = False
        check_eager = mtp_runner.verify_and_commit(slots, base_tokens, draft_tokens, positions)
        for pool, before in zip((p for p in model.pool if "conv" in p), snap_conv, strict=True):
            pool["conv"].copy_(before)
        for pool, before in zip((p for p in model.pool if "rec" in p), snap_rec, strict=True):
            pool["rec"].copy_(before)
        for slot in slots:
            model.hidden_scratch[slot].zero_()
        mtp_runner.enabled = was_enabled
        check_graph = mtp_runner.verify_and_commit(slots, base_tokens, draft_tokens, positions)
        commit_agree = check_eager == check_graph

        for s in range(model.max_batch):
            model.begin(s)
            model.prefill(s, prompt, 0)
        timed_positions = [PROMPT_LEN] * capacity

        def call() -> list[list[int]]:
            return mtp_runner.verify_and_commit(slots, base_tokens, draft_tokens, timed_positions)

        def full_round(
            slots: list[int] = slots,
            base_tokens: list[int] = base_tokens,
            positions: list[int] = timed_positions,
        ) -> list[list[int]]:
            return mtp_runner.speculative_decode(slots, base_tokens, positions)

        mtp_runner.enabled = False
        eager_ms, _ = time_steps(call, WARMUP, STEPS)
        mtp_runner.enabled = was_enabled
        graph_ms, _ = time_steps(call, WARMUP, STEPS)
        # Draft graph + verify graph, what one served MTP step costs (`SEED_MTP_SERVE=1`).
        round_graph_ms, _ = time_steps(full_round, WARMUP, STEPS)

        results.append(
            {
                "bucket": capacity,
                "batch": capacity,
                "mtp_k": k,
                "eager_ms": round(eager_ms, 3),
                "graph_ms": round(graph_ms, 3),
                "round_graph_ms": round(round_graph_ms, 3),
                "speedup": round(eager_ms / graph_ms, 2) if graph_ms else None,
                "commit_agree": commit_agree,
            }
        )
    mtp_runner.enabled = was_enabled
    return results


def run_rank(a: argparse.Namespace, rank: int) -> None:
    """The whole benchmark body, run identically (SPMD, see module docstring) by every rank."""
    t0 = time.time()
    model = build(a, rank)
    load_s = time.time() - t0

    prepare_t0 = time.perf_counter()
    graph_runner = GraphDecodeRunner(model)
    captured = graph_runner.prepare()
    prepare_s = time.perf_counter() - prepare_t0

    decode_results = bench_decode_buckets(model, graph_runner) if captured else []

    mtp_captured = None
    mtp_results: list[dict] = []
    mtp_prepare_s = None
    if model.mtp is not None:
        mtp_t0 = time.perf_counter()
        mtp_runner = MTPVerifyRunner(model, model.mtp)
        mtp_captured = mtp_runner.prepare()
        mtp_prepare_s = time.perf_counter() - mtp_t0
        if mtp_captured:
            mtp_results = bench_mtp_buckets(model, mtp_runner)

    if rank == 0:
        out = {
            "tp": a.tp,
            "max_batch": a.batch,
            "buckets": graph_runner.buckets,
            "captured": captured,
            "prepare_s": round(prepare_s, 1),
            "load_s": round(load_s, 1),
            "decode_buckets": decode_results,
            "mtp_enabled": model.mtp is not None,
            "mtp_captured": mtp_captured,
            "mtp_prepare_s": round(mtp_prepare_s, 1) if mtp_prepare_s is not None else None,
            "mtp_buckets": mtp_results,
        }
        print("RESULT " + json.dumps(out), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29519)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--max-seq", type=int, default=4096)
    a = p.parse_args()

    if a.rank:
        run_rank(a, a.rank)
        return

    base = [sys.executable, "-u", os.path.abspath(__file__)]
    for flag in ("tp", "port", "batch"):
        base += [f"--{flag}", str(getattr(a, flag))]
    base += ["--max-seq", str(a.max_seq)]
    workers = [
        subprocess.Popen([*base, "--rank", str(r)], env=os.environ.copy())
        for r in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for proc in workers:
            proc.wait(timeout=180)


if __name__ == "__main__":
    main()
