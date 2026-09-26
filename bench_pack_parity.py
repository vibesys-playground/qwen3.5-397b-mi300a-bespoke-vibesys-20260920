"""Packed vs per-row captured prefill on real turn deltas (round 15, W9 `SEED_PREFILL_PACK`).

Captures each shape in `--shapes` twice, per-row and packed (`PrefillGraphRunner(pack=...)`);
both `prepare()` on every rank (replay vs uncaptured step vs eager; packed shapes validate a
multi-segment call). Then, per shape and draw, from the same lane state:

- `same`: calls that fit the per-row layout (`rows` turns, each chunked at `width`) through
  the per-row graph and the packed graph of the same shape. Same GEMM `M`, so every DeltaNet
  and attention input should match bit for bit.
- `dense`: as many turns as fit the packed area (aligned) and slots, through the packed graph
  and through the smallest per-row shape among `--shapes` that holds them (different `M`), and
  through the eager `Model.prefill_batch`.

Reported per rank: logits bit-equal rows, max |diff| relative to max |logit|, argmax agreement,
DeltaNet conv/rec state bit-equal and correlation, and replay ms. Rank 0 prints `PAIR` lines.

SPMD: rank 0 spawns ranks 1..tp-1 with the same argv.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from graph_decode import CudaGraphBackend, GraphDecodeRunner
from graph_prefill import PrefillGraphRunner, Shape, _corr, pack_align, parse_shapes

import server


def _trace(path: str) -> list[tuple[int, int]]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [
        (r["cached_tokens"], r["prompt_tokens"] - r["cached_tokens"])
        for r in rows
        if r["ok"] and r["prompt_tokens"] > r["cached_tokens"]
    ]


def _calls(vocab: int, entries: list[tuple[int, int]]) -> list[tuple[int, list[int], int]]:
    return [
        (slot, [((17 * slot + 13 * j + 1) % vocab) for j in range(n)], start)
        for slot, (start, n) in enumerate(entries)
    ]


def _cmp(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict:
    x = torch.cat([t.float() for t in a])
    y = torch.cat([t.float() for t in b])
    eq_rows = sum(bool(torch.equal(p, q)) for p, q in zip(a, b, strict=True))
    rel = float((x - y).abs().max() / y.abs().max().clamp(min=1e-6))
    arg = float((x.argmax(-1) == y.argmax(-1)).float().mean())
    return {"rows": len(a), "bit_equal_rows": eq_rows, "max_rel": rel, "argmax_agree": arg}


def _cmp_state(a, b) -> dict:  # noqa: ANN001
    eq = all(torch.equal(p, q) for (pc, pr), (qc, qr) in zip(a, b, strict=True) for p, q in ((pc, qc), (pr, qr)))
    corr = min(min(_corr(pc, qc), _corr(pr, qr)) for (pc, pr), (qc, qr) in zip(a, b, strict=True))
    return {"state_bit_equal": eq, "state_corr": corr}


def _run(runner: PrefillGraphRunner, shape: Shape, calls, reps: int):  # noqa: ANN001, ANN202
    """Logits and DeltaNet state of one replay from the saved start state, plus median ms."""
    start = runner._deltanet_state()
    got = runner.prefill_batch(shape, calls)
    state = runner._deltanet_state()
    samples = []
    for _ in range(reps):
        runner._rewind(start, calls)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner.prefill_batch(shape, calls)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    runner._rewind(start, calls)
    return [g.clone() for g in got], state, statistics.median(samples) if samples else 0.0


def _eager(model, runner: PrefillGraphRunner, calls):  # noqa: ANN001, ANN202
    start = runner._deltanet_state()
    got = model.prefill_batch([(s, list(ids), p) for s, ids, p in calls])
    state = runner._deltanet_state()
    runner._rewind(start, calls)
    return [g.float().reshape(1, -1).clone() for g in got], state


def run_rank(a: argparse.Namespace) -> None:
    args = argparse.Namespace(
        model_path=a.model_path,
        dtype="bfloat16",
        tp=a.tp,
        tp_port=a.port,
        max_seq_len=a.max_seq,
        max_batch=a.batch,
        devices="",
        rank=a.rank,
    )
    model = server.build_model(args, a.rank)
    decode = GraphDecodeRunner(model, prefill_graphs=False, mixed_graphs=False)
    device = model.devices[-1]
    backend = CudaGraphBackend()
    shapes = parse_shapes(a.shapes, model.max_batch, model.max_seq)
    lt, dirty = decode.lane_tables[device], decode._dirty
    per_row = PrefillGraphRunner(model, backend, lt, dirty, shapes=shapes, pack=False)
    packed = PrefillGraphRunner(model, backend, lt, dirty, shapes=shapes, pack=True)
    ok_r, ok_p = per_row.prepare(), packed.prepare()
    print(f"PREPARE rank={a.rank} per_row={ok_r} packed={ok_p} "
          f"packed_shapes={[s.label for s in packed.shapes]}", flush=True)
    if not (ok_r and ok_p):
        return
    trace = _trace(a.trace_jsonl)
    vocab = model.cfg.vocab
    rng = random.Random(7)
    by_area = sorted(per_row.shapes, key=lambda s: (s.area, s.width))
    for shape in packed.shapes:
        if not shape.packed:
            continue
        for draw in range(a.samples):
            # `same`: one chunk per row at the shape's width.
            same = [rng.choice(trace) for _ in range(shape.rows)]
            same = [(st, min(n, shape.width)) for st, n in same]
            calls = _calls(vocab, same)
            per_row._reset_lanes()
            r_log, r_state, r_ms = _run(per_row, shape, calls, a.reps)
            p_log, p_state, p_ms = _run(packed, shape, calls, a.reps)
            rec = {"shape": shape.label, "draw": draw, "kind": "same", "rank": a.rank,
                   "real": sum(n for _, n in same), "per_row_ms": round(r_ms, 2),
                   "packed_ms": round(p_ms, 2), **_cmp(p_log, r_log), **_cmp_state(p_state, r_state)}
            print("PAIR " + json.dumps(rec), flush=True)

            # `dense`: fill the packed area with whole turns (each capped at 512, the chunk).
            dense, used = [], 0
            while len(dense) < shape.segs:
                st, n = rng.choice(trace)
                n = min(n, 512, shape.area - used)
                if n <= 0 or used + pack_align(n) > shape.area:
                    break
                dense.append((st, n))
                used += pack_align(n)
            calls = _calls(vocab, dense)
            lengths = [n for _, n in dense]
            ref_shape = next(
                (s for s in by_area if s.rows >= len(lengths) and s.width >= max(lengths)), None
            )
            per_row._reset_lanes()
            p_log, p_state, p_ms = _run(packed, shape, calls, a.reps)
            e_log, e_state = _eager(model, packed, calls)
            rec = {"shape": shape.label, "draw": draw, "kind": "dense", "rank": a.rank,
                   "segs": len(dense), "real": sum(lengths), "fill": round(sum(lengths) / shape.area, 3),
                   "packed_ms": round(p_ms, 2), "vs_eager": {**_cmp(p_log, e_log), **_cmp_state(p_state, e_state)}}
            if ref_shape is not None:
                r_log, r_state, r_ms = _run(per_row, ref_shape, calls, a.reps)
                rec["per_row_shape"] = ref_shape.label
                rec["per_row_ms"] = round(r_ms, 2)
                rec["vs_per_row"] = {**_cmp(p_log, r_log), **_cmp_state(p_state, r_state)}
            print("PAIR " + json.dumps(rec), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29651)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--shapes", required=True)
    p.add_argument("--trace-jsonl", required=True)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
        ),
    )
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
    workers = [
        subprocess.Popen([*argv, "--rank", str(rank)], env=os.environ.copy())
        for rank in range(1, a.tp)
    ]
    try:
        run_rank(a)
    finally:
        for proc in workers:
            proc.wait(timeout=1800)


if __name__ == "__main__":
    main()
