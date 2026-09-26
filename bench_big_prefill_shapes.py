"""Captured-prefill shape curve for long prefill steps (round 15, W5 big-prefill).

Captures each shape in `--shapes` as its own `PrefillGraphRunner` sharing one graph pool (the
production `CudaGraphBackend` layout), so the per-shape device-memory delta is the incremental
cost that shape adds on top of the ones before it. `prepare()` checks each replay against its
uncaptured step and the eager prefill on every rank before timing.

Per shape it reports:
- `full`: every row real at full width, resuming at `--full-start` cached tokens.
- `trace`: rows drawn from a benchmark `*.turns.jsonl` (new-token delta chunked at the shape's
  width, cached prefix as the start position), `--samples` independent draws.
ms is the median replay wall over `--reps` synced replays; ms/token uses area (full) or real
tokens (trace).

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
from graph_prefill import PrefillGraphRunner, Shape, parse_shapes

import server


def _trace_rows(path: str) -> list[tuple[int, int]]:
    """(cached_start, new_tokens) per successful turn."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [
        (r["cached_tokens"], r["prompt_tokens"] - r["cached_tokens"])
        for r in rows
        if r["ok"] and r["prompt_tokens"] > r["cached_tokens"]
    ]


def _draw(trace: list[tuple[int, int]], shape: Shape, rng: random.Random) -> list[tuple[int, int]]:
    """`shape.rows` (start, length) rows: each a random turn's first chunk at this width."""
    out = []
    for _ in range(shape.rows):
        start, n = rng.choice(trace)
        out.append((start, min(n, shape.width)))
    return out


def _calls(vocab: int, entries: list[tuple[int, int]]) -> list[tuple[int, list[int], int]]:
    return [
        (slot, [((17 * slot + 13 * j + 1) % vocab) for j in range(n)], start)
        for slot, (start, n) in enumerate(entries)
    ]


def _time(runner: PrefillGraphRunner, shape: Shape, entries, reps: int) -> float:  # noqa: ANN001
    runner._reset_lanes()
    buf, replay = runner.graphs[shape]
    runner.fill(buf, _calls(runner.model.cfg.vocab, entries))
    replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _gib(x: int) -> float:
    return round(x / 2**30, 3)


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
    shapes.sort(key=lambda s: (s.area, s.width))
    runners: dict[Shape, PrefillGraphRunner] = {}
    mem: dict[str, dict] = {}
    torch.cuda.synchronize()
    free0, total = torch.cuda.mem_get_info(device)
    mem["_before"] = {
        "free_gib": _gib(free0),
        "alloc_gib": _gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
    }
    for shape in shapes:
        free_a, _ = torch.cuda.mem_get_info(device)
        t0 = time.perf_counter()
        runner = PrefillGraphRunner(
            model, backend, decode.lane_tables[device], decode._dirty, shapes=[shape]
        )
        ok = runner.prepare()
        torch.cuda.synchronize()
        free_b, _ = torch.cuda.mem_get_info(device)
        name = f"{shape.rows}x{shape.width}"
        mem[name] = {
            "ok": ok,
            "capture_s": round(time.perf_counter() - t0, 1),
            "delta_free_gib": _gib(free_a - free_b),
            "free_gib": _gib(free_b),
            "alloc_gib": _gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
        }
        print(f"MEM rank={a.rank} {name} {json.dumps(mem[name])}", flush=True)
        if ok:
            runners[shape] = runner

    trace = _trace_rows(a.trace_jsonl)
    results = []
    for shape, runner in runners.items():
        name = f"{shape.rows}x{shape.width}"
        full = [(a.full_start, shape.width)] * shape.rows
        full_ms = _time(runner, shape, full, a.reps)
        rng = random.Random(1000 + shape.area)
        draws = []
        for _ in range(a.samples):
            entries = _draw(trace, shape, rng)
            draws.append((_time(runner, shape, entries, a.reps), sum(n for _, n in entries)))
        trace_ms = statistics.median(ms for ms, _ in draws)
        real = statistics.mean(n for _, n in draws)
        res = {
            "shape": name,
            "area": shape.area,
            "full_ms": round(full_ms, 3),
            "full_ms_per_tok": round(full_ms / shape.area, 4),
            "trace_ms": round(trace_ms, 3),
            "trace_real_tokens": round(real, 1),
            "trace_fill": round(real / shape.area, 3),
            "trace_ms_per_real_tok": round(statistics.median(ms / max(n, 1) for ms, n in draws), 4),
        }
        results.append(res)
        if a.rank == 0:
            print("SHAPE " + json.dumps(res), flush=True)
    if a.rank == 0:
        print("RESULT " + json.dumps({"shapes": results, "mem_rank0": mem}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29647)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--shapes", required=True)
    p.add_argument("--trace-jsonl", required=True)
    p.add_argument("--full-start", type=int, default=512)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--reps", type=int, default=5)
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
