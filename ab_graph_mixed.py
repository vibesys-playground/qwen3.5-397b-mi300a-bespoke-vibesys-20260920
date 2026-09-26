"""Paired captured-replay A/B of `SEED_AR_SP` in one process (SPMD, no profiler).

Boots each rank as the server does with `SEED_AR_SP=1` (so the SP buffers exist), captures the
mixed graphs once with the SP all-reduce off and once on, then times the two sets of graphs in
alternating rounds, so node, clocks and weights are shared. Two timings per shape and variant:
`loop` = `--reps` replays back to back, one sync; `step` = fill + replay + sync per replay, as a
served step does. Rank 0 prints one `AB` line per shape.

    srun ... env $FLAGS SEED_AR_SP=1 SEED_MIXED_GRAPH_DECODE=48 SEED_MIXED_GRAPH_TOTALS=256,512 \\
        python3 -u ab_graph_mixed.py --tp 4 --shapes 48+1x208,48+1x464
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allreduce_custom
import torch
from graph_decode import GraphDecodeRunner

import server


def run_rank(a: argparse.Namespace) -> None:
    assert allreduce_custom.SP, "run with SEED_AR_SP=1"
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
    runner = GraphDecodeRunner(model)
    allreduce_custom.SP = False
    runner.prepare()
    mixed = runner.mixed_runner
    assert mixed is not None and mixed.enabled
    names = a.shapes.split(",")
    graphs = {}
    graphs["base"] = {s.name: mixed.graphs[s] for s in mixed.shapes if s.name in names}
    shapes = {s.name: s for s in mixed.shapes}
    allreduce_custom.SP = True
    assert mixed.prepare(), "SP capture/validation failed"
    graphs["sp"] = {s.name: mixed.graphs[s] for s in mixed.shapes if s.name in names}
    res = {(v, n, k): [] for v in graphs for n in names for k in ("loop", "step")}
    for _ in range(a.rounds):
        for v in ("base", "sp"):
            for n in names:
                shape = shapes[n]
                slots, tokens, calls = mixed._synthetic(shape)
                mixed._seed_lanes(slots, calls)
                dbuf, pbuf, replay = graphs[v][n]
                mixed.fill(dbuf, pbuf, slots, tokens, [3] * len(slots), calls)
                replay()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    replay()
                torch.cuda.synchronize()
                res[(v, n, "loop")].append((time.perf_counter() - t0) * 1e3 / a.reps)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    mixed.fill(dbuf, pbuf, slots, tokens, [3] * len(slots), calls)
                    replay()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1e3)
                res[(v, n, "step")].append(statistics.median(ts))
    if a.rank == 0:
        for n in names:
            parts = []
            for k in ("loop", "step"):
                b = statistics.median(res[("base", n, k)])
                s = statistics.median(res[("sp", n, k)])
                parts.append(f"{k} base {b:.2f} sp {s:.2f} delta {s - b:+.2f} ms")
            print(f"AB {n}: " + "; ".join(parts), flush=True)
        print("RAW " + repr({f"{k}": [round(x, 2) for x in v] for k, v in res.items()}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29537)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--shapes", default="48+1x208,48+1x464")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
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
        subprocess.Popen([*argv, "--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)
    ]
    try:
        run_rank(a)
    finally:
        for proc in workers:
            proc.wait(timeout=600)


if __name__ == "__main__":
    main()
