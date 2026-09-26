"""Replay captured mixed graphs one at a time, for a kernel trace of the served step (SPMD).

Builds each rank exactly as the server does (`server.build_model`, which BLAS-tunes, then
`GraphDecodeRunner.prepare`, which captures decode, prefill and mixed graphs under the
environment's flags), then for each requested mixed shape: seed its synthetic lanes, fill,
replay `--reps` times, fenced by a device sync and 1 s host sleeps so `prof_split.py` can
split the trace by gaps. Rank 0 prints `PID` and one `CASE <shape> wall_ms=<per replay>`.

    srun ... rocprofv3 --kernel-trace --output-format csv -d <dir> -- \\
        python3 -u prof_graph_mixed.py --tp 4 --shapes 48+1x208,48+1x464 --reps 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from graph_decode import GraphDecodeRunner

import server


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
    runner = GraphDecodeRunner(model)
    runner.prepare()
    mixed = runner.mixed_runner
    assert mixed is not None, "no mixed runner: set SEED_MIXED_GRAPH=1"
    by_name = {s.name: s for s in mixed.shapes}
    if a.rank == 0:
        print(f"PID {os.getpid()}", flush=True)
    for name in a.shapes.split(","):
        shape = by_name[name]
        slots, tokens, calls = mixed._synthetic(shape)
        mixed._seed_lanes(slots, calls)
        dbuf, pbuf, replay = mixed.graphs[shape]
        mixed.fill(dbuf, pbuf, slots, tokens, [3] * len(slots), calls)
        replay()
        torch.cuda.synchronize()
        time.sleep(1.0)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3 / a.reps
        time.sleep(1.0)
        if a.rank == 0:
            print(f"CASE {name} wall_ms={ms:.1f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29537)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=16384)
    p.add_argument("--shapes", default="48+1x208,48+1x464")
    p.add_argument("--reps", type=int, default=1)
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
