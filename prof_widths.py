"""Per-component kernel time of eager prefill and mixed steps at several widths (SPMD, TP=N).

Run under rocprofv3 so every kernel is traced; each profiled case is fenced by a device sync
and a 1 s host sleep, so `prof_split.py` splits the trace by time gaps (case order is printed
as `CASE` lines on rank 0, with its pid):

    srun ... rocprofv3 --kernel-trace --output-format csv -d <dir> -- \\
        python3 -u prof_widths.py --tp 4 --cases p64,p256,p512,m48+16,m48+208,m48+464

`pT` is one fresh T-token prefill (`Model.prefill`); `mD+W` is one eager mixed step
(`Model.decode_mixed`): D decode lanes (one token each, lanes at position 8) plus one
W-token prefill, i.e. D+W rows through every shared GEMM and the MoE.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blas_tune
import torch
import tp
from model import Model, load_cfg

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)


def ids(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 50000, (n,), generator=g).tolist()


def run_case(model: Model, case: str, decode_slots: int) -> float:
    """One step of `case`; returns synced wall ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if case.startswith("p"):
        t = int(case[1:])
        model.begin(decode_slots)
        model.prefill(decode_slots, ids(t, 2), 0)
    else:
        d, w = (int(v) for v in case[1:].split("+"))
        lanes = list(range(d))
        model.begin(decode_slots)
        model.decode_mixed(lanes, [1234] * d, [8] * d, [(decode_slots, ids(w, 3), 0)])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def run_rank(a: argparse.Namespace, rank: int) -> None:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)
    cases = a.cases.split(",")
    # the server tunes every captured width's GEMMs at boot (`server.build_model`); do the same
    # for this run's widths, or the default heuristic's wide tiles dominate the GEMM term
    widths = {int(c[1:]) if c.startswith("p") else sum(map(int, c[1:].split("+"))) for c in cases}
    blas_tune.tune(model, batches=sorted({*blas_tune.tuned_batches(model.max_batch), *widths}))
    decode_slots = a.batch - 1
    # decode lanes 0..D-1 hold an 8-token prefix so the mixed cases' decode rows are real
    dmax = max([int(c[1:].split("+")[0]) for c in cases if c.startswith("m")] or [0])
    for s in range(dmax):
        model.begin(s)
        model.prefill(s, ids(8, 10 + s), 0)
    for c in cases:  # warmup: JIT, autotune, BLAS
        run_case(model, c, decode_slots)
        run_case(model, c, decode_slots)
    if rank == 0:
        print(f"PID {os.getpid()}", flush=True)
    for c in cases:
        torch.cuda.synchronize()
        time.sleep(1.0)
        ms = run_case(model, c, decode_slots)
        time.sleep(1.0)
        if rank == 0:
            print(f"CASE {c} wall_ms={ms:.1f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29533)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=4608)
    p.add_argument("--cases", default="p64,p256,p512,m48+16,m48+208,m48+464")
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [sys.executable, "-u", os.path.abspath(__file__)]
    base += ["--tp", str(a.tp), "--port", str(a.port), "--batch", str(a.batch)]
    base += ["--max-seq", str(a.max_seq), "--cases", a.cases]
    workers = [
        subprocess.Popen([*base, "--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for proc in workers:
            proc.wait(timeout=300)


if __name__ == "__main__":
    main()
