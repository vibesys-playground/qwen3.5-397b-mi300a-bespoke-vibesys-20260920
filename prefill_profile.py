"""Profile one eager prefill at several shapes (SPMD, TP=N): host time vs device time, per-op.

    srun ... timeout 900 python3 -u prefill_profile.py --tp 4

Rank 0 prints, per case: wall_ms (synced), host_ms (time for `prefill` to return, before the
final sync), kernel_ms (sum of device kernel time from torch.profiler), launches, and the top
ops by device and by host self time. Cases: fresh T=5/128/2048 and a turn-2 continuation
(T=5/128 appended after a 2048-token prefix).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tp
from model import Model, load_cfg

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
OUT = os.environ.get("PROF_OUT", "/tmp/prefill_prof")


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    return Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)


def ids(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 50000, (n,), generator=g).tolist()


def run_case(model: Model, slot: int, prefix: int, t: int) -> tuple[float, float]:
    model.begin(slot)
    if prefix:
        model.prefill(slot, ids(prefix, 1), 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.prefill(slot, ids(t, 2), prefix)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t0) * 1e3


def run_rank(a: argparse.Namespace, rank: int) -> None:
    model = build(a, rank)
    cases = [(0, 5), (0, 128), (0, 2048), (2048, 5), (2048, 128)]
    for prefix, t in cases:  # warmup: JIT, autotune, BLAS
        run_case(model, 0, prefix, t)
        run_case(model, 0, prefix, t)
    results = []
    from torch.profiler import ProfilerActivity, profile

    for prefix, t in cases:
        timings = [run_case(model, 0, prefix, t) for _ in range(3)]
        host_ms = sorted(x[0] for x in timings)[1]
        wall_ms = sorted(x[1] for x in timings)[1]
        model.begin(0)
        if prefix:
            model.prefill(0, ids(prefix, 1), 0)
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
        ) as prof:
            model.prefill(0, ids(t, 2), prefix)
            torch.cuda.synchronize()
        if rank != 0:
            continue
        name = f"p{prefix}_t{t}"
        os.makedirs(OUT, exist_ok=True)
        prof.export_chrome_trace(f"{OUT}/{name}.json")
        ka = prof.key_averages()
        dev_attr = (
            "self_device_time_total"
            if hasattr(ka[0], "self_device_time_total")
            else "self_cuda_time_total"
        )
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernel_ms = sum(e.time_range.elapsed_us() for e in kernels) / 1e3
        by_dev = sorted(ka, key=lambda e: -getattr(e, dev_attr))[:25]
        by_cpu = sorted(ka, key=lambda e: -e.self_cpu_time_total)[:25]
        res = {
            "case": name,
            "wall_ms": round(wall_ms, 1),
            "host_ms": round(host_ms, 1),
            "kernel_ms": round(kernel_ms, 1),
            "launches": len(kernels),
        }
        results.append(res)
        print("CASE " + json.dumps(res), flush=True)
        print(f"--- {name} top by device self time (ms, count)")
        for e in by_dev:
            print(f"  {getattr(e, dev_attr) / 1e3:8.2f} {e.count:6d}  {e.key[:110]}")
        print(f"--- {name} top by host self time (ms, count)")
        for e in by_cpu:
            print(f"  {e.self_cpu_time_total / 1e3:8.2f} {e.count:6d}  {e.key[:110]}")
        sys.stdout.flush()
    if rank == 0:
        print("RESULT " + json.dumps(results), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29531)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-seq", type=int, default=4608)
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--tp",
        str(a.tp),
        "--port",
        str(a.port),
        "--batch",
        str(a.batch),
        "--max-seq",
        str(a.max_seq),
    ]
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
