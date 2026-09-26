"""Decode all-reduce microbenchmark: per-call time inside a 120-call graph, 4 ranks, and
bit-exactness of each fused AR + residual add + RMSNorm variant against the production one.

Paths at each row count (hidden 4096, bf16):
- `read`: production `ar_add_rmsnorm` (publish locally, read three peers over xGMI).
- `push`: `SEED_AR_PUSH` (write into peers' receive buffers, spin and read locally).
- `sp`: `sp_ar_add_rmsnorm` (reduce-scatter + norm on rows/4 + all-gather), rows % 4 == 0.
- `push_read_mix`: push and read calls alternating in one graph (shared device counter).

    srun ... env SEED_AR_SP=1 SEED_AR_GRAPH_SAFE=1 SEED_AR_PUSH=1 \\
      python3 -u bench_ar_decode.py --rows 16,48,96,144,192,256
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from bench_custom_allreduce import bench_graph

HIDDEN = 4096
EPS = 1e-6


def bench_rows(car, w: torch.Tensor, rows: int, a: argparse.Namespace) -> dict:  # noqa: ANN001
    import allreduce_custom as ac

    device = w.device
    g = torch.Generator(device=device).manual_seed(1234 + a.rank)
    m = torch.randn(rows, HIDDEN, device=device, generator=g).to(torch.bfloat16)
    g0 = torch.Generator(device=device).manual_seed(99)  # residual identical on all ranks
    resid = torch.randn(rows, HIDDEN, device=device, generator=g0).to(torch.bfloat16)

    def read(_=None):
        ac.PUSH = False
        return car.ar_add_rmsnorm(m, resid, w, EPS)

    def push(_=None):
        ac.PUSH, ac.PUSH_SYS_FENCE = True, True
        return car.ar_add_rmsnorm(m, resid, w, EPS)

    def push_nf(_=None):
        ac.PUSH, ac.PUSH_SYS_FENCE = True, False
        return car.ar_add_rmsnorm(m, resid, w, EPS)

    def stale_check(fn, calls: int = 40, replays: int = 20) -> bool:
        """Every call in a graph gets fresh inputs; outputs must equal the eager read path."""
        gin = torch.Generator(device=device).manual_seed(4321 + a.rank)
        ins = torch.randn(calls, rows, HIDDEN, device=device, generator=gin).to(torch.bfloat16)
        refs = []
        for k in range(calls):
            m.copy_(ins[k])
            refs.append(read()[1].clone())
        outs = torch.empty(calls, rows, HIDDEN, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            m.copy_(ins[0])
            fn()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for k in range(calls):
                m.copy_(ins[k])
                outs[k].copy_(fn()[1])
        ok = True
        for _ in range(replays):
            outs.zero_()
            g.replay()
            torch.cuda.synchronize()
            ok = ok and all(torch.equal(outs[k], refs[k]) for k in range(calls))
        m.copy_(ins[0])
        return ok

    rx, rh = read()
    px, ph = push()
    torch.cuda.synchronize()
    r = {
        "push_x_bitexact": bool(torch.equal(px, rx)),
        "push_h_bitexact": bool(torch.equal(ph, rh)),
    }
    # The capture-time Python flag is baked per call, so the mixed graph interleaves both.
    state = {"k": 0}

    def mix(_=None):
        state["k"] += 1
        return push() if state["k"] % 2 else read()

    base_m = m.clone()
    r["push_fresh_inputs_exact"] = stale_check(push)
    r["push_nf_fresh_inputs_exact"] = stale_check(push_nf)
    m.copy_(base_m)
    paths = {"read": read, "push": push, "push_nf": push_nf, "push_read_mix": mix}
    if rows % 4 == 0 and rows // 4 <= ac.SP_MAX_ROWS:
        s = rows // 4
        shard = resid[a.rank * s : (a.rank + 1) * s].contiguous()
        paths["sp"] = lambda _: car.sp_ar_add_rmsnorm(m, shard, w, EPS)
    for name, fn in paths.items():
        r[name] = round(bench_graph(fn, None, a.calls, a.replays, 5)["median_us"], 2)
    px2, ph2 = push()
    torch.cuda.synchronize()
    r["push_bitexact_after_graphs"] = bool(torch.equal(ph2, rh) and torch.equal(px2, rx))
    ac.PUSH, ac.PUSH_SYS_FENCE = False, True
    return r


def run_rank(a: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    device = torch.device(f"cuda:{a.rank}")
    torch.cuda.set_device(a.rank)
    dist.init_process_group(backend="nccl", rank=a.rank, world_size=4, device_id=device)
    import allreduce_custom

    assert allreduce_custom.PUSH and allreduce_custom.GRAPH_SAFE
    car = allreduce_custom.CustomAllReduce(a.rank, 4, device)
    gw = torch.Generator(device=device).manual_seed(7)
    w = (torch.randn(HIDDEN, device=device, generator=gw) * 0.1).to(torch.bfloat16)
    res = {rows: bench_rows(car, w, rows, a) for rows in [int(r) for r in a.rows.split(",")]}
    print(f"RESULT rank={a.rank} " + json.dumps(res), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--rows", default="16,48,96,144,192,256")
    p.add_argument("--calls", type=int, default=120)
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--port", type=int, default=29651)
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
    procs = [subprocess.Popen([*argv, "--rank", str(r)]) for r in range(1, 4)]
    try:
        run_rank(a)
    finally:
        for pr in procs:
            pr.wait(timeout=600)


if __name__ == "__main__":
    main()
