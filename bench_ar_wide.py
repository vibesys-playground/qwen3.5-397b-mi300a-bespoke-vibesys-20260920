"""Wide (prefill) SP all-reduce microbenchmark: where the time of `sp_ar_add_rmsnorm` goes at
512-2048 rows, per call inside a graph of `--calls` calls, 4 ranks.

Arms: `sp` (production kernel), `sp_nocopy` (skips the all-gather copy into `h`; wrong
output, timing only), `sp_w8` (8 warps; different `tl.sum` order, timing only), and `rccl`
(fp32 RCCL + add_rmsnorm, today's > 1024-row path).

    srun ... env SEED_CUSTOM_ALLREDUCE=1 SEED_AR_GRAPH_SAFE=1 SEED_AR_SP=1 \\
      SEED_AR_SP_MAX_TOKENS=2048 python3 -u bench_ar_wide.py --rows 512,1024,2048
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


def fresh_check(m, ref_fn, fn, device, rank, rows, calls: int = 16, replays: int = 10) -> bool:  # noqa: ANN001
    """Every call in a graph gets fresh inputs; `fn`'s normed output must equal `ref_fn`'s
    eager output for the same inputs (catches reads of a previous call's data)."""
    gin = torch.Generator(device=device).manual_seed(4321 + rank)
    ins = torch.randn(calls, rows, HIDDEN, device=device, generator=gin).to(torch.bfloat16)
    saved = m.clone()
    refs = []
    for k in range(calls):
        m.copy_(ins[k])
        refs.append(ref_fn(None)[1].clone())
    outs = torch.empty(calls, rows, HIDDEN, dtype=torch.bfloat16, device=device)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn(None)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for k in range(calls):
            m.copy_(ins[k])
            outs[k].copy_(fn(None)[1])
    ok = True
    for _ in range(replays):
        outs.zero_()
        g.replay()
        torch.cuda.synchronize()
        ok = ok and all(torch.equal(outs[k], refs[k]) for k in range(calls))
    m.copy_(saved)
    return ok


def run_rank(a: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    device = torch.device(f"cuda:{a.rank}")
    torch.cuda.set_device(a.rank)
    dist.init_process_group(backend="nccl", rank=a.rank, world_size=4, device_id=device)
    import allreduce_custom as ac
    import rmsnorm_fused as rf

    car = ac.CustomAllReduce(a.rank, 4, device)
    gw = torch.Generator(device=device).manual_seed(7)
    w = (torch.randn(HIDDEN, device=device, generator=gw) * 0.1).to(torch.bfloat16)
    res = {}
    for rows in [int(r) for r in a.rows.split(",")]:
        g = torch.Generator(device=device).manual_seed(1234 + a.rank)
        m = torch.randn(rows, HIDDEN, device=device, generator=g).to(torch.bfloat16)
        g0 = torch.Generator(device=device).manual_seed(99)
        resid = torch.randn(rows, HIDDEN, device=device, generator=g0).to(torch.bfloat16)
        s = rows // 4
        shard = resid[a.rank * s : (a.rank + 1) * s].contiguous()
        r = {}

        def sp(_):
            return car.sp_ar_add_rmsnorm(m, shard, w, EPS)

        def sp_nocopy(_):
            rf.SP_COPY_AG = False
            try:
                return car.sp_ar_add_rmsnorm(m, shard, w, EPS)
            finally:
                rf.SP_COPY_AG = True

        def sp_w8(_):
            old = ac.SP_WARPS
            ac.SP_WARPS = 8
            try:
                return car.sp_ar_add_rmsnorm(m, shard, w, EPS)
            finally:
                ac.SP_WARPS = old

        def sp_relax(_):
            rf.SP_RELAXED_SPIN = True
            try:
                return car.sp_ar_add_rmsnorm(m, shard, w, EPS)
            finally:
                rf.SP_RELAXED_SPIN = False

        def sp_relax1(_):
            rf.SP_RELAXED_SPIN, rf.SP_ONE_RELEASE = True, True
            try:
                return car.sp_ar_add_rmsnorm(m, shard, w, EPS)
            finally:
                rf.SP_RELAXED_SPIN, rf.SP_ONE_RELEASE = False, False

        def mk_rpp(rpp):
            def fn(_):
                rf.SP_RELAXED_SPIN, rf.SP_ONE_RELEASE = False, True
                rf.SP_ROWS_PER_PROG, rf.SP_ROWS_PER_PROG_MIN = rpp, 1
                try:
                    return car.sp_ar_add_rmsnorm(m, shard, w, EPS)
                finally:
                    rf.SP_ONE_RELEASE, rf.SP_ROWS_PER_PROG = False, 1
            return fn

        rpp_arms = {f"sp_r1_rpp{k}": mk_rpp(k) for k in (1, 2, 4, 8) if s % k == 0}

        ref_x, ref_h = sp(None)
        rx, rh = sp_relax(None)
        torch.cuda.synchronize()
        r["relax_exact"] = bool(torch.equal(rx, ref_x) and torch.equal(rh, ref_h))
        r["relax_fresh_inputs_exact"] = fresh_check(m, sp, sp_relax, device, a.rank, rows)
        r["relax1_fresh_inputs_exact"] = fresh_check(m, sp, sp_relax1, device, a.rank, rows)
        for name, fn in rpp_arms.items():
            r[name + "_fresh_exact"] = fresh_check(m, sp, fn, device, a.rank, rows)

        def rccl(_):
            wd = m.float()
            dist.all_reduce(wd)
            return rf.add_rmsnorm(resid, wd.to(torch.bfloat16), w, EPS)

        for name, fn in {"sp": sp, "sp_relax1": sp_relax1, **rpp_arms, "sp_nocopy": sp_nocopy, "rccl": rccl}.items():
            dist.barrier()
            r[name] = round(bench_graph(fn, None, a.calls, a.replays, 3)["median_us"], 1)
        res[rows] = r
        print(f"WIDE rank={a.rank} rows={rows} " + json.dumps(r), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--rows", default="96,512,1024,2048")
    p.add_argument("--calls", type=int, default=40)
    p.add_argument("--replays", type=int, default=15)
    p.add_argument("--port", type=int, default=29671)
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
            pr.wait(timeout=900)


if __name__ == "__main__":
    main()
