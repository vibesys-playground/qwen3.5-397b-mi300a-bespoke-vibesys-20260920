"""Microbenchmark + bit-exactness check: `SEED_AR_SP`'s reduce-scatter/norm/all-gather kernel
vs the paths the mixed step uses today, per call inside a 120-call graph, 4 ranks.

Today at <= 256 rows: `ar_add_rmsnorm` (one-shot + add + norm, one Triton kernel). Above:
RCCL fp32 all-reduce + `add_rmsnorm`, or with `SEED_AR_WIDE` the one-shot + `add_rmsnorm`.

    srun ... env SEED_AR_SP=1 SEED_AR_GRAPH_SAFE=1 python3 -u bench_ar_sp.py \
        --rows 8,16,24,32,48,64,80,96
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
    """Bit-exactness and per-call graph timing of every path at `rows` rows."""
    import allreduce_custom
    import rmsnorm_fused

    device = w.device
    g = torch.Generator(device=device).manual_seed(1234 + a.rank)
    m = torch.randn(rows, HIDDEN, device=device, generator=g).to(torch.bfloat16)
    g0 = torch.Generator(device=device).manual_seed(99)  # residual identical on all ranks
    resid = torch.randn(rows, HIDDEN, device=device, generator=g0).to(torch.bfloat16)
    s = rows // 4
    shard = resid[a.rank * s : (a.rank + 1) * s].contiguous()

    # reference: one-shot + add_rmsnorm (bit-exact equal to ar_add_rmsnorm); above the one-shot
    # cap (`--max-elems`) the RCCL fp32 sum in the same bf16 rounding is the reference instead
    if m.numel() <= car.max_elems:
        red = car(m.clone())
    else:
        red = m.float()
        dist.all_reduce(red)
        red = red.to(torch.bfloat16)
    ref_x, ref_h = rmsnorm_fused.add_rmsnorm(resid, red, w, EPS)
    xo, h = car.sp_ar_add_rmsnorm(m, shard, w, EPS)
    torch.cuda.synchronize()
    own = slice(a.rank * s, (a.rank + 1) * s)
    r = {
        "sp_h_bitexact": bool(torch.equal(h, ref_h)),
        "sp_x_bitexact": bool(torch.equal(xo, ref_x[own])),
        "sp_h_rows_bad": [
            q
            for q in range(4)
            if not torch.equal(h[q * s : (q + 1) * s], ref_h[q * s : (q + 1) * s])
        ],
        "sp_h_maxdiff": float((h.float() - ref_h.float()).abs().max()),
        "sp_h_frac_diff": float((h != ref_h).float().mean()),
    }
    if rows <= allreduce_custom.MAX_FLAG_BLOCKS:
        fx, fh = car.ar_add_rmsnorm(m, resid, w, EPS)
        r["fused_h_bitexact_vs_ref"] = bool(torch.equal(fh, ref_h))
    # RCCL fp32 path (r9 above 256 rows)
    wide = m.float()
    dist.all_reduce(wide)
    rx, rh = rmsnorm_fused.add_rmsnorm(resid, wide.to(torch.bfloat16), w, EPS)
    r["rccl_h_maxdiff_vs_ref"] = float((rh.float() - ref_h.float()).abs().max())
    r["rccl_h_frac_diff"] = float((rh != ref_h).float().mean())

    def rccl_path(_):
        wd = m.float()
        dist.all_reduce(wd)
        return rmsnorm_fused.add_rmsnorm(resid, wd.to(torch.bfloat16), w, EPS)

    def oneshot_path(_):
        return rmsnorm_fused.add_rmsnorm(resid, car(m.clone()), w, EPS)

    def sp_path(_):
        return car.sp_ar_add_rmsnorm(m, shard, w, EPS)

    paths = {"rccl_addnorm": rccl_path, "sp": sp_path}
    if m.numel() <= car.max_elems:
        paths["oneshot_addnorm"] = oneshot_path
    if rows <= allreduce_custom.MAX_FLAG_BLOCKS:
        paths["fused_ar_norm"] = lambda _: car.ar_add_rmsnorm(m, resid, w, EPS)
    for name, fn in paths.items():
        r[name] = bench_graph(fn, None, a.calls, a.replays, 5)["median_us"]
    # repeated calls stay exact (slot/counter protocol across replays)
    xo2, h2 = car.sp_ar_add_rmsnorm(m, shard, w, EPS)
    torch.cuda.synchronize()
    r["sp_h_bitexact_after_graphs"] = bool(torch.equal(h2, ref_h))
    return r


def run_rank(a: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    device = torch.device(f"cuda:{a.rank}")
    torch.cuda.set_device(a.rank)
    dist.init_process_group(backend="nccl", rank=a.rank, world_size=4, device_id=device)
    import allreduce_custom

    assert allreduce_custom.SP and allreduce_custom.GRAPH_SAFE
    car = allreduce_custom.CustomAllReduce(a.rank, 4, device, max_elems=1 << 22)
    gw = torch.Generator(device=device).manual_seed(7)  # replicated, like the model's norms
    w = (torch.randn(HIDDEN, device=device, generator=gw) * 0.1).to(torch.bfloat16)
    res = {rows: bench_rows(car, w, rows, a) for rows in [int(r) for r in a.rows.split(",")]}
    if a.rank == 0:
        print("RESULT " + json.dumps(res), flush=True)
    else:
        flags = [v["sp_h_bitexact"] and v["sp_x_bitexact"] for v in res.values()]
        print(f"rank {a.rank} exact {flags}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--rows", default="8,16,24,32,48,64,80,96")
    p.add_argument("--calls", type=int, default=120)
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--port", type=int, default=29641)
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
