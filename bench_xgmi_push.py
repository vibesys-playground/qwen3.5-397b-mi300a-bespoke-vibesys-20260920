"""xGMI push bandwidth probe: every rank writes `rows` x 4096 bf16 into each of the three
peers' `SEED_AR_PUSH` receive buffers at the same time, split over `rows * split` programs.
No flags: this measures the store path only (per-call time in a graph of 120 calls).

    srun ... env SEED_CUSTOM_ALLREDUCE=1 SEED_AR_GRAPH_SAFE=1 SEED_AR_PUSH=1 \\
      python3 -u bench_xgmi_push.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from bench_custom_allreduce import bench_graph


@triton.jit
def _push(m_ptr, p0, p1, p2, p3, cols, SELF: tl.constexpr, SPLIT: tl.constexpr, CH: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPLIT
    k = pid % SPLIT
    c = k * CH + tl.arange(0, CH)
    live = c < cols
    m = tl.load(m_ptr + row * cols + c, mask=live)
    off = SELF * 256 * 4096 + row * cols
    if SELF != 0:
        tl.store(p0 + off + c, m, mask=live)
    if SELF != 1:
        tl.store(p1 + off + c, m, mask=live)
    if SELF != 2:
        tl.store(p2 + off + c, m, mask=live)
    if SELF != 3:
        tl.store(p3 + off + c, m, mask=live)


def main_rank(rank: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29661")
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=4, device_id=dev)
    import allreduce_custom

    car = allreduce_custom.CustomAllReduce(rank, 4, dev)
    recv, _ = car._push_ptrs
    res = {}
    for rows in (16, 96, 192):
        m = torch.randn(rows, 4096, device=dev).to(torch.bfloat16)
        for split in (1, 2, 4, 8):
            for warps in (4, 8):
                ch = 4096 // split

                def fn(_):
                    _push[(rows * split,)](m, *recv, 4096, SELF=rank, SPLIT=split, CH=ch, num_warps=warps)

                dist.barrier()
                us = bench_graph(fn, None, 120, 20, 3)["median_us"]
                gbs = rows * 4096 * 2 / (us * 1e-6) / 1e9  # per link (per peer)
                res[f"r{rows}_s{split}_w{warps}"] = (round(us, 2), round(gbs, 1))
    # reference: large peer copy per link through the copy engine / blit
    big = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    peer = (rank + 1) % 4
    dst = torch.empty(64 << 20, dtype=torch.uint8, device=torch.device(f"cuda:{peer}"))
    dist.barrier()
    for _ in range(3):
        dst.copy_(big)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        dst.copy_(big)
    e.record()
    torch.cuda.synchronize()
    res["peer_copy_64MiB_gbs"] = round(10 * (64 << 20) / (s.elapsed_time(e) * 1e-3) / 1e9, 1)
    print(f"XGMI rank={rank} " + json.dumps(res), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_rank(int(sys.argv[1]))
    else:
        procs = [subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), str(r)]) for r in (1, 2, 3)]
        try:
            main_rank(0)
        finally:
            for p in procs:
                p.wait(timeout=600)
