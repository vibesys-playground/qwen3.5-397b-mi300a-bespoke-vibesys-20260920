"""Microbenchmark: custom one-shot all-reduce vs RCCL, eager and HIP-graph-replayed.

Sizes match the campaign's ask: 48 KiB (batch 6), 384 KiB (batch 48, production), 1.5 MiB
(batch 192), all bf16 at hidden=4096. Topology matches `tp.py`/`rccl-bench/allreduce_bench.py`:
backend="nccl", TCP rendezvous, one process per rank, rank r on cuda:r.

    srun --jobid=<id> --overlap --environment=<toml> python3 -u bench_custom_allreduce.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

import torch
import torch.distributed as dist

HIDDEN = 4096
SIZES = {"48KiB": 6, "384KiB": 48, "1536KiB": 192}  # batch, at hidden=4096 bf16
DEFAULT_PORT = 29631


def bench_eager(fn, x, iters, warmup):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(x)
        ends[i].record()
    torch.cuda.synchronize()
    us = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(iters)]
    return {"mean_us": statistics.mean(us), "median_us": statistics.median(us), "p90_us": statistics.quantiles(us, n=100)[89]}


def bench_graph(fn, x, calls, replays, warmup):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # capture_error_mode="thread_local": required so the NCCL process-group watchdog thread's
    # background polling doesn't itself trip "operation not permitted when stream is
    # capturing" while an RCCL all_reduce is being captured (see allreduce_graph_bench.py).
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(calls):
            fn(x)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    us = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        us.append(start.elapsed_time(end) * 1e3 / calls)
    us.sort()
    return {"mean_us": statistics.mean(us), "median_us": statistics.median(us), "p90_us": us[int(0.9 * len(us))]}


def run_rank(a: argparse.Namespace) -> dict | None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    device = torch.device(f"cuda:{a.rank}")
    torch.cuda.set_device(a.rank)
    dist.init_process_group(backend="nccl", rank=a.rank, world_size=a.tp, device_id=device)

    import allreduce_custom

    car = allreduce_custom.CustomAllReduce(a.rank, a.tp, device)
    two_shot = allreduce_custom.TwoShotAllReduce(a.rank, a.tp, device) if a.two_shot else None

    def rccl_reduce(x: torch.Tensor) -> torch.Tensor:
        wide = x.float()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
        return x.copy_(wide)

    results: dict[str, dict] = {}
    sizes = {f"{r}rows": int(r) for r in a.rows.split(",")} if a.rows else SIZES
    for tag, batch in sizes.items():
        n = batch * HIDDEN
        x = torch.randn(n, device=device, dtype=torch.bfloat16)
        if n > car.max_elems:  # over the one-shot cap: only RCCL (what TP.all_reduce does)
            results[f"rccl_eager_{tag}"] = bench_eager(rccl_reduce, x, a.iters, a.warmup)
            results[f"rccl_graph_{tag}"] = bench_graph(rccl_reduce, x, a.graph_calls, a.replays, 5)
            continue
        results[f"custom_eager_{tag}"] = bench_eager(car, x, a.iters, a.warmup)
        results[f"rccl_eager_{tag}"] = bench_eager(rccl_reduce, x, a.iters, a.warmup)
        results[f"custom_graph_{tag}"] = bench_graph(car, x, a.graph_calls, a.replays, 5)
        results[f"rccl_graph_{tag}"] = bench_graph(rccl_reduce, x, a.graph_calls, a.replays, 5)
        if two_shot is not None and (tag == "384KiB" or a.rows):
            results[f"twoshot_eager_{tag}"] = bench_eager(two_shot, x, a.iters, a.warmup)
            results[f"twoshot_graph_{tag}"] = bench_graph(two_shot, x, a.graph_calls, a.replays, 5)

    if a.rank == 0:
        print("RESULT " + json.dumps(results), flush=True)
    sys.stdout.flush()
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--graph-calls", type=int, default=60)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--two-shot", action="store_true")
    p.add_argument(
        "--rows",
        default="",
        help="comma-separated row counts at hidden 4096 to bench instead of the default sizes",
    )
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    workers = []
    for r in range(1, a.tp):
        argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
        argv = [tok for tok in argv if not tok.startswith("--rank")]
        argv += ["--rank", str(r)]
        workers.append(subprocess.Popen(argv, env=os.environ.copy()))
    run_rank(a)
    for w in workers:
        w.wait(timeout=180)


if __name__ == "__main__":
    main()
