"""Correctness of `allreduce_custom.CustomAllReduce` against an exact fp32 reference sum.

Real GPU only (IPC handles, HIP graph capture): skipped without CUDA. Both the test and the
worker live in this file, following `test_graph_capture_tp.py`'s layout, so a worker carries
no pytest and no pickled closures. Rank 0 spawns ranks 1.. as subprocesses and drives the
check itself; each rank's own process is the watchdog boundary the harness's outer `timeout`
wraps (see the module docstring in `allreduce_custom.py` for the device-side spin bound,
which bounds a single kernel, not the whole run).

Two things are checked, matching the campaign's ask:

1. **Eager, 1000 iterations, random data.** Every iteration compares the custom kernel's
   output against `sum(rank_inputs)` computed independently in fp32 and rounded to bf16 once
   -- exactly the numerics `tp.TP.all_reduce`'s existing fp32-upcast RCCL path promises (see
   its docstring), so this also stands in for "matches RCCL in fp32-accumulate terms" without
   needing RCCL and the custom path live in the same process at once.
2. **Captured and replayed.** The same all-reduce, called `CALLS_PER_GRAPH` times back to
   back inside one `torch.cuda.graph`, replayed `REPLAYS` times with fresh random data filled
   into the static input buffer before each replay (so a replay exercises the same "stale
   buffer" and "epoch tracks call position, not wall time" concerns a real decode step would).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

WORLD = 4
ITERS_EAGER = 1000
CALLS_PER_GRAPH = 8
REPLAYS = 100
N_ELEMS = 48 * 4096  # the production shape, 384 KiB of bf16
PORT = 29621


def run_worker(rank: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(PORT))
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    import allreduce_custom

    car = allreduce_custom.CustomAllReduce(rank, WORLD, device)
    ok = True

    # -- 1: eager, 1000 iterations of random data -----------------------------------
    gen = torch.Generator(device=device).manual_seed(1000 + rank)
    for i in range(ITERS_EAGER):
        x = torch.randn(N_ELEMS, generator=gen, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )
        want_parts = [None] * WORLD
        dist.all_gather_object(want_parts, x.float().cpu())
        want = sum(want_parts).to(torch.bfloat16).float().to(device)
        got = x.clone()
        car(got)
        err = (got.float() - want).abs().max().item()
        if err > 0.05:
            ok = False
            print(f"rank {rank} eager iter {i}: max err {err}", flush=True)
        # This barrier is already a sync point, so checking the watchdog flag here is free
        # (see CustomAllReduce.check_errors): a spin-limit trip turns into a loud, attributed
        # failure instead of silently passing the err<=0.05 check above by luck.
        try:
            car.check_errors()
        except RuntimeError as exc:
            ok = False
            print(f"rank {rank} eager iter {i}: {exc}", flush=True)
        dist.barrier()

    # -- 2: captured and replayed ----------------------------------------------------
    static_in = torch.zeros(N_ELEMS, device=device, dtype=torch.bfloat16)

    def step() -> None:
        for _ in range(CALLS_PER_GRAPH):
            car(static_in)

    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    # A cold pass over the same code path before capture, per graph_decode.py's own practice.
    static_in.zero_()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize(device)

    for r in range(REPLAYS):
        gen2 = torch.Generator(device=device).manual_seed(5000 + rank + r * 97)
        fresh = torch.randn(N_ELEMS, generator=gen2, device=device, dtype=torch.float32).to(
            torch.bfloat16
        )
        static_in.copy_(fresh)
        want_parts = [None] * WORLD
        dist.all_gather_object(want_parts, fresh.float().cpu())
        # CALLS_PER_GRAPH successive all-reduces of the *same* buffer: after call 1 every
        # rank's buffer already holds the world sum, so calls 2.. reduce world copies of that.
        want = sum(want_parts).to(torch.bfloat16).float().to(device)
        for _ in range(1, CALLS_PER_GRAPH):
            want = (want * WORLD).to(torch.bfloat16).float()
        graph.replay()
        torch.cuda.synchronize(device)
        err = (static_in.float() - want).abs().max().item()
        scale = want.abs().max().clamp(min=1.0).item()
        if err / scale > 0.05:
            ok = False
            print(f"rank {rank} replay {r}: max rel err {err / scale}", flush=True)
        try:
            car.check_errors()
        except RuntimeError as exc:
            ok = False
            print(f"rank {rank} replay {r}: {exc}", flush=True)
        dist.barrier()

    print(f"RESULT rank={rank} ok={ok}", flush=True)
    sys.stdout.flush()
    dist.barrier()
    os._exit(0 if ok else 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs real GPUs for IPC + HIP graphs")
def test_custom_allreduce_eager_and_graph() -> None:
    workers = [
        subprocess.Popen([sys.executable, "-u", __file__, "worker", str(r)])
        for r in range(1, WORLD)
    ]
    try:
        rc0 = subprocess.run(
            [sys.executable, "-u", __file__, "worker", "0"], timeout=580
        ).returncode
    finally:
        for w in workers:
            try:
                w.wait(timeout=60)
            except subprocess.TimeoutExpired:
                w.kill()
    assert rc0 == 0, "rank 0 reported a mismatch or crashed; see its stdout above"
    assert all(w.returncode == 0 for w in workers), "a worker rank reported a mismatch"


if __name__ == "__main__":
    assert sys.argv[1] == "worker"
    run_worker(int(sys.argv[2]))
