"""GPU check + microbench for `SEED_AR_GRAPH_SAFE` (allreduce_custom's device-counter one-shot).

Run from the bundle root on a 4-GPU node (spawns one process per rank):
    timeout 600 python3 -u scratchpad/ar_graph_check.py --tp 4

Per rank: (1) eager correctness at several sizes, (2) two captured graphs with an even and an
odd number of calls, replayed interleaved with eager calls, every result checked against an
fp32 reference computed locally from every rank's seeded inputs, (3) µs per call inside a
120-call graph for rccl / host-counter custom (`graph_safe=0`) / device-counter (`=1`), plus
the fused residual+RMSNorm variant. Prints CHECK/TIME lines from rank 0.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import torch
import torch.distributed as dist

import allreduce_custom as ac

H = 4096


def inputs(seed, world, n, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(n, generator=g, device=device).to(torch.bfloat16) for _ in range(world)]


def ref_sum(parts):
    return torch.stack([p.float() for p in parts]).sum(0).to(torch.bfloat16)


def ref_norm(parts, res, w, eps):
    x = (res.float() + ref_sum(parts).float()).to(torch.bfloat16)
    xf = x.float().view(-1, H)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    return x, y.to(torch.bfloat16).view(-1)


def run_rank(a, rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(a.port)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=a.tp, device_id=dev)
    car = ac.CustomAllReduce(rank, a.tp, dev)
    w = (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.1).to(dev)
    eps = 1e-6
    bad = []

    def say(msg):
        if rank == 0:
            print(msg, flush=True)

    for gs in ((1, 0) if a.check else ()):
        ac.GRAPH_SAFE = bool(gs)
        # (1) eager
        for n in (1000, H, 16 * H, 48 * H, 256 * H):
            parts = inputs(n, a.tp, n, dev)
            x = parts[rank].clone()
            car(x)
            if not torch.equal(x, ref_sum(parts)):
                bad.append(f"gs{gs} eager n={n} maxdiff={(x.float()-ref_sum(parts).float()).abs().max().item()}")
        torch.cuda.synchronize()
        say(f"eager graph_safe={gs} done, failures so far {len(bad)}")

        # (2) graphs: G_even 120 calls @48 rows (alternating plain/fused), G_odd 121 calls @1 row
        def make(rows, calls):
            n = rows * H
            xs = [torch.zeros(n, dtype=torch.bfloat16, device=dev) for _ in range(calls)]
            res = torch.zeros(n, dtype=torch.bfloat16, device=dev)
            outs = []

            def body():
                outs.clear()
                for k in range(calls):
                    if k % 2:
                        outs.append(car.residual_norm(xs[k].view(rows, H), res.view(rows, H), w, eps))
                    else:
                        outs.append(car(xs[k]))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            return g, xs, res, outs, rows, calls

        graphs = [make(48, 120), make(1, 121), make(16, 7)]
        torch.cuda.synchronize()
        dist.barrier()
        order = [0, 1, 1, 2, 0, 2, 2, 1, 0, 0, 1, 2]
        for it, gi in enumerate(order):
            g, xs, res, outs, rows, calls = graphs[gi]
            n = rows * H
            seeds = [1000 * it + k for k in range(calls)]
            allp = []
            for k in range(calls):
                p = inputs(seeds[k], a.tp, n, dev)
                allp.append(p)
                xs[k].copy_(p[rank])
            rp = inputs(seeds[0] + 99, 1, n, dev)[0]
            res.copy_(rp)
            g.replay()
            # an eager call between replays (different size, own parity)
            e = inputs(it + 5, a.tp, 3 * H, dev)
            ex = e[rank].clone()
            car(ex)
            torch.cuda.synchronize()
            if not torch.equal(ex, ref_sum(e)):
                bad.append(f"gs{gs} it{it} eager-between")
            for k in range(calls):
                if k % 2:
                    xr, yr = ref_norm(allp[k], rp, w, eps)
                    nr, nm = outs[k]
                    d1 = (nr.float().view(-1) - xr.float()).abs().max().item()
                    d2 = (nm.float().view(-1) - yr.float()).abs().max().item()
                    if d1 > 0 or d2 > 0.07:
                        bad.append(f"gs{gs} it{it} g{gi} call{k} norm d_res={d1} d_norm={d2}")
                else:
                    if not torch.equal(xs[k], ref_sum(allp[k])):
                        d = (xs[k].float() - ref_sum(allp[k]).float()).abs().max().item()
                        bad.append(f"gs{gs} it{it} g{gi} call{k} ar maxdiff={d}")
        car.check_errors()
        nb = torch.tensor([len(bad)], device=dev)
        dist.all_reduce(nb)
        say(f"CHECK graph_safe={gs} failures_all_ranks={int(nb.item())} first={bad[:3]}")
        bad.clear()
        del graphs
        torch.cuda.synchronize()

    # (3) timing inside a 120-call graph
    def time_graph(fn, rows):
        n = rows * H
        x = torch.randn(n, device=dev).to(torch.bfloat16)
        res = torch.randn(n, device=dev).to(torch.bfloat16)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(x, res, rows)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            for _ in range(120):
                fn(x, res, rows)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(a.replays):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.replays / 120 * 1e6

    def rccl(x, res, rows):
        wide = x.float()
        dist.all_reduce(wide)
        x.copy_(wide)

    def plain(x, res, rows):
        car(x)

    def fused(x, res, rows):
        car.residual_norm(x.view(rows, H), res.view(rows, H), w, eps)

    def unfused_norm(x, res, rows):
        car(x)
        xx = (res + x).view(rows, H).float()
        (xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + eps) * (1 + w)).to(torch.bfloat16)

    for rows in (1, 16, 48):
        line = []
        ac.GRAPH_SAFE = False
        line.append(("custom_host", time_graph(plain, rows)))
        line.append(("custom_host_fused", time_graph(fused, rows)))
        ac.GRAPH_SAFE = True
        line.append(("graph_safe", time_graph(plain, rows)))
        line.append(("graph_safe_fused", time_graph(fused, rows)))
        line.append(("graph_safe_unfused_norm", time_graph(unfused_norm, rows)))
        line.append(("rccl", time_graph(rccl, rows)))
        say(f"TIME rows={rows} " + " ".join(f"{k}={v:.1f}us" for k, v in line))
    car.check_errors()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29547)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--check", type=int, default=1)
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--tp", str(a.tp), "--port", str(a.port),
           "--replays", str(a.replays), "--check", str(a.check)]
    ws = [subprocess.Popen(cmd + ["--rank", str(r)]) for r in range(1, a.tp)]
    try:
        run_rank(a, 0)
    finally:
        for wk in ws:
            wk.wait(timeout=120)


if __name__ == "__main__":
    main()
