"""GPU check for SEED_AR_RMSNORM_FUSED: Triton AR+add+RMSNorm vs oneshot AR + add_rmsnorm.

Run from the bundle root on 4 GPUs: SEED_AR_GRAPH_SAFE=1 python3 -u ar_norm_check.py
(1) eager bit-exactness at rows 1/16/48; (2) graphs mixing fused and plain calls, replayed
interleaved with eager calls, every output compared bit-for-bit; (3) us per call in a
120-call graph: plain AR + add_rmsnorm vs fused."""

import os, subprocess, sys, time

sys.path.insert(0, os.getcwd())
import torch
import torch.distributed as dist
import allreduce_custom as ac
import rmsnorm_fused as rf

H = 4096
WORLD = 4


def inputs(seed, n, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return [torch.randn(n, generator=g, device=dev).to(torch.bfloat16) for _ in range(WORLD)]


def run(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29591"
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=WORLD, device_id=dev)
    assert ac.GRAPH_SAFE
    car = ac.CustomAllReduce(rank, WORLD, dev)
    w = (
        (torch.randn(H, generator=torch.Generator().manual_seed(7)) * 0.1)
        .to(dev)
        .to(torch.bfloat16)
    )
    eps = 1e-6
    bad = []

    def ref(parts_k, res):
        x = parts_k[rank].clone()
        car(x)
        return rf.add_rmsnorm(res, x.view(res.shape), w, eps)

    for rows in (1, 16, 48):
        for it in range(20):
            p = inputs(100 * rows + it, rows * H, dev)
            res = inputs(7 + it, rows * H, dev)[0].view(rows, 1, H)
            m = p[rank].view(rows, 1, H).clone()
            xo, o = car.ar_add_rmsnorm(m, res, w, eps)
            xr, orf = ref(p, res)
            if not (torch.equal(xo, xr) and torch.equal(o, orf)):
                bad.append(
                    f"eager rows{rows} it{it} dres={(xo.float() - xr.float()).abs().max().item()} dnorm={(o.float() - orf.float()).abs().max().item()}"
                )
    torch.cuda.synchronize()
    if rank == 0:
        print(f"eager done bad={len(bad)} {bad[:3]}", flush=True)

    def make(rows, calls):
        n = rows * H
        ms = [torch.zeros(rows, 1, H, dtype=torch.bfloat16, device=dev) for _ in range(calls)]
        res = torch.zeros(rows, 1, H, dtype=torch.bfloat16, device=dev)
        outs = []

        def body():
            outs.clear()
            for k in range(calls):
                if k % 3 == 2:
                    outs.append(car(ms[k].clone()))
                else:
                    outs.append(car.ar_add_rmsnorm(ms[k], res, w, eps))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        return g, ms, res, outs, rows, calls

    graphs = [make(48, 120), make(1, 121), make(16, 7)]
    torch.cuda.synchronize()
    dist.barrier()
    for it, gi in enumerate([0, 1, 1, 2, 0, 2, 2, 1, 0, 0, 1, 2]):
        g, ms, res, outs, rows, calls = graphs[gi]
        allp = []
        for k in range(calls):
            p = inputs(1000 * it + k, rows * H, dev)
            allp.append(p)
            ms[k].copy_(p[rank].view(rows, 1, H))
        res.copy_(inputs(it + 99, rows * H, dev)[0].view(rows, 1, H))
        g.replay()
        e = inputs(it + 5, 3 * H, dev)
        ex = e[rank].clone()
        car(ex)
        torch.cuda.synchronize()
        want = torch.stack([q.float() for q in e]).sum(0).to(torch.bfloat16)
        if not torch.equal(ex, want):
            bad.append(f"it{it} eager-between")
        got = [tuple(t.clone() for t in o) if isinstance(o, tuple) else o.clone() for o in outs]
        for k in range(calls):
            if k % 3 == 2:
                want = torch.stack([q.float() for q in allp[k]]).sum(0).to(torch.bfloat16)
                if not torch.equal(got[k].view(-1), want):
                    bad.append(f"it{it} g{gi} call{k} ar")
            else:
                xr, orf = ref(allp[k], res)
                xo, o = got[k]
                if not (torch.equal(xo, xr) and torch.equal(o, orf)):
                    bad.append(
                        f"it{it} g{gi} call{k} fused dres={(xo.float() - xr.float()).abs().max().item()} dnorm={(o.float() - orf.float()).abs().max().item()}"
                    )
        torch.cuda.synchronize()
    nb = torch.tensor([len(bad)], device=dev)
    dist.all_reduce(nb)
    if rank == 0:
        print(f"CHECK graphs failures_all_ranks={int(nb.item())} first={bad[:3]}", flush=True)

    for rows in (16, 48):
        for mode in ("plain", "fused"):
            ms = [torch.randn(rows, 1, H, device=dev).to(torch.bfloat16) for _ in range(120)]
            res = torch.randn(rows, 1, H, device=dev).to(torch.bfloat16)

            def body():
                x = res
                for k in range(120):
                    if mode == "plain":
                        x, _ = rf.add_rmsnorm(x, car(ms[k].clone()), w, eps)
                    else:
                        x, _ = car.ar_add_rmsnorm(ms[k], x, w, eps)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                body()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / 20 / 120 * 1e6
            if rank == 0:
                print(f"TIME rows{rows} {mode} {us:.2f} us/call", flush=True)
    car.check_errors()
    dist.barrier()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(int(sys.argv[1]))
    else:
        ps = [subprocess.Popen([sys.executable, "-u", __file__, str(r)]) for r in range(1, WORLD)]
        run(0)
        for p in ps:
            p.wait(timeout=300)
