"""Microbench `skinny_hip` against TunableOp-selected hipBLASLt at the decode step's real
per-rank TP=4 shapes, and check its numerics. Weights rotate over >= 1 GB of copies so every
call reads HBM (the 256 MB Infinity Cache would otherwise hold a small weight). Timing is per
call inside a captured graph of REPS back-to-back calls, which is what the decode replay pays.

    SEED_BLAS_CACHE_DIR=... SEED_HIP_CACHE_DIR=... python3 scratchpad/skinny_hip_bench.py \
        [--m 1,16,32,48,64] [--shapes q,o,...] [--out result.json]

Prints, per (shape, M): hipBLASLt µs, best HIP µs + (config, tpp), achieved TB/s, and the max
abs diff vs `F.linear` relative to the output's max magnitude. `--out` writes the best config
per (shape, M) so `skinny_hip.ROUTES` can be filled from it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skinny_hip as sh  # noqa: E402

SHAPES = {  # name: (N, K, calls per step)
    "q": (4096, 4096, 15),
    "kv": (256, 4096, 30),
    "o": (4096, 2048, 15),
    "in_all": (5152, 4096, 45),
    "out": (4096, 2048, 45),
    "router": (513, 4096, 60),
    "sh_gu": (512, 4096, 60),
    "sh_d": (4096, 256, 60),
    "lm_head": (248320, 4096, 1),
}
REPS = 20


def load_tunable() -> None:
    d = os.environ.get("SEED_BLAS_CACHE_DIR")
    files = sorted(glob.glob(f"{d}/tunableop_gfx942_0.csv")) if d else []
    if files:
        t = torch.cuda.tunable
        t.set_filename(f"{d}/tunableop_gfx942_%d.csv", True)
        t.enable(True)
        t.tuning_enable(False)
        t.read_file()
        print("tunable cache", files[0], flush=True)


def graph_time(fn_list) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list[:3]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(REPS):
            fn_list[i % len(fn_list)]()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    it = 10
    for _ in range(it):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / (it * REPS)


def tpp_candidates(n: int, split: int) -> list[int]:
    tiles = (n + 15) // 16
    out = set()
    for target in (228, 456, 684, 912, 1824):
        gx = max(1, target // split)
        out.add(max(1, -(-tiles // gx)))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,16,32,48,64")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    load_tunable()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    results = []
    for name in args.shapes.split(","):
        n, k, calls = SHAPES[name]
        copies = max(1, min(64, (1 << 30) // (2 * n * k)))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for m in (int(v) for v in args.m.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            outs = [torch.empty(m, n, device=dev, dtype=torch.bfloat16) for _ in range(2)]
            t_blas = graph_time([lambda i=i: F.linear(x, ws[i]) for i in range(copies)])
            ref = F.linear(x, ws[0]).float()
            best = None
            for cfg in range(len(sh.CONFIGS)):
                split = sh.split_of(cfg, k)
                if sh.CONFIGS[cfg][0] != sh.mt_of(m) or split == 0:
                    continue
                for tpp in tpp_candidates(n, split):
                    if not sh.valid(cfg, m, n, k, tpp):
                        continue
                    got = sh.gemm(x, ws[0], cfg, tpp).float()
                    got2 = sh.gemm(x, ws[0], cfg, tpp).float()
                    err = ((got - ref).abs().max() / ref.abs().max()).item()
                    det = bool(torch.equal(got, got2))
                    if err > 2e-2 or not det:
                        print(f"  BAD {name} m={m} cfg={cfg} tpp={tpp} err={err:.3g} det={det}", flush=True)
                        continue
                    t = graph_time([lambda i=i: sh.gemm(x, ws[i], cfg, tpp, outs[i % 2]) for i in range(copies)])
                    if best is None or t < best[0]:
                        best = (t, cfg, tpp, err)
            tb = 2 * n * k / 1e6
            row = {"shape": name, "n": n, "k": k, "m": m, "calls": calls, "blas_us": t_blas}
            if best:
                row.update(hip_us=best[0], cfg=best[1], tpp=best[2], err=best[3])
                print(
                    f"{name:8s} m={m:2d} blas {t_blas:7.2f} us ({tb / t_blas:4.2f} TB/s)  "
                    f"hip {best[0]:7.2f} us ({tb / best[0]:4.2f} TB/s) cfg={best[1]} tpp={best[2]} "
                    f"err={best[3]:.2e}  step_saving={calls * (t_blas - best[0]):7.1f} us",
                    flush=True,
                )
            else:
                print(f"{name:8s} m={m:2d} blas {t_blas:7.2f} us  hip: no valid config", flush=True)
            results.append(row)
        del ws
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
