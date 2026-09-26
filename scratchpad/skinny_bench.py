"""Microbench `skinny_gemm.skinny_linear` against TunableOp-selected hipBLASLt at the decode
step's real per-rank TP=4 shapes. Weights rotate over >= 1 GB of copies so every call reads
HBM (MI300A's 256 MB Infinity Cache would otherwise hold a small weight). Timing is per call,
inside a captured graph of `REPS` back-to-back calls, which is what the decode replay pays.

    python3 scratchpad/skinny_bench.py [--sweep] [--m 1,16,48] [--shapes q,o,...]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skinny_gemm as sg  # noqa: E402

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
NARROW = os.environ.get("NARROW", "1") == "1"


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
    """us per call: capture REPS calls cycling over fn_list, replay, time."""
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


def candidates(m: int, n: int, k: int):
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    seen = set()
    grid = ((16, 32), (128, 256), (1, 2, 4, 8, 16), (4,)) if NARROW else ((16, 32, 64), (64, 128, 256, 512), (1, 2, 4, 8, 16), (2, 4, 8))
    for bn, bk, split, nw in itertools.product(*grid):
        if k % (split * bk) or k // split < bk:
            continue
        tiles = -(-n // bn)
        if tiles * split < 64 or tiles * split > 8192:
            continue
        if bn * bk * 2 > 16 * 1024 or bm * bk * 2 > 32 * 1024:
            continue
        if (bn, bk, split, nw) in seen:
            continue
        seen.add((bn, bk, split, nw))
        yield (bm, bn, bk, split, nw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,16,48")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="skinny_bench.json")
    a = ap.parse_args()
    torch.cuda.set_device(0)
    load_tunable()
    res = {}
    for name in a.shapes.split(","):
        n, k, calls = SHAPES[name]
        copies = max(1, min(64, (1 << 30) // (2 * n * k)))
        ws = [torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02 for _ in range(copies)]
        for m in (int(v) for v in a.m.split(",")):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            ref = F.linear(x.float(), ws[0].float())
            outs = [None]
            base = graph_time([lambda w=w: F.linear(x, w) for w in ws])
            cfgs = list(candidates(m, n, k)) if a.sweep else [sg.config(m, n, k)]
            best = None
            for cfg in cfgs:
                try:
                    y = sg.skinny_linear(x, ws[0], cfg=cfg)
                    torch.cuda.synchronize()
                    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                    if err > 2e-2:
                        print(f"  BAD {name} m={m} cfg={cfg} err={err:.3e}", flush=True)
                        continue
                    t = graph_time([lambda w=w, c=cfg: sg.skinny_linear(x, w, cfg=c) for w in ws])
                except Exception as e:  # noqa: BLE001 - sweep keeps going past a bad config
                    print(f"  FAIL {name} m={m} cfg={cfg}: {str(e)[:120]}", flush=True)
                    continue
                if best is None or t < best[0]:
                    best = (t, cfg, err)
            del outs
            gb = 2 * n * k / 1e9
            t, cfg, err = best
            line = {
                "blas_us": round(base, 2),
                "skinny_us": round(t, 2),
                "cfg": cfg,
                "err": err,
                "blas_TBs": round(gb / base * 1e3, 2),
                "skinny_TBs": round(gb / t * 1e3, 2),
                "step_saving_us": round((base - t) * calls, 1),
            }
            res[f"{name}.m{m}"] = line
            print(name, m, json.dumps(line), flush=True)
        del ws
        torch.cuda.empty_cache()
    for m in a.m.split(","):
        tot = sum(v["step_saving_us"] for kk, v in res.items() if kk.endswith(f".m{m}"))
        print(f"m={m} predicted per-step saving {tot / 1e3:.3f} ms", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
