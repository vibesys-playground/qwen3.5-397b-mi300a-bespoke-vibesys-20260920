"""Dense bf16 projections at prefill/mixed widths on one MI300A: achieved TFLOP/s.

Per-rank TP=4 weight shapes (bf16, [N, K]) at M rows, `F.linear` timed three ways:
the default heuristic, TunableOp with `blas_tune`'s 10 ms per-candidate budget, and
TunableOp with a 100 ms budget (fresh in-memory search each, no cache file).

    timeout 900 python3 bench_dense_widths.py --widths 256,512,1024
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

SHAPES = {  # name: (N, K) per rank at TP=4
    "dn_in_proj_all": (5152, 4096),
    "dn_out_proj": (4096, 2048),
    "attn_q_proj": (4096, 4096),
    "attn_kv_proj": (256, 4096),
    "attn_o_proj": (4096, 2048),
    "router_gate": (513, 4096),
    "shared_gate_up": (256, 4096),
    "shared_down": (4096, 128),
}


def time_us(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="256,512,1024")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    tun = torch.cuda.tunable
    tun.set_filename("/tmp/bench_dense_widths_tunableop_%d.csv", False)  # never the served cache
    rows = []
    for m in (int(w) for w in a.widths.split(",")):
        for name, (n, k) in SHAPES.items():
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
            flop = 2 * m * n * k
            row = {"M": m, "gemm": name, "N": n, "K": k}
            tun.enable(False)
            row["default_us"] = round(time_us(lambda: F.linear(x, w)), 1)
            for budget in (10, 100):
                tun.enable(True)
                tun.set_max_tuning_duration(budget)
                tun.tuning_enable(True)
                # a distinct stride per budget forces a fresh search for the same M, N, K
                wb = torch.empty(n, k + budget, device=dev, dtype=torch.bfloat16)[:, :k]
                wb.copy_(w)
                F.linear(x, wb)
                torch.cuda.synchronize()
                tun.tuning_enable(False)
                row[f"tuned{budget}_us"] = round(time_us(lambda wb=wb: F.linear(x, wb)), 1)
                tun.enable(False)
            best = min(row["default_us"], row["tuned10_us"], row["tuned100_us"])
            row["best_tflops"] = round(flop / best / 1e6, 1)
            row["floor_us"] = round(max(flop / 980e6, (n * k * 2 + m * (n + k) * 2) / 5.3e6), 1)
            print(json.dumps(row), flush=True)
            rows.append(row)
    tot = {}
    for r in rows:
        t = tot.setdefault(r["M"], [0.0, 0.0, 0.0])
        t[0] += r["default_us"]
        t[1] += min(r["tuned10_us"], r["tuned100_us"])
        t[2] += r["floor_us"]
    print("SUM us per shape set (default, best tuned, floor):", json.dumps(tot))


if __name__ == "__main__":
    main()
