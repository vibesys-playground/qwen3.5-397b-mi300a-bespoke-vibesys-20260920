"""`SEED_MOE_HIP_BT32` microgate: 16-token units (NB = 1 kernels) vs 32-token units (NB = 2)
on one MI300A rank, one layer, real weights and router. Checks the routed output is
bit-identical and times gate_up + down at each width.

    SEED_MOE_HIP=1 SEED_MOE_HIP_WIDE=1 python bench_moe_bt32.py --widths 512,1024,1536,2048
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SEED_MOE_HIP", "1")
os.environ.setdefault("SEED_MOE_HIP_WIDE", "1")

import torch  # noqa: E402
from bench_prefill_moe import HIDDEN, TOP_K, load_layer, routing_for, time_us  # noqa: E402

import moe_hip  # noqa: E402
import mxfp4_gemv  # noqa: E402


def runner(m, x, ex, routing, span, block_t):
    plan = mxfp4_gemv.BwPlan(
        x.shape[0], TOP_K, HIDDEN, span[1] - span[0], x.device, x.dtype, block_t=block_t,
        permute_x=False,
    )
    mxfp4_gemv.bw_prep(x, routing[0], routing[1], span, plan)
    aw = routing[1].float()
    assignments = x.shape[0] * TOP_K
    inter = torch.zeros(assignments, moe_hip.INTER, dtype=x.dtype, device=x.device)
    y = torch.zeros(assignments, HIDDEN, dtype=x.dtype, device=x.device)
    g = moe_hip.grid(x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    gu, dn = (m.gate_up2, m.down2) if block_t == 32 else (m.gate_up, m.down)

    def call():
        gu(x.data_ptr(), ex["gate_up"].data_ptr(), ex["gate_up_scale"].data_ptr(),
           plan.sorted.data_ptr(), plan.unit.data_ptr(), plan.n_units.data_ptr(),
           inter.data_ptr(), TOP_K, g, stream)
        dn(inter.data_ptr(), ex["down"].data_ptr(), ex["down_scale"].data_ptr(), aw.data_ptr(),
           plan.sorted.data_ptr(), plan.unit.data_ptr(), plan.n_units.data_ptr(), y.data_ptr(),
           g, stream)

    return call, plan, inter, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--widths", default="256,512,1024,1536,2048")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    ex, router = load_layer(a.layer, a.rank, dev, False)
    local = ex["gate_up"].shape[0]
    span = (a.rank * local, (a.rank + 1) * local)
    m = moe_hip.ext()
    rows = []
    for t in (int(w) for w in a.widths.split(",")):
        x = torch.randn(t, HIDDEN, device=dev, dtype=torch.bfloat16)
        x = (x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)).contiguous()
        routing = routing_for(x, router)
        c16, p16, i16, y16 = runner(m, x, ex, routing, span, 16)
        c32, p32, i32, y32 = runner(m, x, ex, routing, span, 32)
        c16(); c32(); torch.cuda.synchronize()
        live = (routing[0] >= span[0]) & (routing[0] < span[1])
        row = {
            "T": t,
            "units16": int(p16.n_units.item()),
            "units32": int(p32.n_units.item()),
            "inter_equal": bool(torch.equal(i16[live], i32[live])),
            "y_equal": bool(torch.equal(y16[live], y32[live])),
            "bt16_us": round(min(time_us(c16, a.reps) for _ in range(3)), 1),
            "bt32_us": round(min(time_us(c32, a.reps) for _ in range(3)), 1),
        }
        row["speedup"] = round(row["bt16_us"] / row["bt32_us"], 3)
        print(json.dumps(row), flush=True)
        rows.append(row)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
