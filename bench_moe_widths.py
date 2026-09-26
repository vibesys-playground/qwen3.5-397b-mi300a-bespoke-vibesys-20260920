"""MoE cost per token across prefill/mixed widths: every routed-MoE kernel, one TP=4 rank.

One MI300A, one rank's real experts of one layer (128 local of 512, MXFP4), routing from that
layer's real router on random unit-RMS hidden states. For each width T it times each kernel
end to end (routing prep and combine included, eager) and reports us/layer, us/token and the
max relative error against the fp32 oracle of `bench_prefill_moe.py`.

    SEED_MOE_HIP=1 timeout 900 python bench_moe_widths.py --out moe_widths.json

`--impls` picks among: hip (`moe_hip.fused_moe_hip`; its work split follows
`SEED_MOE_HIP_FLAT`, default on under `SEED_MOE_HIP_WIDE`, so run once per setting), grouped
(`prefill_moe`, SEED_PREFILL_GROUPED_MOE), legacy (`mxfp4_gemv.fused_moe`).
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

import torch  # noqa: E402
from bench_prefill_moe import (  # noqa: E402
    HIDDEN,
    TOP_K,
    dequant_all,
    load_layer,
    oracle,
    rel_err,
    routing_for,
    time_us,
)

import mxfp4_gemv  # noqa: E402
import prefill_moe  # noqa: E402


def impl_fns(ex: dict, span: tuple[int, int]) -> dict:
    import moe_hip  # noqa: PLC0415

    fns = {
        "hip": lambda x, r: moe_hip.fused_moe_hip(x, ex, r, TOP_K, span),
        "grouped": lambda x, r: prefill_moe.prefill_moe(x, ex, r, TOP_K, span),
        "legacy": lambda x, r: mxfp4_gemv.fused_moe(
            x, ex, r, TOP_K, span, total_experts=512
        ),
    }
    return fns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--widths", default="64,128,256,512,1024,2048")
    ap.add_argument("--impls", default="hip,grouped")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    ex, router = load_layer(a.layer, a.rank, dev, a.random_weights)
    local = ex["gate_up"].shape[0]
    span = (a.rank * local, (a.rank + 1) * local)
    dense = None if a.no_oracle else dequant_all(ex)
    fns = impl_fns(ex, span)
    rows = []
    for t in (int(w) for w in a.widths.split(",")):
        x = torch.randn(t, HIDDEN, device=dev, dtype=torch.bfloat16)
        x = (x / x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)).contiguous()
        r = routing_for(x, router)
        live = ((r[0] >= span[0]) & (r[0] < span[1])).sum().item()
        distinct = torch.unique(r[0][(r[0] >= span[0]) & (r[0] < span[1])]).numel()
        want = None if dense is None else oracle(x, dense, r, span)
        for name in a.impls.split(","):
            if name not in fns:
                continue
            row = {"T": t, "impl": name, "local_assign": live, "distinct": distinct}
            if name == "hip":
                import moe_hip  # noqa: PLC0415

                row["flat"] = moe_hip.KNOBS["MOE_FLAT"]
            try:
                got = fns[name](x, r)
                if want is not None:
                    row["rel_err"] = rel_err(got, want)
                us = time_us(lambda n=name: fns[n](x, r), a.reps)
                row["us_layer"] = round(us, 1)
                row["us_token"] = round(us / t, 3)
                row["ms_60_layers"] = round(us * 60 / 1e3, 2)
            except Exception as exc:  # noqa: BLE001 - a failing impl is a result
                row["error"] = repr(exc)[:300]
            print(json.dumps(row), flush=True)
            rows.append(row)
        del want
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
