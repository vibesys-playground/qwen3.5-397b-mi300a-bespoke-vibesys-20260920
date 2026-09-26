"""DeltaNet prefill glue + chunked kernels per layer: torch glue vs `deltanet_prefill_glue`.

One MI300A, per-rank TP=4 shapes, the `[rows, width]` graph layout. us per layer, eager
(launch gaps included, so it overstates the torch path's cost in a graph; see the
`kernel_us` column from torch.profiler for device time).

    python3 bench_dn_glue.py --shapes 1x16,1x208,1x464,2x232
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "seed_tests"))
os.environ["SEED_DN_PREFILL_GLUE_FUSED"] = "1"

import deltanet_prefill_chunked  # noqa: E402
import deltanet_prefill_glue  # noqa: E402
import torch  # noqa: E402
from test_deltanet_prefill_glue import KC, KD, KH, VD, VH, torch_glue  # noqa: E402


def device_us(fn, reps: int = 20) -> float:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sum(e.time_range.elapsed_us() for e in ev) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1x16,1x208,1x464,2x232")
    a = ap.parse_args()
    dev = torch.device("cuda")
    c = 2 * KH * KD + VH * VD
    p = c + VH * VD + 2 * VH
    for spec in a.shapes.split(","):
        r, t = (int(v) for v in spec.split("x"))
        tc = -(-t // 32) * 32
        proj = torch.randn(r, tc, p, device=dev, dtype=torch.bfloat16)
        conv_w = torch.randn(c, 1, KC, device=dev, dtype=torch.bfloat16) * 0.5
        pool = torch.randn(64, c, KC - 1, device=dev, dtype=torch.bfloat16)
        lanes = torch.arange(r, device=dev)
        length = torch.full((r,), t, device=dev)
        active = torch.ones(r, dtype=torch.bool, device=dev)
        real = torch.arange(tc, device=dev)[None, :] < length[:, None]
        a_log, dt_bias = torch.randn(VH, device=dev), torch.randn(VH, device=dev)
        rec = torch.zeros(r, VH, KD, VD, device=dev)

        def old():
            q, k, v, beta, g = torch_glue(proj, conv_w, pool, lanes, length, active, real, a_log, dt_bias)
            rep = VH // KH
            q = q.view(r * tc, KH, KD).repeat_interleave(rep, 1)
            k = k.view(r * tc, KH, KD).repeat_interleave(rep, 1)
            deltanet_prefill_chunked.chunked_prefill(
                q, k, v.view(r * tc, VH, VD).contiguous(), g.float(), beta.contiguous(),
                rec.clone(), None, uniform=tc,
            )

        def new():
            q, k, v, beta, g = deltanet_prefill_glue.prefill_glue(
                proj, conv_w, pool, lanes, length, active, real, a_log, dt_bias, KH * KD, VH
            )
            deltanet_prefill_chunked.chunked_prefill(
                q.view(r * tc, KH, KD), k.view(r * tc, KH, KD), v.view(r * tc, VH, VD), g, beta,
                rec.clone(), None, uniform=tc,
            )

        row = {"shape": f"{r}x{tc}", "old_us": round(device_us(old), 1), "new_us": round(device_us(new), 1)}
        row["saved_ms_45_layers"] = round((row["old_us"] - row["new_us"]) * 45 / 1e3, 2)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
