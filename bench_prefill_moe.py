"""Prefill MoE microbenchmark: `prefill_moe` (SEED_PREFILL_GROUPED_MOE) vs the shipped path.

One MI300A, one TP=4 rank's real expert weights (128 local experts of one layer, MXFP4 from
the checkpoint), routing from that layer's real router on random unit-RMS hidden states.
For each T: us/layer of the shipped `mxfp4_gemv.fused_moe` (its `_grouped_moe` path at these
shapes) and of `prefill_moe.prefill_moe`, both end to end (routing prep included), eager,
plus each one's max error against an fp32 oracle (dense fp32 dequant, per-expert matmuls).

Also times, for PREFILL_ROOFLINE.md's ranking: a small `prefill_moe` block-shape sweep at
T=2048, the DeltaNet prefill recurrence (`deltanet_fused.fused_recurrent_prefill`, 16 local
heads; warps x block_v x prefetch sweep), the chunked DeltaNet prefill
(`deltanet_prefill_chunked`, SEED_DELTANET_PREFILL_CHUNKED) against it, and one dense bf16
projection at the DeltaNet in_proj shape.

    timeout 600 python bench_prefill_moe.py --out bench_prefill_moe.json     # < 5 min
    python bench_prefill_moe.py --random-weights                             # no checkpoint

`MODEL_PATH` picks the checkpoint (default: the cluster copy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mxfp4_gemv  # noqa: E402
import prefill_moe  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mxfp4 import dequant_mxfp4  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
HIDDEN, INTER, EXPERTS, TOP_K, WORLD = 4096, 1024, 512, 10, 4


def load_layer(
    layer: int, rank: int, dev: torch.device, random_weights: bool
) -> tuple[dict, torch.Tensor]:
    """(this rank's MXFP4 experts, the layer's bf16 router [512, 4096])."""
    local = EXPERTS // WORLD
    if random_weights:
        sys.path.insert(0, str(HERE / "seed_tests"))
        from test_mxfp4_fused_gemv import random_experts  # noqa: PLC0415

        ex = {k: v.to(dev) for k, v in random_experts(local, HIDDEN, INTER, seed=0).items()}
        router = torch.randn(EXPERTS, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
        return ex, router
    from model import PREFIX, load_experts  # noqa: PLC0415
    from tp import Shard  # noqa: PLC0415
    from weights import Checkpoint  # noqa: PLC0415

    ck = Checkpoint(MODEL_PATH)
    p = f"{PREFIX}layers.{layer}.mlp"
    ex = load_experts(ck, p, Shard(rank * local, local), dev, torch.bfloat16)
    router = ck.load(f"{p}.gate.weight", dev, torch.bfloat16)
    return ex, router


def routing_for(x: torch.Tensor, router: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`Model.moe`'s torch routing: softmax over 512, top-10, renormalize; flattened."""
    probs = F.linear(x, router).softmax(-1, dtype=torch.float)
    w, i = probs.topk(TOP_K, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return i.to(torch.int32).reshape(-1).contiguous(), w.reshape(-1).contiguous()


def oracle(x, dense, routing, span) -> torch.Tensor:
    """fp32: per local expert, gather its rows, silu(gate) * up, down, weighted index_add."""
    a_expert, a_weight = routing
    lo, hi = span
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(hi - lo):
        a = ((a_expert == lo + e) & (a_weight != 0)).nonzero().flatten()
        if a.numel() == 0:
            continue
        tok = a // TOP_K
        gu = xf[tok] @ dense["gate_up"][e].T
        y = (F.silu(gu[:, :INTER]) * gu[:, INTER:]) @ dense["down"][e].T
        out.index_add_(0, tok, y * a_weight[a, None])
    return out


def dequant_all(ex: dict, chunk: int = 16) -> dict:
    """fp32 dense copies of every local expert (6.4 GB at 128 experts), `chunk` at a time to
    bound `dequant_mxfp4`'s temporaries."""
    out = {}
    for name in ("gate_up", "down"):
        q, s = ex[name], ex[f"{name}_scale"]
        dense = torch.empty(*q.shape[:-1], q.shape[-1] * 2, dtype=torch.float32, device=q.device)
        for e in range(0, q.shape[0], chunk):
            dense[e : e + chunk] = dequant_mxfp4(q[e : e + chunk], s[e : e + chunk], torch.float32)
        out[name] = dense
    return out


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


def time_us(fn, reps: int, warm: int = 2) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def context_ops(dev: torch.device, reps: int) -> dict:
    """The two non-MoE ops PREFILL_ROOFLINE.md ranks next, at T=2048, one rank."""
    out: dict = {}
    t = 2048
    x = torch.randn(t, HIDDEN, device=dev, dtype=torch.bfloat16)
    w = torch.randn(5152, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02
    us = time_us(lambda: F.linear(x, w), reps)
    out["deltanet_in_proj_us"] = us
    out["deltanet_in_proj_tflops"] = 2 * t * HIDDEN * 5152 / us / 1e6
    try:
        import deltanet_fused  # noqa: PLC0415

        heads, dk = 16, 128
        q = torch.randn(t, heads, dk, device=dev, dtype=torch.bfloat16)
        k = torch.randn(t, heads, dk, device=dev, dtype=torch.bfloat16)
        v = torch.randn(t, heads, dk, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(t, heads, device=dev) * 0.1
        beta = torch.rand(t, heads, device=dev, dtype=torch.bfloat16)
        st = torch.zeros(1, heads, dk, dk, device=dev)
        base, _ = deltanet_fused.fused_recurrent_prefill(
            q, k, v, g, beta, st.clone(), prefetch=False, num_warps=4, block_v=32, chunked=False
        )
        # Timed calls advance their state in place: give them a scratch copy, so `st` stays
        # the zero state every correctness comparison below starts from.
        scratch = st.clone()
        us = time_us(
            lambda: deltanet_fused.fused_recurrent_prefill(
                q, k, v, g, beta, scratch, prefetch=False, num_warps=4, block_v=32, chunked=False
            ),
            reps,
        )
        out["deltanet_recurrence_us"] = us
        out["deltanet_recurrence_us_per_token"] = us / t
        # SEED_DELTANET_PREFILL_{PREFETCH,WARPS,BLOCK_V} sweep; the first row is the shipped config
        sweep = []
        for nw in (4, 2, 1):
            for bv in (32, 16):
                for pf in (False, True):

                    def fn(nw=nw, bv=bv, pf=pf):
                        return deltanet_fused.fused_recurrent_prefill(
                            q,
                            k,
                            v,
                            g,
                            beta,
                            st.clone(),
                            prefetch=pf,
                            num_warps=nw,
                            block_v=bv,
                            chunked=False,
                        )

                    item = {"warps": nw, "block_v": bv, "prefetch": pf}
                    try:
                        item["max_abs_diff"] = (fn()[0] - base).abs().max().item()
                        item["us_per_token"] = time_us(fn, reps) / t
                    except Exception as exc:  # noqa: BLE001 - a failing config is a result
                        item["error"] = repr(exc)[:300]
                    sweep.append(item)
        out["deltanet_recurrence_sweep"] = sweep
        # SEED_DELTANET_PREFILL_CHUNKED: the chunked (WY) kernels, same inputs
        import deltanet_prefill_chunked  # noqa: PLC0415

        chunked = []
        for ch, bv, ns in ((16, 16, 2), (32, 16, 2), (32, 16, 1), (32, 32, 1), (16, 32, 1)):

            def fc(ch=ch, bv=bv, ns=ns):
                return deltanet_prefill_chunked.chunked_prefill(
                    q, k, v, g, beta, st.clone(), chunk=ch, block_v=bv, num_stages=ns
                )

            item = {"chunk": ch, "block_v": bv, "num_stages": ns}
            try:
                got = fc()[0]
                item["rel_err_vs_recurrence"] = (
                    (got - base).abs().max() / base.abs().max().clamp_min(1e-30)
                ).item()
                item["us_per_token"] = time_us(fc, reps) / t
                item["ms_per_layer_t2048"] = item["us_per_token"] * t / 1e3
            except Exception as exc:  # noqa: BLE001 - a failing config is a result
                item["error"] = repr(exc)[:300]
            chunked.append(item)
        out["deltanet_chunked"] = chunked
    except Exception as exc:  # noqa: BLE001 - a context number, never fail the MoE bench
        out["deltanet_recurrence_error"] = repr(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=str(HERE / "bench_prefill_moe.json"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    t0 = time.time()
    ex, router = load_layer(args.layer, args.rank, dev, args.random_weights)
    local = EXPERTS // WORLD
    span = (args.rank * local, (args.rank + 1) * local)
    dense = dequant_all(ex)
    print(f"loaded layer {args.layer} rank {args.rank} in {time.time() - t0:.0f}s", flush=True)
    result: dict = {
        "layer": args.layer,
        "rank": args.rank,
        "random_weights": args.random_weights,
        "config": {
            "BLOCK_M": prefill_moe.BLOCK_M,
            "BLOCK_N": prefill_moe.BLOCK_N,
            "BLOCK_N_DOWN": prefill_moe.BLOCK_N_DOWN,
            "BLOCK_K": prefill_moe.BLOCK_K,
            "PREFETCH": prefill_moe.PREFETCH,
            "STAGES": prefill_moe.STAGES,
        },
        "rows": [],
    }
    gen = torch.Generator(device=dev).manual_seed(0)
    for t in args.tokens:
        x = torch.randn(t, HIDDEN, device=dev, generator=gen).to(torch.bfloat16)
        routing = routing_for(x, router)
        live = ((routing[0] >= span[0]) & (routing[0] < span[1])).sum().item()
        want = oracle(x, dense, routing, span)

        def old(x=x, routing=routing):
            return mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, span, total_experts=EXPERTS)

        def new(x=x, routing=routing):
            return prefill_moe.prefill_moe(x, ex, routing, TOP_K, span)

        row = {
            "tokens": t,
            "local_assignments": live,
            "old_err": rel_err(old(), want),
            "new_err": rel_err(new(), want),
            "old_us": time_us(old, args.reps),
            "new_us": time_us(new, args.reps),
        }
        row["speedup"] = row["old_us"] / row["new_us"]
        row["new_prefill_ms_60_layers"] = row["new_us"] * 60 / 1e3
        row["old_prefill_ms_60_layers"] = row["old_us"] * 60 / 1e3
        a_expert, a_weight = routing
        row["align_us"] = time_us(
            lambda a_expert=a_expert, a_weight=a_weight: prefill_moe.prefill_align(
                a_expert, a_weight, span, prefill_moe.BLOCK_M
            ),
            args.reps,
        )
        result["rows"].append(row)
        print(json.dumps(row), flush=True)

    if not args.no_sweep and 2048 in args.tokens:
        x = torch.randn(2048, HIDDEN, device=dev, generator=gen).to(torch.bfloat16)
        routing = routing_for(x, router)
        want = oracle(x, dense, routing, span)
        sweep = []
        # (block_m, block_n, block_n_down, block_k, warps, prefetch, num_stages); the first is
        # the default. num_stages only applies without prefetch (the prefetch loop compiles at 1).
        configs = [
            (64, 64, 128, 128, 4, True, 1),
            (64, 64, 128, 64, 4, True, 1),
            (64, 64, 128, 64, 4, False, 1),
            (64, 64, 128, 64, 4, False, 2),
            (64, 64, 128, 64, 4, False, 3),
            (64, 64, 128, 128, 4, False, 2),
            (64, 64, 128, 64, 8, False, 2),
            (128, 64, 128, 64, 4, False, 2),
            (64, 64, 128, 64, 8, True, 1),
            (32, 64, 128, 64, 4, True, 1),
            (128, 64, 128, 64, 4, True, 1),
            (64, 64, 64, 64, 8, True, 1),
        ]
        for bm, bn, bnd, bk, nw, pf, ns in configs:

            def fn(bm=bm, bn=bn, bnd=bnd, bk=bk, nw=nw, pf=pf, ns=ns):
                return prefill_moe.prefill_moe(
                    x,
                    ex,
                    routing,
                    TOP_K,
                    span,
                    block_m=bm,
                    block_n=bn,
                    block_n_down=bnd,
                    block_k=bk,
                    num_warps=nw,
                    prefetch=pf,
                    num_stages=ns,
                )

            item = {
                "block_m": bm,
                "block_n": bn,
                "block_n_down": bnd,
                "block_k": bk,
                "warps": nw,
                "prefetch": pf,
                "num_stages": ns,
            }
            try:
                item.update(err=rel_err(fn(), want), us=time_us(fn, args.reps))
            except Exception as exc:  # noqa: BLE001 - a failing config is a sweep result
                item["error"] = repr(exc)[:300]
            sweep.append(item)
            print("sweep", json.dumps(item), flush=True)
        result["sweep_t2048"] = sweep

    result["context_t2048"] = context_ops(dev, args.reps)
    print("context", json.dumps(result["context_t2048"]), flush=True)
    result["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} ({result['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
