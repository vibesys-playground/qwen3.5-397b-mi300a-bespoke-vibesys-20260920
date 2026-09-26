"""GPU check for SEED_DN_DECODE_FUSED at TP=4 per-rank shapes: dn_decode_fused vs
causal_conv_decode + delta_rule_decode(active, lanes) + gated_rmsnorm(fp32 in), bit-for-bit on
the output, the conv state pool and the recurrent state pool. Padding rows share one inactive
lane, like a graph bucket. Repeats several steps (state carries), in a CUDA graph too, and
times both (us per layer)."""

import os, sys, time

sys.path.insert(0, os.getcwd())
import torch
import deltanet_fused as df

dev = torch.device("cuda:0")
KH, VH, D, K, L = 4, 16, 128, 4, 64
C = 2 * KH * D + VH * D
g = torch.Generator(device=dev).manual_seed(0)
cw = (torch.randn(C, 1, K, device=dev, generator=g) * 0.3).to(torch.bfloat16)
alog = torch.randn(VH, device=dev, generator=g) * 0.5
dtb = torch.randn(VH, device=dev, generator=g) * 0.5
nw = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
eps = 1e-6
bad = 0
for B in (1, 16, 48):
    real = max(1, B - 3)
    lanes = torch.cat(
        [torch.randperm(L - 1, device=dev)[:real], torch.full((B - real,), L - 1, device=dev)]
    )
    active = torch.arange(B, device=dev) < real
    cs0 = torch.randn(L, C, K - 1, device=dev, generator=g).to(torch.bfloat16)
    rec0 = torch.randn(L, VH, D, D, device=dev, generator=g) * 0.1
    W = C + VH * D + 2 * VH + 7  # a wider projection row, like in_proj_all's split
    proj = torch.randn(B, 1, W, device=dev, generator=g).to(torch.bfloat16)
    qkv = proj[..., :C]
    z = proj[..., C : C + VH * D].unflatten(-1, (VH, D))[:, 0]
    braw = proj[..., C + VH * D : C + VH * D + VH]
    araw = proj[..., C + VH * D + VH : C + VH * D + 2 * VH]

    def unfused(cs, rec):
        flat = df.causal_conv_decode(qkv, cw, cs, lanes, active)
        q = flat[:, : KH * D].unflatten(-1, (KH, D))
        k = flat[:, KH * D : 2 * KH * D].unflatten(-1, (KH, D))
        v = flat[:, 2 * KH * D :].unflatten(-1, (VH, D))
        o = df.delta_rule_decode(
            (q, k), v, (araw[:, 0], braw[:, 0]), rec, (alog, dtb), active=active, lanes=lanes
        )
        return df.gated_rmsnorm(o, z, nw, eps, torch.bfloat16)

    def fused(cs, rec):
        return df.dn_decode_fused(
            qkv,
            (cw, cs),
            (araw[:, 0], braw[:, 0]),
            rec,
            (alog, dtb),
            (active, lanes),
            (z, nw, eps),
            (KH, VH, D, D),
        )

    def fused_nc(cs, rec):
        flat = df.causal_conv_decode(qkv, cw, cs, lanes, active)
        return df.dn_decode_fused(
            flat,
            (cw, cs),
            (araw[:, 0], braw[:, 0]),
            rec,
            (alog, dtb),
            (active, lanes),
            (z, nw, eps),
            (KH, VH, D, D),
            fuse_conv=False,
        )

    c1, r1, c2, r2 = cs0.clone(), rec0.clone(), cs0.clone(), rec0.clone()
    c3, r3 = cs0.clone(), rec0.clone()
    for step in range(4):
        o3 = fused_nc(c3, r3)
        o1, o2 = unfused(c1, r1), fused(c2, r2)
        torch.cuda.synchronize()
        if not (torch.equal(o1[:real], o3[:real]) and torch.equal(c1, c3) and torch.equal(r1, r3)):
            bad += 1
            print(f"B{B} step{step} nc MISMATCH", flush=True)
        ok = torch.equal(o1[:real], o2[:real]) and torch.equal(c1, c2) and torch.equal(r1, r2)
        if not ok:
            bad += 1
            print(
                f"B{B} step{step} MISMATCH out={(o1[:real].float() - o2[:real].float()).abs().max().item():.3g} "
                f"conv={(c1.float() - c2.float()).abs().max().item():.3g} rec={(r1 - r2).abs().max().item():.3g}",
                flush=True,
            )
    # graph replay, interleaved with eager
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        og = fused(c2, r2)
    for step in range(3):
        o1 = unfused(c1, r1)
        gr.replay()
        torch.cuda.synchronize()
        if not (torch.equal(o1[:real], og[:real]) and torch.equal(c1, c2) and torch.equal(r1, r2)):
            bad += 1
            print(f"B{B} graph step{step} MISMATCH", flush=True)
    print(f"B{B} checked, bad so far {bad}", flush=True)
    for name, fn in (("unfused", unfused), ("fused", fused), ("fused_noconv", fused_nc)):
        cs, rec = cs0.clone(), rec0.clone()
        fn(cs, rec)
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            for _ in range(20):
                fn(cs, rec)
        gg.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            gg.replay()
        torch.cuda.synchronize()
        print(f"TIME B{B} {name} {(time.perf_counter() - t0) / 200 * 1e6:.2f} us/layer", flush=True)
print(f"DONE bad={bad}")
