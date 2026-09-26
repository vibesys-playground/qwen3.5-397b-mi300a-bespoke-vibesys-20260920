"""GPU check for SEED_ADD_RMSNORM / SEED_SKINNY_SWIGLU kernels at decode shapes.

add_rmsnorm must be bit-identical to `x + r` then `rmsnorm_fused.rmsnorm`; the swiglu fold
is compared to the torch chain + hipBLASLt (max abs/rel diff). Times are per call inside a
captured graph (the replay cost)."""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rmsnorm_fused, skinny_gemm as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skinny_bench import graph_time, load_tunable
torch.cuda.set_device(0); load_tunable()
for b in (1, 16, 48):
    x = torch.randn(b, 1, 4096, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(b, 1, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda") * 0.1
    xo, h = rmsnorm_fused.add_rmsnorm(x, r, w, 1e-6)
    xr = x + r; hr = rmsnorm_fused.rmsnorm(xr, w, 1e-6)
    t_f = graph_time([lambda: rmsnorm_fused.add_rmsnorm(x, r, w, 1e-6)])
    t_u = graph_time([lambda: rmsnorm_fused.rmsnorm(x + r, w, 1e-6)])
    print(f"add_rmsnorm b{b}: exact x={torch.equal(xo, xr)} h={torch.equal(h, hr)} fused={t_f:.2f}us unfused(add+norm)={t_u:.2f}us", flush=True)
    hs = torch.randn(b, 4096, dtype=torch.bfloat16, device="cuda")
    gu_w = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda") * 0.02
    dn = [torch.randn(4096, 256, dtype=torch.bfloat16, device="cuda") * 0.05 for _ in range(64)]
    gu = F.linear(hs, gu_w)
    g, u = gu.chunk(2, -1)
    ref = F.linear(F.silu(g) * u, dn[0])
    got = sg.skinny_linear(gu, dn[0], act="swiglu")
    exact_ref = F.linear((F.silu(g) * u).float(), dn[0].float())
    d = (got.float() - exact_ref).abs().max().item(); d0 = (ref.float() - exact_ref).abs().max().item()
    t_f = graph_time([lambda d_=d_: sg.skinny_linear(gu, d_, act="swiglu") for d_ in dn])
    def unf(d_):
        g, u = gu.chunk(2, -1); return F.linear(F.silu(g) * u, d_)
    t_u = graph_time([lambda d_=d_: unf(d_) for d_ in dn])
    print(f"swiglu_down b{b}: maxdiff vs fp32 fold={d:.3e} blas={d0:.3e} (|ref|max={exact_ref.abs().max().item():.3f}) fused={t_f:.2f}us unfused(silu+mul+gemm)={t_u:.2f}us", flush=True)
