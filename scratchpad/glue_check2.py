"""GPU exactness + graph-replay timing for SEED_ELEMWISE_FUSED and SEED_DN_NORM_F32_IN kernels."""
import os, sys
os.environ["SEED_ELEMWISE_FUSED"] = "1"
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decode_glue, deltanet_fused
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skinny_bench import graph_time
torch.cuda.set_device(0)
for b in (1, 16, 48):
    gu = torch.randn(b, 512, dtype=torch.bfloat16, device="cuda") * 3
    g, u = gu.chunk(2, -1)
    ref = F.silu(g) * u; got = decode_glue.silu_mul(gu)
    neq = (ref != got).sum().item()
    print(f"silu_mul b{b}: mismatches {neq}/{ref.numel()} maxdiff {(ref.float()-got.float()).abs().max().item():.3e} "
          f"fused={graph_time([lambda: decode_glue.silu_mul(gu)]):.2f}us unfused={graph_time([lambda: F.silu(gu.chunk(2,-1)[0]) * gu.chunk(2,-1)[1]]):.2f}us", flush=True)
    qg = torch.randn(b, 4096, dtype=torch.bfloat16, device="cuda").view(b, 1, 8, 512) * 3
    q, gate = qg.chunk(2, -1)
    out = torch.randn(b, 1, 2048, dtype=torch.bfloat16, device="cuda")
    ref = out * torch.sigmoid(gate.reshape(b, 1, -1)); got = decode_glue.sigmoid_gate_mul(out, gate)
    print(f"sigmoid_gate b{b}: mismatches {(ref != got).sum().item()}/{ref.numel()} "
          f"fused={graph_time([lambda: decode_glue.sigmoid_gate_mul(out, gate)]):.2f}us unfused={graph_time([lambda: out * torch.sigmoid(gate.reshape(b, 1, -1))]):.2f}us", flush=True)
    x32 = torch.randn(b, 16, 128, device="cuda"); zg = torch.randn(b, 16, 128, dtype=torch.bfloat16, device="cuda")
    wn = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    ref = deltanet_fused.gated_rmsnorm(x32.to(torch.bfloat16), zg, wn, 1e-6)
    got = deltanet_fused.gated_rmsnorm(x32, zg, wn, 1e-6, torch.bfloat16)
    print(f"gated_rmsnorm f32in b{b}: exact={torch.equal(ref, got)} fused={graph_time([lambda: deltanet_fused.gated_rmsnorm(x32, zg, wn, 1e-6, torch.bfloat16)]):.2f}us "
          f"unfused={graph_time([lambda: deltanet_fused.gated_rmsnorm(x32.to(torch.bfloat16), zg, wn, 1e-6)]):.2f}us", flush=True)
