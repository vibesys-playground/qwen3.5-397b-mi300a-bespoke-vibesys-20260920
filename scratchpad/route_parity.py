"""SEED_MOE_ROUTE_FUSED routing parity on real router weights (one GPU), from the bundle root:
    MODEL_PATH=... python3 scratchpad/route_parity.py --layers 0 3 30 59 --tokens 4096

For each layer: routing = F.linear(h, [router; shared_gate]) in bf16 with h ~ N(0,1) bf16 (the
post-RMSNorm input scale), then the production torch chain (softmax fp32, topk, renorm) vs
`router_fused.route(..., index_dtype=int32)`. Reports: tokens whose expert *set* differs, and
for those whether the swapped experts tie exactly in fp32 probability; max |weight diff| after
matching by expert id; tokens whose id *order* differs. Also bw_combine_glue vs the unfused
bw_combine + sigmoid*shared chain on random y.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from safetensors import safe_open

import mxfp4_gemv
import router_fused

p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, nargs="+", default=[0, 3, 30, 59])
p.add_argument("--tokens", type=int, default=4096)
a = p.parse_args()
MP = os.environ["MODEL_PATH"]
wm = json.load(open(os.path.join(MP, "model.safetensors.index.json")))["weight_map"]
dev = torch.device("cuda:0")
K = 10


def get(name):
    with safe_open(os.path.join(MP, wm[name]), "pt", device="cpu") as f:
        return f.get_tensor(name)


for li in a.layers:
    pre = f"model.language_model.layers.{li}.mlp."
    rw = torch.cat([get(pre + "gate.weight"), get(pre + "shared_expert_gate.weight")]).to(dev, torch.bfloat16)
    E = rw.shape[0] - 1
    g = torch.Generator(device=dev).manual_seed(li)
    h = torch.randn(a.tokens, rw.shape[1], generator=g, device=dev).to(torch.bfloat16)
    routing = F.linear(h, rw)
    probs = routing[:, :E].softmax(-1, dtype=torch.float)
    bw, bi = probs.topk(K, dim=-1)
    bw = bw / bw.sum(-1, keepdim=True)
    fw, fi = router_fused.route(routing, E, K, index_dtype=torch.int32)
    fi = fi.long()
    set_b = torch.zeros(a.tokens, E, dtype=torch.bool, device=dev).scatter_(1, bi, True)
    set_f = torch.zeros(a.tokens, E, dtype=torch.bool, device=dev).scatter_(1, fi, True)
    diff_rows = (set_b != set_f).any(1).nonzero().flatten().tolist()
    tie_only = 0
    for r in diff_rows:
        only_b = (set_b[r] & ~set_f[r]).nonzero().flatten()
        only_f = (set_f[r] & ~set_b[r]).nonzero().flatten()
        kth = bw.new_tensor(probs[r].topk(K).values[-1].item())
        if torch.all(probs[r, only_b] == kth) and torch.all(probs[r, only_f] == kth):
            tie_only += 1
    same = [r for r in range(a.tokens) if r not in set(diff_rows)]
    sb = torch.zeros(a.tokens, E, device=dev).scatter_(1, bi, bw)
    sf = torch.zeros(a.tokens, E, device=dev).scatter_(1, fi, fw)
    wdiff = (sb[same] - sf[same]).abs().max().item() if same else 0.0
    order = int((bi != fi).any(1).sum())
    # exact-tie rate inside the top-k (ties anywhere among selected experts)
    srt = probs.sort(-1, descending=True).values[:, : K + 1]
    tie_in = int((srt[:, 1:] == srt[:, :-1]).any(1).sum())
    print(f"layer {li}: tokens={a.tokens} set_differs={len(diff_rows)} (all at exact fp32 ties: "
          f"{tie_only}) order_differs={order} rows_with_any_tie_in_top{K + 1}={tie_in} "
          f"max_weight_diff_same_set={wdiff:.3g}", flush=True)
    for r in diff_rows[:3]:
        print(f"  row {r}: base ids {sorted(bi[r].tolist())} fused ids {sorted(fi[r].tolist())} "
              f"base kth p={probs[r].topk(K + 1).values[-2:].tolist()}", flush=True)

# combine glue vs unfused chain
T, Hd, lo, hi = 48, 4096, 128, 256
g = torch.Generator(device=dev).manual_seed(1)
y = torch.randn(T * K, Hd, generator=g, device=dev).to(torch.bfloat16)
ae = torch.randint(0, 512, (T * K,), generator=g, device=dev, dtype=torch.int32)
aw = torch.rand(T * K, generator=g, device=dev)
shared = torch.randn(T, Hd, generator=g, device=dev).to(torch.bfloat16)
gate = torch.randn(T, 1, generator=g, device=dev).to(torch.bfloat16)
o1 = torch.empty(T, Hd, dtype=torch.bfloat16, device=dev)
mxfp4_gemv.bw_combine(y, ae, aw, (lo, hi), K, o1)
ref = o1 + torch.sigmoid(gate) * shared
o2 = torch.empty_like(o1)
mxfp4_gemv.bw_combine_glue(y, ae, aw, (lo, hi), K, shared, gate, o2)
d = (ref.float() - o2.float()).abs()
print(f"combine_glue vs unfused: max|diff|={d.max().item():.3g} max|ref|={ref.float().abs().max().item():.3g} "
      f"frac_elems_differ={(d > 0).float().mean().item():.3g}", flush=True)
