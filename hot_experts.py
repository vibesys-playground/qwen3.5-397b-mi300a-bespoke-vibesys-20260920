"""Hot-expert bf16 cache for MoE decode (`SEED_HOT_EXPERTS=1`, default off).

Cited ideas:

- **Expert load balancing with redundant experts** (DeepSeek-V3's EPLB): placement decisions
  driven by *monitored* router load rather than a static assignment. Here the monitored load
  drives a per-rank *cache* rather than a placement change: `Model` accumulates each decode
  step's local-expert assignment counts (`observe_routing` in model.py) until
  `finalize_hot_experts` freezes, per layer, the `H` local experts this rank's own traffic hit
  most -- the standing hot set.
- **Caching dequantized hot weights**: a bf16 copy of only those `H` experts is kept per layer
  (`build_layer_cache`, via `mxfp4.dequant_mxfp4`), so decode assignments to them run a dense
  `torch.matmul` batched GEMM (bf16, graph-capturable: `H` and `t` are fixed by capture time,
  never a function of this step's routing) instead of the MXFP4 kernels' per-assignment
  dequant-in-kernel gather, which `model.py`'s module docstring/`DECODE_BOTTLENECK` notes is
  latency-bound (weight-load-per-assignment, not GEMM FLOPs).

Splice into the existing cold path: `hot_expert_forward` returns a `[t, top_k]` `consumed` mask
alongside its dense output. The caller (`Model.moe`) zeros `top_w` at `consumed` positions
before calling `_routed_fused`/`_routed_grouped` unmodified -- both already skip zero-weight
assignments (`mxfp4_gemv.py`'s `a_weight != 0` gate, the same mechanism `FAULT_DROP_EXPERT`
uses), so a hot assignment is computed exactly once, by exactly one path, with no kernel
changes needed.

Size: one expert's dense bf16 `gate_up` + `down` has no MXFP4 scale tensors and every packed
nibble becomes a full bf16 element, so it costs `2x` the packed-uint8 byte count (2 bf16 bytes
per unpacked value vs. 1 packed uint8 byte holding 2 values) plus change for the now-absent
scale bytes: roughly 6.68 MB MXFP4 -> ~26.7 MB bf16 for this model's expert shape (hidden=4096,
moe_intermediate=1024). `budget_h_per_layer` turns a per-rank GiB budget into a uniform
per-layer `H`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mxfp4_moe_bw2
import torch
import torch.nn.functional as F
from mxfp4 import dequant_mxfp4

ENABLED = os.environ.get("SEED_HOT_EXPERTS", "0") != "0"
REDUNDANT_ENABLED = os.environ.get("SEED_REDUNDANT_EXPERTS", "0") != "0"
# "≤20 GiB via SEED_HOT_EXPERT_GIB" (COMMON_BRIEF.md): clamp so a misconfigured env var cannot
# eat into the stage-2 allocator's `SEED_MIN_UNALLOCATED_GIB` (model.py) headroom on its own.
BUDGET_GIB = min(20.0, max(0.0, float(os.environ.get("SEED_HOT_EXPERT_GIB", "16"))))


@dataclass
class LayerCache:
    """One layer's hot-expert bf16 cache.

    `ids` are LOCAL expert indices (offsets into this rank's `[expert_range[0],
    expert_range[1])` window), ascending, no duplicates. An empty cache (`ids.numel() == 0`,
    the state before `finalize_hot_experts` runs, or for a layer no warmup traffic touched) is
    valid: `hot_expert_forward` then contributes nothing and the whole step falls through to the
    cold path exactly as if `SEED_HOT_EXPERTS` were off.
    """

    ids: torch.Tensor
    gate_up: torch.Tensor  # [H, 2*inter, hidden]
    down: torch.Tensor  # [H, hidden, inter]


def per_expert_bf16_bytes(ex: dict) -> int:
    """Bytes for one expert's dense bf16 `gate_up` + `down` (no scale tensors in bf16)."""
    gate_up = ex["gate_up"]
    down = ex["down"]
    gate_up_elems = gate_up.shape[-2] * gate_up.shape[-1] * 2  # packed K/2 uint8 -> K values
    down_elems = down.shape[-2] * down.shape[-1] * 2
    return (gate_up_elems + down_elems) * 2  # 2 bytes/element in bf16


def budget_h_per_layer(
    gib: float, num_moe_layers: int, bytes_per_expert: int, local_experts: int
) -> int:
    """How many hot experts/layer a `gib`-GiB/rank budget affords, split evenly over layers."""
    if num_moe_layers <= 0 or bytes_per_expert <= 0 or gib <= 0:
        return 0
    per_layer_bytes = (gib * (1024**3)) / num_moe_layers
    return max(0, min(int(per_layer_bytes // bytes_per_expert), local_experts))


def select_hot_local(counts: torch.Tensor, h: int) -> torch.Tensor:
    """The `h` local expert indices with the highest `counts`, ascending.

    Ties break by index (lower local id wins): `argsort(..., stable=True)` preserves the
    original ascending order among equal counts, so the result is deterministic run to run for
    the same accumulated counts, which matters for the CPU parity tests and for reproducing a
    cache across ranks that happen to see identical traffic.
    """
    if h <= 0 or counts.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    order = torch.argsort(counts, descending=True, stable=True)
    return torch.sort(order[: min(h, counts.numel())]).values


def build_layer_cache(ex: dict, ids: torch.Tensor, dtype: torch.dtype) -> LayerCache:
    """Dequantize (or cast, for a dense checkpoint) just the `ids` experts into a bf16 cache."""
    device = ex["gate_up"].device
    if ids.numel() == 0:
        return LayerCache(
            ids=ids,
            gate_up=torch.empty(0, ex["gate_up"].shape[-2], 0, dtype=dtype, device=device),
            down=torch.empty(0, ex["down"].shape[-2], 0, dtype=dtype, device=device),
        )
    if "gate_up_scale" in ex:
        gu, gu_s, dn, dn_s = (
            ex[k][ids] for k in ("gate_up", "gate_up_scale", "down", "down_scale")
        )
        if "bw_layout" in ex:
            # `SEED_MOE_BW_VARIANT` reshuffled the MXFP4 bytes in place at load
            # (`mxfp4_moe_bw2.bw_shuffle`); undo it for just these experts before dequant.
            gu_bk, dn_bk = (int(v) for v in ex["bw_layout"].tolist())
            gu = mxfp4_moe_bw2.bw_unshuffle_packed(gu, gu_bk)
            gu_s = mxfp4_moe_bw2.bw_unshuffle_scale(gu_s, gu_bk)
            dn = mxfp4_moe_bw2.bw_unshuffle_packed(dn, dn_bk)
            dn_s = mxfp4_moe_bw2.bw_unshuffle_scale(dn_s, dn_bk)
        gate_up = dequant_mxfp4(gu, gu_s, dtype)
        down = dequant_mxfp4(dn, dn_s, dtype)
    else:
        gate_up, down = ex["gate_up"][ids].to(dtype), ex["down"][ids].to(dtype)
    return LayerCache(ids=ids.clone(), gate_up=gate_up.contiguous(), down=down.contiguous())


def hot_expert_forward(
    h: torch.Tensor, top_i: torch.Tensor, top_w: torch.Tensor, lo: int, cache: LayerCache
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense bf16 output for `cache`'s hot experts, plus the `[t, top_k]` `consumed` mask.

    Every hot expert's GEMM runs against every token (dense, not gathered by assignment): with
    `H` capped in the low tens (`budget_h_per_layer`) and `t` the decode batch, `H * t` is small
    and *fixed* by cache-build time, unlike the MXFP4 path's per-assignment shape -- the whole
    call is a function of `cache` and `h.shape`/`top_i.shape` alone, so it is graph-capturable
    as-is, matching COMMON_BRIEF.md's "torch/hipBLASLt grouped or batched matmul,
    graph-capturable" for this path.

    `token_w[e, j]` is nonzero only for a token whose top-k included hot expert `e`; the router
    never repeats an expert within one token's top-k (`probs.topk`), so at most one of that
    token's `top_k` slots can match a given hot expert and summing the `H` contributions is
    exact, not an approximation of a hard routing choice.
    """
    n_hot = cache.ids.shape[0]
    t = h.shape[0]
    if n_hot == 0:
        return torch.zeros_like(h), torch.zeros_like(top_i, dtype=torch.bool)
    hot_global = lo + cache.ids.to(top_i.device)  # [H]
    match = top_i.unsqueeze(-1) == hot_global  # [t, top_k, H]
    consumed = match.any(-1)  # [t, top_k]
    token_w = (top_w.unsqueeze(-1) * match.to(top_w.dtype)).sum(dim=1)  # [t, H]

    h_rep = h.unsqueeze(0).expand(n_hot, t, h.shape[-1])  # broadcast view, no copy
    gate, up = torch.matmul(h_rep, cache.gate_up.transpose(1, 2)).chunk(2, dim=-1)
    y = torch.matmul(F.silu(gate) * up, cache.down.transpose(1, 2))  # [H, t, hidden]
    out = (y * token_w.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    return out.to(h.dtype), consumed
