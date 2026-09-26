"""Fused Triton kernel for `Model.moe`'s router: logits -> softmax -> top-k -> renormalize.

`Model.moe` spends four torch ops getting from the router's raw logits to the `(weight,
index)` pairs the expert kernels consume: `softmax(-1, dtype=float)`, `topk(top_k, dim=-1)`
(itself at least a sort/select kernel), a `sum(-1, keepdim=True)`, and a `/`. At `top_k=10`,
`experts=512` that is a `[t, 512]` softmax and a `[t, 512] -> [t, 10]` top-k, none of it large
enough to be worth its own launch at this step's dispatch-bound regime (see `deltanet_fused.py`
and `rmsnorm_fused.py`'s module docstrings for the same story at their own call sites; the
router measured 2.0 ms of an 79 ms decode step at batch 48, almost entirely dispatch on a
handful of small kernels). One program per token does the whole thing: a block-wide softmax
over the row, then `top_k` rounds of "find the largest remaining probability and its index,
mask it out" (`tl.argmax`/`tl.max` over the row, `top_k` times -- cheap relative to a sort at
`top_k` this small), then a normalize pass once every weight is known.

Not a reformulation with a different selection rule: ties in the softmax output (two experts
at the exact same float32 probability) can pick a different expert than `torch.topk`'s
implementation-defined tie-break, which is a real, if measure-zero-probability-under-continuous-
router-logits, semantic difference rather than fp32 reduction-order noise -- unlike this
module's siblings, whose only divergence from the torch chain is which bits of the last ULP a
sum lands on. `seed_tests/test_router_fused.py` uses random (not adversarially tied) router
weights, so this is a documented risk, not a validated one, and it is the reason
`SEED_FUSE_GLUE` is its own flag, off by default: enabling it changes routing, not just timing,
until it is checked against the accuracy pins on real hardware.

Behind `SEED_FUSE_GLUE` (default off) at its one call site in `Model.moe`. `available` mirrors
`deltanet_fused`'s and `rmsnorm_fused`'s contract.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

MAX_TOP_K = 32
"""Compile-time cap on `top_k`: the per-iteration selection loop unrolls `top_k` times, and
this bounds how large that gets. The real model's `top_k` is 10."""


def available(device: torch.device) -> bool:
    """Same contract as `deltanet_fused.available`/`rmsnorm_fused.available`."""
    if not HAVE_TRITON:
        return False
    return device.type == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _fused_route_kernel(
        routing_ptr,
        out_w_ptr,
        out_i_ptr,
        sroute_row,
        sw_row,
        si_row,
        experts_out,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per token: softmax over `experts_out` columns, then `TOP_K` rounds of
        argmax-and-mask, then renormalize the selected weights to sum to 1.

        `tl.argmax`'s tie-break (first-occurrence-wins under Triton's reduction, not
        necessarily `torch.topk`'s) is the one place this is not just a reformulation -- see
        the module docstring.

        The selection loop stores each round's (unnormalized) weight and index to
        `out_w_ptr`/`out_i_ptr` immediately, then a second, separate `TOP_K`-round loop
        re-reads and divides by `total` -- rather than buffering `TOP_K` scalars in a Python
        list across iterations and indexing it after the loop, which traces fine under
        `TRITON_INTERPRET=1` (a real Python interpreter, list and all) but is not part of the
        AST subset `triton.compile`'s real compiler path accepts (`list.append`/`__getitem__`
        on a plain Python list of traced values is not a supported op) -- caught by this
        module's own offline `gfx942` ISA compile, not by the interpreter test suite alone.
        """
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        live = c < experts_out
        logits = tl.load(routing_ptr + row * sroute_row + c, mask=live, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(logits, axis=0)
        p = tl.exp(logits - m)
        denom = tl.sum(tl.where(live, p, 0.0), axis=0)
        # Padding/out-of-range lanes get -1.0: strictly below any real softmax probability
        # (which lies in [0, 1]), so `tl.argmax` never selects one.
        remaining = tl.where(live, p / denom, -1.0)

        total = 0.0
        for t in range(TOP_K):
            idx = tl.argmax(remaining, axis=0)
            val = tl.max(remaining, axis=0)
            total += val
            tl.store(out_i_ptr + row * si_row + t, idx)
            tl.store(out_w_ptr + row * sw_row + t, val)  # unnormalized; the second loop divides
            remaining = tl.where(c == idx, -1.0, remaining)

        for t in range(TOP_K):
            raw = tl.load(out_w_ptr + row * sw_row + t)
            tl.store(out_w_ptr + row * sw_row + t, raw / total)


def route(
    routing: torch.Tensor,
    experts_out: int,
    top_k: int,
    *,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`softmax(routing[:, :experts_out]).topk(top_k)`, renormalized -- one kernel.

    `routing` is `[t, >= experts_out]` (the router's raw logits, possibly a wider row that
    also carries the shared-expert gate, as `fuse_moe_dense`'s output does -- only the row
    stride matters here, not the full width). Returns `(top_w, top_i)`: `top_w` is fp32
    `[t, top_k]`, summing to 1 per row; `top_i` is `index_dtype` `[t, top_k]`.

    `index_dtype` defaults to int64, matching `probs.topk`'s output dtype so the `SEED_FUSE_GLUE`
    call site (which hands `top_i` to callers expecting that contract) needs no cast.
    `SEED_MOE_ROUTE_FUSED` (`model.py`'s `_moe_hip_route_fused`) instead passes `torch.int32`:
    its only consumer is `mxfp4_gemv.bw_prep`'s `a_expert_ptr`, which reads whatever integer
    width the tensor already has, so writing int32 directly here removes a separate
    `top_i.to(torch.int32)` cast kernel from that path. `_fused_route_kernel` itself is
    unchanged either way -- `tl.store` casts to the output pointer's element type, so the
    kernel does not need to know which width it is writing.
    """
    if top_k > MAX_TOP_K:
        raise ValueError(f"route: top_k {top_k} exceeds MAX_TOP_K {MAX_TOP_K}")
    if index_dtype not in (torch.int64, torch.int32):
        raise ValueError(f"route: index_dtype must be int32 or int64, got {index_dtype}")
    t = routing.shape[0]
    top_w = torch.empty(t, top_k, dtype=torch.float32, device=routing.device)
    top_i = torch.empty(t, top_k, dtype=index_dtype, device=routing.device)
    _fused_route_kernel[(t,)](
        routing,
        top_w,
        top_i,
        routing.stride(0),
        top_w.stride(0),
        top_i.stride(0),
        experts_out,
        TOP_K=top_k,
        BLOCK=triton.next_power_of_2(experts_out),
    )
    return top_w, top_i
