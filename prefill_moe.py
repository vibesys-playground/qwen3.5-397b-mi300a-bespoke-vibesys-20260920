"""Prefill-shaped MXFP4 MoE: an expert-grouped GEMM on MFMA (`SEED_PREFILL_GROUPED_MOE=1`).

Idea (published, not copied): MegaBlocks-style grouped GEMM (Gale et al., "MegaBlocks:
Efficient Sparse Training with Mixture-of-Experts", 2022) and the fused-MoE kernel concept
from the Triton tutorials/blog posts: sort (token, expert) assignments by expert, pad each
expert's run to a block multiple, and run one tiled GEMM whose M tile is a block of rows
sharing one expert's weights. Weights are dequantized MXFP4 -> bf16 in registers inside the
K loop (no dense weight copy in HBM) and multiplied with `tl.dot`, i.e. MFMA on gfx942.

What it replaces. At prefill `mxfp4_gemv.fused_moe` already takes its de-duplicating
`_grouped_moe` path (T=2048 is 20,480 assignments, far above `use_grouped`'s threshold). That
path was tuned for decode-to-512-token shapes: `BLOCK_M=16`, so a local expert's weights are
re-dequantized once per 16 rows (~3x at 40 rows/expert); the scale is fetched and multiplied
per element; the down kernel writes an un-weighted `y[A, hidden]` for *all* assignments
(zero-filled, 168 MB at T=2048) and the combine is four torch ops over it.

This path, per layer at T=2048 on one rank (128 local experts, ~40 rows each):

1. `prefill_align`: fixed-shape torch sort into `BLOCK_M`-row blocks per local expert, with
   the inverse map `assign_slot` (assignment -> sorted row) for the combine. Non-local and
   zero-weight assignments go to a sentinel bucket whose blocks exit on their first load.
2. `_pf_gate_up_kernel`: one program = (block, `BLOCK_N` intermediate columns); gate and up
   tiles share the gathered activation tile; SiLU(gate) * up in the epilogue; output stored
   at the *sorted* row, so the down kernel reads contiguous rows (no gather).
3. `_pf_down_kernel`: one program = (block, `BLOCK_N_DOWN` hidden columns); the routing
   weight is applied in the epilogue, so `y[slot]` is already weighted.
4. `_pf_combine_kernel`: one program per (token, column tile) sums the token's <= top_k live
   rows of `y` via `assign_slot`. No atomics (deterministic), no zero-fill of `y` or `out`.

Dequant cost is what `BLOCK_M` amortizes: each block dequantizes its expert's whole weight
once, so at `BLOCK_M=64` the ~5 VALU instructions per weight value are spread over 64 MFMA
rows instead of 16 (see PREFILL_ROOFLINE.md for the budget).

Software pipelining is by hand: each K trip issues the next trip's weight words, scales and
activation tile before decoding the current one, then a side-effecting empty asm (the same
device `mxfp4_gemv._bw_fence` uses) keeps LLVM from sinking those loads back to their use.

Everything is launch-shape-static in `T` (no host reads), though prefill is not captured.
"""

from __future__ import annotations

import os

import mxfp4_gemv
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

ENABLED = os.environ.get("SEED_PREFILL_GROUPED_MOE", "0") not in ("0", "", "false", "False")
"""`SEED_PREFILL_GROUPED_MOE=1` routes MoE calls with at least `MIN_TOKENS` tokens here."""

MIN_TOKENS = int(os.environ.get("SEED_PREFILL_MOE_MIN_TOKENS", "256"))
"""Token count at or above which the grouped-GEMM path is used (prefill shapes only).

At 256 tokens (2,560 assignments over 512 experts, 5 per expert) a 64-row block is ~8% live,
so the break-even against `_grouped_moe` (BLOCK_M 16) is somewhere near here;
`bench_prefill_moe.py` measures it."""

BLOCK_M = int(os.environ.get("SEED_PREFILL_MOE_BLOCK_M", "64"))
"""Rows (assignments of one expert) per block. >= 16 for `tl.dot`; power of two."""

BLOCK_N = int(os.environ.get("SEED_PREFILL_MOE_BLOCK_N", "64"))
"""gate_up output columns per program (the matching up columns come along: 2x weight rows)."""

BLOCK_N_DOWN = int(os.environ.get("SEED_PREFILL_MOE_BLOCK_N_DOWN", "128"))
"""down output columns per program."""

BLOCK_K = int(os.environ.get("SEED_PREFILL_MOE_BLOCK_K", "128"))
"""Reduction elements per K trip; a multiple of 32 (the MXFP4 scale block).

128: on MI300A (layer 30, one rank) it beats 64 by 6-11% at T=256..4096 (e.g. 1730 vs
1845 us/layer at T=2048), although offline ISA put it at 264-284 VGPRs (occupancy 1)."""

WARPS = int(os.environ.get("SEED_PREFILL_MOE_WARPS", "4"))
"""Warps per program. 8 halves each wave's tile (occupancy 3-4 at BLOCK_K 64) but spreads
each MFMA tile thinner; `bench_prefill_moe.py`'s sweep measures both."""

PREFETCH = os.environ.get("SEED_PREFILL_MOE_PREFETCH", "1") != "0"
"""Hand software-pipelining of the K loop (see module docstring). `0` for A/B."""

STAGES = int(os.environ.get("SEED_PREFILL_MOE_STAGES", "2"))
"""Triton `num_stages` for the non-prefetch K loop (the compiler's own software pipeliner).
The prefetch loop always compiles with `num_stages=1`: its fence is an inline asm, which the
AMD stream pipeliner cannot predicate (Triton 3.4: "pipeliner doesn't know how to predicate
this op"), and the loop is already pipelined by hand."""

COMBINE_BLOCK = 512

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def available(device: torch.device, tokens: int) -> bool:
    """Whether `Model._routed_fused` should call `prefill_moe` for a `tokens`-row MoE call."""
    return ENABLED and HAVE_TRITON and device.type == "cuda" and tokens >= MIN_TOKENS


def block_count(assignments: int, local_experts: int, block_m: int) -> int:
    """Static upper bound on `prefill_align`'s blocks: one partial block per bucket extra."""
    return -(-assignments // block_m) + local_experts + 1


def prefill_align(
    a_expert: torch.Tensor, a_weight: torch.Tensor, expert_range: tuple[int, int], block_m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort assignments into per-local-expert blocks of `block_m` rows. No host sync.

    Returns `(slot_assign, block_expert, assign_slot)`:

    - `slot_assign[s]`: the assignment in sorted row `s`, or -1 for padding.
    - `block_expert[b]`: block `b`'s local expert, or `local_experts` (the sentinel) for a
      block of dropped assignments or past the last real block.
    - `assign_slot[a]`: assignment `a`'s sorted row, or -1 when this rank drops it (not local,
      or zero routing weight).

    Same construction as `mxfp4_gemv.align_blocks` (counts -> per-bucket block starts ->
    stable sort -> rank within bucket), plus the inverse map the combine needs.
    """
    lo, hi = expert_range
    experts = hi - lo
    assignments = a_expert.numel()
    device = a_expert.device
    live = (a_expert >= lo) & (a_expert < hi) & (a_weight != 0)
    bucket = torch.where(live, a_expert.to(torch.long) - lo, experts)

    counts = torch.zeros(experts + 1, dtype=torch.long, device=device)
    counts.scatter_add_(0, bucket, torch.ones_like(bucket))
    blocks = -(-counts // block_m)
    block_start = blocks.cumsum(0) - blocks

    order = torch.argsort(bucket, stable=True)
    sorted_bucket = bucket[order]
    rank = torch.arange(assignments, device=device) - (counts.cumsum(0) - counts)[sorted_bucket]
    slot = block_start[sorted_bucket] * block_m + rank

    n_blocks = block_count(assignments, experts, block_m)
    slot_assign = torch.full((n_blocks * block_m,), -1, dtype=torch.int32, device=device)
    slot_assign.scatter_(0, slot, order.to(torch.int32))
    assign_slot = torch.empty(assignments, dtype=torch.int32, device=device)
    assign_slot.scatter_(0, order, torch.where(sorted_bucket < experts, slot, -1).to(torch.int32))

    marks = torch.full((n_blocks + 1,), -1, dtype=torch.long, device=device)
    marks.scatter_reduce_(
        0, block_start.clamp(max=n_blocks), torch.arange(experts + 1, device=device), reduce="amax"
    )
    block_expert = marks.cummax(0).values[:n_blocks].to(torch.int32)
    return slot_assign, block_expert, assign_slot


if HAVE_TRITON:
    _MXB = tl.constexpr(32)
    _INTERP = tl.constexpr(_INTERPRET)

    @triton.jit
    def _pf_fence(ptr):
        """A zero the compiler cannot see through, produced after the loads issued above it.

        The asm clobbers memory, so no load crosses it (see `mxfp4_gemv._bw_fence`); callers
        add the zero into the *current* trip's words, so that trip's decode cannot be
        scheduled above it either. Without the second half LLVM hoisted the decode over the
        prefetch and the loads issued at the end of the trip (checked in the gfx942 ISA)."""
        if _INTERP:
            return tl.zeros([1], tl.int32)
        else:
            return tl.inline_asm_elementwise(
                "v_mov_b32 $0, 0",
                "=v,v,~{memory}",
                [ptr + tl.zeros([1], tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

    @triton.jit
    def _pf_load_w(wq_ptr, ws_ptr, rows, k0, K: tl.constexpr, BLOCK_K: tl.constexpr):
        """Raw words `[R, BK/8]` (int32, 8 fp4 each) and scales `[R, BK/32]` for `rows`.

        `wq_ptr`/`ws_ptr` already point at this expert's payload (a 64-bit scalar offset,
        computed once), so the per-element offsets here are 32-bit."""
        kw = tl.arange(0, BLOCK_K // 8)
        g = tl.arange(0, BLOCK_K // _MXB)
        words = tl.load(
            wq_ptr.to(tl.pointer_type(tl.int32))
            + rows[:, None] * (K // 8)
            + (k0 // 8 + kw[None, :])
        )
        sc = tl.load(ws_ptr + rows[:, None] * (K // _MXB) + (k0 // _MXB + g[None, :]))
        return words, sc

    @triton.jit
    def _pf_decode(words, sc, R: tl.constexpr, BLOCK_K: tl.constexpr, PERM: tl.constexpr):
        """`_pf_load_w`'s output -> `[R, BK]` bf16 (fp32 under the interpreter), scale applied.

        fp4 x 2^(s-127) is exact in fp32 and has <= 2 mantissa bits, so dropping the low 16
        bits is an exact fp32 -> bf16 conversion (cheaper than the rounding convert gfx942
        would otherwise emulate)."""
        if PERM:
            vals = mxfp4_gemv._fp4_words_to_values(words)  # [R, BK] fp32
        else:
            b0 = words & 0xFF
            b1 = (words >> 8) & 0xFF
            b2 = (words >> 16) & 0xFF
            b3 = (words >> 24) & 0xFF
            by = tl.interleave(tl.interleave(b0, b2), tl.interleave(b1, b3))  # bytes in order
            vals = tl.interleave(
                mxfp4_gemv._fp4_magnitudes(by & 0xF), mxfp4_gemv._fp4_magnitudes(by >> 4)
            )
        vals = tl.reshape(vals, (R, BLOCK_K // _MXB, _MXB)) * mxfp4_gemv._e8m0(sc)[:, :, None]
        vals = tl.reshape(vals, (R, BLOCK_K))
        if _INTERP:
            return vals
        bits = vals.to(tl.uint32, bitcast=True) >> 16
        return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _pf_gate_up_trip(
        wg, sg, wu, su, xa, acc_g, acc_u, BN: tl.constexpr, BK: tl.constexpr, PERM: tl.constexpr
    ):
        """One K trip of gate_up: decode both weight tiles, two MFMA dots (transposed:
        `acc[n, m]`, so the decoded weights are the A operand)."""
        if _INTERP:
            xa = xa.to(tl.float32)  # the interpreter's bf16 dot is wrong (numpy has no bf16)
        vg = _pf_decode(wg, sg, BN, BK, PERM)
        vu = _pf_decode(wu, su, BN, BK, PERM)
        acc_g = tl.dot(vg, tl.trans(xa), acc_g)
        acc_u = tl.dot(vu, tl.trans(xa), acc_u)
        return acc_g, acc_u

    @triton.jit
    def _pf_down_trip(wd, sd, xa, acc, BN: tl.constexpr, BK: tl.constexpr, PERM: tl.constexpr):
        """One K trip of down."""
        if _INTERP:
            xa = xa.to(tl.float32)
        return tl.dot(_pf_decode(wd, sd, BN, BK, PERM), tl.trans(xa), acc)

    @triton.jit
    def _pf_gate_up_kernel(
        x_ptr,
        wq_ptr,
        ws_ptr,
        slot_assign_ptr,
        block_expert_ptr,
        inter_ptr,
        experts,
        k_trips,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
        PREFETCH: tl.constexpr,
    ):
        """`inter[s, n] = silu(x[t(s)] @ Wg[e, n]) * (x[t(s)] @ Wu[e, n])` for one block."""
        pid_m = tl.program_id(0)
        expert = tl.load(block_expert_ptr + pid_m)
        if (expert < 0) | (expert >= experts):
            return
        slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = tl.load(slot_assign_ptr + slots)
        if tl.max(rows) < 0:
            return
        token = tl.where(rows >= 0, rows, 0) // TOP_K  # padding rows read token 0, never stored
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        e64 = expert.to(tl.int64) * (2 * INTER)
        wq_ptr += e64 * (K // 2)
        ws_ptr += e64 * (K // _MXB)
        rows_g = n
        rows_u = INTER + n
        kk = tl.arange(0, BLOCK_K)
        x_row = x_ptr + token[:, None] * K + kk[None, :]
        acc_g = tl.zeros([BLOCK_N, BLOCK_M], tl.float32)
        acc_u = tl.zeros([BLOCK_N, BLOCK_M], tl.float32)
        if PREFETCH:
            # Ping-pong over two register sets, two trips per loop iteration: set 0 is
            # reloaded only after its own decode, so the loop-carried value needs no copy
            # (a copy of an in-flight load forces a wait on it mid-trip). `k_trips` is even.
            _pf_fence(wq_ptr)
            wg0, sg0 = _pf_load_w(wq_ptr, ws_ptr, rows_g, 0, K, BLOCK_K)
            wu0, su0 = _pf_load_w(wq_ptr, ws_ptr, rows_u, 0, K, BLOCK_K)
            xa0 = tl.load(x_row)
            for t in range(0, k_trips, 2):
                k1 = (t + 1) * BLOCK_K
                wg1, sg1 = _pf_load_w(wq_ptr, ws_ptr, rows_g, k1, K, BLOCK_K)
                wu1, su1 = _pf_load_w(wq_ptr, ws_ptr, rows_u, k1, K, BLOCK_K)
                xa1 = tl.load(x_row + k1)
                z = _pf_fence(wq_ptr)[:, None]
                acc_g, acc_u = _pf_gate_up_trip(
                    wg0 + z, sg0, wu0 + z, su0, xa0, acc_g, acc_u, BLOCK_N, BLOCK_K, PERM
                )
                k2 = (t + 2) * BLOCK_K
                k2 = tl.where(k2 < K, k2, 0)  # the last reload re-reads trip 0: in bounds, unused
                wg0, sg0 = _pf_load_w(wq_ptr, ws_ptr, rows_g, k2, K, BLOCK_K)
                wu0, su0 = _pf_load_w(wq_ptr, ws_ptr, rows_u, k2, K, BLOCK_K)
                xa0 = tl.load(x_row + k2)
                z = _pf_fence(wq_ptr)[:, None]
                acc_g, acc_u = _pf_gate_up_trip(
                    wg1 + z, sg1, wu1 + z, su1, xa1, acc_g, acc_u, BLOCK_N, BLOCK_K, PERM
                )
        else:
            for t in range(0, k_trips):
                k0 = t * BLOCK_K
                wg, sg = _pf_load_w(wq_ptr, ws_ptr, rows_g, k0, K, BLOCK_K)
                wu, su = _pf_load_w(wq_ptr, ws_ptr, rows_u, k0, K, BLOCK_K)
                xa = tl.load(x_row + k0)
                acc_g, acc_u = _pf_gate_up_trip(
                    wg, sg, wu, su, xa, acc_g, acc_u, BLOCK_N, BLOCK_K, PERM
                )
        act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(
            inter_ptr + slots.to(tl.int64)[None, :] * INTER + n[:, None],
            act.to(inter_ptr.dtype.element_ty),
        )

    @triton.jit
    def _pf_down_kernel(
        inter_ptr,
        wq_ptr,
        ws_ptr,
        slot_assign_ptr,
        block_expert_ptr,
        a_weight_ptr,
        y_ptr,
        experts,
        k_trips,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
        PREFETCH: tl.constexpr,
    ):
        """`y[s, n] = w(s) * (inter[s] @ Wd[e, n])` for one block; `inter` rows contiguous."""
        pid_m = tl.program_id(0)
        expert = tl.load(block_expert_ptr + pid_m)
        if (expert < 0) | (expert >= experts):
            return
        slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = tl.load(slot_assign_ptr + slots)
        if tl.max(rows) < 0:
            return
        row_ok = rows >= 0
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        e64 = expert.to(tl.int64) * HIDDEN
        wq_ptr += e64 * (K // 2)
        ws_ptr += e64 * (K // _MXB)
        rows_d = n
        kk = tl.arange(0, BLOCK_K)
        a_row = inter_ptr + slots.to(tl.int64)[:, None] * K + kk[None, :]
        acc = tl.zeros([BLOCK_N, BLOCK_M], tl.float32)
        if PREFETCH:  # same ping-pong as `_pf_gate_up_kernel`
            _pf_fence(wq_ptr)
            wd0, sd0 = _pf_load_w(wq_ptr, ws_ptr, rows_d, 0, K, BLOCK_K)
            xa0 = tl.load(a_row)
            for t in range(0, k_trips, 2):
                k1 = (t + 1) * BLOCK_K
                wd1, sd1 = _pf_load_w(wq_ptr, ws_ptr, rows_d, k1, K, BLOCK_K)
                xa1 = tl.load(a_row + k1)
                z = _pf_fence(wq_ptr)[:, None]
                acc = _pf_down_trip(wd0 + z, sd0, xa0, acc, BLOCK_N, BLOCK_K, PERM)
                k2 = (t + 2) * BLOCK_K
                k2 = tl.where(k2 < K, k2, 0)
                wd0, sd0 = _pf_load_w(wq_ptr, ws_ptr, rows_d, k2, K, BLOCK_K)
                xa0 = tl.load(a_row + k2)
                z = _pf_fence(wq_ptr)[:, None]
                acc = _pf_down_trip(wd1 + z, sd1, xa1, acc, BLOCK_N, BLOCK_K, PERM)
        else:
            for t in range(0, k_trips):
                k0 = t * BLOCK_K
                wd, sd = _pf_load_w(wq_ptr, ws_ptr, rows_d, k0, K, BLOCK_K)
                xa = tl.load(a_row + k0)
                acc = _pf_down_trip(wd, sd, xa, acc, BLOCK_N, BLOCK_K, PERM)
        w = tl.load(a_weight_ptr + tl.where(row_ok, rows, 0), mask=row_ok, other=0.0)
        acc = acc * w.to(tl.float32)[None, :]
        tl.store(
            y_ptr + slots.to(tl.int64)[None, :] * HIDDEN + n[:, None],
            acc.to(y_ptr.dtype.element_ty),
        )

    @triton.jit
    def _pf_combine_kernel(
        y_ptr,
        assign_slot_ptr,
        out_ptr,
        HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`out[t] = sum_j y[assign_slot[t, j]]` over the token's live assignments."""
        token = tl.program_id(0)
        n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], tl.float32)
        for j in range(TOP_K):
            s = tl.load(assign_slot_ptr + token * TOP_K + j)
            if s >= 0:
                acc += tl.load(y_ptr + s.to(tl.int64) * HIDDEN + n).to(tl.float32)
        tl.store(out_ptr + token.to(tl.int64) * HIDDEN + n, acc.to(out_ptr.dtype.element_ty))


def prefill_moe(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_n_down: int | None = None,
    block_k: int | None = None,
    prefetch: bool | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Routed-expert output `[T, hidden]`: `mxfp4_gemv.fused_moe`'s contract, prefill-tuned.

    `routing` is `(a_expert, a_weight)`, both `[T * top_k]` (global expert ids, token-major).
    Returns this rank's partial sum; the caller all-reduces. The `block_*`/`prefetch`/`num_*`
    overrides exist for the microbenchmark and tests; the module constants are the defaults.
    """
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `available`
        raise RuntimeError("triton is required for the prefill grouped MoE")
    bm, bn = block_m or BLOCK_M, block_n or BLOCK_N
    bnd, bk = block_n_down or BLOCK_N_DOWN, block_k or BLOCK_K
    pf = PREFETCH if prefetch is None else prefetch
    warps = num_warps or WARPS
    stages = 1 if pf else (num_stages or STAGES)
    a_expert, a_weight = routing
    lo, hi = expert_range
    tokens, hidden_in = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(f"routing must be two [{assignments}] tensors")
    for name, tensor in (("x", x), ("a_expert", a_expert), ("a_weight", a_weight)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for dim, block, what in (
        (hidden_in, bk, "hidden/BLOCK_K"),
        (intermediate, bk, "intermediate/BLOCK_K"),
        (intermediate, bn, "intermediate/BLOCK_N"),
        (hidden, bnd, "hidden/BLOCK_N_DOWN"),
        (hidden, COMBINE_BLOCK if hidden >= COMBINE_BLOCK else hidden, "hidden/combine"),
    ):
        if dim % block:
            raise ValueError(f"prefill_moe needs {what} to divide exactly ({dim} % {block})")
    if bk % 32:
        raise ValueError("BLOCK_K must be a multiple of 32")
    if pf and ((hidden_in // bk) % 2 or (intermediate // bk) % 2):
        raise ValueError("the prefetching K loop runs two trips per iteration: K/BLOCK_K even")

    slot_assign, block_expert, assign_slot = prefill_align(a_expert, a_weight, expert_range, bm)
    n_blocks = block_expert.numel()
    n_slots = n_blocks * bm
    inter = torch.empty(n_slots, intermediate, dtype=x.dtype, device=x.device)
    y = torch.empty(n_slots, hidden, dtype=x.dtype, device=x.device)
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=x.device)
    perm = mxfp4_gemv.perm_lut(x.device)
    local = hi - lo

    _pf_gate_up_kernel[(n_blocks, intermediate // bn)](
        x,
        experts["gate_up"],
        experts["gate_up_scale"],
        slot_assign,
        block_expert,
        inter,
        local,
        hidden_in // bk,
        K=hidden_in,
        INTER=intermediate,
        TOP_K=top_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        PERM=perm,
        PREFETCH=pf,
        num_warps=warps,
        num_stages=stages,
    )
    _pf_down_kernel[(n_blocks, hidden // bnd)](
        inter,
        experts["down"],
        experts["down_scale"],
        slot_assign,
        block_expert,
        a_weight,
        y,
        local,
        intermediate // bk,
        K=intermediate,
        HIDDEN=hidden,
        BLOCK_M=bm,
        BLOCK_N=bnd,
        BLOCK_K=bk,
        PERM=perm,
        PREFETCH=pf,
        num_warps=warps,
        num_stages=stages,
    )
    cb = min(COMBINE_BLOCK, hidden)
    _pf_combine_kernel[(tokens, hidden // cb)](
        y, assign_slot, out, HIDDEN=hidden, TOP_K=top_k, BLOCK=cb, num_warps=4
    )
    return out
