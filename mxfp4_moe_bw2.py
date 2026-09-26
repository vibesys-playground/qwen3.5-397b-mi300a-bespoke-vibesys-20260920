"""Refined variants of the bandwidth-first MoE decode path (`SEED_MOE_BW=1`).

`SEED_MOE_BW_VARIANT` picks one; `v1` (the default) is `mxfp4_gemv.fused_moe_bw` unchanged, so
one build A/Bs every variant. The design and budget math of v1 are in `mxfp4_gemv`'s module
docstring; this file only states what each variant changes and why.

Variants (`VARIANTS`; knobs below override single fields):

    name  depth  shuffle  gate_up split-K  launches          decode
    v1    (v1: one-ahead prefetch through loop-carried copies, row-major weights)
    v2a   2      no       1                gate_up, down     v1 asm
    v2b   2      yes      1                gate_up, down     v1 asm
    v2c   2      yes      2                gate_up, down     v1 asm
    v2d   2      yes      2                one fused kernel  v1 asm
    v2e   4      yes      2                gate_up, down     v1 asm
    v2f   2      yes      2                gate_up, down     lean asm
    v2g   2      yes      2                one fused kernel  lean asm

1. Register rotation (`depth`, weak point 1). v1 carries "the next trip's loads" as loop
   values, so LLVM copies them into the registers the decode reads (`v_mov_b32` x 32 per
   trip) and must wait for the loads before copying: in the offline ISA the wait lands after
   ~3/4 of the trip's decode, so about half a trip of latency is hidden. Here the K loop is
   unrolled by `depth` with one register buffer per unrolled step (modulo variable
   expansion, Lam, PLDI 1988): step i decodes buffer i, then refills buffer i with trip
   i + depth. A refill writes the registers that step just finished reading, so no copy is
   needed, and `depth - 1` trips stay in flight during every decode. vmcnt retires in issue
   order, so waiting for buffer i never waits for the younger buffers. `depth` must divide
   the trip count (powers of two here, so 2 or 4; a third buffer would need a masked or
   redundant refill on the last group).

2. Load-time weight reshuffle (`shuffle`, weak point 2). v1's `global_load_dwordx4` for one
   register slot reads 16 bytes from each of 64 rows (16 rows x 4 lane groups, stride K/2),
   i.e. 64 separate 16-byte pieces per wave instruction. `bw_shuffle` rewrites each expert
   once at model load so that, for every (16-row block, K trip, register slot), the 64
   lanes' 16-byte pieces are contiguous in lane order: one wave instruction = one aligned
   1 KB run. Scales get the same treatment (each lane's C scale bytes contiguous, one dword
   per lane, 256 B per wave). Only 16-byte pieces move, never nibbles, so the decode, the
   K permutation (`bw_physical_k`) and the activation permute are unchanged. This is the
   same idea as the "preshuffled" weight layouts of CK/hipBLASLt and CUTLASS's
   interleaved layouts: pay a permutation once at load so the hot loop reads linearly.
   Memory: the permutation is applied in place, one tensor at a time; steady state costs 0
   extra bytes, the transient peak is one copy of the largest tensor (a layer's gate_up
   payload: 128 x 2048 x 2048 B = 512 MiB per TP=4 rank; scales 32 MiB). The shuffled
   tensors keep their keys and shapes and are only readable by this path; `bw_layout`
   (added to the experts dict) records the trip sizes they were shuffled for, and
   `fused_moe_bw2` refuses a mismatch.

3. Split-K for gate_up (`splitk`, weak point 3). At ~6 distinct local experts gate_up has
   6 x 32 = 192 items for 228 CUs (b48 p10 is 3 experts). With split S each item reduces
   K/S, so there are S x as many; partial [2*GU, BT] fp32 tiles go to a scratch buffer and
   the last split to arrive (per-tile atomic counter, acq_rel) sums the S partials in
   fixed index order, applies SiLU(gate) * up and stores `inter`. That is Stream-K's
   "fixup" (Osama et al., PPoPP 2023) restricted to a fixed split; the fixed summation
   order keeps the result bit-reproducible whichever split finishes last. Split-K is only
   applied up to `SPLITK_MAX_TOKENS` tokens (decode); prefill has items to spare.

4. One fused launch (`fused`). gate_up and down run in one persistent kernel whose programs
   claim work items from an atomic ticket: all gate_up items first, then all down items. A
   down item for unit u spins until unit u's `INTER // GU` gate_up tiles have published
   (per-unit ready counter). Deadlock-free without any co-residency assumption: a waiting
   down item only depends on items with smaller tickets, which were claimed by programs
   that are already running and never wait themselves (the ordering argument of decoupled
   look-back, Merrill & Garland 2016). Saves one kernel boundary (drain + ramp) and lets
   down items start on finished units while gate_up's tail is still running.

5. Lean decode (`lean`). v1's asm re-materializes nine constants per 32-value block (the
   four table bases, two byte selectors, the sign mask, two pair selectors) and builds the
   scale shift in two instructions. The lean asm takes the constants as loop-invariant
   register operands and forms `(s << 7) | (s << 23)` with one `v_mul_u32_u24`:
   66 instead of 76 VALU instructions per 32 values. Bit-identical output
   (seed test emulates both asm strings instruction by instruction).

Offline gfx942 ISA (Triton 3.4, TP=4 per-rank dims, BK=512; `vmcnt` = the waits inside the
K loop, as outstanding-load counts; v1 waits `vmcnt(0)` with nothing else in flight):

    kernel      VGPR  waves/SIMD  VALU/trip  v_mov/trip  K-loop vmcnt waits
    v1 gate_up   208      2          354         32      0 (at ~3/4 of every trip)
    v2a/v2b      252      2          326          5      13..10 per buffer, 8 at the latch
    v2c/v2f      256*     2        326/286        5      same as v2a
    v2d/v2g      256*     2        326/286        5      same; 39-43 spills, none in K loop
    v2e          369      1          332          7      0 once per 4-trip iteration
    (* with the variant's waves_per_eu=2; 260-297 and 1 wave/SIMD without it)

Two scheduling hazards had to be designed out, both visible only in the ISA: (a) the decode
asm is pure by default, so LLVM hoisted the next buffer's decode above the current buffer's
refill and waited `vmcnt(0)` with nothing in flight; it is side-effecting here
(`SEED_MOE_BW2_ORDERED=0` restores the pure version). (b) u8 scales carried across the loop
back-edge are byte-extracted right after their load, i.e. an immediate wait; they now run
their own ring one iteration ahead (`_v2_accumulate`).

Not implemented, by analysis:
- bf16 MFMA 32x32x8. The decode, not the matrix core, is the VALU/MFMA bottleneck: per
  4 KB trip a wave spends ~350 VALU (x4 clk = ~1400 clk) against 32 x 16 clk = 512 clk of
  16x16x16 MFMA, so MFMA already hides under VALU with 2 waves/SIMD. 32x32x8 needs 32 token
  columns; units average 8-10 live tokens at b48-b192 (routing traces), so half the
  columns would be padding and MFMA time per weight byte doubles (1024 clk per 16 rows),
  with no VALU saved. It would pay only for >= 32 tokens per expert (prefill).
- More warps per item. One warp owns 16 weight rows (the MFMA A-operand placement), so
  warps per program = rows per item / 16 (`SEED_MOE_BW_GU_ROWS`, `SEED_MOE_BW_DN_ROWS`);
  swept by `bench_moe_bw.py --sweep` together with `SEED_MOE_BW2_WAVES_PER_EU`.
"""

from __future__ import annotations

import dataclasses
import os

import mxfp4_gemv as mg
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False


@dataclasses.dataclass(frozen=True)
class Variant:
    """One selectable kernel configuration. `name == "v1"` means `mxfp4_gemv.fused_moe_bw`."""

    name: str
    depth: int = 2
    shuffle: bool = False
    splitk: int = 1
    fused: bool = False
    lean: bool = False
    waves_per_eu: int = 0
    """AMDGPU occupancy hint (0 = compiler default). 2 caps the split-K and fused kernels at
    256 VGPRs, i.e. 2 waves/SIMD, with no spill inside the K loop (offline ISA); without it
    they land at 260-297 VGPRs and 1 wave/SIMD. Depth 4 needs ~370 VGPRs and spills 95
    times inside the K loop under the hint, so it runs at 1 wave/SIMD instead."""


VARIANTS: dict[str, Variant] = {
    "v1": Variant("v1"),
    "v2a": Variant("v2a", depth=2),
    "v2b": Variant("v2b", depth=2, shuffle=True),
    "v2c": Variant("v2c", depth=2, shuffle=True, splitk=2, waves_per_eu=2),
    "v2d": Variant("v2d", depth=2, shuffle=True, splitk=2, fused=True, waves_per_eu=2),
    "v2e": Variant("v2e", depth=4, shuffle=True, splitk=2),
    "v2f": Variant("v2f", depth=2, shuffle=True, splitk=2, lean=True, waves_per_eu=2),
    "v2g": Variant("v2g", depth=2, shuffle=True, splitk=2, fused=True, lean=True, waves_per_eu=2),
}

SPLITK_MAX_TOKENS = int(os.environ.get("SEED_MOE_BW2_SPLITK_MAX_TOKENS", "256"))
"""Split-K for gate_up applies at or below this many tokens (decode batches)."""


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw in (None, "") else int(raw)


def selected_variant() -> Variant:
    """`SEED_MOE_BW_VARIANT` (default v1) plus per-field overrides
    (`SEED_MOE_BW2_DEPTH`, `_SPLITK`, `_FUSED`, `_LEAN`, `_SHUFFLE`, `_WAVES_PER_EU`).
    Unknown names fail."""
    name = os.environ.get("SEED_MOE_BW_VARIANT", "v1")
    if name not in VARIANTS:
        raise ValueError(f"SEED_MOE_BW_VARIANT={name!r}: expected one of {sorted(VARIANTS)}")
    v = VARIANTS[name]
    if v.name == "v1":
        return v
    over = {}
    for field, env in (
        ("depth", "SEED_MOE_BW2_DEPTH"),
        ("splitk", "SEED_MOE_BW2_SPLITK"),
        ("fused", "SEED_MOE_BW2_FUSED"),
        ("lean", "SEED_MOE_BW2_LEAN"),
        ("shuffle", "SEED_MOE_BW2_SHUFFLE"),
        ("waves_per_eu", "SEED_MOE_BW2_WAVES_PER_EU"),
    ):
        val = _env_int(env)
        if val is not None:
            over[field] = bool(val) if field in ("fused", "lean", "shuffle") else val
    v = dataclasses.replace(v, **over)
    if v.depth not in (2, 4) or v.splitk not in (1, 2, 4):
        raise ValueError(f"SEED_MOE_BW variant {v}: depth must be 2 or 4, splitk 1, 2 or 4")
    return v


# -- load-time weight reshuffle -------------------------------------------------------------


def bw_shuffle_packed(packed: torch.Tensor, bk: int) -> torch.Tensor:
    """`[E, N, K/2]` row-major MXFP4 bytes -> the same bytes with, per (16-row block, K trip
    of `bk`, 128-value slot c), the 64 lanes' 16-byte pieces contiguous in lane order
    `g * 16 + r` (lane group g owns physical K `[g*bk/4, (g+1)*bk/4)` of the trip, row r).
    Shape unchanged. Pure permutation; `bw_unshuffle_packed` inverts it."""
    e, n, kh = packed.shape
    c = bk // 128
    v = packed.view(e, n // 16, 16, (2 * kh) // bk, 4, c, 16)  # [E, RB, r, t, g, c, B]
    return v.permute(0, 1, 3, 5, 4, 2, 6).contiguous().view(e, n, kh)  # [E, RB, t, c, g, r, B]


def bw_unshuffle_packed(shuffled: torch.Tensor, bk: int) -> torch.Tensor:
    e, n, kh = shuffled.shape
    c = bk // 128
    v = shuffled.view(e, n // 16, (2 * kh) // bk, c, 4, 16, 16)  # [E, RB, t, c, g, r, B]
    return v.permute(0, 1, 5, 2, 4, 3, 6).contiguous().view(e, n, kh)


def bw_shuffle_scale(scale: torch.Tensor, bk: int) -> torch.Tensor:
    """`[E, N, K/32]` e8m0 -> per (16-row block, K trip), lane `g * 16 + r`'s C scale bytes
    contiguous (one dword per lane when C = 4)."""
    e, n, ks = scale.shape
    c = bk // 128
    v = scale.view(e, n // 16, 16, (32 * ks) // bk, 4, c)  # [E, RB, r, t, g, c]
    return v.permute(0, 1, 3, 4, 2, 5).contiguous().view(e, n, ks)  # [E, RB, t, g, r, c]


def bw_unshuffle_scale(shuffled: torch.Tensor, bk: int) -> torch.Tensor:
    e, n, ks = shuffled.shape
    c = bk // 128
    v = shuffled.view(e, n // 16, (32 * ks) // bk, 4, 16, c)  # [E, RB, t, g, r, c]
    return v.permute(0, 1, 4, 2, 3, 5).contiguous().view(e, n, ks)


def trip_sizes(reduction: int, intermediate: int) -> tuple[int, int]:
    """(gate_up, down) K-trip sizes the kernels use, hence the shuffle granularity."""
    return min(mg.BW_GU_BLOCK_K, reduction), min(mg.BW_DN_BLOCK_K, intermediate)


def bw_shuffle(experts: dict) -> None:
    """One-time, in place, at model load: rewrite the four MXFP4 tensors into the shuffled
    layout and record it as `experts["bw_layout"]`. Idempotent. One tensor at a time, so the
    transient peak is one copy of the largest tensor (see the module docstring)."""
    if "bw_layout" in experts:
        return
    gu_bk, dn_bk = trip_sizes(experts["gate_up"].shape[2] * 2, experts["down"].shape[2] * 2)
    for key, fn, bk in (
        ("gate_up", bw_shuffle_packed, gu_bk),
        ("gate_up_scale", bw_shuffle_scale, gu_bk),
        ("down", bw_shuffle_packed, dn_bk),
        ("down_scale", bw_shuffle_scale, dn_bk),
    ):
        experts[key].copy_(fn(experts[key], bk))
    experts["bw_layout"] = torch.tensor([gu_bk, dn_bk], dtype=torch.int64)


def bw_shuffle_bytes(experts: dict) -> tuple[int, int]:
    """(steady-state extra bytes, transient peak bytes) of `bw_shuffle` for one layer."""
    return 0, max(t.numel() * t.element_size() for k, t in experts.items() if k != "bw_layout")


# -- kernels --------------------------------------------------------------------------------


def _lean_asm() -> str:
    """`mxfp4_gemv._bw_asm` with its nine per-block constants taken as register operands and
    the scale shift in one multiply. Same outputs, bit for bit (emulated in the seed test).

    Operands (pack=4): $0-$15 outputs P0..P3, $16-$31 scratch, $32-$35 words, $36 scale,
    $40 $44 $48 $52 table bases, $56/$60 byte-split selectors, $64 sign mask, $68/$72 pair
    selectors (only the first register of each packed group is read)."""
    q = ["$16", "$17", "$18", "$19"]
    tlo0, tlo1, thi0, thi1 = "$20", "$21", "$22", "$23"
    t, a, b, ha, hb, la, lb = "$24", "$25", "$26", "$27", "$28", "$29", "$30"
    words = ["$32", "$33", "$34", "$35"]
    scale = "$36"
    qb = ["$40", "$44", "$48", "$52"]
    c_lo, c_hi, m80, s01, s23 = "$56", "$60", "$64", "$68", "$72"
    lines = [
        f"v_mul_u32_u24 {t}, 0x800080, {scale}",  # (s << 7) | (s << 23), s <= 255
        f"v_pk_add_u16 {q[0]}, {qb[0]}, {t}",
        f"v_pk_add_u16 {q[1]}, {qb[1]}, {t}",
        f"v_pk_add_u16 {q[2]}, {qb[2]}, {t}",
        f"v_pk_add_u16 {q[3]}, {qb[3]}, {t}",
        f"v_and_b32 {q[0]}, 0xffff0000, {q[0]}",
        f"v_perm_b32 {tlo0}, {q[1]}, {q[0]}, {c_lo}",
        f"v_perm_b32 {tlo1}, {q[3]}, {q[2]}, {c_lo}",
        f"v_perm_b32 {thi0}, {q[1]}, {q[0]}, {c_hi}",
        f"v_perm_b32 {thi1}, {q[3]}, {q[2]}, {c_hi}",
    ]
    for i, w in enumerate(words):
        p0, p1, p2, p3 = f"${i}", f"${4 + i}", f"${8 + i}", f"${12 + i}"
        lines += [
            f"v_and_b32 {a}, 0x07070707, {w}",
            f"v_lshrrev_b32 {b}, 4, {w}",
            f"v_and_b32 {b}, 0x07070707, {b}",
            f"v_perm_b32 {ha}, {thi1}, {thi0}, {a}",
            f"v_perm_b32 {hb}, {thi1}, {thi0}, {b}",
            f"v_perm_b32 {la}, {tlo1}, {tlo0}, {a}",
            f"v_perm_b32 {lb}, {tlo1}, {tlo0}, {b}",
            f"v_lshlrev_b32 {a}, 4, {w}",
            f"v_and_or_b32 {ha}, {a}, {m80}, {ha}",
            f"v_and_or_b32 {hb}, {w}, {m80}, {hb}",
            f"v_perm_b32 {p0}, {ha}, {la}, {s01}",
            f"v_perm_b32 {p1}, {ha}, {la}, {s23}",
            f"v_perm_b32 {p2}, {hb}, {lb}, {s01}",
            f"v_perm_b32 {p3}, {hb}, {lb}, {s23}",
        ]
    return "\n".join(lines)


LEAN_ASM = _lean_asm()
LEAN_ASM_CONSTRAINTS = ",".join(["=&v"] * 32 + ["v"] * 44)
LEAN_CONSTANTS = (
    0xFF800000,  # table bases: bf16 bits of the fp4 magnitudes minus 0x3f80, as u16 pairs
    0x00400000,
    0x00C00080,
    0x01400100,
    0x06040200,  # low-byte split selector
    0x07050301,  # high-byte split selector
    0x80808080,  # sign mask
    0x05010400,  # pair selectors
    0x07030602,
)

if HAVE_TRITON:
    from mxfp4_gemv import (
        _BW_ASM,
        _BW_CONSTRAINTS,
        _BW_INTERPRET,
        _bw_decode_tile,
        _bw_dot,
        _bw_fence,
        _bw_hi,
        _bw_lo,
        _bw_prep_kernel,
    )

    _UNROLL = tl.constexpr(os.environ.get("SEED_MOE_BW2_UNROLL", "0") == "1")
    _ORDERED = tl.constexpr(os.environ.get("SEED_MOE_BW2_ORDERED", "1") == "1")
    _LEAN_ASM = tl.constexpr(LEAN_ASM)
    _LEAN_CONSTRAINTS = tl.constexpr(LEAN_ASM_CONSTRAINTS)
    _K0 = tl.constexpr(LEAN_CONSTANTS[0] - (1 << 32))  # as signed int32
    _K1 = tl.constexpr(LEAN_CONSTANTS[1])
    _K2 = tl.constexpr(LEAN_CONSTANTS[2])
    _K3 = tl.constexpr(LEAN_CONSTANTS[3])
    _K4 = tl.constexpr(LEAN_CONSTANTS[4])
    _K5 = tl.constexpr(LEAN_CONSTANTS[5])
    _K6 = tl.constexpr(LEAN_CONSTANTS[6] - (1 << 32))
    _K7 = tl.constexpr(LEAN_CONSTANTS[7])
    _K8 = tl.constexpr(LEAN_CONSTANTS[8])

    @triton.jit
    def _v2_mem_release():
        """Every wave: make this wave's global stores visible device-wide (write back L2 for
        the other XCDs) before the workgroup barrier that precedes a publishing atomic."""
        if not _BW_INTERPRET:
            tl.inline_asm_elementwise(
                "buffer_wbl2 sc1\ns_waitcnt vmcnt(0)",
                "=v,v,~{memory}",
                [tl.zeros([1], tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.debug_barrier()

    @triton.jit
    def _v2_mem_acquire():
        """Every wave, after observing a published counter: drop stale L2/L1 lines."""
        tl.debug_barrier()
        if not _BW_INTERPRET:
            tl.inline_asm_elementwise(
                "buffer_inv sc1",
                "=v,v,~{memory}",
                [tl.zeros([1], tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

    @triton.jit
    def _v2_rows(row0, R: tl.constexpr, GATE_UP: tl.constexpr, INTER: tl.constexpr):
        """Index tensors of the `[16, 4, W, C, 4]` tile and each element's 16-row block."""
        W: tl.constexpr = R // 16
        r = tl.arange(0, 16)[:, None, None, None, None]
        g = tl.arange(0, 4)[None, :, None, None, None]
        w = tl.arange(0, W)[None, None, :, None, None]
        rb = w
        if GATE_UP:
            rb = rb + (w >= W // 2).to(tl.int32) * ((INTER - R // 2) // 16)
        return r, g, rb

    @triton.jit
    def _v2_load_words(
        wq_ptr, row0, k0, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr,
        GATE_UP: tl.constexpr, INTER: tl.constexpr, SHUF: tl.constexpr,
    ):  # fmt: skip
        """`_bw_load_tile`'s words `[16, 4, W, C, 4]` (int32), from the row-major or the
        shuffled layout. Shuffled: (16-row block, trip) is `8 * BK` bytes at
        `rb * K/BK + t`, slot c is the 1 KB run at `c * 1024`, lane `g * 16 + r` at
        `16 * lane` (`bw_shuffle_packed`), so one wave instruction reads 1 KB contiguous."""
        C: tl.constexpr = BK // 128
        r, g, rb = _v2_rows(row0, R, GATE_UP, INTER)
        c = tl.arange(0, C)[None, None, None, :, None]
        j = tl.arange(0, 4)[None, None, None, None, :]
        wq = wq_ptr.to(tl.pointer_type(tl.int32))
        if SHUF:
            blk = (row0 // 16 + rb.to(tl.int64)) * (K // BK) + k0 // BK
            return tl.load(wq + blk * (2 * BK) + (c * 256 + (g * 16 + r) * 4 + j))
        rows = row0 + (rb * 16 + r).to(tl.int64)
        return tl.load(wq + rows * (K // 8) + (k0 // 8 + g * (BK // 32) + c * 4 + j))

    @triton.jit
    def _v2_load_scale(
        ws_ptr, row0, k0, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr,
        GATE_UP: tl.constexpr, INTER: tl.constexpr, SHUF: tl.constexpr,
    ):  # fmt: skip
        """The tile's e8m0 scales `[16, 4, W, C, 1]` (u8); a lane's C bytes are contiguous in
        both layouts (one dword per lane at C = 4; 256 B contiguous per wave when shuffled)."""
        C: tl.constexpr = BK // 128
        r, g, rb = _v2_rows(row0, R, GATE_UP, INTER)
        c = tl.arange(0, C)[None, None, None, :, None]
        if SHUF:
            blk = (row0 // 16 + rb.to(tl.int64)) * (K // BK) + k0 // BK
            return tl.load(ws_ptr + blk * (BK // 2) + ((g * 16 + r) * C + c))
        rows = row0 + (rb * 16 + r).to(tl.int64)
        return tl.load(ws_ptr + rows * (K // 32) + (k0 // 32 + g * C + c))

    @triton.jit
    def _v2_decode_tile(
        words, sc, R: tl.constexpr, BK: tl.constexpr, PERM: tl.constexpr, LEAN: tl.constexpr
    ):
        """`_bw_decode_tile` with the v1 or the lean asm. With `ORDERED` (default) the asm is
        side-effecting, so LLVM keeps each decode on its side of the loads and fences around
        it: without that, the scheduler hoists the next buffer's decode above the current
        buffer's refill loads, and the wait for the next buffer then happens with nothing
        else in flight (seen in the offline ISA: `vmcnt(0)` before the refill is issued)."""
        C: tl.constexpr = BK // 128
        W: tl.constexpr = R // 16
        shape: tl.constexpr = (16, 4, W, C, 4)
        if not PERM:
            return _bw_decode_tile(words, sc, R, BK, PERM)
        sc = tl.broadcast_to(sc.to(tl.int32), shape)
        if LEAN:
            k0 = tl.full(shape, _K0, tl.int32)
            k1 = tl.full(shape, _K1, tl.int32)
            k2 = tl.full(shape, _K2, tl.int32)
            k3 = tl.full(shape, _K3, tl.int32)
            k4 = tl.full(shape, _K4, tl.int32)
            k5 = tl.full(shape, _K5, tl.int32)
            k6 = tl.full(shape, _K6, tl.int32)
            k7 = tl.full(shape, _K7, tl.int32)
            k8 = tl.full(shape, _K8, tl.int32)
            p0, p1, p2, p3, _s0, _s1, _s2, _s3 = tl.inline_asm_elementwise(
                _LEAN_ASM,
                _LEAN_CONSTRAINTS,
                [words, sc, k0, k1, k2, k3, k4, k5, k6, k7, k8],
                dtype=(tl.int32,) * 8,
                is_pure=not _ORDERED,
                pack=4,
            )
        else:
            p0, p1, p2, p3, _s0, _s1, _s2, _s3, _s4 = tl.inline_asm_elementwise(
                _BW_ASM,
                _BW_CONSTRAINTS,
                [words, sc],
                dtype=(tl.int32,) * 9,
                is_pure=not _ORDERED,
                pack=4,
            )
        lo = tl.join(tl.join(_bw_lo(p0), _bw_lo(p2)), tl.join(_bw_lo(p1), _bw_lo(p3)))
        hi = tl.join(tl.join(_bw_hi(p0), _bw_hi(p2)), tl.join(_bw_hi(p1), _bw_hi(p3)))
        v = tl.reshape(tl.join(lo, hi), (16, 4, W, C, 4, 2, 4))
        v = tl.permute(v, (2, 0, 3, 4, 5, 1, 6))
        return tl.reshape(v, (R, BK))

    @triton.jit
    def _v2_step(
        acc, a, w, s, arow, wq_ptr, row0, knext,
        K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr, GATE_UP: tl.constexpr,
        INTER: tl.constexpr, SHUF: tl.constexpr, PERM: tl.constexpr, LEAN: tl.constexpr,
        FENCE: tl.constexpr,
    ):  # fmt: skip
        """Decode + multiply one buffer, then refill that same buffer's activation and words
        with trip `knext` (the scales run their own ring, see `_v2_accumulate`)."""
        acc = _bw_dot(_v2_decode_tile(w, s, R, BK, PERM, LEAN), a, acc)
        _bw_fence(wq_ptr, FENCE)
        a = tl.load(arow + knext)
        w = _v2_load_words(wq_ptr, row0, knext, K, R, BK, GATE_UP, INTER, SHUF)
        _bw_fence(wq_ptr, FENCE)
        return acc, a, w

    @triton.jit
    def _v2_scale_at(
        ws_ptr, row0, k, klast, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr,
        GATE_UP: tl.constexpr, INTER: tl.constexpr, SHUF: tl.constexpr,
    ):  # fmt: skip
        """Scales of trip `k`, clamped to the item's last trip (the ring runs past the end;
        the clamped re-read is an L2 hit and keeps every address in bounds)."""
        return _v2_load_scale(ws_ptr, row0, tl.minimum(k, klast), K, R, BK, GATE_UP, INTER, SHUF)

    @triton.jit
    def _v2_accumulate(
        acc, arow, wq_ptr, ws_ptr, row0, kb, k_loop,
        NTRIP: tl.constexpr, K: tl.constexpr, R: tl.constexpr, BK: tl.constexpr,
        GATE_UP: tl.constexpr, INTER: tl.constexpr, SHUF: tl.constexpr,
        PERM: tl.constexpr, LEAN: tl.constexpr, FENCE: tl.constexpr, DEPTH: tl.constexpr,
    ):  # fmt: skip
        """acc += W[rows, kb : kb + NTRIP*BK] . A over NTRIP trips, DEPTH rotating register
        buffers for words + activations, and a 2*DEPTH scale ring loaded one whole iteration
        ahead. `k_loop` = (NTRIP - DEPTH) * BK is a runtime value so LLVM cannot unroll the
        loop into straight-line code, where the activation's LDS round trip is hoisted next
        to its load and every trip waits (offline ISA; `SEED_MOE_BW2_UNROLL=1` shows it).

        Why the scales get their own ring: Triton carries a u8 tensor across the back-edge
        as separate bytes, so LLVM extracts them in the block that loads them, i.e. right
        after the load, and in-order vmcnt makes that a wait for the load and everything
        issued before it. Loading a trip's scales one iteration early, first in the step,
        turns that wait into one for loads that are an iteration old."""
        klast = kb + (NTRIP - 1) * BK
        _bw_fence(wq_ptr, FENCE)
        s0 = _v2_scale_at(ws_ptr, row0, kb, klast, K, R, BK, GATE_UP, INTER, SHUF)
        s1 = _v2_scale_at(ws_ptr, row0, kb + BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
        a0 = tl.load(arow + kb)
        w0 = _v2_load_words(wq_ptr, row0, kb, K, R, BK, GATE_UP, INTER, SHUF)
        _bw_fence(wq_ptr, FENCE)
        a1 = tl.load(arow + kb + BK)
        w1 = _v2_load_words(wq_ptr, row0, kb + BK, K, R, BK, GATE_UP, INTER, SHUF)
        _bw_fence(wq_ptr, FENCE)
        if DEPTH == 2:
            s2 = _v2_scale_at(ws_ptr, row0, kb + 2 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            s3 = _v2_scale_at(ws_ptr, row0, kb + 3 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            _bw_fence(wq_ptr, FENCE)
            if _BW_INTERPRET or _UNROLL:
                for k in tl.static_range(0, (NTRIP - 2) * BK, 2 * BK):
                    n4 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 4 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n5 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 5 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    acc, a0, w0 = _v2_step(
                        acc,
                        a0,
                        w0,
                        s0,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 2 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a1, w1 = _v2_step(
                        acc,
                        a1,
                        w1,
                        s1,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 3 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    s0, s1, s2, s3 = s2, s3, n4, n5
            else:
                for k in range(0, k_loop, 2 * BK):
                    n4 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 4 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n5 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 5 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    acc, a0, w0 = _v2_step(
                        acc,
                        a0,
                        w0,
                        s0,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 2 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a1, w1 = _v2_step(
                        acc,
                        a1,
                        w1,
                        s1,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 3 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    s0, s1, s2, s3 = s2, s3, n4, n5
            acc = _bw_dot(_v2_decode_tile(w0, s0, R, BK, PERM, LEAN), a0, acc)
            acc = _bw_dot(_v2_decode_tile(w1, s1, R, BK, PERM, LEAN), a1, acc)
        else:
            s2 = _v2_scale_at(ws_ptr, row0, kb + 2 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            s3 = _v2_scale_at(ws_ptr, row0, kb + 3 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            a2 = tl.load(arow + kb + 2 * BK)
            w2 = _v2_load_words(wq_ptr, row0, kb + 2 * BK, K, R, BK, GATE_UP, INTER, SHUF)
            _bw_fence(wq_ptr, FENCE)
            a3 = tl.load(arow + kb + 3 * BK)
            w3 = _v2_load_words(wq_ptr, row0, kb + 3 * BK, K, R, BK, GATE_UP, INTER, SHUF)
            _bw_fence(wq_ptr, FENCE)
            s4 = _v2_scale_at(ws_ptr, row0, kb + 4 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            s5 = _v2_scale_at(ws_ptr, row0, kb + 5 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            s6 = _v2_scale_at(ws_ptr, row0, kb + 6 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            s7 = _v2_scale_at(ws_ptr, row0, kb + 7 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF)
            _bw_fence(wq_ptr, FENCE)
            if _BW_INTERPRET or _UNROLL:
                for k in tl.static_range(0, (NTRIP - 4) * BK, 4 * BK):
                    n8 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 8 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n9 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 9 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n10 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 10 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n11 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 11 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    acc, a0, w0 = _v2_step(
                        acc,
                        a0,
                        w0,
                        s0,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 4 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a1, w1 = _v2_step(
                        acc,
                        a1,
                        w1,
                        s1,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 5 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a2, w2 = _v2_step(
                        acc,
                        a2,
                        w2,
                        s2,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 6 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a3, w3 = _v2_step(
                        acc,
                        a3,
                        w3,
                        s3,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 7 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    s0, s1, s2, s3, s4, s5, s6, s7 = s4, s5, s6, s7, n8, n9, n10, n11
            else:
                for k in range(0, k_loop, 4 * BK):
                    n8 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 8 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n9 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 9 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n10 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 10 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    n11 = _v2_scale_at(
                        ws_ptr, row0, kb + k + 11 * BK, klast, K, R, BK, GATE_UP, INTER, SHUF
                    )
                    acc, a0, w0 = _v2_step(
                        acc,
                        a0,
                        w0,
                        s0,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 4 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a1, w1 = _v2_step(
                        acc,
                        a1,
                        w1,
                        s1,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 5 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a2, w2 = _v2_step(
                        acc,
                        a2,
                        w2,
                        s2,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 6 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    acc, a3, w3 = _v2_step(
                        acc,
                        a3,
                        w3,
                        s3,
                        arow,
                        wq_ptr,
                        row0,
                        kb + k + 7 * BK,
                        K,
                        R,
                        BK,
                        GATE_UP,
                        INTER,
                        SHUF,
                        PERM,
                        LEAN,
                        FENCE,
                    )
                    s0, s1, s2, s3, s4, s5, s6, s7 = s4, s5, s6, s7, n8, n9, n10, n11
            acc = _bw_dot(_v2_decode_tile(w0, s0, R, BK, PERM, LEAN), a0, acc)
            acc = _bw_dot(_v2_decode_tile(w1, s1, R, BK, PERM, LEAN), a1, acc)
            acc = _bw_dot(_v2_decode_tile(w2, s2, R, BK, PERM, LEAN), a2, acc)
            acc = _bw_dot(_v2_decode_tile(w3, s3, R, BK, PERM, LEAN), a3, acc)
        return acc

    @triton.jit
    def _v2_gate_up_store(
        acc, n0, slot, valid, inter_ptr, INTER: tl.constexpr, GU: tl.constexpr,
        BT: tl.constexpr, DBK: tl.constexpr,
    ):  # fmt: skip
        """SiLU(gate) * up, stored in the down kernel's logical K order (as v1)."""
        g, v = tl.split(tl.permute(tl.reshape(acc, (2, GU, BT)), (1, 2, 0)))
        act = g / (1.0 + tl.exp(-g)) * v
        n = n0 + tl.arange(0, GU)
        nn = n % DBK
        gg = nn // (DBK // 4)
        cc = (nn % (DBK // 4)) // 32
        q = (n - nn) + 128 * cc + 32 * ((nn % 32) // 8) + 16 * (nn % 2) + 4 * gg + (nn % 8) // 2
        tl.store(
            inter_ptr + slot[None, :].to(tl.int64) * INTER + q[:, None],
            act.to(inter_ptr.dtype.element_ty),
            mask=valid[None, :],
        )

    @triton.jit
    def _v2_gate_up_item(
        item,
        xp_ptr,
        wq_ptr,
        ws_ptr,
        sorted_ptr,
        unit_ptr,
        inter_ptr,
        part_ptr,
        cnt_ptr,
        ready_ptr,
        k_loop,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BT: tl.constexpr,
        GU: tl.constexpr,
        BK: tl.constexpr,
        DBK: tl.constexpr,
        S: tl.constexpr,
        DEPTH: tl.constexpr,
        SHUF: tl.constexpr,
        PERM: tl.constexpr,
        LEAN: tl.constexpr,
        FENCE: tl.constexpr,
        READY: tl.constexpr,
    ):
        """Work item = (unit, n-tile, K split), split fastest. The last split of a tile to
        publish sums the S partials in index order and stores `inter`; with READY it then
        bumps the unit's ready counter (fused kernel)."""
        NT: tl.constexpr = INTER // GU
        R: tl.constexpr = 2 * GU
        KS: tl.constexpr = K // S
        tt = tl.arange(0, BT)
        kk = tl.arange(0, BK)
        s = item % S
        tile = item // S
        u = tile // NT
        n0 = (tile % NT) * GU
        expert = tl.load(unit_ptr + 3 * u)
        p0 = tl.load(unit_ptr + 3 * u + 1)
        cnt = tl.load(unit_ptr + 3 * u + 2)
        valid = tt < cnt
        slot = tl.load(sorted_ptr + p0 + tt, mask=valid, other=0)
        tok = (slot // TOP_K).to(tl.int64)
        row0 = expert.to(tl.int64) * (2 * INTER) + n0
        xrow = xp_ptr + tok[:, None] * K + kk[None, :]
        acc = tl.zeros([R, BT], tl.float32)
        acc = _v2_accumulate(
            acc, xrow, wq_ptr, ws_ptr, row0, s * KS, k_loop, KS // BK, K, R, BK, True, INTER,
            SHUF, PERM, LEAN, FENCE, DEPTH,
        )  # fmt: skip
        if S == 1:
            _v2_gate_up_store(acc, n0, slot, valid, inter_ptr, INTER, GU, BT, DBK)
            if READY:
                _v2_mem_release()
                tl.atomic_add(ready_ptr + u, 1, sem="release", scope="gpu")
        else:
            offs = tl.arange(0, R)[:, None] * BT + tt[None, :]
            base = part_ptr + tile.to(tl.int64) * (S * R * BT)
            tl.store(base + s * (R * BT) + offs, acc)
            _v2_mem_release()
            done = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
            if done == S - 1:
                _v2_mem_acquire()
                tot = tl.load(base + offs, cache_modifier=".cv")
                for s2 in tl.static_range(1, S):
                    tot += tl.load(base + s2 * (R * BT) + offs, cache_modifier=".cv")
                _v2_gate_up_store(tot, n0, slot, valid, inter_ptr, INTER, GU, BT, DBK)
                tl.atomic_xchg(cnt_ptr + tile, 0)  # self-cleaning; prep also zeroes
                if READY:
                    _v2_mem_release()
                    tl.atomic_add(ready_ptr + u, 1, sem="release", scope="gpu")

    @triton.jit
    def _v2_down_item(
        item,
        inter_ptr,
        wq_ptr,
        ws_ptr,
        a_weight_ptr,
        sorted_ptr,
        unit_ptr,
        y_ptr,
        k_loop,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BT: tl.constexpr,
        DN: tl.constexpr,
        BK: tl.constexpr,
        DEPTH: tl.constexpr,
        SHUF: tl.constexpr,
        PERM: tl.constexpr,
        LEAN: tl.constexpr,
        FENCE: tl.constexpr,
    ):
        NT: tl.constexpr = HIDDEN // DN
        tt = tl.arange(0, BT)
        kk = tl.arange(0, BK)
        dn = tl.arange(0, DN)
        u = item // NT
        n0 = (item % NT) * DN
        expert = tl.load(unit_ptr + 3 * u)
        p0 = tl.load(unit_ptr + 3 * u + 1)
        cnt = tl.load(unit_ptr + 3 * u + 2)
        valid = tt < cnt
        slot = tl.load(sorted_ptr + p0 + tt, mask=valid, other=0).to(tl.int64)
        row0 = expert.to(tl.int64) * HIDDEN + n0
        arow = inter_ptr + slot[:, None] * K + kk[None, :]
        acc = tl.zeros([DN, BT], tl.float32)
        acc = _v2_accumulate(
            acc, arow, wq_ptr, ws_ptr, row0, 0, k_loop, K // BK, K, DN, BK, False, 0,
            SHUF, PERM, LEAN, FENCE, DEPTH,
        )  # fmt: skip
        cw = tl.load(a_weight_ptr + slot, mask=valid, other=0.0).to(tl.float32)
        tl.store(
            y_ptr + slot[None, :] * HIDDEN + (n0 + dn)[:, None],
            (acc * cw[None, :]).to(y_ptr.dtype.element_ty),
            mask=valid[None, :],
        )

    @triton.jit
    def _v2_prep_kernel(
        a_expert_ptr,
        a_weight_ptr,
        cursor_ptr,
        sorted_ptr,
        unit_ptr,
        n_units_ptr,
        x_ptr,
        xp_ptr,
        sync_ptr,
        lo,
        hi,
        K: tl.constexpr,
        BK: tl.constexpr,
        ASSIGN_PAD: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        NLOCAL: tl.constexpr,
        BUCKET_PAD: tl.constexpr,
        BT: tl.constexpr,
        MAX_UPE: tl.constexpr,
        U_MAX: tl.constexpr,
        NSYNC: tl.constexpr,
    ):
        """v1's prep, plus program 0 zeroing the split-K/fused sync words (`sync_ptr`)."""
        _bw_prep_kernel(
            a_expert_ptr, a_weight_ptr, cursor_ptr, sorted_ptr, unit_ptr, n_units_ptr, x_ptr,
            xp_ptr, lo, hi, K, BK, ASSIGN_PAD, ASSIGNMENTS, NLOCAL, BUCKET_PAD, BT, MAX_UPE,
            U_MAX,
        )  # fmt: skip
        if tl.program_id(0) == 0:
            zb = tl.arange(0, 1024)
            for i in range(0, NSYNC, 1024):
                tl.store(sync_ptr + i + zb, 0, mask=i + zb < NSYNC)

    @triton.jit
    def _v2_gate_up_kernel(
        xp_ptr, wq_ptr, ws_ptr, sorted_ptr, unit_ptr, n_units_ptr, inter_ptr, part_ptr,
        cnt_ptr, k_loop,
        K: tl.constexpr, INTER: tl.constexpr, TOP_K: tl.constexpr, BT: tl.constexpr,
        GU: tl.constexpr, BK: tl.constexpr, DBK: tl.constexpr, S: tl.constexpr,
        GRID: tl.constexpr, ITERS: tl.constexpr, DEPTH: tl.constexpr, SHUF: tl.constexpr,
        PERM: tl.constexpr, LEAN: tl.constexpr, FENCE: tl.constexpr,
    ):  # fmt: skip
        n_items = tl.load(n_units_ptr) * (INTER // GU) * S
        for it in range(ITERS):
            item = tl.program_id(0) + it * GRID
            if item < n_items:
                _v2_gate_up_item(
                    item, xp_ptr, wq_ptr, ws_ptr, sorted_ptr, unit_ptr, inter_ptr, part_ptr,
                    cnt_ptr, cnt_ptr, k_loop, K, INTER, TOP_K, BT, GU, BK, DBK, S, DEPTH,
                    SHUF, PERM, LEAN, FENCE, False,
                )  # fmt: skip

    @triton.jit
    def _v2_down_kernel(
        inter_ptr, wq_ptr, ws_ptr, a_weight_ptr, sorted_ptr, unit_ptr, n_units_ptr, y_ptr,
        k_loop,
        K: tl.constexpr, HIDDEN: tl.constexpr, BT: tl.constexpr, DN: tl.constexpr,
        BK: tl.constexpr, GRID: tl.constexpr, ITERS: tl.constexpr, DEPTH: tl.constexpr,
        SHUF: tl.constexpr, PERM: tl.constexpr, LEAN: tl.constexpr, FENCE: tl.constexpr,
    ):  # fmt: skip
        n_items = tl.load(n_units_ptr) * (HIDDEN // DN)
        for it in range(ITERS):
            item = tl.program_id(0) + it * GRID
            if item < n_items:
                _v2_down_item(
                    item, inter_ptr, wq_ptr, ws_ptr, a_weight_ptr, sorted_ptr, unit_ptr,
                    y_ptr, k_loop, K, HIDDEN, BT, DN, BK, DEPTH, SHUF, PERM, LEAN, FENCE,
                )  # fmt: skip

    @triton.jit
    def _v2_fused_kernel(
        xp_ptr, gu_q_ptr, gu_s_ptr, dn_q_ptr, dn_s_ptr, a_weight_ptr, sorted_ptr, unit_ptr,
        n_units_ptr, inter_ptr, part_ptr, sync_ptr, y_ptr, gu_k_loop, dn_k_loop,
        K: tl.constexpr, INTER: tl.constexpr, HIDDEN: tl.constexpr, TOP_K: tl.constexpr,
        BT: tl.constexpr, GU: tl.constexpr, DN: tl.constexpr, BK: tl.constexpr,
        DBK: tl.constexpr, S: tl.constexpr, U_MAX: tl.constexpr, DEPTH: tl.constexpr,
        SHUF: tl.constexpr, PERM: tl.constexpr, LEAN: tl.constexpr, FENCE: tl.constexpr,
    ):  # fmt: skip
        """gate_up and down in one launch. `sync` = [ticket, ready[U_MAX], cnt[...]], zeroed
        by the prep kernel. Items are claimed in ticket order, gate_up items first, so a
        down item only ever waits on items already claimed by running programs."""
        ready_ptr = sync_ptr + 1
        cnt_ptr = sync_ptr + 1 + U_MAX
        n_units = tl.load(n_units_ptr)
        n_gu = n_units * (INTER // GU) * S
        n_all = n_gu + n_units * (HIDDEN // DN)
        item = tl.atomic_add(sync_ptr, 1, sem="relaxed", scope="gpu")
        while item < n_all:
            if item < n_gu:
                _v2_gate_up_item(
                    item, xp_ptr, gu_q_ptr, gu_s_ptr, sorted_ptr, unit_ptr, inter_ptr,
                    part_ptr, cnt_ptr, ready_ptr, gu_k_loop, K, INTER, TOP_K, BT, GU, BK, DBK,
                    S, DEPTH, SHUF, PERM, LEAN, FENCE, True,
                )  # fmt: skip
            else:
                d = item - n_gu
                u = d // (HIDDEN // DN)
                seen = tl.atomic_add(ready_ptr + u, 0, sem="acquire", scope="gpu")
                while seen < INTER // GU:
                    seen = tl.atomic_add(ready_ptr + u, 0, sem="acquire", scope="gpu")
                _v2_mem_acquire()
                _v2_down_item(
                    d, inter_ptr, dn_q_ptr, dn_s_ptr, a_weight_ptr, sorted_ptr, unit_ptr, y_ptr,
                    dn_k_loop, INTER, HIDDEN, BT, DN, DBK, DEPTH, SHUF, PERM, LEAN, FENCE,
                )  # fmt: skip
            item = tl.atomic_add(sync_ptr, 1, sem="relaxed", scope="gpu")


# -- host side ------------------------------------------------------------------------------


class Bw2Plan(mg.BwPlan):
    """v1's scratch plus the split-K partials and the sync words
    `[ticket, ready[u_max], cnt[u_max * INTER / GU]]` (int32, zeroed by the prep kernel)."""

    def __init__(self, tokens, top_k, reduction, intermediate, local_experts, device, dtype, split):
        super().__init__(tokens, top_k, reduction, local_experts, device, dtype)
        tiles = self.u_max * (intermediate // mg.BW_GU_ROWS)
        self.split = split
        self.sync = torch.empty(1 + self.u_max + tiles, dtype=torch.int32, device=device)
        rows = 2 * mg.BW_GU_ROWS
        n_part = tiles * split * rows * mg.BW_BLOCK_T if split > 1 else 1
        self.part = torch.empty(n_part, dtype=torch.float32, device=device)

    @property
    def cnt(self) -> torch.Tensor:
        return self.sync[1 + self.u_max :]


def _launch_opts(num_warps: int, variant: Variant) -> dict:
    opts = {"num_warps": num_warps, "num_stages": mg.BW_STAGES}
    if variant.waves_per_eu > 0 and torch.version.hip is not None:
        opts["waves_per_eu"] = variant.waves_per_eu
    return opts


def _check_depth(ntrip: int, depth: int, what: str) -> int:
    """The rotation needs `depth` to divide the trip count; a 2-trip projection runs depth 2."""
    depth = min(depth, ntrip)
    if ntrip < 2 or ntrip % depth:
        raise ValueError(f"fused_moe_bw2: {what} has {ntrip} K trips, not a multiple of {depth}")
    return depth


def fused_moe_bw2(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    variant: Variant,
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """`mxfp4_gemv.fused_moe_bw`'s contract (static shapes, no host sync, capturable) for a
    v2 `variant`. With `variant.shuffle` the experts must have been through `bw_shuffle`
    (checked via `experts["bw_layout"]`, a CPU tensor, so no sync)."""
    if variant.name == "v1":
        raise ValueError("fused_moe_bw2 runs v2 variants; v1 is mxfp4_gemv.fused_moe_bw")
    a_expert, a_weight = routing
    lo, hi = expert_range
    tokens, reduction = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    gu_bk, dn_bk = trip_sizes(reduction, intermediate)
    shuffled = "bw_layout" in experts
    if shuffled != variant.shuffle or (
        shuffled and tuple(experts["bw_layout"].tolist()) != (gu_bk, dn_bk)
    ):
        raise ValueError(
            f"fused_moe_bw2 {variant.name}: expert layout "
            f"{experts.get('bw_layout')} does not match shuffle={variant.shuffle} "
            f"trips=({gu_bk}, {dn_bk}); run bw_shuffle at load time"
        )
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(f"routing must be two [{assignments}] tensors")
    if (
        intermediate % mg.BW_GU_ROWS
        or hidden % mg.BW_DN_ROWS
        or reduction % gu_bk
        or intermediate % dn_bk
        or gu_bk % 128
        or dn_bk % 128
    ):
        raise ValueError("fused_moe_bw2: tile sizes must divide the projection shapes")
    split = variant.splitk if tokens <= SPLITK_MAX_TOKENS else 1
    while split > 1 and (reduction // gu_bk) % split:
        split //= 2
    gu_trips = reduction // gu_bk // split
    gu_depth = _check_depth(gu_trips, variant.depth, "gate_up")
    dn_depth = _check_depth(intermediate // dn_bk, variant.depth, "down")
    if variant.fused and (gu_depth != dn_depth or 2 * mg.BW_GU_ROWS != mg.BW_DN_ROWS):
        raise ValueError("fused variant needs equal depth and 2 * GU_ROWS == DN_ROWS")
    device = x.device
    if inter is None or inter.shape != (assignments, intermediate):
        inter = torch.empty(assignments, intermediate, dtype=x.dtype, device=device)
    if y is None or y.shape != (assignments, hidden):
        y = torch.empty(assignments, hidden, dtype=x.dtype, device=device)
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=device)
    perm = mg.perm_lut(device)
    plan = Bw2Plan(tokens, top_k, reduction, intermediate, hi - lo, device, x.dtype, split)
    _v2_prep_kernel[(1 + tokens,)](
        a_expert,
        a_weight,
        plan.cursor,
        plan.sorted,
        plan.unit,
        plan.n_units,
        x,
        plan.xp,
        plan.sync,
        lo,
        hi,
        K=reduction,
        BK=gu_bk,
        ASSIGN_PAD=max(mg._next_pow2(assignments), mg.ROUTE_ORDER_MIN_PAD),
        ASSIGNMENTS=assignments,
        NLOCAL=hi - lo,
        BUCKET_PAD=plan.cursor.numel(),
        BT=mg.BW_BLOCK_T,
        MAX_UPE=triton.cdiv(assignments, mg.BW_BLOCK_T),
        U_MAX=plan.u_max,
        NSYNC=plan.sync.numel(),
        num_warps=4,
    )
    common = {"DEPTH": gu_depth, "SHUF": variant.shuffle, "PERM": perm, "LEAN": variant.lean}
    gu_k_loop = (gu_trips - gu_depth) * gu_bk
    dn_k_loop = (intermediate // dn_bk - dn_depth) * dn_bk
    gu_items = plan.u_max * (intermediate // mg.BW_GU_ROWS) * split
    dn_items = plan.u_max * (hidden // mg.BW_DN_ROWS)
    if variant.fused:
        grid = mg.BW_GRID if mg.BW_GRID > 0 else gu_items + dn_items
        _v2_fused_kernel[(grid,)](
            plan.xp,
            experts["gate_up"],
            experts["gate_up_scale"],
            experts["down"],
            experts["down_scale"],
            a_weight,
            plan.sorted,
            plan.unit,
            plan.n_units,
            inter,
            plan.part,
            plan.sync,
            y,
            gu_k_loop,
            dn_k_loop,
            K=reduction,
            INTER=intermediate,
            HIDDEN=hidden,
            TOP_K=top_k,
            BT=mg.BW_BLOCK_T,
            GU=mg.BW_GU_ROWS,
            DN=mg.BW_DN_ROWS,
            BK=gu_bk,
            DBK=dn_bk,
            S=split,
            U_MAX=plan.u_max,
            FENCE=mg.BW_FENCE,
            **common,
            **_launch_opts(mg.BW_DN_ROWS // 16, variant),
        )
    else:
        grid, iters = mg._bw_launch(gu_items)
        _v2_gate_up_kernel[(grid,)](
            plan.xp,
            experts["gate_up"],
            experts["gate_up_scale"],
            plan.sorted,
            plan.unit,
            plan.n_units,
            inter,
            plan.part,
            plan.cnt,
            gu_k_loop,
            K=reduction,
            INTER=intermediate,
            TOP_K=top_k,
            BT=mg.BW_BLOCK_T,
            GU=mg.BW_GU_ROWS,
            BK=gu_bk,
            DBK=dn_bk,
            S=split,
            GRID=grid,
            ITERS=iters,
            FENCE=mg.BW_FENCE,
            **common,
            **_launch_opts(2 * mg.BW_GU_ROWS // 16, variant),
        )
        grid, iters = mg._bw_launch(dn_items)
        _v2_down_kernel[(grid,)](
            inter,
            experts["down"],
            experts["down_scale"],
            a_weight,
            plan.sorted,
            plan.unit,
            plan.n_units,
            y,
            dn_k_loop,
            K=intermediate,
            HIDDEN=hidden,
            BT=mg.BW_BLOCK_T,
            DN=mg.BW_DN_ROWS,
            BK=dn_bk,
            GRID=grid,
            ITERS=iters,
            FENCE=mg.BW_FENCE,
            **{**common, "DEPTH": dn_depth},
            **_launch_opts(mg.BW_DN_ROWS // 16, variant),
        )
    mg.bw_combine(y, a_expert, a_weight, expert_range, top_k, out)
    return out


def fused_moe_variant(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    variant: Variant,
    **scratch,
) -> torch.Tensor:
    """Dispatch: v1 -> `mxfp4_gemv.fused_moe_bw`, anything else -> `fused_moe_bw2`."""
    if variant.name == "v1":
        return mg.fused_moe_bw(x, experts, routing, top_k, expert_range, **scratch)
    return fused_moe_bw2(x, experts, routing, top_k, expert_range, variant, **scratch)
