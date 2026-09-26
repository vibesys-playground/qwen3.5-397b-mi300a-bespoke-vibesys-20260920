"""Fused MXFP4 grouped-GEMV for the routed-expert path, in Triton.

What this replaces. `Model.moe` used to dequantize the activated experts' MXFP4 payloads into
a dense bf16 tensor in HBM (`mxfp4.dequant_mxfp4`) and then run two `bmm` calls over it. The
dequant's torch op chain moves ~37 bytes of HBM traffic per logical weight value against the
0.53 bytes the MXFP4 payload actually occupies, and the `bmm` then reads the dense copy again:
a ~70x amplification of the term that dominates the decode roofline. Caching the dequantized
weights is not an option either (205 GB of MXFP4 is 773 GB in bf16, against 512 GB of node
HBM), so the dequant has to move *into* the GEMM. See ROOFLINE_BOTTLENECK_ANALYSIS.md.

The two kernels below read the packed uint8 nibbles and the e8m0 scales straight from the
weights `load_experts` already produced (`[E_local, N, K // 2]` plus `[E_local, N, K // 32]`;
no loader or checkpoint change) and dequantize in registers. No dequantized weight value is
ever written to memory.

Static shape is the second, equally load-bearing property. Every (token, expert) assignment
gets its own program, and the two ways an assignment can be a no-op -- this rank does not own
the expert (expert parallelism), or the routing weight is zero (a padded, inactive batch row)
-- are GPU-side branches inside the kernel, not host-side boolean indexing. The torch spelling
could only express that sparsity as a data-dependent shape, which forces a device-to-host round
trip per layer per step; here the launch shape is a function of `max_batch` and `top_k` alone,
so the MoE costs zero host syncs and the step stays capturable.

Layout and math, for one projection:

    out[a, n] = sum_k  W[e(a), n, k] * x[t(a), k]

    W[e, n, k] = fp4_e2m1(nibble k of W_packed[e, n]) * 2 ** (W_scale[e, n, k // 32] - 127)

    packed nibbles: low nibble = even k, high nibble = odd k (mxfp4.py, verified against the
    FP8 source checkpoint). Scales are e8m0, one per 32-element block along k.

Assignments are indexed `a = t * top_k + j`, i.e. token-major, which is what
`top_i.reshape(-1)` gives. `_gate_up_silu_kernel` keeps that indexing for its output and
`_down_combine_kernel` relies on it to sum a token's `top_k` contributions in registers, so
the combine needs no atomics and is bit-reproducible run to run.

The scale is a power of two, so folding it out of the inner 32-element block
(`scale * sum(fp4 * x)` instead of `sum(scale * fp4 * x)`) is exact, not an approximation.
It saves broadcasting the scale over the block dimension.

Bandwidth-first decode path (`SEED_MOE_BW=1`, `fused_moe_bw`). Design note; unmeasured as of
this commit (offline gfx942 ISA only, `bench_moe_bw.py` is the first GPU check).

Why the earlier kernels are latency-bound: every one of them loads one K tile, waits, decodes,
multiplies. Compiling this file's kernels offline shows the mechanism: the AMDGPU scheduler
sinks each weight load to just above its first use, so a wave has one 1 KB load in flight and
pays full HBM latency per K step. Tiling, fp8 and grouping do not change that chain.

Budget (MI300A: 228 CUs, 2.1 GHz, 5.3 TB/s peak, ~3.5 realistic): HBM share is 11 B/clk/CU
(7.3 at 3.5 TB/s). Little's law at ~2 us loaded latency needs ~8 MB in flight, 35 KB per CU.
VALU issue is 1 wave64 instruction per clock per CU, so a wave's 4 KB weight trip may spend
~370 VALU instructions at peak (560 at 3.5 TB/s).

Choices, each against that budget:
- MFMA, not VALU FMA. ~10 tokens per expert means 10 FMAs per weight on VALU: 10 lane-ops per
  element against a budget of 64 lanes / 22 elements per clock = 2.9. bf16 16x16x16 MFMA with
  16 token columns costs 32 MFMAs per 4 KB trip per wave, off the VALU.
- bf16 operands, exact. fp4 x 2^s is exact in bf16, so there is no accuracy trade. A byte-per-
  value fp8 copy would double the bytes (roofline 15 -> 30 us/layer, +51 GB/rank) and fp8
  activations cost 20-25% relative error (the SEED_MOE_FP8MFMA attempt). Rejected.
- Decode = 76 VALU instructions per 32 values (2.4/value), scale included (`_bw_asm`): the
  e8m0 scale is added to the u16 exponent field of an 8-entry bf16 table once per 32-value
  block (one `v_pk_add_u16` per pair), then `v_perm_b32` byte lookups emit packed bf16 pairs
  directly. No per-element multiply, no fp32->bf16 convert, no scale FMA on the accumulator.
  Exact for scales in [2, 252] (`bw_scales_ok`, checked at load). Offline ISA of the K loop:
  354 VALU + 32 MFMA per 4 KB trip per wave, i.e. 0.95x the VALU budget at 5.3 TB/s, 0.63x
  at 3.5 TB/s.
- K order is free. The decode emits values in whatever order is cheapest (`bw_physical_k`)
  and in the MFMA A-operand lane placement, so the decoded tile feeds `tl.dot` by register
  renaming (no LDS, no lane shuffle). The activations are permuted once instead (`xp`, in
  the prep kernel), and gate_up stores `inter` already in down's order.
- Loads stay ahead of use. Each trip issues the next trip's 128-bit loads (4 per lane per
  64-row tile at BK=512), then a side-effecting empty asm (`_bw_fence`) that loads cannot
  cross, then decodes the current trip. The K-loop bound is a runtime value so LLVM cannot
  unroll the loop into straight-line code, where the loads sink anyway.
- Parallelism without split-K. Work unit = (expert, <=16 tokens); gate_up items are (unit, 32
  gate + 32 matching up rows), so SiLU(gate) * up stays in registers; down items are (unit,
  64 hidden rows) with the routing weight applied in the epilogue. At the mean ~12 distinct
  experts that is 384 gate_up and 768 down work items for 228 CUs, each read once. A static
  912-program grid strides over the GPU-side item count, so the launch is capturable and an
  idle program costs one load. At 6 experts gate_up has only 192 items: split-K for gate_up
  is the first knob to add if the bench shows it.
- Combine: one program per token sums its <=10 weighted rows of `y`; no atomics.

Known costs the ISA shows, and where to look first if `bench_moe_bw.py` falls short: the
loop-carried copies of the prefetched registers force the prefetch wait halfway through a
trip (overlap ~0.5 trip); the activation operand goes through LDS (16 KB per trip per
workgroup); row-major weights make each 128-bit load instruction touch 64 separate 16-byte
pieces. A load-time preshuffle to lane-contiguous 1 KB per instruction removes the last one.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

BLOCK_SCALE = 32
"""Elements per e8m0 scale, i.e. the OCP MXFP4 group size. Also the kernels' inner tile."""

BLOCK_N = int(os.environ.get("SEED_MOE_BLOCK_N", "64"))
"""Output rows one program computes. Tuning knob; correctness is independent of it."""

BLOCK_K = int(os.environ.get("SEED_MOE_BLOCK_K", "128"))
"""Reduction elements per inner iteration. Must be a multiple of `BLOCK_SCALE`."""

BLOCK_M = int(os.environ.get("SEED_MOE_BLOCK_M", "16"))
"""Assignments per block in the de-duplicating kernels. Must be at least 16 for `tl.dot`.

The instinct is that bigger is safer, because splitting one expert across two blocks costs a
whole re-read of its weight tile while a partial block only wastes FLOPs. Measurement says
otherwise: 16 is the fastest value tested at every prefill shape, and by a lot. One MI300A,
real dimensions, 512 experts, 512-token chunk (5,120 assignments), against the
per-assignment kernels:

    BLOCK_M   blocks   padded rows       ms   speedup
         16      833        13,328     9.68     4.70x
         32      673        21,536    24.04     1.89x
         64      593        37,952    17.36     2.62x
        128      553        70,784    18.81     2.42x

Note that 32 is worse than both 16 and 64, so this is not a smooth padding-versus-re-read
tradeoff. Fewer blocks does reduce weight traffic monotonically, and 16 has the most blocks
of any value here, so the re-read is not what dominates. What dominates is padded rows, which
16 also minimizes: at ten assignments per expert a 16-row block is 62% live and a 128-row
block is 8%. The non-monotonic dip at 32 is an MFMA tile-selection artifact on gfx942, whose
native shapes are 16x16 and 32x32; 32 rows across BLOCK_N=64 lands on a worse instruction
than either neighbor. Retune per architecture rather than reasoning it out.
"""

GROUPED_MIN_PER_EXPERT = int(os.environ.get("SEED_MOE_GROUPED_MIN", "3"))
"""Assignments per local expert above which the de-duplicating kernels are used.

Measured crossover is about 2.5 (one MI300A, real dimensions, 512 experts): 64 tokens is
0.83x, 128 tokens is 1.38x, 256 is 2.42x, 512 is 4.58x, 1024 is 5.34x. Set just above it, on
the side that keeps decode out: batch 48 at top-10 is 480 assignments, well under the 1,536
this threshold asks for, and the grouped path is a large loss at decode shapes.
"""

NUM_WARPS = int(os.environ.get("SEED_MOE_WARPS", "8"))
"""Warps per program for the per-assignment kernels. Left explicit rather than autotuned: an
autotune probe inside a captured region bakes the probe into the graph, and the shapes here
are server constants. 8 beats 4 by ~10% at one token and ~3% at 48."""

GROUPED_WARPS = int(os.environ.get("SEED_MOE_GROUPED_WARPS", "4"))
"""Warps per program for the de-duplicating kernels, which want the opposite of `NUM_WARPS`.

8 warps is 5x *slower* here (9.73 ms to 50.21 ms at a 512-token chunk) while being faster for
the per-assignment kernels. The difference is `tl.dot`: spreading a 16-row block over 8 warps
leaves most of each MFMA tile idle, whereas the per-assignment kernels reduce with `tl.sum`
and just want more waves in flight to cover the weight loads."""


def perm_lut(device: torch.device) -> bool:
    """Whether to decode fp4 with the `v_perm_b32` byte-table path instead of bit arithmetic.

    The fast path is hand-written gfx9 assembly (`_fp4_words_to_values`), so it needs a real
    AMD accelerator: it is unreachable under `TRITON_INTERPRET=1`, which runs the kernel body
    in Python, and it is not portable to a non-AMD backend. `_fp4_magnitudes` stays as the
    portable spelling of the same 16 values, and both are covered by the same bit-exactness
    test, so the fallback is a tested path rather than dead code.

    `SEED_MOE_PERM_LUT=0` forces the portable path on hardware that could run the assembly,
    which is what the ablation benchmark and any bisect of a suspected decode bug use.
    """
    if os.environ.get("SEED_MOE_PERM_LUT", "1") == "0":
        return False
    if os.environ.get("TRITON_INTERPRET") == "1" or device.type != "cuda":
        return False
    if torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx9")


def available(device: torch.device) -> bool:
    """True when the fused kernels can run for tensors on `device`.

    Triton compiles per target architecture and the kernels below are only reachable on an
    accelerator, so a CPU tensor (the hermetic tests, and any CPU-only run) keeps the torch
    path. `SEED_FUSED_MOE=0` forces the torch path on a machine that could run the kernels,
    which is the escape hatch for bisecting a suspected kernel bug against the oracle.
    """
    if not HAVE_TRITON or os.environ.get("SEED_FUSED_MOE", "1") == "0":
        return False
    return device.type == "cuda"


if HAVE_TRITON:
    _MX = tl.constexpr(BLOCK_SCALE)
    """`BLOCK_SCALE` again, as a Triton constexpr.

    A `@triton.jit` body may only read globals that are `tl.constexpr` instances, so the
    kernels below cannot use the plain int the host-side code wants.
    """

    @triton.jit
    def _fp4_magnitudes(code):
        """fp4 e2m1 code (0-15) to its float value, by writing the fp32 fields directly.

        The portable decode. `_fp4_words_to_values` is faster on gfx9 and is what ships
        there; this is what runs everywhere else, and both are held to the same
        all-256-byte-values bit-exactness test, so neither is the untested one.

        Computed rather than gathered from a 16-entry lookup table: a gather is one
        independent load per lane. Written as bit fields rather than as a select over the
        four magnitude cases because this kernel is bound by exactly this arithmetic, not
        by its loads. Measured on gfx942 at the real dimensions, 48 tokens, with every load
        and the reduction unchanged: the four-case select form is 2.87 ms and a form with
        the magnitude computation deleted outright is 1.24 ms, so the decode is 57% of the
        kernel. This spelling is 2.13 ms, 1.33x, and bit-identical (see below).

        `code` is `[s][e1][e0][m]`. For `e != 0` the value is `2**(e - 1) * (1 + m/2)`,
        which is exactly the fp32 whose exponent field is `e + 126` and whose top mantissa
        bit is `m`: one shift, one mask and one add. Only `e == 0`, where the value is
        `m * 0.5`, needs a select, so there is one instead of four.

        Exact against `mxfp4._FP4_VALUES` on all 16 codes, including the sign of zero:
        code 8 is `-0.0` in that table and comes back `-0.0` here, where the previous
        four-case form gave `+0.0` because the compiler folded the negation of a zero.
        Either is inert, since the accumulators start at `+0.0` and `v + 0.0 == v + -0.0`
        for every finite v, but this one needs no caveat.
        """
        c = code.to(tl.uint32)
        sign = (c >> 3) << 31
        body = ((c << 22) & 0x01C00000) + 0x3F000000
        half = (c & 1) * 0x3F000000
        return tl.where((c & 6) == 0, sign | half, sign | body).to(tl.float32, bitcast=True)

    _PERM_LUT_ASM = tl.constexpr(
        # $14 holds four packed bytes, i.e. eight fp4 codes. Two byte tables cover every
        # magnitude, so the decode is table lookups plus a sign fixup, not arithmetic.
        "v_mov_b32 $8, 0x3f3f3f00\n"  # bf16 high byte of magnitudes 0-3
        "v_mov_b32 $9, 0x40404040\n"  # bf16 high byte of magnitudes 4-7
        "v_and_b32 $10, 0x07070707, $14\n"  # four low-nibble magnitudes, one per byte
        "v_lshrrev_b32 $11, 4, $14\n"
        "v_and_b32 $11, 0x07070707, $11\n"  # four high-nibble magnitudes
        "v_perm_b32 $12, $9, $8, $10\n"  # four high bytes at once
        "v_perm_b32 $13, $9, $8, $11\n"
        "v_lshlrev_b32 $8, 4, $14\n"
        "v_and_b32 $8, 0x80808080, $8\n"
        "v_or_b32 $12, $12, $8\n"  # a low nibble's sign, lifted to the bf16 sign bit
        "v_and_b32 $8, 0x80808080, $14\n"
        "v_or_b32 $13, $13, $8\n"  # a high nibble's is already there
        "v_mov_b32 $8, 0xc0800000\n"  # bf16 low byte of magnitudes 0-3
        "v_mov_b32 $9, 0xc0804000\n"  # bf16 low byte of magnitudes 4-7
        "v_perm_b32 $10, $9, $8, $10\n"  # four low bytes at once
        "v_perm_b32 $11, $9, $8, $11\n"
        # Selector 12 emits a zero byte, so one more permute per value widens a bf16 byte
        # pair to the fp32 [00 00 lo hi] the reduction wants, with no shift and no convert.
        "v_mov_b32 $8, 0x04000c0c\n"
        "v_perm_b32 $0, $12, $10, $8\n"
        "v_perm_b32 $1, $13, $11, $8\n"
        "v_mov_b32 $8, 0x05010c0c\n"
        "v_perm_b32 $2, $12, $10, $8\n"
        "v_perm_b32 $3, $13, $11, $8\n"
        "v_mov_b32 $8, 0x06020c0c\n"
        "v_perm_b32 $4, $12, $10, $8\n"
        "v_perm_b32 $5, $13, $11, $8\n"
        "v_mov_b32 $8, 0x07030c0c\n"
        "v_perm_b32 $6, $12, $10, $8\n"
        "v_perm_b32 $7, $13, $11, $8"
    )
    """gfx9 assembly decoding eight fp4 codes to eight fp32 values in 28 vector instructions.

    `v_perm_b32 d, s0, s1, sel` builds `d` a byte at a time out of the eight bytes of
    `{s0, s1}`: byte `i` of `d` is byte `sel.u8[i]` of that window for a selector of 0-7, and
    the constant `0x00` for a selector of 12. Byte 0 of the window is the low byte of `s1`.
    Taken from measurement rather than from a manual: the bit-exactness tests below cover all
    256 packed byte values, which exercises every selector this uses.

    The window is exactly eight bytes wide and an fp4 magnitude has exactly eight values, so
    the high bytes of all four magnitudes in a packed word come out of one instruction and
    their low bytes out of a second. That is the whole idea: the decode stops being
    arithmetic and becomes two table lookups.

        code & 7:      0     1     2     3     4     5     6     7
        value:       0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0
        bf16 hi:      00    3F    3F    3F    40    40    40    40
        bf16 lo:      00    00    80    C0    00    40    80    C0

    An fp32 whose top two bytes are those and whose bottom two are zero is the same number,
    because an fp4 magnitude needs only two mantissa bits. So widening is a third permute per
    value (selector 12 supplies the zero bytes) rather than a shift and a convert.

    The sign is bit 3 of the code, which sits four bits below the bf16 sign bit for a low
    nibble and exactly at it for a high nibble; hence the shift-and-mask on one side and the
    mask alone on the other.

    Measured on gfx942 at the real dimensions, 48 tokens, 480 assignments, every load and the
    reduction unchanged, against `_fp4_magnitudes`: `gate_up` 2.15 -> 1.26-1.48 ms, `down`
    1.14 -> 0.59-0.64, one MoE layer 3.30 -> 1.92-2.07, i.e. **1.6-1.7x** on the kernel that
    is 96% of the decode step. (The spread is other work on the node, not the kernel; the
    `_fp4_magnitudes` side of the same runs varies by 0.3%.) The compiled inner loop falls
    from 411 vector instructions to 183, i.e. 12.8 ops per decoded weight value to 5.7.

    The regime has changed, which is the point rather than the speedup: the same kernel with
    the decode deleted outright and every load left in place measures 1.33-1.42 ms, which is
    *not* faster than keeping it. The MoE is memory-bound again. See
    DECODE_BOTTLENECK_2026-09-22.md section 4, recommendation 3.

    The eight `v_mov_b32` are the one obvious waste: an asm block is opaque, so they are
    re-executed on every trip of the reduction loop. Passing the constants in as ordinary
    loop-invariant Triton values gets them hoisted and is faster -- 20 instructions, 4.7 ops
    per value, 1.24 ms against 1.26 -- but it takes the block to 24 operands, and at that
    width it stops being trustworthy: `gate_up` comes back exact at the model's own
    dimensions and all-NaN at `INTER=64`, `K=128`, where the reduction is a single trip and
    there is no loop to hoist out of. The cause was not chased down, because 2% of one kernel
    does not justify shipping a decode whose correctness depends on the shape. The constants
    stay inside the block, where every shape the tests reach is exact.
    """

    _PERM_LUT_CONSTRAINTS = tl.constexpr("=v,=v,=v,=v,=v,=v,=v,=v,=&v,=&v,=&v,=&v,=&v,=&v,v")
    """Eight fp32 results, six early-clobber scratch registers, one packed input word.

    An inline-asm result tile has to be 32 bits wide -- an i8 one fails register allocation
    for constraint `v` -- which is why the scratch registers are spelled as six unused int32
    results: at `pack=1` that is exactly one VGPR each, and it lets the block hold the byte
    tables without the compiler having to guess at temporaries.
    """

    @triton.jit
    def _fp4_words_to_values(words):
        """`[..., W]` packed int32 words to `[..., 8 * W]` fp32 fp4 values, in element order.

        Each word holds four bytes and so eight codes, ordered low nibble then high nibble
        within each byte, which is the element order `mxfp4.dequant_mxfp4` produces. The
        assembly returns those eight as eight separate tiles, so they are woven back together
        here: the three-level interleave below is the identity permutation for that order,
        and its two halves are exactly the `low`/`high` byte tiles the portable path builds.
        """
        l0, h0, l1, h1, l2, h2, l3, h3, _s0, _s1, _s2, _s3, _s4, _s5 = tl.inline_asm_elementwise(
            _PERM_LUT_ASM,
            _PERM_LUT_CONSTRAINTS,
            [words],
            dtype=(
                tl.float32,
                tl.float32,
                tl.float32,
                tl.float32,
                tl.float32,
                tl.float32,
                tl.float32,
                tl.float32,
                tl.int32,
                tl.int32,
                tl.int32,
                tl.int32,
                tl.int32,
                tl.int32,
            ),
            is_pure=True,
            pack=1,
        )
        low = tl.interleave(tl.interleave(l0, l2), tl.interleave(l1, l3))
        high = tl.interleave(tl.interleave(h0, h2), tl.interleave(h1, h3))
        return tl.interleave(low, high)

    @triton.jit
    def _e8m0(scale):
        """e8m0 byte to 2 ** (byte - 127), by writing the byte into an fp32 exponent field.

        Exact for 1..254, and 255 becomes +inf, which is what `torch.exp2(255 - 127)` gives
        too (the format calls 255 NaN; mxfp4.py already documents that it is ignored). Byte 0
        is the one difference: this gives +0.0 where torch gives the subnormal 2**-127. A
        zero scale means the whole 32-element block is zero to within 1e-38, so the
        difference cannot move a bf16 result, and `exp2` would trade it for a fast-math
        approximation evaluated on every element.
        """
        return (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _dequant_tile(
        wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K: tl.constexpr, PERM: tl.constexpr
    ):
        """Dequantized weights for rows `n` of one expert, elements `[k0, k0 + BLOCK_K)`.

        `wq_base`/`ws_base` are that expert's int64 element offsets: a `[512, 2048, 2048]`
        uint8 payload is 2**31 bytes, so the expert term has to be computed in 64 bits even
        though every offset within an expert fits in 32.

        Returns `(values, scales)` shaped `[BLOCK_N, BLOCK_K // 32, 32]` and
        `[BLOCK_N, BLOCK_K // 32]`: the fp4 magnitudes and their per-block scale, kept apart
        so the caller can fold the scale out of the innermost reduction.

        `PERM` picks the byte-table decode, which reads the payload four bytes at a time
        because `v_perm_b32` works on whole registers. The addressing is the same addressing
        in units of four bytes: a row is `K // 2` bytes and `K` is a multiple of 32, so every
        offset here divides by four exactly.
        """
        gi = tl.arange(0, BLOCK_K // _MX)
        if PERM:
            word = tl.arange(0, _MX // 8)
            packed = tl.load(
                wq_ptr.to(tl.pointer_type(tl.int32))
                + (wq_base >> 2)
                + n[:, None, None] * (K // 8)
                + (k0 // 8 + gi[None, :, None] * (_MX // 8) + word[None, None, :]),
                mask=n_ok[:, None, None],
                other=0,
            )
            values = _fp4_words_to_values(packed)
        else:
            byte = tl.arange(0, _MX // 2)
            bytes_ = tl.load(
                wq_ptr
                + wq_base
                + n[:, None, None] * (K // 2)
                + (k0 // 2 + gi[None, :, None] * (_MX // 2) + byte[None, None, :]),
                mask=n_ok[:, None, None],
                other=0,
            ).to(tl.int32)
            values = tl.interleave(
                _fp4_magnitudes(bytes_ & 0xF), _fp4_magnitudes((bytes_ >> 4) & 0xF)
            )
        scale = tl.load(
            ws_ptr + ws_base + n[:, None] * (K // _MX) + (k0 // _MX + gi[None, :]),
            mask=n_ok[:, None],
            other=0,
        )
        return values, _e8m0(scale)

    @triton.jit
    def _rows_of(k0, BLOCK_K: tl.constexpr):
        """The `[BLOCK_K // 32, 32]` activation offsets matching `_dequant_tile`'s layout."""
        return k0 + tl.arange(0, BLOCK_K // _MX)[:, None] * _MX + tl.arange(0, _MX)[None, :]

    @triton.jit
    def _gate_up_silu_kernel(
        x_ptr,
        wq_ptr,
        ws_ptr,
        a_expert_ptr,
        a_weight_ptr,
        order_ptr,
        inter_ptr,
        expert_lo,
        expert_hi,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ORDERED: tl.constexpr,
        PERM: tl.constexpr,
    ):
        """`inter[a, n] = silu(gate_up[e, n] @ x[t]) * (gate_up[e, n + INTER] @ x[t])`.

        The gate half and the up half of the same `n` share one load of `x`, so the
        activation is fused into the projection and the only thing written is the already
        gated intermediate: half the traffic a separate elementwise pass would cost.

        `order_ptr` permutes only the program-to-assignment mapping. Sorting it by expert
        makes consecutive programs read the same weight tile, which turns their HBM reads
        into L2 hits, without changing where anything is written.
        """
        a = tl.load(order_ptr + tl.program_id(0)) if ORDERED else tl.program_id(0)
        expert = tl.load(a_expert_ptr + a)
        combine = tl.load(a_weight_ptr + a)
        # The expert-parallel drop and the inactive-row skip, GPU-side. This is what lets the
        # launch shape be a server constant instead of a property of the step's routing.
        if (expert >= expert_lo) & (expert < expert_hi) & (combine != 0.0):
            local = (expert - expert_lo).to(tl.int64)
            wq_base = local * (2 * INTER) * (K // 2)
            ws_base = local * (2 * INTER) * (K // _MX)
            n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
            n_ok = n < INTER
            acc_gate = tl.zeros([BLOCK_N], tl.float32)
            acc_up = tl.zeros([BLOCK_N], tl.float32)
            for k0 in range(0, K, BLOCK_K):
                xv = tl.load(x_ptr + (a // TOP_K) * K + _rows_of(k0, BLOCK_K)).to(tl.float32)
                vg, sg = _dequant_tile(
                    wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K, PERM
                )
                vu, su = _dequant_tile(
                    wq_ptr, ws_ptr, wq_base, ws_base, n + INTER, n_ok, k0, K, BLOCK_K, PERM
                )
                acc_gate += tl.sum(tl.sum(vg * xv[None, :, :], axis=2) * sg, axis=1)
                acc_up += tl.sum(tl.sum(vu * xv[None, :, :], axis=2) * su, axis=1)
            act = acc_gate / (1.0 + tl.exp(-acc_gate)) * acc_up
            tl.store(inter_ptr + a * INTER + n, act.to(inter_ptr.dtype.element_ty), mask=n_ok)

    @triton.jit
    def _down_combine_kernel(
        inter_ptr,
        wq_ptr,
        ws_ptr,
        a_expert_ptr,
        a_weight_ptr,
        out_ptr,
        expert_lo,
        expert_hi,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
    ):
        """`out[t, n] = sum_j combine[t, j] * (down[e(t, j), n] @ inter[t * TOP_K + j])`.

        One program owns one token's whole `top_k` sum for its slice of the hidden dimension,
        so the combine accumulates in registers and the store is a plain write. No atomics,
        hence a result that does not depend on how the scheduler orders the programs.

        A skipped assignment contributes nothing and, because the skip condition is rechecked
        here, `inter` never has to be zero-filled for the rows `_gate_up_silu_kernel` left
        untouched.
        """
        token = tl.program_id(0)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < HIDDEN
        acc = tl.zeros([BLOCK_N], tl.float32)
        for j in range(TOP_K):
            a = token * TOP_K + j
            expert = tl.load(a_expert_ptr + a)
            combine = tl.load(a_weight_ptr + a)
            if (expert >= expert_lo) & (expert < expert_hi) & (combine != 0.0):
                local = (expert - expert_lo).to(tl.int64)
                wq_base = local * HIDDEN * (K // 2)
                ws_base = local * HIDDEN * (K // _MX)
                part = tl.zeros([BLOCK_N], tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    xv = tl.load(inter_ptr + a * K + _rows_of(k0, BLOCK_K)).to(tl.float32)
                    vd, sd = _dequant_tile(
                        wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K, PERM
                    )
                    part += tl.sum(tl.sum(vd * xv[None, :, :], axis=2) * sd, axis=1)
                acc += part * combine
        tl.store(out_ptr + token * HIDDEN + n, acc.to(out_ptr.dtype.element_ty), mask=n_ok)

    @triton.jit
    def _dequant_rows(
        wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K: tl.constexpr, PERM: tl.constexpr
    ):
        """`[BLOCK_N, BLOCK_K]` dequantized weights, laid out for `tl.dot`.

        The same arithmetic as `_dequant_tile`, but 2D and with the scale already applied,
        because a `tl.dot` operand cannot be a 3D tile with the scale held to one side. The
        scale is read once per element rather than once per 32-element group, which is 32x the
        scale loads (the scale array is 1/32 the size of the payload, so it stays in L1) in
        exchange for not reshaping a tile between layouts.

        Applying the scale before the dot is exact: an fp4 magnitude needs 2 mantissa bits and
        the scale is a power of two, so the product is representable in bf16 and casting to
        the activation's dtype for the dot loses nothing.
        """
        ki = tl.arange(0, BLOCK_K)
        if PERM:
            kw = tl.arange(0, BLOCK_K // 8)
            words = tl.load(
                wq_ptr.to(tl.pointer_type(tl.int32))
                + (wq_base >> 2)
                + n[:, None] * (K // 8)
                + (k0 // 8 + kw[None, :]),
                mask=n_ok[:, None],
                other=0,
            )
            values = _fp4_words_to_values(words)
        else:
            kb = tl.arange(0, BLOCK_K // 2)
            packed = tl.load(
                wq_ptr + wq_base + n[:, None] * (K // 2) + (k0 // 2 + kb[None, :]),
                mask=n_ok[:, None],
                other=0,
            ).to(tl.int32)
            values = tl.interleave(
                _fp4_magnitudes(packed & 0xF), _fp4_magnitudes((packed >> 4) & 0xF)
            )
        scale = tl.load(
            ws_ptr + ws_base + n[:, None] * (K // _MX) + ((k0 + ki[None, :]) // _MX),
            mask=n_ok[:, None],
            other=0,
        )
        return values * _e8m0(scale)

    @triton.jit
    def _grouped_gate_up_kernel(
        x_ptr,
        wq_ptr,
        ws_ptr,
        slot_ptr,
        block_expert_ptr,
        inter_ptr,
        experts,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
    ):
        """gate_up + SiLU for one block of `BLOCK_M` assignments that all share an expert.

        This is the de-duplicating form. `_gate_up_silu_kernel` reads the expert's weight tile
        once per assignment, which is right when a batch of 48 tokens activates 311 distinct
        experts, and badly wrong for a 512-token prefill chunk where 5,120 assignments land on
        512 experts and every tile is re-read ten times. Here the tile is loaded once per
        block and multiplied against up to `BLOCK_M` token rows with `tl.dot`.

        Padding a partial block wastes FLOPs, not bytes, which is the right trade at 19x
        memory-bound. What does cost bytes is splitting one expert across several blocks, so
        `BLOCK_M` wants to be at least the average tokens per expert, not below it.

        Output goes to `inter[a]` at the assignment's natural index, so `_down_combine_kernel`
        and the grouped down kernel see the same layout.
        """
        expert = tl.load(block_expert_ptr + tl.program_id(0))
        if (expert < 0) | (expert >= experts):  # sentinel block: dropped or padded-out
            return
        rows = tl.load(slot_ptr + tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M))
        if tl.max(rows) < 0:  # a block past the last real one for this expert
            return
        row_ok = rows >= 0
        # `rows` is -1 on a partial block's padding lanes. Clamp it before it reaches any
        # address, the same way `_grouped_down_kernel` does: the masks below suppress the
        # accesses, but forming `inter_ptr + (-1) * INTER + n` relies on predication to
        # cover an address a kilobyte in front of the buffer, and that is not a property
        # worth depending on. Verified to be behavior-preserving at the prefill shape
        # (512 tokens, 5,120 assignments, 3,280 padding lanes): a poisoned guard region in
        # front of `inter` comes back untouched both before and after this change.
        safe = tl.where(row_ok, rows, 0)
        token = safe // TOP_K
        local = expert.to(tl.int64)
        wq_base = local * (2 * INTER) * (K // 2)
        ws_base = local * (2 * INTER) * (K // _MX)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < INTER
        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        acc_up = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            xv = tl.load(
                x_ptr + token[:, None] * K + (k0 + tl.arange(0, BLOCK_K))[None, :],
                mask=row_ok[:, None],
                other=0.0,
            )
            vg = _dequant_rows(wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K, PERM)
            vu = _dequant_rows(
                wq_ptr, ws_ptr, wq_base, ws_base, n + INTER, n_ok, k0, K, BLOCK_K, PERM
            )
            acc_gate = tl.dot(xv, tl.trans(vg.to(xv.dtype)), acc_gate, input_precision="ieee")
            acc_up = tl.dot(xv, tl.trans(vu.to(xv.dtype)), acc_up, input_precision="ieee")
        act = acc_gate / (1.0 + tl.exp(-acc_gate)) * acc_up
        tl.store(
            inter_ptr + safe[:, None] * INTER + n[None, :],
            act.to(inter_ptr.dtype.element_ty),
            mask=row_ok[:, None] & n_ok[None, :],
        )

    @triton.jit
    def _grouped_down_kernel(
        inter_ptr,
        wq_ptr,
        ws_ptr,
        slot_ptr,
        block_expert_ptr,
        y_ptr,
        experts,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
    ):
        """The down projection for one block of assignments sharing an expert, un-combined.

        It writes `y[a]` per assignment rather than summing a token's `top_k` contributions,
        because a token's assignments are spread over different experts and therefore
        different blocks. Summing here would need atomics, and an atomic combine makes the
        result depend on scheduler order; the combine is a handful of torch ops instead, which
        costs one extra read and write of `[A, hidden]` (2.5% of the expert traffic at a
        512-token chunk) and stays deterministic.
        """
        expert = tl.load(block_expert_ptr + tl.program_id(0))
        if (expert < 0) | (expert >= experts):
            return
        rows = tl.load(slot_ptr + tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M))
        if tl.max(rows) < 0:
            return
        row_ok = rows >= 0
        safe = tl.where(row_ok, rows, 0)
        local = expert.to(tl.int64)
        wq_base = local * HIDDEN * (K // 2)
        ws_base = local * HIDDEN * (K // _MX)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < HIDDEN
        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            xv = tl.load(
                inter_ptr + safe[:, None] * K + (k0 + tl.arange(0, BLOCK_K))[None, :],
                mask=row_ok[:, None],
                other=0.0,
            )
            vd = _dequant_rows(wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K, PERM)
            acc = tl.dot(xv, tl.trans(vd.to(xv.dtype)), acc, input_precision="ieee")
        tl.store(
            y_ptr + safe[:, None] * HIDDEN + n[None, :],
            acc.to(y_ptr.dtype.element_ty),
            mask=row_ok[:, None] & n_ok[None, :],
        )


def _block_k(reduction: int) -> int:
    """Largest power-of-two multiple of `BLOCK_SCALE` that divides `reduction`, up to `BLOCK_K`.

    The inner tile is a `tl.arange`, so it has to be a power of two, and it has to divide the
    reduction exactly or the last iteration reads past the row into the next one. The model's
    reductions (4096 for gate_up, 1024 for down) both take the full `BLOCK_K`; this only bites
    at the small dimensions the tests use.
    """
    if reduction % BLOCK_SCALE:
        raise ValueError(f"MXFP4 reduction must be a multiple of {BLOCK_SCALE}, got {reduction}")
    block = BLOCK_SCALE
    while block < BLOCK_K and reduction % (2 * block) == 0:
        block *= 2
    return block


def grouped_block_count(assignments: int, experts: int, block_m: int) -> int:
    """Upper bound on the blocks `align_blocks` can produce, so the grid is a constant.

    Each of the `experts + 1` buckets (the local experts plus the dropped-assignment
    sentinel) contributes at most one partial block beyond `assignments / block_m`. Blocks
    past the real count are left pointing at no assignments and return immediately, so
    overshooting costs launch slots and nothing else.
    """
    return -(-assignments // block_m) + experts + 1


def align_blocks(
    a_expert: torch.Tensor,
    a_weight: torch.Tensor,
    expert_range: tuple[int, int],
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group assignments into `block_m`-sized, per-expert blocks. No host synchronization.

    Returns `(slot_assign, block_expert)`:

    - `slot_assign[b * block_m + i]` is the assignment index that row `i` of block `b` holds,
      or -1 for padding.
    - `block_expert[b]` is the local expert id block `b` reads, or the sentinel `experts` for
      a block holding only assignments this rank drops.

    Every step is a fixed-shape torch op. That is the whole point: the obvious way to group by
    expert is to split on counts, which produces a data-dependent shape and a device-to-host
    round trip per layer. Here the shape comes from `grouped_block_count`, which depends only
    on the batch size and the expert count.

    The one subtle step is `block_expert`. Scattering each expert's id at its first block marks
    only that block, so the ids are propagated forward with a running maximum. Experts with no
    assignments share a first block with the next non-empty expert, and an `amax` scatter
    resolves that collision to the later (non-empty) one because expert ids increase.
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
    bucket_first = counts.cumsum(0) - counts
    rank = torch.arange(assignments, device=device) - bucket_first[sorted_bucket]
    slot = block_start[sorted_bucket] * block_m + rank

    n_blocks = grouped_block_count(assignments, experts, block_m)
    slot_assign = torch.full((n_blocks * block_m,), -1, dtype=torch.int32, device=device)
    slot_assign.scatter_(0, slot, order.to(torch.int32))

    marks = torch.full((n_blocks + 1,), -1, dtype=torch.long, device=device)
    marks.scatter_reduce_(
        0,
        block_start.clamp(max=n_blocks),
        torch.arange(experts + 1, device=device),
        reduce="amax",
    )
    block_expert = marks.cummax(0).values[:n_blocks].to(torch.int32)
    return slot_assign, block_expert


def use_grouped(assignments: int, experts: int) -> bool:
    """Whether the de-duplicating kernels beat the per-assignment ones at these shapes.

    The grouped form reads an expert's weight tile once per `BLOCK_M`-sized block instead of
    once per assignment, so it wins exactly when assignments pile up on the same expert. A
    512-token prefill chunk puts 5,120 assignments on 512 experts, ten deep, and wins about
    tenfold. A 48-token decode step puts 480 on the same 512 and mostly does not repeat, so
    the grouped form would pay `BLOCK_M`-wide dot products for one or two live rows and gain
    nothing. Both are fixed-shape and sync-free; this is a throughput choice, not a
    correctness one, and it reads only Python ints.

    `experts` is the expert count `assignments` is spread over, which is the *whole* expert
    axis, not the slice one rank owns. Under expert-parallel TP the caller still passes every
    assignment (the kernels drop the out-of-range ones GPU-side), so the pile-up depth on a
    local expert is `assignments / total_experts`, unchanged by the sharding. Passing a
    rank's 128-expert slice here instead would quadruple the apparent depth at TP=4 and
    select the grouped form for a decode step it loses on; see `fused_moe`.
    """
    return assignments >= GROUPED_MIN_PER_EXPERT * max(experts, 1)


def _grouped_moe(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor,
    inter: torch.Tensor,
    reduction: int,
) -> torch.Tensor:
    """`fused_moe`'s de-duplicating path: align into per-expert blocks, two GEMMs, combine."""
    perm = perm_lut(x.device)
    a_expert, a_weight = routing
    lo, hi = expert_range
    local_experts = hi - lo
    tokens = x.shape[0]
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = a_expert.numel()

    slot_assign, block_expert = align_blocks(a_expert, a_weight, expert_range, BLOCK_M)
    n_blocks = block_expert.numel()
    y = torch.zeros(assignments, hidden, dtype=x.dtype, device=x.device)

    _grouped_gate_up_kernel[(n_blocks, triton.cdiv(intermediate, BLOCK_N))](
        x,
        experts["gate_up"],
        experts["gate_up_scale"],
        slot_assign,
        block_expert,
        inter,
        local_experts,
        K=reduction,
        INTER=intermediate,
        TOP_K=top_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(reduction),
        PERM=perm,
        num_warps=GROUPED_WARPS,
    )
    _grouped_down_kernel[(n_blocks, triton.cdiv(hidden, BLOCK_N))](
        inter,
        experts["down"],
        experts["down_scale"],
        slot_assign,
        block_expert,
        y,
        local_experts,
        K=intermediate,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(intermediate),
        PERM=perm,
        num_warps=GROUPED_WARPS,
    )
    # The combine, in torch rather than in the down kernel: a token's `top_k` assignments live
    # in different experts' blocks, so summing them inside the kernel would need atomics and a
    # result that depends on scheduler order. `y` is zeroed, so a dropped assignment adds
    # nothing and needs no mask here. Fixed shape, no host sync.
    weighted = y.view(tokens, top_k, hidden) * a_weight.view(tokens, top_k, 1).to(y.dtype)
    out.copy_(weighted.sum(1))
    return out


def expert_sorted_order(a_expert: torch.Tensor) -> torch.Tensor:
    """Program order for the gate_up kernel that groups assignments sharing an expert.

    `torch.argsort` on a fixed-length `[T * top_k]` tensor: static shape, no host sync, so
    this keeps the whole path capturable. Consecutive programs then read the same weight tile.

    Never wired to a caller: `torch.argsort` is one Python call but, on this backend, lowers
    to several separate kernels (count/scan/permute), each its own graph node. Measured under
    CUDA-graph replay on real routing traces (batch 48, top_k 10, 512 experts, 1320
    step/layer samples): ~29.5 us/layer mean (28.96-30.83 us), ~1.8 ms summed over 60 layers,
    for a permutation whose only job is picking `_gate_up_silu_kernel`'s per-program
    assignment. `fused_expert_order` is the one-launch replacement, at ~14.4 us/layer mean
    (10.2-15.2 us) on the same traces; see it and `SEED_FUSED_MOE_ROUTING` in `model.py`.
    """
    return torch.argsort(a_expert, stable=True).to(torch.int32)


def _next_pow2(n: int) -> int:
    p = 1
    while p < max(n, 1):
        p *= 2
    return p


if HAVE_TRITON:

    @triton.jit
    def _fused_route_order_kernel(
        a_expert_ptr,
        order_ptr,
        cursor_ptr,
        ASSIGN_PAD: tl.constexpr,
        ASSIGNMENTS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        BUCKET_PAD: tl.constexpr,
    ):
        """One-launch stable-grouping counting sort: `expert_sorted_order`, fused.

        Same result `torch.argsort(a_expert, stable=True)` gives, as a permutation: every
        assignment index appears once, and reading `order` in order visits every assignment
        of expert 0, then every assignment of expert 1, and so on. Everything happens in one
        program (`grid=(1,)`), since `ASSIGNMENTS` (batch * top_k, <=480 in production) and
        `NUM_EXPERTS` (512) both fit comfortably in one block, with `tl.debug_barrier()`
        staging the three passes a counting sort needs:

        1. Count: each live lane atomically bumps `cursor[expert]`.
        2. Scan: an in-block `tl.cumsum` over the (tiny, fixed) expert axis turns counts into
           per-expert start offsets, written back into the same buffer.
        3. Scatter: each live lane atomically claims the next free slot in its expert's run
           (the same `cursor` buffer, now holding starts, doubles as the claim counter) and
           writes its own assignment index there.

        `cursor_ptr` (`[NUM_EXPERTS]`, scratch, reused call to call) is zeroed in step 1 here,
        not by the caller: no extra torch op, and always consistent before the atomics.

        Not bit-identical to `argsort(..., stable=True)`: within one expert's run, the atomic
        claim order depends on hardware scheduling, not assignment index. That never reaches a
        result, because `_gate_up_silu_kernel` reads `order` only to pick which assignment a
        program handles -- every assignment is still handled by exactly one program, writing
        its own output row -- so the *set* of assignments per expert is what has to match
        `expert_sorted_order`, and does; the order within a run is free to differ.
        """
        offs = tl.arange(0, ASSIGN_PAD)
        live = offs < ASSIGNMENTS
        key = tl.load(a_expert_ptr + offs, mask=live, other=NUM_EXPERTS).to(tl.int32)

        bucket_offs = tl.arange(0, BUCKET_PAD)
        bucket_ok = bucket_offs < NUM_EXPERTS
        tl.store(cursor_ptr + bucket_offs, 0, mask=bucket_ok)
        tl.debug_barrier()

        tl.atomic_add(cursor_ptr + key, 1, mask=live)
        tl.debug_barrier()

        counts = tl.load(cursor_ptr + bucket_offs, mask=bucket_ok, other=0)
        starts = tl.cumsum(counts, axis=0) - counts
        tl.store(cursor_ptr + bucket_offs, starts, mask=bucket_ok)
        tl.debug_barrier()

        slot = tl.atomic_add(cursor_ptr + key, 1, mask=live)
        tl.store(order_ptr + slot, offs.to(tl.int32), mask=live)


ROUTE_ORDER_MIN_PAD = 256
"""Floor on `ASSIGN_PAD` of the single-program atomic counting sorts
(`_fused_route_order_kernel`, `_bw_prep_kernel`, `mxfp4_moe_bw2`'s prep): their launch's
thread count (4 warps x 64 lanes on gfx942).

Measured on MI300A (Triton 3.4): at `ASSIGN_PAD=128` (80 assignments, i.e. a batch of 8 at
top_k 10), a 128-element tensor spread over 256 threads, the kernel left some `order` slots
unwritten in about half of calls, so `_gate_up_silu_kernel` read an uninitialized
assignment index and faulted (the bucket-8 decode-graph warmup crash). Every size whose
padded length is at least the thread count (160..480 assignments) came out a permutation
every time. Padding up costs only masked lanes in a single-program kernel.
`_bw_prep_kernel` (the `SEED_MOE_HIP`/`SEED_MOE_BW` work list) had the same fault: at
70-120 assignments ~10% of calls produced a bad `sorted`/`unit` list and the HIP kernels
faulted in the first eager warmup.
"""


def fused_expert_order(
    a_expert: torch.Tensor, num_experts: int, *, cursor: torch.Tensor | None = None
) -> torch.Tensor:
    """Fused-Triton drop-in for `expert_sorted_order`: one launch, no host sync, capturable.

    `cursor` is `[num_experts]` int32 scratch the kernel zeroes and reuses itself (see
    `_fused_route_order_kernel`); pass the decode hot path's preallocated buffer to avoid a
    fresh allocation per call. `order` is still freshly allocated: it is the value handed to
    `fused_moe`, which the caller may hold onto past this call.
    """
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `available`
        raise RuntimeError("triton is required for the fused MoE routing path")
    assignments = a_expert.numel()
    device = a_expert.device
    order = torch.empty(assignments, dtype=torch.int32, device=device)
    if cursor is None:
        cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    _fused_route_order_kernel[(1,)](
        a_expert,
        order,
        cursor,
        ASSIGN_PAD=max(_next_pow2(assignments), ROUTE_ORDER_MIN_PAD),
        ASSIGNMENTS=assignments,
        NUM_EXPERTS=num_experts,
        BUCKET_PAD=_next_pow2(num_experts),
        num_warps=4,
    )
    return order


def fused_moe(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    order: torch.Tensor | None = None,
    grouped: bool | None = None,
    total_experts: int | None = None,
) -> torch.Tensor:
    """Routed-expert output `[T, hidden]` for tokens `x` `[T, hidden]`, without dequantizing.

    `routing` is `(a_expert, a_weight)`, both `[T * top_k]`: the global expert id and the
    normalized combine weight of assignment `a = t * top_k + j`. Both are consumed on the
    device; nothing here reads a device value back to the host.

    `expert_range` is the half-open range of global expert ids this rank owns. Assignments
    outside it are dropped inside the kernel, so every rank launches the same shape and the
    caller all-reduces the partial sums.

    `grouped` picks the de-duplicating M-tiled kernels over the per-assignment ones; `None`
    lets `use_grouped` decide from the shapes, which is what callers should normally do. The
    two produce the same arithmetic and differ only in how much weight traffic they need.

    `total_experts` is the size of the whole expert axis, which is what that decision turns
    on. It defaults to the width of `expert_range`, correct when one rank owns every expert;
    an expert-parallel caller must pass the unsharded count, because it still hands in every
    assignment while owning a slice of the experts.

    The shared expert is not part of this: it is dense bf16 and belongs on the torch path.
    """
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `available`
        raise RuntimeError("triton is required for the fused MXFP4 MoE path")
    a_expert, a_weight = routing
    lo, hi = expert_range
    tokens, reduction = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(
            f"routing must be two [{assignments}] tensors, got "
            f"{tuple(a_expert.shape)} and {tuple(a_weight.shape)}"
        )
    # The kernels index every tensor as flat row-major, so a non-contiguous input would be
    # read at the wrong addresses and come back plausible but wrong. Checked rather than
    # silently `.contiguous()`d: a copy on the decode hot path is a bug worth seeing.
    for name, tensor in (("x", x), ("a_expert", a_expert), ("a_weight", a_weight)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if inter is None:
        inter = torch.empty(assignments, intermediate, dtype=x.dtype, device=x.device)
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=x.device)

    perm = perm_lut(x.device)
    spread = hi - lo if total_experts is None else total_experts
    if use_grouped(assignments, spread) if grouped is None else grouped:
        return _grouped_moe(
            x, experts, routing, top_k, expert_range, out=out, inter=inter, reduction=reduction
        )

    _gate_up_silu_kernel[(assignments, triton.cdiv(intermediate, BLOCK_N))](
        x,
        experts["gate_up"],
        experts["gate_up_scale"],
        a_expert,
        a_weight,
        a_expert if order is None else order,  # unread when ORDERED is False
        inter,
        lo,
        hi,
        K=reduction,
        INTER=intermediate,
        TOP_K=top_k,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(reduction),
        ORDERED=order is not None,
        PERM=perm,
        num_warps=NUM_WARPS,
    )
    _down_combine_kernel[(tokens, triton.cdiv(hidden, BLOCK_N))](
        inter,
        experts["down"],
        experts["down_scale"],
        a_expert,
        a_weight,
        out,
        lo,
        hi,
        K=intermediate,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(intermediate),
        PERM=perm,
        num_warps=NUM_WARPS,
    )
    return out


# -- de-duplicating MoE (SEED_MOE_DEDUP) ---------------------------------------------------
#
# Shapes at the real model, TP=4 (verified against `reference/config.json`): hidden = 4096,
# moe_intermediate = 1024 (MoE is expert-parallel, not TP-sharded on this axis -- see
# `tp.py`'s Plan.experts), local_experts = 512 / 4 = 128, top_k = 10. Per local expert per
# rank: gate_up packed [2048, 2048] uint8 (4,194,304 B) + scale [2048, 128] uint8 (262,144 B)
# = 4,456,448 B; down packed [4096, 512] uint8 (2,097,152 B) + scale [4096, 32] uint8
# (131,072 B) = 2,228,224 B. Sum 6,684,672 B = 6.68 MB, matching the measured per-assignment
# traffic (`_gate_up_silu_kernel` + `_down_combine_kernel` stream this per *assignment*, 480 of
# them at batch 48 top_k 10, i.e. up to ~40x reread of the 6-44 distinct local experts real
# traffic touches per layer -- see ROOFLINE_BOTTLENECK_ANALYSIS.md and the routing traces).
#
# Three designs measured and dropped before this one (real shapes, MI300A, batch 48, ~20
# distinct local experts, graph replay; production `fused_moe` is ~790 us/layer there):
#
# 1. Every local expert gets an always-present grid slot (0..127) plus a fixed
#    assignment-M-tile axis (grid size `local_experts * TOKPAD / BLOCK_M`, independent of how
#    many experts a step actually touches), K further split with atomics. 0.25-0.34x
#    production: at a real mean of ~12 distinct local experts out of 128, over 90% of the grid
#    is programs that load one bucket header and return, and dispatch overhead on ~64k mostly-
#    empty programs dominates; splitting K on top made it 2.1x worse at the same grid, because
#    the atomics add contention without buying back the dispatch cost.
# 2. The same idea rebuilt on `align_blocks`/`grouped_block_count` (already used by
#    `_grouped_moe`, already tested), so grid size tracks actual assignment count instead
#    (~159 blocks at batch 48 against 512+ above), still split-K with fp32-atomic gate/up
#    accumulators. 0.6-0.7x production: two fp32 atomic planes cost more than the plain bf16
#    store `_grouped_gate_up_kernel` already does, and SPLIT_K bought nothing once the grid
#    was no longer artificially starved (design 1's actual problem).
# 3. `align_blocks` grid, no split-K, `_grouped_gate_up_kernel` unchanged, but the down
#    kernel's routing-weight combine fused into an atomic add straight into `out[token]`
#    (replacing `_grouped_moe`'s separate `y` buffer + torch weighted-sum). 0.72-0.84x
#    production: a token's `top_k` assignments land in different experts' blocks, so several
#    programs atomic-add into the *same* `out[token]` concurrently; that HBM write contention
#    measured slower than a contention-free plain store plus a separate reduce.
#
# What actually wins, measured the same way: keep `_grouped_gate_up_kernel` and
# `_grouped_down_kernel` exactly as they are (`align_blocks` grid, plain stores, no atomics,
# no split) -- 425 + 162 = 587 us of kernel time, already faster than production on its own --
# and replace only `_grouped_moe`'s *torch-side* combine (`view` + `mul` + `sum` + `copy_`,
# each its own dispatch, starting from a fresh `torch.zeros(assignments, hidden)`) with one
# small fused kernel that loops a token's `top_k` un-combined rows in registers and writes
# `out[token]` once. Contention-free (one program owns one token, like
# `_down_combine_kernel` on the per-assignment path) and one launch instead of four.


DEDUP_BLOCK_M = int(os.environ.get("SEED_MOE_DEDUP_BLOCK_M", "16"))
"""Assignments per `align_blocks` block; reuses `BLOCK_M`'s measured-fastest value (16, see its
docstring) as a separately tunable knob for the dedup path."""

DEDUP_WARPS = int(os.environ.get("SEED_MOE_DEDUP_WARPS", "4"))
"""Warps per program for the two `_grouped_*` kernels. Matches `GROUPED_WARPS`'s measured-
fastest value (4, not `NUM_WARPS`'s 8) for the same reason: these reduce with `tl.dot`, not
`tl.sum`, and a 16-row block spread over 8 warps leaves most of an MFMA tile idle."""

DEDUP_COMBINE_BLOCK_N = int(os.environ.get("SEED_MOE_DEDUP_COMBINE_BLOCK_N", "256"))
"""Hidden-dim tile the combine kernel computes per program. Independent of `BLOCK_N` (no
weight tile involved here, just a `top_k`-deep gather-multiply-add over `y`)."""


if HAVE_TRITON:

    @triton.jit
    def _dedup_combine_kernel(
        y_ptr,
        a_weight_ptr,
        out_ptr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """`out[t, n] = sum_j a_weight[t*TOP_K+j] * y[t*TOP_K+j, n]`, one program per
        (token, N-tile). `y` is `_grouped_down_kernel`'s un-combined per-assignment output
        (zero on a dropped/padded assignment, same as the per-assignment path's `y`), so a
        program owns its whole token and sums in registers: no atomics, and the result does
        not depend on the order `_grouped_down_kernel`'s programs ran in, same determinism
        property `_down_combine_kernel` has on the per-assignment path.
        """
        token = tl.program_id(0)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < HIDDEN
        acc = tl.zeros([BLOCK_N], tl.float32)
        for j in range(TOP_K):
            a = token * TOP_K + j
            w = tl.load(a_weight_ptr + a)
            yv = tl.load(y_ptr + a * HIDDEN + n, mask=n_ok, other=0.0).to(tl.float32)
            acc += yv * w
        tl.store(out_ptr + token * HIDDEN + n, acc.to(out_ptr.dtype.element_ty), mask=n_ok)


def dedup_available(device: torch.device) -> bool:
    """Whether `fused_moe_dedup` can run: same gating as `available`, plus the explicit flag.

    Default is `0` as of integration build #2: the batch-48/96/192 graph-replay margin this
    module's docstring measured did not reproduce as a decode win once re-checked against the
    integrated tree (a no-win, same call as `SEED_MOE_BW`/`SEED_MOE_FP8MFMA`), so this stays
    off the default decode path until it is GPU-revalidated on the merged bundle.
    `SEED_MOE_DEDUP=1` opts back in for that revalidation or for bisecting a regression.
    """
    return available(device) and os.environ.get("SEED_MOE_DEDUP", "0") == "1"


def fused_moe_dedup(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed-expert output `[T, hidden]`, de-duplicating path: `align_blocks` groups
    assignments by local expert (no per-assignment weight reread), `_grouped_gate_up_kernel`/
    `_grouped_down_kernel` run unchanged, and `_dedup_combine_kernel` replaces `_grouped_moe`'s
    torch-side combine. Same contract as `fused_moe`: `routing` is `(a_expert, a_weight)` over
    `[T * top_k]`, `expert_range` is the half-open range of global expert ids this rank owns,
    no device value is read back to the host, and every buffer is sized from `T`/`top_k`/
    `expert_range` alone (all Python ints at call time, so the launch shape is static across
    steps and the whole thing is capturable): `align_blocks` itself is sync-free (its
    docstring covers why).

    `inter`/`y` are optional preallocated scratch (the decode hot path passes its own, same
    idea as `fused_moe`'s `inter`); omitted, they allocate fresh, fine off that path.
    """
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `dedup_available`
        raise RuntimeError("triton is required for the fused MXFP4 MoE path")
    a_expert, a_weight = routing
    lo, hi = expert_range
    local_experts = hi - lo
    tokens, reduction = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(
            f"routing must be two [{assignments}] tensors, got "
            f"{tuple(a_expert.shape)} and {tuple(a_weight.shape)}"
        )
    for name, tensor in (("x", x), ("a_expert", a_expert), ("a_weight", a_weight)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    perm = perm_lut(x.device)
    device = x.device

    slot_assign, block_expert = align_blocks(a_expert, a_weight, expert_range, DEDUP_BLOCK_M)
    n_blocks = block_expert.numel()

    if inter is None or inter.shape != (assignments, intermediate):
        inter = torch.empty(assignments, intermediate, dtype=x.dtype, device=device)
    if y is None or y.shape != (assignments, hidden):
        y = torch.zeros(assignments, hidden, dtype=x.dtype, device=device)
    else:
        y.zero_()
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=device)

    _grouped_gate_up_kernel[(n_blocks, triton.cdiv(intermediate, BLOCK_N))](
        x,
        experts["gate_up"],
        experts["gate_up_scale"],
        slot_assign,
        block_expert,
        inter,
        local_experts,
        K=reduction,
        INTER=intermediate,
        TOP_K=top_k,
        BLOCK_M=DEDUP_BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(reduction),
        PERM=perm,
        num_warps=DEDUP_WARPS,
    )
    _grouped_down_kernel[(n_blocks, triton.cdiv(hidden, BLOCK_N))](
        inter,
        experts["down"],
        experts["down_scale"],
        slot_assign,
        block_expert,
        y,
        local_experts,
        K=intermediate,
        HIDDEN=hidden,
        BLOCK_M=DEDUP_BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=_block_k(intermediate),
        PERM=perm,
        num_warps=DEDUP_WARPS,
    )
    _dedup_combine_kernel[(tokens, triton.cdiv(hidden, DEDUP_COMBINE_BLOCK_N))](
        y,
        a_weight,
        out,
        TOP_K=top_k,
        HIDDEN=hidden,
        BLOCK_N=DEDUP_COMBINE_BLOCK_N,
    )
    return out


# -- fp8 e4m3fnuz MFMA MoE (SEED_MOE_FP8MFMA) ----------------------------------------------
#
# Same `align_blocks`/`_dedup_combine_kernel` scaffolding as `fused_moe_dedup`; only
# `_grouped_gate_up_kernel`/`_grouped_down_kernel` are swapped for fp8-MFMA equivalents.
#
# e2m1 (fp4) weight nibbles convert EXACTLY to fp8 e4m3fnuz: fp8 has 3 mantissa bits against
# fp4's 1, so all 16 fp4 codes round-trip losslessly (verified bit-exact against
# `torch.float8_e4m3fnuz` for every code, both nibbles of all 256 packed bytes). The one
# wrinkle is fnuz's unsigned zero: OR-ing the fp4 sign bit into a zero-magnitude byte
# unconditionally gives 0x80, which decodes as NaN (fnuz reserves that bit pattern), not
# -0.0. The sign is gated on magnitude != 0 below.
#
# The e8m0 weight scale (one per 32-K block, matching BLOCK_SCALE) is applied AFTER the
# fp8xfp8 MFMA dot as one FMA per (output, 32-K block) rather than folded into the operand:
# `acc[m, n] += raw_dot_block[m, n] * wscale[n, block]`. BLOCK_SCALE=32 doubles as the fp8
# dot tile's K width, so one `tl.dot` call is one scale application.
#
# Activations are quantized to fp8 too, and ALSO per 32-K block rather than per whole token
# row (`ascale[m] = max(|x[m, block]|) / 240`, 240 being e4m3fnuz's max magnitude). A single
# per-token scale (absmax over the whole 4096/1024-wide row) was tried first and measured
# badly: mean relative error 25-35% against the bf16 grouped kernel on randn hidden states,
# because outlier elements anywhere in the row set the scale and starve every other 32-block
# of mantissa bits. Per-block scale tracks each block's own dynamic range instead and drops
# the pure fp8-quantization noise (isolated from SiLU/gating) to ~9-12% mean relative error,
# independent of how many blocks are accumulated (checked K=32..4096); the ~9-12% grows to
# ~20-25% on the full gated activation because SiLU near its zero crossing and the
# gate*up product both amplify independent per-block noise. Measured 1 MI300A, real gate_up
# (K=4096) and down (K=1024) dims, BLOCK_M=16: fp8 MFMA is ~1.03-1.10x the shipped bf16
# `_grouped_gate_up_kernel`/`_grouped_down_kernel` at M in {1,4,10,16} -- a real but modest
# win, well short of the roofline gap the per-32-block loop (128 trips for gate_up) still
# leaves on the table; batching 4 scale-blocks per loop trip to cut the trip count did not
# help further. See the PR description for the full ISA/timing/accuracy tables.


if HAVE_TRITON:

    @triton.jit
    def _fp4_to_fp8_bits(code):
        """fp4 e2m1 code (0-15) -> fp8 e4m3fnuz byte, exact. See module docstring above."""
        c = code.to(tl.uint32)
        sign = (c >> 3) & 1
        e = (c >> 1) & 3
        m = c & 1
        sub = m * 0x38
        norm = ((e + 7) << 3) | (m << 2)
        mag = tl.where((c & 6) == 0, sub, norm)
        signbit = tl.where(mag != 0, sign << 7, 0)
        return (mag | signbit).to(tl.uint8)

    _FP8_MAX = tl.constexpr(240.0)
    """e4m3fnuz's max finite magnitude (`torch.finfo(torch.float8_e4m3fnuz).max`)."""

    @triton.jit
    def _dequant_rows_fp8(wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, BLOCK_K: tl.constexpr):
        """`[BLOCK_N, BLOCK_K]` fp8 e4m3fnuz weight tile (exact, unscaled) + its e8m0 scale.

        `BLOCK_K` is meant to be exactly `BLOCK_SCALE`: one scale value per tile, applied by the
        caller after the MFMA dot rather than folded into the operand.
        """
        kb = tl.arange(0, BLOCK_K // 2)
        packed = tl.load(
            wq_ptr + wq_base + n[:, None] * (K // 2) + (k0 // 2 + kb[None, :]),
            mask=n_ok[:, None],
            other=0,
        ).to(tl.int32)
        lo = _fp4_to_fp8_bits(packed & 0xF)
        hi = _fp4_to_fp8_bits((packed >> 4) & 0xF)
        values = tl.interleave(lo, hi).to(tl.float8e4b8, bitcast=True)
        scale = tl.load(ws_ptr + ws_base + n * (K // _MX) + (k0 // _MX), mask=n_ok, other=0)
        e8 = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        return values, e8

    @triton.jit
    def _quantize_block_fp8(xv, row_ok):
        """`[BLOCK_M, 32]` bf16/fp32 activations -> (fp8 tile, per-row block scale).

        One scale per (row, 32-K block), matching the weight's own MXFP4 block granularity. See
        the module docstring for why this beats one scale for the whole row.
        """
        amax = tl.max(tl.where(row_ok[:, None], tl.abs(xv), 0.0), axis=1)
        ascale = tl.where(amax > 0, amax / _FP8_MAX, 1.0)
        xq = tl.clamp(xv / ascale[:, None], -_FP8_MAX, _FP8_MAX).to(tl.float8e4b8)
        return xq, ascale

    @triton.jit
    def _grouped_gate_up_kernel_fp8(
        x_ptr,
        wq_ptr,
        ws_ptr,
        slot_ptr,
        block_expert_ptr,
        inter_ptr,
        experts,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """gate_up + SiLU, fp8 MFMA. Same block/grid contract as `_grouped_gate_up_kernel`."""
        expert = tl.load(block_expert_ptr + tl.program_id(0))
        if (expert < 0) | (expert >= experts):
            return
        rows = tl.load(slot_ptr + tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M))
        if tl.max(rows) < 0:
            return
        row_ok = rows >= 0
        safe = tl.where(row_ok, rows, 0)
        token = safe // TOP_K
        local = expert.to(tl.int64)
        wq_base = local * (2 * INTER) * (K // 2)
        ws_base = local * (2 * INTER) * (K // _MX)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < INTER

        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        acc_up = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, _MX):
            xv = tl.load(
                x_ptr + token[:, None] * K + (k0 + tl.arange(0, _MX))[None, :],
                mask=row_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            xq, ascale = _quantize_block_fp8(xv, row_ok)
            vg, sg = _dequant_rows_fp8(wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, _MX)
            vu, su = _dequant_rows_fp8(
                wq_ptr, ws_ptr, wq_base, ws_base, n + INTER, n_ok, k0, K, _MX
            )
            dg = tl.dot(xq, tl.trans(vg), input_precision="ieee")
            du = tl.dot(xq, tl.trans(vu), input_precision="ieee")
            acc_gate += dg.to(tl.float32) * sg[None, :] * ascale[:, None]
            acc_up += du.to(tl.float32) * su[None, :] * ascale[:, None]
        act = acc_gate / (1.0 + tl.exp(-acc_gate)) * acc_up
        tl.store(
            inter_ptr + safe[:, None] * INTER + n[None, :],
            act.to(inter_ptr.dtype.element_ty),
            mask=row_ok[:, None] & n_ok[None, :],
        )

    @triton.jit
    def _grouped_down_kernel_fp8(
        inter_ptr,
        wq_ptr,
        ws_ptr,
        slot_ptr,
        block_expert_ptr,
        y_ptr,
        experts,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Down projection, fp8 MFMA. Same block/grid contract as `_grouped_down_kernel`."""
        expert = tl.load(block_expert_ptr + tl.program_id(0))
        if (expert < 0) | (expert >= experts):
            return
        rows = tl.load(slot_ptr + tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M))
        if tl.max(rows) < 0:
            return
        row_ok = rows >= 0
        safe = tl.where(row_ok, rows, 0)
        local = expert.to(tl.int64)
        wq_base = local * HIDDEN * (K // 2)
        ws_base = local * HIDDEN * (K // _MX)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < HIDDEN

        acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        for k0 in range(0, K, _MX):
            xv = tl.load(
                inter_ptr + safe[:, None] * K + (k0 + tl.arange(0, _MX))[None, :],
                mask=row_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            xq, ascale = _quantize_block_fp8(xv, row_ok)
            vd, sd = _dequant_rows_fp8(wq_ptr, ws_ptr, wq_base, ws_base, n, n_ok, k0, K, _MX)
            dd = tl.dot(xq, tl.trans(vd), input_precision="ieee")
            acc += dd.to(tl.float32) * sd[None, :] * ascale[:, None]
        tl.store(
            y_ptr + safe[:, None] * HIDDEN + n[None, :],
            acc.to(y_ptr.dtype.element_ty),
            mask=row_ok[:, None] & n_ok[None, :],
        )


def fp8mfma_available(device: torch.device) -> bool:
    """Whether `fused_moe_fp8mfma` can run: `dedup_available` plus the explicit opt-in flag.

    Off by default (`SEED_MOE_FP8MFMA=0`): measured speedup over the shipped bf16 grouped
    kernels is real but modest (~1.03-1.10x per kernel, see the module docstring), and fp8
    activation quantization costs real accuracy (~20-25% mean relative error on the gated
    intermediate, randn hidden states) that has not been validated against real hidden-state
    statistics. `SEED_MOE_FP8MFMA=1` opts in for benchmarking and further tuning.
    """
    return dedup_available(device) and os.environ.get("SEED_MOE_FP8MFMA", "0") == "1"


def fused_moe_fp8mfma(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """`fused_moe_dedup`, with the two `_grouped_*` kernels replaced by fp8 MFMA equivalents.

    Same contract, same `align_blocks`/`_dedup_combine_kernel` scaffolding; see
    `fused_moe_dedup`'s docstring for the shape/capturability argument, which is unchanged
    here, and the module docstring above this section for the fp8 kernel design.
    """
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `fp8mfma_available`
        raise RuntimeError("triton is required for the fused MXFP4 MoE path")
    a_expert, a_weight = routing
    lo, hi = expert_range
    local_experts = hi - lo
    tokens, reduction = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(
            f"routing must be two [{assignments}] tensors, got "
            f"{tuple(a_expert.shape)} and {tuple(a_weight.shape)}"
        )
    for name, tensor in (("x", x), ("a_expert", a_expert), ("a_weight", a_weight)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    device = x.device
    slot_assign, block_expert = align_blocks(a_expert, a_weight, expert_range, DEDUP_BLOCK_M)
    n_blocks = block_expert.numel()

    if inter is None or inter.shape != (assignments, intermediate):
        inter = torch.empty(assignments, intermediate, dtype=x.dtype, device=device)
    if y is None or y.shape != (assignments, hidden):
        y = torch.zeros(assignments, hidden, dtype=x.dtype, device=device)
    else:
        y.zero_()
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=device)

    _grouped_gate_up_kernel_fp8[(n_blocks, triton.cdiv(intermediate, BLOCK_N))](
        x,
        experts["gate_up"],
        experts["gate_up_scale"],
        slot_assign,
        block_expert,
        inter,
        local_experts,
        K=reduction,
        INTER=intermediate,
        TOP_K=top_k,
        BLOCK_M=DEDUP_BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=DEDUP_WARPS,
    )
    _grouped_down_kernel_fp8[(n_blocks, triton.cdiv(hidden, BLOCK_N))](
        inter,
        experts["down"],
        experts["down_scale"],
        slot_assign,
        block_expert,
        y,
        local_experts,
        K=intermediate,
        HIDDEN=hidden,
        BLOCK_M=DEDUP_BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=DEDUP_WARPS,
    )
    _dedup_combine_kernel[(tokens, triton.cdiv(hidden, DEDUP_COMBINE_BLOCK_N))](
        y,
        a_weight,
        out,
        TOP_K=top_k,
        HIDDEN=hidden,
        BLOCK_N=DEDUP_COMBINE_BLOCK_N,
    )
    return out


def reference_moe(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    dequant: Callable[[torch.Tensor, torch.Tensor, torch.dtype], torch.Tensor],
    compute_dtype: torch.dtype | None = None,
    chunk: int = 16,
) -> torch.Tensor:
    """The oracle `fused_moe` must match: dequantize to a dense tensor, then `bmm`.

    Deliberately the slow, obvious spelling of the same arithmetic, with the host-side
    boolean filtering the fused path removes. It is the numerical reference for the tests and
    the readable statement of what the kernels compute; it is not a serving fallback.

    `compute_dtype` selects which oracle: `x.dtype` (bf16) is what the engine's `bmm` path
    actually does today, and `torch.float32` isolates the fused kernel's own error by giving
    the oracle the same fp32 accumulation the kernels use.

    `chunk` bounds how many assignments are dequantized at once. It changes nothing about the
    result; it exists because the dense form of 480 assignments at the real dimensions is
    8 GB in bf16, which is the whole point of the kernel this is checking.
    """
    a_expert, a_weight = routing
    lo, hi = expert_range
    dt = compute_dtype or x.dtype
    keep = ((a_expert >= lo) & (a_expert < hi) & (a_weight != 0)).nonzero(as_tuple=True)[0]
    out = torch.zeros(x.shape[0], experts["down"].shape[1], dtype=dt, device=x.device)
    for start in range(0, keep.numel(), chunk):
        part = keep[start : start + chunk]
        local = (a_expert[part] - lo).to(torch.long)
        token = (part // top_k).to(torch.long)
        gate_up_w = dequant(experts["gate_up"][local], experts["gate_up_scale"][local], dt)
        down_w = dequant(experts["down"][local], experts["down_scale"][local], dt)
        gate, up = torch.bmm(x.to(dt)[token][:, None], gate_up_w.transpose(1, 2)).chunk(2, dim=-1)
        y = torch.bmm(torch.nn.functional.silu(gate) * up, down_w.transpose(1, 2))[:, 0]
        out.index_add_(0, token, y * a_weight[part, None].to(dt))
    return out.to(x.dtype)


# -- bandwidth-first decode MoE (SEED_MOE_BW) ------------------------------------------------
#
# Design note and budget math: see "Bandwidth-first decode path" in the module docstring.

BW_BLOCK_T = int(os.environ.get("SEED_MOE_BW_BLOCK_T", "16"))
"""Token columns per unit (the MFMA N side). 16 is one 16x16 MFMA tile; an expert with more
tokens gets several units, which re-read its weights (from L2/MALL, typically)."""

BW_GU_ROWS = int(os.environ.get("SEED_MOE_BW_GU_ROWS", "32"))
"""gate rows per gate_up work item; the item also owns the same number of up rows, so the
MFMA A tile is `2 * BW_GU_ROWS` rows and SiLU(gate) * up is formed in registers."""

BW_DN_ROWS = int(os.environ.get("SEED_MOE_BW_DN_ROWS", "64"))
"""hidden rows per down work item."""

BW_GU_BLOCK_K = int(os.environ.get("SEED_MOE_BW_GU_BLOCK_K", "512"))
BW_DN_BLOCK_K = int(os.environ.get("SEED_MOE_BW_DN_BLOCK_K", "512"))
"""Reduction elements per K-loop trip. 512 fp4 = 256 B per row per trip, i.e. with 64 rows and
256 lanes four `global_load_dwordx4` per lane per trip, all independent."""

BW_WARPS = int(os.environ.get("SEED_MOE_BW_WARPS", "4"))
BW_STAGES = int(os.environ.get("SEED_MOE_BW_STAGES", "1"))
"""Triton pipelining depth. 1: the kernels software-pipeline by hand (see `_bw_fence`)."""

BW_FENCE = int(os.environ.get("SEED_MOE_BW_FENCE", "2"))
"""How a trip's loads are kept ahead of its decode (see `_bw_fence`): 0 none, 1 workgroup
barrier (a hard scheduling boundary), 2 side-effecting empty asm."""

BW_GRID = int(os.environ.get("SEED_MOE_BW_GRID", "912"))
"""Programs per GEMM launch (static, so the launch is graph-capturable). Each program strides
over the GPU-side work list; `0` launches the static upper bound instead (one item per program,
the surplus exits after one load). 912 = 4 x 228 CUs."""


def bw_available(device: torch.device) -> bool:
    """Whether `fused_moe_bw` is selected: `available` plus the explicit `SEED_MOE_BW=1` opt-in."""
    return available(device) and os.environ.get("SEED_MOE_BW", "0") == "1"


def _bw_asm() -> str:
    """gfx9 assembly: four packed fp4 words (one 32-element scale block) plus its e8m0 scale
    to sixteen registers of scaled bf16 pairs. See `_bw_dequant`."""
    # Operand map (pack=4, every tensor contributes 4 consecutive registers):
    #   $0-$15  outputs P0[w0..w3], P1[..], P2[..], P3[..]    (bf16 pairs, see below)
    #   $16-$35 scratch (five int32 scratch tensors)
    #   $36-$39 the four packed words, $40-$43 the scale (broadcast; only $40 is read)
    q = ["$16", "$17", "$18", "$19"]
    tlo0, tlo1, thi0, thi1 = "$20", "$21", "$22", "$23"
    t, c, m80, s01, s23 = "$24", "$25", "$26", "$27", "$28"
    a, b, ha, hb, la, lb = "$29", "$30", "$31", "$32", "$33", "$34"
    words = ["$36", "$37", "$38", "$39"]
    scale = "$40"
    lines = [
        # bf16 bits of the eight fp4 magnitudes, minus 1.0's bits (0x3f80), as u16 pairs.
        # Adding (s << 7) to a pair then gives bf16(v * 2**(s - 127)): the scale is folded
        # into the table, once per 32 elements, not applied per element or per output.
        f"v_mov_b32 {q[0]}, 0xff800000",  # [0 -> masked below, 0.5]
        f"v_mov_b32 {q[1]}, 0x00400000",  # [1.0, 1.5]
        f"v_mov_b32 {q[2]}, 0x00c00080",  # [2.0, 3.0]
        f"v_mov_b32 {q[3]}, 0x01400100",  # [4.0, 6.0]
        f"v_lshlrev_b32 {t}, 7, {scale}",
        f"v_lshl_or_b32 {t}, {t}, 16, {t}",
        f"v_pk_add_u16 {q[0]}, {q[0]}, {t}",
        f"v_pk_add_u16 {q[1]}, {q[1]}, {t}",
        f"v_pk_add_u16 {q[2]}, {q[2]}, {t}",
        f"v_pk_add_u16 {q[3]}, {q[3]}, {t}",
        f"v_and_b32 {q[0]}, 0xffff0000, {q[0]}",  # magnitude 0 stays exactly 0
        # Split the eight u16 entries into a low-byte table and a high-byte table.
        f"v_mov_b32 {c}, 0x06040200",
        f"v_perm_b32 {tlo0}, {q[1]}, {q[0]}, {c}",
        f"v_perm_b32 {tlo1}, {q[3]}, {q[2]}, {c}",
        f"v_mov_b32 {c}, 0x07050301",
        f"v_perm_b32 {thi0}, {q[1]}, {q[0]}, {c}",
        f"v_perm_b32 {thi1}, {q[3]}, {q[2]}, {c}",
        f"v_mov_b32 {m80}, 0x80808080",
        f"v_mov_b32 {s01}, 0x05010400",
        f"v_mov_b32 {s23}, 0x07030602",
    ]
    for i, w in enumerate(words):
        p0, p1, p2, p3 = f"${i}", f"${4 + i}", f"${8 + i}", f"${12 + i}"
        lines += [
            f"v_and_b32 {a}, 0x07070707, {w}",  # magnitudes of e0 e2 e4 e6 (low nibbles)
            f"v_lshrrev_b32 {b}, 4, {w}",
            f"v_and_b32 {b}, 0x07070707, {b}",  # magnitudes of e1 e3 e5 e7
            f"v_perm_b32 {ha}, {thi1}, {thi0}, {a}",
            f"v_perm_b32 {hb}, {thi1}, {thi0}, {b}",
            f"v_perm_b32 {la}, {tlo1}, {tlo0}, {a}",
            f"v_perm_b32 {lb}, {tlo1}, {tlo0}, {b}",
            f"v_lshlrev_b32 {a}, 4, {w}",
            f"v_and_or_b32 {ha}, {a}, {m80}, {ha}",  # low-nibble sign (bit 3) -> bf16 sign
            f"v_and_or_b32 {hb}, {w}, {m80}, {hb}",  # high-nibble sign is already at bit 7
            f"v_perm_b32 {p0}, {ha}, {la}, {s01}",  # bf16 pair (e0, e2)
            f"v_perm_b32 {p1}, {ha}, {la}, {s23}",  # (e4, e6)
            f"v_perm_b32 {p2}, {hb}, {lb}, {s01}",  # (e1, e3)
            f"v_perm_b32 {p3}, {hb}, {lb}, {s23}",  # (e5, e7)
        ]
    return "\n".join(lines)


BW_ASM = _bw_asm()
BW_ASM_CONSTRAINTS = ",".join(["=&v"] * 36 + ["v"] * 8)
BW_PERM8 = (0, 2, 4, 6, 1, 3, 5, 7)
"""Logical K position within a packed word -> physical element index. `_bw_dequant` emits a
word's eight values in this order (the order the byte tables produce them in); the matching
activation is permuted the same way, which is free, so the weights never need reordering."""


def bw_scales_ok(experts: dict) -> bool:
    """Precondition of the folded-scale decode: every block holding a nonzero fp4 code has an
    e8m0 scale in [2, 252], so `v * 2**(s-127)` is a normal bf16 for every magnitude v.

    One host sync; call once at load time, never inside graph capture. All-zero blocks may
    carry any scale (the zero entry is masked, not scaled), which is what padding blocks do.
    """
    for w, s in (("gate_up", "gate_up_scale"), ("down", "down_scale")):
        packed, scale = experts[w], experts[s]
        bad = (scale < 2) | (scale > 252)
        if not bool(bad.any()):
            continue
        nonzero = ((packed & 0x77) != 0).view(*packed.shape[:-1], scale.shape[-1], -1).any(-1)
        if bool((bad & nonzero).any()):
            return False
    return True


def bw_physical_k(q: int, bk: int) -> int:
    """Physical K index of logical position `q` for K-trip size `bk` (the permutation is per
    trip): logical `128C + 32J + 16H + 4G + I4` holds physical
    `G * bk/4 + 32C + 8J + 2*I4 + H`. See `_bw_weight_tile`."""
    base, q = q - q % bk, q % bk
    c, j, h, g, i4 = q // 128, (q // 32) % 4, (q // 16) % 2, (q // 4) % 4, q % 4
    return base + g * (bk // 4) + 32 * c + 8 * j + 2 * i4 + h


if HAVE_TRITON:
    _BW_ASM = tl.constexpr(BW_ASM)
    _BW_CONSTRAINTS = tl.constexpr(BW_ASM_CONSTRAINTS)
    _BW_INTERPRET = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")
    """The interpreter's bf16 `tl.dot` is wrong (numpy has no bf16), so it gets fp32 operands."""

    @triton.jit
    def _bw_dot(w, a, acc):
        if _BW_INTERPRET:
            return tl.dot(w.to(tl.float32), tl.trans(a).to(tl.float32), acc)
        return tl.dot(w, tl.trans(a), acc)

    @triton.jit
    def _bw_bits(code, scale):
        """bf16 bits (int32, low 16 used) of fp4 `code` times 2**(scale-127), portable path."""
        c = code.to(tl.uint32)
        mag = c & 7
        # bf16(v) - 0x3f80 for magnitudes 0.5 .. 6, as in the asm tables
        base = tl.where(
            mag < 4,
            tl.where(mag < 2, 0xFF80, tl.where(mag == 2, 0x0000, 0x0040)),
            tl.where(
                mag < 6, tl.where(mag == 4, 0x0080, 0x00C0), tl.where(mag == 6, 0x0100, 0x0140)
            ),
        ).to(tl.uint32)
        bits = (base + (scale.to(tl.uint32) << 7)) & 0xFFFF
        bits = tl.where(mag == 0, 0, bits)
        return (bits | ((c >> 3) << 15)).to(tl.int32)

    @triton.jit
    def _bw_pairs_portable(words, scale):
        """Same four outputs as the asm: P_i holds (e[PERM8[2i]], e[PERM8[2i+1]])."""
        w = words.to(tl.uint32)
        e0 = _bw_bits(w & 0xF, scale)
        e1 = _bw_bits((w >> 4) & 0xF, scale)
        e2 = _bw_bits((w >> 8) & 0xF, scale)
        e3 = _bw_bits((w >> 12) & 0xF, scale)
        e4 = _bw_bits((w >> 16) & 0xF, scale)
        e5 = _bw_bits((w >> 20) & 0xF, scale)
        e6 = _bw_bits((w >> 24) & 0xF, scale)
        e7 = _bw_bits((w >> 28) & 0xF, scale)
        return e0 | (e2 << 16), e4 | (e6 << 16), e1 | (e3 << 16), e5 | (e7 << 16)

    @triton.jit
    def _bw_lo(p):
        return p.to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _bw_hi(p):
        return (p >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def _bw_fence(ptr, FENCE: tl.constexpr):
        """Empty side-effecting asm, placed right after a trip's loads are issued. LLVM keeps
        memory operations on their side of it, so the loads stay issued ahead of the decode
        that follows; it reads no loaded register, so it forces no wait. Without it the
        AMDGPU scheduler sinks every weight load to just above its first use, which turns
        the K loop back into a load -> wait -> decode chain (one 1 KB load in flight per
        wave; checked in the offline gfx942 ISA)."""
        if FENCE == 1:
            tl.debug_barrier()
        elif FENCE == 2:
            if not _BW_INTERPRET:
                # The weight pointer is an operand so alias analysis cannot prove the asm leaves
                # its memory alone (the kernel arguments are `noalias`); otherwise LLVM IR
                # passes sink the loads past the fence when no loop separates them. A fence is
                # also placed before the first loads: loads with no possible clobber before
                # them get AMDGPU's `noclobber` treatment and are again free to sink.
                tl.inline_asm_elementwise(
                    "; bw fence",
                    "=v,v,~{memory}",
                    [ptr + tl.zeros([1], tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )

    @triton.jit
    def _bw_load_tile(
        wq_ptr,
        ws_ptr,
        row0,
        k0,
        K: tl.constexpr,
        R: tl.constexpr,
        BK: tl.constexpr,
        GATE_UP: tl.constexpr,
        INTER: tl.constexpr,
    ):
        """Issue the loads for `[R, BK]` weights (rows from `row0`, physical K `[k0, k0+BK)`).
        Returns raw words `[16, 4, W, C, 4]` and their e8m0 scales `[16, 4, W, C, 1]`.

        Dim order matters: Triton's coalescer puts the contiguous dim (J, four words = one
        128-bit load = one 32-element scale block) in registers, then hands lanes and warps to
        the remaining dims in index order: R16 -> lanes 0-15, G -> lanes 16-63 (in groups of
        16), W -> warps, C -> registers. That is the MFMA A-operand placement for 16x16x16
        bf16 with kWidth 4 (lane = row + 16 * k-group), so after decoding, feeding `tl.dot`
        is a register renaming. Lane group g owns physical K `[g * BK/4, (g+1) * BK/4)`, so
        its C scale bytes are contiguous. Unmasked: a masked load costs a branch per load here.
        """
        C: tl.constexpr = BK // 128
        W: tl.constexpr = R // 16
        r = tl.arange(0, 16)[:, None, None, None, None]
        g = tl.arange(0, 4)[None, :, None, None, None]
        w = tl.arange(0, W)[None, None, :, None, None]
        c = tl.arange(0, C)[None, None, None, :, None]
        j = tl.arange(0, 4)[None, None, None, None, :]
        rr = w * 16 + r
        if GATE_UP:
            rr = rr + (w >= W // 2).to(tl.int32) * (INTER - R // 2)
        rows = row0 + rr.to(tl.int64)
        words = tl.load(
            wq_ptr.to(tl.pointer_type(tl.int32))
            + rows * (K // 8)
            + (k0 // 8 + g * (BK // 32) + c * 4 + j)
        )
        sc = tl.load(ws_ptr + rows * (K // 32) + (k0 // 32 + g * C + c))
        return words, sc

    @triton.jit
    def _bw_decode_tile(words, sc, R: tl.constexpr, BK: tl.constexpr, PERM: tl.constexpr):
        """`_bw_load_tile`'s output -> `[R, BK]` bf16, scale applied, logical K order
        (`bw_physical_k`)."""
        C: tl.constexpr = BK // 128
        W: tl.constexpr = R // 16
        sc = tl.broadcast_to(sc.to(tl.int32), (16, 4, W, C, 4))
        if PERM:
            p0, p1, p2, p3, _s0, _s1, _s2, _s3, _s4 = tl.inline_asm_elementwise(
                _BW_ASM,
                _BW_CONSTRAINTS,
                [words, sc],
                dtype=(tl.int32,) * 9,
                is_pure=True,
                pack=4,
            )
        else:
            p0, p1, p2, p3 = _bw_pairs_portable(words, sc)
        # value index within a word = 2 * pair + half, i.e. BW_PERM8 order = 4H + I4
        lo = tl.join(tl.join(_bw_lo(p0), _bw_lo(p2)), tl.join(_bw_lo(p1), _bw_lo(p3)))
        hi = tl.join(tl.join(_bw_hi(p0), _bw_hi(p2)), tl.join(_bw_hi(p1), _bw_hi(p3)))
        v = tl.reshape(tl.join(lo, hi), (16, 4, W, C, 4, 2, 4))  # [R16, G, W, C, J, H, I4]
        v = tl.permute(v, (2, 0, 3, 4, 5, 1, 6))  # [W, R16, C, J, H, G, I4]
        return tl.reshape(v, (R, BK))

    @triton.jit
    def _bw_trip(
        acc,
        an,
        wn,
        sn,
        arow,
        wq_ptr,
        ws_ptr,
        row0,
        k0,
        K: tl.constexpr,
        R: tl.constexpr,
        BK: tl.constexpr,
        PERM: tl.constexpr,
        GATE_UP: tl.constexpr,
        INTER: tl.constexpr,
        FENCE: tl.constexpr,
    ):
        """One K trip: issue trip k0+BK's loads, fence, then decode and multiply trip k0's
        (already in flight since the previous trip). Returns the new accumulator and the
        freshly issued loads."""
        av, wv, sv = an, wn, sn
        an = tl.load(arow + (k0 + BK))
        wn, sn = _bw_load_tile(wq_ptr, ws_ptr, row0, k0 + BK, K, R, BK, GATE_UP, INTER)
        _bw_fence(wq_ptr, FENCE)
        acc = _bw_dot(_bw_decode_tile(wv, sv, R, BK, PERM), av, acc)
        return acc, an, wn, sn

    @triton.jit
    def _bw_prep_kernel(
        a_expert_ptr,
        a_weight_ptr,
        cursor_ptr,
        sorted_ptr,
        unit_ptr,
        n_units_ptr,
        x_ptr,
        xp_ptr,
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
    ):
        """Program 0: counting-sort live assignments by local expert and cut each expert's run
        into `BT`-token units: `sorted` (assignment ids, grouped by expert),
        `unit[3u + {0,1,2}]` = (local expert, first position in `sorted`, live count),
        `n_units`. Program 1 + t: token t's activations into logical K order (`xp`)."""
        pid = tl.program_id(0)
        if pid > 0:
            q = tl.arange(0, K)
            qq = q % BK
            phys = (
                (q - qq)
                + ((qq // 4) % 4) * (BK // 4)
                + 32 * (qq // 128)
                + 8 * ((qq // 32) % 4)
                + 2 * (qq % 4)
                + (qq // 16) % 2
            )
            t = (pid - 1).to(tl.int64)
            tl.store(xp_ptr + t * K + q, tl.load(x_ptr + t * K + phys))
        else:
            offs = tl.arange(0, ASSIGN_PAD)
            in_range = offs < ASSIGNMENTS
            e = tl.load(a_expert_ptr + offs, mask=in_range, other=-1)
            w = tl.load(a_weight_ptr + offs, mask=in_range, other=0.0)
            live = in_range & (e >= lo) & (e < hi) & (w != 0.0)
            local = tl.where(live, e - lo, NLOCAL)
            buckets = tl.arange(0, BUCKET_PAD)
            b_ok = buckets < NLOCAL
            tl.store(cursor_ptr + buckets, 0)
            tl.debug_barrier()
            tl.atomic_add(cursor_ptr + local, 1, mask=live)
            tl.debug_barrier()
            cnt = tl.load(cursor_ptr + buckets, mask=b_ok, other=0)
            start = tl.cumsum(cnt, axis=0) - cnt
            units = (cnt + BT - 1) // BT
            ustart = tl.cumsum(units, axis=0) - units
            tl.debug_barrier()
            tl.store(cursor_ptr + buckets, start, mask=b_ok)
            tl.debug_barrier()
            pos = tl.atomic_add(cursor_ptr + local, 1, mask=live)
            tl.store(sorted_ptr + pos, offs.to(tl.int32), mask=live)
            for jj in range(MAX_UPE):
                m = b_ok & (jj < units)
                u = ustart + jj
                tl.store(unit_ptr + 3 * u, buckets, mask=m)
                tl.store(unit_ptr + 3 * u + 1, start + jj * BT, mask=m)
                tl.store(unit_ptr + 3 * u + 2, tl.minimum(cnt - jj * BT, BT), mask=m)
            tl.store(n_units_ptr, tl.minimum(tl.sum(units, axis=0), U_MAX))

    @triton.jit
    def _bw_gate_up_kernel(
        xp_ptr,
        wq_ptr,
        ws_ptr,
        sorted_ptr,
        unit_ptr,
        n_units_ptr,
        inter_ptr,
        k_loop,
        K: tl.constexpr,
        INTER: tl.constexpr,
        TOP_K: tl.constexpr,
        BT: tl.constexpr,
        GU: tl.constexpr,
        BK: tl.constexpr,
        DBK: tl.constexpr,
        GRID: tl.constexpr,
        ITERS: tl.constexpr,
        PERM: tl.constexpr,
        FENCE: tl.constexpr,
    ):
        """`inter[a, q(n)] = silu(gate[e, n] . x[t(a)]) * (up[e, n] . x[t(a)])` for one unit's
        tokens and `GU` values of n per work item. Work item = (unit, n-tile), n-tile fastest.
        `xp` is `x` in logical K order; `q` stores the intermediate in the down kernel's
        logical K order, so every activation load here and in down is contiguous."""
        NT: tl.constexpr = INTER // GU
        R: tl.constexpr = 2 * GU
        n_items = tl.load(n_units_ptr) * NT
        tt = tl.arange(0, BT)
        kk = tl.arange(0, BK)
        for it in range(ITERS):
            item = tl.program_id(0) + it * GRID
            if item < n_items:
                u = item // NT
                n0 = (item % NT) * GU
                expert = tl.load(unit_ptr + 3 * u)
                p0 = tl.load(unit_ptr + 3 * u + 1)
                cnt = tl.load(unit_ptr + 3 * u + 2)
                valid = tt < cnt
                slot = tl.load(sorted_ptr + p0 + tt, mask=valid, other=0)
                tok = (slot // TOP_K).to(tl.int64)
                row0 = expert.to(tl.int64) * (2 * INTER) + n0
                acc = tl.zeros([R, BT], tl.float32)
                # Manual one-trip-ahead prefetch, last trip peeled so no load is masked; see
                # `_bw_fence`. `k_loop` (= K - BK) is a runtime value on purpose: a loop LLVM
                # can fully unroll becomes straight-line code, where the loads sink anyway. Activations are issued before weights (loads retire in order).
                # Padded token columns read token 0's row and are never stored.
                xrow = xp_ptr + tok[:, None] * K + kk[None, :]
                _bw_fence(wq_ptr, FENCE)  # makes the loads below "clobbered", see `_bw_fence`
                xn = tl.load(xrow)
                wn, sn = _bw_load_tile(wq_ptr, ws_ptr, row0, 0, K, R, BK, True, INTER)
                _bw_fence(wq_ptr, FENCE)
                if _BW_INTERPRET:  # the interpreter cannot bound a loop by a runtime arg
                    for k0 in range(0, K - BK, BK):
                        acc, xn, wn, sn = _bw_trip(
                            acc,
                            xn,
                            wn,
                            sn,
                            xrow,
                            wq_ptr,
                            ws_ptr,
                            row0,
                            k0,
                            K,
                            R,
                            BK,
                            PERM,
                            True,
                            INTER,
                            FENCE,
                        )
                else:
                    for k0 in range(0, k_loop, BK):
                        acc, xn, wn, sn = _bw_trip(
                            acc,
                            xn,
                            wn,
                            sn,
                            xrow,
                            wq_ptr,
                            ws_ptr,
                            row0,
                            k0,
                            K,
                            R,
                            BK,
                            PERM,
                            True,
                            INTER,
                            FENCE,
                        )
                acc = _bw_dot(_bw_decode_tile(wn, sn, R, BK, PERM), xn, acc)
                g, v = tl.split(tl.permute(tl.reshape(acc, (2, GU, BT)), (1, 2, 0)))
                act = g / (1.0 + tl.exp(-g)) * v
                n = n0 + tl.arange(0, GU)
                # inverse of `bw_physical_k` for the down kernel's trip size
                nn = n % DBK
                gg = nn // (DBK // 4)
                cc = (nn % (DBK // 4)) // 32
                q = (
                    (n - nn)
                    + 128 * cc
                    + 32 * ((nn % 32) // 8)
                    + 16 * (nn % 2)
                    + 4 * gg
                    + (nn % 8) // 2
                )
                tl.store(
                    inter_ptr + slot[None, :].to(tl.int64) * INTER + q[:, None],
                    act.to(inter_ptr.dtype.element_ty),
                    mask=valid[None, :],
                )

    @triton.jit
    def _bw_down_kernel(
        inter_ptr,
        wq_ptr,
        ws_ptr,
        a_weight_ptr,
        sorted_ptr,
        unit_ptr,
        n_units_ptr,
        y_ptr,
        k_loop,
        K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BT: tl.constexpr,
        DN: tl.constexpr,
        BK: tl.constexpr,
        GRID: tl.constexpr,
        ITERS: tl.constexpr,
        PERM: tl.constexpr,
        FENCE: tl.constexpr,
    ):
        """`y[a, n] = a_weight[a] * (down[e, n] . inter[a])`: the routing weight is applied in
        the epilogue, so the combine that follows is a plain sum."""
        NT: tl.constexpr = HIDDEN // DN
        n_items = tl.load(n_units_ptr) * NT
        tt = tl.arange(0, BT)
        kk = tl.arange(0, BK)
        dn = tl.arange(0, DN)
        for it in range(ITERS):
            item = tl.program_id(0) + it * GRID
            if item < n_items:
                u = item // NT
                n0 = (item % NT) * DN
                expert = tl.load(unit_ptr + 3 * u)
                p0 = tl.load(unit_ptr + 3 * u + 1)
                cnt = tl.load(unit_ptr + 3 * u + 2)
                valid = tt < cnt
                slot = tl.load(sorted_ptr + p0 + tt, mask=valid, other=0).to(tl.int64)
                row0 = expert.to(tl.int64) * HIDDEN + n0
                acc = tl.zeros([DN, BT], tl.float32)
                arow = inter_ptr + slot[:, None] * K + kk[None, :]
                _bw_fence(wq_ptr, FENCE)  # makes the loads below "clobbered", see `_bw_fence`
                an = tl.load(arow)
                wn, sn = _bw_load_tile(wq_ptr, ws_ptr, row0, 0, K, DN, BK, False, 0)
                _bw_fence(wq_ptr, FENCE)
                if _BW_INTERPRET:
                    for k0 in range(0, K - BK, BK):
                        acc, an, wn, sn = _bw_trip(
                            acc,
                            an,
                            wn,
                            sn,
                            arow,
                            wq_ptr,
                            ws_ptr,
                            row0,
                            k0,
                            K,
                            DN,
                            BK,
                            PERM,
                            False,
                            0,
                            FENCE,
                        )
                else:
                    for k0 in range(0, k_loop, BK):
                        acc, an, wn, sn = _bw_trip(
                            acc,
                            an,
                            wn,
                            sn,
                            arow,
                            wq_ptr,
                            ws_ptr,
                            row0,
                            k0,
                            K,
                            DN,
                            BK,
                            PERM,
                            False,
                            0,
                            FENCE,
                        )
                acc = _bw_dot(_bw_decode_tile(wn, sn, DN, BK, PERM), an, acc)
                cw = tl.load(a_weight_ptr + slot, mask=valid, other=0.0).to(tl.float32)
                tl.store(
                    y_ptr + slot[None, :] * HIDDEN + (n0 + dn)[:, None],
                    (acc * cw[None, :]).to(y_ptr.dtype.element_ty),
                    mask=valid[None, :],
                )

    @triton.jit
    def _bw_combine_kernel(
        y_ptr,
        a_expert_ptr,
        a_weight_ptr,
        out_ptr,
        lo,
        hi,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """`out[t] = sum_j y[t * TOP_K + j]` over this rank's live assignments. One program
        owns a token, so no atomics and a fixed sum order. Dropped rows of `y` are never read,
        so `y` needs no zero-fill."""
        token = tl.program_id(0)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], tl.float32)
        for j in range(TOP_K):
            a = token * TOP_K + j
            e = tl.load(a_expert_ptr + a)
            w = tl.load(a_weight_ptr + a)
            if (e >= lo) & (e < hi) & (w != 0.0):
                acc += tl.load(y_ptr + a.to(tl.int64) * HIDDEN + n).to(tl.float32)
        tl.store(out_ptr + token * HIDDEN + n, acc.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _bw_combine_glue_kernel(
        y_ptr,
        a_expert_ptr,
        a_weight_ptr,
        shared_ptr,
        gate_ptr,
        out_ptr,
        lo,
        hi,
        sgate_row,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """`_bw_combine_kernel`, plus the shared-expert glue `Model.moe` otherwise does as three
        separate elementwise kernels: `out[t] = sum_j y[t*TOP_K+j] + sigmoid(gate[t]) * shared[t]`.

        `gate[t]` is the shared-expert gate's pre-sigmoid logit (`routing[:, experts_out:]`,
        width 1 -- one column, broadcast over `HIDDEN`), loaded through its own row stride
        (`sgate_row`, since it is typically a view into the wider fused router+gate GEMM
        output, not a freestanding contiguous tensor -- only column 0 is ever read) and
        sigmoided in fp32 to match the opmath `torch.sigmoid` uses internally on a bf16 input.
        Only the accumulation order
        differs from the unfused chain: there, `bw_combine`'s sum is rounded to `out`'s dtype
        (bf16) *before* the `+ sigmoid(gate) * shared` add, all three of those in bf16 tensor
        ops; here every term accumulates in fp32 and rounds once, at the final store. Same class
        of divergence as this codebase's other fused kernels (see `deltanet_fused.py`,
        `rmsnorm_fused.py`): last-ULP fp32-rounding-order noise, not a different result.
        """
        token = tl.program_id(0)
        n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], tl.float32)
        for j in range(TOP_K):
            a = token * TOP_K + j
            e = tl.load(a_expert_ptr + a)
            w = tl.load(a_weight_ptr + a)
            if (e >= lo) & (e < hi) & (w != 0.0):
                acc += tl.load(y_ptr + a.to(tl.int64) * HIDDEN + n).to(tl.float32)
        g = tl.load(gate_ptr + token * sgate_row).to(tl.float32)
        # Reproduce the unfused chain's rounding points exactly (out dtype after each of
        # `bw_combine`, `sigmoid`, `*`, `+`), so the flag changes launch count, not numerics.
        odt = out_ptr.dtype.element_ty
        gate_sig = (1.0 / (1.0 + tl.exp(-g))).to(odt).to(tl.float32)
        shared_v = tl.load(shared_ptr + token.to(tl.int64) * HIDDEN + n).to(tl.float32)
        glued = (gate_sig * shared_v).to(odt).to(tl.float32)
        routed = acc.to(odt).to(tl.float32)
        tl.store(out_ptr + token * HIDDEN + n, (routed + glued).to(odt))


def bw_unit_bound(assignments: int, local_experts: int, block_t: int) -> int:
    """Static upper bound on units: each touched expert adds at most one partial unit."""
    return min(local_experts, assignments) + assignments // block_t


class BwPlan:
    """GPU-side scratch for one `fused_moe_bw` call. Every shape is a Python int, so a
    captured graph replays with the same buffers; the prep kernel rewrites all of them.

    `block_t` is the unit size (tokens per unit); `permute_x=False` skips the permuted
    activation copy, for consumers that read `x` in its own K order (`moe_hip`)."""

    def __init__(
        self,
        tokens: int,
        top_k: int,
        reduction: int,
        local_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        block_t: int = BW_BLOCK_T,
        permute_x: bool = True,
    ) -> None:
        assignments = tokens * top_k
        self.block_t = block_t
        self.u_max = bw_unit_bound(assignments, local_experts, block_t)
        self.cursor = torch.empty(_next_pow2(local_experts + 1), dtype=torch.int32, device=device)
        self.sorted = torch.empty(assignments, dtype=torch.int32, device=device)
        self.unit = torch.empty(3 * self.u_max, dtype=torch.int32, device=device)
        self.n_units = torch.empty(1, dtype=torch.int32, device=device)
        self.xp = torch.empty(tokens, reduction, dtype=dtype, device=device) if permute_x else None


def bw_prep(x, a_expert, a_weight, expert_range, plan: BwPlan) -> None:
    """Routing work list (`sorted`, `unit`, `n_units`) and, when the plan has one, the
    permuted activation copy `xp`. Grid: program 0 routes, programs 1..T permute."""
    lo, hi = expert_range
    assignments = a_expert.numel()
    permute = plan.xp is not None
    _bw_prep_kernel[(1 + x.shape[0] if permute else 1,)](
        a_expert,
        a_weight,
        plan.cursor,
        plan.sorted,
        plan.unit,
        plan.n_units,
        x,
        plan.xp if permute else x,
        lo,
        hi,
        K=x.shape[1],
        BK=min(BW_GU_BLOCK_K, x.shape[1]),
        ASSIGN_PAD=max(_next_pow2(assignments), ROUTE_ORDER_MIN_PAD),
        ASSIGNMENTS=assignments,
        NLOCAL=hi - lo,
        BUCKET_PAD=plan.cursor.numel(),
        BT=plan.block_t,
        MAX_UPE=triton.cdiv(assignments, plan.block_t),
        U_MAX=plan.u_max,
        num_warps=4,
    )


def _bw_launch(items_max: int) -> tuple[int, int]:
    grid = items_max if BW_GRID <= 0 else min(BW_GRID, items_max)
    return grid, triton.cdiv(items_max, grid)


def bw_gate_up(experts, plan: BwPlan, top_k: int, inter: torch.Tensor, *, perm: bool) -> None:
    reduction = plan.xp.shape[1]
    intermediate = experts["gate_up"].shape[1] // 2
    grid, iters = _bw_launch(plan.u_max * (intermediate // BW_GU_ROWS))
    _bw_gate_up_kernel[(grid,)](
        plan.xp,
        experts["gate_up"],
        experts["gate_up_scale"],
        plan.sorted,
        plan.unit,
        plan.n_units,
        inter,
        reduction - min(BW_GU_BLOCK_K, reduction),
        K=reduction,
        INTER=intermediate,
        TOP_K=top_k,
        BT=BW_BLOCK_T,
        GU=BW_GU_ROWS,
        BK=min(BW_GU_BLOCK_K, reduction),
        DBK=min(BW_DN_BLOCK_K, intermediate),
        GRID=grid,
        ITERS=iters,
        PERM=perm,
        FENCE=BW_FENCE,
        num_warps=2 * BW_GU_ROWS // 16,
        num_stages=BW_STAGES,
    )


def bw_down(inter, experts, a_weight, plan: BwPlan, y: torch.Tensor, *, perm: bool) -> None:
    intermediate = inter.shape[1]
    hidden = experts["down"].shape[1]
    grid, iters = _bw_launch(plan.u_max * (hidden // BW_DN_ROWS))
    _bw_down_kernel[(grid,)](
        inter,
        experts["down"],
        experts["down_scale"],
        a_weight,
        plan.sorted,
        plan.unit,
        plan.n_units,
        y,
        intermediate - min(BW_DN_BLOCK_K, intermediate),
        K=intermediate,
        HIDDEN=hidden,
        BT=BW_BLOCK_T,
        DN=BW_DN_ROWS,
        BK=min(BW_DN_BLOCK_K, intermediate),
        GRID=grid,
        ITERS=iters,
        PERM=perm,
        FENCE=BW_FENCE,
        num_warps=BW_DN_ROWS // 16,
        num_stages=BW_STAGES,
    )


def bw_combine(y, a_expert, a_weight, expert_range, top_k: int, out: torch.Tensor) -> None:
    lo, hi = expert_range
    tokens, hidden = out.shape
    block_n = min(1024, hidden)
    _bw_combine_kernel[(tokens, triton.cdiv(hidden, block_n))](
        y, a_expert, a_weight, out, lo, hi, TOP_K=top_k, HIDDEN=hidden, BLOCK_N=block_n
    )


def bw_combine_glue(
    y: torch.Tensor,
    a_expert: torch.Tensor,
    a_weight: torch.Tensor,
    expert_range: tuple[int, int],
    top_k: int,
    shared: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """`bw_combine`, folding in the shared-expert glue (`_bw_combine_glue_kernel`): `out[t] =
    sum_j y[t*top_k+j] + sigmoid(gate[t]) * shared[t]`, one launch in place of `bw_combine`
    plus a `sigmoid`, a multiply, and an add. `gate` is `[tokens, >=1]`; only column 0 is read
    (the shared-expert gate is a single logit per token), so a wider slice -- as
    `routing[:, experts_out:]` may be when `fuse_moe_dense` packs exactly one extra column --
    is accepted as-is."""
    lo, hi = expert_range
    tokens, hidden = out.shape
    if shared.shape != (tokens, hidden) or not shared.is_contiguous():
        raise ValueError(
            f"bw_combine_glue: shared must be a contiguous [{tokens}, {hidden}] tensor, "
            f"got shape {shared.shape}"
        )
    if gate.shape[0] != tokens or gate.stride(1) != 1:
        raise ValueError(
            f"bw_combine_glue: gate must have {tokens} rows and unit column stride, "
            f"got shape {gate.shape}, stride {gate.stride()}"
        )
    block_n = min(1024, hidden)
    _bw_combine_glue_kernel[(tokens, triton.cdiv(hidden, block_n))](
        y,
        a_expert,
        a_weight,
        shared,
        gate,
        out,
        lo,
        hi,
        gate.stride(0),
        TOP_K=top_k,
        HIDDEN=hidden,
        BLOCK_N=block_n,
    )


def fused_moe_bw(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed-expert output `[T, hidden]`, bandwidth-first path. Same contract as
    `fused_moe_dedup`: static launch shapes, no host sync, capturable. Four launches: prep
    (routing + activation permute), gate_up(+SiLU), down(+routing weight), combine.
    `inter`/`y` are optional `[T * top_k, intermediate]` / `[T * top_k, hidden]` scratch;
    neither needs zeroing. Callers must have checked `bw_scales_ok` once at load time."""
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `bw_available`
        raise RuntimeError("triton is required for the fused MXFP4 MoE path")
    a_expert, a_weight = routing
    lo, hi = expert_range
    tokens, reduction = x.shape
    intermediate = experts["gate_up"].shape[1] // 2
    hidden = experts["down"].shape[1]
    assignments = tokens * top_k
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(
            f"routing must be two [{assignments}] tensors, got "
            f"{tuple(a_expert.shape)} and {tuple(a_weight.shape)}"
        )
    for name, tensor in (("x", x), ("a_expert", a_expert), ("a_weight", a_weight)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if (
        intermediate % BW_GU_ROWS
        or hidden % BW_DN_ROWS
        or reduction % min(BW_GU_BLOCK_K, reduction)
        or intermediate % min(BW_DN_BLOCK_K, intermediate)
        or min(BW_GU_BLOCK_K, reduction) % 128
        or min(BW_DN_BLOCK_K, intermediate) % 128
    ):
        raise ValueError("fused_moe_bw: tile sizes must divide the projection shapes")
    device = x.device
    if inter is None or inter.shape != (assignments, intermediate):
        inter = torch.empty(assignments, intermediate, dtype=x.dtype, device=device)
    if y is None or y.shape != (assignments, hidden):
        y = torch.empty(assignments, hidden, dtype=x.dtype, device=device)
    if out is None:
        out = torch.empty(tokens, hidden, dtype=x.dtype, device=device)
    perm = perm_lut(device)
    plan = BwPlan(tokens, top_k, reduction, hi - lo, device, x.dtype)
    bw_prep(x, a_expert, a_weight, expert_range, plan)
    bw_gate_up(experts, plan, top_k, inter, perm=perm)
    bw_down(inter, experts, a_weight, plan, y, perm=perm)
    bw_combine(y, a_expert, a_weight, expert_range, top_k, out)
    return out
