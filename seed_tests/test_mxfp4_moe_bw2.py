"""`mxfp4_moe_bw2`: the SEED_MOE_BW_VARIANT kernels, the load-time reshuffle and the lean asm.

The kernel tests run under `TRITON_INTERPRET=1` on CPU (`seed_tests` convention) or on a GPU;
the reshuffle and asm tests are pure torch/Python and run anywhere."""

from __future__ import annotations

import os
import struct

import mxfp4_gemv
import mxfp4_moe_bw2 as bw2
import pytest
import torch
from mxfp4 import _FP4_VALUES, dequant_mxfp4
from test_mxfp4_fused_gemv import DEVICE, TOP_K, random_experts, random_routing, relative_error

NEEDS_KERNEL = pytest.mark.skipif(
    not bw2.HAVE_TRITON
    or not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)


@pytest.mark.parametrize("bk", [128, 512])
def test_shuffle_round_trips_exactly(bk: int) -> None:
    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (3, 64, 1024 // 2), dtype=torch.uint8, generator=g)
    scale = torch.randint(0, 256, (3, 64, 1024 // 32), dtype=torch.uint8, generator=g)
    sp, ss = bw2.bw_shuffle_packed(packed, bk), bw2.bw_shuffle_scale(scale, bk)
    assert sp.shape == packed.shape and ss.shape == scale.shape
    assert not torch.equal(sp, packed)
    assert torch.equal(bw2.bw_unshuffle_packed(sp, bk), packed)
    assert torch.equal(bw2.bw_unshuffle_scale(ss, bk), scale)
    # a permutation: same multiset of bytes
    assert torch.equal(sp.flatten().sort().values, packed.flatten().sort().values)


@pytest.mark.parametrize("bk", [128, 512])
def test_shuffle_matches_kernel_addressing(bk: int) -> None:
    """The byte the kernel reads for (row, lane group g, slot c, byte b) of trip t, at the
    `_v2_load_tile` address, is row-major byte (row, t*bk/2 + g*bk/8 + c*16 + b); likewise
    each lane's scale bytes. Every byte carries its own (row, column) id."""
    e, n, k = 2, 32, 1024
    rows = torch.arange(e * n).view(e, n, 1)
    cols = torch.arange(k // 2).view(1, 1, -1)
    ident = rows * (k // 2) + cols  # unique id per row-major byte
    flat = bw2.bw_shuffle_packed(ident, bk).reshape(-1)
    sid = rows * (k // 32) + torch.arange(k // 32).view(1, 1, -1)
    sflat = bw2.bw_shuffle_scale(sid, bk).reshape(-1)
    c_n = bk // 128
    for row in (0, 5, 17, 63):
        for t in range(k // bk):
            blk = (row // 16) * (k // bk) + t
            for g in range(4):
                lane = g * 16 + row % 16
                for c in range(c_n):
                    for b in (0, 7, 15):
                        got = flat[blk * 8 * bk + c * 1024 + lane * 16 + b]
                        want = row * (k // 2) + t * bk // 2 + g * bk // 8 + c * 16 + b
                        assert int(got) == want
                    got = sflat[blk * (bk // 2) + lane * c_n + c]
                    assert int(got) == row * (k // 32) + t * bk // 32 + g * c_n + c


def _perm(s0: int, s1: int, sel: int) -> int:
    win = (s0 << 32) | s1
    out = 0
    for i in range(4):
        b = (sel >> (8 * i)) & 0xFF
        v = (win >> (8 * b)) & 0xFF if b <= 7 else (0 if b == 12 else 0xFF)
        out |= v << (8 * i)
    return out


def _emulate(asm: str, regs: dict[str, int]) -> dict[str, int]:
    """gfx9 semantics of the handful of VALU ops the decode asm uses."""
    m = 0xFFFFFFFF

    def val(o: str) -> int:
        return int(o, 16) if o.startswith("0x") else (int(o) if o.isdigit() else regs[o])

    for line in asm.splitlines():
        op, rest = line.split(None, 1)
        d, *src = (x.strip() for x in rest.split(","))
        s = [val(x) for x in src]
        if op == "v_mov_b32":
            r = s[0]
        elif op == "v_and_b32":
            r = s[0] & s[1]
        elif op == "v_lshlrev_b32":
            r = (s[1] << s[0]) & m
        elif op == "v_lshrrev_b32":
            r = s[1] >> s[0]
        elif op == "v_lshl_or_b32":
            r = ((s[0] << s[1]) & m) | s[2]
        elif op == "v_mul_u32_u24":
            r = ((s[0] & 0xFFFFFF) * (s[1] & 0xFFFFFF)) & m
        elif op == "v_pk_add_u16":
            lo = ((s[0] & 0xFFFF) + (s[1] & 0xFFFF)) & 0xFFFF
            r = lo | ((((s[0] >> 16) + (s[1] >> 16)) & 0xFFFF) << 16)
        elif op == "v_perm_b32":
            r = _perm(s[0], s[1], s[2])
        elif op == "v_and_or_b32":
            r = (s[0] & s[1]) | s[2]
        else:
            raise ValueError(op)
        regs[d] = r
    return regs


def _bf16_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0] >> 16


def test_lean_asm_matches_v1_asm_and_fp4_values() -> None:
    """Both decode asm strings, emulated on every byte value and the scale range the
    folded-scale decode admits: identical registers, and the exact bf16 of fp4 x 2^(s-127)."""
    checked = 0
    for s in (2, 64, 118, 127, 200, 252):
        for base in range(0, 256, 16):
            words = [
                sum((base + 4 * w + i) << (8 * i) for i in range(4)) & 0xFFFFFFFF for w in range(4)
            ]
            v1 = {f"${36 + i}": w for i, w in enumerate(words)} | {"$40": s}
            v1 = _emulate(mxfp4_gemv.BW_ASM, v1)
            lean = {f"${32 + i}": w for i, w in enumerate(words)} | {"$36": s}
            lean |= {f"${40 + 4 * i}": c for i, c in enumerate(bw2.LEAN_CONSTANTS)}
            lean = _emulate(bw2.LEAN_ASM, lean)
            for p in range(16):
                assert lean[f"${p}"] == v1[f"${p}"], (s, base, p)
            for wi, w in enumerate(words):
                for pair in range(4):
                    for h in range(2):
                        e = mxfp4_gemv.BW_PERM8[2 * pair + h]
                        code = (w >> (4 * e)) & 0xF
                        want = _bf16_bits(_FP4_VALUES[code] * 2.0 ** (s - 127))
                        got = (lean[f"${4 * pair + wi}"] >> (16 * h)) & 0xFFFF
                        assert got == want or (want & 0x7FFF == 0 and got & 0x7FFF == 0)
                        checked += 1
    assert checked == 6 * 16 * 4 * 8
    assert len(bw2.LEAN_ASM.splitlines()) == 66
    assert len(mxfp4_gemv.BW_ASM.splitlines()) == 76


@pytest.fixture
def small_trips(monkeypatch):
    """128-value K trips so tiny shapes still have enough trips for depth 4 x split 2."""
    monkeypatch.setattr(mxfp4_gemv, "BW_GU_BLOCK_K", 128)
    monkeypatch.setattr(mxfp4_gemv, "BW_DN_BLOCK_K", 128)


@NEEDS_KERNEL
@pytest.mark.parametrize("name", [n for n in bw2.VARIANTS if n != "v1"])
def test_variant_matches_reference(name: str, small_trips) -> None:
    """Every v2 variant against the fp32 oracle, and against v1 bit for bit where the
    reduction order is v1's (no split-K). Covers multi-unit experts, dropped assignments,
    inactive rows and an expert-parallel slice."""
    variant = bw2.VARIANTS[name]
    tokens, hidden, intermediate, (lo, hi) = 20, 1024, 256, (4, 12)
    ex = random_experts(hi - lo, hidden, intermediate, seed=7)
    a_expert, a_weight = random_routing(tokens, 2 * hi, TOP_K // 2, seed=3, zero_rows=(0,))
    a_expert[: tokens * TOP_K // 4] = lo + 1  # > BW_BLOCK_T tokens on one expert
    routing = (a_expert.contiguous(), a_weight.to(torch.bfloat16).contiguous())
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(5))
    x = x.to(torch.bfloat16).to(DEVICE)
    want = mxfp4_gemv.reference_moe(
        x, ex, routing, TOP_K // 2, (lo, hi), dequant_mxfp4, torch.float32
    )
    v1 = mxfp4_gemv.fused_moe_bw(x, ex, routing, TOP_K // 2, (lo, hi))
    ex2 = {k: t.clone() for k, t in ex.items()}
    if variant.shuffle:
        bw2.bw_shuffle(ex2)
    got = bw2.fused_moe_variant(x, ex2, routing, TOP_K // 2, (lo, hi), variant)
    assert relative_error(got, want) < 2e-2
    if variant.splitk == 1:
        assert torch.equal(got, v1)
    # scratch is reusable: a second call on the same buffers gives the same answer
    again = bw2.fused_moe_variant(x, ex2, routing, TOP_K // 2, (lo, hi), variant)
    assert torch.equal(again, got)


@NEEDS_KERNEL
@pytest.mark.parametrize("name", ["v2c", "v2d"])
def test_variant_at_production_trip_size(name: str, monkeypatch) -> None:
    """gate_up at the shipped 512-value trip (C = 4 scale bytes per lane, the dword the
    shuffle makes contiguous) with split-K 2, against the fp32 oracle."""
    monkeypatch.setattr(mxfp4_gemv, "BW_GU_BLOCK_K", 512)
    monkeypatch.setattr(mxfp4_gemv, "BW_DN_BLOCK_K", 128)
    tokens, hidden, intermediate, (lo, hi) = 12, 2048, 256, (0, 6)
    ex = random_experts(hi - lo, hidden, intermediate, seed=11)
    routing = random_routing(tokens, hi, TOP_K // 2, seed=12, zero_rows=(1,))
    routing = (routing[0].contiguous(), routing[1].to(torch.bfloat16).contiguous())
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(13))
    x = x.to(torch.bfloat16).to(DEVICE)
    want = mxfp4_gemv.reference_moe(
        x, ex, routing, TOP_K // 2, (lo, hi), dequant_mxfp4, torch.float32
    )
    bw2.bw_shuffle(ex)
    got = bw2.fused_moe_variant(x, ex, routing, TOP_K // 2, (lo, hi), bw2.VARIANTS[name])
    assert relative_error(got, want) < 2e-2


def test_layout_mismatch_is_refused(small_trips) -> None:
    ex = random_experts(4, 1024, 256, seed=1)
    routing = random_routing(4, 4, 2, seed=2)
    x = torch.randn(4, 1024).to(torch.bfloat16).to(DEVICE)
    with pytest.raises(ValueError, match="bw_shuffle"):
        bw2.fused_moe_bw2(x, ex, routing, 2, (0, 4), bw2.VARIANTS["v2b"])


def test_unknown_variant_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SEED_MOE_BW_VARIANT", "v9")
    with pytest.raises(ValueError, match="SEED_MOE_BW_VARIANT"):
        bw2.selected_variant()
