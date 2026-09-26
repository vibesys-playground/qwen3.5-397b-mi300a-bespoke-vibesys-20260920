"""`moe_hip` (SEED_MOE_HIP): the gfx942 decode MoE kernels, checked in three layers.

1. numpy: the decode algorithm (e8m0 scale folded into a bf16 byte table, `v_perm_b32`
   lookups) re-stated in numpy, exhaustively against `mxfp4`'s fp4 values: every nibble in
   every byte position, every scale in the supported range [2, 252]. No compiler needed.
2. host C++ (`moe_hip_host_test.cpp`, needs any clang++): the kernel file's own helpers,
   compiled for the host, must produce the numpy decode bit for bit over all 256 scales, and
   a lane-level emulation of both kernels at the real shapes (the kernel's work split, lane
   offsets, MFMA register layout and LDS reduction order) must match a float64 reference.
3. GPU (gfx942 + SEED_MOE_HIP=1): `fused_moe_hip` against `mxfp4_gemv.reference_moe`.

    python -m pytest .../seed_tests/test_moe_hip.py -p no:cacheprovider --no-cov
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seed_tests"))

from mxfp4 import _FP4_VALUES, dequant_mxfp4  # noqa: E402

# ---- 1. numpy statement of the decode ------------------------------------------------------


def perm(hi: np.ndarray, lo: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """v_perm_b32 for selector bytes 0-7 (all the decode uses): byte i = byte sel[i] of hi:lo."""
    v = (hi.astype(np.uint64) << np.uint64(32)) | lo.astype(np.uint64)
    out = np.zeros(np.broadcast(hi, lo, sel).shape, np.uint64)
    for i in range(4):
        s = (sel.astype(np.uint64) >> np.uint64(8 * i)) & np.uint64(0xFF)
        assert (s < 8).all()
        out |= ((v >> (np.uint64(8) * s)) & np.uint64(0xFF)) << np.uint64(8 * i)
    return out.astype(np.uint32)


def pk_add_u16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = (a & 0xFFFF) + (b & 0xFFFF)
    hi = (a >> 16) + (b >> 16)
    return ((lo & 0xFFFF) | ((hi & 0xFFFF) << 16)).astype(np.uint32)


def make_tab(s: np.ndarray) -> tuple[np.ndarray, ...]:
    """moe_make_tab: (lo0, lo1, hi0, hi1) byte tables of bf16(fp4(v) * 2^(s-127)), v = 0..7."""
    s = s.astype(np.uint32)
    t = s * np.uint32(0x00800080)
    q = [pk_add_u16(np.uint32(c), t) for c in (0xFF800000, 0x00400000, 0x00C00080, 0x01400100)]
    q[0] = q[0] & np.uint32(0xFFFF0000)
    c_lo, c_hi = np.uint32(0x06040200), np.uint32(0x07050301)
    return (
        perm(q[1], q[0], c_lo),
        perm(q[3], q[2], c_lo),
        perm(q[1], q[0], c_hi),
        perm(q[3], q[2], c_hi),
    )


def decode_word(w: np.ndarray, tab) -> list[np.ndarray]:
    """moe_decode_word: 8 packed fp4 -> bf16 pairs (e0,e2), (e4,e6), (e1,e3), (e5,e7)."""
    lo0, lo1, hi0, hi1 = tab
    w = w.astype(np.uint32)
    a = w & np.uint32(0x07070707)
    b = (w >> np.uint32(4)) & np.uint32(0x07070707)
    ha, hb = perm(hi1, hi0, a), perm(hi1, hi0, b)
    la, lb = perm(lo1, lo0, a), perm(lo1, lo0, b)
    ha = ha | ((w << np.uint32(4)) & np.uint32(0x80808080))
    hb = hb | (w & np.uint32(0x80808080))
    s01, s23 = np.uint32(0x05010400), np.uint32(0x07030602)
    return [perm(ha, la, s01), perm(ha, la, s23), perm(hb, lb, s01), perm(hb, lb, s23)]


def words_for_bytes() -> np.ndarray:
    """The host test's words: byte i of word b is (b + 37 i) & 0xff."""
    b = np.arange(256, dtype=np.uint32)
    return sum((((b + 37 * i) & 0xFF) << (8 * i)).astype(np.uint32) for i in range(4))


def decode_all(scales: np.ndarray) -> np.ndarray:
    """[len(scales), 256 words, 4 pairs] uint32."""
    s = scales[:, None].astype(np.uint32)
    return np.stack(decode_word(words_for_bytes()[None, :], make_tab(s)), axis=-1)


def reference_bf16_bits(scales: np.ndarray) -> np.ndarray:
    """[len(scales), 256 words, 8 elements] bf16 bits of fp4 * 2^(s-127), element order as
    the pairs: (e0, e2, e4, e6, e1, e3, e5, e7)."""
    words = words_for_bytes()
    nib = np.stack([(words >> (4 * n)) & 15 for n in range(8)], axis=-1)  # element n = nibble n
    vals = np.asarray(_FP4_VALUES, np.float32)[nib]  # [256, 8]
    f = vals[None] * np.exp2(scales[:, None, None].astype(np.float32) - 127.0)
    bits = f.astype(np.float32).view(np.uint32)
    assert ((bits & 0xFFFF) == 0).all()  # exactly representable in bf16
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    return (bits >> 16)[..., order]


def pairs_to_elems(p: np.ndarray) -> np.ndarray:
    return np.stack([p & 0xFFFF, p >> 16], axis=-1).reshape(*p.shape[:-1], 8)


def test_numpy_decode_is_exact_for_every_code_and_supported_scale() -> None:
    scales = np.arange(2, 253)
    got = pairs_to_elems(decode_all(scales))
    want = reference_bf16_bits(scales)
    zero = (want & 0x7FFF) == 0
    # fp4 has +-0; the decode keeps the sign bit, so -0 decodes to bf16 -0 (0x8000)
    assert (got[~zero] == want[~zero]).all()
    assert ((got[zero] & 0x7FFF) == 0).all()
    # all 16 codes appear in every one of the 8 element positions
    words = words_for_bytes()
    for n in range(8):
        assert len(set(((words >> (4 * n)) & 15).tolist())) == 16


def test_scale_precondition_is_needed() -> None:
    """Outside [2, 252] the folded add over/underflows the exponent field: the load-time check
    (`mxfp4_gemv.bw_scales_ok`) is load-bearing, not decorative."""
    words = words_for_bytes()
    nib = np.stack([(words >> (4 * n)) & 15 for n in range(8)], axis=-1)[
        :, [0, 2, 4, 6, 1, 3, 5, 7]
    ]
    true = np.asarray(_FP4_VALUES, np.float64)[nib]
    for s in (0, 1, 253, 254):
        bits = pairs_to_elems(decode_all(np.array([s])))[0].astype(np.uint32) << 16
        got = bits.view(np.float32).astype(np.float64)
        assert not np.array_equal(got, true * 2.0 ** (s - 127))


# ---- 2. the kernel file's helpers, compiled for the host -----------------------------------


def find_clang() -> str | None:
    for cand in (
        os.environ.get("MOE_HIP_CLANG"),
        shutil.which("clang++"),
        "/opt/rocm/llvm/bin/clang++",
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/clang++"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


@pytest.fixture(scope="module", params=[0, 1], ids=["split", "flat"])
def host_test(tmp_path_factory, request) -> Path:
    """Built once per work split: `MOE_FLAT=0` (default) and `1` (`SEED_MOE_HIP_WIDE`)."""
    clang = find_clang()
    if clang is None:
        pytest.skip("no clang++ (set MOE_HIP_CLANG)")
    exe = tmp_path_factory.mktemp("moe_hip") / "moe_hip_host_test"
    src = ROOT / "seed_tests" / "moe_hip_host_test.cpp"
    flat = f"-DMOE_FLAT={request.param}"
    subprocess.run(
        [clang, "-std=c++17", "-O2", "-DMOE_HIP_HOST_TEST", flat, "-o", str(exe), str(src)],
        check=True,
        timeout=300,
    )
    return exe


def test_host_decode_matches_numpy_for_all_scales(host_test: Path, tmp_path: Path) -> None:
    out = tmp_path / "decode.bin"
    subprocess.run([str(host_test), "decode", str(out)], check=True, timeout=120)
    got = np.fromfile(out, np.uint32).reshape(256, 256, 4)
    assert (got == decode_all(np.arange(256))).all()


def test_host_emulation_of_both_kernels_matches_reference(host_test: Path) -> None:
    # 228: one segment per program (units <= grid); 5: several whole units per program;
    # 1000: most programs empty.
    p = subprocess.run(
        [str(host_test), "emulate", "228", "5", "1000"], capture_output=True, text=True, timeout=900
    )
    print(p.stdout)
    assert p.returncode == 0, p.stdout + p.stderr


# ---- 3. the real kernels ---------------------------------------------------------------------


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    import moe_hip

    return moe_hip.available(torch.device("cuda"))


@pytest.mark.skipif(not _gpu_ready(), reason="needs gfx942 and SEED_MOE_HIP=1")
@pytest.mark.parametrize(("tokens", "pile_up"), [(48, False), (48, True), (7, False), (1, False)])
def test_fused_moe_hip_matches_reference(tokens: int, pile_up: bool) -> None:
    """Real shapes (hidden 4096, intermediate 1024), 16 local experts of a 64-expert axis:
    dropped (other rank) and zero-weight rows, and with `pile_up` > 16 tokens on one expert
    (two units). Intermediate rounds to bf16 like the shipped kernels, hence their bar."""
    import moe_hip
    import mxfp4_gemv
    from test_mxfp4_fused_gemv import random_experts, random_routing

    dev = torch.device("cuda")
    span = (16, 32)
    ex = random_experts(16, moe_hip.HIDDEN, moe_hip.INTER, seed=11)
    ex = {k: v.to(dev) for k, v in ex.items()}
    assert moe_hip.supports(ex)
    a_expert, a_weight = random_routing(tokens, 64, 10, seed=4, zero_rows=(0,))
    if pile_up:
        a_expert[: tokens * 10 // 3] = span[0] + 1
    x = torch.randn(tokens, moe_hip.HIDDEN, generator=torch.Generator().manual_seed(6))
    x = x.to(torch.bfloat16).to(dev)
    routing = (a_expert.to(dev).contiguous(), a_weight.to(dev).float().contiguous())
    got = moe_hip.fused_moe_hip(x, ex, routing, 10, span)
    want = mxfp4_gemv.reference_moe(x, ex, routing, 10, span, dequant_mxfp4, torch.float32)
    err = float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-6))
    assert err < 2e-2


@pytest.mark.skipif(not _gpu_ready(), reason="needs gfx942 and SEED_MOE_HIP=1")
def test_fused_moe_hip_no_local_assignment() -> None:
    """A batch routed entirely to other ranks' experts (n_units 0; at T=1 and TP=4 about 1 in
    18 calls) launches with no work and returns zeros. Faulted before `moe_segment` checked."""
    import moe_hip
    from test_mxfp4_fused_gemv import random_experts

    dev = torch.device("cuda")
    ex = {k: v.to(dev) for k, v in random_experts(16, moe_hip.HIDDEN, moe_hip.INTER, seed=11).items()}
    x = torch.randn(1, moe_hip.HIDDEN, device=dev).to(torch.bfloat16)
    a_expert = torch.arange(40, 50, dtype=torch.int32, device=dev)
    a_weight = torch.full((10,), 0.1, device=dev)
    for _ in range(20):
        out = moe_hip.fused_moe_hip(x, ex, (a_expert, a_weight), 10, (16, 32))
        torch.cuda.synchronize()
        assert not out.any()
