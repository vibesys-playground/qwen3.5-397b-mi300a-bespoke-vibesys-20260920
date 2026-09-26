"""`SEED_HIP_SKINNY_GEMM`: hand-written HIP weight-streaming GEMM for decode's skinny dense layers.

Every dense projection in a decode step is `out[M, N] = x[M, K] @ w[N, K]^T` with M = batch
<= 64: its cost is the weight bytes over HBM bandwidth. hipBLASLt's TunableOp-selected
Tensile solutions reach 0.4-2.2 TB/s here, and the Triton split-K kernel (`skinny_gemm`)
0.2-1.2 TB/s, because Triton stages both operands through LDS with a barrier per K step, so a
wave has few loads in flight. Kernel source: `skinny_hip.hip`; build and load follow
`moe_hip.py` (content-addressed `SEED_HIP_CACHE_DIR` build, one gfx942 code object).

Design (ideas: software pipelining with a register ring, as in CUTLASS's multistage
mainloop; split-K with a deterministic "last block reduces" fixup, as in Stream-K, Osama et
al., PPoPP 2023, and CUDA's threadFenceReduction sample):
  - Weights stream global -> registers, never through LDS: a wave step is 16 rows x 128 K
    (4 KB, four `global_load_dwordx4` per lane), and a DEPTH-deep register ring keeps
    DEPTH - 1 steps (12-16 KB per wave) in flight.
  - Activations `x[:M, wave's K slice]` are loaded once per program into registers
    (16 * MT * KSW VGPRs) and are the MFMA B operand; weights are the A operand, so the MFMA
    output rows are weight rows and columns are tokens (`v_mfma_f32_16x16x16_bf16`; MT
    column tiles cover M <= 16 MT; columns >= M are computed from a clamped row, never stored).
  - A program is KW waves along K; the KW partial 16 x 16 MT tiles are reduced through LDS
    (double-buffered, one barrier per tile). K / (128 KSW KW) programs along K (`split`)
    write fp32 partials, and the last one to arrive sums them in split order, so the result
    is deterministic run to run (not bit-identical to hipBLASLt: another fp32 sum order).
  - N is split across programs in contiguous runs of `tpp` 16-row tiles.

Routing: `ROUTES[(N, K, MT)] = (config id, tpp)` holds the configs that beat TunableOp's
hipBLASLt in `scratchpad/skinny_hip_bench.py` on MI300A (per-shape µs in the table's
comments); a shape not in the table keeps `F.linear`.

Offline ISA (ROCm 7.13 clang, gfx942): steady-state loop `s_waitcnt vmcnt(15..12)` before
each step's MFMAs, i.e. 3 steps (12 loads, 12 KB/wave) in flight at DEPTH 4; VGPRs 100-188,
no scratch (except config 23), 32 MT MFMAs per 4 KB step.

Decode step (`scratchpad/decode_profile_glue.py`, glue phase vs `+hipgemm`, TP=4, graph
replay, prompt 1024): b1 9.58 -> 9.46 ms, b16 18.27 -> 16.19 ms, b48 25.29 -> 25.26 ms (no
M 48 route). Graph validation passes; logits max |diff| 0.56 vs hipBLASLt, argmax equal.

Measured (MI300A, graph replay, vs TunableOp hipBLASLt; `scratchpad/skinny_hip_bench.py`):
  - Wins where hipBLASLt's tile is badly sized: o/out_proj at M 16 (12.2 vs 29.7 µs) and 32,
    shared down at M 16/32 (3.8 vs 16.1 µs), k/v_proj and router at M <= 32 (8-40%).
  - Loses 1.2-2.2x on q_proj, in_proj_all, shared gate_up and every shape at M >= 48. At M 48
    the x slice limits a wave to 128 K, so K = 4096 needs split 4 and ~2 WGs/CU of 170-VGPR
    waves: the run is a chain of latencies (x/weights, LDS reduce, partial store, arrival,
    partial reload), not a stream.
  - Streaming ceiling: lm_head (2 GB) reaches 2.6 TB/s (hipBLASLt 3.1). With the MFMAs
    removed the same loop reaches 2.8 TB/s, and 3.1 TB/s when each load instruction covers
    8 rows x 128 B instead of 16 rows x 64 B (the MFMA A-operand placement). So the lane
    layout costs ~10% and the MFMA chain ~7%; a layout that reads whole 128-B lines per
    instruction and then permutes into MFMA order (e.g. `ds_bpermute`) is the next step.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
HIP_SRC = HERE / "skinny_hip.hip"

ENABLED = os.environ.get("SEED_HIP_SKINNY_GEMM", "0") == "1"
MAX_M = 64

CONFIGS = [  # id -> (MT, KSW, KW, DEPTH); mirrors SK_CONFIGS in skinny_hip.hip
    (1, 4, 8, 4), (1, 4, 4, 4), (1, 2, 8, 4), (1, 2, 4, 4),
    (1, 1, 8, 4), (1, 1, 4, 4), (1, 2, 2, 4), (1, 1, 2, 4),
    (2, 2, 8, 4), (2, 2, 4, 4), (2, 1, 8, 4), (2, 1, 4, 4),
    (2, 1, 16, 2), (2, 2, 16, 2), (2, 1, 2, 4),
    (3, 1, 8, 4), (3, 1, 4, 4), (3, 1, 8, 2), (3, 1, 2, 4),
    (4, 1, 8, 4), (4, 1, 4, 4), (4, 1, 8, 2), (4, 1, 2, 4),
    (1, 4, 16, 4), (1, 2, 16, 4), (1, 1, 16, 4),
]  # fmt: skip

WS_BYTES = 32 << 20
"""fp32 split-K partials workspace per (device, stream); configs that need more are skipped."""
MAX_TILES = 1 << 15
"""Arrival counters per (device, stream): N <= 16 * MAX_TILES for split > 1."""

BUCKETS = (1, 16, 32, 48, 64)
"""M buckets the routing table is keyed on: a call with M rows takes the smallest bucket >= M
(the bucket's measurement is at M = bucket, its most expensive case)."""

ROUTES: dict[tuple[int, int, int], tuple[int, int]] = {
    # (N, K, M bucket): (config, tpp)  # HIP µs vs TunableOp hipBLASLt µs, MI300A, graph replay
    (256, 4096, 1): (1, 1),  # k/v_proj        7.10 vs  7.75
    (513, 4096, 1): (1, 1),  # router+sh gate  7.26 vs  8.02
    (256, 4096, 16): (6, 1),  # k/v_proj        8.25 vs 13.64
    (513, 4096, 16): (3, 1),  # router+sh gate  8.66 vs 10.14
    (4096, 2048, 16): (1, 2),  # o_proj/out_proj 12.2 vs 29.7
    (4096, 256, 16): (7, 2),  # shared down      3.84 vs 16.11
    (513, 4096, 32): (9, 1),  # router+sh gate 11.45 vs 14.19
    (4096, 2048, 32): (12, 2),  # o_proj/out_proj 14.9 vs 21.6
    (4096, 256, 32): (14, 2),  # shared down      4.10 vs 12.96
}
"""(N, K, M bucket) -> (config id, tpp): only the shapes where the kernel beat TunableOp's
hipBLASLt in `scratchpad/skinny_hip_bench.py` (per-rank TP=4 shapes). Every other shape --
q_proj, in_proj_all, shared gate_up, lm_head at any M, everything at M > 32 -- keeps
`F.linear`: there the kernel was 1.2-2.2x slower (see the module docstring)."""

_BINDING = r"""
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

extern "C" int skinny_hip_gemm(int, const void*, const void*, void*, void*, void*, int, int, int,
                               int, int, int, int, int, void*);
#define P(x) reinterpret_cast<void*>(x)

void gemm(int64_t cfg, int64_t x, int64_t w, int64_t out, int64_t ws, int64_t cnt, int64_t m,
          int64_t n, int64_t k, int64_t ldx, int64_t ldw, int64_t gx, int64_t split, int64_t tpp,
          int64_t stream) {
    int err = skinny_hip_gemm((int)cfg, P(x), P(w), P(out), P(ws), P(cnt), (int)m, (int)n, (int)k,
                              (int)ldx, (int)ldw, (int)gx, (int)split, (int)tpp, P(stream));
    if (err) throw std::runtime_error("skinny_hip_gemm: error " + std::to_string(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }
"""

_ext = None
_scratch: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def build_dir() -> Path:
    h = hashlib.sha256()
    for part in (HIP_SRC.read_bytes(), _BINDING.encode()):
        h.update(part)
    h.update(f"{torch.__version__} {torch.version.hip}".encode())
    root = os.environ.get("SEED_HIP_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "torch_extensions"
    return base / f"seed_skinny_hip_{h.hexdigest()[:16]}"


def ext():
    """Compile (first call per cache directory) or load the extension."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        d = build_dir()
        d.mkdir(parents=True, exist_ok=True)
        cpp, hip = d / "skinny_hip_binding.cpp", d / "skinny_hip_kernels.hip"
        for path, text in ((cpp, _BINDING), (hip, HIP_SRC.read_text())):
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
        prev_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
        try:
            _ext = load(
                name=d.name,
                sources=[str(cpp), str(hip)],
                build_directory=str(d),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=os.environ.get("SEED_HIP_SKINNY_VERBOSE") == "1",
            )
        finally:
            if prev_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = prev_arch
    return _ext


def device_ok(device: torch.device) -> bool:
    if device.type != "cuda" or torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx942"


def mt_of(m: int) -> int:
    return (m + 15) // 16


def split_of(cfg: int, k: int) -> int:
    """Programs along K for config `cfg` at this K, or 0 when K does not divide."""
    _, ksw, kw, _ = CONFIGS[cfg]
    kp = 128 * ksw * kw
    return k // kp if k % kp == 0 else 0


def valid(cfg: int, m: int, n: int, k: int, tpp: int) -> bool:
    mt = CONFIGS[cfg][0]
    split = split_of(cfg, k)
    if mt != mt_of(m) or split == 0 or tpp < 1:
        return False
    tiles = (n + 15) // 16
    return split == 1 or (tiles <= MAX_TILES and split * tiles * mt * 1024 <= WS_BYTES)


def _get_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K partials and arrival counters, one pair per (device, stream): launches on one
    stream are ordered, so they share it; the counters return to zero after every launch."""
    stream = torch.cuda.current_stream(device)
    key = (device.index if device.index is not None else torch.cuda.current_device(), stream.cuda_stream)
    s = _scratch.get(key)
    if s is None:
        ws = torch.empty(WS_BYTES // 4, dtype=torch.float32, device=device)
        cnt = torch.zeros(MAX_TILES, dtype=torch.int32, device=device)
        s = _scratch[key] = (ws, cnt)
    return s


def gemm(x2: torch.Tensor, w: torch.Tensor, cfg: int, tpp: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """`x2 @ w.T` for 2-D bf16 `x2` [M, K] (unit last stride, 16-byte aligned rows) and `w`
    [N, K] (same). Static launch shape, no host sync: capturable."""
    m, k = x2.shape
    n = w.shape[0]
    split = split_of(cfg, k)
    tiles = (n + 15) // 16
    gx = (tiles + tpp - 1) // tpp
    if out is None:
        out = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
    ws, cnt = _get_scratch(x2.device)
    ext().gemm(
        cfg, x2.data_ptr(), w.data_ptr(), out.data_ptr(), ws.data_ptr(), cnt.data_ptr(),
        m, n, k, x2.stride(0), w.stride(0), gx, split, tpp,
        torch.cuda.current_stream(x2.device).cuda_stream,
    )  # fmt: skip
    return out


def _aligned(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0


def bucket(m: int) -> int:
    return next(b for b in BUCKETS if b >= m)


_DEV_OK: dict[int, bool] = {}


def route(x2: torch.Tensor, w: torch.Tensor) -> tuple[int, int] | None:
    m, k = x2.shape
    if not (1 <= m <= MAX_M) or w.dim() != 2 or w.shape[1] != k:
        return None
    r = ROUTES.get((w.shape[0], k, bucket(m)))
    if r is None or x2.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return None
    if not x2.is_cuda or not (_aligned(x2) and _aligned(w)):
        return None
    idx = x2.device.index
    if idx not in _DEV_OK:
        _DEV_OK[idx] = device_ok(x2.device)
    return r if _DEV_OK[idx] else None


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`F.linear(x, w)`, through the HIP kernel when `SEED_HIP_SKINNY_GEMM=1` and the
    (N, K, M bucket) shape is routed (`ROUTES`)."""
    if not ENABLED:
        return F.linear(x, w)
    x2 = x.reshape(-1, x.shape[-1])
    r = route(x2, w)
    if r is None:
        return F.linear(x, w)
    return gemm(x2, w, *r).reshape(*x.shape[:-1], w.shape[0])
