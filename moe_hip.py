"""MXFP4 routed-expert decode MoE as hand-written HIP for gfx942 (`SEED_MOE_HIP=1`, off by default).

Why HIP. Every Triton MoE kernel in `mxfp4_gemv` is latency-bound at decode: the AMDGPU
scheduler sinks each weight load to just above its first use, so a wave has one load in flight
and pays full HBM latency per K step (~640 us/layer measured against a ~15 us bandwidth
roofline). In HIP the ring of in-flight loads is explicit and the offline ISA can be checked
for it. Kernel source: `moe_hip.hip`. Idea sources: software pipelining with a register ring
(the classic "prefetch distance" loop, as in CUTLASS's multistage mainloop and AMD's
Composable Kernel pipelines); fp4 decode by byte-permute lookup with the block scale folded
into the table (the `SEED_MOE_BW` design in `mxfp4_gemv`, itself an instance of LUT-based
low-bit dequant as in "LUT-GEMM", Park et al. 2022).

Layout (no repacking; the weights `load_experts` produces are used as-is): one wave tile is
16 weight rows x 512 K. Lane l = r + 16 g owns row r and K [128 g, 128 g + 128) of the wave's
slice: four 16-byte loads (64 contiguous bytes) plus one scale dword. That is the MFMA
16x16x16 bf16 A-operand placement, so decoded values feed `v_mfma_f32_16x16x16_bf16` from
registers. The decode emits a word's even elements, then its odd ones (one MFMA each); the
activation B operand is split the same way with two `v_perm_b32` per 8 values once per
segment, so no activation permute kernel is needed and `inter` is written in plain K order.

Decode: per 32 values (one scale), the e8m0 scale is added into the exponent field of an
8-entry bf16 table (4 `v_pk_add_u16`, 1 mask, 4 `v_perm_b32` to split low/high bytes); per 8
values, 3 ops extract the magnitudes, 4 `v_perm_b32` look up low/high bytes, 3 ops apply the
signs, 4 `v_perm_b32` interleave into bf16 pairs. 2.06 VALU instructions per weight, exact for
scales in [2, 252] (`mxfp4_gemv.bw_scales_ok`, checked at load).

Kernels (`moe_kernel` template; 8 waves per program):
  gate_up: wave kw owns K slice kw (of 8); item = 16 neurons (one gate tile + one up tile per
    wave); the 8 partial 16x16 tiles are reduced through LDS, SiLU(gate) * up is formed there.
  down: waves = 2 K slices x 4 row groups; item = 64 rows; K partials reduced through LDS; the
    routing weight is applied in the epilogue, so `mxfp4_gemv.bw_combine` is a plain sum.
Work list: `mxfp4_gemv._bw_prep_kernel` (units of <= 16 tokens of one expert; each distinct
expert's weights are read once per 16 tokens). Static grid (graph-capturable); a program runs
whole (unit, item-range) segments, so activations stay in registers for the segment and the
steady-state loop issues no load but the ring's. Programs with no work exit before loading.

Offline gfx942 ISA (ROCm 7.13 clang, `python moe_hip_isa.py --rocm <rocm>`): gate_up 167
VGPRs, down 160, no scratch, so one 8-wave program per CU (2 waves/SIMD). Steady-state loop,
per tile per wave: 4 `global_load_dwordx4` + 1 `global_load_dword` issued, then
`s_waitcnt vmcnt(15)` before the decode, i.e. three tiles (12 KB per wave, 96 KB per CU) stay
in flight while one decodes; 316 (gate_up) / 282 (down) VALU + 32 MFMA per 4 KB of weights,
epilogue included. Three compiler hazards had to be designed out to get there, each of which
drained the ring on every trip in an earlier draft: the machine scheduler sinking the loads
below the decode (`sched_barrier`), a dependent global load in the loop (the unit table; now
read per segment), and control flow that makes the waitcnt pass's loop-head merge
pessimistic (mid-body exits, a ring tile issued after the activation loads). The host test
and `moe_hip_isa.py` catch regressions of the first and third kind.

Predicted b48 decode, per layer and rank. MI300A: 228 CUs, 2.1 GHz, 5.3 TB/s peak; assume
4.0 TB/s achieved streaming. D = distinct local experts (the bench reports it; ~12 on the
real routing traces, ~75 for uniform random routing).
  bytes: D x 6.68 MB (gate_up 4.46 + down 2.23).
  in flight: 96 KB/CU x 228 = 22 MB, 10x the ~2 MB Little's law needs at 4 TB/s x 0.5 us
    latency, so the loop is HBM-bound, not latency-bound.
  VALU: a gate_up item per SIMD is 2 waves x 2 tiles x 316 x 4 clk = 5.1k clk against its
    66 KB per CU at 8.4 B/clk/CU (4 TB/s) = 7.9k clk: 64% VALU busy (85% at 5.3 TB/s).
  gate_up ~3 us start-up (unit table -> slot -> activations, dependent round trips) +
    D x 4.46 MB / 4 TB/s = 3 + 1.12 D us; down ~3 + 0.56 D us; prep (Triton, one program)
    ~5 us; combine ~3 us; graph gaps ~2 us.
  Total ~16 + 1.67 D us: D = 12 -> ~36 us/layer against ~640 us for the Triton paths, so
  ~2.2 ms instead of ~38 ms of MoE per 60-layer step. Refuted if gate_up alone
  (`bench_moe_bw.py --impl hip --gate-up-only`) takes more than ~2x its byte time; then look
  at achieved bandwidth and the start-up chain before the decode.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mxfp4_gemv
import torch

HERE = Path(__file__).resolve().parent
HIP_SRC = HERE / "moe_hip.hip"
HIDDEN, INTER = 4096, 1024
"""The kernels' compile-time shapes (Qwen3.5-397B-A17B per rank at TP=4); anything else
takes the Triton paths."""

BLOCK_T = 16
"""Tokens per unit: the MFMA N dimension."""

WIDE = os.environ.get("SEED_MOE_HIP_WIDE", "0") not in ("0", "", "false", "False")
"""`SEED_MOE_HIP_WIDE=1` (off by default; needs `SEED_MOE_HIP=1`): the HIP kernels serve
prefill and mixed widths too, and split their work evenly over programs (`MOE_FLAT`).

Measured on one MI300A rank (layer 30, real weights and router, `bench_moe_widths.py`), the
unchanged HIP kernels already beat `prefill_moe` (SEED_PREFILL_GROUPED_MOE) at every width:
644 vs 1364 us/layer at T=256, 681 vs 1430 at 512, 768 vs 1542 at 1024, 1337 vs 1779 at 2048.
What held them back is the work split: a gate_up unit (one expert, <= 16 tokens) is ~0.2 ms
of decode + MFMA on one CU, and with ~120-250 units against a 228-program grid the default
split leaves some programs a whole unit (or two) while others hold half of one. `MOE_FLAT`
lays the (unit, item) pairs out flat and gives every program the same count (see
`moe_segment` in `moe_hip.hip`). With this flag, `MAX_TOKENS` defaults to 4096 and
`Model._routed_fused` prefers these kernels over `prefill_moe` up to it."""

MAX_TOKENS = int(os.environ.get("SEED_MOE_HIP_MAX_TOKENS", "4096" if WIDE else "256"))
"""Largest call the HIP kernels take. Default 256 (decode and mixed-step widths), larger
calls keep the Triton paths; 4096 under `SEED_MOE_HIP_WIDE`."""

BT32 = os.environ.get("SEED_MOE_HIP_BT32", "0") not in ("0", "", "false", "False")
"""`SEED_MOE_HIP_BT32=1` (off by default): calls of at least `BT32_MIN_TOKENS` tokens build
32-token units and run the `NB = 2` kernels (`moe_hip.hip`), which decode each weight word
once for two 16-token MFMA groups. Why: at prefill widths most local experts get 17-60 tokens
(T=1024: 1.56 units per expert, T=2048: 2.65), and every 16-token unit streams and decodes the
expert's full 6.7 MB again. Per token the arithmetic is the `NB = 1` kernels', so the output
is bit-identical; only the work list (unit size) changes. The choice depends only on the
call's token count, so every rank (and every replay of a captured shape) takes the same one."""

BT32_MIN_TOKENS = int(os.environ.get("SEED_MOE_HIP_BT32_MIN_TOKENS", "512"))
"""Smallest call (tokens) that takes the 32-token units under `SEED_MOE_HIP_BT32`."""


def block_t(tokens: int) -> int:
    """Tokens per work unit for a call of `tokens` tokens (see `BT32`)."""
    return 2 * BLOCK_T if BT32 and tokens >= BT32_MIN_TOKENS else BLOCK_T


GRID = int(os.environ.get("SEED_MOE_HIP_GRID", "0"))
"""Programs per launch; 0 = one per CU (a program is 8 waves at ~167 VGPRs, so one fits)."""

KNOBS = {
    "MOE_DEPTH": os.environ.get("SEED_MOE_HIP_DEPTH", "4"),
    "MOE_GU_PT": os.environ.get("SEED_MOE_HIP_GU_PT", "1"),
    "MOE_DN_RW": os.environ.get("SEED_MOE_HIP_DN_RW", "4"),
    "MOE_DN_TPW": os.environ.get("SEED_MOE_HIP_DN_TPW", "1"),
    "MOE_NT": os.environ.get("SEED_MOE_HIP_NT", "0"),
    "MOE_FLAT": os.environ.get("SEED_MOE_HIP_FLAT", "1" if WIDE else "0"),
    "MOE_DEPTH2": os.environ.get("SEED_MOE_HIP_DEPTH2", os.environ.get("SEED_MOE_HIP_DEPTH", "4")),
}
"""Compile-time knobs of `moe_hip.hip` (ring depth, tile shapes, nontemporal weight loads)."""

_BINDING = r"""
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

extern "C" int moe_hip_gate_up(const void*, const void*, const void*, const void*, const void*,
                               const void*, void*, int, int, void*);
extern "C" int moe_hip_down(const void*, const void*, const void*, const void*, const void*,
                            const void*, const void*, void*, int, void*);
extern "C" int moe_hip_gate_up2(const void*, const void*, const void*, const void*, const void*,
                                const void*, void*, int, int, void*);
extern "C" int moe_hip_down2(const void*, const void*, const void*, const void*, const void*,
                             const void*, const void*, void*, int, void*);
extern "C" int moe_hip_gate_up_item_rows();
extern "C" int moe_hip_down_item_rows();

#define P(x) reinterpret_cast<void*>(x)

static void check(int err, const char* what) {
    if (err) throw std::runtime_error(std::string(what) + ": hip error " + std::to_string(err));
}

void gate_up(int64_t x, int64_t wq, int64_t ws, int64_t sorted, int64_t unit, int64_t n_units,
             int64_t inter, int64_t top_k, int64_t grid, int64_t stream) {
    check(moe_hip_gate_up(P(x), P(wq), P(ws), P(sorted), P(unit), P(n_units), P(inter),
                          (int)top_k, (int)grid, P(stream)), "moe_hip_gate_up");
}

void down(int64_t inter, int64_t wq, int64_t ws, int64_t a_weight, int64_t sorted, int64_t unit,
          int64_t n_units, int64_t y, int64_t grid, int64_t stream) {
    check(moe_hip_down(P(inter), P(wq), P(ws), P(a_weight), P(sorted), P(unit), P(n_units), P(y),
                       (int)grid, P(stream)), "moe_hip_down");
}

void gate_up2(int64_t x, int64_t wq, int64_t ws, int64_t sorted, int64_t unit, int64_t n_units,
              int64_t inter, int64_t top_k, int64_t grid, int64_t stream) {
    check(moe_hip_gate_up2(P(x), P(wq), P(ws), P(sorted), P(unit), P(n_units), P(inter),
                           (int)top_k, (int)grid, P(stream)), "moe_hip_gate_up2");
}

void down2(int64_t inter, int64_t wq, int64_t ws, int64_t a_weight, int64_t sorted, int64_t unit,
           int64_t n_units, int64_t y, int64_t grid, int64_t stream) {
    check(moe_hip_down2(P(inter), P(wq), P(ws), P(a_weight), P(sorted), P(unit), P(n_units), P(y),
                        (int)grid, P(stream)), "moe_hip_down2");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_up", &gate_up);
    m.def("down", &down);
    m.def("gate_up2", &gate_up2);
    m.def("down2", &down2);
    m.def("gate_up_item_rows", &moe_hip_gate_up_item_rows);
    m.def("down_item_rows", &moe_hip_down_item_rows);
}
"""

_ext = None


def enabled() -> bool:
    return os.environ.get("SEED_MOE_HIP", "0") == "1"


def available(device: torch.device) -> bool:
    """`SEED_MOE_HIP=1` on a gfx942 device (the kernels use gfx9 MFMA and `v_perm_b32`)."""
    if not enabled() or device.type != "cuda" or torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx942"


def supports(experts: dict) -> bool:
    """The compiled shapes and the folded-scale precondition. One host sync (the scale scan):
    call once at load time, never inside graph capture."""
    gate_up, down = experts["gate_up"], experts["down"]
    return (
        gate_up.shape[1:] == (2 * INTER, HIDDEN // 2)
        and down.shape[1:] == (HIDDEN, INTER // 2)
        and all(
            experts[k].is_contiguous() for k in ("gate_up", "gate_up_scale", "down", "down_scale")
        )
        and mxfp4_gemv.bw_scales_ok(experts)
    )


def build_dir() -> Path:
    """Content-addressed build directory: `SEED_HIP_CACHE_DIR` (put it on /path/to/scratch so a boot
    reuses the .so; the 4 TP ranks share one build through torch's file lock), else torch's
    default extension root. The hash covers the sources, the knobs and the torch/HIP build."""
    h = hashlib.sha256()
    for part in (HIP_SRC.read_bytes(), _BINDING.encode(), repr(sorted(KNOBS.items())).encode()):
        h.update(part)
    h.update(f"{torch.__version__} {torch.version.hip}".encode())
    root = os.environ.get("SEED_HIP_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "torch_extensions"
    return base / f"seed_moe_hip_{h.hexdigest()[:16]}"


def ext():
    """Compile (first call per cache directory, ~1 min) or load the extension."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        d = build_dir()
        d.mkdir(parents=True, exist_ok=True)
        cpp = d / "moe_hip_binding.cpp"
        hip = d / "moe_hip_kernels.hip"
        for path, text in ((cpp, _BINDING), (hip, HIP_SRC.read_text())):
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
        prev_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"  # one code object, not torch's arch list
        try:
            _ext = load(
                name=d.name,
                sources=[str(cpp), str(hip)],
                build_directory=str(d),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", *(f"-D{k}={v}" for k, v in KNOBS.items())],
                with_cuda=True,
                verbose=os.environ.get("SEED_MOE_HIP_VERBOSE") == "1",
            )
        finally:
            if prev_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = prev_arch
    return _ext


def grid(device: torch.device) -> int:
    return GRID if GRID > 0 else torch.cuda.get_device_properties(device).multi_processor_count


def prep(
    x: torch.Tensor,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
) -> tuple[mxfp4_gemv.BwPlan, torch.Tensor]:
    """The work list alone (`bw_prep`): `(plan, a_expert)`. Split out of `_fused_moe_hip_core` so
    a caller can run it on a side stream, concurrently with work that does not depend on routing
    (`SEED_MOE_PREP_FORK`, `model.py`'s `_moe_hip_route_fused`), and pass the result back in as
    `prepared`. Reads only `x.shape`, not `x`'s contents."""
    a_expert, a_weight = routing
    lo, hi = expert_range
    plan = mxfp4_gemv.BwPlan(
        x.shape[0], top_k, x.shape[1], hi - lo, x.device, x.dtype, block_t=block_t(x.shape[0]),
        permute_x=False,
    )
    mxfp4_gemv.bw_prep(x, a_expert, a_weight, expert_range, plan)
    return plan, a_expert


def _fused_moe_hip_core(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    out: torch.Tensor | None,
    inter: torch.Tensor | None,
    y: torch.Tensor | None,
    caller: str,
    prepared: tuple[mxfp4_gemv.BwPlan, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Everything `fused_moe_hip`/`fused_moe_hip_glued` share: validate, prep, gate_up, down.
    Returns `(y, out)`; the caller still owns the combine launch (`bw_combine` or
    `bw_combine_glue`), since that is the one step the two entry points differ on.
    `prepared`: `prep`'s result, already computed (same `x`/`routing`/`top_k`/`expert_range`),
    to skip the prep launch here (`SEED_MOE_PREP_FORK`)."""
    a_expert, a_weight = routing
    tokens, hidden = x.shape
    assignments = tokens * top_k
    if hidden != HIDDEN or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError(f"{caller}: x must be contiguous bf16 [T, {HIDDEN}]")
    if a_expert.shape != (assignments,) or a_weight.shape != (assignments,):
        raise ValueError(f"routing must be two [{assignments}] tensors")
    if not (a_expert.is_contiguous() and a_weight.is_contiguous()):
        raise ValueError(f"{caller}: routing tensors must be contiguous")
    # the down kernel reads fp32 routing weights (the model's `top_w` already is)
    a_weight_f32 = a_weight if a_weight.dtype == torch.float32 else a_weight.float()
    device = x.device
    if inter is None or inter.shape != (assignments, INTER):
        inter = torch.empty(assignments, INTER, dtype=x.dtype, device=device)
    if y is None or y.shape != (assignments, HIDDEN):
        y = torch.empty(assignments, HIDDEN, dtype=x.dtype, device=device)
    if out is None:
        out = torch.empty(tokens, HIDDEN, dtype=x.dtype, device=device)
    if prepared is not None:
        plan, _ = prepared
    else:
        plan, _ = prep(x, routing, top_k, expert_range)
    m, g = ext(), grid(device)
    stream = torch.cuda.current_stream(device).cuda_stream
    wide = plan.block_t == 2 * BLOCK_T
    (m.gate_up2 if wide else m.gate_up)(
        x.data_ptr(),
        experts["gate_up"].data_ptr(),
        experts["gate_up_scale"].data_ptr(),
        plan.sorted.data_ptr(),
        plan.unit.data_ptr(),
        plan.n_units.data_ptr(),
        inter.data_ptr(),
        top_k,
        g,
        stream,
    )
    (m.down2 if wide else m.down)(
        inter.data_ptr(),
        experts["down"].data_ptr(),
        experts["down_scale"].data_ptr(),
        a_weight_f32.data_ptr(),
        plan.sorted.data_ptr(),
        plan.unit.data_ptr(),
        plan.n_units.data_ptr(),
        y.data_ptr(),
        g,
        stream,
    )
    return y, out


def fused_moe_hip(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    prepared: tuple[mxfp4_gemv.BwPlan, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Routed-expert output `[T, hidden]`. Same contract as `mxfp4_gemv.fused_moe_bw`: static
    launch shapes, no host sync, capturable; `inter`/`y` are optional `[T * top_k, *]` scratch
    that needs no zeroing. Callers must have checked `supports(experts)` once at load time.
    Four launches: prep (Triton, routing only), gate_up + SiLU, down + routing weight, combine.
    `prepared`: see `_fused_moe_hip_core`."""
    a_expert, a_weight = routing
    y, out = _fused_moe_hip_core(
        x, experts, routing, top_k, expert_range, out, inter, y, "fused_moe_hip", prepared
    )
    mxfp4_gemv.bw_combine(y, a_expert, a_weight, expert_range, top_k, out)
    return out


def fused_moe_hip_glued(
    x: torch.Tensor,
    experts: dict,
    routing: tuple[torch.Tensor, torch.Tensor],
    top_k: int,
    expert_range: tuple[int, int],
    shared: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    inter: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    prepared: tuple[mxfp4_gemv.BwPlan, torch.Tensor] | None = None,
) -> torch.Tensor:
    """`fused_moe_hip`, but the combine launch also folds in the shared-expert glue
    (`mxfp4_gemv.bw_combine_glue` in place of `bw_combine`): `out = routed_sum +
    sigmoid(gate) * shared`, one launch instead of `bw_combine` plus a `sigmoid`, a multiply,
    and an add at the `Model.moe` call site. Same contract as `fused_moe_hip` otherwise --
    static shapes, no host sync, capturable -- with `shared` `[T, hidden]` (the already-computed
    `swiglu_mlp` shared-expert output) and `gate` `[T, >=1]` (the shared-expert gate's raw,
    pre-sigmoid logit; only column 0 is read). Part of `SEED_MOE_ROUTE_FUSED`
    (`model.py`'s `_moe_hip_route_fused`). `prepared`: see `_fused_moe_hip_core`."""
    a_expert, a_weight = routing
    y, out = _fused_moe_hip_core(
        x, experts, routing, top_k, expert_range, out, inter, y, "fused_moe_hip_glued", prepared
    )
    mxfp4_gemv.bw_combine_glue(y, a_expert, a_weight, expert_range, top_k, shared, gate, out)
    return out
