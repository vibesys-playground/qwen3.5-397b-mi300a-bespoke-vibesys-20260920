"""Chunked (WY-representation) Gated DeltaNet prefill, with the intra-chunk triangular
solve as a Triton kernel instead of `torch.linalg.solve_triangular`.

Why this exists. `model.delta_rule_chunked` already turns the O(T) sequential per-token
recurrence into O(T/C) sequential chunk updates plus, per chunk, one C x C unit-lower-
triangular solve (C = `DELTA_CHUNK`, 64). That solve is what `model.Model.delta_rule`'s
docstring reports as ~10x *slower* than the plain recurrence on this hardware (13-token
prompt: ~130s chunked vs ~13.6s recurrent) when spelled with `torch.linalg.solve_triangular`,
so production always took the recurrent path. This module keeps every other operation in
`delta_rule_chunked` (which are ordinary batched matmuls -- decay, the intra-chunk attention
matrix, the state carry -- and are not the bottleneck) and replaces only the triangular solve
with a hand-written forward-substitution kernel, following the fused-kernel style of
`deltanet_fused.py`: one Triton launch instead of a library call whose ROCm cost profile is
opaque from here.

The math (see `model.py`'s module docstring for the recurrence this mirrors). Per chunk, with
q,k l2-normalized and inputs promoted to fp32, decay[i,j] = exp(G_i - G_j) for j <= i (G the
within-chunk cumulative log-gate) else 0, and

    A = tril(k_beta @ k^T * decay, -1)                         # strictly lower triangular
    (I + A) [U W] = [v_beta, k_beta * exp(G)]                  # <- the solve this module speeds up

`U` is the chunk's own delta contribution and `W` maps the incoming state onto the same basis;
solving for both in one call is why the right-hand side is `[v_beta, k_beta*exp(G)]`
concatenated on the last axis. Because `A` is strictly lower triangular of size C, `I + A` is
unit lower triangular and nilpotent (`A^C == 0`), so forward substitution is exact, not an
iterative approximation: row i of the solution is `rhs_i - sum_{j<i} A[i,j] * x_j`, which needs
only rows already computed. That is the whole kernel below (`_lower_tri_solve_kernel`):
C sequential row-updates, each an elementwise multiply-reduce over the whole (C, W) tile,
executed once per (batch, head, chunk) system in parallel and once per column tile of the
right-hand side (`BLOCK_W` wide) in parallel, so the *sequential depth* the solve adds is C
(64), not T, regardless of how many chunks or heads there are: all of them are independent
programs on the grid. That is the source of the speedup this module exists to capture: the
recurrent path's serial depth is T; the chunked path's is C (per-chunk solve) + T/C (the
state carry across chunks, `_state_carry` below) -- for T=4096, C=64 that is 64 + 64 = 128
sequential steps in place of 4096.

Column-blocking (`BLOCK_W`) rather than one wide tile per system keeps each program's working
set to `C x C` (the shared `A` tile) plus `C x BLOCK_W` (the running solution), independent of
the solve's total width (`v_dim + k_dim`, 256 at this model's real per-head shape): the two
right-hand sides are solved by whichever column tiles they land in, with no cross-tile
dependence, because `A` acts only on the chunk axis, never on the right-hand-side's columns.

`C` (the chunk length) must be a power of two: `tl.arange` requires it, and the one value this
module is ever run at in production is `model.DELTA_CHUNK` (64). `_chunk` below pads an
arbitrary sequence length out to a multiple of `chunk`, exactly as `model._delta_rule_chunks`
does; the last chunk's padding is rows of zero k/v/beta/g, which zero their rows and columns of
`A` and their rows of `rhs`, so they solve to zero and contribute nothing to the outputs or the
carried state (`model._delta_rule_chunks`'s docstring covers why).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

DELTA_CHUNK = 64
"""Default chunk length. Must stay a power of two (`tl.arange` requires it). Production
callers (`model.Model.delta_rule`) pass `model.DELTA_CHUNK` explicitly so there is one source
of truth for the value actually deployed; this default is for direct callers and tests."""

L2_EPS = 1e-6
"""Same epsilon as `model.l2norm`, which every delta-rule spelling uses."""


def available(device: torch.device) -> bool:
    """True when the chunked Triton solve can run for tensors on `device`.

    Defaults OFF (`SEED_CHUNKED_DELTANET` defaults to `"0"`): rocBLAS's batched-matmul
    heuristics make this path ~100x slower than the recurrence on 4x MI300A (gfx942), so
    leaving it on by default would regress prefill in an integration build. It stays here,
    and `SEED_CHUNKED_DELTANET=1` re-enables it, for direct callers, the parity tests, and
    future rocBLAS-heuristic or shape-specific fixes. CPU tensors (and the CPU-side
    correctness tests, which drive the kernel directly under `TRITON_INTERPRET=1`) always take
    the torch fallback (`_lower_tri_solve_torch`) regardless of the flag.
    """
    if not HAVE_TRITON or os.environ.get("SEED_CHUNKED_DELTANET", "0") == "0":
        return False
    return device.type == "cuda"


def l2norm(x: torch.Tensor, eps: float = L2_EPS) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _delta_rule_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same promotion as `model._delta_rule_inputs`, duplicated to keep this module import-free
    of `model` (which imports this module for the prefill dispatch; importing back would be a
    cycle)."""
    return (
        l2norm(q.float()) * q.shape[-1] ** -0.5,
        l2norm(k.float()),
        v.float(),
        beta.float(),
    )


def _chunk(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, ...]:
    """Split [1,T,H,d] inputs into [1,H,NC,chunk,d], right-padding the last chunk with zeros.

    Identical in effect to `model._delta_rule_chunks` (duplicated for the same import-cycle
    reason as `_delta_rule_inputs`). `g` comes back as the within-chunk cumulative sum.
    """
    t = q.shape[1]
    pad = -t % chunk
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    g, beta = g.float().transpose(1, 2), beta.transpose(1, 2)
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    nc = (t + pad) // chunk
    q, k, v = (x.unflatten(2, (nc, chunk)) for x in (q, k, v))
    return q, k, v, g.unflatten(2, (nc, chunk)).cumsum(-1), beta.unflatten(2, (nc, chunk))


if HAVE_TRITON:

    @triton.jit
    def _lower_tri_solve_kernel(
        a_ptr,
        rhs_ptr,
        out_ptr,
        sa_n,
        sa_r,
        sa_c,
        sr_n,
        sr_r,
        sr_c,
        so_n,
        so_r,
        so_c,
        w,
        C: tl.constexpr,
        BS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Solve `(I + A) X = rhs` for one (batch*head*chunk) system and one column tile.

        `A` [C, C] is strictly lower triangular (caller guarantees `A[i, j] == 0` for `j >=
        i`), so `I + A` is unit lower triangular and forward substitution is exact. Rather than
        one flat C-row forward substitution, this splits `C` into `C // BS` blocks of `BS` rows
        and follows the standard block-forward-substitution identity: block row `r`'s equation
        is

        A first version of this kernel did the flat, single-block C-row substitution (one
        `[C, C]`/`[C, BLOCK_W]` tile carried and rewritten across all C sequential steps).
        Measured on gfx942 at this model's real per-rank shape (T=512, one DeltaNet layer): 7.07
        s versus 62 ms for `delta_rule_recurrent` over the same call, i.e. slower than the
        recurrence it was meant to replace, not faster. The block form below is the fix that
        was measured to work (see this change's perf report for the full table); the likely
        mechanism -- a resident `[64, up to 128]` fp32 tile kept live and rewritten across 64
        dependent steps outrunning the register file and spilling every step -- is a plausible
        explanation for *why* the flat form was slow, not something separately profiled, so it
        is offered as that and not as a measured root cause.

            (I + A_rr) X_r = rhs_r - sum_{c < r} A_rc @ X_c

        The sum over already-solved blocks is `BS`-sized matmuls (`tl.dot`, which targets the
        tensor-core / MFMA path and is not the part that was slow), leaving only the BS x BS
        diagonal block's own unit-lower-triangular system to forward-substitute row by row --
        `BS` (16) small steps on a `[BS, BLOCK_W]` tile instead of `C` (64) steps on a `[C,
        BLOCK_W]` one. Total sequential depth is `nb` block-steps (4) each doing `BS - 1` (15)
        inner steps, so the same C-ish total step count as the flat version, but every tile
        touched by an individual step is `(BS/C)^2` the size, which is where the difference
        actually comes from.

        `BS` must divide `C`, and both must be powers of two (`tl.arange` requires it, and the
        inner solve's masking trick, mirrored from the single-block version, needs a power of
        two range). The caller (`lower_tri_solve`) picks `BS = min(16, C)`.

        Earlier blocks' solved rows (`X_c`, `c < r`) are read back from `out_ptr` rather than
        kept in a Python-level list of live tiles: Triton's real (non-interpreter) compiler
        does not support mutating a Python list of tensors inside a jitted function the way the
        reference interpreter tolerates, and re-reading a just-written output is well-defined
        here because one grid program executes its unrolled instruction sequence in order --
        the store for block `c` is issued strictly before the load for block `r > c`.
        """
        pid_n = tl.program_id(0)
        pid_w = tl.program_id(1)
        nb: tl.constexpr = C // BS
        local = tl.arange(0, BS)
        cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
        col_live = cols < w

        for r in tl.static_range(0, nb):
            row_idx = r * BS + local
            acc = tl.load(
                rhs_ptr + pid_n * sr_n + row_idx[:, None] * sr_r + cols[None, :] * sr_c,
                mask=col_live[None, :],
                other=0.0,
            ).to(tl.float32)
            for c in tl.static_range(0, r):
                col_idx = c * BS + local
                a_rc = tl.load(
                    a_ptr + pid_n * sa_n + row_idx[:, None] * sa_r + col_idx[None, :] * sa_c
                ).to(tl.float32)
                x_c = tl.load(
                    out_ptr + pid_n * so_n + col_idx[:, None] * so_r + cols[None, :] * so_c,
                    mask=col_live[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc -= tl.dot(a_rc, x_c)

            a_rr = tl.load(
                a_ptr + pid_n * sa_n + row_idx[:, None] * sa_r + row_idx[None, :] * sa_c
            ).to(tl.float32)
            xr = acc
            for i in tl.static_range(1, BS):
                is_i = local == i
                a_row = tl.sum(tl.where(is_i[:, None], a_rr, 0.0), axis=0)  # A_rr[i, :]
                correction = tl.sum(a_row[:, None] * xr, axis=0)
                cur = tl.sum(tl.where(is_i[:, None], xr, 0.0), axis=0)  # xr[i, :], still acc
                xr = tl.where(is_i[:, None], (cur - correction)[None, :], xr)

            tl.store(
                out_ptr + pid_n * so_n + row_idx[:, None] * so_r + cols[None, :] * so_c,
                xr,
                mask=col_live[None, :],
            )


def _lower_tri_solve_torch(a: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Forward substitution in plain torch: the CPU fallback, and the algorithm the kernel
    above follows (this is the reference the kernel is checked against, not an approximation
    of it -- both compute the same row-by-row recurrence, just vectorized differently)."""
    n, c, w = rhs.shape
    x = rhs.clone()
    for i in range(1, c):
        correction = torch.einsum("nj,njw->nw", a[:, i, :i], x[:, :i].clone())
        x[:, i] = rhs[:, i] - correction
    return x


def lower_tri_solve(a: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Batched solve of `(I + a) x = rhs`, `a` strictly lower triangular in its last two axes.

    `a`: [..., C, C] with `a[..., i, j] == 0` for `j >= i`. `rhs`/the result: [..., C, W]. `C`
    must be a power of two. Dispatches to the Triton kernel on an accelerator (or under
    `TRITON_INTERPRET=1`, which the CPU tests use directly), else the torch fallback.
    """
    *batch, c, c2 = a.shape
    if c != c2:
        raise ValueError(f"a must be square in its last two axes, got {tuple(a.shape)}")
    if c & (c - 1):
        raise ValueError(f"chunk length must be a power of two, got {c}")
    w = rhs.shape[-1]
    if rhs.shape[:-1] != (*batch, c):
        raise ValueError(f"rhs shape {tuple(rhs.shape)} does not match a's batch/chunk {(*batch, c)}")

    n = 1
    for d in batch:
        n *= d
    a2 = a.reshape(n, c, c).contiguous()
    rhs2 = rhs.reshape(n, c, w).contiguous()

    use_triton = HAVE_TRITON and (
        a.device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"
    )
    if use_triton:
        out = torch.empty_like(rhs2)
        block_w = min(triton.next_power_of_2(max(w, 1)), 128)
        block_s = min(16, c)
        grid = (max(n, 1), triton.cdiv(w, block_w))
        _lower_tri_solve_kernel[grid](
            a2,
            rhs2,
            out,
            *a2.stride(),
            *rhs2.stride(),
            *out.stride(),
            w,
            C=c,
            BS=block_s,
            BLOCK_W=block_w,
        )
    else:
        out = _lower_tri_solve_torch(a2, rhs2)
    return out.reshape(*batch, c, w)


DEFAULT_TUNE_CACHE = Path.home() / ".cache" / "qwen35-bespoke"
"""Same cache root as `blas_tune.py`'s, a distinct file within it (`tunableop_chunked_...`):
this module never imports or calls into `blas_tune.py` (no coupling to the decode-shape
tuning it does), but persisting to the same directory means one `~/.cache` to know about."""

_TUNED_STATE_CARRY_SHAPES: set[tuple] = set()
"""Per-process memo of (heads, chunk, k_dim, v_dim, device index) already sent through
`_ensure_state_carry_tuned`, so a long-lived server pays the one-time search once per shape,
not once per prefill call."""


def _tunable_cache_file(device: torch.device) -> Path:
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return DEFAULT_TUNE_CACHE / f"tunableop_chunked_deltanet_{arch}_%d.csv"


def _ensure_state_carry_tuned(step, state: torch.Tensor) -> None:
    """Warm PyTorch's TunableOp for the state-carry loop's batched-matmul shapes, once.

    `step`'s five batched matmuls (`v_new = w_i @ state`, `q_i @ k_i^T`, `(q_i*decay) @
    state`, `intra @ v_new`, the transposed `k_i @ v_new` state update) all carry a *head*
    batch dimension (this model's real per-rank shape: 16 heads at TP=4) over small per-matrix
    tiles (chunk x k_dim or k_dim x v_dim, both <= 128). Measured on gfx942: rocBLAS/hipBLASLt's
    default heuristic for that shape of batched GEMM is not just suboptimal (the scale
    `blas_tune.py` documents for decode's skinny dense projections) but catastrophic -- the
    state-carry loop measured 5.58 s for 8 chunks (T=512) against 2.4 ms once TunableOp had a
    tuned solution for these exact shapes, roughly 2,300x. That gap, not the intra-chunk solve,
    is what made earlier chunked-prefill spellings (this module's first version, and
    `model.delta_rule_chunked`'s `torch.linalg.solve_triangular`) measure far slower than the
    recurrence: solve_triangular and this module's Triton kernel were never the dominant cost,
    the untuned batched matmuls around them were.

    This mirrors `blas_tune.tune`'s technique (enable TunableOp, run the real op once so the
    search sees the real strides -- a slice of a larger tensor is a different solution key than
    a freshly allocated one of the same shape -- then freeze tuning and persist the result) but
    is self-contained here rather than a change to `blas_tune.py`: that module only knows about
    the decode step's `F.linear` dense-projection shapes, this is a different operator
    (batched `@`) at prefill. `step(state.clone(), 0)`'s result is discarded; it exists to make
    TunableOp measure the real sequence of ops once, not to advance any real state.
    """
    # Checked directly against the device (not through `available`, which also gates on
    # `SEED_CHUNKED_DELTANET` and is what tests monkeypatch to force the dispatch in
    # `model.Model.delta_rule` without a real accelerator): TunableOp is a ROCm/CUDA-only
    # facility, so this has to stay off on CPU regardless of that flag.
    if state.device.type != "cuda" or os.environ.get("SEED_CHUNKED_DELTANET_TUNE", "1") == "0":
        return
    key = (*state.shape, state.device.index)
    if key in _TUNED_STATE_CARRY_SHAPES:
        return
    _TUNED_STATE_CARRY_SHAPES.add(key)

    tunable = torch.cuda.tunable
    path = _tunable_cache_file(state.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(path), True)
    tunable.enable(True)
    tunable.set_max_tuning_duration(10)
    resolved = Path(tunable.get_filename())
    if resolved.exists():
        tunable.read_file()

    started = time.perf_counter()
    tunable.tuning_enable(True)
    try:
        step(state.clone(), 0)
    finally:
        tunable.tuning_enable(False)
    tunable.write_file()
    print(
        f"deltanet_chunked: tuned state-carry matmuls for shape {key} in "
        f"{time.perf_counter() - started:.1f}s, cached at {resolved}",
        flush=True,
    )


def delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    rec: torch.Tensor,
    chunk: int = DELTA_CHUNK,
) -> torch.Tensor:
    """Gated delta rule, chunked WY-representation form. Same contract as
    `model.delta_rule_recurrent`/`model.delta_rule_chunked`: q,k,v [1,T,H,d], g,beta [1,T,H],
    `rec` [1,H,k_dim,v_dim] advanced in place, returns fp32 [1,T,H,v_dim].

    Identical math to `model.delta_rule_chunked`; the intra-chunk solve goes through
    `lower_tri_solve` (this module's Triton kernel) instead of `torch.linalg.solve_triangular`,
    and the state-carry loop is preceded by a one-time-per-shape TunableOp warmup
    (`_ensure_state_carry_tuned`) -- see its docstring for why that, not the solve, was the
    actual bottleneck. See the module docstring for the solve kernel's derivation.
    """
    t, dv = q.shape[1], v.shape[-1]
    chunk = min(chunk, t) if t else chunk
    chunk = 1 << (chunk - 1).bit_length()  # round up to a power of two if t < chunk shrank it
    q, k, v, beta = _delta_rule_inputs(q, k, v, beta)
    q, k, v, g, beta = _chunk(q, k, v, g, beta, chunk)

    # decay[i,j] = exp(G_i - G_j) for j <= i else 0, G the within-chunk cumulative gate (<= 0).
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    a = ((k_beta @ k.transpose(-1, -2)) * decay).tril(-1)
    rhs = torch.cat([v_beta, k_beta * g.exp()[..., None]], dim=-1)
    sol = lower_tri_solve(a, rhs)
    u, w = sol.split([dv, k.shape[-1]], dim=-1)

    def step(state: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # `.contiguous()` on the three sliced matmul operands: `q[:, :, i]` etc. are views into
        # a [1, H, NC, chunk, d] parent, so their *stride* (not shape) depends on NC = T //
        # chunk, which differs by prompt length even though chunk, H and d never do. A
        # TunableOp-selected GEMM solution is keyed on stride as well as shape (see
        # `_ensure_state_carry_tuned`'s docstring), so leaving these as raw slices meant a
        # solution tuned at one T's NC silently missed the cache -- and fell back to the
        # catastrophic default heuristic -- at every other T. Copying to a canonical,
        # NC-independent layout here is what makes one tuning pass (any T) cover every T.
        q_i, k_i, g_i = q[:, :, i].contiguous(), k[:, :, i].contiguous(), g[:, :, i]
        w_i = w[:, :, i].contiguous()
        v_new = u[:, :, i] - w_i @ state
        intra = (q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]
        out_i = (q_i * g_i[..., None].exp()) @ state + intra @ v_new
        tail = g_i[..., -1:]  # total decay across the chunk
        new_state = (
            state * tail[..., None].exp()
            + (k_i * (tail - g_i)[..., None].exp()).transpose(-1, -2) @ v_new
        )
        return out_i, new_state

    _ensure_state_carry_tuned(step, rec)

    state, out = rec.clone(), torch.empty_like(v)
    for i in range(q.shape[2]):  # sequential across chunks only; T/chunk steps, not T
        out[:, :, i], state = step(state, i)
    rec.copy_(state)
    return out.flatten(2, 3)[:, :, :t].transpose(1, 2).contiguous()
