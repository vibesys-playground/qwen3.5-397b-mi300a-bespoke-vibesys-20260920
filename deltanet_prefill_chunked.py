"""Chunked gated-delta-rule prefill as two Triton kernels on MFMA (`SEED_DELTANET_PREFILL_CHUNKED=1`).

Idea (published): the chunkwise-parallel WY form of the (gated) delta rule, Yang et al.,
"Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (NeurIPS 2024)
and "Gated Delta Networks" (ICLR 2025). The math is exactly `model.delta_rule_chunked`'s (the
torch reference in this bundle); this module only changes where it runs.

Why. `deltanet_fused.fused_recurrent_prefill` walks all `T` positions serially in each of 64
programs per sequence (a quarter of the CUs), so its time is `T` x one step's latency
(loads, four cross-warp reductions) however large `T` is. The chunked form splits the work:

1. `_dn_chunk_prep_kernel`, one program per (chunk, head), all in parallel: everything that
   does not depend on the carried state. Per chunk of `C` positions (l2-normalized q, k; G
   the within-chunk cumulative log-decay):

       A     = tril(k_beta k^T * exp(G_i - G_j), -1)                  (tl.dot)
       (I + A) [U W] = [v_beta, k_beta * exp(G)]                      (forward substitution)
       intra = tril(q k^T * exp(G_i - G_j))                           (tl.dot)
       QG = q * exp(G),  KG = k * exp(G_last - G),  gl = exp(G_last)

   The solve is `C` serial row updates, but per chunk, and all chunks run at once.
2. `_dn_chunk_state_kernel`, one program per (sequence, head, value block): the only serial
   part, `T / C` steps of small `tl.dot`s with the state tile `S` in registers:

       v_new = U - W S;   out = QG S + intra v_new;   S = gl * S + KG^T v_new

So the serial depth is `C` (prep) + `T / C` (state) instead of `T`: 32 + 64 at T=2048.

Contract: same as `deltanet_fused.fused_recurrent_prefill` (raw q/k before l2-norm, g the
per-token log-decay, beta already sigmoid'd, `initial_state` fp32 `[n_seqs, H, K, V]`
advanced in place, fp32 `[T, H, V]` output). `cu_seqlens` is read on the host once per call
(one small device-to-host copy) to build the chunk table: prefill is not graph-captured.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False

ENABLED = os.environ.get("SEED_DELTANET_PREFILL_CHUNKED", "0") not in ("0", "", "false", "False")
"""`SEED_DELTANET_PREFILL_CHUNKED=1`: the model's prefill DeltaNet calls (`Model.delta_rule`
for T > 1, `Model._delta_rule_packed`) of at least `MIN_ROWS` rows run here instead of the
per-position recurrence (they pass `chunked=use_chunked(rows)` to
`deltanet_fused.fused_recurrent_prefill`)."""

MIN_ROWS = int(os.environ.get("SEED_DELTANET_PREFILL_CHUNKED_MIN", "128"))
CHUNK = int(os.environ.get("SEED_DELTANET_PREFILL_CHUNK", "16"))
"""Positions per chunk (a power of two, >= 16 for `tl.dot`). At 64 the state kernel spills
(512 VGPRs + scratch) on gfx942. On MI300A at T=2048 (BLOCK_V 16, num_stages 2) 16 runs
0.72 ms/layer vs 0.76 at 32. The serial depth is C + T/C."""
BLOCK_V = int(os.environ.get("SEED_DELTANET_PREFILL_CHUNKED_BLOCK_V", "16"))
"""Value columns per state-kernel program: 16 gives 8 programs per head (128 per sequence at
16 local heads). On MI300A at T=2048 (C=32, num_stages 1) it runs 0.80 ms/layer vs 1.08 at 32; 64 spills."""
_STAGES = os.environ.get("SEED_DELTANET_PREFILL_CHUNKED_STAGES", "")
"""Triton `num_stages` for both kernels; unset picks 2 at `block_v <= 16`, else 1. At 2 the
AMD pipeliner double-buffers the state kernel's per-chunk tiles in LDS, which at C=32, K=128,
BV=32 needs 69,632 B, over gfx942's 65,536 B. At BV=16 it fits and is ~3% faster than 1."""
L2_EPS = 1e-6
"""Same epsilon as `model.l2norm` / `deltanet_fused.L2_EPS`."""


def use_chunked(rows: int) -> bool:
    """Whether a prefill call of `rows` packed rows should take the chunked kernels. Never
    while a CUDA graph is being captured: the chunk table is built on the host."""
    if not (ENABLED and HAVE_TRITON and rows >= MIN_ROWS):
        return False
    return not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


if HAVE_TRITON:
    _EPS = tl.constexpr(L2_EPS)

    @triton.jit
    def _dn_chunk_prep_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        chunk_start_ptr,
        chunk_len_ptr,
        w_ptr,
        u_ptr,
        qg_ptr,
        kg_ptr,
        intra_ptr,
        gl_ptr,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sg_b,
        sbeta_b,
        heads,
        scale,
        C: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        QK_REP: tl.constexpr = 1,
        UNIFORM: tl.constexpr = False,
    ):
        """Everything per (chunk, head) that does not depend on the carried state. `q`/`k`
        may hold the key heads only (`QK_REP` value heads per key head, GQA-style), in place
        of a `repeat_interleave`d copy. `UNIFORM`: chunk `c` starts at `c * C` and is full, so
        the chunk table is not read (its pointers may be any tensor)."""
        c = tl.program_id(0)
        head = tl.program_id(1)
        if UNIFORM:
            start = c * C
            length = C
        else:
            start = tl.load(chunk_start_ptr + c)
            length = tl.load(chunk_len_ptr + c)
        r = tl.arange(0, C)
        ok = r < length
        kc = tl.arange(0, K)
        vc = tl.arange(0, V)
        row = start + r

        q = tl.load(
            q_ptr + row[:, None] * sq_b + (head // QK_REP) * sq_h + kc[None, :],
            mask=ok[:, None],
            other=0.0,
        )
        k = tl.load(
            k_ptr + row[:, None] * sk_b + (head // QK_REP) * sk_h + kc[None, :],
            mask=ok[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + row[:, None] * sv_b + head * sv_h + vc[None, :], mask=ok[:, None], other=0.0
        )
        g = tl.load(g_ptr + row * sg_b + head, mask=ok, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + row * sbeta_b + head, mask=ok, other=0.0).to(tl.float32)
        q = q.to(tl.float32)
        k = k.to(tl.float32)
        v = v.to(tl.float32)
        q = q * (tl.rsqrt(tl.sum(q * q, axis=1) + _EPS) * scale)[:, None]
        k = k * tl.rsqrt(tl.sum(k * k, axis=1) + _EPS)[:, None]

        big_g = tl.cumsum(g, axis=0)  # padded rows add 0, so the tail value is the last real one
        g_last = tl.sum(tl.where(r == C - 1, big_g, 0.0), axis=0)
        lower = r[:, None] >= r[None, :]
        diff = tl.where(lower, big_g[:, None] - big_g[None, :], 0.0)
        decay = tl.where(lower, tl.exp(diff), 0.0)

        kb = k * beta[:, None]
        a = tl.dot(kb, tl.trans(k), input_precision="ieee") * decay
        a = tl.where(r[:, None] > r[None, :], a, 0.0)
        intra = tl.dot(q, tl.trans(k), input_precision="ieee") * decay

        # Forward substitution, (I + A) X = rhs, A strictly lower: row i is final once rows
        # < i are. `at` is A transposed so column i of it (A's row i) comes out indexed like
        # X's rows.
        at = tl.trans(a)
        xu = v * beta[:, None]
        xw = kb * tl.exp(big_g)[:, None]
        for i in range(1, C):
            ai = tl.sum(tl.where(r[None, :] == i, at, 0.0), axis=1)  # A[i, :] over rows j
            cu = tl.sum(ai[:, None] * xu, axis=0)
            cw = tl.sum(ai[:, None] * xw, axis=0)
            xu = tl.where(r[:, None] == i, xu - cu[None, :], xu)
            xw = tl.where(r[:, None] == i, xw - cw[None, :], xw)

        base = (c * heads + head).to(tl.int64) * C
        tl.store(w_ptr + (base + r[:, None]) * K + kc[None, :], xw)
        tl.store(u_ptr + (base + r[:, None]) * V + vc[None, :], xu)
        tl.store(qg_ptr + (base + r[:, None]) * K + kc[None, :], q * tl.exp(big_g)[:, None])
        kg = k * tl.exp(tl.where(ok, g_last - big_g, 0.0))[:, None]
        tl.store(kg_ptr + (base + r[:, None]) * K + kc[None, :], kg)
        tl.store(intra_ptr + (base + r[:, None]) * C + r[None, :], intra)
        tl.store(gl_ptr + c * heads + head, tl.exp(g_last))

    @triton.jit
    def _dn_chunk_state_kernel(
        w_ptr,
        u_ptr,
        qg_ptr,
        kg_ptr,
        intra_ptr,
        gl_ptr,
        chunk_start_ptr,
        chunk_len_ptr,
        seq_first_ptr,
        seq_count_ptr,
        rec_ptr,
        o_ptr,
        lane_ptr,
        active_ptr,
        sr_s,
        sr_h,
        sr_k,
        so_b,
        so_h,
        heads,
        per,
        C: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        UNIFORM: tl.constexpr = False,
        LANES: tl.constexpr = False,
        SEQ_TABLE: tl.constexpr = False,
    ):
        """The serial chunk-to-chunk state carry for one (sequence, head, value block).

        `SEQ_TABLE` (with `UNIFORM`, `SEED_PREFILL_PACK`): chunks are the uniform full-width
        grid, but each sequence's run of chunks comes from `seq_first`/`seq_count`, so several
        sequences can share a `[rows, width]` token matrix. A sequence with `count == 0` (an
        unused segment slot) neither reads nor writes the state.

        `UNIFORM`: every sequence is `per` full chunks, so the chunk table is not read.
        `LANES`: `rec` is the whole lane pool; sequence `seq` reads lane `lane_ptr[seq]` and
        writes it back only when `active_ptr[seq]` is nonzero (padding rows share a filler
        lane and must not write it). Each program owns a disjoint state tile, so reading and
        writing the pool in place has no cross-program hazard."""
        pid = tl.program_id(0)
        seq, head = pid // heads, pid % heads
        r = tl.arange(0, C)
        kc = tl.arange(0, K)
        vc = tl.program_id(1) * BV + tl.arange(0, BV)
        if LANES:
            lane = tl.load(lane_ptr + seq).to(tl.int64)
        else:
            lane = seq.to(tl.int64)
        rec_off = rec_ptr + lane * sr_s + head * sr_h + kc[:, None] * sr_k + vc[None, :]
        if SEQ_TABLE:
            first = tl.load(seq_first_ptr + seq)
            count = tl.load(seq_count_ptr + seq)
            s = tl.load(rec_off, mask=count > 0, other=0.0).to(tl.float32)
        elif UNIFORM:
            s = tl.load(rec_off).to(tl.float32)
            first = seq * per
            count = per
        else:
            s = tl.load(rec_off).to(tl.float32)
            first = tl.load(seq_first_ptr + seq)
            count = tl.load(seq_count_ptr + seq)
        for c in range(first, first + count):
            base = (c * heads + head).to(tl.int64) * C
            w = tl.load(w_ptr + (base + r[:, None]) * K + kc[None, :])
            qg = tl.load(qg_ptr + (base + r[:, None]) * K + kc[None, :])
            kg = tl.load(kg_ptr + (base + r[:, None]) * K + kc[None, :])
            u = tl.load(u_ptr + (base + r[:, None]) * V + vc[None, :])
            intra = tl.load(intra_ptr + (base + r[:, None]) * C + r[None, :])
            gl = tl.load(gl_ptr + c * heads + head)
            if UNIFORM:
                start = c * C
                length = C
            else:
                start = tl.load(chunk_start_ptr + c)
                length = tl.load(chunk_len_ptr + c)
            v_new = u - tl.dot(w, s, input_precision="ieee")
            out = tl.dot(qg, s, input_precision="ieee") + tl.dot(
                intra, v_new, input_precision="ieee"
            )
            tl.store(
                o_ptr + (start + r)[:, None] * so_b + head * so_h + vc[None, :],
                out,
                mask=(r < length)[:, None],
            )
            s = s * gl + tl.dot(tl.trans(kg), v_new, input_precision="ieee")
        if SEQ_TABLE:
            if count > 0:
                if tl.load(active_ptr + seq) != 0:
                    tl.store(rec_off, s)
        elif LANES:
            if tl.load(active_ptr + seq) != 0:
                tl.store(rec_off, s)
        else:
            tl.store(rec_off, s)


def chunked_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    chunk: int | None = None,
    block_v: int | None = None,
    num_stages: int | None = None,
    uniform: int | None = None,
    lanes: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    table_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`deltanet_fused.fused_recurrent_prefill`'s contract, chunked. See the module docstring.

    `uniform=W`: every sequence is exactly `W` rows (`rows == n_seqs * W`, `W` a multiple of
    the chunk) and `cu_seqlens` is ignored. The chunk table is then built with device ops
    only, no host read and no host-to-device copy, so the call can be captured in a graph
    (`graph_prefill`'s `[rows, width]` layout, `SEED_PREFILL_GRAPH_DN_CHUNKED`).

    `q`/`k` may carry the key heads only (`[rows, key_heads, K]`, `key_heads` dividing `v`'s
    heads): value head `h` then reads key head `h // (heads // key_heads)`, the same pairing
    as `repeat_interleave` on the head axis, without the widened copy.

    `lanes` `[n_seqs]` int and `active` `[n_seqs]` bool (both or neither; `SEED_PREFILL_DN_STATE_INPLACE`):
    `initial_state` is then the whole lane pool `[L, H, K, V]`; sequence `s` resumes lane
    `lanes[s]` in place and writes it back only if `active[s]`. Bit-identical to gathering
    `pool[lanes]`, running on the copy, and scattering `where(active, new, old)` back, without
    those four launches. `table_free` (with `uniform`): the kernels derive the uniform chunk
    table from their program ids instead of reading four device-built tables."""
    if not HAVE_TRITON:  # pragma: no cover - callers gate on `use_chunked`
        raise RuntimeError("triton is required for the chunked DeltaNet prefill")
    c = chunk or CHUNK
    rows, heads, v_dim = v.shape
    k_dim = q.shape[-1]
    bv = min(block_v or BLOCK_V, v_dim)
    stages = num_stages or (int(_STAGES) if _STAGES else (2 if bv <= 16 else 1))
    for name, dim in (("chunk", c), ("k_dim", k_dim), ("v_dim", v_dim), ("block_v", bv)):
        if dim < 16 or dim & (dim - 1):
            raise ValueError(f"chunked prefill needs {name} a power of two >= 16, got {dim}")
    if v_dim % bv:
        raise ValueError(f"v_dim {v_dim} must be a multiple of block_v {bv}")
    if q.shape[1] != k.shape[1] or heads % q.shape[1]:
        raise ValueError("q/k heads must match and divide v's (key heads, GQA-style)")
    for name, x in (("q", q), ("k", k), ("v", v)):
        if x.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous along its last axis")
    if g.stride(-1) != 1 or beta.stride(-1) != 1:
        raise ValueError("g and beta must be contiguous along their last axis")
    if (lanes is None) != (active is None):
        raise ValueError("lanes and active go together")
    if lanes is not None and uniform is None:
        raise ValueError("lanes (in-place lane state) needs uniform")
    if table_free and uniform is None:
        raise ValueError("table_free needs uniform")
    n_seqs = initial_state.shape[0] if lanes is None else lanes.shape[0]
    if uniform is not None:
        if uniform % c or rows != n_seqs * uniform:
            raise ValueError(f"uniform={uniform}: need rows == n_seqs * W and W % chunk == 0")
        per = uniform // c
        nc = n_seqs * per
        dev = q.device
        if table_free:
            return _launch(
                q, k, v, g, beta, initial_state, None, nc, c, bv, stages, per, lanes, active
            )
        i32 = {"dtype": torch.int32, "device": dev}
        tables = (
            torch.arange(0, rows, c, **i32),
            torch.full((nc,), c, **i32),
            torch.arange(0, nc, per, **i32),
            torch.full((n_seqs,), per, **i32),
        )
        return _launch(
            q, k, v, g, beta, initial_state, tables, nc, c, bv, stages, per, lanes, active
        )
    bounds = [0, rows] if cu_seqlens is None else [int(b) for b in cu_seqlens.tolist()]
    if len(bounds) != n_seqs + 1:
        raise ValueError(f"cu_seqlens must be [{n_seqs + 1}]")

    starts: list[int] = []
    lengths: list[int] = []
    first: list[int] = []
    count: list[int] = []
    for s in range(n_seqs):
        lo, hi = bounds[s], bounds[s + 1]
        first.append(len(starts))
        for c0 in range(lo, hi, c):
            starts.append(c0)
            lengths.append(min(c, hi - c0))
        count.append(len(starts) - first[-1])
    nc = len(starts)
    if nc == 0:
        return torch.empty((rows, heads, v_dim), dtype=torch.float32, device=v.device), initial_state
    dev = q.device
    as_i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)  # noqa: E731
    tables = (as_i32(starts), as_i32(lengths), as_i32(first), as_i32(count))
    return _launch(q, k, v, g, beta, initial_state, tables, nc, c, bv, stages)


def chunked_prefill_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    seq_first: torch.Tensor,
    seq_count: torch.Tensor,
    lanes: torch.Tensor,
    active: torch.Tensor,
    *,
    chunk: int | None = None,
    block_v: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Several sequences packed into one token stream (`SEED_PREFILL_PACK`, `graph_prefill`).

    The stream's `rows` tokens form `rows / chunk` full chunks. Sequence `s` owns chunks
    `seq_first[s] .. seq_first[s] + seq_count[s] - 1` (so it starts on a chunk boundary), resumes
    lane `lanes[s]` of the state pool `pool` `[L, H, K, V]` in place, and writes it back only if
    `active[s]`; `seq_count[s] == 0` marks an unused slot. Tokens past a sequence's length inside
    its last chunk must be identity steps (`beta = g = 0`), as in the per-row layout. Every
    table is a device tensor of fixed size, so the call is capturable and every replay may
    pack differently. Per sequence, the chunks and their arithmetic are the ones
    `chunked_prefill(..., uniform=W, lanes=...)` runs for the same tokens at a row start,
    so the results are bit-identical to that per-row layout. Output rows no sequence owns
    are zero."""
    if not HAVE_TRITON:  # pragma: no cover - callers gate on availability
        raise RuntimeError("triton is required for the chunked DeltaNet prefill")
    c = chunk or CHUNK
    rows, heads, v_dim = v.shape
    if rows % c:
        raise ValueError(f"packed prefill needs rows % chunk == 0, got {rows} % {c}")
    for name, x in (("seq_count", seq_count), ("lanes", lanes), ("active", active)):
        if x.shape != seq_first.shape:
            raise ValueError(f"{name} must match seq_first's shape")
    bv = min(block_v or BLOCK_V, v_dim)
    stages = num_stages or (int(_STAGES) if _STAGES else (2 if bv <= 16 else 1))
    return _launch(
        q,
        k,
        v,
        g,
        beta,
        pool,
        None,
        rows // c,
        c,
        bv,
        stages,
        0,
        lanes,
        active,
        seq_table=(seq_first, seq_count),
    )


def _launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    nc: int,
    c: int,
    bv: int,
    stages: int,
    per: int = 0,
    lanes: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    seq_table: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both kernels over a built chunk table `(chunk_start, chunk_len, seq_first, seq_count)`,
    or, with `tables=None`, the uniform table (`per` full chunks per sequence) derived in-kernel.
    `seq_table` (with `tables=None` and `lanes`): uniform chunks, per-sequence chunk runs from
    `(seq_first, seq_count)` (`chunked_prefill_packed`)."""
    rows, heads, v_dim = v.shape
    k_dim = q.shape[-1]
    n_seqs = initial_state.shape[0] if lanes is None else lanes.shape[0]
    uniform = tables is None
    dev = q.device
    if uniform:
        chunk_start = chunk_len = seq_first = seq_count = v  # never read (UNIFORM)
        if seq_table is not None:
            seq_first, seq_count = seq_table
    else:
        chunk_start, chunk_len, seq_first, seq_count = tables
    has_lanes = lanes is not None
    lane_arg = lanes if has_lanes else v  # never read without LANES
    active_arg = (active.view(torch.int8) if active.dtype == torch.bool else active) if has_lanes else v
    # Packed: chunks no segment claims are never stored by the state kernel; zero them so the
    # rows after the last segment stay finite through the norm, MoE routing and all-reduce.
    alloc = torch.zeros if seq_table is not None else torch.empty
    out = alloc((rows, heads, v_dim), dtype=torch.float32, device=dev)
    f32 = {"dtype": torch.float32, "device": dev}
    w = torch.empty(nc, heads, c, k_dim, **f32)
    qg = torch.empty(nc, heads, c, k_dim, **f32)
    kg = torch.empty(nc, heads, c, k_dim, **f32)
    u = torch.empty(nc, heads, c, v_dim, **f32)
    intra = torch.empty(nc, heads, c, c, **f32)
    gl = torch.empty(nc, heads, **f32)

    _dn_chunk_prep_kernel[(nc, heads)](
        q,
        k,
        v,
        g,
        beta,
        chunk_start,
        chunk_len,
        w,
        u,
        qg,
        kg,
        intra,
        gl,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        g.stride(0),
        beta.stride(0),
        heads,
        k_dim**-0.5,
        C=c,
        K=k_dim,
        V=v_dim,
        QK_REP=heads // q.shape[1],
        UNIFORM=uniform,
        num_warps=4,
        num_stages=stages,
    )
    _dn_chunk_state_kernel[(n_seqs * heads, v_dim // bv)](
        w,
        u,
        qg,
        kg,
        intra,
        gl,
        chunk_start,
        chunk_len,
        seq_first,
        seq_count,
        initial_state,
        out,
        lane_arg,
        active_arg,
        *initial_state.stride()[:3],
        *out.stride()[:2],
        heads,
        per,
        C=c,
        K=k_dim,
        V=v_dim,
        BV=bv,
        UNIFORM=uniform,
        LANES=has_lanes,
        SEQ_TABLE=seq_table is not None,
        num_warps=4,
        num_stages=stages,
    )
    return out, initial_state
