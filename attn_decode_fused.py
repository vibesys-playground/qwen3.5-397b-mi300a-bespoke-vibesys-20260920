"""Fused Triton kernel for the full-attention decode step's RoPE + KV-cache write.

`graph_decode.attn_decode_static` (the captured/replayed decode path) used to do this in five
separate torch ops per tensor: `apply_rope`'s `cat`/`rotate_half`/two-multiply chain for q and
for k (4 dispatches), then, for k and v, a *read* of `pool[...][write_rows]` (a gather), a
`torch.where(keep, new, old)` select, and a *write* back to `pool[...][write_rows]` (a
scatter) -- 3 more dispatches each, 6 total, only to leave an inactive replay row's cache
untouched (see that function's docstring for why the no-op-via-select is needed: an inactive
row still indexes a real, previously-committed cache position under `--enable-graph-capture`,
so skipping the op outright is not an option and the write must be *conditional in the store*,
not skipped in Python). None of this needs to be a torch op: RoPE is an elementwise function of
q/k and the per-row cos/sin, and "write only where active" is exactly what a Triton `tl.store`
`mask` argument already does for free, without ever reading the cache's prior value back to
compute a `where` against it.

One kernel replaces all of it: for every row, apply RoPE to every q head (writing a fresh,
still-unpadded `[B, nq, 1, head_dim]` tensor for the attention kernel to read) and to every kv
head (writing straight into the paged pool at `write_rows[row]`, gated on `active[row]`, no
intermediate tensor and no gather-then-select), and copy v into the pool under the same gate,
no rope. `active` is optional so the identical kernel serves the eager batched-decode path
(`Model.attn_decode`, which only ever processes real rows and has no "keep" concept) with
`HAS_ACTIVE=False`, one launch that always writes.

Idea: this is the RoPE+KV-cache-write fusion TensorRT-LLM's and vLLM's attention backends both
ship (apply rotary embeddings and append to the paged KV cache in one kernel, rather than a
separate rotary kernel followed by a separate cache-append kernel) -- cited for the *shape* of
the fusion, no code taken from either; the mask-gated store that makes an inactive captured-
replay row a true no-op is specific to this codebase's paged-KV-under-CUDA-graph design
(`paged-kv-design.md` Stage 1, `graph_decode.py`'s module docstring).

Behind `SEED_FUSE_GLUE` (default off) at the two call sites in `graph_decode.py` and
`model.py`; this module itself has no flag of its own; `available` mirrors `deltanet_fused`'s
and `rmsnorm_fused`'s contract (CPU/`TRITON_INTERPRET=1` for tests, an accelerator to ship).

`rmsnorm_rope_and_kv_write` (`SEED_ATTN_ROPE_KV_FUSED`, `graph_decode.attn_decode_static`
only) goes one step further: it also folds in the `q_norm`/`k_norm` RMSNorm that runs ahead of
RoPE in every case, removing the two `rmsnorm_fused` launches those normally cost. It is a
second kernel rather than a mode of `rope_and_kv_write`, because its callers pass q/k/v in
their natural pre-`.transpose(1, 2)` layout (before norm, there is no reason to transpose at
all) and it needs the norm weights and `eps` besides. See its own docstring for the flag's
relationship to `SEED_FUSE_GLUE`, which is orthogonal (fuses RoPE+KV-write only, no norm) and
off in production.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    HAVE_TRITON = False


def available(device: torch.device) -> bool:
    """Same contract as `deltanet_fused.available`/`rmsnorm_fused.available`."""
    if not HAVE_TRITON:
        return False
    return device.type == "cuda"


if HAVE_TRITON:

    @triton.jit
    def _rope_kv_write_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        cos_ptr,
        sin_ptr,
        active_ptr,
        write_rows_ptr,
        out_q_ptr,
        pool_k_ptr,
        pool_v_ptr,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sc_b,
        soq_b,
        soq_h,
        spk_row,
        spk_h,
        spv_row,
        spv_h,
        head_dim,
        rot_dim,
        NQ: tl.constexpr,
        NKV: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per row: RoPE every q/k head, scatter-write k/v into the paged pool.

        `pair`/`sign` reproduce `model.rotate_half` + `model.apply_rope` exactly: for channel
        `d < rot_dim // 2`, `rotate_half(x)[d] = -x[d + rot_dim // 2]`; for `d >= rot_dim // 2`,
        `rotate_half(x)[d] = x[d - rot_dim // 2]`. Channels `>= rot_dim` (a partial-rotary
        config) pass through unrotated, matching `apply_rope`'s `rest` slice.
        """
        row = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        live = d < head_dim
        half = rot_dim // 2
        rot_live = d < rot_dim
        pair = tl.where(d < half, d + half, d - half)
        sign = tl.where(d < half, -1.0, 1.0)

        cos = tl.load(cos_ptr + row * sc_b + d, mask=rot_live, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + row * sc_b + d, mask=rot_live, other=0.0).to(tl.float32)

        store_mask = live
        if HAS_ACTIVE:
            act = tl.load(active_ptr + row) != 0
            store_mask = live & act
        dest_row = tl.load(write_rows_ptr + row)

        for h in range(NQ):
            x = tl.load(q_ptr + row * sq_b + h * sq_h + d, mask=live, other=0.0).to(tl.float32)
            xp = tl.load(q_ptr + row * sq_b + h * sq_h + pair, mask=rot_live, other=0.0).to(
                tl.float32
            )
            roped = tl.where(rot_live, x * cos + sign * xp * sin, x)
            tl.store(
                out_q_ptr + row * soq_b + h * soq_h + d,
                roped.to(out_q_ptr.dtype.element_ty),
                mask=live,
            )

        for h in range(NKV):
            x = tl.load(k_ptr + row * sk_b + h * sk_h + d, mask=live, other=0.0).to(tl.float32)
            xp = tl.load(k_ptr + row * sk_b + h * sk_h + pair, mask=rot_live, other=0.0).to(
                tl.float32
            )
            roped = tl.where(rot_live, x * cos + sign * xp * sin, x)
            v = tl.load(v_ptr + row * sv_b + h * sv_h + d, mask=live, other=0.0)
            tl.store(
                pool_k_ptr + dest_row * spk_row + h * spk_h + d,
                roped.to(pool_k_ptr.dtype.element_ty),
                mask=store_mask,
            )
            tl.store(pool_v_ptr + dest_row * spv_row + h * spv_h + d, v, mask=store_mask)

    @triton.jit
    def _rmsnorm_rope_kv_write_kernel(  # noqa: PLR0915 - one fused expression, not a procedure
        q_ptr,
        k_ptr,
        v_ptr,
        qn_ptr,
        kn_ptr,
        cos_ptr,
        sin_ptr,
        active_ptr,
        write_rows_ptr,
        out_q_ptr,
        pool_k_ptr,
        pool_v_ptr,
        sq_b,
        sq_h,
        sk_b,
        sk_h,
        sv_b,
        sv_h,
        sc_b,
        soq_b,
        soq_h,
        spk_row,
        spk_h,
        spv_row,
        spv_h,
        head_dim,
        rot_dim,
        eps,
        NQ: tl.constexpr,
        NKV: tl.constexpr,
        HAS_ACTIVE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per row: RMSNorm every q/k head, then RoPE, then scatter-write roped-k
        and v into the paged pool.

        `model.rmsnorm` (`q_norm`/`k_norm`) folded in ahead of `_rope_kv_write_kernel`'s own
        RoPE math, at the same two rounding points running the two kernels back to back would
        have: once at the norm's output (`model.rmsnorm`'s own single rounding, `(1 + w)`
        convention), once more at the final store -- see `rmsnorm_rope_and_kv_write`'s
        docstring. `pair`/`sign` reproduce `model.rotate_half` + `model.apply_rope` exactly, as
        in `_rope_kv_write_kernel`.
        """
        row = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        live = d < head_dim
        half = rot_dim // 2
        rot_live = d < rot_dim
        pair = tl.where(d < half, d + half, d - half)
        sign = tl.where(d < half, -1.0, 1.0)
        dt_q = out_q_ptr.dtype.element_ty
        dt_k = pool_k_ptr.dtype.element_ty

        cos = tl.load(cos_ptr + row * sc_b + d, mask=rot_live, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + row * sc_b + d, mask=rot_live, other=0.0).to(tl.float32)

        store_mask = live
        if HAS_ACTIVE:
            act = tl.load(active_ptr + row) != 0
            store_mask = live & act
        dest_row = tl.load(write_rows_ptr + row)

        for h in range(NQ):
            raw = tl.load(q_ptr + row * sq_b + h * sq_h + d, mask=live, other=0.0).to(tl.float32)
            scale = tl.rsqrt(tl.sum(raw * raw) / head_dim + eps)
            w = tl.load(qn_ptr + d, mask=live, other=0.0).to(tl.float32)
            xn = (raw * scale * (1.0 + w)).to(dt_q).to(tl.float32)
            raw_p = tl.load(q_ptr + row * sq_b + h * sq_h + pair, mask=rot_live, other=0.0).to(
                tl.float32
            )
            wp = tl.load(qn_ptr + pair, mask=rot_live, other=0.0).to(tl.float32)
            xnp = (raw_p * scale * (1.0 + wp)).to(dt_q).to(tl.float32)
            roped = tl.where(rot_live, xn * cos + sign * xnp * sin, xn)
            tl.store(
                out_q_ptr + row * soq_b + h * soq_h + d,
                roped.to(dt_q),
                mask=live,
            )

        for h in range(NKV):
            raw = tl.load(k_ptr + row * sk_b + h * sk_h + d, mask=live, other=0.0).to(tl.float32)
            scale = tl.rsqrt(tl.sum(raw * raw) / head_dim + eps)
            w = tl.load(kn_ptr + d, mask=live, other=0.0).to(tl.float32)
            xn = (raw * scale * (1.0 + w)).to(dt_k).to(tl.float32)
            raw_p = tl.load(k_ptr + row * sk_b + h * sk_h + pair, mask=rot_live, other=0.0).to(
                tl.float32
            )
            wp = tl.load(kn_ptr + pair, mask=rot_live, other=0.0).to(tl.float32)
            xnp = (raw_p * scale * (1.0 + wp)).to(dt_k).to(tl.float32)
            roped = tl.where(rot_live, xn * cos + sign * xnp * sin, xn)
            v = tl.load(v_ptr + row * sv_b + h * sv_h + d, mask=live, other=0.0)
            tl.store(
                pool_k_ptr + dest_row * spk_row + h * spk_h + d,
                roped.to(dt_k),
                mask=store_mask,
            )
            tl.store(pool_v_ptr + dest_row * spv_row + h * spv_h + d, v, mask=store_mask)


def rope_and_kv_write(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rot_dim: int,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    write_rows: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """RoPE q and k, then write roped-k and v into the paged pool at `write_rows`.

    `q`/`k`/`v` are `[B, heads, 1, head_dim]` (as `attn_decode_static`/`Model.attn_decode`
    produce after their own `.transpose(1, 2)`) or `[B, heads, head_dim]`; the singleton time
    axis, if present, costs nothing (it does not change the flat memory layout of a
    freshly-projected, contiguous tensor). `cos`/`sin` are `[B, rot_dim]`. `pool_k`/`pool_v`
    are `[rows, kv_heads, head_dim]`. `write_rows` is `[B]`. `active`, when given, gates the
    pool write only (see the kernel's docstring); `q`'s RoPE and its output are never gated,
    matching `attn_decode_static`'s existing behavior of computing every row's `q` regardless
    of `active` (only the *cache* has a "leave alone" requirement).

    Returns the roped `q`, `[B, heads, 1, head_dim]`, contiguous.
    """
    if q.dim() == 4:
        q, k, v = (
            q.reshape(*q.shape[:2], q.shape[-1]),
            k.reshape(*k.shape[:2], k.shape[-1]),
            v.reshape(*v.shape[:2], v.shape[-1]),
        )
    b, nq, head_dim = q.shape
    nkv = k.shape[1]
    if k.shape != (b, nkv, head_dim) or v.shape != (b, nkv, head_dim):
        raise ValueError(
            f"k/v shape mismatch: k {tuple(k.shape)}, v {tuple(v.shape)}, expected nkv={nkv}"
        )
    if cos.shape != (b, rot_dim) or sin.shape != (b, rot_dim):
        raise ValueError(f"cos/sin must be [{b}, {rot_dim}], got {tuple(cos.shape)}")
    if write_rows.shape != (b,):
        raise ValueError(f"write_rows must be [{b}], got {tuple(write_rows.shape)}")

    out_q = torch.empty(b, nq, head_dim, dtype=q.dtype, device=q.device)
    active_arg = active if active is not None else q  # unused dummy pointer when HAS_ACTIVE=False
    _rope_kv_write_kernel[(b,)](
        q,
        k,
        v,
        cos,
        sin,
        active_arg,
        write_rows,
        out_q,
        pool_k,
        pool_v,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(0),
        out_q.stride(0),
        out_q.stride(1),
        pool_k.stride(0),
        pool_k.stride(1),
        pool_v.stride(0),
        pool_v.stride(1),
        head_dim,
        rot_dim,
        NQ=nq,
        NKV=nkv,
        HAS_ACTIVE=active is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out_q.unsqueeze(2)


def rmsnorm_rope_and_kv_write(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rot_dim: int,
    eps: float,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    write_rows: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """`SEED_ATTN_ROPE_KV_FUSED=1`: q/k RMSNorm, RoPE, and the KV-cache write, in one kernel.

    `attn_decode_static` used to run this as three separate stages: two `rmsnorm_fused`
    launches (`model.rmsnorm` on q and k), then either the unfused `apply_rope` plus a
    gather/`torch.where`/scatter per tensor (`SEED_FUSE_GLUE=0`, production today) or
    `rope_and_kv_write` above (`SEED_FUSE_GLUE=1`, RoPE + KV-write only, no norm). This
    collapses all of it -- 2 norm launches plus however many RoPE/KV-write ops the other flag
    leaves -- into one launch that takes q/k straight off `q_proj`/`k_proj`, before any norm.

    Precision: preserves the same two rounding points running `rmsnorm_fused` into
    `rope_and_kv_write` back to back would have, just with no intermediate HBM round trip for
    q/k between them. Per head: normalize in fp32 (`model.rmsnorm`'s `(1 + w)` convention,
    `tl.sum` order rather than `torch.sum`'s -- the same reduction-order caveat every other
    kernel in this campaign carries), round once to the storage dtype (matching
    `model.rmsnorm`'s own single rounding, which is what `apply_rope`/`rope_and_kv_write` would
    read as their *input*), then do the RoPE math in fp32 on that rounded value and round again
    at the store (matching `rope_and_kv_write`'s own single final rounding). Not bit-identical
    to the unfused `SEED_FUSE_GLUE=0` production path, whose `apply_rope` runs its multiply-add
    chain directly on bf16 tensors (extra intermediate torch-op roundings `rope_and_kv_write`
    already does not reproduce either); held to the same `BF16_TOL` as this module's other
    fused kernel in `seed_tests/test_attn_decode_fused.py`.

    `q`/`k`/`v` are `[B, 1, heads, head_dim]` or `[B, heads, head_dim]` -- the natural,
    pre-`.transpose(1, 2)` shape straight off `F.linear(...).view(...)`, so this needs no
    transpose either, unlike `rope_and_kv_write`'s expected post-transpose layout. `q_norm`/
    `k_norm` are `[head_dim]`. `cos`/`sin` are `[B, rot_dim]`. `pool_k`/`pool_v` are
    `[rows, kv_heads, head_dim]`. `write_rows` is `[B]`. `active`, when given, gates the pool
    write only, matching `rope_and_kv_write`.

    Returns the normed-and-roped `q`, `[B, heads, 1, head_dim]`, contiguous.
    """
    if q.dim() == 4:
        if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
            raise ValueError("q/k/v must be [B, 1, heads, head_dim] when 4-dimensional")
        q, k, v = (
            q.reshape(q.shape[0], q.shape[2], q.shape[3]),
            k.reshape(k.shape[0], k.shape[2], k.shape[3]),
            v.reshape(v.shape[0], v.shape[2], v.shape[3]),
        )
    b, nq, head_dim = q.shape
    nkv = k.shape[1]
    if k.shape != (b, nkv, head_dim) or v.shape != (b, nkv, head_dim):
        raise ValueError(
            f"k/v shape mismatch: k {tuple(k.shape)}, v {tuple(v.shape)}, expected nkv={nkv}"
        )
    if q_norm.shape != (head_dim,) or k_norm.shape != (head_dim,):
        raise ValueError(f"q_norm/k_norm must be [{head_dim}]")
    if cos.shape != (b, rot_dim) or sin.shape != (b, rot_dim):
        raise ValueError(f"cos/sin must be [{b}, {rot_dim}], got {tuple(cos.shape)}")
    if write_rows.shape != (b,):
        raise ValueError(f"write_rows must be [{b}], got {tuple(write_rows.shape)}")

    out_q = torch.empty(b, nq, head_dim, dtype=q.dtype, device=q.device)
    active_arg = active if active is not None else q  # unused dummy pointer when HAS_ACTIVE=False
    _rmsnorm_rope_kv_write_kernel[(b,)](
        q,
        k,
        v,
        q_norm,
        k_norm,
        cos,
        sin,
        active_arg,
        write_rows,
        out_q,
        pool_k,
        pool_v,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos.stride(0),
        out_q.stride(0),
        out_q.stride(1),
        pool_k.stride(0),
        pool_k.stride(1),
        pool_v.stride(0),
        pool_v.stride(1),
        head_dim,
        rot_dim,
        eps,
        NQ=nq,
        NKV=nkv,
        HAS_ACTIVE=active is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out_q.unsqueeze(2)
