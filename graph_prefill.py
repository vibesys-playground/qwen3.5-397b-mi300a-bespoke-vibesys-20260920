"""Opt-in HIP/CUDA graph capture for small prefill calls (`SEED_PREFILL_GRAPHS=1` plus
`--enable-graph-capture`).

**Why.** An eager prefill call costs ~120 ms before its first token matters (a 5-token call
measured 124-128 ms at C48, then ~1 ms per token). That fixed cost is host dispatch: the eager
forward issues several thousand `aten`/Triton launches over 60 layers plus 120 collectives,
and a small call's device work is a fraction of that (`TP_BYTELUT_BOTTLENECK_2026-09-22.md`
measured the same effect on the eager decode step: 186 ms host issue vs 122 ms device). Every
prefill step stalls every decoding lane, so the fixed cost lands on TPOT. Replaying a captured
graph removes the host issue time, leaving the device time.

**Shape.** A call is a set of `(lane, ids, start)` chunks. A captured shape is `rows x width`:
up to `rows` chunks of up to `width` tokens each, laid out as a `[rows, width]` token matrix
(the MTP verify step's layout, `graph_mtp.py`, with per-row lengths). The shape ladder is
`SEED_PREFILL_GRAPH_SHAPES` (default `DEFAULT_SHAPES`); a call replays from the smallest-area
shape that holds it and falls back to the eager path otherwise. Everything that varies per call
is a buffer *value*: tokens, positions, lengths, lane ids, block tables.

**Padding is exact, not approximate.** A padded column (index `>= length` in its row) and a
padding row (beyond the call's chunk count) must leave every lane's state as the eager call
would:

- *KV.* Its write goes to `block_pool.RESERVED_BLOCK` rows with the value already there
  (`torch.where(keep, new, old)`, the decode graph's trick). A real query never reads it: a
  real column `c` attends to positions `<= start + c`, all real.
- *DeltaNet recurrence.* `beta = 0` and `g = 0` on padded columns make the step the identity
  (`S * exp(0) + k * 0 = S` bit for bit), so the recurrent state after the row is the state
  after its real tokens only; no rollback pass is needed (contrast `graph_mtp`'s verify
  rollback, which must truncate at a length unknown until after the forward). Padding rows'
  state writes are additionally masked by `active`.
- *Conv state.* The new conv window is gathered from `[old window, raw inputs]` at the row's
  own length (`graph_mtp.deltanet_verify_rollback_static`'s gather), not taken from the end.
- *Positions.* A padded column computes at its row's last real position (clamped), so RoPE
  and the block-table gather never index past `max_seq` or past the lane's reserved blocks.
- *Logits.* Only each real row's last real column goes through the LM head.

**Attention.** Each `(row, column)` query is issued as one decode-attention query at its own
position (`decode_attention_paged`, the fused paged kernel decode graphs already replay), with
its row's block table: causal masking is the kernel's own `pos` bound, and the KV of the whole
chunk is written before the attention reads it. `verify_attention_paged` is not used: it reads
`block_valid.max()` on the host, which a capture forbids.

**TP.** `replayable` reads only the broadcast `calls`, so every rank takes the same path;
`prepare` agrees the verdict across ranks like `GraphDecodeRunner.agree`; blocks are reserved on
rank 0 by the scheduler before the command (`Model.grow_lane` is a capacity check), exactly as
for decode.

**MTP** (`model.mtp`, `SEED_MTP_SERVE`). The eager MTP prefill also extends the lane's MTP
KV (`mtp.prefill_cache`) and seeds `Model.hidden_scratch` (the next draft round's seed), so the
captured step ends with the same tail (`_mtp_tail`): the final raw residual rows (all-gathered
first when the residual is row-sharded), `mtp.cache_target_rows` over the real columns (the
same KV rows the target writes, padded columns masked), and each active row's residual at its
last real column scattered into `hidden_scratch[lane]`. `_check` validates the tail like the
DeltaNet state: against the uncaptured step and against the eager MTP prefill.

    <python-with-torch> -m pytest seed_tests/test_graph_prefill.py -q -o addopts=
"""

from __future__ import annotations

import glob
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import attn_decode_fused
import block_pool
import decode_glue
import deltanet_fused
import deltanet_prefill_chunked
import deltanet_prefill_glue
import graph_decode
import mem_timeline
import mtp as mtp_mod
import torch
import torch.nn.functional as F
from graph_decode import CaptureBackend, log
from model import (
    Model,
    apply_rope,
    copy_from_host,
    decode_attention_paged,
    delta_rule_recurrent,
    gated_rmsnorm,
    in_proj_sizes,
    rmsnorm,
)

PREFILL_GRAPHS = os.environ.get("SEED_PREFILL_GRAPHS", "0") not in ("0", "", "false", "False")
"""Capture small prefill calls (see the module docstring). Needs `--enable-graph-capture`."""

DEFAULT_SHAPES = "1x16,2x16,4x16,8x16,1x64,2x64,4x64,2x128,1x256,1x384"
"""`rows x width` ladder: short follow-up chunks batch up to 8 wide. Two chunks up to 128
tokens share a replay, and longer chunks up to 384 tokens replay alone.
`SEED_PREFILL_GRAPH_SHAPES` overrides it."""

VALIDATE_REL_TOL = graph_decode.VALIDATE_REL_TOL
"""Capture fidelity: replay vs the same static step run uncaptured (same kernels, so any gap is
a capture bug: a baked-in host value, a stale pointer, a replay that does nothing)."""

EAGER_MIN_CORR = float(os.environ.get("SEED_PREFILL_GRAPH_MIN_CORR", "0.99"))
"""Semantic check against the eager `Model.prefill_batch`: per-row Pearson correlation of the
logits (and of the DeltaNet state) must reach this, and each side's argmax must be in the
other's top `EAGER_TOPK`.

Why not `VALIDATE_REL_TOL` against eager, as the decode runner does: the decode graph replays
the eager decode step's own kernels, so the two agree to rounding. This static step does not
(decode attention per query instead of masked SDPA, `in_proj_all` instead of four GEMMs, the
varlen recurrent kernel, different GEMM tiles), so over 60 bf16 layers with top-10 routing the
two drift apart smoothly. Measured on 4x MI300A, 1x16 call resuming a 3-token prefix, per-layer
max-abs error relative to max |x| of the residual stream: 1e-4 at layer 0, 4e-3 by layer 4,
1-3% from layer 20 on, 3.3% at layer 59, with no single-layer jump; DeltaNet state 3-4%. That
drift alone fails a 2% max-abs gate on the final logits, which is why `SEED_PREFILL_GRAPHS=1`
never engaged ("prefill shape 1x16 logits disagree with eager" at every boot). A real bug
(wrong position, mask, lane or state row) decorrelates the logits instead."""

EAGER_MIN_STATE_CORR = float(os.environ.get("SEED_PREFILL_GRAPH_MIN_STATE_CORR", "0.98"))
"""The correlation gate on the DeltaNet state (`conv`, `rec`, min over layers). The state is
this rank's head shard, so unlike the logits it differs by rank, and the verdict is agreed
across ranks before it is acted on (`graph_decode.all_ranks`).

Lower than `EAGER_MIN_CORR` because the state drifts more than the logits, most at the
widths that switch the recurrence to the chunked kernel (`DN_CHUNKED`). Measured on 4x MI300A,
per rank, at boot: 2x64 state corr 0.98974 / 0.99185 / 0.99593 / 0.99704 in one job and 0.99130
to 0.99763 in another, 1x64 0.99157 to 0.99757, every other shape 0.995 to 0.9998, while the
logits (identical on every rank) stayed at 0.997 or above with the top-5 agreeing. The one
rank at 0.98974 against the old 0.99 is drift, not a wrong shape; a wrong lane, position or
state row decorrelates the state far below 0.98."""
VALIDATE_LANES = os.environ.get("SEED_PREFILL_VALIDATE_LANES", "0") == "1"
"""Boot memory: validation clones only the DeltaNet lanes the synthetic call touches (at most
8 of 96) instead of the whole pool, cutting the per-rank transient from ~17.5 GiB to ~1.5 GiB.
Every other lane is zero after `_reset_lanes` and must stay zero; each run checks that, and a
nonzero lane outside the touched range reruns the shape with whole-pool clones. The
statistics are the whole-pool ones (the zero lanes enter the correlation as zeros), so the
verdicts are the same up to float64 vs float32 rounding."""
VALIDATE_LOWMEM = os.environ.get("SEED_PREFILL_VALIDATE_LOWMEM", "0") == "1"
"""Boot memory: validation holds at most three DeltaNet-pool clones instead of four and
returns them to the device after each shape. Same checks, same verdicts."""

EAGER_TOPK = 5

DN_CHUNKED = os.environ.get("SEED_PREFILL_GRAPH_DN_CHUNKED", "0") not in ("0", "", "false", "False")
"""Captured prefill (and `graph_mixed`) shapes of width `>= DN_CHUNKED_MIN` run the DeltaNet
recurrence through `deltanet_prefill_chunked` (chunkwise-parallel WY form, Yang et al.,
NeurIPS'24) instead of `deltanet_fused.fused_recurrent_prefill`, whose serial depth is the
width: `T` dependent steps per layer, which is why a captured prefill costs about as much per
token as a decode step costs per row. Chunked, the serial depth is `C + T/C`."""

DN_CHUNKED_MIN = int(os.environ.get("SEED_PREFILL_GRAPH_DN_CHUNKED_MIN", "64"))

DN_STATE_INPLACE = os.environ.get("SEED_PREFILL_DN_STATE_INPLACE", "0") not in ("0", "", "false", "False")
"""Fused-glue chunked DeltaNet prefill (`_deltanet_prefill_core_fused`): the state kernel reads
and writes each row's lane of the recurrent-state pool in place (inactive rows never write)
and derives the uniform chunk table from its program ids. Removes the lane gather, the clone,
the `where`, the scatter and four table-building launches per layer; bit-identical."""

ATTN_FUSED = os.environ.get("SEED_PREFILL_ATTN_FUSED", "0") not in ("0", "", "false", "False")
"""Captured prefill full attention: q/k RMSNorm, RoPE and the masked paged-KV write run as the
decode path's one kernel (`attn_decode_fused.rmsnorm_rope_and_kv_write`, gated per token on
`buf.real`), and the output gate as `decode_glue.sigmoid_gate_mul` when `SEED_ELEMWISE_FUSED`
is on, instead of ~20 torch launches per layer. Same rounding points as the decode path
(`SEED_ATTN_ROPE_KV_FUSED`), not bit-identical to the torch `apply_rope` chain."""

DN_NORM_FUSED = os.environ.get("SEED_PREFILL_DN_NORM_FUSED", "0") not in ("0", "", "false", "False")
"""Fused-glue chunked DeltaNet prefill: the fp32 recurrence output goes straight into
`deltanet_fused.gated_rmsnorm` (one launch, bf16 rounding in-kernel, the decode path's kernel)
instead of `.to(bf16)` plus the ~13-launch torch `gated_rmsnorm`."""

PACK = os.environ.get("SEED_PREFILL_PACK", "0") not in ("0", "", "false", "False")
"""`SEED_PREFILL_PACK=1`: captured shapes whose DeltaNet runs the fused-glue chunked path
(`pack_eligible`) are captured *packed*: the `[rows, width]` token matrix is one stream of
`rows * width` tokens holding up to `segs` segments (one per call), each starting on a
DeltaNet chunk boundary (`PACK_ALIGN`) and free to span rows. Why: per-row padding measured
~23% of captured prefill area at C96 under `SEED_PREFILL_ACCUM` (rows are sized to the
longest chunk), and a replay's cost is affine in area, not real tokens.

What is per segment instead of per row: the DeltaNet recurrence (each segment's chunk run
from `seg_first`/`seg_count`, `deltanet_prefill_chunked.chunked_prefill_packed`), the conv
window (taps before a segment's start read its lane's carried window,
`deltanet_prefill_glue.prefill_glue_packed`), and the LM-head row (`seg_last`). Attention,
KV writes, RoPE, GEMMs, MoE and norms are per token already. Per segment, the DeltaNet chunks
are the ones the per-row layout runs for the same tokens at a row start, so packing changes
no DeltaNet arithmetic; GEMM/MoE rows can round differently only through a different `M`
(another shape) or row placement."""

PACK_ALIGN = deltanet_prefill_chunked.CHUNK

PACK_SYNTH_SEGS = 12
"""Segments of a packed shape's boot validation call (`PrefillGraphRunner._synthetic`)."""
"""Segment start alignment in the packed stream: the chunked kernels' chunk length."""

PACK_SEG_MIN = int(os.environ.get("SEED_PREFILL_PACK_SEG_MIN", "32"))
"""Segment slots of a packed shape: `min(max_batch, max(rows, area // PACK_SEG_MIN))`. An
unused slot costs a table entry and one skipped state-kernel program per head and value
block; a used one also one LM-head row."""


def graph_dn_chunked(width: int, device: torch.device) -> bool:
    """Whether a captured `[rows, width]` DeltaNet call takes the chunked kernels (static per
    shape, so every replay of a graph takes the same path)."""
    return (
        DN_CHUNKED
        and width >= DN_CHUNKED_MIN
        and width % deltanet_prefill_chunked.CHUNK == 0
        and deltanet_prefill_chunked.HAVE_TRITON
        and deltanet_fused.available_prefill(device)
    )

TUNE = os.environ.get("SEED_PREFILL_GRAPH_TUNE", "1") not in ("0", "", "false", "False")
"""Tune BLAS solutions (`blas_tune`) at every captured shape's GEMM width (`rows * width`).
Untuned skinny widths take the default tile: measured per call at M=5..47, `k_proj`/`v_proj`
(N=256) 126 us vs 13 us tuned, shared `gate_up` (N=512) 113 vs 21, `in_proj_all` 147 vs 20;
~20 ms of a small prefill's ~28 ms dense-GEMM device time."""


def gemm_widths(max_batch: int, max_seq: int) -> tuple[int, ...]:
    """GEMM M values the captured prefill shapes run at, for `blas_tune.tune`'s `batches`.
    Empty unless `SEED_PREFILL_GRAPHS` and `SEED_PREFILL_GRAPH_TUNE` are both on."""
    if not (PREFILL_GRAPHS and TUNE):
        return ()
    spec = os.environ.get("SEED_PREFILL_GRAPH_SHAPES", DEFAULT_SHAPES)
    return tuple(sorted({s.area for s in parse_shapes(spec, max_batch, max_seq)}))


@dataclass(frozen=True, order=True)
class Shape:
    rows: int
    width: int
    segs: int = field(default=0, compare=False)
    """Packed segment slots (`SEED_PREFILL_PACK`); 0 for the per-row layout. Not part of the
    shape's identity: a deployment captures each `rows x width` one way only."""

    @property
    def area(self) -> int:
        return self.rows * self.width

    @property
    def packed(self) -> bool:
        return self.segs > 0

    @property
    def label(self) -> str:
        """`prefill_path` / log name: `graphRxW`, with a `p` suffix when packed."""
        return f"graph{self.rows}x{self.width}{'p' if self.packed else ''}"


def parse_shapes(spec: str, max_batch: int, max_seq: int) -> list[Shape]:
    """`"RxW,..."` -> shapes, dropping any the deployment cannot use (`rows > max_batch` or
    `width > max_seq`). Raises on a malformed entry rather than silently ignoring it."""
    shapes = set()
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        rows, sep, width = item.partition("x")
        if not sep or not rows.isdigit() or not width.isdigit() or int(rows) < 1 or int(width) < 1:
            raise ValueError(f"SEED_PREFILL_GRAPH_SHAPES: bad entry {item!r} (want ROWSxWIDTH)")
        if int(rows) <= max_batch and int(width) <= max_seq:
            shapes.add(Shape(int(rows), int(width)))
    return sorted(shapes, key=lambda s: (s.area, s.width))


def shape_for(lengths: Sequence[int], shapes: Sequence[Shape]) -> Shape | None:
    """The smallest-area shape holding `len(lengths)` rows of `max(lengths)` tokens, or None.
    A packed shape holds them when their aligned lengths fit its area and slots
    (`packed_fits`); it always holds what its per-row layout would."""
    need_rows, need_width = len(lengths), max(lengths)
    for shape in shapes:  # sorted by area
        if shape.packed:
            if packed_fits(lengths, shape):
                return shape
        elif shape.rows >= need_rows and shape.width >= need_width:
            return shape
    return None


def pack_align(n: int) -> int:
    """Stream tokens a segment of `n` real tokens occupies (rounded up to `PACK_ALIGN`)."""
    return -(-n // PACK_ALIGN) * PACK_ALIGN


def packed_fits(lengths: Sequence[int], shape: Shape) -> bool:
    return (
        shape.packed
        and 1 <= len(lengths) <= shape.segs
        and sum(pack_align(n) for n in lengths) <= shape.area
    )


def pack_eligible(shape: Shape, device: torch.device, pack: bool = PACK) -> bool:
    """Whether `SEED_PREFILL_PACK` captures `shape` packed: its DeltaNet takes the fused-glue
    chunked kernels (the packed variants exist only for that path)."""
    return (
        pack
        and shape.width % PACK_ALIGN == 0
        and graph_dn_chunked(shape.width, device)
        and deltanet_prefill_glue.available(device)
    )


def packed(shape: Shape, max_batch: int) -> Shape:
    """`shape` with its packed segment slots (see `PACK_SEG_MIN`)."""
    segs = min(max_batch, max(shape.rows, shape.area // max(1, PACK_SEG_MIN)))
    return replace(shape, segs=segs)


class PrefillBuffers:
    """One shape's static tensors. Inputs are rewritten by `PrefillGraphRunner.fill` before
    every replay; `out` (`[rows, vocab]` fp32, row `j`'s last real column's logits) is read
    after it. Query-level tensors are flattened row-major (`j * width + c`)."""

    def __init__(self, model: Model, shape: Shape, device: torch.device) -> None:
        r, w = shape.rows, shape.width
        self.shape = shape
        self.tokens = torch.zeros(r, w, dtype=torch.long, device=device)
        self.pos = torch.zeros(r, dtype=torch.long, device=device)
        self.length = torch.ones(r, dtype=torch.long, device=device)
        self.active = torch.zeros(r, dtype=torch.bool, device=device)
        self.slot_rows = torch.zeros(r, dtype=torch.long, device=device)
        self.real = torch.zeros(r, w, dtype=torch.bool, device=device)  # active & c < length
        self.q_pos = torch.zeros(r * w, dtype=torch.long, device=device)
        self.q_table = torch.zeros(
            r * w, model.max_blocks_per_lane, dtype=torch.int32, device=device
        )
        self.q_valid = torch.zeros(r * w, dtype=torch.int32, device=device)
        self.write_rows = torch.zeros(r * w, dtype=torch.long, device=device)
        self.packed = shape.packed
        n_out = shape.segs if shape.packed else r
        self.out = torch.zeros(n_out, model.cfg.vocab, dtype=torch.float32, device=device)
        if shape.packed:  # per segment slot, then per stream token (`PrefillGraphRunner.fill`)
            s, i32 = shape.segs, {"dtype": torch.int32, "device": device}
            self.seg_first = torch.zeros(s, **i32)
            self.seg_count = torch.zeros(s, **i32)
            self.seg_start = torch.zeros(s, **i32)
            self.seg_len = torch.ones(s, **i32)
            self.seg_lanes = torch.zeros(s, dtype=torch.long, device=device)
            self.seg_active = torch.zeros(s, dtype=torch.bool, device=device)
            self.seg_last = torch.zeros(s, dtype=torch.long, device=device)
            self.tok_seg = torch.zeros(r * w, **i32)
            self.tok_off = torch.zeros(r * w, **i32)
            self.tok_lane = torch.zeros(r * w, dtype=torch.long, device=device)
            self.tok_pos = torch.zeros(r * w, dtype=torch.long, device=device)
            self.tok_in = torch.zeros(r * w, dtype=torch.bool, device=device)


# ---------------------------------------------------------------- static-shape layers


def attn_prefill_static(model: Model, i: int, x: torch.Tensor, buf: PrefillBuffers) -> torch.Tensor:
    """Full attention for a `[rows, width]` chunk matrix: project, RoPE at each query's own
    (clamped) position, masked KV write, then one decode-attention query per `(row, column)`.
    Head counts from `model.tp.plan` (see `graph_decode.attn_decode_static`)."""
    w = model.layers[i]
    qg = F.linear(x, w["q_proj"])
    k, v = F.linear(x, w["k_proj"]), F.linear(x, w["v_proj"])
    return F.linear(attn_prefill_core(model, i, qg, k, v, buf), w["o_proj"])


def attn_prefill_core(
    model: Model, i: int, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor, buf: PrefillBuffers
) -> torch.Tensor:
    """`attn_prefill_static` from the projections' outputs (`[rows, width, ...]`) to the gated
    attention output `o_proj` reads; split out for `graph_mixed` (see
    `graph_decode.attn_decode_core`)."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    r, t = qg.shape[0], qg.shape[1]
    pl = model.tp.plan
    nq, nkv = pl.q.count, pl.kv.count
    q, gate = qg.view(r, t, nq, 2 * c.head_dim).chunk(2, dim=-1)
    if ATTN_FUSED and attn_decode_fused.available(qg.device):
        return _attn_prefill_core_fused(model, i, q, gate, k, v, buf)
    q = rmsnorm(q, w["q_norm"], c.eps)
    k = rmsnorm(k.view(r, t, nkv, c.head_dim), w["k_norm"], c.eps)
    v = v.view(r, t, nkv, c.head_dim)
    cos, sin = model.rope_at(buf.q_pos, qg.dtype)  # [r*t, rot]
    cos, sin = cos.view(r, t, 1, -1), sin.view(r, t, 1, -1)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    keep = buf.real.reshape(-1)[:, None, None]
    rows = buf.write_rows
    new_k, new_v = k.reshape(r * t, nkv, c.head_dim), v.reshape(r * t, nkv, c.head_dim)
    pool["k"][rows] = torch.where(keep, new_k, pool["k"][rows])
    pool["v"][rows] = torch.where(keep, new_v, pool["v"][rows])

    out = decode_attention_paged(
        q.reshape(r * t, nq, 1, c.head_dim),
        pool["k"],
        pool["v"],
        buf.q_table,
        buf.q_valid,
        buf.q_pos,
        model.block_size,
        c.head_dim**-0.5,
    )
    out = out.reshape(r, t, -1)
    return out * torch.sigmoid(gate.reshape(r, t, -1))


def _attn_prefill_core_fused(
    model: Model,
    i: int,
    q: torch.Tensor,
    gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buf: PrefillBuffers,
) -> torch.Tensor:
    """`attn_prefill_core` under `SEED_PREFILL_ATTN_FUSED`: every `(row, column)` query is one
    row of the decode kernels' batch (`b = rows * width`)."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    r, t, nq = q.shape[0], q.shape[1], q.shape[2]
    b, hd = r * t, c.head_dim
    nkv = k.shape[-1] // hd
    cos, sin = model.rope_at(buf.q_pos, q.dtype)  # [r*t, rot]
    qn = attn_decode_fused.rmsnorm_rope_and_kv_write(
        q.view(b, nq, hd),
        k.view(b, nkv, hd),
        v.view(b, nkv, hd),
        w["q_norm"],
        w["k_norm"],
        cos,
        sin,
        c.rot_dim,
        c.eps,
        pool["k"],
        pool["v"],
        buf.write_rows,
        buf.real.reshape(-1),
    )
    out = decode_attention_paged(
        qn,
        pool["k"],
        pool["v"],
        buf.q_table,
        buf.q_valid,
        buf.q_pos,
        model.block_size,
        hd**-0.5,
    )
    out = out.reshape(b, 1, -1)
    if decode_glue.available(out):
        return decode_glue.sigmoid_gate_mul(out, gate.view(b, 1, nq, hd)).view(r, t, -1)
    return (out * torch.sigmoid(gate.reshape(b, 1, -1))).view(r, t, -1)


def deltanet_prefill_static(
    model: Model, i: int, x: torch.Tensor, buf: PrefillBuffers
) -> torch.Tensor:
    """Gated DeltaNet for a `[rows, width]` chunk matrix, each row resuming its lane's state.
    Padded columns are identity steps (`beta = g = 0`); the conv window is gathered at each
    row's length; padding rows write back what they gathered (see the module docstring)."""
    w = model.layers[i]
    proj = F.linear(x, w["in_proj_all"])
    return F.linear(deltanet_prefill_core(model, i, proj, buf), w["out_proj"])


def deltanet_prefill_core(
    model: Model, i: int, proj: torch.Tensor, buf: PrefillBuffers
) -> torch.Tensor:
    """`deltanet_prefill_static` from `in_proj_all`'s output (`[rows, width, ...]`) to the
    gated-norm output `out_proj` reads; split out for `graph_mixed`."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    r, t = proj.shape[0], proj.shape[1]
    lanes = buf.slot_rows
    active = buf.active
    key_dim, val_dim = c.k_heads * c.k_dim, c.v_heads * c.v_dim

    if graph_dn_chunked(t, proj.device) and deltanet_prefill_glue.available(proj.device):
        return _deltanet_prefill_core_fused(model, i, proj, buf)
    qkv, z, beta_raw, a_raw = proj.split(in_proj_sizes(c), dim=-1)
    pre_conv = pool["conv"][lanes]  # [r, C, K-1], a gathered copy
    full = torch.cat([pre_conv, qkv.transpose(1, 2)], dim=-1)  # [r, C, K-1 + t]
    mixed = F.silu(F.conv1d(full, w["conv"], groups=full.shape[1]))  # [r, C, t]
    win = torch.arange(pre_conv.shape[-1], device=proj.device)
    idx = (buf.length[:, None] + win[None, :])[:, None, :].expand(-1, full.shape[1], -1)
    new_conv = torch.gather(full, -1, idx)
    pool["conv"][lanes] = torch.where(active[:, None, None], new_conv, pre_conv)

    q, k, v = mixed.transpose(1, 2).split([key_dim, key_dim, val_dim], dim=-1)
    q, k = q.reshape(r, t, c.k_heads, c.k_dim), k.reshape(r, t, c.k_heads, c.k_dim)
    v = v.reshape(r, t, c.v_heads, c.v_dim).contiguous()
    real = buf.real[:, :, None]
    beta = torch.where(real, beta_raw.sigmoid(), 0.0)
    g = torch.where(real, -w["A_log"].exp() * F.softplus(a_raw.float() + w["dt_bias"]), 0.0)
    rep = c.v_heads // c.k_heads
    q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)

    pre_rec = pool["rec"][lanes]
    rec = pre_rec.clone()
    heads = c.v_heads
    if graph_dn_chunked(t, proj.device):
        # `SEED_PREFILL_GRAPH_DN_CHUNKED`: the chunked (WY) kernels with a device-built chunk
        # table (`uniform=t`, capturable). Padded columns are identity steps there too
        # (`beta = g = 0` zero their rows of `U` and `W` and leave the decay at 1).
        out, _ = deltanet_prefill_chunked.chunked_prefill(
            q.reshape(r * t, heads, c.k_dim),
            k.reshape(r * t, heads, c.k_dim),
            v.reshape(r * t, heads, c.v_dim),
            g.reshape(r * t, heads).float(),
            beta.reshape(r * t, heads).contiguous(),
            rec,
            None,
            uniform=t,
        )
        out = out.reshape(r, t, heads, c.v_dim).to(proj.dtype)
    elif deltanet_fused.available_prefill(proj.device):
        cu = torch.arange(0, r * t + 1, t, dtype=torch.int32, device=proj.device)
        out, _ = deltanet_fused.fused_recurrent_prefill(
            q.reshape(r * t, heads, c.k_dim),
            k.reshape(r * t, heads, c.k_dim),
            v.reshape(r * t, heads, c.v_dim),
            g.reshape(r * t, heads).float(),
            beta.reshape(r * t, heads),
            rec,
            cu,
        )
        out = out.reshape(r, t, heads, c.v_dim).to(proj.dtype)
    else:
        out = delta_rule_recurrent(q, k, v, g, beta, rec).to(proj.dtype)
    pool["rec"][lanes] = torch.where(active[:, None, None, None], rec, pre_rec)

    out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
    return out.reshape(r, t, -1)


def _deltanet_prefill_core_fused(
    model: Model, i: int, proj: torch.Tensor, buf: PrefillBuffers
) -> torch.Tensor:
    """`deltanet_prefill_core` with its glue in one kernel (`SEED_DN_PREFILL_GLUE_FUSED`, see
    `deltanet_prefill_glue`) and the chunked kernels reading key-head `q`/`k`. Same state
    contract: conv window and recurrent state written back only for active rows."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    r, t = proj.shape[0], proj.shape[1]
    lanes, active = buf.slot_rows, buf.active
    heads = c.v_heads
    if getattr(buf, "packed", False):
        return _deltanet_prefill_core_packed(model, i, proj, buf)
    q, k, v, beta, g = deltanet_prefill_glue.prefill_glue(
        proj,
        w["conv"],
        pool["conv"],
        lanes,
        buf.length,
        active,
        buf.real,
        w["A_log"],
        w["dt_bias"],
        c.k_heads * c.k_dim,
        heads,
    )
    qkv3 = (
        q.view(r * t, c.k_heads, c.k_dim),
        k.view(r * t, c.k_heads, c.k_dim),
        v.view(r * t, heads, c.v_dim),
    )
    if DN_STATE_INPLACE:
        out, _ = deltanet_prefill_chunked.chunked_prefill(
            *qkv3, g, beta, pool["rec"], None, uniform=t, lanes=lanes, active=active,
            table_free=True,
        )
    else:
        pre_rec = pool["rec"][lanes]
        rec = pre_rec.clone()
        out, _ = deltanet_prefill_chunked.chunked_prefill(
            *qkv3, g, beta, rec, None, uniform=t
        )
        pool["rec"][lanes] = torch.where(active[:, None, None, None], rec, pre_rec)
    z = proj[..., c.k_heads * c.k_dim * 2 + heads * c.v_dim :][..., : heads * c.v_dim]
    if DN_NORM_FUSED:
        out = deltanet_fused.gated_rmsnorm(
            out.view(r * t, heads, c.v_dim),
            z.reshape(r * t, heads, c.v_dim),
            w["dn_norm"],
            c.eps,
            proj.dtype,
        )
        return out.view(r, t, -1)
    out = out.to(proj.dtype)
    out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
    return out.reshape(r, t, -1)


def _deltanet_prefill_core_packed(
    model: Model, i: int, proj: torch.Tensor, buf: PrefillBuffers
) -> torch.Tensor:
    """`_deltanet_prefill_core_fused` on a packed stream (`SEED_PREFILL_PACK`): conv windows,
    recurrent state and chunk runs per segment slot instead of per row. The state is advanced
    in the lane pool in place (the `SEED_PREFILL_DN_STATE_INPLACE` contract, bit-identical to
    gather/scatter); the output norm follows `SEED_PREFILL_DN_NORM_FUSED` as the per-row path."""
    c, w, pool = model.cfg, model.layers[i], model.pool[i]
    r, t = proj.shape[0], proj.shape[1]
    heads = c.v_heads
    q, k, v, beta, g = deltanet_prefill_glue.prefill_glue_packed(
        proj,
        w["conv"],
        pool["conv"],
        buf.seg_lanes,
        buf.seg_start,
        buf.seg_len,
        buf.seg_active,
        buf.tok_seg,
        buf.tok_off,
        buf.real,
        w["A_log"],
        w["dt_bias"],
        c.k_heads * c.k_dim,
        heads,
    )
    out, _ = deltanet_prefill_chunked.chunked_prefill_packed(
        q.view(r * t, c.k_heads, c.k_dim),
        k.view(r * t, c.k_heads, c.k_dim),
        v.view(r * t, heads, c.v_dim),
        g,
        beta,
        pool["rec"],
        buf.seg_first,
        buf.seg_count,
        buf.seg_lanes,
        buf.seg_active,
    )
    z = proj[..., c.k_heads * c.k_dim * 2 + heads * c.v_dim :][..., : heads * c.v_dim]
    if DN_NORM_FUSED:
        out = deltanet_fused.gated_rmsnorm(
            out.view(r * t, heads, c.v_dim),
            z.reshape(r * t, heads, c.v_dim),
            w["dn_norm"],
            c.eps,
            proj.dtype,
        )
        return out.view(r, t, -1)
    out = out.to(proj.dtype)
    out = gated_rmsnorm(out.reshape(-1, c.v_dim), z.reshape(-1, c.v_dim), w["dn_norm"], c.eps)
    return out.reshape(r, t, -1)


def prefill_step(model: Model, buf: PrefillBuffers) -> Callable[[], None]:
    """The captured callable: embed, 60 layers (both all-reduces inside, as in
    `graph_decode.decode_layer_static`), then the LM head on each row's last real column."""

    def step() -> None:
        c = model.cfg
        x = F.embedding(buf.tokens, model.embed)
        r, t = x.shape[:2]
        total = r * t
        chained = graph_decode.AR_RMSNORM_FUSED
        n = len(model.layers)
        if chained:
            h = rmsnorm(x, model.layers[0]["in_norm"], c.eps)
        # Mirror graph_mixed's proven row-sharded residual contract: each collective consumes
        # a full mixer, carries only this rank's residual rows between calls, and returns the
        # full normalized rows required by the next projection. Shapes are static at capture.
        cr = getattr(model.tp, "custom_reduce", None)
        sp = chained and cr is not None and _sp_rows_ok(model, total)
        if sp:
            shard = total // 4
            flat_x = x.reshape(total, -1)
            x = flat_x[model.tp.rank * shard : (model.tp.rank + 1) * shard].contiguous()
        for i in range(n):
            w = model.layers[i]
            if not chained:
                h = rmsnorm(x, w["in_norm"], c.eps)
            h_static = h.reshape(r, t, -1)
            mixer = (
                attn_prefill_static(model, i, h_static, buf)
                if c.layer_types[i] == "full_attention"
                else deltanet_prefill_static(model, i, h_static, buf)
            )
            if sp:
                nxt = model.layers[i + 1]["in_norm"] if i + 1 < n else model.final_norm
                x, h_mid = cr.sp_ar_add_rmsnorm(
                    mixer.reshape(total, -1).contiguous(), x, w["post_norm"], c.eps
                )
                x, h = cr.sp_ar_add_rmsnorm(model.moe(i, h_mid).contiguous(), x, nxt, c.eps)
                continue
            if chained:
                nxt = model.layers[i + 1]["in_norm"] if i + 1 < n else model.final_norm
                x, h_mid = graph_decode._residual_norm(model, mixer, x, w["post_norm"])
                x, h = graph_decode._residual_norm(model, model.moe(i, h_mid), x, nxt)
                continue
            x = x + model.tp.all_reduce(mixer)
            x = x + model.tp.all_reduce(model.moe(i, rmsnorm(x, w["post_norm"], c.eps)))
        if getattr(model, "mtp", None) is not None:
            _mtp_tail(model, buf, _full_residual(model, x, total, sp))
        rows = torch.arange(r, device=h.device if chained else x.device)
        if getattr(buf, "packed", False):  # one LM-head row per segment slot, at its last real token
            if chained:
                normed = h.reshape(total, -1)[buf.seg_last]
            else:
                normed = rmsnorm(x.reshape(total, -1)[buf.seg_last], model.final_norm, c.eps)
        elif chained:
            last = h.reshape(r, t, -1)[rows, buf.length - 1]
            normed = last
        else:
            last = x[rows, buf.length - 1]
            normed = rmsnorm(last, model.final_norm, c.eps)
        buf.out.copy_(model.unembed(normed).float())

    return step


def _full_residual(model: Model, x: torch.Tensor, total: int, sp: bool) -> torch.Tensor:
    """The final raw residual as `[total, hidden]` on every rank. Under the row-sharded
    residual each rank holds rows `[rank * total/world, ...)`; one all-gather (exact)
    reassembles them in rank order."""
    if not sp:
        return x.reshape(total, -1)
    world = model.tp.world
    gathered = model.tp.all_gather_last(x)  # [shard, world * hidden]
    return gathered.view(total // world, world, -1).transpose(0, 1).reshape(total, -1)


def _mem_note(device: torch.device, tag: str) -> None:
    """Log this rank's free device memory (the APU's share of unified memory on MI300A)."""
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    log(
        f"[mem] {device} {tag}: free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB; "
        f"NUMA used GiB {_numa_used()}"
    )


def _numa_used() -> str:
    """Per-NUMA-node MemUsed from sysfs, `n0=.. n1=..` (empty where sysfs has none)."""
    out = []
    for node in sorted(glob.glob("/sys/devices/system/node/node[0-9]*/meminfo")):
        try:
            text = open(node).read()  # noqa: SIM115, PTH123
        except OSError:
            continue
        for line in text.splitlines():
            if "MemUsed:" in line:
                n = node.split("/node")[-1].split("/")[0]
                out.append(f"n{n}={int(line.split()[-2]) / 2**20:.1f}")
    return " ".join(out)


def _mtp_tail(model: Model, buf: PrefillBuffers, x: torch.Tensor) -> None:
    """`Model.forward`/`forward_packed`'s MTP epilogue for a `[rows, width]` call. `x` is the
    final raw residual, `[rows * width, hidden]`. Column `c` of row `j` caches MTP K/V at
    position `pos + c` from token `c` and the target hidden at `pos + c - 1` (the lane's
    `hidden_scratch` row for `c = 0`); then `hidden_scratch[lane]` becomes the residual at the
    row's last real column. Padded columns and padding rows write nothing."""
    if getattr(buf, "packed", False):
        _mtp_tail_packed(model, buf, x)
        return
    r, t = buf.shape.rows, buf.shape.width
    xs = x.view(r, t, -1)
    seed = model.hidden_scratch[buf.slot_rows]  # a gathered copy, read before the write below
    previous = torch.cat([seed[:, None], xs[:, :-1]], dim=1)
    mtp_mod.cache_target_rows(
        model,
        model.mtp,
        buf.tokens,
        previous,
        buf.q_pos.view(r, t),
        buf.write_rows.view(r, t),
        buf.real,
    )
    mtp_mod.mtp_dense_moe.scatter_time_active_rows(
        model.hidden_scratch, buf.slot_rows, xs, buf.length - 1, buf.active
    )


def _mtp_tail_packed(model: Model, buf: PrefillBuffers, x: torch.Tensor) -> None:
    """`_mtp_tail` for a packed stream (`SEED_PREFILL_PACK`): stream token `i` of a segment
    caches MTP K/V at its own position from its token and the residual of stream token `i - 1`
    (the segment lane's `hidden_scratch` row at the segment's first token). Then each active
    segment's lane takes the residual at its last real token (`seg_last`). Tokens past a
    segment's length and outside every segment write nothing (`buf.real`)."""
    area = buf.shape.area
    xs = x.view(area, -1)
    seed = mtp_mod.mtp_dense_moe.gather_first_dim(model.hidden_scratch, buf.tok_lane)
    shifted = torch.cat([seed[:1], xs[:-1]], dim=0)
    first = (buf.tok_off == 0)[:, None]
    previous = torch.where(first, seed, shifted)
    mtp_mod.cache_target_rows(
        model,
        model.mtp,
        buf.tokens.view(1, area),
        previous.view(1, area, -1),
        buf.q_pos.view(1, area),
        buf.write_rows.view(1, area),
        buf.real.view(1, area),
    )
    last = mtp_mod.mtp_dense_moe.gather_first_dim(xs.contiguous(), buf.seg_last)
    mtp_mod.mtp_dense_moe.scatter_active_first_dim(
        model.hidden_scratch, buf.seg_lanes, last, buf.seg_active
    )


def _sp_rows_ok(model: Model, rows: int) -> bool:
    """Whether a static prefill shape can keep the residual row-sharded across collectives."""
    import allreduce_custom  # noqa: PLC0415

    cr = getattr(model.tp, "custom_reduce", None)
    return (
        allreduce_custom.SP
        and cr is not None
        and model.tp.world == 4
        and model.embed.dtype == torch.bfloat16
        and rows % 4 == 0
        and rows // 4 <= allreduce_custom.SP_MAX_ROWS
        and rows * model.cfg.hidden <= allreduce_custom.SP_SLOT_ELEMS
    )


# ---------------------------------------------------------------- the runner


class PrefillGraphRunner:
    """Captures one graph per `Shape` and replays prefill calls that fit one. Shares the decode
    runner's device block-table mirror and dirty flags (like `graph_mtp.MTPVerifyRunner`), so
    any host-table change the decode runner tracks also resyncs this runner's view."""

    def __init__(
        self,
        model: Model,
        backend: CaptureBackend,
        lane_table: graph_decode.LaneBlockTables,
        dirty: list[bool],
        shapes: Sequence[Shape] | None = None,
        pack: bool | None = None,
    ) -> None:
        """`pack`: capture eligible shapes packed (`pack_eligible`); default `PACK`."""
        self.model, self.backend = model, backend
        self.lane_table, self._dirty = lane_table, dirty
        spec = os.environ.get("SEED_PREFILL_GRAPH_SHAPES", DEFAULT_SHAPES)
        self.shapes = (
            list(shapes)
            if shapes is not None
            else parse_shapes(spec, model.max_batch, model.max_seq)
        )
        self.device = model.devices[-1]
        self.shapes = [
            packed(s, model.max_batch)
            if pack_eligible(s, self.device, PACK if pack is None else pack)
            else replace(s, segs=0)
            for s in self.shapes
        ]
        self.graphs: dict[Shape, tuple[PrefillBuffers, Callable[[], None]]] = {}
        self.enabled = False
        self.last_shape: Shape | None = None  # tests
        self.replays = 0  # served calls since `prepare` (tests)

    def supported(self) -> tuple[bool, str]:
        if len(graph_decode.plan_segments(self.model.layer_dev)) != 1:
            return False, "needs a single-segment layout (TP owns one device)"
        if not self.shapes:
            return False, "no usable shape in SEED_PREFILL_GRAPH_SHAPES"
        return True, ""

    # -- dispatch -------------------------------------------------------------
    def replayable(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> Shape | None:
        """The shape that serves `calls`, or None for the eager path. Reads only the broadcast
        call list, so every rank decides the same way."""
        if not self.enabled or not self.valid_calls(calls):
            return None
        return shape_for([len(ids) for _, ids, _ in calls], self.shapes)

    def valid_calls(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> bool:
        """Whether `calls` can be laid out as rows at all: distinct in-range lanes, nonempty
        chunks, every chunk inside `max_seq`. Shape-independent (`graph_mixed` reuses it)."""
        if not calls:
            return False
        lanes = [slot for slot, _, _ in calls]
        if len(set(lanes)) != len(lanes) or not all(0 <= s < self.model.max_batch for s in lanes):
            return False
        lengths = [len(ids) for _, ids, _ in calls]
        return min(lengths) >= 1 and all(
            0 <= start and start + n <= self.model.max_seq
            for (_, _, start), n in zip(calls, lengths, strict=True)
        )

    def prefill_batch(
        self, shape: Shape, calls: Sequence[tuple[int, Sequence[int], int]]
    ) -> list[torch.Tensor]:
        """`Model.prefill_batch`'s contract (one `[1, vocab]` logits row per call) from a
        replay. A failed replay disables the runner and re-raises: it may have advanced some
        lanes' state, so it is not retried eagerly (same policy as `GraphDecodeRunner`)."""
        buf, replay = self.graphs[shape]
        self.fill(buf, calls)
        try:
            replay()
        except Exception as exc:
            self.enabled = False
            log(f"prefill replay failed, prefill eager from here on: {exc!r}")
            raise
        self.last_shape = shape
        self.replays += 1
        out = buf.out[: len(calls)].clone()
        return [out[j : j + 1] for j in range(len(calls))]

    def fill(
        self,
        buf: PrefillBuffers,
        calls: Sequence[tuple[int, Sequence[int], int]],
        avoid: Sequence[int] = (),
    ) -> None:
        """Copy one call's inputs into `buf`: host lists for tokens/positions/lengths/lanes,
        then the per-query block tables, positions and KV write rows on the device. Padding
        rows avoid the lanes in `avoid` when a free one exists (`graph_mixed`: the decode
        lanes of the same replay)."""
        model = self.model
        rows, width = buf.shape.rows, buf.shape.width
        for slot, ids, start in calls:
            table = model.block_tables[slot]
            before = len(table.blocks)
            model.grow_lane(slot, start + len(ids))
            if self._dirty[slot] or len(table.blocks) != before:
                self.lane_table.sync_lane(slot, table.blocks)
                self._dirty[slot] = False
        if buf.packed:
            self._fill_packed(buf, calls, avoid)
            return

        lanes = [slot for slot, _, _ in calls]
        pad = graph_decode.pad_slot_for(lanes, model.max_batch, avoid)
        row_tokens = [[0] * width for _ in range(rows)]
        row_pos, row_len = [0] * rows, [1] * rows
        row_active = [False] * rows
        for j, (_, ids, start) in enumerate(calls):
            row_tokens[j][: len(ids)] = list(ids)
            row_pos[j], row_len[j], row_active[j] = start, len(ids), True
        copy_from_host(buf.tokens, torch.tensor(row_tokens, dtype=torch.long))
        copy_from_host(buf.pos, torch.tensor(row_pos, dtype=torch.long))
        copy_from_host(buf.length, torch.tensor(row_len, dtype=torch.long))
        copy_from_host(buf.active, torch.tensor(row_active, dtype=torch.bool))
        copy_from_host(
            buf.slot_rows, torch.tensor(lanes + [pad] * (rows - len(calls)), dtype=torch.long)
        )

        bs, dev = model.block_size, buf.pos.device
        col = torch.arange(width, device=dev)[None, :]
        buf.real.copy_(buf.active[:, None] & (col < buf.length[:, None]))
        eff = buf.pos[:, None] + torch.minimum(col, buf.length[:, None] - 1)  # [rows, width]
        table = torch.where(
            buf.active[:, None], self.lane_table.table[buf.slot_rows], block_pool.RESERVED_BLOCK
        )
        block_id = torch.gather(table.long(), 1, eff // bs)
        reserved = block_pool.RESERVED_BLOCK * bs + col % bs
        buf.write_rows.copy_(torch.where(buf.real, block_id * bs + eff % bs, reserved).reshape(-1))
        buf.q_pos.copy_(eff.reshape(-1))
        buf.q_valid.copy_(((eff + bs) // bs).to(torch.int32).reshape(-1))
        buf.q_table.copy_(table.repeat_interleave(width, dim=0))

    def _fill_packed(
        self,
        buf: PrefillBuffers,
        calls: Sequence[tuple[int, Sequence[int], int]],
        avoid: Sequence[int] = (),
    ) -> None:
        """`fill` for a packed shape: call `j` is segment slot `j`, laid out at the next
        `PACK_ALIGN` boundary of the stream. Tokens past a segment's length (up to its
        alignment) are its padding columns, as in the per-row layout: clamped to its last real
        position, identity DeltaNet steps, masked KV writes. Tokens after the last segment belong
        to no segment: position 0 on the reserved table, masked writes, and zero DeltaNet output
        (`chunked_prefill_packed`)."""
        model, shape = self.model, buf.shape
        n_tok, segs = shape.area, shape.segs
        if not packed_fits([len(ids) for _, ids, _ in calls], shape):
            raise ValueError(f"calls do not fit packed shape {shape.label}")
        lanes = [slot for slot, _, _ in calls]
        pad = graph_decode.pad_slot_for(lanes, model.max_batch, avoid)
        tokens = [0] * n_tok
        tok_lane = [pad] * n_tok
        tok_pos = [0] * n_tok
        tok_real = [False] * n_tok
        tok_in = [False] * n_tok  # inside some segment's aligned span
        tok_seg = [0] * n_tok
        tok_off = [1 << 30] * n_tok  # outside every segment: conv taps read the stream only
        seg_first, seg_count = [0] * segs, [0] * segs
        seg_start, seg_len = [0] * segs, [1] * segs
        seg_lanes, seg_active, seg_last = [pad] * segs, [False] * segs, [0] * segs
        at = 0
        for j, (slot, ids, start) in enumerate(calls):
            n, span = len(ids), pack_align(len(ids))
            tokens[at : at + n] = list(ids)
            for o in range(span):
                x = at + o
                tok_lane[x], tok_seg[x], tok_off[x], tok_in[x] = slot, j, o, True
                tok_pos[x] = start + min(o, n - 1)
                tok_real[x] = o < n
            seg_first[j], seg_count[j] = at // PACK_ALIGN, span // PACK_ALIGN
            seg_start[j], seg_len[j] = at, n
            seg_lanes[j], seg_active[j], seg_last[j] = slot, True, at + n - 1
            at += span
        i32, i64 = torch.int32, torch.long
        copy_from_host(buf.tokens, torch.tensor(tokens, dtype=i64).view(shape.rows, shape.width))
        copy_from_host(buf.seg_first, torch.tensor(seg_first, dtype=i32))
        copy_from_host(buf.seg_count, torch.tensor(seg_count, dtype=i32))
        copy_from_host(buf.seg_start, torch.tensor(seg_start, dtype=i32))
        copy_from_host(buf.seg_len, torch.tensor(seg_len, dtype=i32))
        copy_from_host(buf.seg_lanes, torch.tensor(seg_lanes, dtype=i64))
        copy_from_host(buf.seg_active, torch.tensor(seg_active, dtype=torch.bool))
        copy_from_host(buf.seg_last, torch.tensor(seg_last, dtype=i64))
        copy_from_host(buf.tok_seg, torch.tensor(tok_seg, dtype=i32))
        copy_from_host(buf.tok_off, torch.tensor(tok_off, dtype=i32))
        copy_from_host(buf.real, torch.tensor(tok_real, dtype=torch.bool).view(shape.rows, shape.width))
        # Per-token paged-attention inputs, as `fill` builds them per row. Tokens outside every
        # segment use the reserved table (their writes are masked, their reads unused).
        bs, dev = model.block_size, buf.q_pos.device
        copy_from_host(buf.tok_lane, torch.tensor(tok_lane, dtype=i64))
        copy_from_host(buf.tok_pos, torch.tensor(tok_pos, dtype=i64))
        copy_from_host(buf.tok_in, torch.tensor(tok_in, dtype=torch.bool))
        lane_d, eff, inside = buf.tok_lane, buf.tok_pos, buf.tok_in
        real = buf.real.reshape(-1)
        table = torch.where(
            inside[:, None], self.lane_table.table[lane_d], block_pool.RESERVED_BLOCK
        )
        block_id = torch.gather(table.long(), 1, (eff // bs)[:, None])[:, 0]
        idx = torch.arange(n_tok, device=dev)
        reserved = block_pool.RESERVED_BLOCK * bs + idx % bs
        buf.write_rows.copy_(torch.where(real, block_id * bs + eff % bs, reserved))
        buf.q_pos.copy_(eff)
        buf.q_valid.copy_(((eff + bs) // bs).to(torch.int32))
        buf.q_table.copy_(table)

    # -- setup ----------------------------------------------------------------
    def prepare(self) -> bool:
        """Warm, capture and validate every shape; agree across ranks. Never raises (a
        failure leaves prefill eager on every rank). Boot-only, like `GraphDecodeRunner.
        prepare`: warmup and validation grow and reset lanes with the model's own allocator."""
        ok, why = self.supported()
        if not ok:
            log(f"prefill capture off: {why}")
            self.enabled = False
            return self._agree(False)
        try:
            self.graphs = {}
            for shape in self.shapes:
                t0 = time.perf_counter()
                buf = PrefillBuffers(self.model, shape, self.device)
                step = prefill_step(self.model, buf)
                self._warm(buf, step, shape)
                self.graphs[shape] = (buf, self.backend.capture(step, self.device))
                mem_timeline.mark(f"prefill_graph_{shape.label.removeprefix('graph')}", self.device)
                log(
                    f"captured prefill {shape.label.removeprefix('graph')} in "
                    f"{(time.perf_counter() - t0) * 1e3:.0f} ms"
                )
            self.enabled = True
        except Exception as exc:
            self.enabled = False
            log(f"prefill capture failed, prefill stays eager: {exc!r}")
        _mem_note(self.device, "after prefill capture")
        ok = self._agree(self.enabled)
        mem_timeline.mark("prefill_captured", self.device)
        if ok:
            ok = self._agree(self._validate())
        self._reset_lanes()
        mem_timeline.mark("prefill_validate", self.device)
        self.replays = 0
        if ok:
            log(f"prefill graphs enabled for shapes {[s.label.removeprefix('graph') for s in self.shapes]}")
        return ok

    def _agree(self, ok: bool) -> bool:
        self.enabled = graph_decode.all_ranks(self.model.tp, ok)
        return self.enabled

    def _reset_lanes(self) -> None:
        if self.model.scheduler_owns_blocks:
            raise RuntimeError("prefill capture is boot-only: the scheduler owns lane blocks now")
        for slot in range(self.model.max_batch):
            self.model._release_lane_blocks(slot)
            self.model.begin(slot)
            self._dirty[slot] = True
        self.model.bind(0)

    def _synthetic(self, shape: Shape) -> list[tuple[int, list[int], int]]:
        """One call per row, lengths 1..width, resuming at start 3 (a cached prefix). Packed:
        up to `PACK_SYNTH_SEGS` segments of mixed lengths (the first spans a row boundary when
        the shape has several rows), leaving at least one aligned block and one slot unused."""
        vocab = self.model.cfg.vocab
        calls = []
        if shape.packed:
            pattern = [shape.width + 37 if shape.area >= 4 * shape.width else shape.width // 2]
            pattern += [1, 17, 5, 40, 16, 33, 64, 3, 100, 15, 48]
            limit = max(1, min(shape.segs - 1, PACK_SYNTH_SEGS))
            used = 0
            for j, n in enumerate(pattern[:limit]):
                n = max(1, min(n, shape.area - PACK_ALIGN - used))
                if used + pack_align(n) > shape.area - PACK_ALIGN and calls:
                    break
                used += pack_align(n)
                calls.append((j, [(11 * j + 5 * t + 1) % vocab for t in range(n)], 3))
            return calls
        for j in range(shape.rows):
            n = shape.width if j == 0 else 1 + (7 * j) % shape.width
            calls.append((j, [(11 * j + 5 * t + 1) % vocab for t in range(n)], 3))
        return calls

    def _warm(self, buf: PrefillBuffers, step: Callable[[], None], shape: Shape) -> None:
        self.fill(buf, [(slot, ids, 0) for slot, ids, _ in self._synthetic(shape)])
        for _ in range(graph_decode.WARMUP_STEPS):
            step()
        self._reset_lanes()

    def _deltanet_state(self, lanes: int | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Clones of every DeltaNet layer's conv and recurrent state: all lanes, or `[:lanes]`."""
        return [
            (p["conv"][:lanes].clone(), p["rec"][:lanes].clone())
            for p in self.model.pool
            if "conv" in p
        ]

    def _tail_zero(self, lanes: int) -> bool:
        """Every DeltaNet lane from `lanes` on is all zero (one reduction per tensor, no clone)."""
        return all(
            not bool(p[k][lanes:].any())
            for p in self.model.pool
            if "conv" in p
            for k in ("conv", "rec")
        )

    def _restore(self, state: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        pools = [p for p in self.model.pool if "conv" in p]
        for pool, (conv, rec) in zip(pools, state, strict=True):
            pool["conv"][: conv.shape[0]].copy_(conv)
            pool["rec"][: rec.shape[0]].copy_(rec)

    def _hidden(self) -> torch.Tensor | None:
        """A copy of `hidden_scratch` under MTP (the rewind point), else None."""
        if getattr(self.model, "mtp", None) is None:
            return None
        return self.model.hidden_scratch.clone()

    def _mtp_state(self, calls: Sequence[tuple[int, Sequence[int], int]]) -> list[torch.Tensor]:
        """Under MTP: the calls' lanes' `hidden_scratch` rows, then this rank's MTP K and V
        at every row the calls wrote. Empty without MTP."""
        model = self.model
        if getattr(model, "mtp", None) is None:
            return []
        dev = model.hidden_scratch.device
        lanes = torch.tensor([slot for slot, _, _ in calls], device=dev)
        rows = torch.cat(
            [model._physical_rows_range(s, p, p + len(ids), dev) for s, ids, p in calls]
        )
        pool = model.mtp.pool
        return [model.hidden_scratch[lanes].clone(), pool["k"][rows].clone(), pool["v"][rows].clone()]

    def _others_untouched(
        self, calls: Sequence[tuple[int, Sequence[int], int]], before: torch.Tensor | None
    ) -> bool:
        """Every lane outside `calls` (a padding row's filler lane included) kept its
        `hidden_scratch` row bit for bit. True without MTP."""
        if before is None:
            return True
        keep = torch.ones(before.shape[0], dtype=torch.bool, device=before.device)
        keep[[slot for slot, _, _ in calls]] = False
        return torch.equal(self.model.hidden_scratch[keep], before[keep])

    def _validate(self) -> bool:
        """Every shape on one synthetic call (`_check`), each verdict agreed across ranks
        before it is acted on: a shape any rank rejects is dropped on every rank, and the
        others stay. Deciding alone would hang the group (`graph_decode.all_ranks`). True
        when at least one shape survives."""
        kept = []
        for shape in self.shapes:
            _mem_note(self.device, f"before check {shape.rows}x{shape.width}")
            if graph_decode.all_ranks(self.model.tp, self._check(shape)):
                kept.append(shape)
            else:
                log(f"prefill shape {shape.rows}x{shape.width} off on every rank")
                del self.graphs[shape]
        self.shapes = kept
        if kept:
            log(f"prefill: {len(kept)} shape(s) match the eager prefill on every rank")
        return bool(kept)

    def _check(self, shape: Shape, whole_pool: bool = False) -> bool:
        """One shape on one synthetic call (every row resuming a 3-token prefix, lengths
        1..width, a padding row when the shape has more rows than the call), two checks:

        1. Capture fidelity: the replay against the same static step run uncaptured, logits and
           DeltaNet state within `VALIDATE_REL_TOL` (same kernels, so only a capture bug fails).
        2. Semantics: the replay against the eager `Model.prefill_batch`, by correlation and
           top-k agreement (`EAGER_MIN_CORR`, `EAGER_MIN_STATE_CORR`), not max-abs error; see
           those constants for why.
        """
        model = self.model
        name = shape.label.removeprefix("graph")
        try:
            calls = self._synthetic(shape)
            if shape.rows > 1 and not shape.packed:
                calls = calls[: shape.rows - 1] or calls  # leave a padding row
            self._reset_lanes()
            lanes = None
            if VALIDATE_LANES and not whole_pool:
                lanes = max(slot for slot, _, _ in calls) + 1
                if not self._tail_zero(0):
                    return self._check(shape, whole_pool=True)
                lanes = lanes if lanes < model.max_batch else None
            for slot, _, _ in calls:
                model.prefill(slot, [2, 3, 4], 0)
            before = self._deltanet_state(lanes)
            hidden_before = self._hidden()
            want = model.prefill_batch([(s, list(ids), p) for s, ids, p in calls])
            want_state = self._deltanet_state(lanes)
            want_mtp = self._mtp_state(calls)
            if lanes is not None and not self._tail_zero(lanes):
                log(f"prefill shape {name}: a lane outside the call changed; whole-pool check")
                return self._check(shape, whole_pool=True)
            self._rewind(before, calls, hidden_before)
            got = self.prefill_batch(shape, calls)
            got_state = self._deltanet_state(lanes)
            got_mtp = self._mtp_state(calls)
            others_ok = self._others_untouched(calls, hidden_before)
            if lanes is not None and not self._tail_zero(lanes):
                log(f"prefill shape {name}: a lane outside the call changed; whole-pool check")
                return self._check(shape, whole_pool=True)
            total = None if lanes is None else model.max_batch
            state_corr = None
            if VALIDATE_LOWMEM:
                # Same statistics, shorter lifetimes: each clone is the whole DeltaNet pool
                # (~4.4 GiB/rank at 96 lanes), and four at once pushed the node holding the
                # OS image into exhaustion, spilling that rank's allocations to a remote node.
                state_corr = _state_corr(got_state, want_state, total)
                del want_state
            self._rewind(before, calls, hidden_before)
            if VALIDATE_LOWMEM:
                del before
            buf, _ = self.graphs[shape]
            self.fill(buf, calls)
            prefill_step(model, buf)()
            ref = buf.out[: len(calls)].clone()
            ref_state = self._deltanet_state(lanes)
            ref_mtp = self._mtp_state(calls)
            if lanes is not None and not self._tail_zero(lanes):
                log(f"prefill shape {name}: a lane outside the call changed; whole-pool check")
                return self._check(shape, whole_pool=True)

            faithful = (
                all(_close(g[0], r) for g, r in zip(got, ref, strict=True))
                and all(
                    _close(gc, rc) and _close(gr, rr)
                    for (gc, gr), (rc, rr) in zip(got_state, ref_state, strict=True)
                )
                and all(_close(g, r) for g, r in zip(got_mtp, ref_mtp, strict=True))
            )
            if not faithful:
                log(f"prefill shape {name} replay disagrees with its uncaptured step")
                return False
            if not others_ok:
                log(f"prefill shape {name} wrote hidden_scratch outside the call's lanes")
                return False
            if got_mtp:
                mtp_corr = min(_corr(g, w) for g, w in zip(got_mtp, want_mtp, strict=True))
                log(f"prefill shape {name} vs eager MTP: hidden/kv corr {mtp_corr:.5f}")
                if mtp_corr < EAGER_MIN_STATE_CORR:
                    log(f"prefill shape {name} MTP tail disagrees with eager on this rank")
                    return False
            corr = min(_corr(g[0], w[0]) for g, w in zip(got, want, strict=True))
            top_ok = all(_top_agree(g[0], w[0]) for g, w in zip(got, want, strict=True))
            if state_corr is None:
                state_corr = _state_corr(got_state, want_state, total)
            log(
                f"prefill shape {name} vs eager: logit corr {corr:.5f}, state corr "
                f"{state_corr:.5f}, top-{EAGER_TOPK} agree {top_ok}"
            )
            if corr < EAGER_MIN_CORR or state_corr < EAGER_MIN_STATE_CORR or not top_ok:
                log(f"prefill shape {name} disagrees with eager on this rank")
                return False
        except Exception as exc:
            log(f"prefill shape {name} validation could not run: {exc!r}")
            return False
        finally:
            if VALIDATE_LOWMEM and torch.cuda.is_available():
                torch.cuda.empty_cache()  # hand this check's clones back to the node
        return True

    def _rewind(
        self,
        state: list[tuple[torch.Tensor, torch.Tensor]],
        calls: Sequence[tuple],
        hidden: torch.Tensor | None = None,
    ) -> None:
        """Back to the pre-call DeltaNet state (and MTP hidden seed). Target KV needs no
        rewind: every path rewrites the same rows with the same prefix visible. The calls' MTP
        KV rows are zeroed instead, so a replay that skips the MTP tail cannot pass by reading
        the previous run's rows (nothing reads them before the tail writes them)."""
        self._restore(state)
        if hidden is not None:
            model = self.model
            model.hidden_scratch.copy_(hidden)
            dev = model.hidden_scratch.device
            for slot, ids, start in calls:
                rows = model._physical_rows_range(slot, start, start + len(ids), dev)
                model.mtp.pool["k"][rows] = 0
                model.mtp.pool["v"][rows] = 0
        for slot, _, _ in calls:
            self._dirty[slot] = True


def _state_corr(got: list, want: list, lanes_total: int | None = None) -> float:
    """Minimum correlation over every DeltaNet layer's conv and recurrent state. With
    `lanes_total`, the clones hold the first lanes of a `lanes_total`-lane pool whose other
    lanes are zero on both sides, and the correlation is the whole pool's."""
    if lanes_total is None:
        return min(
            min(_corr(gc, wc), _corr(gr, wr))
            for (gc, gr), (wc, wr) in zip(got, want, strict=True)
        )
    return min(
        min(_corr_zero_padded(gc, wc, lanes_total), _corr_zero_padded(gr, wr, lanes_total))
        for (gc, gr), (wc, wr) in zip(got, want, strict=True)
    )


def _corr_zero_padded(got: torch.Tensor, want: torch.Tensor, lanes_total: int) -> float:
    """`_corr` of the two tensors each extended along dim 0 with zero lanes to `lanes_total`
    lanes, from float64 moments of the stored lanes only."""
    if not got.any() and not want.any():
        return 1.0 if torch.equal(got, want) else 0.0
    n = got.numel() / got.shape[0] * lanes_total
    g, w = got.double().flatten(), want.double().flatten()
    sg, sw = g.sum(), w.sum()
    vg = (g @ g) - sg * sg / n
    vw = (w @ w) - sw * sw / n
    if vg <= 0 or vw <= 0:
        return 1.0 if torch.equal(got, want) else 0.0
    return float(((g @ w) - sg * sw / n) / torch.sqrt(vg * vw))


def _corr(got: torch.Tensor, want: torch.Tensor) -> float:
    """Pearson correlation of two same-shape tensors (flattened, fp32). 1.0 if both are
    constant and equal (a zero state), 0.0 if only one is constant."""
    g, w = got.float().flatten(), want.float().flatten()
    g, w = g - g.mean(), w - w.mean()
    gn, wn = g.norm(), w.norm()
    if gn == 0 or wn == 0:
        return 1.0 if gn == wn and torch.equal(got, want) else 0.0
    return float((g @ w) / (gn * wn))


def _top_agree(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Each side's argmax is in the other's top `EAGER_TOPK` (a near-tie may swap the top two
    under rounding drift; a real bug moves the argmax far)."""
    k = min(EAGER_TOPK, got.numel())
    return bool(
        (got.argmax() == want.topk(k).indices).any()
        and (want.argmax() == got.topk(k).indices).any()
    )


def _close(got: torch.Tensor, want: torch.Tensor) -> bool:
    scale = want.float().abs().max().clamp(min=1e-6)
    return bool(((got.float() - want.float()).abs().max() / scale) <= VALIDATE_REL_TOL)
