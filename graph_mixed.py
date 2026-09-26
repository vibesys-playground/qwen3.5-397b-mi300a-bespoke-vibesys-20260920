"""Graph-captured mixed decode+prefill step (`SEED_MIXED_GRAPH=1` plus `--enable-graph-capture`).

**Why.** With separate steps, every prefill forward stalls every decoding lane: the overlap
scheduler alternates decode and prefill steps, and at C48 prefill was 65-69% of the per-token
time (r4 logs) while the b48 decode graph itself is ~35 ms. The old `SEED_MIXED_BATCH` put both
halves into one forward but ran it eager (~260 ms per step), so it lost more to host dispatch
than it saved. This module captures that one forward as a graph.

**Idea.** Stall-free batching with chunked prefill (Sarathi-Serve, Agrawal et al., OSDI'24; the
same idea as vLLM's chunked-prefill scheduling): every iteration carries all decoding lanes'
next tokens plus a bounded prefill chunk, so decode never pauses for a whole prefill forward,
and the per-token work both halves share (norms, MoE, the two all-reduces per layer, the LM
head) runs once over the concatenated rows, reading each weight once. Re-derived here from the
paper; no code from either project.

**Shape.** `MixedShape(decode=D, prefill=RxW)`: `D` decode rows (a decode bucket, padded like
`graph_decode`'s buckets) followed by an `R x W` prefill chunk matrix (`graph_prefill`'s layout
and padding). The captured set is, per decode bucket in `SEED_MIXED_GRAPH_DECODE`, one prefill
row of width `T - D` for each total `T` in `SEED_MIXED_GRAPH_TOTALS` (`RxT`: `R` rows of
`(T - D) // R`, so several queued requests advance in one step; see `DEFAULT_TOTALS` for why
totals), plus the product with any `SEED_MIXED_GRAPH_PREFILL` shapes. A step replays the
smallest-area shape holding it and otherwise runs as separate decode and prefill steps (never
the eager mixed forward).

**One layer.** `x` is `[D + R*W, hidden]`, decode rows first. Every GEMM runs once over all
rows: the mixer's input projections (`q_proj`/`k_proj`/`v_proj`, or `in_proj_all`), its output
projection, the router, the routed and shared experts. Only the mixer cores split: decode rows
go through `graph_decode.attn_decode_core`/`deltanet_decode_core` against the decode `Buffers`
(one query per lane, recurrent state step), prefill rows through
`graph_prefill.attn_prefill_core`/`deltanet_prefill_core` against the `PrefillBuffers`
(causal chunk, varlen recurrence). Then one all-reduce, one MoE, one all-reduce. Exact for the
reason `Model.layer_mixed` gives: neither core reads or writes another row's KV or recurrent
state, and every other op is per row. This is where the saving is: a separate prefill step
reads every dense weight and every activated expert's weights again and pays the step's fixed
per-layer cost (launches, 120 all-reduces) again; here a short chunk costs its extra rows.

**State semantics.** Each half keeps its own path's contract unchanged: the prefill rows'
DeltaNet state, conv window and KV after the replay are what `PrefillGraphRunner` leaves
(identity steps on padded columns, conv gathered at the row's length, masked KV writes), and
the scheduler publishes snapshots after the step exactly as after a separate prefill
(`scheduler._finish_prefill_chunk`). Decode rows are `GraphDecodeRunner`'s replay rows.
Padding rows of either half point at a lane outside both halves when one exists (the `avoid`
argument of both `fill`s); when none exists (the pool is full), the decode mixer runs first in
every layer, so a prefill padding row pointed at a decode lane gathers and writes back that
lane's already-updated state unchanged, and a decode padding row writes back its gathered
value before the prefill mixer reads it.

**TP.** `replayable` reads only the broadcast call (`tp_driver.Op.DECODE_MIXED`), so every rank
decides the same way and issues the same collective sequence; blocks for both halves are
reserved on rank 0 before the command (`scheduler._mixed_step`'s one `_grow_lanes`).

    <python-with-torch> -m pytest seed_tests/test_graph_mixed.py -q -o addopts=
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import graph_decode
import graph_prefill
import moe_hip
import skinny_hip
import step_timing
import torch
import torch.nn.functional as F
from graph_decode import Buffers, CaptureBackend, GraphDecodeRunner, log
from graph_prefill import PrefillBuffers, PrefillGraphRunner, Shape
from model import Model, rmsnorm

MIXED_GRAPH = os.environ.get("SEED_MIXED_GRAPH", "0") not in ("0", "", "false", "False")
"""Capture and serve mixed decode+prefill steps (see the module docstring)."""

DEFAULT_DECODE = "16,32,48,64"
DEFAULT_TOTALS = "64,128,256,2x128,2x256"
"""Default shape set: for each decode bucket `D` and total `T` here, one prefill row of width
`T - D` (when `>= 16`); `RxT` gives `R` rows of `(T - D) // R`. The two-row totals let a short
chunk and the next queued request share a step: measured ramp at C48/C64, TTFT p95 1.35 s /
3.0 s without them vs 1.11 s / 1.52 s with, throughput 631 / 579 vs 623 / 611 tok/s. Totals, not prefill widths, because every shared GEMM runs at width
`D + R*W`: measured on 4x MI300A (`SEED_MIXED_GRAPH_BENCH`), a total that is not a BLAS-tuned
width (decode buckets and prefill areas are) costs up to +25 ms per step (64+1x16 = 80 rows:
62.7 ms vs 38.0 ms for 48+1x16 = 64 rows), and a total above `moe_hip.MAX_TOKENS` (256)
leaves the HIP MoE kernel (48+1x256: 148 ms mixed vs 103 ms as two separate graphs)."""

WIDE_TOTALS = "64,128,256,512,1024,2x128,2x256,2x512"
"""Default totals under `SEED_MOE_HIP_WIDE` (`moe_hip.WIDE`), where the HIP MoE serves every
width up to `moe_hip.MAX_TOKENS` and its cost grows far slower than the width: one rank,
real layer, 465 us at 256 rows, 562 at 512, 758 at 1024 (`bench_moe_widths.py`), so per
prefill token a 512- or 1024-row step is much cheaper than a 256-row one. The new totals'
GEMM widths are tuned at boot (`gemm_widths`)."""

FINE = os.environ.get("SEED_MIXED_FINE_TOTALS", "0") not in ("0", "", "false", "False")
"""`SEED_MIXED_FINE_TOTALS=1` (off by default; meant with `SEED_MOE_HIP_WIDE`): `FINE_TOTALS`,
and every mixed width BLAS-tuned at boot. Why: a step replays the smallest captured shape
holding its chunks, and with `WIDE_TOTALS` the ramp's mixed steps carried 57% real prefill
rows (512-row shapes 61%: 240-330 real of 464; 64/128-row shapes 13-19%, mostly the 5-token
chat-suffix chunk every turn ends with, riding a 64+2x32 shape). An intermediate total (384)
cuts the 512-row padding, and 80 gives the full decode bucket (64) a 1x16 row for those
suffix chunks instead of 2x32.

The set is capped at `WIDE_TOTALS`' size and 512 rows on purpose: every captured shape holds
its own buffers in the GPU's share of MI300A's unified memory, and a 13-total set (adding
320, 448, 2x384 and keeping 1024, 2x512: 51 graphs per rank) exhausted it during capture on
two nodes (the kernel OOM took the node down: NODE_FAIL). 1024-row steps were 2 of 94
sampled mixed steps in the WIDE ramp."""

FINE_TOTALS = "64,80,128,256,384,512,2x128,2x256,2x384"

if FINE:
    DEFAULT_TOTALS = FINE_TOTALS
elif moe_hip.WIDE:
    DEFAULT_TOTALS = WIDE_TOTALS
DEFAULT_PREFILL = ""

TUNE = os.environ.get("SEED_MIXED_GRAPH_TUNE", "0") not in ("0", "", "false", "False")
"""Tune BLAS solutions at every mixed shape's concatenated width (`D + R*W`, the router and
shared-expert GEMMs). Off by default: each new width costs minutes of cold-cache search at boot
(`blas_tune.tuned_batches`), and the mixer projections already run at tuned decode and prefill
widths."""

BENCH = os.environ.get("SEED_MIXED_GRAPH_BENCH", "0") not in ("0", "", "false", "False")
"""After capture, time each mixed shape's replay against its decode bucket's replay plus its
prefill shape's replay (when `SEED_PREFILL_GRAPHS` captured it) and log both."""

EAGER_MIN_CORR = graph_prefill.EAGER_MIN_CORR
EAGER_MIN_STATE_CORR = graph_prefill.EAGER_MIN_STATE_CORR


@dataclass(frozen=True, order=True)
class MixedShape:
    decode: int
    prefill: Shape

    @property
    def area(self) -> int:
        return self.decode + self.prefill.area

    @property
    def name(self) -> str:
        return f"{self.decode}+{self.prefill.rows}x{self.prefill.width}"


def parse_mixed_shapes(
    decode_spec: str, prefill_spec: str, max_batch: int, max_seq: int, totals_spec: str = ""
) -> list[MixedShape]:
    """Decode buckets times prefill shapes, plus for every decode bucket `D` and total `T` in
    `totals_spec` the one-row shape `D + 1x(T - D)` (when `T - D >= 16`), sorted by area.
    Drops decode sizes outside `[1, max_batch]` and prefill shapes whose rows leave no lane
    for a decode row."""
    decode = set()
    for item in (s.strip() for s in decode_spec.split(",")):
        if not item:
            continue
        if not item.isdigit() or int(item) < 1:
            raise ValueError(f"SEED_MIXED_GRAPH_DECODE: bad entry {item!r} (want an int)")
        if int(item) <= max_batch:
            decode.add(int(item))
    prefill = [
        s for s in graph_prefill.parse_shapes(prefill_spec, max_batch, max_seq) if s.rows < max_batch
    ]
    shapes = {MixedShape(d, p) for d in decode for p in prefill}
    for item in (s.strip() for s in totals_spec.split(",")):
        if not item:
            continue
        rows_s, sep, total_s = item.rpartition("x")
        rows = int(rows_s) if sep and rows_s.isdigit() else 1
        if not total_s.isdigit() or (sep and not rows_s.isdigit()) or rows < 1:
            raise ValueError(f"SEED_MIXED_GRAPH_TOTALS: bad entry {item!r} (want T or RxT)")
        for d in decode:
            width = (int(total_s) - d) // rows
            if 16 <= width <= max_seq and rows < max_batch:
                shapes.add(MixedShape(d, Shape(rows, width)))
    return sorted(shapes, key=lambda s: (s.area, s.decode, s.prefill.width))


def mixed_shape_for(
    n_decode: int, lengths: Sequence[int], shapes: Sequence[MixedShape]
) -> MixedShape | None:
    """The smallest-area shape with at least `n_decode` decode rows whose prefill half holds
    `len(lengths)` rows of `max(lengths)` tokens, or None."""
    if n_decode < 1 or not lengths:
        return None
    need_rows, need_width = len(lengths), max(lengths)
    for shape in shapes:  # sorted by area
        p = shape.prefill
        if shape.decode >= n_decode and p.rows >= need_rows and p.width >= need_width:
            return shape
    return None


def gemm_widths(max_batch: int, max_seq: int) -> tuple[int, ...]:
    """Concatenated GEMM widths (`D + R*W`) for `blas_tune.tune`: all of them under
    `SEED_MIXED_GRAPH_TUNE` or `SEED_MIXED_FINE_TOTALS`, empty without `SEED_MIXED_GRAPH`.
    Under `SEED_MOE_HIP_WIDE` alone the widths above 256 are tuned: no decode bucket or default prefill shape covers them,
    and the default heuristic's tiles cost several ms per step there."""
    if not MIXED_GRAPH or not (TUNE or FINE or moe_hip.WIDE):
        return ()
    areas = {s.area for s in _env_shapes(max_batch, max_seq)}
    return tuple(sorted(areas if TUNE or FINE else {a for a in areas if a > 256}))


def _env_shapes(max_batch: int, max_seq: int) -> list[MixedShape]:
    return parse_mixed_shapes(
        os.environ.get("SEED_MIXED_GRAPH_DECODE", DEFAULT_DECODE),
        os.environ.get("SEED_MIXED_GRAPH_PREFILL", DEFAULT_PREFILL),
        max_batch,
        max_seq,
        os.environ.get("SEED_MIXED_GRAPH_TOTALS", DEFAULT_TOTALS),
    )


def mixed_step(model: Model, dbuf: Buffers, pbuf: PrefillBuffers) -> Callable[[], None]:
    """The captured callable: embed the prefill tokens (the decode rows' embedding is
    `dbuf.x_in`, written by `GraphDecodeRunner.fill`), 60 layers with split mixers and shared
    norms/MoE/all-reduces, then the LM head on every decode row and on each prefill row's last
    real column. Writes `dbuf.out` (`[D, vocab]`) and `pbuf.out` (`[R, vocab]`)."""

    def step() -> None:
        c = model.cfg
        d = dbuf.capacity
        r, w = pbuf.shape.rows, pbuf.shape.width
        x_pre = F.embedding(pbuf.tokens, model.embed).reshape(r * w, c.hidden)
        x = torch.cat([dbuf.x_in.reshape(d, c.hidden), x_pre.to(dbuf.x_in.dtype)])
        def dec(t: torch.Tensor) -> torch.Tensor:
            return t[:d].reshape(d, 1, -1)

        def pre(t: torch.Tensor) -> torch.Tensor:
            return t[d:].reshape(r, w, -1)

        # `SEED_AR_RMSNORM_FUSED`: each all-reduce carries the residual add and the RMSNorm
        # after it (`graph_decode._residual_norm`, bit-exact against the unfused pair), and `h`
        # chains from one layer's second all-reduce into the next layer (or the final norm).
        chained = graph_decode.AR_RMSNORM_FUSED
        n = len(model.layers)
        if chained:
            h = rmsnorm(x, model.layers[0]["in_norm"], c.eps)
        # `SEED_AR_SP`: the residual `x` is kept as this rank's `rows / 4` row shard and each
        # all-reduce is a reduce-scatter + add + norm + all-gather (`sp_ar_add_rmsnorm`), so
        # `h` is whole again after every call. Bit-identical to the chained path.
        cr = getattr(model.tp, "custom_reduce", None)
        sp = chained and cr is not None and _sp_rows_ok(model, d + r * w)
        if sp:
            s_rows = (d + r * w) // 4
            x = x[model.tp.rank * s_rows : (model.tp.rank + 1) * s_rows].contiguous()
        for i in range(n):
            lw = model.layers[i]
            if not chained:
                h = rmsnorm(x, lw["in_norm"], c.eps)
            # Every GEMM runs once over all `d + r*w` rows; only the cores split.
            if c.layer_types[i] == "full_attention":
                qg = skinny_hip.linear(h, lw["q_proj"])
                k, v = skinny_hip.linear(h, lw["k_proj"]), skinny_hip.linear(h, lw["v_proj"])
                core_dec = graph_decode.attn_decode_core(model, i, dec(qg), dec(k), dec(v), dbuf)
                core_pre = graph_prefill.attn_prefill_core(model, i, pre(qg), pre(k), pre(v), pbuf)
                out_w = lw["o_proj"]
            else:
                proj = skinny_hip.linear(h, lw["in_proj_all"])
                core_dec = graph_decode.deltanet_decode_core(model, i, dec(proj), dbuf)
                core_pre = graph_prefill.deltanet_prefill_core(model, i, pre(proj), pbuf)
                out_w = lw["out_proj"]
            core = torch.cat([core_dec.reshape(d, -1), core_pre.reshape(r * w, -1)])
            if sp:
                nxt = model.layers[i + 1]["in_norm"] if i + 1 < n else model.final_norm
                x, h_mid = cr.sp_ar_add_rmsnorm(
                    skinny_hip.linear(core, out_w).contiguous(), x, lw["post_norm"], c.eps
                )
                x, h = cr.sp_ar_add_rmsnorm(model.moe(i, h_mid).contiguous(), x, nxt, c.eps)
                continue
            if chained:
                nxt = model.layers[i + 1]["in_norm"] if i + 1 < n else model.final_norm
                x, h_mid = graph_decode._residual_norm(
                    model, skinny_hip.linear(core, out_w), x, lw["post_norm"]
                )
                x, h = graph_decode._residual_norm(model, model.moe(i, h_mid), x, nxt)
                continue
            x = x + model.tp.all_reduce(skinny_hip.linear(core, out_w))
            x = x + model.tp.all_reduce(model.moe(i, rmsnorm(x, lw["post_norm"], c.eps)))
        last = d + torch.arange(r, device=x.device) * w + pbuf.length - 1
        rows = torch.cat([torch.arange(d, device=x.device), last])
        # A row-wise norm: `h[rows]` is `rmsnorm(x[rows])` bit for bit.
        normed = h[rows] if chained else rmsnorm(x[rows], model.final_norm, c.eps)
        logits = model.unembed(normed).float()
        dbuf.out.copy_(logits[:d])
        pbuf.out.copy_(logits[d:])

    return step


def _sp_rows_ok(model: Model, rows: int) -> bool:
    """Whether a `rows`-row mixed step takes the `SEED_AR_SP` all-reduce: 4 ranks, bf16, and
    `rows / 4` within the SP kernel's flag table and buffers (`CustomAllReduce.sp_ok`)."""
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


class MixedGraphRunner:
    """Captures one graph per `MixedShape` and replays mixed steps that fit one. Shares the
    decode runner's device block-table mirror and dirty flags, and uses its `fill` for the
    decode half and a `PrefillGraphRunner`'s `fill` for the prefill half, so each half's
    buffers are built exactly as its own graph's are."""

    def __init__(
        self,
        decode_runner: GraphDecodeRunner,
        backend: CaptureBackend,
        shapes: Sequence[MixedShape] | None = None,
    ) -> None:
        self.dr = decode_runner
        self.model = model = decode_runner.model
        self.backend = backend
        self.shapes = (
            list(shapes) if shapes is not None else _env_shapes(model.max_batch, model.max_seq)
        )
        self.device = model.devices[-1]
        lane_table = decode_runner.lane_tables.get(self.device)
        self.prefill = PrefillGraphRunner(model, backend, lane_table, decode_runner._dirty, [])
        self.graphs: dict[MixedShape, tuple[Buffers, PrefillBuffers, Callable[[], None]]] = {}
        self.enabled = False
        self.last_shape: MixedShape | None = None  # tests
        self.replays = 0  # served steps since `prepare` (tests)
        self._timing_replays = 0

    def supported(self) -> tuple[bool, str]:
        segments = graph_decode.plan_segments(self.model.layer_dev)
        if len(segments) != 1:
            return False, "needs a single-segment layout (TP owns one device)"
        if self.model.mtp is not None:
            return False, "MTP's eager prefill also seeds hidden_scratch"
        if not self.dr.enabled:
            return False, "decode graphs are off"
        if not self.shapes:
            return False, "no usable shape (SEED_MIXED_GRAPH_DECODE/_TOTALS/_PREFILL)"
        return True, ""

    # -- dispatch -------------------------------------------------------------
    def fit(self, n_decode: int) -> tuple[Callable[[Sequence[int]], bool], int, int] | None:
        """`(fits(lengths), max_width, max_tokens)` for a step with `n_decode` decode rows, or
        None when no captured shape has that many decode rows. Rank 0's scheduler shapes a
        mixed step's prefill chunks with it."""
        if not self.enabled:
            return None
        usable = [s for s in self.shapes if s.decode >= n_decode]
        if not usable:
            return None
        return (
            lambda lengths: mixed_shape_for(n_decode, lengths, usable) is not None,
            max(s.prefill.width for s in usable),
            max(s.prefill.area for s in usable),
        )

    def replayable(
        self,
        slots: Sequence[int],
        positions: Sequence[int],
        calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> MixedShape | None:
        """The shape serving this step, or None. Reads only the broadcast step, so every rank
        decides the same way."""
        if not self.enabled or not slots or not calls:
            return None
        if not self.dr.replayable(slots, positions):
            return None
        lanes = [slot for slot, _, _ in calls]
        if set(lanes) & set(slots):
            return None
        if not self.prefill.valid_calls(calls):
            return None
        return mixed_shape_for(len(slots), [len(ids) for _, ids, _ in calls], self.shapes)

    def run(
        self,
        shape: MixedShape,
        slots: Sequence[int],
        tokens: Sequence[int],
        positions: Sequence[int],
        calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`Model.decode_mixed`'s contract from a replay. A failed replay disables the runner
        and re-raises (it may have advanced some lanes' state; same policy as the decode and
        prefill runners)."""
        dbuf, pbuf, replay = self.graphs[shape]
        timed = False
        if step_timing.ENABLED:
            self._timing_replays += 1
            timed = self._timing_replays % step_timing.EVERY == 0
            t0 = time.perf_counter()
        self.fill(dbuf, pbuf, list(slots), list(tokens), list(positions), calls)
        if timed:
            t_fill = time.perf_counter()
        try:
            replay()
        except Exception as exc:
            self.enabled = False
            log(f"mixed replay failed, mixed steps off from here on: {exc!r}")
            raise
        if timed:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()
            step_timing.log(
                f"rank {self.model.tp.plan.rank} graph mixed n={self._timing_replays} "
                f"shape={shape.name} decode={len(slots)} prefill_tokens="
                f"{sum(len(ids) for _, ids, _ in calls)} fill_host_ms={(t_fill - t0) * 1e3:.2f} "
                f"replay_ms={(t1 - t_fill) * 1e3:.1f}"
            )
        self.last_shape = shape
        self.replays += 1
        out = pbuf.out[: len(calls)].clone()
        return dbuf.out[: len(slots)].clone(), [out[j : j + 1] for j in range(len(calls))]

    def fill(
        self,
        dbuf: Buffers,
        pbuf: PrefillBuffers,
        slots: list[int],
        tokens: list[int],
        positions: list[int],
        calls: Sequence[tuple[int, Sequence[int], int]],
    ) -> None:
        lanes = [slot for slot, _, _ in calls]
        self.dr.fill([dbuf], dbuf.capacity, slots, tokens, positions, avoid=lanes)
        self.prefill.fill(pbuf, calls, avoid=slots)

    # -- setup ----------------------------------------------------------------
    def prepare(self) -> bool:
        """Warm, capture and validate every shape; agree across ranks. Never raises (a failure
        leaves mixed steps off on every rank). Boot-only: warmup and validation grow and reset
        lanes with the model's own allocator."""
        ok, why = self.supported()
        if not ok:
            log(f"mixed capture off: {why}")
            self.enabled = False
            return self._agree(False)
        segment = graph_decode.plan_segments(self.model.layer_dev)[0]
        try:
            self.graphs = {}
            for shape in self.shapes:
                t0 = time.perf_counter()
                dbuf = Buffers(self.model, segment, shape.decode)
                pbuf = PrefillBuffers(self.model, shape.prefill, self.device)
                step = mixed_step(self.model, dbuf, pbuf)
                self._warm(dbuf, pbuf, step, shape)
                self.graphs[shape] = (dbuf, pbuf, self.backend.capture(step, self.device))
                log(f"captured mixed {shape.name} in {(time.perf_counter() - t0) * 1e3:.0f} ms")
            self.enabled = True
        except Exception as exc:
            self.enabled = False
            log(f"mixed capture failed, mixed steps off: {exc!r}")
        ok = self._agree(self.enabled)
        if ok:
            ok = self._agree(self._validate())
        self.dr.reset_slots()
        if ok and BENCH:
            self._bench()
            self.dr.reset_slots()
        self.replays = 0
        if ok:
            log(f"mixed graphs enabled for shapes {[s.name for s in self.shapes]}")
        return ok

    def _agree(self, ok: bool) -> bool:
        self.enabled = self.dr.agree(ok)
        return self.enabled

    def _synthetic(
        self, shape: MixedShape
    ) -> tuple[list[int], list[int], list[tuple[int, list[int], int]]]:
        """Decode lanes first (as many as the bucket holds, leaving a lane per prefill row),
        then one prefill call per row but the last when the shape has several rows (a padding
        row), lengths 1..width, each resuming at start 3."""
        vocab = self.model.cfg.vocab
        p = shape.prefill
        n_calls = p.rows - 1 if p.rows > 1 else 1
        n_dec = min(shape.decode, self.model.max_batch - n_calls)
        if shape.decode > 1 and n_dec == shape.decode:
            n_dec -= 1  # leave a decode padding row too
        slots = list(range(n_dec))
        tokens = [(7 * j + 3) % vocab for j in range(n_dec)]
        calls = []
        for j in range(n_calls):
            n = p.width if j == 0 else 1 + (7 * j) % p.width
            calls.append((n_dec + j, [(11 * j + 5 * t + 1) % vocab for t in range(n)], 3))
        return slots, tokens, calls

    def _warm(
        self, dbuf: Buffers, pbuf: PrefillBuffers, step: Callable[[], None], shape: MixedShape
    ) -> None:
        slots, tokens, calls = self._synthetic(shape)
        self.fill(dbuf, pbuf, slots, tokens, [0] * len(slots), [(s, i, 0) for s, i, _ in calls])
        for _ in range(graph_decode.WARMUP_STEPS):
            step()
        self.dr.reset_slots()

    def _seed_lanes(self, slots: Sequence[int], calls: Sequence[tuple]) -> None:
        """Every lane of the synthetic step resumes a 3-token prefix, prefilled eagerly."""
        self.dr.reset_slots()
        for lane in [*slots, *(s for s, _, _ in calls)]:
            self.model.prefill(lane, [2, 3, 4], 0)
            self.dr._dirty[lane] = True

    def _validate(self) -> bool:
        """Every shape on its synthetic step (`_check`), each verdict agreed across ranks
        before it is acted on (`graph_prefill.PrefillGraphRunner._validate`'s policy): a shape
        any rank rejects is dropped on every rank. True when at least one shape survives."""
        kept = []
        for shape in self.shapes:
            if self.dr.agree(self._check(shape)):
                kept.append(shape)
            else:
                log(f"mixed shape {shape.name} off on every rank")
                del self.graphs[shape]
        self.shapes = kept
        if kept:
            log(f"mixed: {len(kept)} shape(s) match the eager mixed step on every rank")
        return bool(kept)

    def _check(self, shape: MixedShape) -> bool:
        """One shape on its synthetic step, two checks (`graph_prefill`'s policy):

        1. Capture fidelity: replay vs the same static step run uncaptured, within
           `VALIDATE_REL_TOL` (only a capture bug fails).
        2. Semantics: replay vs the eager `Model.decode_mixed` (whose prefill half is the
           eager packed prefill), by per-row logit correlation and top-k agreement, and
           DeltaNet state correlation.
        """
        model = self.model
        try:
            slots, tokens, calls = self._synthetic(shape)
            positions = [3] * len(slots)
            lanes = [*slots, *(s for s, _, _ in calls)]
            self._seed_lanes(slots, calls)
            before = self.prefill._deltanet_state()
            want_dec, want_pre = model.decode_mixed(slots, tokens, positions, calls)
            want_state = self.prefill._deltanet_state()
            self._rewind(before, lanes)
            got_dec, got_pre = self.run(shape, slots, tokens, positions, calls)
            got_state = self.prefill._deltanet_state()
            self._rewind(before, lanes)
            dbuf, pbuf, _ = self.graphs[shape]
            self.fill(dbuf, pbuf, slots, tokens, positions, calls)
            mixed_step(model, dbuf, pbuf)()
            ref_dec = dbuf.out[: len(slots)].clone()
            ref_pre = pbuf.out[: len(calls)].clone()
            ref_state = self.prefill._deltanet_state()

            close = graph_prefill._close
            faithful = (
                close(got_dec, ref_dec)
                and all(close(g[0], r) for g, r in zip(got_pre, ref_pre, strict=True))
                and all(
                    close(gc, rc) and close(gr, rr)
                    for (gc, gr), (rc, rr) in zip(got_state, ref_state, strict=True)
                )
            )
            if not faithful:
                log(f"mixed shape {shape.name} replay disagrees with its uncaptured step")
                return False
            pairs = [(got_dec[j], want_dec[j]) for j in range(len(slots))]
            pairs += [(g[0], w[0]) for g, w in zip(got_pre, want_pre, strict=True)]
            corr = min(graph_prefill._corr(g, w) for g, w in pairs)
            top_ok = all(graph_prefill._top_agree(g, w) for g, w in pairs)
            state_corr = min(
                min(graph_prefill._corr(gc, wc), graph_prefill._corr(gr, wr))
                for (gc, gr), (wc, wr) in zip(got_state, want_state, strict=True)
            )
            log(
                f"mixed shape {shape.name} vs eager: logit corr {corr:.5f}, state corr "
                f"{state_corr:.5f}, top-{graph_prefill.EAGER_TOPK} agree {top_ok}"
            )
            if corr < EAGER_MIN_CORR or state_corr < EAGER_MIN_STATE_CORR or not top_ok:
                log(f"mixed shape {shape.name} disagrees with eager on this rank")
                return False
        except Exception as exc:
            log(f"mixed shape {shape.name} validation could not run: {exc!r}")
            return False
        return True

    def _rewind(self, state: list, lanes: Sequence[int]) -> None:
        """Back to the pre-step DeltaNet state (KV needs none: every path rewrites the same
        rows with the same prefix visible)."""
        self.prefill._restore(state)
        for lane in lanes:
            self.dr._dirty[lane] = True

    def _bench(self, reps: int = 5) -> None:
        """`SEED_MIXED_GRAPH_BENCH`: per shape, mean ms of the mixed replay vs its decode
        bucket's replay plus its prefill shape's replay (same rows, run as two graphs)."""
        if self.device.type != "cuda":
            return
        pr = self.dr.prefill_runner

        def timed(fn: Callable[[], None]) -> float:
            fn()
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize(self.device)
            return (time.perf_counter() - t0) * 1e3 / reps

        for shape in self.shapes:
            slots, tokens, calls = self._synthetic(shape)
            positions = [3] * len(slots)
            self._seed_lanes(slots, calls)
            dbuf, pbuf, replay = self.graphs[shape]
            self.fill(dbuf, pbuf, slots, tokens, positions, calls)
            mixed_ms = timed(replay)
            dec = self.dr.graphs.get(shape.decode)
            dec_ms = timed(dec.replays[0]) if dec is not None else float("nan")
            sep = pr.graphs.get(shape.prefill) if pr is not None and pr.enabled else None
            pre_ms = timed(sep[1]) if sep is not None else float("nan")
            log(
                f"mixed bench {shape.name}: mixed {mixed_ms:.1f} ms vs decode b{shape.decode} "
                f"{dec_ms:.1f} ms + prefill {shape.prefill.rows}x{shape.prefill.width} "
                f"{pre_ms:.1f} ms = {dec_ms + pre_ms:.1f} ms"
            )
