"""Captured prefill under MTP (round 15, W10 mtp-serve): tail exactness and cost per shape.

Run with `SEED_MTP_SERVE=1` and the production prefill flags. Captures every shape in
`--shapes` in one `PrefillGraphRunner` (its `prepare` validates each shape, MTP tail
included, against the uncaptured step and the eager MTP prefill on every rank), then per shape:

- `tail`: the uncaptured static step's own final residual fed through the eager
  `mtp.prefill_cache` per row, versus the MTP K/V and `hidden_scratch` the step wrote (max
  abs error relative to max |ref|), and the replay versus the uncaptured step. Every rank.
- `graph_ms` / `eager_ms`: median synced wall of a captured replay versus the eager
  `Model.prefill_batch` (the only prefill path under MTP before this change), full rows at
  `--start` cached tokens.
- `notail_ms` (`--notail-shapes`): the same shape captured with the MTP head hidden, i.e.
  the incumbent step, to price the tail.

SPMD: rank 0 spawns ranks 1..tp-1 with the same argv.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graph_prefill
import mtp as mtp_mod
import torch
from graph_decode import CudaGraphBackend, GraphDecodeRunner
from graph_prefill import PrefillGraphRunner, Shape, parse_shapes

import server


def _calls(vocab: int, shape: Shape, start: int) -> list[tuple[int, list[int], int]]:
    return [
        (slot, [((17 * slot + 13 * j + 1) % vocab) for j in range(shape.width)], start)
        for slot in range(shape.rows)
    ]


def _sync_ms(fn, reps: int) -> float:  # noqa: ANN001
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.float().abs().max().clamp(min=1e-6)
    return float((got.float() - ref.float()).abs().max() / scale)


def _mtp_rows(model, calls):  # noqa: ANN001, ANN202
    dev = model.hidden_scratch.device
    rows = torch.cat([model._physical_rows_range(s, p, p + len(ids), dev) for s, ids, p in calls])
    lanes = torch.tensor([s for s, _, _ in calls], device=dev)
    pool = model.mtp.pool
    return [model.hidden_scratch[lanes].clone(), pool["k"][rows].clone(), pool["v"][rows].clone()]


def _tail_check(runner: PrefillGraphRunner, shape: Shape, start: int) -> dict:
    """Replay vs uncaptured step vs the eager `mtp.prefill_cache` on the step's residual."""
    model = runner.model
    runner._reset_lanes()
    calls = _calls(model.cfg.vocab, shape, start)
    buf, replay = runner.graphs[shape]
    runner.fill(buf, calls)
    gen = torch.Generator(device=model.hidden_scratch.device).manual_seed(shape.area)
    seed = (
        torch.randn(model.hidden_scratch.shape, generator=gen, device=gen.device).to(
            model.hidden_scratch.dtype
        )
        * 0.5
    )
    model.hidden_scratch.copy_(seed)
    dn = runner._deltanet_state()
    replay()
    got = _mtp_rows(model, calls)

    runner._restore(dn)
    model.hidden_scratch.copy_(seed)
    seen: dict = {}
    real_tail = graph_prefill._mtp_tail

    def spy(m, b, x):  # noqa: ANN001, ANN202
        seen["x"] = x.clone()
        real_tail(m, b, x)

    graph_prefill._mtp_tail = spy
    try:
        graph_prefill.prefill_step(model, buf)()
    finally:
        graph_prefill._mtp_tail = real_tail
    ref = _mtp_rows(model, calls)

    x = seen["x"].view(shape.rows, shape.width, -1)
    for j, (slot, ids, p) in enumerate(calls):
        mtp_mod.prefill_cache(
            model,
            model.mtp,
            slot,
            torch.tensor(ids, device=x.device),
            x[j, : len(ids)],
            p,
            seed[slot : slot + 1],
        )
        model.cache_hidden(slot, x[j, len(ids) - 1 : len(ids)])
    eager_tail = _mtp_rows(model, calls)
    torch.cuda.synchronize()
    names = ("hidden", "k", "v")
    return {
        "replay_vs_step": {n: _rel(g, r) for n, g, r in zip(names, got, ref, strict=True)},
        "step_vs_eager_tail": {
            n: _rel(r, e) for n, r, e in zip(names, ref, eager_tail, strict=True)
        },
        "step_eq_eager_tail": {
            n: bool(torch.equal(r, e)) for n, r, e in zip(names, ref, eager_tail, strict=True)
        },
    }


def _row_rel(got: torch.Tensor, ref: torch.Tensor) -> list[float]:
    """Per-row max abs error relative to that row's max |ref|."""
    g, r = got.float(), ref.float()
    return ((g - r).abs().amax(-1) / r.abs().amax(-1).clamp(min=1e-6)).tolist()


def _draft_check(runner: PrefillGraphRunner, shape: Shape, prefix: int) -> dict:
    """Captured vs eager MTP prefill as the draft sees it. Every lane first gets the same
    eager `prefix`-token prefill; then one full-width call per row runs (a) eagerly through
    `Model.prefill_batch`, (b) eagerly one row at a time (`Model.prefill`, the eager noise
    floor), (c) as the captured replay. Reports per-row hidden-seed error vs (a), target
    argmax and MTP draft agreement vs (a), and the SP gather check: the step's gathered
    residual, final-normed and unembedded at each row's last column, against the step's own
    logits (a wrong row order or shard offset moves the argmax)."""
    model = runner.model
    mtp = model.mtp
    runner._reset_lanes()
    vocab = model.cfg.vocab
    calls = _calls(vocab, shape, prefix)
    for slot, _, _ in calls:
        model.prefill(slot, [(7 * slot + 3 * j + 2) % vocab for j in range(prefix)], 0)
    dn, hid0 = runner._deltanet_state(), model.hidden_scratch.clone()
    lanes = [s for s, _, _ in calls]
    nxt_pos = [p + len(ids) for _, ids, p in calls]

    def outcome(logits):  # noqa: ANN001, ANN202
        top = [int(x.argmax()) for x in logits]
        for slot, p in zip(lanes, nxt_pos, strict=True):
            model.grow_lane(slot, p + mtp.k + 1)
        hidden = model.hidden_scratch[torch.tensor(lanes, device=hid0.device)].clone()
        drafts = mtp_mod.draft(model, mtp, hidden, top, lanes, nxt_pos)
        return hidden, top, drafts

    def rewind() -> None:
        runner._restore(dn)
        model.hidden_scratch.copy_(hid0)

    res = {}
    want_h, want_top, want_d = outcome(model.prefill_batch(calls))
    rewind()
    solo = [model.prefill(s, list(ids), p) for s, ids, p in calls]
    solo_h, solo_top, solo_d = outcome(solo)
    rewind()
    buf, replay = runner.graphs[shape]
    runner.fill(buf, calls)
    replay()
    got_logits = [buf.out[j : j + 1].clone() for j in range(len(calls))]
    got_h, got_top, got_d = outcome(got_logits)
    rewind()

    seen: dict = {}
    real_tail = graph_prefill._mtp_tail

    def spy(m, b, x):  # noqa: ANN001, ANN202
        seen["x"] = x.clone()
        real_tail(m, b, x)

    graph_prefill._mtp_tail = spy
    try:
        runner.fill(buf, calls)
        graph_prefill.prefill_step(model, buf)()
    finally:
        graph_prefill._mtp_tail = real_tail
    rewind()
    xs = seen["x"].view(shape.rows, shape.width, -1)
    last = torch.stack([xs[j, len(ids) - 1] for j, (_, ids, _) in enumerate(calls)])
    from model import rmsnorm  # noqa: PLC0415

    recomputed = model.unembed(rmsnorm(last, model.final_norm, model.cfg.eps)).float()
    step_out = buf.out[: len(calls)]
    if step_out.is_cuda:
        torch.cuda.synchronize()

    def agree(a, b) -> float:  # noqa: ANN001
        return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)

    res["gather_argmax_agree"] = agree(recomputed.argmax(-1).tolist(), step_out.argmax(-1).tolist())
    res["gather_logit_rel"] = max(_row_rel(recomputed, step_out))
    res["hidden_rel_graph_vs_batch"] = max(_row_rel(got_h, want_h))
    res["hidden_rel_solo_vs_batch"] = max(_row_rel(solo_h, want_h))
    res["top_agree_graph"] = agree(got_top, want_top)
    res["top_agree_solo"] = agree(solo_top, want_top)
    res["draft_agree_graph"] = agree([d[0] for d in got_d], [d[0] for d in want_d])
    res["draft_agree_solo"] = agree([d[0] for d in solo_d], [d[0] for d in want_d])
    res["draft_all_agree_graph"] = agree(got_d, want_d)
    res["draft_all_agree_solo"] = agree(solo_d, want_d)
    return {k: round(v, 5) for k, v in res.items()}


def run_rank(a: argparse.Namespace) -> None:
    args = argparse.Namespace(
        model_path=a.model_path,
        dtype="bfloat16",
        tp=a.tp,
        tp_port=a.port,
        max_seq_len=a.max_seq,
        max_batch=a.batch,
        devices="",
        rank=a.rank,
    )
    model = server.build_model(args, a.rank)
    assert model.mtp is not None, "set SEED_MTP_SERVE=1"
    decode = GraphDecodeRunner(model, prefill_graphs=False, mixed_graphs=False)
    device = model.devices[-1]
    backend = CudaGraphBackend()
    shapes = parse_shapes(a.shapes, model.max_batch, model.max_seq)
    t0 = time.perf_counter()
    runner = PrefillGraphRunner(model, backend, decode.lane_tables[device], decode._dirty, shapes)
    ok = runner.prepare()
    free, _ = torch.cuda.mem_get_info(device)
    print(
        f"PREPARE rank={a.rank} ok={ok} kept={[f'{s.rows}x{s.width}' for s in runner.shapes]} "
        f"s={time.perf_counter() - t0:.0f} free_gib={free / 2**30:.1f}",
        flush=True,
    )

    notail: dict[Shape, PrefillGraphRunner] = {}
    if a.notail_shapes:
        head = model.mtp
        model.mtp = None  # capture the incumbent step (no tail) for the same shapes
        try:
            for shape in parse_shapes(a.notail_shapes, model.max_batch, model.max_seq):
                r = PrefillGraphRunner(
                    model, backend, decode.lane_tables[device], decode._dirty, [shape]
                )
                if r.prepare():
                    notail[shape] = r
        finally:
            model.mtp = head

    if a.check_drafts:
        for shape in list(runner.shapes):
            name = f"{shape.rows}x{shape.width}"
            res = _draft_check(runner, shape, a.start)
            print(f"DRAFT rank={a.rank} {name} {json.dumps(res)}", flush=True)
        return

    results = []
    for shape in list(runner.shapes):
        name = f"{shape.rows}x{shape.width}"
        tail = _tail_check(runner, shape, a.start)
        print(f"TAIL rank={a.rank} {name} {json.dumps(tail)}", flush=True)
        runner._reset_lanes()
        calls = _calls(model.cfg.vocab, shape, a.start)
        buf, replay = runner.graphs[shape]
        runner.fill(buf, calls)
        graph_ms = _sync_ms(replay, a.reps)
        eager_ms = _sync_ms(lambda calls=calls: model.prefill_batch(calls), a.reps)
        res = {
            "shape": name,
            "area": shape.area,
            "graph_ms": round(graph_ms, 2),
            "eager_ms": round(eager_ms, 2),
        }
        if shape in notail:
            r = notail[shape]
            r._reset_lanes()
            nbuf, nreplay = r.graphs[shape]
            model.mtp, head = None, model.mtp
            try:
                r.fill(nbuf, calls)
            finally:
                model.mtp = head
            res["notail_ms"] = round(_sync_ms(nreplay, a.reps), 2)
        results.append(res)
        if a.rank == 0:
            print("SHAPE " + json.dumps(res), flush=True)
    if a.rank == 0:
        print("RESULT " + json.dumps(results), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29651)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--shapes", required=True)
    p.add_argument("--notail-shapes", default="")
    p.add_argument("--start", type=int, default=512)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--check-drafts", action="store_true")
    p.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
        ),
    )
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
    workers = [
        subprocess.Popen([*argv, "--rank", str(rank)], env=os.environ.copy())
        for rank in range(1, a.tp)
    ]
    try:
        run_rank(a)
    finally:
        for proc in workers:
            proc.wait(timeout=3600)


if __name__ == "__main__":
    main()
