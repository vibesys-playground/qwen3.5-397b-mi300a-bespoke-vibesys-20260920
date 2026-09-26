"""Round-15 W6 go/no-go: MTP verify ceiling at B96 on TP=4 (campaign diagnostic, not served).

Measures, in one TP=4 process group, on captured graphs:

1. Production B96 decode replay (the control), at the given positions.
2. A verify-shaped forward: a `96 x t` captured prefill (`graph_prefill.prefill_step`), one row
   per lane resuming at the lane's position, `t = k + 1` tokens per row. This is the
   verify forward's layout (per-query paged attention over the full context, DeltaNet resuming
   each lane's state, MoE/dense/collectives at `96 t` rows); it omits the LM head on
   non-final columns and the DeltaNet rollback, which are measured or bounded separately.
3. The LM head at `96 t` rows versus 96 rows (the extra verify logits).
4. The MTP draft (`graph_mtp.draft_round_step`, `k` MTP-layer steps) at B96, eager and
   captured.
5. Optionally, the existing exact verify graph (`graph_mtp.verify_round_step`) at B96: eager
   time, then capture and replay, which reproduces or clears the round-14 illegal memory access.

Every rank runs the same sequence. Results are JSON lines prefixed `CEILING`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch
from torch.profiler import ProfilerActivity, profile

import blas_tune
import graph_decode
import graph_prefill
import tp
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg

MODEL_PATH = os.environ["MODEL_PATH"]
OUT: Path


def emit(rank: int, name: str, **fields) -> None:  # noqa: ANN003
    rec = {"rank": rank, "name": name, **fields}
    with (OUT / f"ceiling_r{rank}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if rank == 0:
        print("CEILING " + json.dumps(rec, separators=(",", ":")), flush=True)


def windows(fn, steps: int, reps: int) -> list[float]:  # noqa: ANN001
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3 / steps)
    return out


def summary(ms: list[float]) -> dict:
    return {
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "windows_ms": ms,
    }


def profile_table(fn, steps: int, path: Path) -> None:  # noqa: ANN001
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        dev = getattr(ev, "self_device_time_total", None)
        if dev is None:
            dev = getattr(ev, "self_cuda_time_total", 0)
        if dev > 0:
            rows.append((dev / steps / 1e3, ev.count // steps, ev.key))
    rows.sort(reverse=True)
    with path.open("w") as fh:
        fh.write(f"# per-step self device ms, calls per step, kernel ({steps} steps)\n")
        fh.write(f"# total {sum(r[0] for r in rows):.3f} ms\n")
        for ms, n, key in rows:
            fh.write(f"{ms:9.4f} {n:6d} {key[:160]}\n")


def phase_accept(args: argparse.Namespace, rank: int, model: Model, runner, mtp) -> None:  # noqa: ANN001
    """Teacher-forced MTP acceptance on the benchmark's own first turns, eager, exact routing.

    Each step drafts `mtp.k` tokens from the lane's cached target hidden and last committed
    token, then runs one eager target decode step. `drafts[s][j]` predicts `target[s + 1 + j]`;
    tokens committed per round at draft length k is `1 + matching prefix of drafts[s][:k]`,
    averaged over every step `s` as a round start.
    """
    import importlib.util  # noqa: PLC0415

    from transformers import AutoTokenizer  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("bench_run", os.path.join(os.getcwd(), "benchmark", "run.py"))
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench_run"] = bench  # dataclasses resolve annotations through sys.modules
    sys.path.insert(0, os.path.join(os.getcwd(), "benchmark"))
    spec.loader.exec_module(bench)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    slots = list(range(args.batch))
    prompts = []
    for sid in slots:
        session = bench.make_session(bench.SEED, sid)
        msgs = [{"role": "user", "content": session.turns[0].user_text}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not all(isinstance(v, int) for v in ids):
            raise TypeError(f"tokenizer returned {type(ids[0])}")
        prompts.append(list(ids))
    dev = model.devices[-1]
    t0 = time.perf_counter()
    tokens, positions = [], []
    for lane, ids in zip(slots, prompts, strict=True):
        runner.begin(lane)
        logits = model.prefill(lane, ids, 0)
        tokens.append(int(logits[-1].argmax()))
        positions.append(len(ids))
    torch.cuda.synchronize()
    emit(rank, "accept_prefill", seconds=time.perf_counter() - t0,
         prompt_len_mean=statistics.mean(len(p) for p in prompts))
    k = mtp.k
    target = [list(tokens)]
    drafts = []
    slot_t = torch.tensor(slots, device=dev)
    t0 = time.perf_counter()
    for _ in range(args.accept_steps):
        for lane, p in zip(slots, positions, strict=True):
            model.grow_lane(lane, p + k + 1)
        hidden = model.hidden_scratch[slot_t].clone()
        drafts.append(mtp_mod_draft(model, mtp, hidden, tokens, slots, positions))
        logits = model.decode(slots, tokens, positions)
        tokens = logits.argmax(-1).tolist()
        positions = [p + 1 for p in positions]
        target.append(list(tokens))
    torch.cuda.synchronize()
    steps = len(drafts)
    per_k = {}
    for kk in range(1, k + 1):
        vals = []
        for s in range(steps - kk):
            for lane in slots:
                acc = 1
                for j in range(kk):
                    if drafts[s][lane][j] != target[s + 1 + j][lane]:
                        break
                    acc += 1
                vals.append(acc)
        per_k[kk] = {"mean_committed_per_round": statistics.mean(vals), "rounds": len(vals)}
    first_pos = [statistics.mean(
        1.0 if drafts[s][lane][0] == target[s + 1][lane] else 0.0 for s in range(steps - 1)
    ) for lane in slots]
    emit(rank, "acceptance", k=k, steps=steps, seconds=time.perf_counter() - t0, per_k=per_k,
         first_draft_hit_lane_min=min(first_pos), first_draft_hit_lane_max=max(first_pos),
         first_draft_hit_mean=statistics.mean(first_pos))
    if rank == 0:
        with (OUT / "accept_tokens.json").open("w") as fh:
            json.dump({"target": target, "drafts": drafts, "prompt_lens": [len(p) for p in prompts]}, fh)
    for lane in slots:
        runner.begin(lane)
    torch.cuda.synchronize()


def mtp_mod_draft(model, mtp, hidden, tokens, slots, positions):  # noqa: ANN001, ANN201
    import mtp as mtp_mod  # noqa: PLC0415

    return mtp_mod.draft(model, mtp, hidden, list(tokens), list(slots), list(positions))


def fill_wide(model: Model, runner, buf, rows_slot, rows_pos, rows_tok) -> None:  # noqa: ANN001
    """`GraphDecodeRunner.fill` for a bucket wider than `max_batch`: rows may repeat a lane."""
    from model import copy_from_host  # noqa: PLC0415

    dev = buf.pos.device
    need = {}
    for slot, pos in zip(rows_slot, rows_pos, strict=True):
        need[slot] = max(need.get(slot, 0), pos + 1)
    for slot, n in need.items():
        table = model.block_tables[slot]
        before = len(table.blocks)
        model.grow_lane(slot, n)
        if runner._dirty[slot] or len(table.blocks) != before:
            for lane_table in runner.lane_tables.values():
                lane_table.sync_lane(slot, table.blocks)
            runner._dirty[slot] = False
    ids = torch.tensor(rows_tok, dtype=torch.long, device=dev)[:, None]
    buf.x_in.copy_(torch.nn.functional.embedding(ids, model.embed))
    copy_from_host(buf.pos, torch.tensor(rows_pos, dtype=torch.long))
    copy_from_host(buf.active, torch.ones(len(rows_pos), dtype=torch.bool))
    copy_from_host(buf.slot_rows, torch.tensor(rows_slot, dtype=torch.long))
    bs = model.block_size
    table = runner.lane_tables[dev].table
    buf.block_table.copy_(table[buf.slot_rows])
    buf.block_valid.copy_(((buf.pos + bs) // bs).to(torch.int32))
    block_id = buf.block_table.gather(1, (buf.pos // bs).long()[:, None]).squeeze(1).long()
    buf.write_rows.copy_(block_id * bs + buf.pos % bs)


def phase_wide(args: argparse.Namespace, rank: int, model: Model, runner, pos_sets, tokens) -> None:  # noqa: ANN001
    """Verify forward through the decode graph's own kernels at `96 t` rows (lanes repeated,
    positions `p..p+t-1`), plus per-component captured graphs at the same row counts.

    Upper bound for a decode-quality verify: DeltaNet runs `t` independent rows per lane
    (each loads and stores the lane state), where a t-step kernel loads it once; the
    recurrence is therefore wrong but costs at least as much. Attention, MoE, projections and
    collectives are the decode graph's at `96 t` rows.
    """
    seg = runner.segments[0]
    dev = seg.device
    widths = [int(t) for t in args.wide_widths.split(",")]
    slots = list(range(args.batch))
    old_scratch = model.moe_scratch
    keep = []
    wide = {}
    rmax = args.batch * max(widths)
    scratch = [torch.empty(rmax, model.cfg.hidden, dtype=model.dtype, device=d) for d in model.layer_dev]
    keep.append(scratch)
    for t in widths:
        rows = args.batch * t
        buf = graph_decode.Buffers(model, seg, rows)
        rs = [lane for lane in slots for _ in range(t)]
        rt = [(tokens[lane] + 31 * c) % model.cfg.vocab for lane in slots for c in range(t)]
        rp = [pos_sets["profile"][lane] + c for lane in slots for c in range(t)]
        fill_wide(model, runner, buf, rs, rp, rt)
        model.moe_scratch = scratch
        try:
            step = graph_decode.segment_step(model, seg, buf)
            for _ in range(3):
                step()
            replay = runner.backend.capture(step, dev)
        finally:
            model.moe_scratch = old_scratch
        wide[t] = (buf, replay, rs, rt)
        emit(rank, "wide_capture", t=t, rows=rows, mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30)
    for pname, positions in pos_sets.items():
        for t in widths:
            buf, replay, rs, rt = wide[t]
            rp = [positions[lane] + c for lane in slots for c in range(t)]
            fill_wide(model, runner, buf, rs, rp, rt)
            replay()
            torch.cuda.synchronize()
            emit(rank, "wide_forward", positions=pname, t=t, rows=args.batch * t,
                 **summary(windows(replay, args.steps, args.reps)),
                 finite=bool(torch.isfinite(buf.out).all()))
        ms = windows(lambda positions=positions: runner.decode(slots, tokens, positions), args.steps, args.reps)
        emit(rank, "decode_pair", positions=pname, **summary(ms))
    if args.profile:
        for t in widths:
            buf, replay, rs, rt = wide[t]
            rp = [pos_sets["profile"][lane] + c for lane in slots for c in range(t)]
            fill_wide(model, runner, buf, rs, rp, rt)
            profile_table(replay, 3, OUT / f"kernels_r{rank}_wide{t}.txt")

    # Component graphs at each row count (profile positions), from the wide buffers.
    c = model.cfg
    attn_layers = [i for i, lt in enumerate(c.layer_types) if lt == "full_attention"]
    dn_layers = [i for i, lt in enumerate(c.layer_types) if lt != "full_attention"]
    gen = torch.Generator(device=dev).manual_seed(1234 + rank * 0)
    for t in widths:
        buf, _, rs, rt = wide[t]
        rows = args.batch * t
        rp = [pos_sets["profile"][lane] + cc for lane in slots for cc in range(t)]
        fill_wide(model, runner, buf, rs, rp, rt)
        h = (torch.randn(rows, 1, c.hidden, generator=gen, device=dev) * 0.5).to(model.dtype)
        part = (torch.randn(rows, 1, c.hidden, generator=gen, device=dev) * 0.5).to(model.dtype)
        comps = {
            "moe60": lambda h=h, rows=rows: [model.moe(i, h, scratch[i][:rows]) for i in range(len(model.layers))],
            "arnorm120": lambda h=h, part=part: [
                graph_decode._residual_norm(model, part, h, model.layers[i % len(model.layers)]["post_norm"])
                for i in range(2 * len(model.layers))
            ],
            "attn15": lambda h=h, buf=buf: [graph_decode.attn_decode_static(model, i, h, buf) for i in attn_layers],
            "dn45": lambda h=h, buf=buf: [graph_decode.deltanet_decode_static(model, i, h, buf) for i in dn_layers],
            "head": lambda h=h: model.unembed(h),
        }
        for name, fn in comps.items():
            try:
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                g = runner.backend.capture(fn, dev)
                g()
                torch.cuda.synchronize()
                emit(rank, "component", comp=name, t=t, rows=rows, **summary(windows(g, 5, 3)))
                keep.append(g)
            except Exception as exc:  # noqa: BLE001
                emit(rank, "component_failed", comp=name, t=t, error=repr(exc))
    keep.append(wide)
    runner._wide_keep = keep
    for lane in slots:
        runner.begin(lane)
    torch.cuda.synchronize()


def run_rank(args: argparse.Namespace, rank: int) -> None:
    reduce = tp.init(rank, args.tp, port=args.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, args.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, args.max_seq, args.batch, tp=handle)
    widths = [int(t) for t in args.widths.split(",")]
    verify_rows = sorted({args.batch * t for t in widths}
                         | {args.batch * int(t) for t in args.wide_widths.split(",")})
    prefill_widths = graph_prefill.gemm_widths(model.max_batch, model.max_seq)
    blas_tune.tune(
        model,
        batches=sorted({*blas_tune.tuned_batches(model.max_batch), *prefill_widths, *verify_rows}),
    )
    mtp = model.mtp
    model.mtp = None  # decode/prefill capture refuse MTP; restored for the draft phase
    runner = GraphDecodeRunner(model)
    if not runner.prepare():
        raise RuntimeError(f"rank {rank}: decode graph capture failed")
    model.mtp = mtp
    dev = model.devices[-1]
    emit(rank, "boot", mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30,
         mem_reserved_gib=torch.cuda.memory_reserved(dev) / 2**30, mtp_loaded=mtp is not None)

    slots = list(range(args.batch))
    tokens = [(7919 * lane + 1009) % model.cfg.vocab for lane in slots]
    pos_sets = {"profile": [int(v) for v in args.positions.split(",")]}
    if args.uniform_pos:
        pos_sets[f"u{args.uniform_pos}"] = [args.uniform_pos] * args.batch
    for pname, positions in pos_sets.items():
        if len(positions) != args.batch:
            raise ValueError(f"{pname}: {len(positions)} positions for batch {args.batch}")

    pr = runner.prefill_runner
    if pr is None:
        raise RuntimeError("prefill runner missing: set SEED_PREFILL_GRAPHS=1")

    phases = set(args.phases.split(","))
    if "accept" in phases:
        if mtp is None:
            raise RuntimeError("accept phase needs SEED_MTP=1")
        phase_accept(args, rank, model, runner, mtp)
    if "wide" in phases:
        phase_wide(args, rank, model, runner, pos_sets, tokens)
    if "verify" not in phases:
        return

    # Verify-shaped captured forwards, one per width, built once and replayed at each position set.
    verify = {}
    for t in widths:
        shape = graph_prefill.Shape(args.batch, t)
        buf = graph_prefill.PrefillBuffers(model, shape, dev)
        step = graph_prefill.prefill_step(model, buf)
        calls = [
            (lane, [(tokens[lane] + 31 * c) % model.cfg.vocab for c in range(t)], pos_sets["profile"][lane])
            for lane in slots
        ]
        pr.fill(buf, calls)
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        ref = buf.out.clone()
        t0 = time.perf_counter()
        replay = runner.backend.capture(step, dev)
        pr.fill(buf, calls)
        replay()
        torch.cuda.synchronize()
        rel = float((buf.out - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        verify[t] = (buf, replay, step)
        emit(rank, "verify_capture", t=t, rows=shape.area,
             capture_ms=(time.perf_counter() - t0) * 1e3, replay_vs_eager_rel=rel,
             mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30)

    for pname, positions in pos_sets.items():
        for _ in range(args.warmup):
            runner.decode(slots, tokens, positions)
        torch.cuda.synchronize()
        if runner.decode_path != "graph" or runner.decode_bucket != args.batch:
            raise RuntimeError(f"expected graph bucket {args.batch}, got {runner.decode_path}")
        ms = windows(lambda: runner.decode(slots, tokens, positions), args.steps, args.reps)
        emit(rank, "decode", positions=pname, pos_mean=statistics.mean(positions), **summary(ms))
        for t in widths:
            buf, replay, _ = verify[t]
            calls = [
                (lane, [(tokens[lane] + 31 * c) % model.cfg.vocab for c in range(t)], positions[lane])
                for lane in slots
            ]
            pr.fill(buf, calls)
            replay()
            torch.cuda.synchronize()
            ms_replay = windows(replay, args.steps, args.reps)

            def fill_replay(buf=buf, replay=replay, calls=calls) -> None:  # noqa: ANN001
                pr.fill(buf, calls)
                replay()

            ms_fill = windows(fill_replay, args.steps, max(2, args.reps // 2))
            emit(rank, "verify_forward", positions=pname, t=t, rows=args.batch * t,
                 replay=summary(ms_replay), fill_replay=summary(ms_fill),
                 finite=bool(torch.isfinite(buf.out).all()))
        # Paired re-measure of decode after the verify windows (drift check).
        ms = windows(lambda: runner.decode(slots, tokens, positions), args.steps, args.reps)
        emit(rank, "decode_after", positions=pname, **summary(ms))

    # Kernel attribution (same process, after timing windows).
    if args.profile:
        positions = pos_sets["profile"]
        profile_table(lambda: runner.decode(slots, tokens, positions), 3,
                      OUT / f"kernels_r{rank}_decode.txt")
        for t in widths:
            buf, replay, _ = verify[t]
            calls = [
                (lane, [(tokens[lane] + 31 * c) % model.cfg.vocab for c in range(t)], positions[lane])
                for lane in slots
            ]
            pr.fill(buf, calls)
            profile_table(replay, 3, OUT / f"kernels_r{rank}_verify{t}.txt")
        emit(rank, "profile_done")

    # Extra LM-head work of a verify round: logits on every column, not only the last.
    h = torch.randn(args.batch * max(widths), model.cfg.hidden, dtype=model.dtype, device=dev)
    for rows in [args.batch, *verify_rows]:
        x = h[:rows].contiguous()

        def head(x=x) -> None:  # noqa: ANN001
            model.unembed(x).float().argmax(-1)

        for _ in range(3):
            head()
        emit(rank, "lm_head", rows=rows, **summary(windows(head, 20, 3)))

    for buf, _, _ in verify.values():
        del buf
    verify.clear()
    torch.cuda.synchronize()

    if mtp is None:
        emit(rank, "draft_skipped", why="model.mtp is None (set SEED_MTP=1)")
        return
    import graph_mtp  # noqa: PLC0415

    t = mtp.k + 1
    positions = pos_sets["profile"]
    vr = graph_mtp.MTPVerifyRunner(model, mtp, runner.backend, lane_table=runner.lane_tables[dev],
                                   dirty=runner._dirty)
    vbuf = graph_mtp.MTPVerifyBuffers(model, args.batch, t, dev)
    model.hidden_scratch.normal_(0.0, 1.0)
    vr.fill(vbuf, slots, tokens, None, positions)
    draft = graph_mtp.draft_round_step(model, mtp, vbuf, t)
    for _ in range(3):
        draft()
    torch.cuda.synchronize()
    emit(rank, "draft_eager", k=mtp.k, **summary(windows(draft, 5, 3)))
    try:
        dreplay = runner.backend.capture(draft, dev)
        vr.fill(vbuf, slots, tokens, None, positions)
        dreplay()
        torch.cuda.synchronize()
        emit(rank, "draft_graph", k=mtp.k, **summary(windows(dreplay, args.steps, args.reps)))
        if args.profile:
            profile_table(dreplay, 3, OUT / f"kernels_r{rank}_draft.txt")
    except Exception as exc:  # noqa: BLE001
        emit(rank, "draft_graph_failed", error=repr(exc), tb=traceback.format_exc()[-2000:])
        return

    if not args.exact_verify:
        return
    # The existing exact verify graph (round 13/14 branch) at B96, forced drafts.
    step = graph_mtp.verify_round_step(model, mtp, vbuf, t)
    drafts = [[(tok + 1 + s) % model.cfg.vocab for s in range(mtp.k)] for tok in tokens]
    vr.fill(vbuf, slots, tokens, drafts, positions)
    step()
    torch.cuda.synchronize()

    def eager_verify() -> None:
        vr.fill(vbuf, slots, tokens, drafts, positions)
        step()

    emit(rank, "exact_verify_eager", t=t, **summary(windows(eager_verify, 3, 3)))
    emit(rank, "exact_verify_capture_start", t=t)
    vreplay = runner.backend.capture(step, dev)
    emit(rank, "exact_verify_captured", t=t)
    vr.fill(vbuf, slots, tokens, drafts, positions)
    vreplay()
    torch.cuda.synchronize()
    emit(rank, "exact_verify_first_replay_ok", t=t)
    emit(rank, "exact_verify_graph", t=t, **summary(windows(vreplay, args.steps, args.reps)))


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=29653)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--widths", default="2,3,4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--uniform-pos", type=int, default=4096)
    parser.add_argument("--profile", type=int, default=1)
    parser.add_argument("--exact-verify", type=int, default=1)
    parser.add_argument("--phases", default="verify")
    parser.add_argument("--wide-widths", default="1,2,3,4")
    parser.add_argument("--accept-steps", type=int, default=128)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.rank:
        run_rank(args, args.rank)
        return
    forwarded = [sys.executable, "-u", os.path.abspath(__file__)] + [
        a for a in sys.argv[1:]
    ]
    workers = [
        subprocess.Popen([*forwarded, "--rank", str(r)], env=os.environ.copy())
        for r in range(1, args.tp)
    ]
    try:
        run_rank(args, 0)
    finally:
        for w in workers:
            w.wait(timeout=900)


if __name__ == "__main__":
    main()
