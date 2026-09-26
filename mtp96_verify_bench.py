"""Round-15 W6 milestone: captured wide verify (`graph_verify_wide`) at B96, timing and a
sequential-decode differential. Campaign diagnostic, TP=4, every rank runs the same sequence.

1. Timing: verify graph at 96 active lanes (profile positions), paired with the B96 decode
   graph on the same process.
2. Differential: 48 benchmark first-turn prompts prefilled twice (lanes j and j + 48). Group A
   decodes greedily with the production decode graph. Group B runs two verify rounds with
   drafts forced from A's tokens (accept lengths 0/1/2 by lane), then plain decode graph steps.
   B's committed tokens and continuation must reproduce A's, which checks accept, KV, DeltaNet
   rollback, and hidden commit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch

import blas_tune
import graph_decode
import graph_mtp
import graph_prefill
import graph_verify_wide as gvw
import tp
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg

MODEL_PATH = os.environ["MODEL_PATH"]
OUT: Path


def emit(rank: int, name: str, **fields) -> None:  # noqa: ANN003
    rec = {"rank": rank, "name": name, **fields}
    with (OUT / f"verify_r{rank}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if rank == 0:
        print("VERIFY " + json.dumps(rec, separators=(",", ":")), flush=True)


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
    return {"median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms)}


def prompts(n: int) -> list[list[int]]:
    from transformers import AutoTokenizer  # noqa: PLC0415

    sys.path.insert(0, os.path.join(os.getcwd(), "benchmark"))
    spec = importlib.util.spec_from_file_location("bench_run", "benchmark/run.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench_run"] = bench
    spec.loader.exec_module(bench)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    out = []
    for sid in range(n):
        msgs = [{"role": "user", "content": bench.make_session(bench.SEED, sid).turns[0].user_text}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(list(tok(text, add_special_tokens=False)["input_ids"]))
    return out


def mtp_runner_fill_forced(buf, forced, k) -> None:  # noqa: ANN001
    """The forced part of `MTPVerifyRunner.fill`, for this bench's own fill."""
    from model import copy_from_host  # noqa: PLC0415

    n = [0] * buf.capacity
    ids = [[0] * k for _ in range(buf.capacity)]
    for j, f in enumerate(forced):
        n[j], ids[j][: len(f)] = len(f), list(f)
    copy_from_host(buf.forced_n, torch.tensor(n, dtype=torch.long))
    copy_from_host(buf.forced_ids, torch.tensor(ids, dtype=torch.long))


def agreement(args, rank, model, mtp, runner, t, scratch) -> None:  # noqa: ANN001, PLR0913, PLR0915
    """Numerics policy test: teacher-forced logits of a candidate path vs the production
    decode graph, over the benchmark's first turns.

    Three groups of `n = batch // 3` lanes: A and B hold the same prompts, C other prompts
    (filler). Each round, A decodes `t` steps alone along its own greedy continuation (the
    teacher, bucket n). Then B feeds the same `t` teacher tokens through the candidate:
    `control`, the decode graph with C in the same batch (bucket 2n: the incumbent's own
    batch-composition drift, the noise floor); `verify`, one wide verify round with the teacher
    as forced drafts. Every B row is compared with A's: top-1 agreement and KL(A || B).
    """
    import torch.nn.functional as F  # noqa: PLC0415

    dev = model.devices[-1]
    n = args.batch // 3
    a_slots, b_slots, c_slots = list(range(n)), list(range(n, 2 * n)), list(range(2 * n, 3 * n))
    abuf = graph_mtp.MTPVerifyBuffers(model, args.batch, t, dev)
    arow = gvw.row_buffers(model, args.batch * t)
    logits_out = torch.zeros(args.batch * t, model.cfg.vocab, dtype=torch.float32, device=dev)
    astep = gvw.build_verify_step(model, mtp, abuf, arow, t, scratch, logits_out)
    ps = prompts(2 * n)

    def reset() -> None:
        for lane in range(args.batch):
            model._release_lane_blocks(lane)
            runner.begin(lane)

    reset()
    warm = list(range(args.batch))
    gvw.host_fill(model, runner, abuf, arow, t, warm, [1] * args.batch, None, [8] * args.batch)
    astep()
    astep()
    torch.cuda.synchronize()
    areplay = runner.backend.capture(astep, dev)

    for mode in ("control", "verify"):
        reset()
        first, cfirst = [], []
        for j in range(n):
            for lane in (a_slots[j], b_slots[j]):
                logits = model.prefill(lane, ps[j], 0)
            first.append(int(logits[-1].argmax()))
            logits = model.prefill(c_slots[j], ps[n + j], 0)
            cfirst.append(int(logits[-1].argmax()))
        for lane in range(args.batch):
            runner._dirty[lane] = True
        lens = [len(ps[j]) for j in range(n)]
        clens = [len(ps[n + j]) for j in range(n)]
        teacher = [[f] for f in first]
        cseq = [[f] for f in cfirst]
        top1 = total = 0
        kl_sum = kl_max = 0.0
        at = 0
        for _ in range(args.agree_rounds):
            ref = []
            for c in range(t):
                lg = runner.decode(a_slots, [teacher[j][at + c] for j in range(n)],
                                   [lens[j] + at + c for j in range(n)])
                ref.append(lg.float())
                nxt = lg.argmax(-1).tolist()
                for j in range(n):
                    teacher[j].append(nxt[j])
            got = []
            if mode == "control":
                for c in range(t):
                    lg = runner.decode(
                        b_slots + c_slots,
                        [teacher[j][at + c] for j in range(n)] + [cseq[j][at + c] for j in range(n)],
                        [lens[j] + at + c for j in range(n)] + [clens[j] + at + c for j in range(n)],
                    )
                    got.append(lg[:n].float())
                    cn = lg[n:].argmax(-1).tolist()
                    for j in range(n):
                        cseq[j].append(cn[j])
            else:
                gvw.host_fill(model, runner, abuf, arow, t, b_slots, [teacher[j][at] for j in range(n)],
                              [[0] * (t - 1) for _ in range(n)], [lens[j] + at for j in range(n)])
                mtp_runner_fill_forced(abuf, [teacher[j][at + 1 : at + t] for j in range(n)], t - 1)
                areplay()
                view = logits_out.view(args.batch, t, -1)
                got = [view[:n, c].float() for c in range(t)]
            for c in range(t):
                p = F.log_softmax(ref[c], dim=-1)
                q = F.log_softmax(got[c], dim=-1)
                kl = (p.exp() * (p - q)).sum(-1)
                kl_sum += float(kl.sum())
                kl_max = max(kl_max, float(kl.max()))
                top1 += int((ref[c].argmax(-1) == got[c].argmax(-1)).sum())
                total += n
            at += t
        emit(rank, "agreement", mode=mode, tokens=total, top1=top1, top1_rate=top1 / total,
             mean_kl=kl_sum / total, max_kl=kl_max, rounds=args.agree_rounds, t=t, lanes=n)


def run_rank(args: argparse.Namespace, rank: int) -> None:
    reduce = tp.init(rank, args.tp, port=args.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, args.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, args.max_seq, args.batch, tp=handle)
    t = args.k + 1
    rows = args.batch * t
    prefill_widths = graph_prefill.gemm_widths(model.max_batch, model.max_seq)
    blas_tune.tune(
        model,
        batches=sorted({*blas_tune.tuned_batches(model.max_batch), *prefill_widths, rows}),
    )
    mtp = model.mtp
    if mtp is None or mtp.k != args.k:
        raise RuntimeError("set SEED_MTP=1 and SEED_MTP_K to --k")
    model.mtp = None
    runner = GraphDecodeRunner(model)
    if not runner.prepare():
        raise RuntimeError("decode capture failed")
    model.mtp = mtp
    dev = model.devices[-1]
    vocab = model.cfg.vocab

    slots = list(range(args.batch))
    positions = [int(v) for v in args.positions.split(",")]
    base = [(7919 * s + 1009) % vocab for s in slots]
    if args.runner_only:
        runner_phase(args, rank, model, mtp, runner, slots, base, positions)
        return
    buf = graph_mtp.MTPVerifyBuffers(model, args.batch, t, dev)
    rowbuf = gvw.row_buffers(model, rows)
    scratch = [torch.empty(rows, model.cfg.hidden, dtype=model.dtype, device=d) for d in model.layer_dev]
    step = gvw.build_verify_step(model, mtp, buf, rowbuf, t, scratch)

    drafts = [[(base[s] + 17 * (c + 1)) % vocab for c in range(args.k)] for s in slots]
    gvw.host_fill(model, runner, buf, rowbuf, t, slots, base, drafts, positions)
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    replay = runner.backend.capture(step, dev)
    gvw.host_fill(model, runner, buf, rowbuf, t, slots, base, drafts, positions)
    replay()
    torch.cuda.synchronize()
    emit(rank, "captured", t=t, rows=rows, capture_ms=(time.perf_counter() - t0) * 1e3,
         mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30)

    def fill_replay() -> None:
        gvw.host_fill(model, runner, buf, rowbuf, t, slots, base, drafts, positions)
        replay()

    for rep in range(2):
        ms_dec = windows(lambda: runner.decode(slots, base, positions), 10, 5)
        emit(rank, "decode", rep=rep, **summary(ms_dec))
        emit(rank, "verify_replay", rep=rep, t=t, **summary(windows(replay, 10, 5)))
        emit(rank, "verify_fill_replay", rep=rep, t=t, **summary(windows(fill_replay, 10, 3)))
    if args.profile:
        from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                replay()
            torch.cuda.synchronize()
        rows_out = sorted(
            ((getattr(e, "self_device_time_total", 0) / 3e3, e.count // 3, e.key) for e in prof.key_averages()),
            reverse=True,
        )
        with (OUT / f"kernels_r{rank}_verify.txt").open("w") as fh:
            fh.write(f"# total {sum(r[0] for r in rows_out):.3f} ms/step\n")
            for ms, n, key in rows_out[:80]:
                fh.write(f"{ms:9.4f} {n:6d} {key[:150]}\n")

    # ---------------------------------------------------------------- differential
    half = args.batch // 2
    ps = prompts(half)
    for lane in slots:
        model._release_lane_blocks(lane)
        runner.begin(lane)
    first = []
    for j, ids in enumerate(ps):
        for lane in (j, j + half):
            logits = model.prefill(lane, ids, 0)
        first.append(int(logits[-1].argmax()))
        runner._dirty[j] = runner._dirty[j + half] = True
    torch.cuda.synchronize()
    lens = [len(p) for p in ps]
    # Group A: plain decode graph, G tokens after the first.
    g = args.gen
    a_slots = list(range(half))
    seq = [[f] for f in first]  # seq[j][m] = token at position lens[j] + m
    for m in range(g):
        logits = runner.decode(a_slots, [seq[j][m] for j in a_slots], [lens[j] + m for j in a_slots])
        nxt = logits.argmax(-1).tolist()
        for j in a_slots:
            seq[j].append(nxt[j])
    torch.cuda.synchronize()

    # Group B: two verify rounds with forced drafts, then plain decode to the same length.
    b_slots = [j + half for j in a_slots]
    at = [0] * half  # index into seq of B's last committed token (not yet fed)
    rounds = []
    for rnd in range(2):
        want = [(j + rnd) % t for j in a_slots]  # forced accept length per lane (0..k)
        dr = []
        for j in a_slots:
            d = [seq[j][at[j] + 1 + s] for s in range(args.k)]
            if want[j] < args.k:
                d[want[j]] = (d[want[j]] + 1) % vocab
            dr.append(d)
        gvw.host_fill(model, runner, buf, rowbuf, t, b_slots, [seq[j][at[j]] for j in a_slots], dr,
                      [lens[j] + at[j] for j in a_slots])
        replay()
        acc = buf.accept_len[:half].tolist()
        am = buf.step_argmax[:half].tolist()
        ok_acc = sum(1 for j in a_slots if acc[j] == want[j])
        ok_tok = sum(
            1 for j in a_slots if am[j][: acc[j] + 1] == seq[j][at[j] + 1 : at[j] + 2 + acc[j]]
        )
        rounds.append({"round": rnd, "accept_match": ok_acc, "committed_match": ok_tok, "lanes": half})
        for j in a_slots:
            at[j] += acc[j] + 1
            runner._dirty[b_slots[j]] = True
    emit(rank, "differential_rounds", rounds=rounds)
    match_lanes, first_div = 0, []
    cont = {j: [] for j in a_slots}
    while True:
        live = [j for j in a_slots if at[j] < g]
        if not live:
            break
        logits = runner.decode([b_slots[j] for j in live], [seq[j][at[j]] for j in live],
                               [lens[j] + at[j] for j in live])
        nxt = logits.argmax(-1).tolist()
        for r, j in enumerate(live):
            cont[j].append((at[j] + 1, nxt[r]))
            at[j] += 1
    for j in a_slots:
        bad = [p for p, tok in cont[j] if tok != seq[j][p]]
        if not bad:
            match_lanes += 1
        else:
            first_div.append(bad[0])
    emit(rank, "differential_continuation", lanes=half, lanes_all_match=match_lanes,
         first_divergence_positions=first_div[:20], gen=g)

    # Forced feed (folded chat suffix): both groups now sit after seq[:g]. A feeds seq[g] and
    # the forced ids with plain decode; B runs one round with them as forced drafts (f = 2 on
    # even lanes, 1 on odd). B must accept exactly f and commit A's predictions.
    forced_ids = [248046, 198][: args.k]
    preds = [[] for _ in a_slots]
    feed = [seq[j][g] for j in a_slots]
    for c in range(args.k + 1):
        logits = runner.decode(a_slots, feed, [lens[j] + g + c for j in a_slots])
        nxt = logits.argmax(-1).tolist()
        for j in a_slots:
            preds[j].append(nxt[j])
        if c < args.k:
            feed = [forced_ids[c]] * half
    fn = [args.k if j % 2 == 0 else 1 for j in a_slots]
    gvw.host_fill(model, runner, buf, rowbuf, t, b_slots, [seq[j][g] for j in a_slots],
                  [[0] * args.k for _ in a_slots], [lens[j] + g for j in a_slots])
    mtp_runner_fill_forced(buf, [forced_ids[: fn[j]] for j in a_slots], args.k)
    replay()
    acc = buf.accept_len[:half].tolist()
    am = buf.step_argmax[:half].tolist()
    emit(rank, "differential_forced", lanes=half,
         accept_match=sum(1 for j in a_slots if acc[j] == fn[j]),
         committed_match=sum(1 for j in a_slots if am[j][: fn[j] + 1] == preds[j][: fn[j] + 1]))

    if args.agree_rounds:
        agreement(args, rank, model, mtp, runner, t, scratch)
    if args.runner:
        runner_phase(args, rank, model, mtp, runner, slots, base, positions)


def runner_phase(args, rank, model, mtp, runner, slots, base, positions) -> None:  # noqa: ANN001, PLR0913
    dev = model.devices[-1]
    # Served path: `MTPVerifyRunner` with `SEED_MTP_VERIFY_WIDE=1` (every bucket, boot
    # validation against the eager round), then full rounds (draft graph + wide verify).
    if not graph_mtp.VERIFY_WIDE:
        raise RuntimeError("set SEED_MTP_VERIFY_WIDE=1 for --runner")
    t0 = time.perf_counter()
    vr = graph_mtp.MTPVerifyRunner(
        model, mtp, runner.backend, lane_table=runner.lane_tables[dev], dirty=runner._dirty
    )
    ok = vr.prepare()
    emit(rank, "runner_prepare", ok=ok, seconds=time.perf_counter() - t0,
         mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30)
    if not ok:
        return
    emit(rank, "runner_buckets", buckets=vr.buckets)
    for n in (32, 48, 64, 80, 96):
        if n > args.batch:
            continue
        s_, b_, p_ = slots[:n], base[:n], positions[:n]
        budgets, stops = [1 << 20] * n, [[] for _ in s_]

        def round_(s_=s_, b_=b_, p_=p_, budgets=budgets, stops=stops) -> None:  # noqa: ANN001
            vr.speculative_decode(s_, b_, p_, budgets, stops)

        round_()
        emit(rank, "full_round", k=args.k, lanes=n, **summary(windows(round_, 5, 3)))
        emit(rank, "decode_pair_round", lanes=n,
             **summary(windows(lambda s_=s_, b_=b_, p_=p_: runner.decode(s_, b_, p_), 10, 3)))
    emit(rank, "eager_rounds", n=vr.eager_rounds)

    # Served loop: eager prefill of real first-turn prompts (seeds `hidden_scratch` and MTP KV
    # as the server does). Lanes [0, h) run captured MTP rounds feeding back committed tokens;
    # lanes [h, 2h) hold the same prompts and run the plain decode graph. Reports the per-depth
    # draft hit rate (draft s == target argmax at column s, given drafts < s hit), tokens per
    # round, and whether the MTP stream equals plain greedy decode.
    h = args.batch // 2
    ps = prompts(h)
    for lane in slots:
        model._release_lane_blocks(lane)
        runner.begin(lane)
    nxt, pos = [], []
    for j, ids in enumerate(ps):
        for lane in (j, j + h):
            logits = model.prefill(lane, ids, 0)
            runner._dirty[lane] = True
        nxt.append(int(logits[-1].argmax()))
        pos.append(len(ids))
    torch.cuda.synchronize()
    a = list(range(h))
    budgets, stops = [1 << 20] * h, [[] for _ in a]
    buf = vr.graphs[graph_decode.bucket_for(h, vr.buckets)].buf
    k = args.k
    hits, reach, per_round = [0] * k, [0] * k, []
    mtp_stream = [[x] for x in nxt]
    first = list(nxt)
    for _ in range(args.serve_rounds):
        out = vr.speculative_decode(a, nxt, pos, budgets, stops)
        tm = buf.token_matrix[:h, 1:].tolist()
        sa = buf.step_argmax[:h, :k].tolist()
        for j in a:
            for s_ in range(k):
                reach[s_] += 1
                if tm[j][s_] != sa[j][s_]:
                    break
                hits[s_] += 1
        per_round.append(sum(len(c) for c in out) / h)
        for j, c in enumerate(out):
            mtp_stream[j].extend(c)
            pos[j] += len(c)
            nxt[j] = c[-1]
    n_cmp = min(len(x) for x in mtp_stream)
    plain = [[x] for x in first]
    b = [j + h for j in a]
    for m in range(n_cmp - 1):
        logits = runner.decode(b, [plain[j][m] for j in a], [len(ps[j]) + m for j in a])
        for j, x in enumerate(logits.argmax(-1).tolist()):
            plain[j].append(x)
    torch.cuda.synchronize()
    same = sum(1 for j in a if mtp_stream[j][:n_cmp] == plain[j][:n_cmp])
    first_div = [next((m for m in range(n_cmp) if mtp_stream[j][m] != plain[j][m]), -1) for j in a]
    emit(rank, "served_loop", lanes=h, rounds=len(per_round),
         tok_per_lane_round=sum(per_round) / len(per_round), first4=per_round[:4],
         last4=per_round[-4:], draft_hit=[hits[s_] / max(reach[s_], 1) for s_ in range(k)],
         eager_rounds=vr.eager_rounds, compared_tokens=n_cmp, streams_equal=same,
         first_divergence=sorted(d for d in first_div if d >= 0)[:12])

def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=29661)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--gen", type=int, default=16)
    parser.add_argument("--profile", type=int, default=1)
    parser.add_argument("--runner", type=int, default=1)
    parser.add_argument("--agree-rounds", type=int, default=22)
    parser.add_argument("--runner-only", type=int, default=0)
    parser.add_argument("--serve-rounds", type=int, default=24)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.rank:
        run_rank(args, args.rank)
        return
    fwd = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]
    workers = [subprocess.Popen([*fwd, "--rank", str(r)], env=os.environ.copy()) for r in range(1, args.tp)]
    try:
        run_rank(args, 0)
    finally:
        for w in workers:
            w.wait(timeout=900)


if __name__ == "__main__":
    main()
