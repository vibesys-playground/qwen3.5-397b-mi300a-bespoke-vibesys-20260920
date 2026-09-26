"""Round-15 W7: MTP draft on production kernels versus the original draft, at B96 on TP=4.

Campaign diagnostic, not served. One TP=4 process group; every rank runs the same sequence.

1. Timing (W6's "before" condition): random `hidden_scratch`, the ceiling harness's position
   profile, captured draft graphs (`graph_mtp.draft_round_step`) for k in `--ks`, original
   (`SEED_MTP_DRAFT_FAST=0` spelling) and fast, interleaved windows so the pair is same-node
   and same-process. Kernel tables for both at k=2.
2. Agreement and acceptance on real hidden states: the benchmark's first-turn prompts prefilled
   into `--batch` lanes, then `--accept-steps` eager target decode steps. Before each step both
   captured draft graphs (largest k) run from the same lane state; per-position token agreement
   and teacher-forced acceptance per k are reported for each.

JSON lines prefixed `DRAFT`, also written to `<out>/draft_r<rank>.jsonl`.
"""

from __future__ import annotations

import argparse
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
import mtp as mtp_mod
import tp
from model import Model, load_cfg
from mtp96_ceiling import profile_table, summary, windows

MODEL_PATH = os.environ["MODEL_PATH"]
OUT: Path

W6_POSITIONS = (
    "44,83,103,130,193,232,247,264,274,289,303,310,337,358,364,380,397,404,408,417,424,454,466,"
    "477,495,508,528,544,585,603,613,628,641,655,672,708,718,732,745,752,758,760,772,799,811,819,"
    "833,847,871,897,901,905,916,927,938,957,966,976,979,986,1003,1015,1036,1054,1075,1120,1151,"
    "1168,1195,1215,1233,1249,1287,1313,1342,1350,1372,1421,1468,1482,1526,1543,1556,1598,1648,"
    "1702,1741,1771,1852,1996,2062,2131,2270,2343,2472,2556"
)


def emit(rank: int, name: str, **fields) -> None:  # noqa: ANN003
    rec = {"rank": rank, "name": name, **fields}
    with (OUT / f"draft_r{rank}.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if rank == 0:
        print("DRAFT " + json.dumps(rec, separators=(",", ":")), flush=True)


def build(model: Model, mtp, runner, buf, t: int, fast: bool):  # noqa: ANN001, ANN201
    """Warm and capture one draft graph with `mtp.DRAFT_FAST = fast` baked in."""
    mtp_mod.DRAFT_FAST = fast
    step = graph_mtp.draft_round_step(model, mtp, buf, t)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    eager = windows(step, 3, 3)
    replay = runner.backend.capture(step, model.devices[-1])
    replay()
    torch.cuda.synchronize()
    return step, replay, eager


def prompts_for(batch: int) -> list[list[int]]:
    import importlib.util  # noqa: PLC0415

    from transformers import AutoTokenizer  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "bench_run", os.path.join(os.getcwd(), "benchmark", "run.py")
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules["bench_run"] = bench
    sys.path.insert(0, os.path.join(os.getcwd(), "benchmark"))
    spec.loader.exec_module(bench)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    out = []
    for sid in range(batch):
        session = bench.make_session(bench.SEED, sid)
        msgs = [{"role": "user", "content": session.turns[0].user_text}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(list(tok(text, add_special_tokens=False)["input_ids"]))
    return out


def acceptance(drafts: list[list[list[int]]], target: list[list[int]], k: int) -> float:
    vals = []
    for s in range(len(drafts) - k):
        for lane in range(len(drafts[s])):
            acc = 1
            for j in range(k):
                if drafts[s][lane][j] != target[s + 1 + j][lane]:
                    break
                acc += 1
            vals.append(acc)
    return statistics.mean(vals)


def bucket_check(args: argparse.Namespace, rank: int, model: Model, mtp, base) -> None:  # noqa: ANN001
    """Capture the fast draft at small and serving buckets, with padding rows, and replay it
    against its own eager run over several fills. Host-side bounds checks on every index the
    draft reads (run once under AMD_SERIALIZE_KERNEL=3 to localize a device fault)."""
    import random  # noqa: PLC0415

    k = max(int(v) for v in args.ks.split(","))
    t, dev = k + 1, model.devices[-1]
    rows_cap = mtp.pool["k"].shape[0]
    rnd = random.Random(1234)  # same on every rank: identical fills, identical collectives
    mtp_mod.DRAFT_FAST = True
    model.hidden_scratch.normal_(0.0, 1.0, generator=torch.Generator(device=dev).manual_seed(5))
    for capacity in [int(v) for v in args.buckets.split(",")]:
        runner = graph_mtp.MTPVerifyRunner(model, mtp, base.backend, lane_table=base.lane_table,
                                           dirty=base._dirty)
        runner.t = t
        buf = graph_mtp.MTPVerifyBuffers(model, capacity, t, dev)

        def fill(n: int) -> None:
            lanes = rnd.sample(range(model.max_batch), n)
            # positions straddle block boundaries: 16 k - 1 - s for small s
            pos = [rnd.choice([rnd.randrange(1, 3000), 16 * rnd.randrange(1, 180) - 1 - rnd.randrange(t)])
                   for _ in lanes]
            toks = [rnd.randrange(model.cfg.vocab) for _ in lanes]
            runner.fill(buf, lanes, toks, None, pos)
            bad = []
            if int(buf.write_rows.max()) >= rows_cap or int(buf.write_rows.min()) < 0:
                bad.append("write_rows")
            if int(buf.block_table.max()) >= model.num_kv_blocks or int(buf.block_table.min()) < 0:
                bad.append("block_table")
            if int((buf.pos + t).max()) > model.max_seq:
                bad.append("pos")
            if int(buf.slot_rows.max()) >= model.max_batch:
                bad.append("slot_rows")
            if int(buf.token_matrix[:, 0].max()) >= model.cfg.vocab:
                bad.append("tokens")
            if bad:
                raise RuntimeError(f"bucket {capacity}: out-of-range {bad}")

        fill(capacity)
        step = graph_mtp.build_draft_step(model, mtp, buf, t)
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        replay = base.backend.capture(step, dev)
        torch.cuda.synchronize()
        mismatches = 0
        for trial in range(args.bucket_trials):
            n = capacity if trial % 2 == 0 else max(1, capacity - 1 - rnd.randrange(max(1, capacity // 2)))
            fill(n)
            step()
            want = buf.token_matrix[:n, 1:].clone()
            replay()
            torch.cuda.synchronize()
            got = buf.token_matrix[:n, 1:]
            mismatches += int((got != want).sum())
        in_range = bool((buf.token_matrix[:, 1:] >= 0).all() and (buf.token_matrix[:, 1:] < model.cfg.vocab).all())
        emit(rank, "bucket_check", capacity=capacity, trials=args.bucket_trials,
             replay_vs_eager_mismatches=mismatches, tokens_in_range=in_range,
             replay_ms=summary(windows(replay, 5, 3))["median_ms"])
        for lane in range(model.max_batch):
            model._release_lane_blocks(lane)
            base.begin(lane)


def moe_probe(rank: int, model: Model, mtp, batch: int, tag: str) -> None:  # noqa: ANN001
    """Per-rank `grouped_moe` time at `batch` rows, uniform routing (rank-local, no collectives)."""
    import mtp_dense_moe  # noqa: PLC0415

    c, dev = model.cfg, model.devices[-1]
    gen = torch.Generator(device=dev).manual_seed(0)
    h = (torch.randn(batch, c.hidden, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    ids = torch.rand(batch, c.experts, device=dev, generator=gen).topk(c.top_k, -1).indices
    w = torch.rand(batch, c.top_k, device=dev, generator=gen)
    w = (w / w.sum(-1, keepdim=True)).contiguous()
    ex = mtp.weights["experts"]

    def run() -> None:
        mtp_dense_moe.grouped_moe(h, ex, (ids, w), c.top_k, model.expert_range,
                                  mtp.expert_inter_scratch, mtp.expert_out_scratch,
                                  mtp.grouped_scratch)

    run()
    emit(rank, "moe_probe", tag=tag, **summary(windows(run, 10, 3)))


def moe_sweep(rank: int, model: Model, mtp, batch: int) -> None:  # noqa: ANN001
    """Time `grouped_moe` launch shapes at `batch` rows on the MTP experts (uniform routing)."""
    import mtp_dense_moe  # noqa: PLC0415

    c, dev = model.cfg, model.devices[-1]
    gen = torch.Generator(device=dev).manual_seed(rank)
    h = (torch.randn(batch, c.hidden, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    ids = torch.rand(batch, c.experts, device=dev, generator=gen).topk(c.top_k, -1).indices
    w = torch.rand(batch, c.top_k, device=dev, generator=gen)
    w = (w / w.sum(-1, keepdim=True)).contiguous()
    ex = mtp.weights["experts"]
    ref = mtp_dense_moe.fused_moe(h, ex, (ids, w), c.top_k, model.expert_range,
                                  mtp.expert_inter_scratch, mtp.expert_out_scratch).clone()
    emit(rank, "moe_per_assignment", **summary(windows(
        lambda: mtp_dense_moe.fused_moe(h, ex, (ids, w), c.top_k, model.expert_range,
                                        mtp.expert_inter_scratch, mtp.expert_out_scratch), 10, 3)))
    base = dict(mtp_dense_moe.GROUPED)
    configs = [{}]
    configs += [{"gu_bn": bn, "gu_bk": bk, "gu_warps": wp}
                for bn in (32, 64) for bk in (64, 128) for wp in (4, 8)]
    configs += [{"dn_bn": bn, "dn_bk": bk, "dn_warps": wp}
                for bn in (32, 64, 128) for bk in (64, 128) for wp in (4, 8)]
    for cfg in configs:
        mtp_dense_moe.GROUPED.clear()
        mtp_dense_moe.GROUPED.update({**base, **cfg})

        def run() -> torch.Tensor:
            return mtp_dense_moe.grouped_moe(h, ex, (ids, w), c.top_k, model.expert_range,
                                             mtp.expert_inter_scratch, mtp.expert_out_scratch,
                                             mtp.grouped_scratch)

        try:
            got = run().clone()
            diff = float((got.float() - ref.float()).abs().max())
            emit(rank, "moe_grouped", cfg=cfg, max_abs_diff=diff,
                 identical=float((got == ref).float().mean()), **summary(windows(run, 10, 3)))
        except Exception as exc:  # noqa: BLE001
            emit(rank, "moe_grouped_failed", cfg=cfg, error=repr(exc)[:300])
    mtp_dense_moe.GROUPED.clear()
    mtp_dense_moe.GROUPED.update(base)


def run_rank(args: argparse.Namespace, rank: int) -> None:
    reduce = tp.init(rank, args.tp, port=args.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, args.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, args.max_seq, args.batch, tp=handle)
    blas_tune.tune(model, batches=sorted(blas_tune.tuned_batches(model.max_batch)))
    mtp = model.mtp
    if mtp is None:
        raise RuntimeError("set SEED_MTP=1")
    dev = model.devices[-1]
    emit(rank, "boot", mem_alloc_gib=torch.cuda.memory_allocated(dev) / 2**30,
         vocab_tp=bool(getattr(model, "vocab_tp", False)), mtp_k=mtp.k)

    moe_probe(rank, model, mtp, args.batch, "boot")
    if args.realloc_experts:
        # Diagnostic: on one node rank 0's MTP gate_up stack streamed ~9x slower than the
        # other ranks' (same kernel, same shapes). A fresh allocation after boot tests
        # whether that is the allocation's placement rather than the GPU.
        # Every tensor the draft reads: MTP layer weights (experts included), fc and norms,
        # and the target's lm_head shard and embedding table.
        def fresh(t: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(t).copy_(t)

        def walk(d: dict) -> None:
            for key, val in d.items():
                if isinstance(val, dict):
                    walk(val)
                elif torch.is_tensor(val) and val.is_cuda:
                    d[key] = fresh(val)

        walk(mtp.weights)
        for name in ("fc", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            setattr(mtp, name, fresh(getattr(mtp, name)))
        model.lm_head = fresh(model.lm_head)
        model.embed = fresh(model.embed)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        moe_probe(rank, model, mtp, args.batch, "realloc")
    if args.moe_sweep:
        moe_sweep(rank, model, mtp, args.batch)
    if args.moe_cfg:
        import mtp_dense_moe  # noqa: PLC0415

        mtp_dense_moe.GROUPED.update(json.loads(args.moe_cfg))
        emit(rank, "moe_cfg", cfg=mtp_dense_moe.GROUPED)

    ks = [int(v) for v in args.ks.split(",")]
    base = graph_mtp.MTPVerifyRunner(model, mtp, graph_decode.CudaGraphBackend())
    runners, bufs = {}, {}
    for k in ks:
        r = graph_mtp.MTPVerifyRunner(
            model, mtp, base.backend, lane_table=base.lane_table, dirty=base._dirty
        )
        r.t = k + 1
        runners[k] = r
        bufs[k] = graph_mtp.MTPVerifyBuffers(model, args.batch, k + 1, dev)
    slots = list(range(args.batch))

    if args.buckets:
        bucket_check(args, rank, model, mtp, base)
        return

    # ---- 1. timing, W6's condition
    positions = [int(v) for v in args.positions.split(",")]
    tokens = [(7919 * lane + 1009) % model.cfg.vocab for lane in slots]
    for lane in slots:
        base.begin(lane)
    model.hidden_scratch.normal_(0.0, 1.0)
    graphs = {}
    for k in ks:
        runners[k].fill(bufs[k], slots, tokens, None, positions)
        for fast in (False, True):
            _, replay, eager = build(model, mtp, runners[k], bufs[k], k + 1, fast)
            graphs[(k, fast)] = replay
            emit(rank, "draft_eager", k=k, fast=fast, **summary(eager))
    # One-piece-off variants of the fast draft at the largest k (agreement attribution).
    kmax = max(ks)
    variants = {}
    for name in [v for v in args.variants.split(",") if v]:
        saved = (os.environ.get("SEED_DECODE_ATTN_SPLITK"), mtp_mod.DRAFT_FUSED_ROUTE,
                 graph_decode.ATTN_ROPE_KV_FUSED)
        if name == "no_splitk":
            os.environ["SEED_DECODE_ATTN_SPLITK"] = "0"
        elif name == "fused_route":
            mtp_mod.DRAFT_FUSED_ROUTE = True
        elif name == "no_rope_fused":
            graph_decode.ATTN_ROPE_KV_FUSED = False
        runners[kmax].fill(bufs[kmax], slots, tokens, None, positions)
        _, replay, _ = build(model, mtp, runners[kmax], bufs[kmax], kmax + 1, True)
        variants[name] = replay
        os.environ["SEED_DECODE_ATTN_SPLITK"] = saved[0] or "0"
        mtp_mod.DRAFT_FUSED_ROUTE, graph_decode.ATTN_ROPE_KV_FUSED = saved[1], saved[2]
        emit(rank, "variant_graph", variant=name, k=kmax, **summary(windows(replay, args.steps, 3)))
    # Same-state token check at W6's positions (random hidden), then interleaved timing.
    for k in ks:
        runners[k].fill(bufs[k], slots, tokens, None, positions)
        mtp_mod.DRAFT_FAST = False
        graph_mtp.draft_round_step(model, mtp, bufs[k], k + 1)()
        ref = bufs[k].token_matrix[:, 1:].clone()
        graphs[(k, False)]()
        old = bufs[k].token_matrix[:, 1:].clone()
        graphs[(k, True)]()
        new = bufs[k].token_matrix[:, 1:].clone()
        emit(rank, "random_hidden_agree", k=k, fast_vs_ref=float((new == ref).float().mean()),
             old_graph_vs_ref=float((old == ref).float().mean()))
    pairs = {k: {False: [], True: []} for k in ks}
    for _ in range(args.reps):
        for k in ks:
            for fast in (False, True):
                pairs[k][fast] += windows(graphs[(k, fast)], args.steps, 1)
    for k in ks:
        emit(rank, "draft_graph", k=k, old=summary(pairs[k][False]), fast=summary(pairs[k][True]),
             positions="w6")
    if args.profile:
        for k in ks:
            for fast in (False, True):
                profile_table(graphs[(k, fast)], 3,
                              OUT / f"kernels_r{rank}_k{k}_{'fast' if fast else 'old'}.txt")
    if args.uniform_pos:
        upos = [args.uniform_pos] * args.batch
        for k in ks:
            runners[k].fill(bufs[k], slots, tokens, None, upos)
            ms = {f: windows(graphs[(k, f)], args.steps, args.reps) for f in (False, True)}
            emit(rank, "draft_graph", k=k, old=summary(ms[False]), fast=summary(ms[True]),
                 positions=f"u{args.uniform_pos}")

    # ---- 2. agreement and acceptance on real hidden states
    if args.accept_steps <= 0:
        return
    prompts = prompts_for(args.batch)
    t0 = time.perf_counter()
    tokens, positions = [], []
    for lane in slots:
        model._release_lane_blocks(lane)
        base.begin(lane)
    for lane, ids in zip(slots, prompts, strict=True):
        logits = model.prefill(lane, ids, 0)
        tokens.append(int(logits[-1].argmax()))
        positions.append(len(ids))
    torch.cuda.synchronize()
    emit(rank, "prefill", seconds=time.perf_counter() - t0,
         prompt_len_mean=statistics.mean(len(p) for p in prompts))
    target = [list(tokens)]
    # ref: the original draft, eager (the correct reference); old: its captured replay;
    # fast: the production-kernel draft's captured replay; fast_eager: sampled eager check.
    names = ("w6", "ref", "ref2", "old", "old2", "fast_eager", "fast", *variants)
    drafts = {name: [] for name in names}
    slot_t = torch.tensor(slots, device=dev)
    for step in range(args.accept_steps):
        buf = bufs[kmax]
        runners[kmax].fill(buf, slots, tokens, None, positions)
        hidden = model.hidden_scratch[slot_t].clone()
        mtp_mod.DRAFT_FAST = False
        saved_k, mtp.k = mtp.k, kmax
        drafts["w6"].append(mtp_mod.draft(model, mtp, hidden, list(tokens), slots, list(positions)))
        mtp.k = saved_k
        for name, fast, replay in (("ref", False, False), ("ref2", False, False),
                                   ("old", False, True), ("old2", False, True),
                                   ("fast_eager", True, False), ("fast", True, True)):
            runners[kmax].fill(buf, slots, tokens, None, positions)
            if replay:
                graphs[(kmax, fast)]()
            else:
                mtp_mod.DRAFT_FAST = fast
                graph_mtp.draft_round_step(model, mtp, buf, kmax + 1)()
            drafts[name].append(buf.token_matrix[:, 1:].tolist())
        for name, replay in variants.items():
            runners[kmax].fill(buf, slots, tokens, None, positions)
            replay()
            drafts[name].append(buf.token_matrix[:, 1:].tolist())
        logits = model.decode(slots, tokens, positions)
        tokens = logits.argmax(-1).tolist()
        positions = [p + 1 for p in positions]
        target.append(list(tokens))
    torch.cuda.synchronize()

    def agree(a: list, b: list) -> dict:
        total = same = 0
        cond = [[0, 0] for _ in range(kmax)]  # step s given agreement at every earlier step
        for s in range(len(a)):
            for lane in slots:
                prefix = True
                for j in range(kmax):
                    eq = a[s][lane][j] == b[s][lane][j]
                    total += 1
                    same += eq
                    if prefix:
                        cond[j][0] += eq
                        cond[j][1] += 1
                    prefix = prefix and eq
        return {"positions": total, "identical": same, "rate": same / total,
                "conditional": [x / max(y, 1) for x, y in cond]}

    pairs = [("fast", "w6"), ("fast", "ref"), ("ref", "w6"), ("ref2", "ref"), ("old", "ref"),
             ("old2", "old"), ("fast_eager", "fast"), ("fast_eager", "w6"),
             *((name, "w6") for name in variants)]
    emit(rank, "agreement", steps=len(drafts["ref"]),
         **{f"{a}_vs_{b}": agree(drafts[a], drafts[b]) for a, b in pairs})
    for k in range(1, kmax + 1):
        emit(rank, "acceptance", k=k, **{
            name: acceptance([[d[:k] for d in row] for row in rows], target, k)
            for name, rows in drafts.items()
        })
    if rank == 0:
        with (OUT / "drafts.json").open("w") as fh:
            json.dump({"target": target, **drafts}, fh)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=29663)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--ks", default="1,2,3")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--uniform-pos", type=int, default=4096)
    parser.add_argument("--profile", type=int, default=1)
    parser.add_argument("--accept-steps", type=int, default=64)
    parser.add_argument("--positions", default=W6_POSITIONS)
    parser.add_argument("--moe-sweep", type=int, default=0)
    parser.add_argument("--moe-cfg", default="")
    parser.add_argument("--realloc-experts", type=int, default=0)
    parser.add_argument("--variants", default="")
    parser.add_argument("--buckets", default="")
    parser.add_argument("--bucket-trials", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    import mtp96_ceiling  # noqa: PLC0415

    mtp96_ceiling.OUT = OUT
    if args.rank:
        run_rank(args, args.rank)
        return
    forwarded = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]
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
