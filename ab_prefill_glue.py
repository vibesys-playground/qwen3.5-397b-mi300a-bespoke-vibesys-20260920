"""Same-process A/B of captured-prefill variants (round 15, W3 prefill-nonmoe).

Boots every rank once as the server does (`server.build_model`, BLAS tune), then for each
variant: sets module flags, builds a fresh `PrefillGraphRunner` (capture, capture-vs-uncaptured
and graph-vs-eager validation, agreed across ranks), times replays of each shape, records the
logits and DeltaNet state after two chained deterministic calls, and frees the graphs. Variants
run in the order given, so repeating the base arm at the end measures drift.

    python3 -u ab_prefill_glue.py --tp 4 --shapes 1x512,2x256,4x64 \\
        --variants base,state,norm,attn,all,sp_off,base
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import allreduce_custom
import graph_prefill
import torch
from graph_decode import GraphDecodeRunner
from graph_prefill import PrefillGraphRunner, Shape

import server

FLAGS = ("DN_STATE_INPLACE", "DN_NORM_FUSED", "ATTN_FUSED")
VARIANTS = {
    "base": {},
    "state": {"DN_STATE_INPLACE": True},
    "norm": {"DN_NORM_FUSED": True},
    "attn": {"ATTN_FUSED": True},
    "dn": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True},
    "all": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True, "ATTN_FUSED": True},
    "sp_off": {"SP": False},
    # W14: SP past 1024 rows and the SP kernel's flag protocol (boot with
    # SEED_AR_SP_MAX_TOKENS=2048 so the 2048-row buffers exist; `SP_MAX_ROWS` caps per variant)
    "prod": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True, "ATTN_FUSED": True, "SP_ROWS": 256},
    "prod_relax1": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True, "ATTN_FUSED": True,
                    "SP_ROWS": 256, "RELAX1": True},
    "sp2k": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True, "ATTN_FUSED": True, "SP_ROWS": 512},
    "sp2k_relax1": {"DN_STATE_INPLACE": True, "DN_NORM_FUSED": True, "ATTN_FUSED": True,
                    "SP_ROWS": 512, "RELAX1": True},
}


def apply(variant: str) -> None:
    v = VARIANTS[variant]
    for name in FLAGS:
        setattr(graph_prefill, name, bool(v.get(name, False)))
    allreduce_custom.SP = bool(v.get("SP", True))
    import rmsnorm_fused  # noqa: PLC0415

    if "SP_ROWS" in v:
        allreduce_custom.SP_MAX_ROWS = min(v["SP_ROWS"], allreduce_custom.SP_MAX_TOKENS // 4)
    new = bool(v.get("RELAX1", False))  # the W14 flag protocol: one release, 4 rows/program
    rmsnorm_fused.SP_ONE_RELEASE = new
    rmsnorm_fused.SP_ROWS_PER_PROG = 4 if new else 1
    rmsnorm_fused.SP_ROWS_PER_PROG_MIN = 128


def calls_for(shape: Shape, vocab: int, start: int, seed: int) -> list[tuple[int, list[int], int]]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for j in range(shape.rows):
        ids = torch.randint(1000, min(vocab, 150000), (shape.width,), generator=g).tolist()
        out.append((j, ids, start))
    return out


def gemm_probe(model, m: int, say) -> None:  # noqa: ANN001
    """Dense prefill GEMMs at `m` rows with the served (tuned) solutions, all ranks at once:
    one weight repeated (weights hot in the 256 MB MALL) versus cycling through every layer's
    weight in order (what a captured step does)."""
    import torch.nn.functional as F  # noqa: PLC0415

    names = ("in_proj_all", "out_proj", "q_proj", "k_proj", "o_proj")
    for name in names:
        ws = [w[name] for w in model.layers if name in w]
        if not ws:
            continue
        x = torch.randn(m, ws[0].shape[1], device=ws[0].device, dtype=ws[0].dtype)
        res = {}
        for label, seq in (("hot", [ws[0]] * len(ws)), ("cycle", ws)):
            for _ in range(2):
                for w in seq:
                    F.linear(x, w)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                for w in seq:
                    F.linear(x, w)
            e.record()
            torch.cuda.synchronize()
            res[label] = s.elapsed_time(e) * 1e3 / (5 * len(seq))
        n, k = ws[0].shape
        say(f"GEMM M={m} {name} N={n} K={k} layers={len(ws)} hot_us={res['hot']:.1f} "
            f"cycle_us={res['cycle']:.1f} floor_us={2 * m * n * k / 980e6:.1f}")


def profile_variant(pr, variant: str, a: argparse.Namespace, say) -> None:  # noqa: ANN001
    """Per-kernel device time of one replay per shape (torch profiler, all ranks replay; rank 0
    reports). Kernel durations only; the replay wall is the timed number above."""
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    for shape in pr.shapes:
        _, replay = pr.graphs[shape]
        replay()
        torch.cuda.synchronize()
        # CPU + CUDA and two replays, as round 13's calibrated profile did: ROCTracer expands
        # graph kernels for one replay of the scope.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            replay()
            replay()
            torch.cuda.synchronize()
        if a.rank:
            continue
        name = f"{shape.rows}x{shape.width}"
        path = f"/path/to/home/r15/prefill-nonmoe/trace_{variant}_{name}.json"
        prof.export_chrome_trace(path)
        with open(path) as f:
            evs = [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel"]
        tot = sum(float(e.get("dur", 0)) for e in evs)
        say(f"PROFILE {variant} {name} sum_ms={tot / 1e3:.2f} kernels={len(evs)} -> {path}")


def run_rank(a: argparse.Namespace) -> None:
    args = argparse.Namespace(
        model_path=a.model_path, dtype="bfloat16", tp=a.tp, tp_port=a.port,
        max_seq_len=a.max_seq, max_batch=a.batch, devices="", rank=a.rank,
    )
    t0 = time.perf_counter()
    model = server.build_model(args, a.rank)
    gd = GraphDecodeRunner(model)
    lane_table = gd.lane_tables[model.devices[-1]]
    shapes = [Shape(*map(int, s.split("x"))) for s in a.shapes.split(",")]
    say = (lambda *m: print(*m, flush=True)) if a.rank == 0 else (lambda *m: None)
    say(f"BOOT {time.perf_counter() - t0:.1f}s host={os.uname().nodename}")
    for m in (int(v) for v in a.gemm_m.split(",") if v):
        gemm_probe(model, m, say)
    ref: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}
    dn_pools = [p for p in model.pool if "rec" in p]
    for vi, variant in enumerate(a.variants.split(",")):
        apply(variant)
        pr = PrefillGraphRunner(model, gd.backend, lane_table, gd._dirty, shapes)
        t1 = time.perf_counter()
        ok = pr.prepare()
        say(f"VARIANT {vi}:{variant} prepare ok={ok} kept={[f'{s.rows}x{s.width}' for s in pr.shapes]}"
            f" in {time.perf_counter() - t1:.1f}s mem_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB")
        if not ok:
            continue
        rec: dict = {"variant": variant, "idx": vi, "shapes": {}}
        for shape in pr.shapes:
            name = f"{shape.rows}x{shape.width}"
            buf, replay = pr.graphs[shape]
            # numerics: two chained calls on fresh lanes, then logits + touched DN state
            pr._reset_lanes()
            lanes = list(range(shape.rows))
            pr.fill(buf, calls_for(shape, model.cfg.vocab, 0, 7))
            replay()
            pr.fill(buf, calls_for(shape, model.cfg.vocab, shape.width, 8))
            replay()
            torch.cuda.synchronize()
            logits = buf.out.clone()
            state = [p["rec"][lanes].clone() for p in dn_pools]
            cmp = {}
            if name in ref:
                rl, rs = ref[name]
                cmp = {
                    "logits_bitexact": bool(torch.equal(logits, rl)),
                    "logits_maxabs": float((logits - rl).abs().max()),
                    "top1_agree": float((logits.argmax(-1) == rl.argmax(-1)).float().mean()),
                    "state_bitexact": all(torch.equal(x, y) for x, y in zip(state, rs, strict=True)),
                    "state_maxabs": max(float((x - y).abs().max()) for x, y in zip(state, rs, strict=True)),
                }
            else:
                ref[name] = (logits, state)
            # timing: replay only, fixed inputs, cuda events per window
            for _ in range(3):
                replay()
            torch.cuda.synchronize()
            wins = []
            for _ in range(a.windows):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(a.reps):
                    replay()
                e.record()
                torch.cuda.synchronize()
                wins.append(s.elapsed_time(e) / a.reps)
            rec["shapes"][name] = {"median_ms": statistics.median(wins), "min_ms": min(wins),
                                   "max_ms": max(wins), **cmp}
            say(f"CASE {variant} {name} median={statistics.median(wins):.3f} ms "
                f"spread=[{min(wins):.3f},{max(wins):.3f}] {json.dumps(cmp)}")
        if variant in a.profile.split(","):
            profile_variant(pr, variant, a, say)
        say("RESULT " + json.dumps(rec))
        pr.graphs = {}
        pr._reset_lanes()
        del pr
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29561)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--shapes", default="1x512,2x256,4x64")
    p.add_argument("--variants", default="base,state,norm,attn,all,sp_off,base")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--gemm-m", default="512,256")
    p.add_argument("--profile", default="")
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"))
    a = p.parse_args()
    if a.rank:
        run_rank(a)
        return
    argv = [sys.executable, "-u", os.path.abspath(__file__)] + sys.argv[1:]
    workers = [subprocess.Popen([*argv, "--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)]
    try:
        run_rank(a)
    finally:
        for proc in workers:
            proc.wait(timeout=900)


if __name__ == "__main__":
    main()
