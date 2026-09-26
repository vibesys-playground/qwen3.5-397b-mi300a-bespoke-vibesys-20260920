"""Validate and time `mxfp4_gemv.fused_moe_bw` (SEED_MOE_BW), `mxfp4_moe_bw2` variants
(SEED_MOE_BW_VARIANT), and `moe_hip.fused_moe_hip` (SEED_MOE_HIP) against the shipped MoE
paths.

One MI300A, real TP=4 per-rank dims (hidden 4096, moe_intermediate 1024, 128 local experts,
top-10), random MXFP4 weights (`seed_tests.random_experts`), real routing traces.

    python bench_moe_bw.py                        # everything, JSON to --out
    python bench_moe_bw.py --traces 64 --quick    # smoke: fewer traces, fewer reps
    python bench_moe_bw.py --sweep                # variant x knob sweep, b48 real traces only
    python bench_moe_bw.py --rank                 # every impl (per_assign, dedup, bw v1,
                                                    # every v2 variant, hip), b48/b96/b192
    python bench_moe_bw.py --impl hip bw --batches 48   # the HIP kernels (builds the extension)
    python bench_moe_bw.py --impl hip --batches 48 --gate-up-only   # HIP gate_up alone, us/layer

`SEED_MOE_BW_VARIANT` (v1, v2a..v2g, see `mxfp4_moe_bw2`) selects the kernel the "bw" rows
time; variants that read the reshuffled layout get `bw_shuffle` applied to the timing layers
at setup (the one-time model-load step), and correctness compares against the fp32 oracle on
an unshuffled copy.

Sections:
  1. correctness: every trace, every requested impl (plus dedup/per_assign, always) against
     the fp32 oracle (`reference_moe`); the bar is "no worse than shipped" (both round the
     intermediate to bf16).
  2. us/layer under CUDA-graph replay at b48 (real traces), b96/b192 (2 or 4 real traces
     concatenated, so distinct-expert counts stay realistic). Each graph runs NLAYERS layers
     with distinct weights (>256 MB MALL), so no layer reads its weights from cache.
  3. single-expert gate_up and full-MoE time at M = 1/4/10/32 tokens.
  GB/s = distinct local experts x 6.68 MB / time (the bytes the layer must read).

Runbook (the test cluster; D=/path/to/scratch/bespoke-opt-round2/moe-bw/bundle,
E=/path/to/scratch/runtime.toml):
  rsync -a <worktree>/examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/ cluster:$D/
  salloc -A <account> -p mi300 -N1 -t 01:30:00 --no-shell          # prints <id>
  timeout 900 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && python bench_moe_bw.py --quick --out quick.json"
  timeout 2700 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && python bench_moe_bw.py --out full.json"
  timeout 3000 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && python bench_moe_bw.py --rank --traces 256 --timing-traces 32 --reps 10 --out rank.json"
  timeout 5400 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && python bench_moe_bw.py --sweep --traces 128 --timing-traces 16 --reps 5 --out sweep.json"

SEED_MOE_HIP (H=/path/to/scratch/hip-cache: the extension builds once, ~1 min,
into a content-hashed directory there and later runs load it):
  timeout 1200 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && SEED_MOE_HIP=1 SEED_HIP_CACHE_DIR=$H python -m pytest seed_tests/test_moe_hip.py -p no:cacheprovider --no-cov -q"
  timeout 1800 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && SEED_MOE_HIP=1 SEED_HIP_CACHE_DIR=$H python bench_moe_bw.py --impl hip bw dedup --batches 48 96 --traces 64 --check-traces 16 --out hip.json"
  timeout 900 srun --jobid=<id> --overlap --environment=$E bash -c "cd $D && SEED_MOE_HIP=1 SEED_HIP_CACHE_DIR=$H python bench_moe_bw.py --impl hip --gate-up-only --no-check --batches 48 --traces 64 --out hip_gu.json"
  knobs (each a separate cached build): SEED_MOE_HIP_DEPTH=2|4, SEED_MOE_HIP_GRID=228|456,
  SEED_MOE_HIP_NT=0|1, SEED_MOE_HIP_DN_TPW=1|2.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "seed_tests"))

import moe_hip  # noqa: E402
import mxfp4_gemv as mg  # noqa: E402
import mxfp4_moe_bw2 as bw2  # noqa: E402
from mxfp4 import dequant_mxfp4  # noqa: E402
from test_mxfp4_fused_gemv import random_experts  # noqa: E402

TRACE_DIR = "/path/to/scratch/bespoke-opt-round2/routing_traces"
HIDDEN, INTER, LOCAL, TOP_K, NUM_EXPERTS = 4096, 1024, 128, 10, 512
EXPERT_BYTES = 2 * INTER * (HIDDEN // 2 + HIDDEN // 32) + HIDDEN * (INTER // 2 + INTER // 32)
GU_BYTES = 2 * INTER * (HIDDEN // 2 + HIDDEN // 32)
DEV = torch.device("cuda")

# impl x SEED_MOE_BW_VARIANT/SEED_MOE_HIP knobs the `--rank` shootout runs by default: the
# production per-assignment kernel, dedup, bw v1, every v2 variant, and hip.
CANDIDATES = ["per_assign", "dedup"] + [f"bw:{v}" for v in bw2.VARIANTS] + ["hip"]


def load_traces(limit: int | None) -> list[dict]:
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "step*_layer*.pt")))
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    out = []
    for fp in files:
        d = torch.load(fp, map_location="cpu")
        out.append(
            {
                "name": os.path.basename(fp),
                "a_expert": d["a_expert"].to(torch.int32),
                "a_weight": d["a_weight"].to(torch.bfloat16),
                "lo": int(d["lo"]),
                "hi": int(d["hi"]),
            }
        )
    return out


def distinct(a_expert, a_weight, lo, hi) -> int:
    live = (a_expert >= lo) & (a_expert < hi) & (a_weight != 0)
    return int(torch.unique(a_expert[live]).numel())


def graph_ms(fn, reps: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps, g


class Static:
    """Static routing/activation buffers a captured graph reads; refilled per trace."""

    def __init__(self, tokens: int):
        self.x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=DEV)
        self.a_expert = torch.full((tokens * TOP_K,), -1, dtype=torch.int32, device=DEV)
        self.a_weight = torch.zeros(tokens * TOP_K, dtype=torch.bfloat16, device=DEV)
        self.out = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEV)
        self.inter = torch.empty(tokens * TOP_K, INTER, dtype=torch.bfloat16, device=DEV)
        self.y = torch.empty(tokens * TOP_K, HIDDEN, dtype=torch.bfloat16, device=DEV)
        self.cursor = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=DEV)

    def fill(self, a_expert, a_weight):
        self.a_expert.copy_(a_expert)
        self.a_weight.copy_(a_weight)


def runner(kind: str, st: Static, layers: list[dict], rng: tuple[int, int]):
    routing = (st.a_expert, st.a_weight)
    variant = bw2.selected_variant()

    def bw():
        for ex in layers:
            bw2.fused_moe_variant(
                st.x, ex, routing, TOP_K, rng, variant, out=st.out, inter=st.inter, y=st.y
            )

    def hip():
        for ex in layers:
            moe_hip.fused_moe_hip(st.x, ex, routing, TOP_K, rng, out=st.out, inter=st.inter, y=st.y)

    def dedup():
        for ex in layers:
            mg.fused_moe_dedup(st.x, ex, routing, TOP_K, rng, out=st.out, inter=st.inter, y=st.y)

    def per_assign():
        for ex in layers:
            order = mg.fused_expert_order(st.a_expert, NUM_EXPERTS, cursor=st.cursor)
            mg.fused_moe(
                st.x,
                ex,
                routing,
                TOP_K,
                rng,
                out=st.out,
                inter=st.inter,
                order=order,
                total_experts=NUM_EXPERTS,
            )

    return {"bw": bw, "hip": hip, "dedup": dedup, "per_assign": per_assign}[kind]


def time_traces(kind, tokens, routings, layers, reps) -> dict:
    """Capture once, replay per routing (routing copied into the static buffers, untimed)."""
    st = Static(tokens)
    rng = (routings[0][2], routings[0][3])
    st.fill(routings[0][0], routings[0][1])
    _, g = graph_ms(runner(kind, st, layers, rng), 1)
    per, dist = [], []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for ae, aw, lo, hi in routings:
        st.fill(ae, aw)
        g.replay()
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        per.append(a.elapsed_time(b) / reps / len(layers) * 1000.0)
        dist.append(distinct(ae, aw, lo, hi))
    per_t = torch.tensor(per)
    d = torch.tensor(dist, dtype=torch.float64)
    gbs = d * EXPERT_BYTES / (per_t.double() * 1e-6) / 1e9
    return {
        "us_per_layer_mean": float(per_t.mean()),
        "us_per_layer_p50": float(per_t.median()),
        "us_per_layer_max": float(per_t.max()),
        "distinct_mean": float(d.mean()),
        "gbs_mean": float(gbs.mean()),
        "n": len(per),
    }


def batched_routings(traces, tokens) -> list[tuple]:
    """b48 traces as-is; b96/b192 concatenate 2/4 consecutive traces (all share rank 0's lo/hi
    range in the files; we use the file's own lo/hi)."""
    k = tokens // 48
    out = []
    for i in range(0, len(traces) - k + 1, k):
        grp = traces[i : i + k]
        ae = torch.cat([t["a_expert"] for t in grp]).to(DEV)
        aw = torch.cat([t["a_weight"] for t in grp]).to(DEV)
        out.append((ae, aw, grp[0]["lo"], grp[0]["hi"]))
    return out


def correctness(traces, experts, kinds) -> dict:
    """Every requested kind (plus dedup/per_assign, always, as the shipped references) against
    the fp32 oracle. Tracks both max relative error (the existing pass bar: no worse than
    1.5x the shipped kernels' worst) and max absolute error (for the `--rank` table)."""
    variant = bw2.selected_variant()
    bw_experts = {k: t.clone() for k, t in experts.items()}
    if variant.shuffle:
        bw2.bw_shuffle(bw_experts)
    x = torch.randn(48, HIDDEN, dtype=torch.bfloat16, device=DEV)
    fns = {
        "bw": lambda ae, aw, rng: bw2.fused_moe_variant(x, bw_experts, (ae, aw), TOP_K, rng, variant),
        "hip": lambda ae, aw, rng: moe_hip.fused_moe_hip(x, experts, (ae, aw), TOP_K, rng),
        "dedup": lambda ae, aw, rng: mg.fused_moe_dedup(x, experts, (ae, aw), TOP_K, rng),
        "per_assign": lambda ae, aw, rng: mg.fused_moe(
            x, experts, (ae, aw), TOP_K, rng, total_experts=NUM_EXPERTS
        ),
    }
    kinds = list(dict.fromkeys([*kinds, "dedup", "per_assign"]))  # always compare to shipped
    worst_rel = {k: 0.0 for k in kinds}
    worst_abs = {k: 0.0 for k in kinds}
    mean_rel = {k: 0.0 for k in kinds}
    worst_name = {k: "" for k in kinds}
    t0 = time.time()
    for t in traces:
        ae, aw = t["a_expert"].to(DEV), t["a_weight"].to(DEV)
        rng = (t["lo"], t["hi"])
        ref = mg.reference_moe(
            x, experts, (ae, aw), TOP_K, rng, dequant_mxfp4, compute_dtype=torch.float32
        ).float()
        scale = ref.abs().max().clamp_min(1e-6)
        for k in kinds:
            v = fns[k](ae, aw, rng)
            if not torch.isfinite(v).all():
                abs_err, rel_err = float("inf"), float("inf")
            else:
                abs_err = float((v.float() - ref).abs().max())
                rel_err = abs_err / float(scale)
            mean_rel[k] += rel_err / len(traces)
            if rel_err > worst_rel[k]:
                worst_rel[k], worst_abs[k], worst_name[k] = rel_err, abs_err, t["name"]
    shipped = max(worst_rel["dedup"], worst_rel["per_assign"])
    res = {
        "n_traces": len(traces),
        "seconds": round(time.time() - t0, 1),
        "max_rel_err": worst_rel,
        "max_abs_err": worst_abs,
        "mean_rel_err": mean_rel,
        "worst_trace": worst_name,
    }
    res["pass_per_kind"] = {k: worst_rel[k] <= 1.5 * shipped + 1e-3 for k in kinds}
    res["pass"] = all(res["pass_per_kind"][k] for k in kinds if k not in ("dedup", "per_assign"))
    return res


def gate_up_only(layers, plans, inter, perm):
    def gu():
        for ex, plan in zip(layers, plans, strict=True):
            mg.bw_gate_up(ex, plan, TOP_K, inter, perm=perm)

    return gu


def single_expert(bw_layers, layers, reps) -> dict:
    out = {}
    for m in (1, 4, 10, 32):
        st = Static(m)
        ae = torch.full((m, TOP_K), -1, dtype=torch.int32)
        ae[:, 0] = 0
        aw = torch.zeros(m, TOP_K)
        aw[:, 0] = 1.0
        st.fill(ae.reshape(-1).to(DEV), aw.reshape(-1).to(torch.bfloat16).to(DEV))
        rng = (0, LOCAL)
        full_ms, _ = graph_ms(runner("bw", st, bw_layers, rng), reps)
        # gate_up alone (v1 kernels only): prep once, then time only the gate_up kernel
        gu_ms = float("nan")
        if bw2.selected_variant().name == "v1":
            plans = [mg.BwPlan(m, TOP_K, HIDDEN, LOCAL, DEV, torch.bfloat16) for _ in layers]
            for plan in plans:
                mg.bw_prep(st.x, st.a_expert, st.a_weight, rng, plan)
            gu = gate_up_only(layers, plans, st.inter, mg.perm_lut(DEV))
            gu_ms, _ = graph_ms(gu, reps)
        dd_ms, _ = graph_ms(runner("dedup", st, layers, rng), reps)
        n = len(layers)
        out[f"M{m}"] = {
            "bw_full_us": full_ms / n * 1000,
            "bw_gate_up_us": gu_ms / n * 1000,
            "bw_full_gbs": EXPERT_BYTES / (full_ms / n * 1e-3) / 1e9,
            "bw_gate_up_gbs": GU_BYTES / (gu_ms / n * 1e-3) / 1e9,
            "dedup_full_us": dd_ms / n * 1000,
        }
        print(
            f"single expert M={m:3d}: bw full {out[f'M{m}']['bw_full_us']:7.1f} us "
            f"({out[f'M{m}']['bw_full_gbs']:6.0f} GB/s)  gate_up {out[f'M{m}']['bw_gate_up_us']:6.1f} us "
            f"({out[f'M{m}']['bw_gate_up_gbs']:6.0f} GB/s)  dedup full {dd_ms / n * 1000:7.1f} us",
            flush=True,
        )
    return out


def hip_gate_up_timing(tokens, traces, layers, args) -> dict:
    """The HIP gate_up kernel alone under graph replay (prep run once, outside the graph)."""
    m, grid = moe_hip.ext(), moe_hip.grid(DEV)
    per, dist = [], []
    for ae, aw, lo, hi in batched_routings(traces, tokens)[: args.timing_traces]:
        st = Static(tokens)
        st.fill(ae, aw)
        plans = []
        for _ in layers:
            plan = mg.BwPlan(
                tokens, TOP_K, HIDDEN, hi - lo, DEV, torch.bfloat16, block_t=16, permute_x=False
            )
            mg.bw_prep(st.x, st.a_expert, st.a_weight, (lo, hi), plan)
            plans.append(plan)

        def gu(plans=plans, st=st):
            stream = torch.cuda.current_stream().cuda_stream
            for ex, plan in zip(layers, plans, strict=True):
                m.gate_up(
                    st.x.data_ptr(),
                    ex["gate_up"].data_ptr(),
                    ex["gate_up_scale"].data_ptr(),
                    plan.sorted.data_ptr(),
                    plan.unit.data_ptr(),
                    plan.n_units.data_ptr(),
                    st.inter.data_ptr(),
                    TOP_K,
                    grid,
                    stream,
                )

        ms, _ = graph_ms(gu, args.reps)
        per.append(ms / len(layers) * 1000.0)
        dist.append(distinct(ae, aw, lo, hi))
    us, d = torch.tensor(per), torch.tensor(dist, dtype=torch.float64)
    r = {
        "us_per_layer_mean": float(us.mean()),
        "distinct_mean": float(d.mean()),
        "gbs_mean": float((d * GU_BYTES / (us.double() * 1e-6) / 1e9).mean()),
        "n": len(per),
    }
    print(
        f"b{tokens} hip gate_up: {r['us_per_layer_mean']:.1f} us/layer, distinct "
        f"{r['distinct_mean']:.1f}, {r['gbs_mean']:.0f} GB/s",
        flush=True,
    )
    return r


def main_bench(args) -> dict:
    torch.manual_seed(0)
    print(
        f"torch {torch.__version__} hip {torch.version.hip} "
        f"{torch.cuda.get_device_properties(0).gcnArchName} "
        f"variant: {bw2.selected_variant()} knobs: BT={mg.BW_BLOCK_T} GU={mg.BW_GU_ROWS} DN={mg.BW_DN_ROWS} "
        f"GU_BK={mg.BW_GU_BLOCK_K} DN_BK={mg.BW_DN_BLOCK_K} stages={mg.BW_STAGES} "
        f"fence={mg.BW_FENCE} grid={mg.BW_GRID}",
        flush=True,
    )
    layers = [random_experts(LOCAL, HIDDEN, INTER, seed=100 + i) for i in range(args.layers)]
    variant = bw2.selected_variant()
    bw_layers = layers
    if variant.shuffle:  # the one-time model-load step, timed and costed
        bw_layers = [{k: t.clone() for k, t in ex.items()} for ex in layers]
        torch.cuda.synchronize()
        t0 = time.time()
        for ex in bw_layers:
            bw2.bw_shuffle(ex)
        torch.cuda.synchronize()
        extra, peak = bw2.bw_shuffle_bytes(bw_layers[0])
        print(
            f"bw_shuffle: {(time.time() - t0) / len(bw_layers) * 1e3:.1f} ms/layer, "
            f"steady-state extra {extra} B, transient peak {peak / 2**20:.0f} MiB",
            flush=True,
        )
    res = {
        "variant": dataclasses.asdict(variant),
        "knobs": {
            k: getattr(mg, k)
            for k in (
                "BW_BLOCK_T",
                "BW_GU_ROWS",
                "BW_DN_ROWS",
                "BW_GU_BLOCK_K",
                "BW_DN_BLOCK_K",
                "BW_STAGES",
                "BW_FENCE",
                "BW_GRID",
            )
        },
    }
    res["scales_ok"] = mg.bw_scales_ok(layers[0])
    if "hip" in args.impl:
        res["knobs"]["moe_hip"] = {**moe_hip.KNOBS, "grid": moe_hip.grid(DEV)}
    traces = load_traces(args.traces)
    print(f"{len(traces)} traces", flush=True)
    if not args.no_check:
        res["correctness"] = correctness(traces[: args.check_traces or None], layers[0], args.impl)
        print("correctness:", json.dumps(res["correctness"]), flush=True)
        if not res["correctness"]["pass"] and not args.force:
            print("FAIL: new-kernel error exceeds shipped; timing skipped (--force to time)")
            return res
    if args.gate_up_only:
        res["timing"] = {
            f"b{t}_hip_gate_up": hip_gate_up_timing(t, traces, layers, args) for t in args.batches
        }
        return res
    kinds = ["bw"] if args.bw_only else args.impl
    res["timing"] = {}
    for tokens in args.batches:
        routings = batched_routings(traces, tokens)[: args.timing_traces]
        for kind in kinds:
            r = time_traces(
                kind, tokens, routings, bw_layers if kind == "bw" else layers, args.reps
            )
            res["timing"][f"b{tokens}_{kind}"] = r
            print(
                f"b{tokens:4d} {kind:10s}: {r['us_per_layer_mean']:7.1f} us/layer (p50 "
                f"{r['us_per_layer_p50']:.1f}, max {r['us_per_layer_max']:.1f}), distinct "
                f"{r['distinct_mean']:.1f}, {r['gbs_mean']:.0f} GB/s",
                flush=True,
            )
    if not args.bw_only and "bw" in args.impl:
        res["single_expert"] = single_expert(bw_layers, layers, args.reps)
    return res


SWEEP = {
    "SEED_MOE_BW_VARIANT": list(bw2.VARIANTS),
    "SEED_MOE_BW2_WAVES_PER_EU": ["", "0", "2"],  # "" = the variant's default
    "SEED_MOE_BW_GU_ROWS": ["32", "16"],
}
"""Variant x occupancy hint x warps. DN_ROWS follows 2 * GU_ROWS (the fused variants need
equal warps for both item kinds). `--sweep-variants` restricts the SEED_MOE_BW_VARIANT axis
(e.g. to the top-2 Triton variants a prior `--rank` run found)."""


def _child(env_over: dict, args, batches: list[int], check: int, impl: str = "bw") -> dict:
    env = dict(os.environ, **env_over)
    cmd = [
        sys.executable,
        __file__,
        "--impl",
        impl,
        "--batches",
        *map(str, batches),
        "--check-traces",
        str(check),
        "--timing-traces",
        str(args.timing_traces),
        "--reps",
        str(args.reps),
        "--layers",
        str(args.layers),
        "--traces",
        str(args.traces or 0),
        "--out",
        "/dev/null",
        "--json-stdout",
    ]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
        line = [x for x in p.stdout.splitlines() if x.startswith("JSON ")]
        return json.loads(line[-1][5:]) if line else {"error": p.stderr[-800:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}


def sweep(args) -> None:
    """One subprocess per combination (knobs are read at import): b48 real traces, bw only,
    16-trace correctness smoke check per combination."""
    rows = []
    knob_space = dict(SWEEP)
    if args.sweep_variants:
        knob_space["SEED_MOE_BW_VARIANT"] = args.sweep_variants
    keys = list(knob_space)
    for combo in itertools.product(*(knob_space[k] for k in keys)):
        knobs = dict(zip(keys, combo, strict=True))
        if knobs["SEED_MOE_BW_VARIANT"] == "v1" and knobs["SEED_MOE_BW2_WAVES_PER_EU"]:
            continue  # v1 ignores the hint
        knobs["SEED_MOE_BW_DN_ROWS"] = str(2 * int(knobs["SEED_MOE_BW_GU_ROWS"]))
        r = _child(knobs, args, [48], 16, impl="bw")
        t = r.get("timing", {}).get("b48_bw", {})
        rows.append(
            {
                "knobs": knobs,
                "us": t.get("us_per_layer_mean"),
                "gbs": t.get("gbs_mean"),
                "pass": r.get("correctness", {}).get("pass"),
                "error": r.get("error"),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    rows.sort(key=lambda r: r["us"] if r["us"] is not None and r["pass"] else 1e9)
    print("BEST:", json.dumps(rows[0]))
    Path(args.out).write_text(json.dumps(rows, indent=2))


def _candidate_env(cand: str) -> dict:
    if cand.startswith("bw:"):
        return {"SEED_MOE_BW_VARIANT": cand.split(":", 1)[1]}
    if cand == "hip":
        return {"SEED_MOE_HIP": "1"}
    return {}


def _candidate_impl(cand: str) -> str:
    return "bw" if cand.startswith("bw:") else cand


def rank(args) -> None:
    """Shootout across every impl -- production per-assignment, dedup, bw v1, every
    SEED_MOE_BW_VARIANT, and hip -- each correctness-checked against the fp32 reference on
    real traces, then timed at every --batches size (default 48/96/192). One subprocess per
    candidate; ranked by b48 (the decode operating point)."""
    rows = []
    for cand in args.candidates or CANDIDATES:
        env = _candidate_env(cand)
        impl = _candidate_impl(cand)
        r = _child(env, args, args.batches, args.check_traces or 64, impl=impl)
        corr = r.get("correctness", {})
        row = {
            "impl": cand,
            "pass": corr.get("pass"),
            "pass_per_kind": corr.get("pass_per_kind"),
            "max_rel_err": corr.get("max_rel_err", {}).get(impl),
            "max_abs_err": corr.get("max_abs_err", {}).get(impl),
            "error": r.get("error"),
        }
        for b in args.batches:
            row[f"b{b}_us"] = r.get("timing", {}).get(f"b{b}_{impl}", {}).get("us_per_layer_mean")
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.sort(key=lambda r: r["b48_us"] if r.get("b48_us") is not None and r["pass"] else 1e9)
    cols = " ".join(f"{('b' + str(b)):>8s}" for b in args.batches)
    print(f"{'impl':10s} {'pass':5s} {cols}  max_abs_err  max_rel_err")
    for r in rows:
        cells = " ".join(
            f"{r[f'b{b}_us']:8.1f}" if r.get(f"b{b}_us") is not None else "     n/a"
            for b in args.batches
        )
        print(f"{r['impl']:10s} {str(r['pass']):5s} {cells}  {r['max_abs_err']}  {r['max_rel_err']}")
    Path(args.out).write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=0, help="subsample the trace set (0 = all)")
    ap.add_argument("--check-traces", type=int, default=0, help="correctness traces (0 = all)")
    ap.add_argument("--timing-traces", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[48, 96, 192])
    ap.add_argument("--layers", type=int, default=4, help="distinct weight sets per graph")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bw-only", action="store_true")
    ap.add_argument(
        "--impl",
        nargs="+",
        default=["bw", "dedup", "per_assign"],
        choices=["bw", "hip", "dedup", "per_assign"],
        help="implementations to check and time (hip needs SEED_MOE_HIP-capable gfx942)",
    )
    ap.add_argument("--gate-up-only", action="store_true", help="time only the HIP gate_up kernel")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument(
        "--sweep-variants",
        nargs="+",
        default=None,
        help="restrict --sweep's SEED_MOE_BW_VARIANT axis (e.g. a prior --rank's top-2)",
    )
    ap.add_argument("--rank", action="store_true")
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="restrict --rank to these impls (default: per_assign, dedup, bw:<every variant>, hip)",
    )
    ap.add_argument("--json-stdout", action="store_true")
    ap.add_argument("--out", default=str(HERE / "bench_moe_bw_result.json"))
    args = ap.parse_args()
    if args.quick:
        args.traces = args.traces or 64
        args.timing_traces = 16
        args.reps = 5
    if args.sweep:
        sweep(args)
        return
    if args.rank:
        rank(args)
        return
    res = main_bench(args)
    if args.json_stdout:
        print("JSON " + json.dumps(res))
    if args.out != "/dev/null":
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
