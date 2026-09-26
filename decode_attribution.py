"""Low-overhead attribution of the captured decode step (round 15, workstream decode96).

For each `(batch, position)` case this times the production decode graph and a second,
diagnostic capture of the identical step that also holds `decode_stamps` markers at every
component boundary (dense projections, attention core, DeltaNet core, all-reduce+norm, shared
expert, routed MoE, lm head). The two replays alternate on the same buffers, so the stamped
graph's wall over the production graph's wall is the observer ratio (gate: <= 1.05). Stamp
differences give each component's stream time, summed over layers, median over replays.

Floors are bytes over the measured copy bandwidth of this device (weights, DeltaNet state
read+write, KV read), per rank. Routed-MoE bytes use the distinct local experts this step's
routing touched, counted by wrapping `router_fused.route` for one eager step.

SPMD like `graph_bucket_bench.py`: every rank runs the same calls, one JSON line per rank:

    env <production SEED_* flags> SEED_PREFILL_GRAPHS=0 \\
      python3 -u decode_attribution.py --tp 4 --batch 96 --max-seq 8192 \\
        --cases 96x2048,80x2048,64x2048,48x2048,96x512,96x4096
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blas_tune
import decode_attn_splitk
import decode_stamps
import model as model_mod
import torch
import tp
from graph_decode import CudaGraphBackend, GraphDecodeRunner, segment_step
from model import Model, load_cfg

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
ROUNDS = int(os.environ.get("ATTR_ROUNDS", "6"))
REAL_IDS = os.environ.get("ATTR_REAL_IDS", "")
"""Optional path to a JSON with `pins[*].prompt_ids`/`continuation_ids` (the campaign gate's
`continuations.json`). When set, every lane is prefilled with `pos` real tokens (the pins'
ids concatenated, each lane starting at a different offset) so KV, DeltaNet state and hence
routing come from real text instead of the synthetic zero-KV state."""
REAL_CHUNK = 512


def real_stream() -> list[int]:
    with open(REAL_IDS) as f:
        pins = json.load(f)["pins"]
    return [t for p in pins for t in (*p["prompt_ids"], *p["continuation_ids"])]


def prefill_real(runner: GraphDecodeRunner, b: int, pos: int) -> list[int]:
    """Prefill lanes `0..b-1` with `pos` real tokens each; returns each lane's next token."""
    stream = real_stream()
    n = len(stream)
    runner.reset_slots()
    nxt = []
    for s in range(b):
        off = (s * 997) % n
        ids = [stream[(off + j) % n] for j in range(pos + 1)]
        runner.begin(s)
        for start in range(0, pos, REAL_CHUNK):
            runner.prefill(s, ids[start : min(pos, start + REAL_CHUNK)], start)
        nxt.append(ids[pos])
    return nxt


PER_ROUND = int(os.environ.get("ATTR_PER_ROUND", "10"))
RAW_DIR = os.environ.get("ATTR_RAW_DIR", "")
"""Optional directory: each rank writes `raw_r<rank>_<b>x<pos>.json` with the stamp tags and
the absolute stamp ticks of every stamped replay, for cross-rank per-call analysis (replay `j`
on every rank is the same lockstep step: the all-reduces keep ranks aligned)."""


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    model = Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)
    # As `server.build_model`: without the tuned GEMM table the dense projections take the
    # default heuristic's tiles and the step is ~20 ms slower than production.
    blas_tune.tune(model, batches=sorted(blas_tune.tuned_batches(model.max_batch)))
    return model


def nbytes(t: object) -> int:
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    if isinstance(t, dict):
        return sum(nbytes(v) for v in t.values())
    if isinstance(t, (list, tuple)):
        return sum(nbytes(v) for v in t)
    return 0


def copy_bandwidth(dev: torch.device) -> float:
    """Device copy bandwidth, bytes/s counting read + write (2 GiB source)."""
    src = torch.empty(2**30, dtype=torch.int16, device=dev)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    bw = 2 * nbytes(src) * 10 / (time.perf_counter() - t0)
    del src, dst
    torch.cuda.empty_cache()
    return bw


ROUTE_HIST: dict[str, object] = {}
"""`ATTR_RAW_DIR` only: per-layer expert pick counts over the real-id prefill (`hist`, the
placement calibration set) and the decode step's own top-k ids (`step`)."""


class _RouteHist:
    """Counts expert picks per layer for every `router_fused.route` call while installed."""

    def __init__(self, model: Model) -> None:
        self.n = len(model.layers)
        self.e = model.cfg.experts
        self.hist = torch.zeros(self.n, self.e, dtype=torch.int64, device=model.devices[0])
        self.calls = 0
        self.real = model_mod.router_fused.route

    def __enter__(self):
        def spy(*args, **kw):
            out = self.real(*args, **kw)
            ids = out[1].reshape(-1).long()
            self.hist[self.calls % self.n] += torch.bincount(ids, minlength=self.e)[: self.e]
            self.calls += 1
            return out

        model_mod.router_fused.route = spy
        return self

    def __exit__(self, *exc) -> None:
        model_mod.router_fused.route = self.real


def touched_experts(model: Model, runner: GraphDecodeRunner, slots, tokens, positions) -> list[int]:
    """Distinct local experts per layer for one eager step on these inputs."""
    seen: list[torch.Tensor] = []
    real = model_mod.router_fused.route

    def spy(*args, **kw):
        out = real(*args, **kw)
        seen.append(out[1].detach().reshape(-1).clone())
        if RAW_DIR:
            ROUTE_HIST.setdefault("step", []).append(out[1].detach().cpu().clone())
        return out

    model_mod.router_fused.route = spy
    try:
        runner.enabled = False
        runner.decode(slots, tokens, positions)
    finally:
        model_mod.router_fused.route = real
        runner.enabled = True
    lo, hi = model.expert_range
    return [int(s[(s >= lo) & (s < hi)].unique().numel()) for s in seen]


def floors(model: Model, b: int, pos: int, touched: list[int], bw: float) -> dict[str, float]:
    """Byte floors in ms per step, this rank."""
    c = model.cfg
    by: dict[str, float] = defaultdict(float)
    for i, w in enumerate(model.layers):
        if c.layer_types[i] == "full_attention":
            by["dense_qkv"] += nbytes([w["q_proj"], w["k_proj"], w["v_proj"]])
            by["dense_o"] += nbytes(w["o_proj"])
            pool = model.pool[i]
            per_tok = nbytes(pool["k"][0]) + nbytes(pool["v"][0])
            by["attn_core"] += per_tok * b * (pos + 1)
        else:
            by["dense_in_proj"] += nbytes(w["in_proj_all"])
            by["dense_out_proj"] += nbytes(w["out_proj"])
            st = model.pool[i]
            per_lane = sum(nbytes(t[0]) for t in st.values())
            by["dn_core"] += 2 * per_lane * b
        by["dense_router"] += nbytes(w["router_gate"])
        by["dense_shared"] += nbytes(
            [w["shared_expert.gate_up_proj"], w["shared_expert.down_proj"]]
        )
        ex = w["experts"]
        local = model.expert_range[1] - model.expert_range[0]
        if i < len(touched):
            by["moe_routed"] += nbytes(ex) / local * touched[i]
    by["lm_head"] += nbytes(model.lm_head)
    return {k: v / bw * 1e3 for k, v in by.items()} | {"_bytes_gb": sum(by.values()) / 1e9}


def run_case(model: Model, runner: GraphDecodeRunner, b: int, pos: int, bw: float) -> dict:
    dev = model.devices[0]
    slots = list(range(b))
    tokens = [1000 + 17 * s for s in slots]
    positions = [pos] * b
    ROUTE_HIST.clear()
    if REAL_IDS and RAW_DIR:
        with _RouteHist(model) as rh:
            tokens = prefill_real(runner, b, pos)
        ROUTE_HIST["hist"] = rh.hist.cpu()
        ROUTE_HIST["hist_calls"] = rh.calls
    elif REAL_IDS:
        tokens = prefill_real(runner, b, pos)
    else:
        runner.reset_slots()
        runner.decode(slots, tokens, positions)  # grows every lane's blocks to `pos`
    touched = touched_experts(model, runner, slots, tokens, positions)

    graphs = runner.graphs[b]
    bufs = graphs.buffers
    assert len(bufs) == 1, "attribution assumes the single-segment TP layout"
    rec = decode_stamps.Recorder(dev)
    backend = CudaGraphBackend()  # private pool: never shares scratch with production graphs
    steps = [segment_step(model, s, buf) for s, buf in zip(runner.segments, bufs, strict=True)]
    runner.fill(bufs, b, slots, tokens, positions)
    decode_stamps.RECORDER = rec
    try:
        for _ in range(2):
            runner.run(bufs, steps)
        rec.reset()
        stamped = [backend.capture(steps[0], dev)]
    finally:
        decode_stamps.RECORDER = None
    n_stamps = len(rec.tags)

    def timed(replays) -> float:  # noqa: ANN001
        runner.fill(bufs, b, slots, tokens, positions)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        runner.run(bufs, replays)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1e3

    prod_ms: list[float] = []
    stamp_ms: list[float] = []
    step_ms: list[float] = []
    per_tag: dict[str, list[float]] = defaultdict(list)
    spans: list[float] = []
    raw_ticks: list[list[int]] = []
    for _ in range(3):
        timed(graphs.replays)
        timed(stamped)
    for _ in range(ROUNDS):
        for _ in range(PER_ROUND):
            prod_ms.append(timed(graphs.replays))
        for _ in range(PER_ROUND):
            stamp_ms.append(timed(stamped))
            segs = rec.segments_ms()
            acc: dict[str, float] = defaultdict(float)
            for tag, ms in segs:
                acc[tag] += ms
            for tag, ms in acc.items():
                per_tag[tag].append(ms)
            spans.append(sum(ms for _, ms in segs))
            if RAW_DIR:
                raw_ticks.append(rec.buf[: len(rec.tags)].tolist())
        # Full production call: fill (host) + replay + gather, back to back, one sync per step.
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(PER_ROUND):
            out = runner.decode(slots, tokens, positions)
        torch.cuda.synchronize(dev)
        step_ms.append((time.perf_counter() - t0) * 1e3 / PER_ROUND)
        del out
    del stamped
    torch.cuda.synchronize(dev)

    if RAW_DIR:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        os.makedirs(RAW_DIR, exist_ok=True)
        with open(os.path.join(RAW_DIR, f"raw_r{rank}_{b}x{pos}.json"), "w") as f:
            json.dump({"tags": rec.tags, "ticks": raw_ticks, "hz": decode_stamps.CLOCK_HZ}, f)
        if rank == 0 and ROUTE_HIST:
            torch.save(
                dict(ROUTE_HIST, expert_range=model.expert_range),
                os.path.join(RAW_DIR, f"route_{b}x{pos}.pt"),
            )
    med = statistics.median
    prod, stamp = med(prod_ms), med(stamp_ms)
    return {
        "batch": b,
        "pos": pos,
        "real_ids": bool(REAL_IDS),
        "prod_replay_ms": round(prod, 3),
        "prod_replay_iqr": [round(q, 3) for q in statistics.quantiles(prod_ms, n=4)[::2]],
        "stamped_replay_ms": round(stamp, 3),
        "observer_ratio": round(stamp / prod, 4),
        "decode_call_ms": round(med(step_ms), 3),
        "stamp_span_ms": round(med(spans), 3),
        "n_stamps": n_stamps,
        "touched_local_experts_mean": round(sum(touched) / max(1, len(touched)), 1),
        "components_ms": {k: round(med(v), 3) for k, v in sorted(per_tag.items())},
        "floors_ms": {k: round(v, 3) for k, v in floors(model, b, pos, touched, bw).items()},
    }


VARIANT_KEYS = {"pc": "SPLITS_PER_CU", "tile": "TILE", "max": "MAX_SPLITS"}


def parse_variant(spec: str) -> dict[str, float]:
    """`pc=8,tile=64` -> `decode_attn_splitk` module overrides."""
    out: dict[str, float] = {}
    for kv in spec.split(","):
        k, v = kv.split("=")
        out[VARIANT_KEYS[k]] = float(v) if k == "pc" else int(v)
    return out


def run_variants(
    model: Model, runner: GraphDecodeRunner, b: int, pos: int, specs: list[str]
) -> dict:
    """Same-process A/B of `decode_attn_splitk` launch settings inside the captured step.

    Each variant is captured twice (plain and stamped) over the production buffers. Plain
    variant replays alternate with production replays for the step delta; the stamped one
    gives the attention-core time. Numerics: from one DeltaNet-state snapshot, the production
    replay and the variant replay must give the same argmax on every row (rounding-order
    change only: the split count changes the combine order)."""
    dev = model.devices[0]
    slots = list(range(b))
    tokens = [1000 + 17 * s for s in slots]
    positions = [pos] * b
    runner.reset_slots()
    runner.decode(slots, tokens, positions)
    graphs = runner.graphs[b]
    bufs = graphs.buffers
    steps = [segment_step(model, s, buf) for s, buf in zip(runner.segments, bufs, strict=True)]
    dn = [t for p in model.pool if "rec" in p for t in p.values()]
    snap = [t.clone() for t in dn]

    def restore() -> None:
        for t, s0 in zip(dn, snap, strict=True):
            t.copy_(s0)

    def timed(replays) -> float:  # noqa: ANN001
        runner.fill(bufs, b, slots, tokens, positions)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        runner.run(bufs, replays)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1e3

    def output(replays) -> torch.Tensor:  # noqa: ANN001
        restore()
        runner.fill(bufs, b, slots, tokens, positions)
        runner.run(bufs, replays)
        return bufs[-1].out[:b].clone()

    ref = output(graphs.replays)
    base_splits = decode_attn_splitk._num_splits(
        b, 1, bufs[0].block_table.shape[1] * model.block_size, decode_attn_splitk._num_cus(dev)
    )
    rows = []
    saved = {k: getattr(decode_attn_splitk, k) for k in VARIANT_KEYS.values()}
    for spec in specs:
        over = parse_variant(spec)
        rec = decode_stamps.Recorder(dev)
        try:
            for k, v in over.items():
                setattr(decode_attn_splitk, k, v)
            splits = decode_attn_splitk._num_splits(
                b,
                1,
                bufs[0].block_table.shape[1] * model.block_size,
                decode_attn_splitk._num_cus(dev),
            )
            runner.fill(bufs, b, slots, tokens, positions)
            runner.run(bufs, steps)
            plain = [CudaGraphBackend().capture(steps[0], dev)]
            decode_stamps.RECORDER = rec
            runner.run(bufs, steps)
            rec.reset()
            stamped = [CudaGraphBackend().capture(steps[0], dev)]
        finally:
            decode_stamps.RECORDER = None
            for k, v in saved.items():
                setattr(decode_attn_splitk, k, v)
        got = output(plain)
        scale = ref.abs().max().clamp(min=1.0)
        err = float((got - ref).abs().max() / scale)
        mism = int((got.argmax(-1) != ref.argmax(-1)).sum())
        base_ms, var_ms, attn = [], [], []
        for _ in range(3):
            timed(graphs.replays)
            timed(plain)
        for _ in range(ROUNDS):
            base_ms += [timed(graphs.replays) for _ in range(PER_ROUND)]
            var_ms += [timed(plain) for _ in range(PER_ROUND)]
            for _ in range(PER_ROUND // 2):
                timed(stamped)
                attn.append(sum(ms for tag, ms in rec.segments_ms() if tag == "attn_core"))
        del plain, stamped
        med = statistics.median
        rows.append(
            {
                "variant": spec,
                "splits": splits,
                "base_splits": base_splits,
                "base_ms": round(med(base_ms), 3),
                "var_ms": round(med(var_ms), 3),
                "delta_ms": round(med(var_ms) - med(base_ms), 3),
                "attn_core_ms": round(med(attn), 3),
                "rel_err": err,
                "argmax_mismatch": mism,
                "bit_exact": bool(torch.equal(got, ref)),
            }
        )
    restore()
    return {"batch": b, "pos": pos, "variants": rows}


AB_VARIANTS = {
    "none": {},  # null re-capture: controls for capture order / graph memory placement
    "push": {"allreduce_custom.PUSH": True},
    "sp": {"graph_decode.AR_SP_DECODE": True},
    "sp_push": {"graph_decode.AR_SP_DECODE": True, "allreduce_custom.PUSH": True},
}
"""`ATTR_AB_VARIANTS` names: module flags a variant capture sets (`run_flag_ab`)."""


def _set_flags(flags: dict[str, object]) -> dict[str, object]:
    import allreduce_custom
    import graph_decode

    mods = {"allreduce_custom": allreduce_custom, "graph_decode": graph_decode}
    old = {}
    for key, v in flags.items():
        mod, name = key.split(".")
        old[key] = getattr(mods[mod], name)
        setattr(mods[mod], name, v)
    return old


def run_flag_ab(model: Model, runner: GraphDecodeRunner, b: int, pos: int, names: list[str]) -> dict:
    """Same-process A/B of all-reduce variants inside the captured decode step.

    Each variant is captured (plain and stamped) over the production buffers with its module
    flags set, then the flags are restored. Plain variant replays alternate with production
    replays for the step delta; the stamped one gives per-component times. Numerics: from one
    state snapshot, both replays must give bit-identical logits and DeltaNet state."""
    dev = model.devices[0]
    slots = list(range(b))
    tokens = [1000 + 17 * s for s in slots]
    positions = [pos] * b
    if REAL_IDS:
        tokens = prefill_real(runner, b, pos)
    else:
        runner.reset_slots()
        runner.decode(slots, tokens, positions)
    graphs = runner.graphs[b]
    bufs = graphs.buffers
    dn = [t for p in model.pool if "rec" in p for t in p.values()]
    snap = [t.clone() for t in dn]

    def restore() -> None:
        for t, s0 in zip(dn, snap, strict=True):
            t.copy_(s0)

    def timed(replays) -> float:  # noqa: ANN001
        runner.fill(bufs, b, slots, tokens, positions)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        runner.run(bufs, replays)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1e3

    def output(replays) -> tuple[torch.Tensor, list[torch.Tensor]]:  # noqa: ANN001
        restore()
        runner.fill(bufs, b, slots, tokens, positions)
        runner.run(bufs, replays)
        torch.cuda.synchronize(dev)
        return bufs[-1].out[:b].clone(), [t.clone() for t in dn]

    ref, ref_state = output(graphs.replays)
    rows = []
    for name in names:
        rec = decode_stamps.Recorder(dev)
        old = _set_flags(AB_VARIANTS[name])
        try:
            steps = [segment_step(model, s, buf) for s, buf in zip(runner.segments, bufs, strict=True)]
            runner.fill(bufs, b, slots, tokens, positions)
            runner.run(bufs, steps)
            plain = [CudaGraphBackend().capture(steps[0], dev)]
            decode_stamps.RECORDER = rec
            runner.run(bufs, steps)
            rec.reset()
            stamped = [CudaGraphBackend().capture(steps[0], dev)]
        finally:
            decode_stamps.RECORDER = None
            _set_flags(old)
        got, got_state = output(plain)
        state_exact = all(torch.equal(x, y) for x, y in zip(got_state, ref_state, strict=True))
        base_ms, var_ms = [], []
        comp: dict[str, list[float]] = defaultdict(list)
        for _ in range(3):
            timed(graphs.replays)
            timed(plain)
        for _ in range(ROUNDS):
            base_ms += [timed(graphs.replays) for _ in range(PER_ROUND)]
            var_ms += [timed(plain) for _ in range(PER_ROUND)]
            for _ in range(PER_ROUND // 2):
                timed(stamped)
                acc: dict[str, float] = defaultdict(float)
                for tag, ms in rec.segments_ms():
                    acc[tag] += ms
                for tag, ms in acc.items():
                    comp[tag].append(ms)
        del plain, stamped
        med = statistics.median
        rows.append(
            {
                "variant": name,
                "base_ms": round(med(base_ms), 3),
                "base_iqr": [round(q, 3) for q in statistics.quantiles(base_ms, n=4)[::2]],
                "var_ms": round(med(var_ms), 3),
                "var_iqr": [round(q, 3) for q in statistics.quantiles(var_ms, n=4)[::2]],
                "delta_ms": round(med(var_ms) - med(base_ms), 3),
                "var_components_ms": {k: round(med(v), 3) for k, v in sorted(comp.items())},
                "logits_bit_exact": bool(torch.equal(got, ref)),
                "argmax_mismatch": int((got.argmax(-1) != ref.argmax(-1)).sum()),
                "state_bit_exact": state_exact,
            }
        )
    restore()
    return {"batch": b, "pos": pos, "ab": rows}


def run_rank(a: argparse.Namespace, rank: int) -> None:
    t0 = time.time()
    import allreduce_custom

    # `SEED_AR_PUSH=1` only allocates the push buffers here; production graphs stay on the
    # read path unless `ATTR_BASE_PUSH=1`, and `run_flag_ab` switches variants on.
    allreduce_custom.PUSH = os.environ.get("ATTR_BASE_PUSH", "0") == "1"
    model = build(a, rank)
    runner = GraphDecodeRunner(model)
    ok = runner.prepare()
    dev = model.devices[0]
    bw = copy_bandwidth(dev)
    print(
        f"ATTR_READY rank={rank} captured={ok} buckets={runner.buckets} "
        f"boot_s={time.time() - t0:.0f} copy_bw_tbs={bw / 1e12:.3f}",
        flush=True,
    )
    if not ok:
        raise SystemExit("decode graphs did not capture")
    for case in filter(None, a.variant_cases.split(",")):
        b, pos = (int(v) for v in case.split("x"))
        res = run_variants(model, runner, b, pos, a.variants.split(";")) | {"rank": rank}
        print("VAR " + json.dumps(res), flush=True)
    for case in filter(None, os.environ.get("ATTR_AB_CASES", "").split(",")):
        b, pos = (int(v) for v in case.split("x"))
        names = os.environ.get("ATTR_AB_VARIANTS", "push,sp").split(",")
        res = run_flag_ab(model, runner, b, pos, names) | {"rank": rank}
        print("AB " + json.dumps(res), flush=True)
    for case in filter(None, a.cases.split(",")):
        b, pos = (int(v) for v in case.split("x"))
        res = run_case(model, runner, b, pos, bw) | {
            "rank": rank,
            "copy_bw_tbs": round(bw / 1e12, 3),
        }
        print("ATTR " + json.dumps(res), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29531)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--cases", default="96x2048,80x2048,64x2048,48x2048,96x512,96x4096")
    p.add_argument("--variant-cases", default="")
    p.add_argument("--variants", default="pc=4;pc=8;pc=16")
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [sys.executable, "-u", os.path.abspath(__file__)]
    for flag in ("tp", "port", "batch", "max_seq", "cases", "variant_cases", "variants"):
        base += [f"--{flag.replace('_', '-')}", str(getattr(a, flag))]
    workers = [
        subprocess.Popen([*base, "--rank", str(r)], env=os.environ.copy()) for r in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for proc in workers:
            proc.wait(timeout=300)


if __name__ == "__main__":
    main()
