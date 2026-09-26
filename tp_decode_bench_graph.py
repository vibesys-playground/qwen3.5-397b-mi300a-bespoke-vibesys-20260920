"""Full decode step, batch 48, real checkpoint, under HIP graph replay, TP=4.

Adapted from the prior campaign's `tp_decode_bench.py` (eager-only) to also wrap the model in
`graph_decode.GraphDecodeRunner`, so this measures the same thing `server.py
--enable-graph-capture` serves: `SEED_CUSTOM_ALLREDUCE=1` (default) routes each layer's two
all-reduces through the custom kernel when the bf16 payload fits; `SEED_CUSTOM_ALLREDUCE=0`
is the RCCL baseline, all else identical. Every rank's subprocess inherits the parent's
environment (`env=os.environ.copy()` below), so any `SEED_*` flag set before invoking this
script reaches every worker, including `SEED_FUSE_GLUE` and `SEED_FUSED_AR_NORM`
(`opt-decode-fusion`'s two flags: per-tensor decode glue -- RoPE, masked KV-cache/DeltaNet-
state writes, the MoE router -- and the fused all-reduce+residual+RMSNorm, respectively; see
`attn_decode_fused.py`, `deltanet_fused.py`, `router_fused.py`, and
`allreduce_custom.CustomAllReduce.residual_norm` for what each replaces).

Baseline (this campaign's pre-fusion number, r2 tip):

    srun --jobid=<id> --overlap --environment=/path/to/scratch/runtime.toml \\
        timeout 300 env SEED_CUSTOM_ALLREDUCE=1 python3 -u tp_decode_bench_graph.py --tp 4

Every fusion in this branch on, same shape, for the A/B this branch predicts:

    srun --jobid=<id> --overlap --environment=/path/to/scratch/runtime.toml \\
        timeout 300 env SEED_CUSTOM_ALLREDUCE=1 SEED_FUSE_GLUE=1 SEED_FUSED_AR_NORM=1 \\
        python3 -u tp_decode_bench_graph.py --tp 4

Each of the two flags alone (isolates which fusion moved the number):

    ... env SEED_CUSTOM_ALLREDUCE=1 SEED_FUSE_GLUE=1  python3 -u tp_decode_bench_graph.py --tp 4
    ... env SEED_CUSTOM_ALLREDUCE=1 SEED_FUSED_AR_NORM=1  python3 -u tp_decode_bench_graph.py --tp 4
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tp
from graph_decode import GraphDecodeRunner
from model import Model, load_cfg
from tp_driver import Broadcaster, Channel, serve_worker

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    return Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)


def run_rank0(a: argparse.Namespace, workers: list) -> None:
    t0 = time.time()
    model = build(a, 0)
    load_s = time.time() - t0

    graph_runner = GraphDecodeRunner(model)
    captured = graph_runner.prepare()
    runner = graph_runner if captured else model
    if a.tp > 1:
        runner = Broadcaster(runner, Channel(model.tp.device).send)

    slots = list(range(a.batch))
    prompt = list(range(10, 10 + a.prompt))
    for s in slots:
        runner.begin(s)
        runner.prefill(s, prompt, 0)
        runner.save_prefix(s)

    tokens = [1000 + s for s in slots]
    positions = [a.prompt] * a.batch
    for _ in range(a.warmup):
        runner.decode(slots, tokens, positions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(a.steps):
        logits = runner.decode(slots, tokens, positions)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / a.steps * 1e3

    out = {
        "tp": a.tp,
        "batch": a.batch,
        "captured": captured,
        "custom_allreduce": os.environ.get("SEED_CUSTOM_ALLREDUCE", "1"),
        "fuse_glue": os.environ.get("SEED_FUSE_GLUE", "0"),
        "fused_ar_norm": os.environ.get("SEED_FUSED_AR_NORM", "0"),
        "load_s": round(load_s, 1),
        "ms_per_step": round(ms, 3),
        "tok_s": round(a.batch / (ms / 1e3), 1),
        "argmax": logits.argmax(-1).tolist()[: min(8, a.batch)],
    }
    print("RESULT " + json.dumps(out), flush=True)
    if a.tp > 1:
        runner.stop() if hasattr(runner, "stop") else None
    for p in workers:
        p.wait(timeout=180)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29518)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--prompt", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()

    if a.rank:
        model = build(a, a.rank)
        graph_runner = GraphDecodeRunner(model)
        captured = graph_runner.prepare()
        runner = graph_runner if captured else model
        serve_worker(runner, Channel(model.tp.device).recv)
        return

    base = [sys.executable, "-u", os.path.abspath(__file__)]
    for flag in ("tp", "port", "batch", "prompt", "steps", "warmup"):
        base += [f"--{flag}", str(getattr(a, flag))]
    base += ["--max-seq", str(a.max_seq)]
    workers = [
        subprocess.Popen([*base, "--rank", str(r)], env=os.environ.copy())
        for r in range(1, a.tp)
    ]
    try:
        run_rank0(a, workers)
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
