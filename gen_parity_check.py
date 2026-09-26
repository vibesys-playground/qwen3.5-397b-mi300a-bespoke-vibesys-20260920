"""32-token greedy generation, TP=4, real checkpoint: identical with the custom all-reduce on
vs off.

Runs the same prompt/slot twice in one process group, one model load -- once with
`model.tp.custom_reduce` forced off (pure RCCL) and once left as built (the custom path, on
by default) -- and asserts the greedy argmax token stream matches exactly at every one of 32
steps. This is the gate `graph_decode.py`'s own `VALIDATE_REL_TOL` describes for capture; here
it's applied across the two all-reduce backends instead of across eager-vs-replay.

One load, not two: `SEED_CUSTOM_ALLREDUCE` only gates *construction*, and toggling
`model.tp.custom_reduce` after construction is a plain Python attribute write, so both passes
run against the same weights without paying the ~4-minute checkpoint load twice. The toggle
happens at the same *command count* on every rank (`BATCH` begins, `BATCH` prefills, `STEPS`
decodes per pass) rather than through the broadcast protocol, which is what makes it safe
under TP: every rank must pick the same backend on every call or the two protocols deadlock
against each other (RCCL waiting on a collective the custom kernel never issues, and vice
versa), and command counts are symmetric across ranks by construction.

    srun --jobid=<id> --overlap ... python3 -u gen_parity_check.py --tp 4
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tp
from model import Model, load_cfg
from tp_driver import Broadcaster, Channel, apply

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)
STEPS = 32


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    return Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, a.batch, tp=handle)


def generate(runner, batch: int, prompt: list[int]) -> list[list[int]]:
    slots = list(range(batch))
    for s in slots:
        runner.begin(s)
        runner.prefill(s, prompt, 0)
    tokens = [prompt[-1]] * batch
    positions = [len(prompt) - 1] * batch
    out = [[] for _ in slots]
    for _ in range(STEPS):
        logits = runner.decode(slots, tokens, positions)
        tokens = logits.argmax(-1).tolist()
        for s, t in zip(slots, tokens, strict=True):
            out[s].append(t)
        positions = [p + 1 for p in positions]
    return out


def run_rank0(a: argparse.Namespace, workers: list) -> None:
    model = build(a, 0)
    runner = Broadcaster(model, Channel(model.tp.device).send) if a.tp > 1 else model
    prompt = list(range(10, 10 + a.prompt))
    real_custom = model.tp.custom_reduce

    model.tp.custom_reduce = None
    out_rccl = generate(runner, a.batch, prompt)
    model.tp.custom_reduce = real_custom
    out_custom = generate(runner, a.batch, prompt)

    match = out_rccl == out_custom
    print(
        "RESULT "
        + json.dumps(
            {
                "match": match,
                "steps": STEPS,
                "batch": a.batch,
                "had_custom_reduce": real_custom is not None,
                "rccl_tokens_row0": out_rccl[0],
                "custom_tokens_row0": out_custom[0],
            }
        ),
        flush=True,
    )
    if a.tp > 1:
        runner.stop()
    for p in workers:
        p.wait(timeout=180)


def run_worker_two_pass(a: argparse.Namespace, model: Model) -> None:
    ch = Channel(model.tp.device)
    real_custom = model.tp.custom_reduce
    for pass_i in range(2):
        model.tp.custom_reduce = None if pass_i == 0 else real_custom
        for _ in range(a.batch):  # BEGIN x batch
            apply(model, ch.recv())
        for _ in range(a.batch):  # PREFILL x batch
            apply(model, ch.recv())
        for _ in range(STEPS):  # DECODE x STEPS
            apply(model, ch.recv())
    ch.recv()  # STOP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29519)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=128)
    a = p.parse_args()

    if a.rank:
        model = build(a, a.rank)
        run_worker_two_pass(a, model)
        return

    base = [sys.executable, "-u", os.path.abspath(__file__)]
    for flag in ("tp", "port", "batch", "prompt"):
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
