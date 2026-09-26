"""Four-rank MI300A gate for one captured MTP round against sequential decode.

Runs only after ``MTPVerifyRunner.prepare`` has compiled and validated the captured graph.
For prompts around KV block boundaries, forces every accept length 0..k and compares committed
ids, the next-step logits/argmax, attention KV, DeltaNet conv/recurrent state, and the cached
MTP hidden seed. Every rank executes the same SPMD sequence and rank 0 prints the JSON result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tp
from graph_mtp import MTPVerifyRunner
from model import Model, load_cfg

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/path/to/scratch/models/Qwen3.5-397B-A17B-MXFP4"
)


@dataclass
class LaneState:
    conv_rec: list[tuple[torch.Tensor, torch.Tensor]]
    hidden: torch.Tensor
    kv: list[tuple[torch.Tensor, torch.Tensor]]


def build(a: argparse.Namespace, rank: int) -> Model:
    reduce = tp.init(rank, a.tp, port=a.port)
    handle = tp.TP(tp.plan(load_cfg(MODEL_PATH), rank, a.tp), tp.device_for(rank), reduce)
    return Model(MODEL_PATH, [handle.device], torch.bfloat16, a.max_seq, 4, tp=handle)


def recurrent_state(
    model: Model, slot: int
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    state = [
        (pool["conv"][slot].clone(), pool["rec"][slot].clone())
        for pool in model.pool
        if "conv" in pool
    ]
    return state, model.hidden_scratch[slot].clone()


def restore_recurrent(model: Model, slot: int, state, hidden: torch.Tensor) -> None:  # noqa: ANN001
    for pool, (conv, rec) in zip((p for p in model.pool if "conv" in p), state, strict=True):
        pool["conv"][slot].copy_(conv)
        pool["rec"][slot].copy_(rec)
    model.hidden_scratch[slot].copy_(hidden)


def lane_state(model: Model, slot: int, consumed: int) -> LaneState:
    rows = model.block_tables[slot].physical_rows(consumed, model.block_size)
    kv = [(p["k"][rows].clone(), p["v"][rows].clone()) for p in model.pool if "k" in p]
    state, hidden = recurrent_state(model, slot)
    return LaneState(state, hidden, kv)


def max_diff(got: LaneState, want: LaneState) -> dict[str, float]:
    def diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    conv = max(
        (diff(a[0], b[0]) for a, b in zip(got.conv_rec, want.conv_rec, strict=True)), default=0.0
    )
    rec = max(
        (diff(a[1], b[1]) for a, b in zip(got.conv_rec, want.conv_rec, strict=True)), default=0.0
    )
    key = max((diff(a[0], b[0]) for a, b in zip(got.kv, want.kv, strict=True)), default=0.0)
    val = max((diff(a[1], b[1]) for a, b in zip(got.kv, want.kv, strict=True)), default=0.0)
    return {
        "conv": conv,
        "rec": rec,
        "key": key,
        "value": val,
        "hidden": diff(got.hidden, want.hidden),
    }


def decode_tokens(
    model: Model, token: int, pos: int, count: int, slot: int = 0
) -> tuple[list[int], torch.Tensor]:
    out = []
    logits = None
    for step in range(count):
        logits = model.decode([slot], [token], [pos + step])
        token = int(logits.argmax(-1)[0])
        out.append(token)
    assert logits is not None
    return out, logits


def forced_draft(truth: list[int], accept: int, k: int, vocab: int) -> list[int]:
    if accept == k:
        return truth[:k]
    wrong = (truth[accept] + 1) % vocab
    return truth[:accept] + [wrong] + [0] * (k - accept - 1)


def one_case(model: Model, runner: MTPVerifyRunner, prompt_len: int, accept: int) -> dict:
    prompt = [10 + i % 1000 for i in range(prompt_len)]
    model._release_lane_blocks(0)
    runner.begin(0)
    logits = model.prefill(0, prompt, 0)
    base = int(logits.argmax(-1)[0])
    initial, initial_hidden = recurrent_state(model, 0)

    truth, _ = decode_tokens(model, base, prompt_len, model.mtp.k + 1)
    restore_recurrent(model, 0, initial, initial_hidden)

    expected_tokens, _ = decode_tokens(model, base, prompt_len, accept + 1)
    next_pos = prompt_len + accept + 1
    expected_state = lane_state(model, 0, next_pos)
    expected_next = model.decode([0], [expected_tokens[-1]], [next_pos]).clone()
    restore_recurrent(model, 0, initial, initial_hidden)

    draft = forced_draft(truth, accept, model.mtp.k, model.cfg.vocab)
    committed = runner.verify_and_commit([0], [base], [draft], [prompt_len])[0]
    got_state = lane_state(model, 0, next_pos)
    got_next = model.decode([0], [committed[-1]], [next_pos])
    diffs = max_diff(got_state, expected_state)
    logits_max = float((got_next.float() - expected_next.float()).abs().max().item())
    argmax_equal = bool(torch.equal(got_next.argmax(-1), expected_next.argmax(-1)))
    state_ok = max(diffs.values(), default=0.0) <= 0.02
    return {
        "prompt_len": prompt_len,
        "accept": accept,
        "committed_equal": committed == expected_tokens,
        "argmax_equal": argmax_equal,
        "logits_max_abs": logits_max,
        "state_max_abs": diffs,
        "ok": committed == expected_tokens and argmax_equal and logits_max <= 0.2 and state_ok,
    }


def mixed_case(model: Model, runner: MTPVerifyRunner, mode: str) -> dict:
    """Replay one B=4 capture with per-lane accepts 0..k, including scheduler clamps."""
    k = model.mtp.k
    slots = list(range(4))
    prompt_lens = [
        model.block_size - 1,
        model.block_size,
        model.block_size + 1,
        2 * model.block_size - 1,
    ]
    prompts = [
        [100 + 1000 * slot + i % 997 for i in range(n)]
        for slot, n in zip(slots, prompt_lens, strict=True)
    ]
    bases = []
    for slot, prompt in zip(slots, prompts, strict=True):
        model._release_lane_blocks(slot)
        runner.begin(slot)
        bases.append(int(model.prefill(slot, prompt, 0).argmax(-1)[0]))

    # Exercise the ordinary-decode-to-MTP transition in the same captured runner state.
    if mode == "transition":
        logits = model.decode(slots, bases, prompt_lens)
        bases = [int(row.argmax(-1)) for row in logits]
        prompt_lens = [n + 1 for n in prompt_lens]

    initial = [recurrent_state(model, slot) for slot in slots]
    truths = []
    for slot, base, pos, (state, hidden) in zip(slots, bases, prompt_lens, initial, strict=True):
        truth, _ = decode_tokens(model, base, pos, k + 1, slot)
        truths.append(truth)
        restore_recurrent(model, slot, state, hidden)

    effective = list(range(k + 1)) + [k] * (4 - (k + 1))
    drafts = [
        forced_draft(truth, accept, k, model.cfg.vocab)
        for truth, accept in zip(truths, effective, strict=True)
    ]
    budgets = None
    stops = None
    if mode == "budget":
        drafts = [truth[:k] for truth in truths]
        budgets = [accept + 1 for accept in effective]
    elif mode == "stop":
        drafts = [truth[:k] for truth in truths]
        stops = [[truth[accept]] for truth, accept in zip(truths, effective, strict=True)]

    expected_states = []
    expected_next = []
    for slot, base, pos, accept, (state, hidden) in zip(
        slots, bases, prompt_lens, effective, initial, strict=True
    ):
        tokens, _ = decode_tokens(model, base, pos, accept + 1, slot)
        expected_states.append(lane_state(model, slot, pos + accept + 1))
        expected_next.append(model.decode([slot], [tokens[-1]], [pos + accept + 1]).clone())
        restore_recurrent(model, slot, state, hidden)

    committed = runner.verify_and_commit(slots, bases, drafts, prompt_lens, budgets, stops)
    rows = []
    for slot, pos, accept, truth, got, want_state, want_logits in zip(
        slots,
        prompt_lens,
        effective,
        truths,
        committed,
        expected_states,
        expected_next,
        strict=True,
    ):
        got_state = lane_state(model, slot, pos + accept + 1)
        got_logits = model.decode([slot], [got[-1]], [pos + accept + 1])
        diffs = max_diff(got_state, want_state)
        logits_max = float((got_logits.float() - want_logits.float()).abs().max().item())
        rows.append(
            {
                "slot": slot,
                "accept": accept,
                "committed_equal": got == truth[: accept + 1],
                "argmax_equal": bool(torch.equal(got_logits.argmax(-1), want_logits.argmax(-1))),
                "logits_max_abs": logits_max,
                "state_max_abs": diffs,
                "ok": got == truth[: accept + 1]
                and bool(torch.equal(got_logits.argmax(-1), want_logits.argmax(-1)))
                and logits_max <= 0.2
                and max(diffs.values(), default=0.0) <= 0.02,
            }
        )
    return {"mode": mode, "rows": rows, "ok": all(row["ok"] for row in rows)}


def run_rank(a: argparse.Namespace, rank: int) -> None:
    model = build(a, rank)
    if model.mtp is None:
        raise RuntimeError("SEED_MTP_SERVE=1 is required")
    runner = MTPVerifyRunner(model, model.mtp)
    captured = runner.prepare()
    cases = []
    if captured:
        contexts = [
            model.block_size - 1,
            model.block_size,
            model.block_size + 1,
            2 * model.block_size - 1,
        ]
        for prompt_len in contexts:
            for accept in range(model.mtp.k + 1):
                cases.append(one_case(model, runner, prompt_len, accept))
        for mode in ("accept", "budget", "stop", "transition"):
            cases.append(mixed_case(model, runner, mode))
    local_ok = captured and all(case["ok"] for case in cases)
    vote = torch.tensor([1.0 if local_ok else 0.0], device=model.tp.device)
    model.tp.all_reduce(vote)
    all_ok = int(vote.item()) == model.tp.plan.world
    if rank == 0:
        print(
            "DIFFERENTIAL "
            + json.dumps({"captured": captured, "all_ranks_ok": all_ok, "cases": cases}),
            flush=True,
        )
    if not all_ok:
        raise SystemExit(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--port", type=int, default=29632)
    p.add_argument("--max-seq", type=int, default=256)
    a = p.parse_args()
    if a.rank:
        run_rank(a, a.rank)
        return
    base = [
        sys.executable,
        "-u",
        os.path.abspath(__file__),
        "--tp",
        str(a.tp),
        "--port",
        str(a.port),
        "--max-seq",
        str(a.max_seq),
    ]
    workers = [
        subprocess.Popen([*base, "--rank", str(rank)], env=os.environ.copy())
        for rank in range(1, a.tp)
    ]
    try:
        run_rank(a, 0)
    finally:
        for worker in workers:
            worker.wait(timeout=180)


if __name__ == "__main__":
    main()
