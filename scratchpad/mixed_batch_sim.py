#!/usr/bin/env python3
"""Predicted TPOT p95 and aggregate throughput at C=48, `SEED_MIXED_BATCH` vs plain
alternation, from the measured cost model in the campaign brief.

This is a closed-form step trace over the same iteration-level quantities `scheduler.py`
actually uses (`MAX_PREFILL_BURST`, `PREFILL_CHUNK`, `_mixed_budget`'s own formula, imported
directly rather than re-derived, so this cannot silently drift from the real policy), not a
stochastic simulator or a substitute for the GPU run -- it exists to give the GPU run a number
to confirm or refute (per COMMON_BRIEF.md's validation requirement), not to replace it.

Model: C=48 decode lanes, always full (worst case: sustained load, no lane idle -- the
benchmark's own load-ramp scenario at its top concurrency). One session's turn boundary is
modeled as a single fresh prefill burst of `--prefill-tokens` (default 2048, the campaign's own
measured "packed prefill call" size) arriving while every other lane is mid-decode -- the exact
"turn 2+ prefills stall decode" scenario the brief describes. Bursts repeat every
`--turn-gap-tokens` decode iterations so there are enough TPOT samples for a stable p95.

  - alternating: `MAX_PREFILL_BURST` consecutive `PREFILL_CHUNK`-sized prefill iterations run
    before a forced decode iteration, exactly `scheduler.step()`'s pre-mixed-batch branching.
    A prefill iteration stalls *every* decode lane for its own duration (one shared iteration
    axis); a lane's TPOT sample is the gap between its two surrounding decode iterations.
  - mixed: every iteration is a decode iteration; when prefill work is pending, its cost adds
    `budget * prefill_ms_per_token`, `budget` from `_mixed_budget`'s own formula (imported).

Usage:
    python3 mixed_batch_sim.py
    python3 mixed_batch_sim.py --prefill-ms-per-token 0.3   # post-vectorization prediction
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scheduler import MAX_PREFILL_BURST, MIXED_BUDGET_SLACK, PREFILL_CHUNK  # noqa: E402


def mixed_budget(decode_step_ms: float, prefill_ms_per_token: float, ceiling: int) -> int:
    """`Scheduler._mixed_budget`'s own formula, reproduced here (not reimported: it is an
    instance method reading instance config) so this script and the real scheduler cannot
    silently disagree about the shape of the bound -- see `scheduler.py`'s docstring for the
    derivation (`MIXED_BUDGET_SLACK` = 0.5 => <=1.5x a decode-only step)."""
    if prefill_ms_per_token <= 0:
        return ceiling
    by_cost = int(MIXED_BUDGET_SLACK * decode_step_ms / prefill_ms_per_token)
    return max(1, min(ceiling, by_cost))


@dataclass
class Result:
    policy: str
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float
    throughput_tok_s: float
    wall_s: float
    tokens: int


def simulate(
    policy: str,
    concurrency: int,
    decode_step_ms: float,
    prefill_ms_per_token: float,
    prefill_tokens: int,
    turn_gap_tokens: int,
    n_turns: int,
    mixed_token_budget_ceiling: int,
) -> Result:
    t = 0.0
    last_token_time = [0.0] * concurrency
    tpot: list[float] = []
    tokens_generated = 0
    pending_prefill = 0
    tokens_since_turn = 0
    turns_issued = 0
    burst_run = 0  # consecutive prefill-only iterations so far (alternating only)

    while turns_issued < n_turns or pending_prefill > 0:
        if policy == "alternating" and pending_prefill > 0 and burst_run < MAX_PREFILL_BURST:
            chunk = min(PREFILL_CHUNK, pending_prefill)
            t += chunk * prefill_ms_per_token
            pending_prefill -= chunk
            burst_run += 1
            continue  # pure prefill iteration: no lane advances, no TPOT sample this iteration

        burst_run = 0
        if policy == "mixed" and pending_prefill > 0:
            budget = mixed_budget(decode_step_ms, prefill_ms_per_token, mixed_token_budget_ceiling)
            chunk = min(budget, pending_prefill)
            step_cost = decode_step_ms + chunk * prefill_ms_per_token
            pending_prefill -= chunk
        else:
            step_cost = decode_step_ms
        t += step_cost
        for lane in range(concurrency):
            tpot.append(t - last_token_time[lane])
            last_token_time[lane] = t
        tokens_generated += concurrency
        tokens_since_turn += 1

        if tokens_since_turn >= turn_gap_tokens and pending_prefill == 0 and turns_issued < n_turns:
            pending_prefill = prefill_tokens
            tokens_since_turn = 0
            turns_issued += 1

    tpot.sort()

    def pct(p: float) -> float:
        return tpot[min(len(tpot) - 1, int(p * len(tpot)))]

    wall_s = t / 1e3
    return Result(
        policy=policy,
        p50_tpot_ms=pct(0.50),
        p95_tpot_ms=pct(0.95),
        p99_tpot_ms=pct(0.99),
        throughput_tok_s=tokens_generated / wall_s,
        wall_s=wall_s,
        tokens=tokens_generated,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--decode-step-ms", type=float, default=100.0, help="80-120 ms measured range")
    ap.add_argument(
        "--prefill-ms-per-token",
        type=float,
        default=1.0,
        help="1.0 measured pre-vectorization; try 0.3 for the post-vectorization prediction",
    )
    ap.add_argument("--prefill-tokens", type=int, default=2048, help="one turn's fresh prompt")
    ap.add_argument(
        "--turn-gap-tokens",
        type=int,
        default=8,
        help=(
            "decode iterations between new-turn arrivals (aggregate across all C lanes, not "
            "per-lane): default 8 approximates 48 sessions each turning over roughly every "
            "~380 of their own generated tokens (8 * 48), a sustained-load C=48 regime; raise "
            "it to see the low-contention end of the tradeoff (mixed batching's small "
            "always-on tax against alternating's rare-but-huge stall)"
        ),
    )
    ap.add_argument("--n-turns", type=int, default=30)
    ap.add_argument("--mixed-token-budget", type=int, default=512, help="SEED_MIXED_TOKEN_BUDGET")
    ap.add_argument("--tpot-guardrail-ms", type=float, default=250.0)
    args = ap.parse_args()

    print(
        f"C={args.concurrency} decode_step_ms={args.decode_step_ms} "
        f"prefill_ms_per_token={args.prefill_ms_per_token} prefill_tokens/turn="
        f"{args.prefill_tokens} MAX_PREFILL_BURST={MAX_PREFILL_BURST} PREFILL_CHUNK={PREFILL_CHUNK}"
    )
    for policy in ("alternating", "mixed"):
        r = simulate(
            policy,
            args.concurrency,
            args.decode_step_ms,
            args.prefill_ms_per_token,
            args.prefill_tokens,
            args.turn_gap_tokens,
            args.n_turns,
            args.mixed_token_budget,
        )
        breach = "BREACHES" if r.p95_tpot_ms > args.tpot_guardrail_ms else "within"
        print(
            f"{policy:>11}: p50={r.p50_tpot_ms:8.1f} ms  p95={r.p95_tpot_ms:8.1f} ms "
            f"({breach} {args.tpot_guardrail_ms:.0f} ms guardrail)  p99={r.p99_tpot_ms:8.1f} ms "
            f"throughput={r.throughput_tok_s:8.1f} tok/s  wall={r.wall_s:7.2f} s"
        )


if __name__ == "__main__":
    main()
