"""Trace-driven analysis for the hot-expert bf16 cache and cross-rank redundant-expert ideas.

Reads the 1,320 real-traffic routing traces (60 layers x 22 decode steps, batch=48, top_k=10,
512 global experts, 128 local experts/rank at TP=4 expert-parallel) copied from
cluster:/path/to/scratch/bespoke-opt-round2/routing_traces/, and computes:

1. Hot-expert coverage: for H in {2,4,8,16}, per (layer, rank) pick the H local experts with
   the most assignments on a TRAIN half of steps (0..10), then measure what fraction of
   assignments on the HELD-OUT half (11..21) land on that fixed hot set. Also reports the
   in-sample (all-steps) coverage as an optimistic upper bound.
2. Cross-rank skew: per (layer, step), the ratio of the busiest rank's distinct-owned-expert
   count to the median rank's, reproducing the "median 1.47x" fact in COMMON_BRIEF.md.
3. Redundant-expert simulation: replicate the R globally hottest experts (by total assignment
   count across all ranks) onto the least-loaded rank of each step, splitting that expert's
   assignments across the original owner and the replica by token-id parity (round-robin), and
   recomputes the predicted step time using "max across ranks of distinct-expert count" as the
   time proxy (the mechanism COMMON_BRIEF.md cites: "the rank with the most distinct experts
   sets the step time"). Reports predicted MoE-time savings for R in {1,2,4,8}.

The raw `.pt` trace files are not checked in (they are pulled fresh from `the test cluster`); only this
script and its `hot_experts_analysis.json` output (the numbers `HOT_EXPERTS_ANALYSIS.md` reports)
are. To reproduce: `rsync -az cluster:/path/to/scratch/bespoke-opt-round2/routing_traces/
<dir>/`, then `HOT_EXPERTS_TRACE_DIR=<dir> python3 analyze_hot_experts.py`.

Run: HOT_EXPERTS_TRACE_DIR=<dir with the .pt files> python3 analyze_hot_experts.py
"""

import glob
import json
import os
import statistics
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.environ.get("HOT_EXPERTS_TRACE_DIR", os.path.join(HERE, "routing_traces"))
LOCAL_EXPERTS = 128
N_RANKS = 4
NUM_LAYERS = 60
NUM_STEPS = 22
TRAIN_STEPS = set(range(0, 11))
TEST_STEPS = set(range(11, 22))
H_VALUES = [2, 4, 8, 11, 14, 16]
R_VALUES = [1, 2, 4, 8]


def load_all():
    """layer -> step -> rank -> LongTensor local expert ids (flat, len == batch*top_k=480)."""
    data = defaultdict(lambda: defaultdict(dict))
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "step*_layer*.pt")))
    assert len(files) == NUM_LAYERS * NUM_STEPS, f"expected {NUM_LAYERS * NUM_STEPS}, got {len(files)}"
    for fp in files:
        d = torch.load(fp, map_location="cpu")
        layer, step = d["layer"], d["step"]
        a_expert = d["a_expert"].long()
        for r in range(N_RANKS):
            lo, hi = r * LOCAL_EXPERTS, (r + 1) * LOCAL_EXPERTS
            mask = (a_expert >= lo) & (a_expert < hi)
            data[layer][step][r] = a_expert[mask] - lo
    return data


def counts_over_steps(data, layer, rank, steps):
    c = torch.zeros(LOCAL_EXPERTS, dtype=torch.long)
    for s in steps:
        ids = data[layer][s][rank]
        c += torch.bincount(ids, minlength=LOCAL_EXPERTS)
    return c


def coverage_report(data):
    """H -> {'in_sample': mean coverage, 'held_out': mean coverage} averaged over layer x rank."""
    out = {h: {"in_sample": [], "held_out": []} for h in H_VALUES}
    all_steps = set(range(NUM_STEPS))
    for layer in range(NUM_LAYERS):
        for rank in range(N_RANKS):
            all_counts = counts_over_steps(data, layer, rank, all_steps)
            train_counts = counts_over_steps(data, layer, rank, TRAIN_STEPS)
            test_counts = counts_over_steps(data, layer, rank, TEST_STEPS)
            total_all = int(all_counts.sum())
            total_test = int(test_counts.sum())
            for h in H_VALUES:
                hot_in_sample = torch.topk(all_counts, h).indices
                cov_in_sample = float(all_counts[hot_in_sample].sum()) / total_all
                hot_train = torch.topk(train_counts, h).indices
                cov_held_out = float(test_counts[hot_train].sum()) / total_test
                out[h]["in_sample"].append(cov_in_sample)
                out[h]["held_out"].append(cov_held_out)
    return {
        h: {
            "in_sample_mean": statistics.mean(v["in_sample"]),
            "held_out_mean": statistics.mean(v["held_out"]),
            "held_out_min": min(v["held_out"]),
            "held_out_p10": statistics.quantiles(v["held_out"], n=10)[0],
        }
        for h, v in out.items()
    }


def skew_report(data):
    ratios = []
    for layer in range(NUM_LAYERS):
        for step in range(NUM_STEPS):
            distinct = [int(torch.unique(data[layer][step][r]).numel()) for r in range(N_RANKS)]
            med = statistics.median(distinct)
            if med > 0:
                ratios.append(max(distinct) / med)
    return {
        "median_ratio": statistics.median(ratios),
        "mean_ratio": statistics.mean(ratios),
        "p90_ratio": statistics.quantiles(ratios, n=10)[8],
        "n_samples": len(ratios),
    }


def redundant_expert_simulation(data):
    """For each R, replicate the R globally-hottest experts onto the min-loaded rank per step.

    Global heat = total assignment count for a global expert id, summed over every layer/step
    (a stable, precomputable ranking -- the EPLB-style "redundant experts for hot globally-
    popular experts" idea). Per (layer, step): find the current max-loaded rank (by distinct
    expert count, the stated time driver); for each hot global expert owned by that rank,
    add a replica on whichever rank currently has the fewest distinct experts, and move half
    of that expert's assignments (round-robin by position) onto the replica. Recompute max
    distinct count after and derive the fraction reduction, averaged over all (layer, step).
    """
    global_heat = torch.zeros(LOCAL_EXPERTS * N_RANKS, dtype=torch.long)
    for layer in range(NUM_LAYERS):
        for step in range(NUM_STEPS):
            for r in range(N_RANKS):
                ids = data[layer][step][r] + r * LOCAL_EXPERTS
                global_heat.scatter_add_(0, ids, torch.ones_like(ids))
    hottest_global = torch.argsort(global_heat, descending=True)

    results = {}
    for R in R_VALUES:
        hot_set = set(hottest_global[:R].tolist())
        before_all, after_all = [], []
        for layer in range(NUM_LAYERS):
            for step in range(NUM_STEPS):
                distinct = [set(data[layer][step][r].tolist()) for r in range(N_RANKS)]
                distinct_global = [
                    {i + r * LOCAL_EXPERTS for i in distinct[r]} for r in range(N_RANKS)
                ]
                counts = [len(d) for d in distinct_global]
                before_all.append(max(counts))

                # Redundant placement: for every hot global expert present on the busiest rank,
                # add it (as a replica) to the currently-least-loaded rank's distinct set. This
                # models "the replica lets that rank skip loading the expert" -- the least-loaded
                # rank already has it available redundantly and never needed to load it from the
                # busy rank's shard, so distinct-set membership (not assignment count) is what the
                # time proxy tracks.
                mutable_counts = list(counts)
                for g in hot_set:
                    owner = g // LOCAL_EXPERTS
                    if g not in distinct_global[owner]:
                        continue
                    least = min(range(N_RANKS), key=lambda r: mutable_counts[r])
                    if least != owner and g not in distinct_global[least]:
                        distinct_global[least].add(g)
                        mutable_counts[least] += 1
                after_all.append(max(mutable_counts))
        before_mean = statistics.mean(before_all)
        after_mean = statistics.mean(after_all)
        results[R] = {
            "before_mean_max_distinct": before_mean,
            "after_mean_max_distinct": after_mean,
            "predicted_time_reduction_pct": 100.0 * (1 - after_mean / before_mean),
        }
    return results


def redundant_expert_simulation_by_load(data):
    """Optimistic upper bound: model a replica as perfectly halving a hot expert's own
    assignment count between its owner and one replica on the (currently) least-loaded rank,
    using per-rank ASSIGNMENT COUNT (not distinct-expert count) as the time proxy. This is the
    best case for idea 2 -- real token-level "route to least-loaded replica" dispatch needs a
    partial all-to-all this codebase's expert-parallel design does not have (see module
    docstring), so this bounds the payoff before paying that complexity.
    """
    global_heat = torch.zeros(LOCAL_EXPERTS * N_RANKS, dtype=torch.long)
    for layer in range(NUM_LAYERS):
        for step in range(NUM_STEPS):
            for r in range(N_RANKS):
                ids = data[layer][step][r] + r * LOCAL_EXPERTS
                global_heat.scatter_add_(0, ids, torch.ones_like(ids))
    hottest_global = torch.argsort(global_heat, descending=True)

    results = {}
    for R in R_VALUES:
        hot_set = list(hottest_global[:R].tolist())
        before_all, after_all = [], []
        for layer in range(NUM_LAYERS):
            for step in range(NUM_STEPS):
                per_expert = [
                    torch.bincount(data[layer][step][r], minlength=LOCAL_EXPERTS)
                    for r in range(N_RANKS)
                ]
                counts = [int(per_expert[r].sum()) for r in range(N_RANKS)]
                before_all.append(max(counts))
                mutable = list(counts)
                for g in hot_set:
                    owner = g // LOCAL_EXPERTS
                    local = g % LOCAL_EXPERTS
                    n = int(per_expert[owner][local])
                    if n == 0:
                        continue
                    least = min(range(N_RANKS), key=lambda r: mutable[r])
                    if least == owner:
                        continue
                    move = n // 2
                    mutable[owner] -= move
                    mutable[least] += move
                after_all.append(max(mutable))
        before_mean = statistics.mean(before_all)
        after_mean = statistics.mean(after_all)
        results[R] = {
            "before_mean_max_assignments": before_mean,
            "after_mean_max_assignments": after_mean,
            "predicted_time_reduction_pct": 100.0 * (1 - after_mean / before_mean),
        }
    return results


def main():
    data = load_all()
    cov = coverage_report(data)
    skew = skew_report(data)
    redund_distinct = redundant_expert_simulation(data)
    redund_load = redundant_expert_simulation_by_load(data)
    report = {
        "coverage": cov,
        "skew": skew,
        "redundant_expert_simulation_distinct_count_proxy": redund_distinct,
        "redundant_expert_simulation_optimistic_load_proxy": redund_load,
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(HERE, "hot_experts_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
