"""Split a rocprofv3 kernel trace of `prof_widths.py` into cases and components.

    python3 prof_split.py <trace dir> <run.log> [--top 25]

Reads rank 0's `*_kernel_trace.csv` (rank 0's pid is the `PID` line of run.log), splits the
kernels by host gaps > 0.5 s (`prof_widths.py` sleeps 1 s around each case), keeps the last
len(cases) groups, and prints per case the kernel ms per component (MoE, dense GEMM,
DeltaNet, attention, all-reduce, glue) plus the top kernels.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict

RULES = [
    ("allreduce", r"nccl|rccl|all_?reduce|allreduce|cross_device|custom_ar|two_?shot|one_?shot"),
    ("moe", r"moe_kernel|_pf_|_bw_|bw_combine|_fused_route|route_kernel|_grouped_moe|gate_up_silu"
     r"|mxfp4|moe"),
    ("gemm", r"Cijk|gemm|Gemm|GEMM|skinny|wvSplit|hipblaslt|MT\d+x\d+"),
    ("deltanet", r"deltanet|delta|_dn_|chunk|recurr|conv1d|causal_conv|_wy|_state_kernel|prep_kernel"),
    ("attention", r"attn|attention|flash|varlen|paged|sdpa|fmha|rope|softmax_kv"),
]


def classify(name: str) -> str:
    for comp, pat in RULES:
        if re.search(pat, name):
            return comp
    return "glue"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("log")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    log = open(a.log).read()
    pid = re.search(r"^PID (\d+)", log, re.M).group(1)
    cases = re.findall(r"^CASE (\S+) wall_ms=([\d.]+)", log, re.M)
    files = [f for f in glob.glob(f"{a.trace}/**/*kernel_trace.csv", recursive=True) if pid in f]
    rows = []
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()
    groups, cur, last_end = [], [], None
    for s, e, n in rows:
        if last_end is not None and s - last_end > 500_000_000:
            groups.append(cur)
            cur = []
        cur.append((s, e, n))
        last_end = max(last_end or 0, e)
    groups.append(cur)
    groups = groups[-len(cases) :]
    out = []
    for (case, wall), g in zip(cases, groups, strict=True):
        comp = defaultdict(float)
        per = defaultdict(lambda: [0.0, 0])
        for s, e, n in g:
            ms = (e - s) / 1e6
            comp[classify(n)] += ms
            per[n][0] += ms
            per[n][1] += 1
        span = (g[-1][1] - g[0][0]) / 1e6
        total = sum(comp.values())
        res = {
            "case": case,
            "wall_ms": float(wall),
            "gpu_span_ms": round(span, 1),
            "kernel_ms": round(total, 1),
            "launches": len(g),
            **{k: round(v, 2) for k, v in sorted(comp.items())},
        }
        out.append(res)
        print("CASE " + json.dumps(res))
        for n, (ms, cnt) in sorted(per.items(), key=lambda kv: -kv[1][0])[: a.top]:
            print(f"  {ms:8.2f} {cnt:5d} {classify(n):9s} {n[:120]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
