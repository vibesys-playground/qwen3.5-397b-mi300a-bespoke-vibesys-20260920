"""Group rocprofv3 kernel traces from decode_prof_rp.py into per-step ms by kernel group.

usage: parse_rp.py <rp dir> <labels comma list, e.g. base:b16,base:b48> [psteps] [--top N]
Each pid's trace is cut at spin_kernel markers (pairs bracket one window of psteps steps)."""

import collections, csv, glob, os, re, sys

GROUPS = [
    ("moe", r"moe_kernel|_bw_combine|mxfp4|_moe_"),
    ("routing", r"route|_bw_prep|topk|_eplb"),
    ("allreduce", r"oneshot|_ar_add_rmsnorm|allreduce|nccl|rccl"),
    ("attention", r"splitk|_combine_kernel|rope|attn|paged"),
    ("deltanet", r"delta_rule|causal_conv|gated_rmsnorm|conv"),
    ("dense", r"Cijk|sk_kernel|skinny|gemv|wvSplit|gemm|_linear"),
    ("glue", r"."),
]


def group(name):
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return "glue"


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    int(r["Start_Timestamp"]),
                    int(r["End_Timestamp"]),
                    r["Kernel_Name"],
                    r.get("Agent_Id", ""),
                )
            )
    rows.sort()
    return rows


def windows(rows):
    marks = [i for i, r in enumerate(rows) if "spin_kernel" in r[2] or "sleep" in r[2].lower()]
    return [(marks[j] + 1, marks[j + 1]) for j in range(0, len(marks) - 1, 2)]


def main():
    d, labels = sys.argv[1], sys.argv[2].split(",")
    psteps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0
    files = sorted(glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True))
    per_rank = []
    for fpath in files:
        rows = load(fpath)
        ws = windows(rows)
        if len(ws) < len(labels):
            continue
        ws = ws[-len(labels) :]
        per_rank.append((fpath, rows, ws))
    print(f"{len(per_rank)} traces with {len(labels)} windows")
    for li, lab in enumerate(labels):
        tab = collections.defaultdict(list)
        steps = []
        for fpath, rows, ws in per_rank:
            a, b = ws[li]
            seg = rows[a:b]
            g = collections.Counter()
            n = collections.Counter()
            for s, e, name, _ in seg:
                g[group(name)] += (e - s) / 1e6 / psteps
                n[group(name)] += 1 / psteps
            span = (seg[-1][1] - seg[0][0]) / 1e6 / psteps if seg else 0
            steps.append((span, len(seg) / psteps, sum(g.values())))
            for k in g:
                tab[k].append((g[k], n[k]))
        print(
            f"== {lab}: step span ms per rank {[round(x[0], 2) for x in steps]}, kernels/step {[round(x[1]) for x in steps]}, busy ms {[round(x[2], 2) for x in steps]}"
        )
        for k, _ in GROUPS:
            if k in tab:
                v = tab[k]
                print(
                    f"  {k:10s} ms avg {sum(x[0] for x in v) / len(v):6.2f}  ranks {[round(x[0], 2) for x in v]}  kernels {round(v[0][1])}"
                )
        if top:
            fpath, rows, ws = per_rank[0]
            a, b = ws[li]
            c = collections.Counter()
            cn = collections.Counter()
            for s, e, name, _ in rows[a:b]:
                c[name[:90]] += (e - s) / 1e6 / psteps
                cn[name[:90]] += 1 / psteps
            for name, t in c.most_common(top):
                print(f"    {t:7.3f} ms {cn[name]:6.0f}x {group(name):9s} {name}")


if __name__ == "__main__":
    main()
