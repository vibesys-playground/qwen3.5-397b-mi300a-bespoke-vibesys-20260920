"""Merge per-rank TunableOp caches: for every GEMM key keep the solution with the lowest
recorded time on any rank, and write that merged table as every rank's file.
usage: merge_tunableop.py <src dir> <dst dir>"""

import csv, glob, os, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "tunableop_*_[0-9].csv")))
best, header = {}, None
for f in files:
    rows = list(csv.reader(open(f)))
    if header is None:
        header = [r for r in rows if r and r[0] == "Validator"]
    for r in rows:
        if r and r[0].startswith("Gemm"):
            k = (r[0], r[1])
            t = float(r[3])
            if k not in best or t < best[k][1]:
                best[k] = (r[2], t)
for f in files:
    with open(os.path.join(dst, os.path.basename(f)), "w", newline="") as out:
        w = csv.writer(out, lineterminator="\n")
        for r in header:
            w.writerow(r)
        for (op, shape), (sol, t) in sorted(best.items()):
            w.writerow([op, shape, sol, f"{t:g}"])
print(f"merged {len(best)} keys from {len(files)} files into {dst}")
