import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# traj3.py runs plus all 129 round-15 C96 runs (orders 143-271)
ORDER_FIX = {138: 119.5}
DROP = {47, 49, 52, 73, 74, 84, 85, 90, 91, 128, 129, 130, 131, 132}
rows = [r for r in sorted(csv.DictReader(open("bespoke_runs_r15.csv")),
                          key=lambda r: ORDER_FIX.get(int(r["order"]), float(r["order"])))
        if int(r["order"]) >= 47 and int(r["order"]) not in DROP]
y = np.array([float(r["tok_s"]) for r in rows])
x = np.arange(len(y))
best = np.maximum.accumulate(y)
pos = {int(r["order"]): i for i, r in enumerate(rows)}

BLUE = "#2f6aa8"
plt.rcParams.update({"font.size": 15, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(14, 4.95), dpi=170)

# gradient fill under the best-so-far step
YMAX = 2600
xs = np.repeat(x, 2)[1:]; ys = np.repeat(best, 2)[:-1]
fill = ax.fill_between(xs, ys, 0, color="none")
cmap = LinearSegmentedColormap.from_list("g", ["#f4f7fb", "#a9c3de"])
grad = ax.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", cmap=cmap, origin="lower",
                 extent=(0, x[-1], 0, best.max()), zorder=1)
grad.set_clip_path(fill.get_paths()[0], transform=ax.transData)

ax.scatter(x, y, s=9, color="#b0b0b0", zorder=2, label="Individual runs")
ax.step(x, best, where="post", color=BLUE, lw=2, zorder=3)
ax.plot(x, best, "o", color=BLUE, ms=3.5, zorder=4, label="VibeServe (ours)")
ax.axhline(961, color="gray", ls="--", lw=1.3, zorder=2, label="SGLang baseline")

NOTES = [  # (run order, label, absolute text position in data units)
    (51, "Prefix cache", (9, 640)),
    (68, "Decode\ngraphs +\nMoE kernel", (9.5, 1330)),
    (72, "Prefill\ngraphs", (22, 1850)),
    (83, "One-shot\nall-reduce", (32, 1330)),
    (97, "Mixed-step\ngraph", (43, 1850)),
    (105, "Wide MoE +\nGEMM tuning", (63, 330)),
    (112, "Sequence-\nparallel\nall-reduce", (53, 1330)),
    (133, "Decode graphs\nto batch 128", (76, 1480)),
    (136, "Merge tiny prefills\ninto decode", (93, 700)),
    (175, "Wide prefill chunks +\nfused prefill kernels", (107, 1530)),
    (189, "Pack prefills\nwithout padding", (124.5, 2020)),
    (211, "Better prefix\ncache eviction", (151.5, 2280)),
    (259, "Faster SP\nall-reduce", (174, 2000)),
    (267, "MTP speculative decoding", (185.5, 2450)),
]
STRAIGHT = {68, 72, 83, 97, 112, 133}
box = dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1.2)
for order, text, (tx, ty) in NOTES:
    i = pos[order]
    ax.annotate(text, (i, best[i]), xytext=(tx, ty), color=BLUE, fontsize=13,
                ha="center", va="center", bbox=box, zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3,
                                connectionstyle=f"arc3,rad={0 if order in STRAIGHT else 0.15}", shrinkB=3))

ax.set_xlim(-1, x[-1] + 1); ax.set_ylim(0, YMAX)
ax.set_xlabel("Round"); ax.set_ylabel("Goodput (tok/s) ↑")
ax.grid(axis="y", color="#dddddd", lw=0.8)
ax.legend(loc="lower right", fontsize=14, frameon=True, edgecolor="#cccccc")
fig.tight_layout()
fig.savefig("trajectory_styled_r15.png")
