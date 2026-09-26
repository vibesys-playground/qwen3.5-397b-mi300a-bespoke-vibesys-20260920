"""Real-row fraction and cost per real prefill token of the mixed steps in a server log.

    python3 mixed_fill.py server.log [server.log ...]

Reads rank 0's `[step-timing] sched mixed ... tokens=N wall_ms=W ... path=graphD+RxW` lines
(`SEED_STEP_TIMING=1`, one line every `SEED_STEP_TIMING_EVERY` steps). For each log prints
the prefill rows carried (tokens) over the prefill rows replayed (R*W), the decode rows used
over the decode bucket, and wall ms per real prefill token after subtracting the same
log's mean decode-step wall ms for that bucket, overall and per replayed total.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"sched (mixed|decode) n=\d+ batch=(\d+) tokens=(\d+) wall_ms=([\d.]+).*?path=(\S+)"
)
SHAPE = re.compile(r"graph(\d+)\+(\d+)x(\d+)")


def main() -> None:
    for path in sys.argv[1:]:
        dec = defaultdict(list)
        mixed = []
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if not m:
                continue
            kind, batch, tokens, wall, shape = m.groups()
            if kind == "decode":
                dec[int(batch)].append(float(wall))
                continue
            s = SHAPE.match(shape)
            if s:
                d, r, w = map(int, s.groups())
                mixed.append((d, r * w, int(batch), int(tokens), float(wall)))
        if not mixed:
            print(path, "no mixed steps")
            continue
        dec_ms = {b: sum(v) / len(v) for b, v in dec.items()}

        def dec_cost(d: int) -> float:
            near = [b for b in dec_ms if b <= d] or list(dec_ms)
            return dec_ms[max(near)] if near else 0.0

        real = sum(t for *_, t, _ in mixed)
        area = sum(a for _, a, *_ in mixed)
        extra = sum(w - dec_cost(d) for d, _, _, _, w in mixed)
        print(
            f"{path}: {len(mixed)} mixed steps, prefill rows real/replayed {real}/{area} = "
            f"{real / area:.2f}, decode rows {sum(b for *_, b, _, _ in mixed)}/"
            f"{sum(d for d, *_ in mixed)}, mean wall {sum(w for *_, w in mixed) / len(mixed):.1f}"
            f" ms, (wall - decode) per real prefill token {extra / real:.3f} ms"
        )
        by = defaultdict(lambda: [0, 0, 0, 0.0])
        for d, a, _, t, w in mixed:
            k = by[d + a]
            k[0] += 1
            k[1] += t
            k[2] += a
            k[3] += w
        for tot, (n, t, a, w) in sorted(by.items()):
            print(f"  total {tot:5d}: n={n:4d} fill {t / a:.2f} mean_ms {w / n:6.1f}")


if __name__ == "__main__":
    main()
