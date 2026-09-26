"""In-graph device timestamps for decode-step attribution. Diagnostic only; inert by default.

ROCm's torch refuses timing events inside a captured graph ("External events are disallowed
in rocm"), and a tracer inflated the captured B96 decode step 10.4x in round 12. This records
time from inside the graph instead: `mark(tag)` launches a one-thread Triton kernel that
writes the GPU's constant 100 MHz clock (`s_memrealtime`) into slot `k` of a device buffer.
On one stream each stamp starts after the previous kernel finishes, so the difference between
consecutive stamps is the stream time of the segment that ended at the later stamp. Measured
cost on MI300A: about 1.7 us per stamp in a graph.

`mark` does nothing unless a `Recorder` is installed (`RECORDER`), which only
`decode_attribution.py` does, and only around the capture of a separate diagnostic graph. The
production graphs never contain a stamp. Call sites cost one global read at trace time.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _stamp_kernel(buf, idx):
        t = tl.inline_asm_elementwise(
            "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
            "=s",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        tl.store(buf + idx, t)

except ImportError:  # pragma: no cover - CPU environments without triton
    _stamp_kernel = None

CLOCK_HZ = 100e6
"""`s_memrealtime` frequency on gfx942 (confirmed: stamp span equals replay wall)."""


class Recorder:
    """A device buffer of `capacity` int64 stamps and the host-side tag of each slot."""

    def __init__(self, device: torch.device, capacity: int = 4096) -> None:
        self.buf = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.tags: list[str] = []

    def stamp(self, tag: str) -> None:
        k = len(self.tags)
        if k >= self.buf.numel():
            raise RuntimeError("decode_stamps: recorder capacity exceeded")
        self.tags.append(tag)
        _stamp_kernel[(1,)](self.buf, k)

    def reset(self) -> None:
        self.tags = []

    def segments_ms(self) -> list[tuple[str, float]]:
        """`(tag, ms)` for each segment: the time from the previous stamp to this one."""
        ticks = self.buf[: len(self.tags)].tolist()
        return [
            (self.tags[k], (ticks[k] - ticks[k - 1]) / CLOCK_HZ * 1e3)
            for k in range(1, len(self.tags))
        ]


RECORDER: Recorder | None = None


def mark(tag: str) -> None:
    """End the current segment as `tag` (no-op unless a `Recorder` is installed)."""
    rec = RECORDER
    if rec is not None:
        rec.stamp(tag)
