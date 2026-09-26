"""Host-side NUMA placement for the rank processes, and a boot probe of per-rank read bandwidth.

On MI300A a GPU's allocations come from its own socket's share of the unified pool until that
NUMA node is nearly full; after that they are slower, and once the node is exhausted they spill
to a neighbor node, where reads run 30x slower. Host pages count against the same per-node
budget: page cache from the weight reads, each rank's host RSS, and tmpfs files. All ranks run
unpinned, and node 0 also holds the container image's pages, so GPU0 fills first and rank 0's
device work runs slowest. Tensor parallelism then runs every step at that pace.

`apply()` sets this process's memory policy (and optionally its CPU affinity) from
environment flags before anything is allocated. Children inherit both. Device allocations do
not follow the process memory policy (the driver places them on the GPU's own node), so only
host pages move:

- `SEED_HOST_MEM_NODES=1,2,3`: interleave this process's host pages over these nodes.
- `SEED_HOST_CPU_NODES=1,2,3`: run this process's threads only on these nodes' CPUs.

`probe()` (`SEED_RANK_BW_PROBE=1`) times reads of this rank's largest device tensors after
boot and prints one `[rank-bw]` line per rank with the per-node memory it sees. It is a
diagnostic and changes no numbers.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
from pathlib import Path

_MPOL_INTERLEAVE = 3
_SYS_SET_MEMPOLICY = 238  # x86_64


def parse_nodes(spec: str) -> list[int]:
    """`"1,2,3"` or `"1-3"` -> sorted unique node ids. Empty spec -> []."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def node_cpus(node: int, root: Path = Path("/sys/devices/system/node")) -> set[int]:
    return set(parse_nodes((root / f"node{node}" / "cpulist").read_text().strip()))


def apply(env: dict[str, str] | None = None) -> str:
    """Apply the host placement flags to this process. Returns a one-line summary ("" if none)."""
    env = os.environ if env is None else env
    notes = []
    cpu_nodes = parse_nodes(env.get("SEED_HOST_CPU_NODES", ""))
    if cpu_nodes:
        os.sched_setaffinity(0, set().union(*(node_cpus(n) for n in cpu_nodes)))
        notes.append(f"cpu_nodes={cpu_nodes}")
    mem_nodes = parse_nodes(env.get("SEED_HOST_MEM_NODES", ""))
    if mem_nodes:
        mask = ctypes.c_ulong(sum(1 << n for n in mem_nodes))
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_INTERLEAVE, ctypes.byref(mask), 64)
        if rc != 0:
            raise OSError(ctypes.get_errno(), f"set_mempolicy(INTERLEAVE, {mem_nodes})")
        notes.append(f"mem_interleave={mem_nodes}")
    return " ".join(notes)


def node_meminfo_gib() -> dict[str, dict[str, float]]:
    """Per-NUMA-node MemUsed/MemFree/FilePages in GiB (empty off Linux)."""
    out = {}
    for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        vals = {}
        for line in (d / "meminfo").read_text().splitlines():
            p = line.split()
            if p[2].rstrip(":") in ("MemUsed", "MemFree", "FilePages"):
                vals[p[2].rstrip(":")] = round(int(p[3]) / 2**20, 1)
        out[d.name] = vals
    return out


def probe(rank: int, device: object, top: int = 96) -> None:
    """Print `[rank-bw]`: read TB/s over this rank's `top` largest device tensors (by bytes).

    Tensors are found by walking the garbage collector's objects once, so this belongs at boot.
    Reports the aggregate, the slowest single tensor, and how many run below 60% of the median.
    """
    import torch

    seen, tensors = set(), []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device == device and obj.numel():
                key = (obj.untyped_storage().data_ptr(), obj.untyped_storage().nbytes())
                if key not in seen:
                    seen.add(key)
                    tensors.append(obj)
        except Exception:  # noqa: BLE001, S112 - some objects refuse inspection
            continue
    tensors.sort(key=lambda t: t.untyped_storage().nbytes(), reverse=True)
    views = []
    for t in tensors[:top]:
        flat = torch.empty(0, dtype=torch.uint8, device=device).set_(t.untyped_storage())
        views.append(flat[: flat.numel() // 4 * 4].view(torch.float32))
    # Start together: a rank that finished early waits in a collective, and its peers' polling
    # of this rank's memory would be timed as this rank's bandwidth.
    torch.cuda.synchronize(device)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    rows = []
    for view in views:
        for _ in range(2):
            view.sum()
        torch.cuda.synchronize(device)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            view.sum()
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e) / 5
        rows.append((view.numel() * 4, ms))
    if not rows:
        return
    total_b, total_ms = sum(b for b, _ in rows), sum(ms for _, ms in rows)
    rates = sorted(b / (ms * 1e-3) / 1e12 for b, ms in rows)
    median = rates[len(rates) // 2]
    rec = {
        "rank": rank,
        "tensors": len(rows),
        "gib": round(total_b / 2**30, 2),
        "read_tbs": round(total_b / (total_ms * 1e-3) / 1e12, 3),
        "slowest_tbs": round(rates[0], 3),
        "median_tbs": round(median, 3),
        "n_below_60pct": sum(r < 0.6 * median for r in rates),
        "numa": node_meminfo_gib(),
        "largest": [(round(b / 2**30, 2), round(b / (ms * 1e-3) / 1e12, 3)) for b, ms in rows[:6]],
        "mem_nodes": os.environ.get("SEED_HOST_MEM_NODES", ""),
        "cpu_nodes": os.environ.get("SEED_HOST_CPU_NODES", ""),
    }
    print(f"[rank-bw] {json.dumps(rec)}", flush=True)
