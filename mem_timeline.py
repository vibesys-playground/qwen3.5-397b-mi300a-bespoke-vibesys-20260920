"""Boot memory timeline and per-buffer read bandwidth (`SEED_MEM_TIMELINE=1`). Diagnostic only.

On MI300A a device allocation's read bandwidth is fixed when it is allocated: made while its
GPU's NUMA node still has plenty of free memory it reads at ~2.7 TB/s, made while the node is
low it reads at 1.6-2.0 TB/s, and made on an exhausted node it spills to a neighbor node at
~0.1 TB/s. This module records, at named points during boot (`mark`), every node's MemFree and
which caching-allocator segments appeared since the previous point. `report` then times a
read of every segment and prints, per buffer class (weights, DeltaNet pool, KV pool, snapshot
pool, scratch, graph pools, other), how many bytes were allocated at which free level and how
fast they read. Nothing here changes numerics; with the flag off every call is a no-op.
"""

from __future__ import annotations

import bisect
import json
import os
import time
from pathlib import Path

import torch

ENABLED = os.environ.get("SEED_MEM_TIMELINE", "0") == "1"
OUT_DIR = os.environ.get("SEED_MEM_TIMELINE_DIR", "")
"""Where `report` writes `mem_segments_rank<r>.json` (every segment); empty: not written."""
MIN_PROBE_BYTES = 32 << 20
"""Segments smaller than this are counted but not timed (launch overhead dominates)."""

_T0 = time.time()
_segs: dict[int, dict] = {}
_prev_free: list[float] | None = None
_rank = int(os.environ.get("SEED_MEM_TIMELINE_RANK", "-1"))


def set_rank(rank: int) -> None:
    global _rank  # noqa: PLW0603
    _rank = rank


def node_free_gib(root: Path = Path("/sys/devices/system/node")) -> list[float]:
    """Per-NUMA-node MemFree in GiB, node order. Empty off Linux."""
    out = []
    for d in sorted(root.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        for line in (d / "meminfo").read_text().splitlines():
            p = line.split()
            if len(p) >= 4 and p[2] == "MemFree:":
                out.append(round(int(p[3]) / 2**20, 2))
    return out


def _device(device: torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda", torch.cuda.current_device())


def _node_of(device: torch.device) -> int:
    """MI300A: GPU i is the accelerator of NUMA node i."""
    return device.index or 0


def mark(label: str, device: torch.device | str | None = None) -> None:
    """Log every node's MemFree and tag the segments allocated since the previous mark."""
    global _prev_free  # noqa: PLW0603
    if not ENABLED or not torch.cuda.is_available():
        return
    dev = _device(device)
    if dev.type != "cuda":
        return
    torch.cuda.synchronize(dev)
    free = node_free_gib()
    node = _node_of(dev)
    before = _prev_free if _prev_free is not None else free
    live = set()
    new_bytes = 0
    for seg in torch.cuda.memory_snapshot():
        if seg["device"] != dev.index:
            continue
        addr = seg["address"]
        live.add(addr)
        if addr in _segs and _segs[addr]["size"] == seg["total_size"]:
            continue
        pool = tuple(seg.get("segment_pool_id", (0, 0)))
        _segs[addr] = {
            "addr": addr,
            "size": seg["total_size"],
            "graph_pool": pool != (0, 0),
            "phase": label,
            "free_prev": before[node] if node < len(before) else None,
            "free_now": free[node] if node < len(free) else None,
        }
        new_bytes += seg["total_size"]
    for addr in [a for a in _segs if a not in live]:
        del _segs[addr]  # released (empty_cache); a later segment may reuse the address
    _prev_free = free
    rec = {
        "rank": _rank,
        "t": round(time.time() - _T0, 1),
        "label": label,
        "free": free,
        "new_gib": round(new_bytes / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(dev) / 2**30, 2),
    }
    print(f"[mem-tl] {json.dumps(rec)}", flush=True)


def _tensors(obj, out: list[torch.Tensor], depth: int = 0) -> None:  # noqa: ANN001
    if depth > 6 or obj is None:
        return
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _tensors(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _tensors(v, out, depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, torch.nn.Module)):
        for v in vars(obj).values():
            _tensors(v, out, depth + 1)


def hot_classes(model) -> dict[str, list[torch.Tensor]]:  # noqa: ANN001
    """The buffers a decode or prefill step reads, by class, in priority order."""
    pools = [p for p in model.pool if "conv" in p]
    kv = [p for p in model.pool if "k" in p]
    out: dict[str, list[torch.Tensor]] = {}
    for name, objs in (
        ("weights", [model.embed, model.layers, model.final_norm, model.lm_head]),
        ("mtp", [getattr(model, "mtp", None)]),
        ("dn_pool", pools),
        ("kv_pool", kv),
        ("snapshot", [model.snapshot_pool, model.node_logits_scratch]),
        (
            "scratch",
            [
                model.delta_scratch,
                model.moe_scratch,
                model.moe_inter_scratch,
                model.moe_dedup_y_scratch,
                model.moe_route_cursor,
            ],
        ),
    ):
        ts: list[torch.Tensor] = []
        _tensors(objs, ts)
        out[name] = [t for t in ts if t.is_cuda and t.numel()]
    return out


class _Raw:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes // 4,),
            "typestr": "<f4",
            "data": (ptr, False),
            "version": 2,
        }


def _read_tbs(view: torch.Tensor) -> float:
    view.sum()
    torch.cuda.synchronize(view.device)
    best = float("inf")
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        view.sum()
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e))
    return view.numel() * 4 / (best * 1e-3) / 1e12


def _bucket(free: float | None) -> str:
    if free is None:
        return "?"
    for hi in (5, 10, 15, 20, 25, 30, 40):
        if free < hi:
            return f"<{hi}"
    return ">=40"


def report(model, device: torch.device | str | None = None) -> None:  # noqa: ANN001
    """Time a read of every tagged segment and print `[mem-seg]` per buffer class."""
    if not ENABLED or not torch.cuda.is_available():
        return
    dev = _device(device)
    mark("report", dev)
    starts = sorted(_segs)
    cls_of: dict[int, str] = {}
    for name, ts in hot_classes(model).items():
        for t in ts:
            ptr = t.untyped_storage().data_ptr()
            i = bisect.bisect_right(starts, ptr) - 1
            if i >= 0 and ptr < starts[i] + _segs[starts[i]]["size"]:
                cls_of.setdefault(starts[i], name)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    raw_ok = True
    for addr in starts:
        s = _segs[addr]
        s["cls"] = cls_of.get(addr, "graph_pool" if s["graph_pool"] else "other")
        s["tbs"] = None
        if s["size"] < MIN_PROBE_BYTES or not raw_ok:
            continue
        try:
            view = torch.as_tensor(_Raw(addr, s["size"]), device=dev)
        except Exception as exc:  # noqa: BLE001
            print(f"[mem-seg] rank {_rank}: raw segment views unavailable ({exc!r})", flush=True)
            raw_ok = False
            continue
        s["tbs"] = round(_read_tbs(view), 3)
        del view
    summary: dict[str, dict] = {}
    for s in _segs.values():
        c = summary.setdefault(s["cls"], {"gib": 0.0, "timed_gib": 0.0, "ms": 0.0, "by_free": {}})
        c["gib"] += s["size"] / 2**30
        b = c["by_free"].setdefault(_bucket(s["free_now"]), {"gib": 0.0, "timed_gib": 0.0, "ms": 0.0})
        b["gib"] += s["size"] / 2**30
        if s["tbs"]:
            ms = s["size"] / (s["tbs"] * 1e12) * 1e3
            for d in (c, b):
                d["timed_gib"] += s["size"] / 2**30
                d["ms"] += ms
    for c in summary.values():
        for d in (c, *c["by_free"].values()):
            d["tbs"] = round(d["timed_gib"] * 2**30 / (d["ms"] * 1e-3) / 1e12, 3) if d["ms"] else None
            d["gib"], d["timed_gib"], d["ms"] = round(d["gib"], 2), round(d["timed_gib"], 2), round(d["ms"], 2)
    for name, c in sorted(summary.items()):
        print(f"[mem-seg] {json.dumps({'rank': _rank, 'cls': name, **c})}", flush=True)
    if OUT_DIR:
        path = Path(OUT_DIR) / f"mem_segments_rank{_rank}.json"
        path.write_text(json.dumps(sorted(_segs.values(), key=lambda s: s["addr"])))


PREFLIGHT_MIN_FREE_GIB = float(os.environ.get("SEED_BOOT_MEM_PREFLIGHT_GIB", "0"))
"""Launcher preflight: before any rank starts, wait until every NUMA node has at least this
much MemFree (GiB). 0 (the default) skips it. A previous server's ranks, gpucore dumps or
tmpfs files still holding a node's memory make this boot's late allocations slow or spill
them to another node; booting on top of a previous server's residency can OOM the node."""
PREFLIGHT_WAIT_S = float(os.environ.get("SEED_BOOT_MEM_PREFLIGHT_WAIT_S", "300"))


def preflight(
    min_free_gib: float = PREFLIGHT_MIN_FREE_GIB,
    wait_s: float = PREFLIGHT_WAIT_S,
    read=node_free_gib,  # noqa: ANN001
    sleep=time.sleep,  # noqa: ANN001
    clock=time.monotonic,  # noqa: ANN001
) -> None:
    """Block until every node's MemFree >= `min_free_gib`; raise `RuntimeError` after `wait_s`."""
    if min_free_gib <= 0:
        return
    deadline = clock() + wait_s
    while True:
        free = read()
        if free and min(free) >= min_free_gib:
            print(f"[mem-preflight] ok: node MemFree {free} GiB >= {min_free_gib}", flush=True)
            return
        if clock() >= deadline:
            raise RuntimeError(
                f"[mem-preflight] node MemFree {free} GiB still below {min_free_gib} GiB after "
                f"{wait_s:.0f} s; a previous server or stale /dev/shm files hold memory"
            )
        sleep(5)
