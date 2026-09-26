"""Pick a GEMM solution per decode shape, instead of taking rocBLAS's default heuristic.

The problem. A decode step's dense projections are `[B, hidden] @ [hidden, out]` with B the
batch, so on this deployment every one of them is M = 48 against K and N in the thousands:
GEMV-shaped, and bounded by the weight bytes, not by the FLOPs. rocBLAS picks a macro tile
for them as if M were large. Measured on one MI300A (gfx942, 228 CUs) at the real per-rank
TP=4 shapes, the shared expert's `gate_proj` (K = 4096, N = 256, 2.1 MB of weights) gets
`MT256x224x64`. With M = 48 there is exactly one tile on the M axis, so the launch grid is
`ceil(256 / 224) = 2` workgroups: two CUs out of 228 do the whole GEMM, and it takes 126 us
where the bytes say 0.6 us. The same mis-selection, to varying degrees, applies to every
dense projection in the step; together they were the largest device-time term in the step
(`TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 7 item 4).

The fix. PyTorch's TunableOp benchmarks every rocBLAS and hipBLASLt solution for a shape the
first time it sees it and remembers the fastest. Given the choice, it takes tiles that are
small on M and N and deep on K -- `MT16x16x256` for that `gate_proj`, `MT32x64x256` for
`q_proj` -- which is the shape of a grid that fills the machine. Measured, same GPU, sum over
one decode step's dense GEMM calls at their real per-rank shapes and per-step counts:

    default heuristic, before `fuse_moe_dense` (451 calls)   34.87 ms
    tuned solutions, before `fuse_moe_dense`                  8.09 ms   4.3x
    default heuristic, after `fuse_moe_dense` (331 calls)    20.15 ms
    tuned solutions, after `fuse_moe_dense`                   5.74 ms   3.5x

and 3.5-3.9x at every batch width the engine serves, not just the full pool: 20.09 -> 5.17 ms
at batch 1, 19.94 -> 5.48 at 8, 19.99 -> 5.38 at 16.

Forcing hipBLASLt instead of rocBLAS (`TORCH_BLAS_PREFER_HIPBLASLT`) changes nothing: both
routes reach the same Tensile solution and so the same bad tile. A hand-written Triton
skinny-M GEMM was also measured, swept over tile and split-K configurations per shape, and at
17.66 ms against selection's 8.09 it is less than half the win for an order of magnitude more
code, so it is not worth its maintenance.

Why tuning is a startup step and not left on. Tuning a shape costs seconds and synchronizes,
so a shape first seen mid-request would stall it, and a synchronizing GEMM cannot be captured
into a HIP graph. `tune` therefore walks the decode shapes up front, freezes the choices, and
leaves the search off for everything afterwards. Shapes that were never tuned -- prefill's,
and any batch outside `batches` -- keep the default heuristic, which is the behavior this
module replaces and so cannot regress: measured at prefill widths, leaving TunableOp enabled
with nothing tuned for them costs nothing (M = 512, six projections, 469.0 us against 468.9;
M = 2048, 723.8 against 734.2). Results are cached per GPU architecture, so only the first
server start on a machine pays for the search.

Set `SEED_BLAS_TUNE=0` to skip all of this, which is the ablation the measurement above uses
and the escape hatch for bisecting a suspected solution-selection bug.

One table for every rank (`SEED_BLAS_TUNE_MERGE`). Under TP each rank tunes alone, and the
search is noisy at `max_tuning_ms`: 105 of 121 decode shapes in one job's four caches picked
different solutions per rank, and one rank took b16 GEMMs at 22.1 us where a peer's pick ran
9.7 us, which added about 2.3 ms to that rank's dense time per step. Every all-reduce waits for
the slowest rank, so that is 2.3 ms on the step. With the flag, `tune` gives every rank the
same table, holding for each shape the fastest solution any rank measured (`_tune_merged`).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence

    from model import Model

DENSE_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_all",
    "out_proj",
    "router_gate",
    "shared_expert.gate_up_proj",
    "shared_expert.down_proj",
)
"""Per-layer weights the decode step feeds to `F.linear`, by their key in a layer dict.

A layer holds only the ones its mixer type uses, and several of them are views into a
concatenation (`fuse_in_proj`, `fuse_moe_dense`), whose parents appear here rather than the
views: the view names are never the argument to a `F.linear` on the decode path.
"""

DEFAULT_CACHE = Path(
    os.environ.get("SEED_BLAS_CACHE_DIR") or (Path.home() / ".cache" / "qwen35-bespoke")
)
"""Where tuning results live when the caller names no directory.

`SEED_BLAS_CACHE_DIR` overrides the default `~/.cache/qwen35-bespoke`: that path is shared
across every job on a node, so concurrent unrelated jobs read and rewrite the same per-arch
CSVs while tuning, which serializes what should be independent per-job tuning runs (measured
as ranks stuck over 20 minutes in `tune` below, all contending on the same files). Point this
at a job-private directory to isolate one job's cache from the rest.

The file is keyed by GPU architecture and by the ROCm, hipBLASLt and PyTorch versions, which
TunableOp writes into it as validators and checks on read, so a stale file from another
machine or another container image is rejected rather than used.
"""


MERGE = os.environ.get("SEED_BLAS_TUNE_MERGE", "0") not in ("0", "", "false", "False")
"""Give every TP rank one tuning table, per shape the fastest solution any rank measured.
Off by default (each rank tunes and keeps its own). See the module docstring."""


CHILD_TIMEOUT_S = int(os.environ.get("SEED_BLAS_TUNE_CHILD_TIMEOUT_S", "600"))
"""Upper bound on one rank's tuning child under `SEED_BLAS_TUNE_MERGE` (`_tune_merged`)."""


def enabled() -> bool:
    """Whether solution selection should be used at all.

    TunableOp is a ROCm/CUDA BLAS facility, so it is unreachable on CPU, which is where the
    hermetic tests run.
    """
    return os.environ.get("SEED_BLAS_TUNE", "1") != "0" and torch.cuda.is_available()


def cache_file(cache_dir: Path | str | None = None) -> Path:
    """The results file for this process's GPU, `%d` being TunableOp's device-ordinal slot."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return root / f"tunableop_{arch}_%d.csv"


def decode_gemms(model: Model) -> list[torch.Tensor]:
    """The distinct weight tensors the decode step's dense projections multiply against.

    Distinct by shape *and* stride, because a solution is keyed by the leading dimensions as
    well as by M, N and K; a row slice of a wider matrix is a different key from a standalone
    one of the same shape. The real tensors are handed to `tune` rather than fresh ones of
    the same shape for exactly that reason.
    """
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    out: list[torch.Tensor] = []
    for layer in [*model.layers, {"lm_head": model.lm_head}]:
        for name in (*DENSE_PROJECTIONS, "lm_head"):
            w = layer.get(name)
            if w is None or w.dim() != 2:
                continue
            key = (tuple(w.shape), tuple(w.stride()))
            if key not in seen:
                seen.add(key)
                out.append(w)
    return out


def tuned_batches(max_batch: int) -> tuple[int, ...]:
    """Decode batch widths to tune for by default: a single request, and a full slot pool.

    Those are the two the deployment is scored on. The full pool is what a saturated server
    runs and the only width a captured graph replays (`graph_decode`); a batch of one is the
    latency case. Widths in between keep the default heuristic, which is today's behavior and
    so cannot regress, and a deployment that cares about one can name it in `batches`.

    The reason not to simply tune every width up to `max_batch` is cost: the search takes
    roughly 20 s per shape on a cold cache, about ten distinct shapes per width, so each extra
    width is minutes of startup the first time a machine runs this.
    """
    if os.environ.get("SEED_BLAS_TUNE_BUCKETS", "0") not in ("0", "false", "False"):
        # Every captured decode bucket (`graph_decode.build_buckets`), not just the two ends:
        # the graph replays at 2..32 rows too, and an untuned width takes the default
        # heuristic's wide macro tiles (measured at bucket 16: dense GEMMs 20.2 ms/step
        # against 4.4 ms at the tuned 48). Lazy import: `graph_decode` imports `model`.
        from graph_decode import build_buckets

        return tuple(build_buckets(max_batch))
    return (1, max_batch) if max_batch > 1 else (1,)


def tune(
    model: Model,
    batches: Sequence[int] | None = None,
    cache_dir: Path | str | None = None,
    max_tuning_ms: int = 10,
) -> int:
    """Select and freeze a GEMM solution for each decode shape. Returns the number tuned.

    Call once per rank after its weights are loaded and before it serves anything: the search
    synchronizes, so it must not overlap a request or a graph capture. Tuning is per process
    and per device and needs no collective, so ranks do this independently.

    `max_tuning_ms` caps the time TunableOp spends measuring one candidate solution. The
    default trades a little of the search's precision for a startup cost that is minutes
    rather than tens of minutes on a cold cache; the result is written to `cache_dir`, so
    later starts on the same machine pay nothing.
    """
    if not enabled():
        return 0
    if MERGE and model.tp.world > 1:
        return _tune_merged(model, batches, cache_dir, max_tuning_ms)
    tunable = torch.cuda.tunable
    path = cache_file(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(path), True)
    tunable.enable(True)
    tunable.set_max_tuning_duration(max_tuning_ms)
    # `set_filename` resolves the device ordinal into the name, so ask it rather than
    # substituting again here. A missing file is the cold-cache case, not an error.
    resolved = Path(tunable.get_filename())
    if resolved.exists():
        tunable.read_file()

    widths = tuple(batches) if batches is not None else tuned_batches(model.max_batch)
    weights = decode_gemms(model)
    started = time.perf_counter()
    if getattr(model, "vocab_tp", False):
        # Search the full head's shapes, then give the shard shapes their solutions before
        # the loop below reaches them (`pin_vocab_shards`; a loaded key is never re-searched).
        full = full_head(model)
        tunable.tuning_enable(True)
        try:
            for m in widths:
                F.linear(torch.randn(m, full.shape[1], dtype=full.dtype, device=full.device), full)
            torch.cuda.synchronize()
        finally:
            tunable.tuning_enable(False)
        del full
        tunable.write_file()
        validators, table = read_table(resolved)
        write_table(resolved, validators, pin_vocab_shards(table, model))
        tunable.read_file(str(resolved))
    tunable.tuning_enable(True)
    try:
        for m in widths:
            for w in weights:
                # Real values, not zeros: the search is a timing measurement, and it is also
                # what `PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1` would compare if anyone turns it
                # on to audit a selected solution.
                x = torch.randn(m, w.shape[1], dtype=w.dtype, device=w.device)
                F.linear(x, w)
        torch.cuda.synchronize()
    finally:
        # Off for good: everything after this point (prefill, any untuned batch width, and
        # anything a graph captures) takes the default heuristic rather than stalling to
        # search. Failing partway through still leaves the process in that state.
        tunable.tuning_enable(False)
    tunable.write_file()
    count = len(widths) * len(weights)
    print(
        f"blas_tune: {count} decode shapes over batches {widths} "
        f"in {time.perf_counter() - started:.1f}s, cached at {resolved}",
        flush=True,
    )
    return count


# ---------------------------------------------------------------- one table for every rank

Table = dict[tuple[str, str], tuple[str, float]]
"""TunableOp results: `(op signature, params signature) -> (solution, time in ms)`."""


def read_table(path: Path) -> tuple[list[list[str]], Table]:
    """A TunableOp CSV's validator rows and its results. A missing file is empty."""
    validators: list[list[str]] = []
    table: Table = {}
    if not path.exists():
        return validators, table
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0] == "Validator":
                validators.append(row)
            elif len(row) >= 4 and row[0].startswith("Gemm"):
                table[(row[0], row[1])] = (row[2], float(row[3]))
    return validators, table


def merge_tables(tables: Sequence[Table]) -> Table:
    """Per key, the entry with the lowest measured time across `tables` (the first on a tie)."""
    best: Table = {}
    for table in tables:
        for key, (solution, ms) in table.items():
            if key not in best or ms < best[key][1]:
                best[key] = (solution, ms)
    return best


def write_table(path: Path, validators: Sequence[Sequence[str]], table: Table) -> None:
    """Write a TunableOp CSV (validators first, then one row per key, sorted), atomically."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in validators:
            w.writerow(row)
        for (op, params), (solution, ms) in sorted(table.items()):
            w.writerow([op, params, solution, f"{ms:g}"])
    tmp.replace(path)


def pin_vocab_shards(table: Table, model: Model) -> Table:
    """Under `SEED_LMHEAD_VOCAB_TP`: each `lm_head` shard shape takes the solution the search
    picked for the full head at the same width, where the table has it.

    The shard's own best solution is often a different kernel (rocBLAS vs hipBLASLt, another
    tile or split), which reduces over `hidden` in another order, so its logits differed from
    the replicated head's in the last bit at b1 to b16 on MI300A. The full head's kernel run
    on a quarter of the columns computes every logit exactly as before (checked bit for bit at
    b1 to b48, `scratchpad/lmhead_split_check.py --pin`), for about 20-30 us more than the
    shard's own pick (still about a quarter of the full head's time).
    """
    n_shard = model.lm_head.shape[0]
    n_full = n_shard * model.tp.world
    out = dict(table)
    for (op, params), (_solution, ms) in table.items():
        f = params.split("_")  # tn_<N>_<M>_<K>_ld_<lda>_<ldb>_<ldc>
        if len(f) == 8 and f[4] == "ld" and f[1] == f[7] == str(n_shard):
            full = (op, "_".join([f[0], str(n_full), *f[2:7], str(n_full)]))
            if full in table:
                out[(op, params)] = (table[full][0], ms)
    return out


def full_head(model: Model) -> torch.Tensor:
    """A random tensor with the unsharded `lm_head`'s shape and layout (for the search only)."""
    n, k = model.lm_head.shape
    return torch.randn(
        n * model.tp.world, k, dtype=model.lm_head.dtype, device=model.lm_head.device
    )


def _rank_files(root: Path, arch: str) -> list[Path]:
    return sorted(
        p for p in root.glob(f"tunableop_{arch}_*.csv") if p.stem.rsplit("_", 1)[1].isdigit()
    )


def _barrier(model: Model) -> None:
    flag = torch.zeros(1, dtype=torch.float32, device=model.tp.device)
    model.tp.all_reduce(flag)
    flag.item()


def _tune_merged(
    model: Model,
    batches: Sequence[int] | None,
    cache_dir: Path | str | None,
    max_tuning_ms: int,
) -> int:
    """`tune` under `SEED_BLAS_TUNE_MERGE`: every rank ends with the same table.

    TunableOp keeps its first result for a key: `read_file` does not replace a solution the
    process already holds (checked on this image), so a rank that tuned in-process could not
    adopt a peer's faster pick. The search therefore runs in a child process per rank (same
    device, same GEMM shapes and strides on random data), and this process only reads tables:

    1. Every rank min-merges the per-rank caches already on disk (`tunableop_<arch>_<r>.csv`,
       a previous boot's) and hands the result to its child as the starting table, so a warm
       cache searches nothing.
    2. Each child tunes whatever is missing, in parallel on the four devices, and writes
       `tunableop_<arch>_<r>.tune.csv`.
    3. Barrier; every rank min-merges the four `.tune.csv` tables, writes the result as its
       own `tunableop_<arch>_<r>.csv` (identical on every rank) and loads it with the search
       off, so this process never tunes and every rank runs the same solution per shape.
    """
    tunable = torch.cuda.tunable
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    root.mkdir(parents=True, exist_ok=True)
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    ordinal = torch.cuda.current_device()
    own = root / f"tunableop_{arch}_{ordinal}.csv"
    staged = root / f"tunableop_{arch}_{ordinal}.tune.csv"
    started = time.perf_counter()

    validators: list[list[str]] = []
    prior = []
    for path in _rank_files(root, arch):
        v, t = read_table(path)
        validators = validators or v
        prior.append(t)
    if validators:
        write_table(staged, validators, merge_tables(prior))
    else:
        staged.unlink(missing_ok=True)  # cold cache: the child starts empty

    widths = tuple(batches) if batches is not None else tuned_batches(model.max_batch)
    gemms = [(list(w.shape), list(w.stride()), str(w.dtype)) for w in decode_gemms(model)]
    if getattr(model, "vocab_tp", False):
        shape = [model.lm_head.shape[0] * model.tp.world, model.lm_head.shape[1]]
        gemms.append((shape, [shape[1], 1], str(model.lm_head.dtype)))
    specs = [
        {"m": m, "shape": shape, "stride": stride, "dtype": dtype}
        for m in widths
        for shape, stride, dtype in gemms
    ]
    spec_file = root / f"tune_spec_{ordinal}.json"
    spec_file.write_text(json.dumps({"device": ordinal, "max_ms": max_tuning_ms, "gemms": specs}))
    # A candidate solution can hang the GPU (seen on MI300A at new widths: one child died
    # with "HW Exception ... GPU Hang", another never returned), so the child is bounded and
    # a failed or timed-out child only loses this rank's new picks: the peers' fill in.
    try:
        child = subprocess.run(  # noqa: S603 -- this file, same interpreter
            [sys.executable, str(Path(__file__).resolve()), "--child", str(spec_file), str(staged)],
            check=False,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
        if child.returncode != 0:
            print(
                f"blas_tune: tuning child failed (rc {child.returncode}): {child.stderr[-2000:]}",
                flush=True,
            )
    except subprocess.TimeoutExpired:
        print(f"blas_tune: tuning child timed out after {CHILD_TIMEOUT_S}s", flush=True)

    _barrier(model)
    tables = []
    for path in sorted(root.glob(f"tunableop_{arch}_*.tune.csv")):
        v, t = read_table(path)
        validators = validators or v
        tables.append(t)
    merged = merge_tables(tables)
    if getattr(model, "vocab_tp", False):
        merged = pin_vocab_shards(merged, model)
    write_table(own, validators, merged)
    _barrier(model)  # every rank has read every `.tune.csv` before any is rewritten again

    tunable.set_filename(str(own), False)
    tunable.enable(True)
    tunable.tuning_enable(False)
    tunable.read_file(str(own))
    mine = merge_tables([read_table(staged)[1]])
    differ = sum(1 for k, v in mine.items() if merged.get(k, v)[0] != v[0])
    print(
        f"blas_tune: merged {len(merged)} shapes from {len(tables)} ranks "
        f"({differ} differ from this rank's own search) over batches {widths} "
        f"in {time.perf_counter() - started:.1f}s, cached at {own}",
        flush=True,
    )
    return len(specs)


def _child(spec_file: Path, out: Path) -> None:
    """Tuning child for `_tune_merged`: start from `out`'s table, search every GEMM in the
    spec that is not in it, write the table back to `out`."""
    spec = json.loads(spec_file.read_text())
    torch.cuda.set_device(spec["device"])
    tunable = torch.cuda.tunable
    tunable.set_filename(str(out), False)
    tunable.enable(True)
    tunable.set_max_tuning_duration(spec["max_ms"])
    if out.exists():
        tunable.read_file(str(out))
    tunable.tuning_enable(True)
    dev = torch.device("cuda", spec["device"])
    for g in spec["gemms"]:
        dtype = getattr(torch, g["dtype"].removeprefix("torch."))
        shape, stride = g["shape"], g["stride"]
        extent = 1 + sum((n - 1) * st for n, st in zip(shape, stride, strict=True))
        w = torch.randn(extent, dtype=dtype, device=dev).as_strided(shape, stride)
        x = torch.randn(g["m"], shape[1], dtype=dtype, device=dev)
        F.linear(x, w)
    torch.cuda.synchronize()
    tunable.tuning_enable(False)
    tunable.write_file(str(out))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--child":
    _child(Path(sys.argv[2]), Path(sys.argv[3]))
