"""CPU-only coverage for `allreduce_custom.py`'s host-side logic (no GPU, no HIP compile).

`test_custom_allreduce.py` checks the numerics on real hardware and is skipped without CUDA.
This file covers the surrounding host-side machinery that does not need a GPU at all, and that
a GPU-only test would not exercise well even on hardware (a build failure, a spin-limit trip,
and a build-directory race are all rare, hard-to-provoke events on real hardware but trivial to
force here). See individual test docstrings for the bug each one is a regression test for.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_custom_allreduce_cpu.py \\
        -p no:cacheprovider --no-cov
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import allreduce_custom  # noqa: E402

TIMEOUT_S = 120

# ---------------------------------------------------------------- build-dir uniqueness


def test_build_dir_is_unique_per_call() -> None:
    """Two ranks' build dirs never collide, and neither do two calls for the same rank.

    Regression for the bug: every rank wrote `allreduce_ext.cpp`/`.hip` into one shared
    `os.path.join(tempfile.gettempdir(), "custom_allreduce_src")`, so concurrent ranks raced to
    create, write, and compile the same two files.
    """
    dirs = [allreduce_custom._build_dir(rank) for rank in (0, 1, 2, 3)]
    dirs.append(allreduce_custom._build_dir(0))  # a second call for rank 0: still unique
    try:
        assert len(set(dirs)) == len(dirs), f"build dirs collided: {dirs}"
        for d in dirs:
            assert Path(d).is_dir()
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- the vote


def run_vote_worker(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--fail-rank", type=int, required=True, help="-1 means nobody fails")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    dist.init_process_group(
        backend="gloo", init_method=f"file://{args.store}", rank=args.rank, world_size=args.world
    )
    try:
        local_ok = args.rank != args.fail_rank
        local_error = RuntimeError("simulated build failure") if not local_ok else None
        raised = None
        try:
            allreduce_custom._vote_or_raise(local_ok, local_error, torch.device("cpu"), "test")
        except RuntimeError as exc:
            raised = str(exc)
        # A rank that issued a different number of collectives than its peers -- e.g. one that
        # raised straight out of a bare try and never reached `_vote_or_raise` at all -- would
        # deadlock here instead of reaching this barrier.
        dist.barrier()
        args.out.write_text(json.dumps({"rank": args.rank, "raised": raised}))
    finally:
        dist.destroy_process_group()


def spawn(*argv: str) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), "vote-worker", *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_all(procs: list[subprocess.Popen]) -> list[str]:
    """Collect every worker's output, killing the rest if one hangs.

    A rank stuck on a collective its peers never issue is exactly the failure mode `_vote_or_
    raise` exists to prevent; the timeout+kill here turns a regression into a test failure
    instead of a hung test suite.
    """
    logs: list[str] = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=TIMEOUT_S)[0])
    except subprocess.TimeoutExpired:
        logs.append("worker timed out (likely a hang on a mismatched collective)")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return logs


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not built into torch")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("fail_rank", [-1, 0])
def test_vote_falls_back_together_on_one_ranks_build_failure(
    tmp_path: Path, world: int, fail_rank: int
) -> None:
    """One rank's local build failure makes every rank raise; no failure means none do.

    `fail_rank=-1` is the control: every rank's local build "succeeds," so `_vote_or_raise`
    must not raise on any of them. `fail_rank=0` simulates rank 0's extension build or IPC
    allocation failing before it reaches `_vote_or_raise` -- exactly the case that used to
    leave ranks 1.. blocked on a collective rank 0 never issued (or racing ahead with a mixed
    RCCL/custom state) because nothing forced the ranks to agree before continuing.
    """
    store = tmp_path / f"store{world}_{fail_rank}"
    outs = [tmp_path / f"rank{r}.json" for r in range(world)]
    procs = [
        spawn(
            *("--rank", str(r), "--world", str(world)),
            *("--fail-rank", str(fail_rank)),
            *("--store", str(store), "--out", str(outs[r])),
        )
        for r in range(world)
    ]
    logs = wait_all(procs)
    for r, (proc, log) in enumerate(zip(procs, logs, strict=False)):
        assert proc.returncode == 0, f"rank {r} of {world} (fail_rank={fail_rank}) failed:\n{log}"

    reported = [json.loads(path.read_text()) for path in outs]
    assert sorted(r["rank"] for r in reported) == list(range(world))
    if fail_rank == -1:
        assert all(r["raised"] is None for r in reported), reported
    else:
        # Every rank -- including the one that did not fail locally -- falls back together.
        assert all(r["raised"] is not None for r in reported), reported


# ---------------------------------------------------------------- the spin-limit error flag


def test_check_error_flag_raises_and_clears() -> None:
    """A tripped watchdog flag raises loudly once, then goes quiet until it trips again.

    Regression for the bug: `SPIN_TIMEOUT_S` expiry inside `gather_kernel` used to `break` and
    silently reduce whatever garbage sat in the peer's still-unset slot, with no signal the
    host could check. The flag stands in for what a real kernel run would set via
    `atomicExch(error_flag, 1)` on that path.
    """
    flag = torch.zeros(1, dtype=torch.int32)
    allreduce_custom._check_error_flag(flag, "test")  # not tripped: quiet

    flag.fill_(1)  # simulates the kernel's atomicExch(error_flag, 1) on a SPIN_TIMEOUT_S trip
    with pytest.raises(RuntimeError, match="SPIN_TIMEOUT_S"):
        allreduce_custom._check_error_flag(flag, "test")
    assert flag.item() == 0, "the flag must clear on read, or every later check re-raises"
    allreduce_custom._check_error_flag(flag, "test")  # quiet again


def test_custom_all_reduce_check_errors_reads_its_own_flag() -> None:
    """`CustomAllReduce.check_errors` is wired to `self._error_flag`, not a fresh one."""
    car = object.__new__(allreduce_custom.CustomAllReduce)
    car._error_flag = torch.zeros(1, dtype=torch.int32)
    car.check_errors()  # quiet

    car._error_flag.fill_(1)
    with pytest.raises(RuntimeError, match="CustomAllReduce"):
        car.check_errors()


# ---------------------------------------------------------------- the non-contiguous write path


def _make_car(max_elems: int) -> allreduce_custom.CustomAllReduce:
    """A `CustomAllReduce` with `__init__` skipped: no HIP compile, no process group, no CUDA.

    Only the attributes `__call__` reads are set; the extension is a mock that records its
    arguments and (for `gather`) writes a fixed pattern at whatever pointer it is given, which
    is enough to observe where the "kernel" wrote without a real one.
    """
    car = object.__new__(allreduce_custom.CustomAllReduce)
    car.rank, car.world, car.device, car.max_elems = 0, 4, torch.device("cpu"), max_elems
    car._pos = 0
    car._my_buf = car._my_flag = 0
    car._buf_ptr_table = car._flag_ptr_table = 0
    car._error_flag = torch.zeros(1, dtype=torch.int32)
    return car


def test_gather_writes_to_the_contiguous_buffer_not_to_x() -> None:
    """`gather` is called with `xc.data_ptr()`, and a non-contiguous `x` is copied back from it.

    Regression for the bug: `__call__` read from `xc` (the contiguous copy) for `publish` but
    passed `x.data_ptr()` to `gather`, so a non-contiguous `x` had `n_elems` contiguous bf16
    values written starting at its base pointer -- correct only when `x` happens to already be
    contiguous. Here `x` is a strided view (every other element of a wider buffer): the mock
    "kernel" fills the pointer it receives with a known pattern, and the assertions check that
    pattern lands in `x`'s strided positions (via `x.copy_(xc)`) and nowhere else, which only
    happens if the write actually went to a genuinely contiguous scratch buffer first.
    """
    n = 6
    storage = torch.arange(2 * n, dtype=torch.float32).to(torch.bfloat16).clone()
    x = storage[0::2]  # n elements, stride 2: non-contiguous
    assert not x.is_contiguous()
    other = storage[1::2].clone()  # the interleaved elements `x` must not disturb

    pattern = torch.full((n,), 9.0, dtype=torch.bfloat16)
    captured_out_ptr = {}

    def fake_publish(*_args: object) -> None:
        return None

    def fake_gather(buf_ptrs, flag_ptrs, out_ptr, error_flag_ptr, n_elems, *_rest) -> None:  # noqa: ANN001
        captured_out_ptr["ptr"] = out_ptr
        dst = torch.empty(n_elems, dtype=torch.bfloat16)
        dst.copy_(pattern[:n_elems])
        import ctypes

        ctypes.memmove(out_ptr, dst.data_ptr(), n_elems * dst.element_size())

    car = _make_car(max_elems=64)
    car._ext = SimpleNamespace(publish=fake_publish, gather=fake_gather)

    with mock.patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=0)):
        out = car(x)

    assert out is x
    xc_ptr_used = captured_out_ptr["ptr"]
    assert xc_ptr_used != x.data_ptr(), "gather must not write to x's own (strided) memory"
    assert torch.equal(x, pattern), "x must be updated from the contiguous scratch buffer"
    assert torch.equal(storage[1::2], other), "the interleaved elements x does not own must survive"


def test_gather_writes_directly_to_x_when_already_contiguous() -> None:
    """When `x` is already contiguous, `xc is x`, so `gather` targets `x.data_ptr()` directly."""
    n = 6
    x = torch.zeros(n, dtype=torch.bfloat16)
    assert x.is_contiguous()

    def fake_publish(*_args: object) -> None:
        return None

    def fake_gather(buf_ptrs, flag_ptrs, out_ptr, error_flag_ptr, n_elems, *_rest) -> None:  # noqa: ANN001
        assert out_ptr == x.data_ptr()

    car = _make_car(max_elems=64)
    car._ext = SimpleNamespace(publish=fake_publish, gather=fake_gather)

    with mock.patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=0)):
        car(x)


if __name__ == "__main__":
    assert sys.argv[1] == "vote-worker"
    run_vote_worker(sys.argv[2:])
