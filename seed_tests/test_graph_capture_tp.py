"""The captured decode step under real tensor parallelism, over gloo collectives.

`test_graph_capture.py` covers the runner at `--tp 1`, where `graph_decode`'s static step
has no collectives in it and every head count it reads happens to be the global one. Neither
holds under TP, and both are silent: reading `Model.cfg.heads` instead of this rank's
`tp.plan.q.count` reshapes a sharded projection, and a static step missing
`Model.decode_layer`'s two all-reduces returns a partial sum that still has the right shape.
So this runs the runner in `world` real processes, drives it through `GraphDecodeRunner`, and
compares against the same model decoded unsharded in one process.

The capture backend is `EagerBackend`: it returns the static step itself rather than a
replay, which is the equivalence a correct capture has to satisfy (both read every input from
the static buffers and write every output to them). What a GPU adds on top is whether HIP
will record those launches, and that is not something gloo on CPU can answer. It is measured
separately on hardware; see `graph_decode.py`'s module docstring.

Layout follows `test_deltanet_tp_gloo.py`: this file is both the test and the worker, so the
worker processes carry no pytest and no pickled closures.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deltanet_tp_fixture as fx  # noqa: E402
import graph_decode  # noqa: E402
import model as seed_model  # noqa: E402
import tp as tp_mod  # noqa: E402

WORLDS = [2, 4]
PROMPT, DECODE_STEPS = 12, 3
MAX_BATCH, MAX_SEQ = 2, 64
ATOL = RTOL = 1e-5
TIMEOUT_S = 300


class EagerBackend:
    """A `CaptureBackend` that "captures" by handing back the step.

    A replay and this must be interchangeable, because both read every input from the static
    buffers and write their output into `Buffers.out`. Anything the static step got wrong
    about sharding shows up here, without needing a device that can capture.
    """

    def capture(self, step, device):  # noqa: ANN001, ANN202 -- matches CaptureBackend
        return step


def build(ckpt: Path, tp: tp_mod.TP) -> seed_model.Model:
    return seed_model.Model(
        ckpt, [tp.device], torch.float32, max_seq=MAX_SEQ, max_batch=MAX_BATCH, tp=tp
    )


def prompts() -> list[list[int]]:
    return [[(5 * s + i) % fx.VOCAB for i in range(PROMPT)] for s in range(MAX_BATCH)]


def decode_logits(runner, model: seed_model.Model) -> list[torch.Tensor]:
    """Prefill every slot, then take `DECODE_STEPS` full-batch decode steps through `runner`.

    Full batch on purpose: this exercises the `max_batch`-sized bucket specifically (the
    only bucket with no padding rows at all), so a narrower batch would also cover
    `pad_slot_for`'s padding-row path but not add anything sharding-specific; the padding
    logic itself is covered CPU-side, without TP, in `test_graph_capture.py`.
    """
    slots = list(range(MAX_BATCH))
    for slot, ids in zip(slots, prompts(), strict=True):
        model.begin(slot)
        model.prefill(slot, ids, 0)
    out = []
    for step in range(DECODE_STEPS):
        tokens = [(11 * s + step) % fx.VOCAB for s in slots]
        positions = [PROMPT + step] * MAX_BATCH
        out.append(runner.decode(slots, tokens, positions).clone())
    return out


def run_reference(args: argparse.Namespace) -> None:
    """The unsharded answer: one process, no collectives, the model's own eager decode."""
    model = build(args.ckpt, tp_mod.TP.single(seed_model.load_cfg(args.ckpt), "cpu"))
    torch.save(decode_logits(model, model), args.ref)


def run_rank(args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"file://{args.store}", rank=args.rank, world_size=args.world
    )
    try:
        tp = tp_mod.TP.from_torch_distributed(seed_model.load_cfg(args.ckpt), "cpu")
        model = build(args.ckpt, tp)
        runner = graph_decode.GraphDecodeRunner(model, backend=EagerBackend())
        if not runner.prepare():
            raise AssertionError(f"rank {args.rank}: prepare() did not enable the captured path")

        got = decode_logits(runner, model)
        want = torch.load(args.ref, weights_only=True)
        errors = []
        for step, (mine, ref) in enumerate(zip(got, want, strict=True)):
            if mine.shape != ref.shape:
                raise AssertionError(f"step {step}: {mine.shape} != reference {ref.shape}")
            errors.append((mine - ref).abs().max().item() / ref.abs().max().clamp(min=1.0).item())
            if not torch.allclose(mine, ref, atol=ATOL, rtol=RTOL):
                raise AssertionError(f"step {step}: relative error {errors[-1]:.3e}")

        # A rank that issued a different number of collectives than its peers would have
        # deadlocked above rather than reach this.
        dist.barrier()
        args.out.write_text(json.dumps({"rank": args.rank, "errors": errors}))
    finally:
        dist.destroy_process_group()


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)  # -1 runs the reference
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--store", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(1)  # same kernels in the reference and in every rank
    run_reference(args) if args.rank < 0 else run_rank(args)
    return 0


# ---------------------------------------------------------------- the test


def spawn(*argv: str) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_all(procs: list[subprocess.Popen]) -> list[str]:
    """Collect every worker's output, killing the rest if one fails or hangs.

    A rank that dies leaves its peers blocked in `all_reduce` forever, so the timeout and the
    kill are what turn a deadlock into a test failure instead of a hung suite.
    """
    logs: list[str] = []
    try:
        for proc in procs:
            logs.append(proc.communicate(timeout=TIMEOUT_S)[0])
    except subprocess.TimeoutExpired:
        logs.append("worker timed out")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return logs


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fx.write_checkpoint(tmp_path_factory.mktemp("graph-tp-ckpt"))


@pytest.fixture(scope="module")
def reference(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    ref = tmp_path_factory.mktemp("graph-tp-ref") / "reference.pt"
    proc = spawn("--rank", "-1", "--world", "1", "--ckpt", str(ckpt), "--ref", str(ref))
    log = wait_all([proc])[0]
    assert proc.returncode == 0, log
    return ref


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not built into torch")
@pytest.mark.parametrize("world", WORLDS)
def test_the_static_step_under_tp_matches_the_unsharded_decode(
    ckpt: Path, reference: Path, tmp_path: Path, world: int
) -> None:
    """`world` ranks driving `GraphDecodeRunner` agree with one unsharded eager process.

    This is the test that fails if `attn_decode_static` reads `Model.cfg`'s global head
    counts (a reshape error at TP=4, a wrong reading wherever it divides) or if
    `decode_layer_static` drops either of `Model.decode_layer`'s two all-reduces (a partial
    sum of the right shape).
    """
    store = tmp_path / f"store{world}"  # file:// rendezvous: no port to pick
    outs = [tmp_path / f"rank{r}.json" for r in range(world)]
    procs = [
        spawn(
            *("--rank", str(r), "--world", str(world)),
            *("--ckpt", str(ckpt), "--ref", str(reference)),
            *("--store", str(store), "--out", str(outs[r])),
        )
        for r in range(world)
    ]
    logs = wait_all(procs)
    for r, (proc, log) in enumerate(zip(procs, logs, strict=False)):
        assert proc.returncode == 0, f"rank {r} of {world} failed:\n{log}"

    reported = [json.loads(path.read_text()) for path in outs]
    assert sorted(r["rank"] for r in reported) == list(range(world))
    # Every rank all-reduces the same buffer, so they must agree with each other exactly.
    assert len({json.dumps(r["errors"], sort_keys=True) for r in reported}) == 1
    assert max(max(r["errors"]) for r in reported) < RTOL


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
