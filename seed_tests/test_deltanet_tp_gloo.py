"""The DeltaNet tensor-parallel layer over real `torch.distributed` collectives (gloo).

`test_deltanet_tp.py` hand-shards in one process and combines the ranks' partial outputs
with `torch.stack(...).sum(0)`. That checks the math but never touches the distributed
API, so it would not catch a group that is never formed, a collective on a
non-contiguous or wrong-dtype tensor, a rank that reduces a different number of times
than its peers (which deadlocks), or a layout that only reassembles because the test
did the reassembling. This module runs the same model in `world_size` real processes on
the gloo backend and drives the DeltaNet layer through `Model.layer` /
`Model.decode_layer`, where every `all_reduce` of a forward pass is issued, so the
collectives are real ones: the DeltaNet `out_proj` partial and the expert-parallel MoE
combine both cross the wire. gloo is the CPU stand-in for RCCL: same `torch.distributed`
call path, no GPU needed.

Layout: this file is both the test and the worker. The test launches
`python test_deltanet_tp_gloo.py --rank R --world W ...` as subprocesses, which keeps
the workers free of pytest and of any pickling of closures. `--rank -1` is the
single-process reference run that produces the tensors the ranks compare against, run
the same way so that thread counts and kernels match.

    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_deltanet_tp_gloo.py \
        -p no:cacheprovider --no-cov
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
import model as seed_model  # noqa: E402
import tp as tp_mod  # noqa: E402

WORLDS = [2, 4]
LAYER = 0  # LAYER_TYPES[0] is a linear_attention layer
PREFILL, DECODE_STEPS, SLOTS = 70, 3, [0, 1]
ATOL, RTOL = 1e-5, 1e-5
TIMEOUT_S = 300


def layer_outputs(m: seed_model.Model) -> dict[str, torch.Tensor]:
    """Prefill both slots, then run batched decode steps, through the real collective sites.

    `Model.layer` and `Model.decode_layer` are where every all_reduce of a forward pass is
    issued (see `tp.py`), so driving the DeltaNet layer through them is what puts real
    collectives on the wire: one for the DeltaNet `out_proj` partial and one for the
    expert-parallel MoE combine, per call. `decode_layer` reduces once for the whole batch,
    not once per slot, which is the property the batched path has to keep.
    """
    out = {}
    for slot in SLOTS:
        m.begin(slot)
        out[f"prefill{slot}"] = m.layer(LAYER, fx.hidden_states(PREFILL, seed=slot), 0)
    for step in range(DECODE_STEPS):
        x = fx.hidden_states(1, seed=200 + step, batch=len(SLOTS))
        positions = [PREFILL + step] * len(SLOTS)
        out[f"decode{step}"] = m.decode_layer(LAYER, x, SLOTS, positions)
    return out


def deltanet_partial(m: seed_model.Model) -> torch.Tensor:
    """This rank's un-reduced DeltaNet output for a fixed input, on a freshly reset slot.

    The reference run stores the whole output here (its world is 1). A rank of a real group
    stores a quarter of it, and the test asserts the two differ: without that, a run where
    every collective silently moved nothing would still match the reference.
    """
    m.begin(SLOTS[0])
    x = fx.hidden_states(PREFILL, seed=SLOTS[0])
    h = seed_model.rmsnorm(x, m.layers[LAYER]["in_norm"], m.cfg.eps)
    return m.deltanet(LAYER, h).clone()


def build(ckpt: Path, tp: tp_mod.TP) -> seed_model.Model:
    return seed_model.Model(ckpt, ["cpu"], torch.float32, max_seq=128, max_batch=2, tp=tp)


def run_reference(args: argparse.Namespace) -> None:
    model = build(args.ckpt, tp_mod.TP.single(seed_model.load_cfg(args.ckpt), "cpu"))
    out = layer_outputs(model)
    out["deltanet_partial"] = deltanet_partial(model)
    torch.save(out, args.ref)


def run_rank(args: argparse.Namespace) -> None:
    """One tensor-parallel rank: form the group, run the layer, check against the reference."""
    dist.init_process_group(
        backend="gloo", init_method=f"file://{args.store}", rank=args.rank, world_size=args.world
    )
    try:
        tp = tp_mod.TP.from_torch_distributed(seed_model.load_cfg(args.ckpt), "cpu")
        if (tp.rank, tp.world_size) != (args.rank, args.world):
            raise AssertionError(f"group reports rank {tp.rank}/{tp.world_size}")
        model = build(args.ckpt, tp)

        # This rank's un-reduced DeltaNet output. If it matched the reference's, the ranks
        # would agree below whether or not any collective moved anything.
        partial = deltanet_partial(model)

        got = layer_outputs(model)
        want = torch.load(args.ref, weights_only=True)
        reference_partial = want.pop("deltanet_partial")
        errors = {}
        for name, ref in want.items():
            mine = got[name]
            if mine.shape != ref.shape:
                raise AssertionError(f"{name}: {mine.shape} != reference {ref.shape}")
            # Relative to the reference's own scale: these are whole-layer outputs, so they
            # carry the residual stream and are an order of magnitude larger than the
            # DeltaNet partial that produced the error.
            errors[name] = (mine - ref).abs().max().item() / ref.abs().max().clamp(min=1.0).item()
            if not torch.allclose(mine, ref, atol=ATOL, rtol=RTOL):
                raise AssertionError(f"{name}: relative error {errors[name]:.3e}")
        if torch.allclose(partial, reference_partial, atol=ATOL, rtol=RTOL):
            raise AssertionError("this rank's un-reduced DeltaNet output is already the whole one")

        # Every rank must issue the same number of collectives; a rank that reduced a
        # different number of times would have deadlocked above rather than reach this.
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
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_all(procs: list[subprocess.Popen]) -> list[str]:
    """Collect every worker's output, killing the rest if one fails or hangs.

    A rank that dies leaves its peers blocked in `all_reduce` forever, so the timeout
    and the kill are what turn a deadlock into a test failure instead of a hung suite.
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
    return fx.write_checkpoint(tmp_path_factory.mktemp("deltanet-tp-gloo-ckpt"))


@pytest.fixture(scope="module")
def reference(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    ref = tmp_path_factory.mktemp("deltanet-tp-gloo-ref") / "reference.pt"
    proc = spawn("--rank", "-1", "--world", "1", "--ckpt", str(ckpt), "--ref", str(ref))
    log = wait_all([proc])[0]
    assert proc.returncode == 0, log
    return ref


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not built into torch")
@pytest.mark.parametrize("world", WORLDS)
def test_real_gloo_collectives_match_the_single_process_layer(
    ckpt: Path, reference: Path, tmp_path: Path, world: int
) -> None:
    """`world` processes, real `all_reduce`, same layer outputs as one unsharded process."""
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
    # Every rank all_reduces the same buffer, so they must agree with each other exactly.
    assert len({json.dumps(r["errors"], sort_keys=True) for r in reported}) == 1
    assert max(max(r["errors"].values()) for r in reported) < RTOL


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
