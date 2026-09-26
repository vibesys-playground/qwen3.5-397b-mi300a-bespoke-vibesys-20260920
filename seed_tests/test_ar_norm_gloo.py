"""`SEED_FUSED_AR_NORM`'s plumbing over real `torch.distributed` collectives (gloo).

`tp.TP.all_reduce_residual_norm` (see its docstring) has a fused GPU kernel path
(`allreduce_custom.CustomAllReduce.residual_norm`, only reachable with CUDA) and a fallback
path that is exactly the sequence it replaces: `x = residual + self.all_reduce(mixer);
return x, rmsnorm_fn(x, norm_w, eps)`. No GPU is available for this campaign round (see
`COMMON_BRIEF.md`), so the fused *kernel*'s numerics cannot be checked here -- that is
`test_custom_allreduce.py`'s `test_residual_norm_eager_and_graph`, GPU-only, for the next
cluster round. What CAN be checked without a GPU, and is the point of this file: that
`Model.decode_layer`'s new call to `all_reduce_residual_norm` (in place of the old inline
`x = x + all_reduce(mixer); moe_out = self.moe(i, rmsnorm(x, ...), ...)`) is a pure
refactor -- byte-identical output, at any `SEED_FUSED_AR_NORM` setting, since gloo/CPU always
takes the fallback branch (`TP.__init__` only builds `custom_reduce` when
`torch.cuda.is_available()`) -- and that it does not change how many collectives each rank
issues or in what order, over a *real* process group with `world` in `{2, 4}`, both layer
types (`deltanet_tp_fixture`'s tiny checkpoint has one `full_attention` layer and two
`linear_attention` ones), and multiple decode steps.

Layout mirrors `test_deltanet_tp_gloo.py` exactly: this file is both the test and the worker,
spawned as subprocesses so gloo's rendezvous and collectives are the real thing, not a
same-process stand-in.

    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_ar_norm_gloo.py \
        -p no:cacheprovider --no-cov
"""

import argparse
import json
import os
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
PREFILL, DECODE_STEPS, SLOTS = 12, 4, [0, 1]
# Looser than `test_deltanet_tp_gloo.py`'s 1e-5: that test compares one layer's output: this
# one chains through every layer (full_attention and linear_attention both), so fp32
# reduction-order noise between the world=1 reference's single accumulation and each world>1
# rank's partial-sum-then-all-reduce compounds layer over layer. `test_seed_parity.py`'s own
# whole-model, multi-step comparisons use this same 1e-4, which is the right precedent here.
ATOL, RTOL = 1e-4, 1e-4
TIMEOUT_S = 300


def layer_outputs(m: seed_model.Model) -> dict[str, torch.Tensor]:
    """Prefill both slots on every layer, then batched decode steps through `decode_layer`.

    Every layer, not one: `deltanet_tp_fixture`'s checkpoint has `full_attention` at index 1
    and `linear_attention` at 0 and 2 (see its `LAYER_TYPES`), and `all_reduce_residual_norm`
    is reached by both mixer kinds identically (the fusion is after the mixer's own
    all-reduce, not inside it).
    """
    out = {}
    for slot in SLOTS:
        m.begin(slot)
        x = fx.hidden_states(PREFILL, seed=slot)
        for i in range(len(m.cfg.layer_types)):
            x = m.layer(i, x, 0)
        out[f"prefill{slot}"] = x
    for step in range(DECODE_STEPS):
        x = fx.hidden_states(1, seed=300 + step, batch=len(SLOTS))
        positions = [PREFILL + step] * len(SLOTS)
        for i in range(len(m.cfg.layer_types)):
            x = m.decode_layer(i, x, SLOTS, positions)
        out[f"decode{step}"] = x
    return out


def build(ckpt: Path, tp: tp_mod.TP) -> seed_model.Model:
    return seed_model.Model(ckpt, ["cpu"], torch.float32, max_seq=64, max_batch=2, tp=tp)


def run_reference(args: argparse.Namespace) -> None:
    model = build(args.ckpt, tp_mod.TP.single(seed_model.load_cfg(args.ckpt), "cpu"))
    torch.save(layer_outputs(model), args.ref)


def run_rank(args: argparse.Namespace) -> None:
    os.environ["SEED_FUSED_AR_NORM"] = args.fused_ar_norm
    dist.init_process_group(
        backend="gloo", init_method=f"file://{args.store}", rank=args.rank, world_size=args.world
    )
    try:
        tp = tp_mod.TP.from_torch_distributed(seed_model.load_cfg(args.ckpt), "cpu")
        if (tp.rank, tp.world_size) != (args.rank, args.world):
            raise AssertionError(f"group reports rank {tp.rank}/{tp.world_size}")
        assert tp.custom_reduce is None, "gloo/CPU must never build the CUDA-only custom all-reduce"

        model = build(args.ckpt, tp)
        got = layer_outputs(model)
        want = torch.load(args.ref, weights_only=True)
        errors = {}
        for name, ref in want.items():
            mine = got[name]
            if mine.shape != ref.shape:
                raise AssertionError(f"{name}: {mine.shape} != reference {ref.shape}")
            errors[name] = (mine - ref).abs().max().item() / ref.abs().max().clamp(min=1.0).item()
            if not torch.allclose(mine, ref, atol=ATOL, rtol=RTOL):
                raise AssertionError(f"{name}: relative error {errors[name]:.3e}")

        # Every rank must issue the same collectives in the same order regardless of the
        # flag: a rank that took a different branch count would deadlock here.
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
    p.add_argument("--fused-ar-norm", default="0")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
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
    return fx.write_checkpoint(tmp_path_factory.mktemp("ar-norm-gloo-ckpt"))


@pytest.fixture(scope="module")
def reference(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    ref = tmp_path_factory.mktemp("ar-norm-gloo-ref") / "reference.pt"
    proc = spawn("--rank", "-1", "--world", "1", "--ckpt", str(ckpt), "--ref", str(ref))
    log = wait_all([proc])[0]
    assert proc.returncode == 0, log
    return ref


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not built into torch")
@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("fused_ar_norm", ["0", "1"])
def test_decode_layer_matches_single_process_at_any_flag_setting(
    ckpt: Path, reference: Path, tmp_path: Path, world: int, fused_ar_norm: str
) -> None:
    """`world` real gloo ranks, `SEED_FUSED_AR_NORM` on or off: same layer outputs either way.

    On CPU both settings take `all_reduce_residual_norm`'s fallback branch (no CUDA, so
    `custom_reduce` is never built), so this is really "the refactor changed nothing" plus
    "the flag's own plumbing does not crash or desync a real process group" -- the fused
    kernel's own numerics need a GPU, see the module docstring.
    """
    store = tmp_path / f"store{world}-{fused_ar_norm}"
    outs = [tmp_path / f"rank{r}.json" for r in range(world)]
    procs = [
        spawn(
            *("--rank", str(r), "--world", str(world)),
            *("--ckpt", str(ckpt), "--ref", str(reference)),
            *("--store", str(store), "--out", str(outs[r])),
            *("--fused-ar-norm", fused_ar_norm),
        )
        for r in range(world)
    ]
    logs = wait_all(procs)
    for r, (proc, log) in enumerate(zip(procs, logs, strict=False)):
        assert proc.returncode == 0, f"rank {r} of {world} failed:\n{log}"

    reported = [json.loads(path.read_text()) for path in outs]
    assert sorted(r["rank"] for r in reported) == list(range(world))
    assert max(max(r["errors"].values()) for r in reported) < RTOL


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
