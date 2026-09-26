"""End-to-end: `SEED_FUSE_GLUE`/`SEED_FUSED_AR_NORM` through the real captured-decode harness.

The per-kernel tests (`test_attn_decode_fused.py`, `test_router_fused.py`,
`test_deltanet_fused.py`'s `active`/`masked_row_copy` cases) check each fused kernel against
its own torch oracle in isolation. This file drives the *whole* captured decode step
(`graph_decode.GraphDecodeRunner`, the same harness `test_graph_capture.py` uses) with both
flags on, and leans on that harness's own built-in check: `GraphDecodeRunner.validate` runs
one synthetic full-batch step both ways -- eager `Model.decode` and the replay -- and disables
the captured path if they disagree by more than `VALIDATE_REL_TOL`. `graph_runner` below
asserts `prepare()` succeeded, so a wiring bug in either flag (wrong argument order, a masked
write that touches the wrong row, a norm fused into the wrong layer's weight) fails the test
here rather than silently serving from the eager fallback.

`available()` on every new fused module gates on `device.type == "cuda"` (the same contract
`deltanet_fused`/`rmsnorm_fused` already have), so on CPU it has to be forced to exercise the
kernels themselves through Triton's interpreter -- exactly what `test_deltanet_fused.py`'s
`test_decode_matches_the_torch_path` already does for the pre-existing DeltaNet kernels; this
file does the same for the three new modules together.

    TRITON_INTERPRET=1 python -m pytest seed_tests/test_fuse_glue.py -q -o addopts=
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attn_decode_fused  # noqa: E402
import deltanet_fused  # noqa: E402
import graph_decode  # noqa: E402
import model as seed_model  # noqa: E402
import router_fused  # noqa: E402
from test_batched_decode import prompt_of  # noqa: E402
from test_graph_capture import EagerBackend  # noqa: E402
from test_graph_capture import graph_runner as _graph_runner  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

pytestmark = pytest.mark.skipif(
    not INTERPRET and not torch.cuda.is_available(),
    reason="needs TRITON_INTERPRET=1 (no GPU this round; see COMMON_BRIEF.md)",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-fuse-glue")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture
def fused_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every new fused kernel's `available()` to True, and both flags on.

    Module-level constants (`graph_decode.FUSE_GLUE`, `model.FUSE_GLUE`) are frozen at import
    time, so the env var alone would not reach code that already ran `... = os.environ.get(...)`
    at module load; they are patched directly instead. `SEED_FUSED_AR_NORM` is read fresh
    inside `segment_step`/`TP.all_reduce_residual_norm` on every call, so the env var alone is
    enough for that one.
    """
    monkeypatch.setattr(graph_decode, "FUSE_GLUE", True)
    monkeypatch.setattr(seed_model, "FUSE_GLUE", True)
    monkeypatch.setattr(attn_decode_fused, "available", lambda _dev: True)
    monkeypatch.setattr(deltanet_fused, "available", lambda _dev: True)
    monkeypatch.setattr(router_fused, "available", lambda _dev: True)
    monkeypatch.setenv("SEED_FUSED_AR_NORM", "1")
    monkeypatch.setenv("SEED_FUSE_GLUE", "1")


def graph_runner(checkpoint: Path, prompts: list[list[int]], **kw):  # noqa: ANN001
    return _graph_runner(checkpoint, prompts, backend=EagerBackend(), **kw)


def test_capture_validates_with_every_fusion_on(checkpoint: Path, fused_everywhere: None) -> None:
    """`graph_runner`'s own `assert runner.prepare()` is the check: it fails if the fused
    replay disagrees with eager `Model.decode` by more than `VALIDATE_REL_TOL`."""
    prompts = [prompt_of(1, 6), prompt_of(2, 7), prompt_of(3, 5)]
    runner = graph_runner(checkpoint, prompts)
    assert runner.enabled


def test_decode_output_matches_every_fusion_off(checkpoint: Path, fused_everywhere: None) -> None:
    """Same prompts, same steps, fusions on vs. off: the *served* tokens must not move."""
    prompts = [prompt_of(10, 6), prompt_of(11, 7)]
    fused = graph_runner(checkpoint, prompts)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(graph_decode, "FUSE_GLUE", False)
        mp.setattr(seed_model, "FUSE_GLUE", False)
        mp.setenv("SEED_FUSED_AR_NORM", "0")
        mp.setenv("SEED_FUSE_GLUE", "0")
        plain = graph_runner(checkpoint, prompts)

    batch, tokens = [0, 1], [prompts[0][-1], prompts[1][-1]]
    positions = [len(prompts[0]), len(prompts[1])]
    want = plain.decode(batch, tokens, positions)
    got = fused.decode(batch, tokens, positions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_padding_rows_leave_state_alone_with_every_fusion_on(
    checkpoint: Path, fused_everywhere: None
) -> None:
    """The same property `test_graph_capture.py::test_padding_rows_leave_their_slots_state_alone`
    checks, now with `SEED_FUSE_GLUE=1`'s masked-store kernels actually doing the masking."""
    prompts = [prompt_of(50, 5), prompt_of(51, 6), prompt_of(52, 5)]
    runner = graph_runner(checkpoint, prompts)
    before = [t[2].clone() for pool in runner.model.pool for t in pool.values()]

    runner.decode([0, 1], [prompts[0][-1], prompts[1][-1]], [len(prompts[0]), len(prompts[1])])

    after = [t[2].clone() for pool in runner.model.pool for t in pool.values()]
    for old, new in zip(before, after, strict=True):
        assert torch.equal(old, new), "an inactive row wrote into its slot's state"
