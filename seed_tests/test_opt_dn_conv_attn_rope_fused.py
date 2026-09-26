"""End-to-end: `SEED_DN_CONV_INPLACE`/`SEED_ATTN_ROPE_KV_FUSED` through the captured-decode
harness, mirroring `test_fuse_glue.py` for these two independent flags.

The per-kernel tests (`test_deltanet_fused.py`'s `causal_conv_decode` cases,
`test_attn_decode_fused.py`'s `rmsnorm_rope_and_kv_write` cases) check each fused kernel
against its own torch oracle in isolation. This file drives the *whole* captured decode step
(`graph_decode.GraphDecodeRunner`, the same harness `test_graph_capture.py`/`test_fuse_glue.py`
use) with both flags on, and leans on `GraphDecodeRunner.validate`: `prepare()` runs one
synthetic full-batch step both ways (eager `Model.decode` and the replay) and disables the
captured path if they disagree by more than `VALIDATE_REL_TOL`, so a wiring bug in either flag
(wrong lane indexing, a norm applied with the wrong weight, a state write that touches the
wrong row) fails here rather than silently serving from the eager fallback.

    TRITON_INTERPRET=1 python -m pytest seed_tests/test_opt_dn_conv_attn_rope_fused.py -q -o addopts=
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
    out = tmp_path_factory.mktemp("tiny-dn-conv-attn-rope")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


@pytest.fixture
def both_flags_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force both flags on and both kernels' `available()` to True, the same shape
    `test_fuse_glue.py::fused_everywhere` uses: the module constants are frozen at import
    time, so the env var alone does not reach code that already read it.
    """
    monkeypatch.setattr(graph_decode, "DN_CONV_INPLACE", True)
    monkeypatch.setattr(graph_decode, "ATTN_ROPE_KV_FUSED", True)
    monkeypatch.setattr(attn_decode_fused, "available", lambda _dev: True)
    monkeypatch.setattr(deltanet_fused, "available", lambda _dev: True)
    monkeypatch.setenv("SEED_DN_CONV_INPLACE", "1")
    monkeypatch.setenv("SEED_ATTN_ROPE_KV_FUSED", "1")


def graph_runner(checkpoint: Path, prompts: list[list[int]], **kw):  # noqa: ANN001
    return _graph_runner(checkpoint, prompts, backend=EagerBackend(), **kw)


def test_capture_validates_with_both_flags_on(checkpoint: Path, both_flags_on: None) -> None:
    """`graph_runner`'s own `assert runner.prepare()` is the check: it fails if the fused
    replay disagrees with eager `Model.decode` by more than `VALIDATE_REL_TOL`."""
    prompts = [prompt_of(1, 6), prompt_of(2, 7), prompt_of(3, 5)]
    runner = graph_runner(checkpoint, prompts)
    assert runner.enabled


def test_decode_output_matches_with_flags_off(checkpoint: Path, both_flags_on: None) -> None:
    """Same prompts, same steps, flags on vs. off: the *served* tokens must not move."""
    prompts = [prompt_of(10, 6), prompt_of(11, 7)]
    fused = graph_runner(checkpoint, prompts)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(graph_decode, "DN_CONV_INPLACE", False)
        mp.setattr(graph_decode, "ATTN_ROPE_KV_FUSED", False)
        mp.setenv("SEED_DN_CONV_INPLACE", "0")
        mp.setenv("SEED_ATTN_ROPE_KV_FUSED", "0")
        plain = graph_runner(checkpoint, prompts)

    batch, tokens = [0, 1], [prompts[0][-1], prompts[1][-1]]
    positions = [len(prompts[0]), len(prompts[1])]
    want = plain.decode(batch, tokens, positions)
    got = fused.decode(batch, tokens, positions)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_padding_rows_leave_state_alone_with_both_flags_on(
    checkpoint: Path, both_flags_on: None
) -> None:
    """The same property `test_graph_capture.py::test_padding_rows_leave_their_slots_state_alone`
    checks, now with both flags' in-place kernels actually doing the masking: an unfilled slot's
    DeltaNet conv/rec state and attention KV cache must be bit-identical before and after a
    partial-batch replay."""
    prompts = [prompt_of(50, 5), prompt_of(51, 6), prompt_of(52, 5)]
    runner = graph_runner(checkpoint, prompts)
    before = [t[2].clone() for pool in runner.model.pool for t in pool.values()]

    runner.decode([0, 1], [prompts[0][-1], prompts[1][-1]], [len(prompts[0]), len(prompts[1])])

    after = [t[2].clone() for pool in runner.model.pool for t in pool.values()]
    for old, new in zip(before, after, strict=True):
        assert torch.equal(old, new), "an inactive row wrote into its slot's state"


def test_dn_conv_inplace_alone_matches_eager(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SEED_DN_CONV_INPLACE` in isolation (`SEED_ATTN_ROPE_KV_FUSED` off): the DeltaNet
    conv-state gather/scatter removal must not depend on the attention flag."""
    monkeypatch.setattr(graph_decode, "DN_CONV_INPLACE", True)
    monkeypatch.setattr(deltanet_fused, "available", lambda _dev: True)
    monkeypatch.setenv("SEED_DN_CONV_INPLACE", "1")
    prompts = [prompt_of(20, 6), prompt_of(21, 7)]
    runner = graph_runner(checkpoint, prompts)
    assert runner.enabled


def test_attn_rope_kv_fused_alone_matches_eager(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`SEED_ATTN_ROPE_KV_FUSED` in isolation (`SEED_DN_CONV_INPLACE` off)."""
    monkeypatch.setattr(graph_decode, "ATTN_ROPE_KV_FUSED", True)
    monkeypatch.setattr(attn_decode_fused, "available", lambda _dev: True)
    monkeypatch.setenv("SEED_ATTN_ROPE_KV_FUSED", "1")
    prompts = [prompt_of(30, 6), prompt_of(31, 7)]
    runner = graph_runner(checkpoint, prompts)
    assert runner.enabled
