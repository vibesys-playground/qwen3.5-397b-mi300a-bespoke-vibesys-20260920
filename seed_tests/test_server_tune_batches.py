"""`server.tune_batches`: the GEMM widths `build_model` hands `blas_tune.tune`, with and
without an MTP head. Regression: with `SEED_MTP_VERIFY_WIDE` the verify widths were appended
to a tuple as a list, a TypeError on every rank at boot (round 15, graphpf3).

    /tmp/torchenv/bin/python -m pytest seed_tests/test_server_tune_batches.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import blas_tune  # noqa: E402
import graph_decode  # noqa: E402
import graph_mixed  # noqa: E402
import graph_mtp  # noqa: E402
import graph_prefill  # noqa: E402
import server  # noqa: E402


def base(max_batch: int, max_seq: int) -> set[int]:
    return {
        *blas_tune.tuned_batches(max_batch),
        *graph_prefill.gemm_widths(max_batch, max_seq),
        *graph_mixed.gemm_widths(max_batch, max_seq),
    }


def test_production_mtp_widths_are_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_mtp, "VERIFY_WIDE", True)
    monkeypatch.setattr(graph_mtp, "VERIFY_WIDE_MIN_BUCKET", 48)
    got = server.tune_batches(96, 8192, 3)
    assert {144, 192, 240, 288} <= set(got)
    assert set(got) == base(96, 8192) | {144, 192, 240, 288}
    assert got == sorted(set(got))


def test_no_mtp_or_narrow_verify_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert server.tune_batches(96, 8192, None) == sorted(base(96, 8192))
    monkeypatch.setattr(graph_mtp, "VERIFY_WIDE", False)
    assert server.tune_batches(96, 8192, 3) == sorted(base(96, 8192))


@settings(max_examples=50, deadline=None)
@given(
    max_batch=st.integers(min_value=1, max_value=128),
    t=st.integers(min_value=2, max_value=4),
    min_bucket=st.integers(min_value=1, max_value=128),
    wide=st.booleans(),
)
def test_widths_are_the_union_of_every_source(max_batch, t, min_bucket, wide) -> None:  # noqa: ANN001
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(graph_mtp, "VERIFY_WIDE", wide)
        mp.setattr(graph_mtp, "VERIFY_WIDE_MIN_BUCKET", min_bucket)
        got = server.tune_batches(max_batch, 4096, t)
        want = base(max_batch, 4096)
        if wide:
            buckets = [b for b in graph_decode.build_buckets(max_batch) if b >= min_bucket]
            want |= {b * t for b in buckets or [max_batch]}
        assert got == sorted(want)
    finally:
        mp.undo()
