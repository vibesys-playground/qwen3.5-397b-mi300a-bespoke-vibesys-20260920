"""CPU contracts for row-sharded residuals in captured ordinary decode."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_decode  # noqa: E402


class FakeSPReduce:
    """Single-process spelling of the SP kernel's shard/full-output contract."""

    def __init__(self, residual: torch.Tensor, rank: int, world: int) -> None:
        self.full = residual.clone()
        self.rank = rank
        self.world = world
        self.calls = 0

    def sp_ok(self, mixer: torch.Tensor, residual_shard: torch.Tensor) -> bool:
        assert mixer.shape == self.full.shape
        assert residual_shard.shape[0] * self.world == mixer.shape[0]
        return True

    def sp_ar_add_rmsnorm(self, mixer, residual_shard, norm_w, eps):
        assert mixer.ndim == 3
        assert residual_shard.ndim == 3
        rows = self.full.shape[0] // self.world
        lo = self.rank * rows
        torch.testing.assert_close(residual_shard, self.full[lo : lo + rows])
        self.full = self.full + mixer
        self.calls += 1
        # RMSNorm is mocked to the identity in this structural contract test.
        return (
            self.full[lo : lo + rows].flatten(0, 1).clone(),
            self.full.flatten(0, 1).clone(),
        )


def test_full_tp_segment_keeps_residual_row_sharded(monkeypatch) -> None:
    batch, hidden, rank, world = 8, 4, 2, 4
    initial = torch.arange(batch * hidden, dtype=torch.float32).reshape(batch, 1, hidden)
    cr = FakeSPReduce(initial, rank, world)
    layers = [{"in_norm": torch.ones(hidden), "post_norm": torch.ones(hidden)} for _ in range(2)]
    model = SimpleNamespace(
        tp=SimpleNamespace(rank=rank, world=world, custom_reduce=cr),
        cfg=SimpleNamespace(eps=1e-6, layer_types=["full_attention"] * 2),
        layers=layers,
        final_norm=torch.ones(hidden),
        unembed=lambda h: h,
    )
    buf = SimpleNamespace(capacity=batch, x_in=initial, out=torch.empty_like(initial[:, 0]))

    monkeypatch.setattr(graph_decode, "AR_RMSNORM_FUSED", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE_MIN", 0)
    monkeypatch.setattr(graph_decode, "rmsnorm", lambda x, weight, eps: x)
    monkeypatch.setattr(graph_decode, "attn_decode_static", lambda model, i, h, buf: h + 1)
    monkeypatch.setattr(graph_decode, "moe_static", lambda model, i, h, buf: h + 2)

    graph_decode.segment_step(
        model, graph_decode.Segment(0, len(layers), torch.device("cpu"), True), buf
    )()

    expected = initial
    for _ in layers:
        expected = expected + (expected + 1)
        expected = expected + (expected + 2)
    assert cr.calls == 2 * len(layers)
    torch.testing.assert_close(buf.out, expected[:, 0])


def test_sp_is_limited_to_the_single_full_tp_segment(monkeypatch) -> None:
    calls = 0

    class Reducer:
        def sp_ok(self, mixer, residual):
            nonlocal calls
            calls += 1
            return True

    model = SimpleNamespace(tp=SimpleNamespace(world=4, custom_reduce=Reducer()))
    buf = SimpleNamespace(capacity=8, x_in=torch.empty(8, 1, 4))
    monkeypatch.setattr(graph_decode, "AR_RMSNORM_FUSED", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE_MIN", 0)

    graph_decode.segment_step(model, graph_decode.Segment(1, 2, torch.device("cpu"), True), buf)
    graph_decode.segment_step(model, graph_decode.Segment(0, 1, torch.device("cpu"), False), buf)

    assert calls == 0


def test_sp_skips_buckets_below_the_minimum(monkeypatch) -> None:
    calls = 0

    class Reducer:
        def sp_ok(self, mixer, residual):
            nonlocal calls
            calls += 1
            return True

    model = SimpleNamespace(tp=SimpleNamespace(world=4, custom_reduce=Reducer()))
    monkeypatch.setattr(graph_decode, "AR_RMSNORM_FUSED", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE", True)
    monkeypatch.setattr(graph_decode, "AR_SP_DECODE_MIN", 48)
    seg = graph_decode.Segment(0, 1, torch.device("cpu"), True)

    graph_decode.segment_step(model, seg, SimpleNamespace(capacity=32, x_in=torch.empty(32, 1, 4)))
    assert calls == 0
    graph_decode.segment_step(model, seg, SimpleNamespace(capacity=48, x_in=torch.empty(48, 1, 4)))
    assert calls == 1
