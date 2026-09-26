"""Boot-memory flags: the launcher preflight (`SEED_BOOT_MEM_PREFLIGHT_GIB`) and touched-lane
prefill validation (`SEED_PREFILL_VALIDATE_LANES`), which must give the whole-pool verdicts.

    <python-with-torch> -m pytest seed_tests/test_alloc_order.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import graph_prefill  # noqa: E402
import mem_timeline  # noqa: E402
from test_graph_prefill import SHAPES, checkpoint, drifting_step, runner_for  # noqa: E402, F401


def test_preflight_off_by_default() -> None:
    mem_timeline.preflight(0, read=lambda: pytest.fail("read with preflight off"))


def test_preflight_waits_then_passes() -> None:
    reads = iter([[90.0, 120.0], [99.0, 120.0], [101.0, 118.0]])
    slept = []
    mem_timeline.preflight(100, wait_s=60, read=lambda: next(reads), sleep=slept.append, clock=lambda: 0.0)
    assert len(slept) == 2


def test_preflight_refuses_after_deadline() -> None:
    t = iter(range(0, 1000, 10))
    with pytest.raises(RuntimeError, match="still below"):
        mem_timeline.preflight(100, wait_s=30, read=lambda: [50.0], sleep=lambda _: None, clock=lambda: float(next(t)))


def test_mark_is_noop_when_disabled(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(mem_timeline, "ENABLED", False)
    mem_timeline.mark("x")  # no device, no snapshot, no error


@settings(max_examples=60, deadline=None)
@given(
    lanes=st.integers(1, 6),
    total=st.integers(6, 12),
    seed=st.integers(0, 2**31 - 1),
    shape=st.sampled_from([(3, 4), (2, 2, 5)]),
    noise=st.sampled_from([0.0, 1e-3, 0.3, 3.0]),
)
def test_zero_padded_corr_matches_whole_pool(lanes, total, seed, shape, noise) -> None:  # noqa: ANN001
    g = torch.Generator().manual_seed(seed)
    got = torch.randn(lanes, *shape, generator=g)
    want = got + noise * torch.randn(lanes, *shape, generator=g)
    pad = torch.zeros(total - lanes, *shape)
    full = graph_prefill._corr(torch.cat([got, pad]), torch.cat([want, pad]))
    part = graph_prefill._corr_zero_padded(got, want, total)
    assert part == pytest.approx(full, abs=1e-4)  # _corr rounds in float32
    assert graph_prefill._state_corr([(got, got)], [(want, want)], total) == pytest.approx(full, abs=1e-4)  # _corr rounds in float32


def test_zero_padded_corr_constant_cases() -> None:
    z = torch.zeros(2, 3)
    assert graph_prefill._corr_zero_padded(z, z, 5) == graph_prefill._corr(torch.zeros(5, 3), torch.zeros(5, 3))
    one = torch.zeros(2, 3)
    one[0, 0] = 1.0
    full = graph_prefill._corr(torch.cat([one, torch.zeros(3, 3)]), torch.zeros(5, 3))
    assert graph_prefill._corr_zero_padded(one, z, 5) == full


@pytest.mark.parametrize("max_batch", [3, 6])
def test_touched_lane_validation_keeps_every_shape(checkpoint: Path, monkeypatch, max_batch) -> None:  # noqa: ANN001, F811
    seen = []
    real = graph_prefill.PrefillGraphRunner._deltanet_state

    def state(self, lanes=None):  # noqa: ANN001, ANN202
        seen.append(lanes)
        return real(self, lanes)

    monkeypatch.setattr(graph_prefill, "VALIDATE_LANES", True)
    monkeypatch.setattr(graph_prefill.PrefillGraphRunner, "_deltanet_state", state)
    _, runner = runner_for(checkpoint, monkeypatch, max_batch=max_batch)
    pr = runner.prefill_runner
    assert pr.enabled
    assert sorted(pr.graphs) == sorted(graph_prefill.parse_shapes(SHAPES, max_batch, 96))
    assert any(n is not None and n < max_batch for n in seen)  # clones were partial


def test_touched_lane_validation_still_rejects_decorrelated_logits(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setattr(graph_prefill, "VALIDATE_LANES", True)
    drifting_step(monkeypatch, lambda out: out.roll(7, dims=-1))
    _, runner = runner_for(checkpoint, monkeypatch, max_batch=6)
    assert not runner.prefill_runner.enabled


def test_a_write_outside_the_touched_lanes_falls_back_to_whole_pool(checkpoint: Path, monkeypatch) -> None:  # noqa: ANN001, F811
    """A replay that writes a lane the call does not own is what the whole-pool clones exist to
    catch; the touched-lane path must notice and rerun the shape with whole-pool clones."""
    real = graph_prefill.prefill_step
    whole = []

    def step_for(model, buf):  # noqa: ANN001, ANN202
        inner = real(model, buf)

        def step() -> None:
            inner()
            for p in model.pool:
                if "conv" in p:
                    p["rec"][-1].add_(1.0)

        return step

    real_check = graph_prefill.PrefillGraphRunner._check

    def check(self, shape, whole_pool=False):  # noqa: ANN001, ANN202
        whole.append(whole_pool)
        return real_check(self, shape, whole_pool)

    monkeypatch.setattr(graph_prefill, "VALIDATE_LANES", True)
    monkeypatch.setattr(graph_prefill, "prefill_step", step_for)
    monkeypatch.setattr(graph_prefill.PrefillGraphRunner, "_check", check)
    runner_for(checkpoint, monkeypatch, max_batch=6)
    assert True in whole
