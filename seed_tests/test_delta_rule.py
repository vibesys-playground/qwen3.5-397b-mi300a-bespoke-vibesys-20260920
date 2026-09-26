"""Hermetic CPU tests: the chunked delta rule matches the per-token recurrence exactly enough.

The per-token form (`delta_rule_recurrent`) is the ground truth here; it is the seed's original
implementation and mirrors `torch_recurrent_gated_delta_rule` in the reference modeling code.
Only torch is needed, no checkpoint and no GPU:

    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_delta_rule.py \
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402

HEADS, K_DIM, V_DIM = 4, 16, 24
ATOL, RTOL = 1e-5, 1e-4

# DELTA_CHUNK is 64, so these span a single partial chunk, exact multiples, and 2-3 boundaries.
LENGTHS = [1, 2, 7, 63, 64, 65, 128, 129, 190]
SEEDS = [0, 1, 2, 3, 4]


def random_inputs(seed: int, t: int, heads: int = HEADS) -> dict[str, torch.Tensor]:
    """Delta-rule inputs with the ranges the model produces: beta in (0,1), g <= 0, unnormalized qkv."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    a_log = torch.rand(heads, generator=gen) * 16.0  # exp(A_log) in the checkpoint's (0, 16)
    g = -a_log * torch.nn.functional.softplus(randn(1, t, heads))
    return {
        "q": randn(1, t, heads, K_DIM),
        "k": randn(1, t, heads, K_DIM),
        "v": randn(1, t, heads, V_DIM),
        "g": g,
        "beta": randn(1, t, heads).sigmoid(),
        "rec": randn(1, heads, K_DIM, V_DIM) * 0.1,
    }


def run_both(args: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """(out, final state) from the per-token form and from the chunked form, same inputs."""
    results = []
    for fn in (seed_model.delta_rule_recurrent, seed_model.delta_rule_chunked):
        rec = args["rec"].clone()
        out = fn(args["q"], args["k"], args["v"], args["g"], args["beta"], rec)
        results.append((out, rec))
    return tuple(results)


def assert_close(
    got: tuple[torch.Tensor, torch.Tensor], want: tuple[torch.Tensor, torch.Tensor], label: str
) -> None:
    for name, a, b in (("out", got[0], want[0]), ("state", got[1], want[1])):
        err = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=ATOL, rtol=RTOL), f"{label}: {name} max abs err {err:.3e}"


@pytest.mark.parametrize("t", LENGTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chunked_matches_recurrent(t: int, seed: int) -> None:
    ref, chunked = run_both(random_inputs(seed, t))
    assert chunked[0].shape == ref[0].shape == (1, t, HEADS, V_DIM)
    assert_close(chunked, ref, f"t={t} seed={seed}")


@pytest.mark.parametrize("chunk", [1, 2, 3, 8])
def test_small_chunk_sizes_match(chunk: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force many chunk boundaries (including chunk=1, the degenerate per-token case)."""
    monkeypatch.setattr(seed_model, "DELTA_CHUNK", chunk)
    for seed in SEEDS:
        ref, got = run_both(random_inputs(seed, 17))
        assert_close(got, ref, f"chunk={chunk} seed={seed}")


def test_zero_initial_state_matches() -> None:
    """A fresh sequence starts from a zeroed state, the common prefill case."""
    args = random_inputs(0, 100)
    args["rec"] = torch.zeros_like(args["rec"])
    ref, got = run_both(args)
    assert_close(got, ref, "zero state")


def test_decode_step_degenerates_to_the_recurrence() -> None:
    """The chunked form on one token is the per-token update, so the dispatch is a speed choice."""
    args = random_inputs(3, 1)
    ref, got = run_both(args)
    assert_close(got, ref, "t=1")
    rec = args["rec"].clone()
    dispatched = seed_model.Model.delta_rule(
        args["q"], args["k"], args["v"], args["g"], args["beta"], rec
    )
    assert torch.equal(dispatched, ref[0])
    assert torch.equal(rec, ref[1])


def test_state_carries_across_calls() -> None:
    """Chunked prefill then decode steps must equal one per-token pass over the whole sequence."""
    total, split = 150, [70, 60, 1, 1, 1, 1, 16]
    args = random_inputs(7, total)
    want_rec = args["rec"].clone()
    want = seed_model.delta_rule_recurrent(
        args["q"], args["k"], args["v"], args["g"], args["beta"], want_rec
    )

    rec, outs, pos = args["rec"].clone(), [], 0
    for n in split:
        piece = {key: args[key][:, pos : pos + n] for key in ("q", "k", "v", "g", "beta")}
        outs.append(seed_model.Model.delta_rule(**piece, rec=rec))
        pos += n
    assert pos == total
    assert_close((torch.cat(outs, dim=1), rec), (want, want_rec), "incremental")
