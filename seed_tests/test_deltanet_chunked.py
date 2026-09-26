"""`deltanet_chunked`'s Triton forward-substitution solve, and the chunked prefill it enables,
against `model.delta_rule_recurrent` (the ground truth every delta-rule spelling is checked
against; see `test_delta_rule.py`).

Two ways to run it, as for `test_deltanet_fused.py`:

    # real kernel, needs an accelerator
    python -m pytest .../seed_tests/test_deltanet_chunked.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 /tmp/torchenv/bin/python -m pytest ... -p no:cacheprovider --no-cov

`SMALL` shapes (small heads/dims, matching `test_delta_rule.py`'s convention) cover every
length under the interpreter, including a full 4096-token prefill, cheaply. `REAL_TP4` uses
this model's actual per-rank DeltaNet shape (16 value heads, k_dim = v_dim = 128, TP=4) at a
subset of lengths -- the interpreter is a Python-level simulator per program, and a 4096-token
call at the real width is a GPU-only timing, not a CPU-interpreter one; see
`deltanet_chunked.py`'s benchmark story in the perf report for that shape at full length.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import deltanet_chunked  # noqa: E402
import model as seed_model  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not deltanet_chunked.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

ATOL, RTOL = 1e-4, 1e-3
"""Looser than `test_delta_rule.py`'s 1e-5/1e-4: the Triton solve reduces in a different order
(and, under the interpreter, at a different internal precision -- see `test_deltanet_fused.py`'s
`BF16_TOL` note for the same shape of caveat) than `model.delta_rule_chunked`'s
`solve_triangular`, and both accumulate C - 1 sequential correction terms per row instead of
one `lapack` call, which is more roundings, not a different result."""

LENGTHS = [1, 13, 63, 64, 65, 512, 4096]
SEEDS = [0, 1, 2]

# Small heads (fast under the interpreter even at T=4096): matches test_delta_rule.py.
SMALL_HEADS, SMALL_K, SMALL_V = 4, 16, 24

# This model's real per-rank DeltaNet shape at TP=4 (see deltanet_tp.py / reference/config.json):
# 64 global value heads / 4 ranks = 16, k_dim = v_dim = 128.
REAL_HEADS, REAL_K, REAL_V = 16, 128, 128
REAL_LENGTHS = [1, 13, 64, 65, 512]  # 4096 at this width is a GPU-only timing, not interpreter


def random_inputs(
    seed: int, t: int, heads: int, k_dim: int, v_dim: int, *, zero_state: bool = False
) -> dict[str, torch.Tensor]:
    """Same recipe as test_delta_rule.py's: beta in (0,1), g <= 0, unnormalized qkv."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    a_log = torch.rand(heads, generator=gen) * 16.0
    g = -a_log * torch.nn.functional.softplus(randn(1, t, heads))
    rec = torch.zeros(1, heads, k_dim, v_dim) if zero_state else randn(1, heads, k_dim, v_dim) * 0.1
    return {
        "q": randn(1, t, heads, k_dim),
        "k": randn(1, t, heads, k_dim),
        "v": randn(1, t, heads, v_dim),
        "g": g,
        "beta": randn(1, t, heads).sigmoid(),
        "rec": rec,
    }


def run_both(args: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    results = []
    for fn in (seed_model.delta_rule_recurrent, deltanet_chunked.delta_rule_chunked):
        rec = args["rec"].clone()
        out = fn(args["q"], args["k"], args["v"], args["g"], args["beta"], rec)
        results.append((out, rec))
    return tuple(results)


def errors(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    diff = (got.float() - want.float()).abs()
    max_abs = diff.max().item()
    scale = want.float().abs().max().clamp_min(1e-30)
    max_rel = (diff / scale).max().item()
    return max_abs, max_rel


def assert_close(
    got: tuple[torch.Tensor, torch.Tensor], want: tuple[torch.Tensor, torch.Tensor], label: str
) -> None:
    for name, a, b in (("out", got[0], want[0]), ("state", got[1], want[1])):
        max_abs, max_rel = errors(a, b)
        assert torch.allclose(a, b, atol=ATOL, rtol=RTOL), (
            f"{label}: {name} max abs err {max_abs:.3e}, max rel err {max_rel:.3e}"
        )


# ---------------------------------------------------------------- the solve kernel in isolation


def test_lower_tri_solve_matches_torch_solve_triangular() -> None:
    """The Triton (or interpreter) forward substitution against `torch.linalg.solve_triangular`
    directly, isolated from the rest of the chunked delta rule: this is the one op being
    replaced, so it gets its own oracle comparison."""
    gen = torch.Generator().manual_seed(0)
    for n, c, w in [(1, 8, 4), (3, 64, 1), (5, 64, 256), (2, 16, 37)]:
        a = torch.randn(n, c, c, generator=gen) * 0.1
        a = a.tril(-1)
        rhs = torch.randn(n, c, w, generator=gen)
        want = torch.linalg.solve_triangular(torch.eye(c) + a, rhs, upper=False, unitriangular=True)
        got = deltanet_chunked.lower_tri_solve(a, rhs)
        max_abs, max_rel = errors(got, want)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), (
            f"n={n} c={c} w={w}: max abs err {max_abs:.3e}, max rel err {max_rel:.3e}"
        )


def test_lower_tri_solve_rejects_non_power_of_two_chunk() -> None:
    a = torch.zeros(1, 3, 3).tril(-1)
    with pytest.raises(ValueError, match="power of two"):
        deltanet_chunked.lower_tri_solve(a, torch.zeros(1, 3, 2))


# ---------------------------------------------------------------- the whole chunked delta rule


@pytest.mark.parametrize("t", LENGTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chunked_matches_recurrent_small_shape(t: int, seed: int) -> None:
    ref, got = run_both(random_inputs(seed, t, SMALL_HEADS, SMALL_K, SMALL_V))
    assert got[0].shape == ref[0].shape == (1, t, SMALL_HEADS, SMALL_V)
    assert_close(got, ref, f"small t={t} seed={seed}")


@pytest.mark.parametrize("t", REAL_LENGTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_chunked_matches_recurrent_real_tp4_shape(t: int, seed: int) -> None:
    """This model's real per-rank DeltaNet shape at TP=4: 16 value heads, k_dim = v_dim = 128."""
    ref, got = run_both(random_inputs(seed, t, REAL_HEADS, REAL_K, REAL_V))
    assert_close(got, ref, f"real t={t} seed={seed}")


@pytest.mark.parametrize("t", [1, 13, 64, 65, 512])
def test_chunked_matches_recurrent_from_zero_state(t: int) -> None:
    """A fresh sequence starts from a zeroed state, the common cold-start prefill case."""
    args = random_inputs(0, t, SMALL_HEADS, SMALL_K, SMALL_V, zero_state=True)
    ref, got = run_both(args)
    assert_close(got, ref, f"zero state t={t}")


@pytest.mark.parametrize("t", [1, 13, 64, 65, 512])
def test_chunked_matches_recurrent_from_nonzero_state(t: int) -> None:
    """Resuming from a prefix snapshot: `rec` is whatever the prior turn left it at, not zero.

    This is the shape `p95_ttft_turn2plus_ms` actually exercises: turn 2+'s prefill continues
    from turn 1's saved recurrent state (`Model.save_snapshot`/`load_snapshot`), never from zero.
    """
    args = random_inputs(1, t, REAL_HEADS, REAL_K, REAL_V)
    assert args["rec"].abs().max().item() > 0
    ref, got = run_both(args)
    assert_close(got, ref, f"nonzero state t={t}")


def test_state_carries_across_calls() -> None:
    """Chunked prefill in several pieces, resuming state each time, must equal one straight-
    through recurrent pass over the whole sequence -- the actual multi-turn shape: each new
    turn's prompt chunk resumes the previous turn's saved `rec` (see `Model.prefill`)."""
    total, split = 150, [70, 60, 1, 1, 1, 1, 16]
    args = random_inputs(7, total, REAL_HEADS, REAL_K, REAL_V)
    want_rec = args["rec"].clone()
    want = seed_model.delta_rule_recurrent(
        args["q"], args["k"], args["v"], args["g"], args["beta"], want_rec
    )

    rec, outs, pos = args["rec"].clone(), [], 0
    for n in split:
        piece = {key: args[key][:, pos : pos + n] for key in ("q", "k", "v", "g", "beta")}
        outs.append(deltanet_chunked.delta_rule_chunked(**piece, rec=rec))
        pos += n
    assert pos == total
    assert_close((torch.cat(outs, dim=1), rec), (want, want_rec), "incremental")


def test_dispatch_uses_chunked_path_for_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Model.delta_rule` routes T > 1 through the chunked path when it is available, and keeps
    T == 1 on the recurrence (decode's fast path, unchanged)."""
    monkeypatch.setattr(deltanet_chunked, "available", lambda _dev: True)
    calls = []
    real_chunked = deltanet_chunked.delta_rule_chunked

    def spy(*a, **k):
        calls.append(k.get("chunk"))
        return real_chunked(*a, **k)

    monkeypatch.setattr(deltanet_chunked, "delta_rule_chunked", spy)

    args = random_inputs(2, 20, SMALL_HEADS, SMALL_K, SMALL_V)
    seed_model.Model.delta_rule(
        args["q"], args["k"], args["v"], args["g"], args["beta"], args["rec"].clone()
    )
    assert len(calls) == 1, "T > 1 must dispatch to the chunked path when it is available"

    calls.clear()
    args1 = random_inputs(2, 1, SMALL_HEADS, SMALL_K, SMALL_V)
    seed_model.Model.delta_rule(
        args1["q"], args1["k"], args1["v"], args1["g"], args1["beta"], args1["rec"].clone()
    )
    assert len(calls) == 0, "T == 1 (decode) must stay on the recurrence, not the chunked path"
