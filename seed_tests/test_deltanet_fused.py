"""The fused DeltaNet decode kernels against the torch chain they replace.

The oracles are the torch expressions in `model.py`: `model.gated_rmsnorm` for the output
norm, and `_delta_rule_inputs` plus `delta_rule_recurrent` (with the gate activations
`deltanet_decode` applies before them) for the recurrence. The kernels are a reformulation of
exactly those, not an approximation, so the bar is fp32 reduction-order noise.

Two ways to run it, as for `test_mxfp4_fused_gemv.py`:

    # real kernels, needs an accelerator
    python -m pytest .../seed_tests/test_deltanet_fused.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 /tmp/torchenv/bin/python -m pytest ... -p no:cacheprovider --no-cov

`test_decode_matches_the_torch_path` is the end-to-end one: it runs a whole `deltanet_decode`
both ways on the tiny checkpoint and compares the layer output *and* the recurrent and conv
state it left behind, which is what a wrong in-place update would show up in.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import deltanet_fused  # noqa: E402
import model as seed_model  # noqa: E402
from test_seed_parity import build_hf, write_checkpoint  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not deltanet_fused.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

FP32_TOL = 1e-5
BF16_TOL = 5e-2 if INTERPRET else 1e-2
"""Agreement bar against the torch oracle, relative to the oracle's largest magnitude.

The kernels round to the storage dtype wherever torch does, so on an accelerator the only
difference is `tl.sum`'s reduction order and the bf16 arm stays inside an ulp or two. Triton's
*interpreter* additionally truncates fp32 to bf16 where the hardware rounds to nearest (see
`tl.cast`), which biases every rounding a fraction of an ulp low and stacks up over the three
the norm makes, so the interpreter arm gets a looser bar rather than a false pass on device.
"""


def tolerance(dtype: torch.dtype) -> float:
    return BF16_TOL if dtype is torch.bfloat16 else FP32_TOL


SHAPES = pytest.mark.parametrize(
    "rows,k_heads,rep,k_dim,v_dim",
    [
        (1, 2, 1, 16, 16),  # batch 1, no head repetition: the simplest case
        (5, 2, 4, 16, 24),  # v_dim not a power of two, so the v block is masked
        (8, 4, 4, 32, 32),  # rep 4 and k_heads 4, the real model's ratio
        (3, 1, 8, 8, 64),  # v_dim wider than one block, so the grid's second axis is live
    ],
)


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    scale = want.float().abs().max().item()
    return ((got.float() - want.float()).abs().max() / max(scale, 1e-30)).item()


# ---------------------------------------------------------------- the output norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows,heads,cols", [(1, 1, 8), (4, 3, 16), (5, 2, 24), (2, 4, 64)])
def test_gated_rmsnorm_matches_torch(rows: int, heads: int, cols: int, dtype: torch.dtype) -> None:
    """One kernel against the twelve torch ops, at the dtype the engine ships."""
    gen = torch.Generator().manual_seed(rows * 100 + cols)
    x = torch.randn(rows, heads, cols, generator=gen).to(DEVICE).to(dtype)
    gate = torch.randn(rows, heads, cols, generator=gen).to(DEVICE).to(dtype)
    w = torch.randn(cols, generator=gen).to(DEVICE).to(dtype)

    want = seed_model.gated_rmsnorm(x, gate, w, 1e-6)
    got = deltanet_fused.gated_rmsnorm(x, gate, w, 1e-6)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative_error(got, want) < tolerance(dtype)


def test_gated_rmsnorm_reads_strided_operands() -> None:
    """The gate arrives as a slice of the concatenated input projection, never contiguous.

    `fuse_in_proj` makes one `F.linear` produce all four projections, so `deltanet_decode`
    hands the gate in as a narrow view whose row stride is the whole projection's width. A
    kernel that assumed row-major packing would read the wrong addresses and come back
    plausible but wrong, so the strided case is the one that ships.
    """
    rows, heads, cols = 4, 3, 16
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(rows, heads, cols, generator=gen).to(DEVICE)
    wide = torch.randn(rows, heads * cols * 3, generator=gen).to(DEVICE)
    gate = wide[:, cols : cols + heads * cols].unflatten(-1, (heads, cols))
    assert not gate.is_contiguous() and gate.stride(0) != heads * cols
    w = torch.randn(cols, generator=gen).to(DEVICE)

    want = seed_model.gated_rmsnorm(x, gate, w, 1e-6)
    got = deltanet_fused.gated_rmsnorm(x, gate, w, 1e-6)
    assert relative_error(got, want) < FP32_TOL


def test_gated_rmsnorm_rejects_a_shape_mismatch() -> None:
    x = torch.randn(2, 2, 8, device=DEVICE)
    with pytest.raises(ValueError, match="shape mismatch"):
        deltanet_fused.gated_rmsnorm(x, x[:, :1], torch.randn(8, device=DEVICE), 1e-6)


# ---------------------------------------------------------------- the recurrence


def torch_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: tuple[torch.Tensor, torch.Tensor],
    rec: torch.Tensor,
    params: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """`deltanet_decode`'s torch spelling of one decode step, on the kernel's operand shapes.

    Transcribed from `Model.deltanet_decode`'s fallback branch so the oracle stays the code
    that shipped: the same `sigmoid` in the projection's dtype, the same `softplus` in fp32,
    the same `repeat_interleave`, and `delta_rule_recurrent` itself.
    """
    a_raw, beta_raw = gate
    a_log, dt_bias = params
    rep = v.shape[1] // q.shape[1]
    beta = beta_raw.sigmoid()[:, None]
    g = -a_log.exp() * F.softplus(a_raw.float() + dt_bias)
    out = seed_model.delta_rule_recurrent(
        q[:, None].repeat_interleave(rep, dim=2),
        k[:, None].repeat_interleave(rep, dim=2),
        v[:, None],
        g[:, None],
        beta,
        rec,
    )
    return out[:, 0]


def decode_inputs(
    rows: int, k_heads: int, rep: int, k_dim: int, v_dim: int, *, seed: int, dtype: torch.dtype
) -> dict:
    """Operands in the ranges the model produces: conv-shaped q/k/v, raw gate projections."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE)

    v_heads = k_heads * rep
    return {
        "q": randn(rows, k_heads, k_dim).to(dtype),
        "k": randn(rows, k_heads, k_dim).to(dtype),
        "v": randn(rows, v_heads, v_dim).to(dtype),
        "gate": (randn(rows, v_heads).to(dtype), randn(rows, v_heads).to(dtype)),
        "rec": randn(rows, v_heads, k_dim, v_dim) * 0.1,
        "params": ((torch.rand(v_heads, generator=gen) * 16.0).to(DEVICE), randn(v_heads)),
    }


@SHAPES
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_rule_decode_matches_torch(
    rows: int, k_heads: int, rep: int, k_dim: int, v_dim: int, dtype: torch.dtype
) -> None:
    """Output and the advanced state, both. The state is in-place, so it needs its own check."""
    args = decode_inputs(rows, k_heads, rep, k_dim, v_dim, seed=rows + k_dim, dtype=dtype)
    rec_fused, rec_torch = args["rec"].clone(), args["rec"].clone()

    want = torch_delta_rule_decode(
        args["q"], args["k"], args["v"], args["gate"], rec_torch, args["params"]
    )
    got = deltanet_fused.delta_rule_decode(
        (args["q"], args["k"]), args["v"], args["gate"], rec_fused, args["params"]
    )
    assert got.shape == want.shape and got.dtype == torch.float32
    tol = tolerance(dtype)
    assert relative_error(got, want) < tol
    assert relative_error(rec_fused, rec_torch) < tol, "the recurrent state must advance alike"


@SHAPES
def test_delta_rule_decode_active_mask_matches_manual_where(
    rows: int, k_heads: int, rep: int, k_dim: int, v_dim: int
) -> None:
    """`active=` inside the kernel must equal `clone()` + kernel + `torch.where` outside it.

    This is `graph_decode.deltanet_decode_static`'s `SEED_FUSE_GLUE=1` optimization (see the
    kernel's `HAS_ACTIVE` docstring): the whole point is that a False row's `rec` bytes are
    never touched, so this checks both that a True row's state advances exactly like the
    `torch.where`-selected version and that a False row's state is bit-identical to what it
    was before the call (not just numerically close to the "correct" unchanged value -- it
    must genuinely be untouched, no float round-trip at all).
    """
    args = decode_inputs(
        rows, k_heads, rep, k_dim, v_dim, seed=rows + k_dim + 17, dtype=torch.float32
    )
    gen = torch.Generator().manual_seed(rows * 7 + 3)
    active = torch.rand(rows, generator=gen) > 0.5
    if not active.any():
        active[0] = True
    if active.all() and rows > 1:
        active[0] = False

    rec_masked = args["rec"].clone()
    rec_before = args["rec"].clone()
    out_masked = deltanet_fused.delta_rule_decode(
        (args["q"], args["k"]), args["v"], args["gate"], rec_masked, args["params"], active=active
    )

    rec_unmasked = args["rec"].clone()
    out_unmasked = deltanet_fused.delta_rule_decode(
        (args["q"], args["k"]), args["v"], args["gate"], rec_unmasked, args["params"]
    )
    # The "clone, run unconditionally, torch.where back in" pattern the kernel replaces.
    rec_via_where = torch.where(active[:, None, None, None], rec_unmasked, rec_before)

    assert torch.equal(rec_masked[~active], rec_before[~active]), "inactive rows must be untouched"
    assert relative_error(rec_masked, rec_via_where) < FP32_TOL
    assert relative_error(out_masked, out_unmasked) < FP32_TOL, "output is computed for every row"


def test_delta_rule_decode_rejects_mismatched_active_shape() -> None:
    args = decode_inputs(3, 2, 4, 16, 24, seed=1, dtype=torch.float32)
    with pytest.raises(ValueError, match="active must be"):
        deltanet_fused.delta_rule_decode(
            (args["q"], args["k"]),
            args["v"],
            args["gate"],
            args["rec"],
            args["params"],
            active=torch.ones(2, dtype=torch.bool),
        )


def test_delta_rule_decode_rejects_mismatched_heads() -> None:
    args = decode_inputs(2, 3, 1, 8, 8, seed=1, dtype=torch.float32)
    with pytest.raises(ValueError, match="not a multiple"):
        deltanet_fused.delta_rule_decode(
            (args["q"][:, :2], args["k"][:, :2]),
            args["v"],
            args["gate"],
            args["rec"],
            args["params"],
        )


@pytest.mark.parametrize(
    "rows,width,k_heads,rep,k_dim,v_dim",
    [(1, 2, 2, 1, 16, 16), (4, 3, 2, 4, 16, 24), (4, 4, 2, 4, 16, 24)],
)
def test_delta_rule_verify_exact_matches_four_decode_updates(
    rows: int, width: int, k_heads: int, rep: int, k_dim: int, v_dim: int
) -> None:
    """Each fixed-width kernel must reproduce the deployed decode kernel step for step."""
    gen = torch.Generator().manual_seed(rows * 101 + v_dim)
    v_heads = k_heads * rep

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(dtype)

    q = randn(rows, width, k_heads, k_dim)
    k = randn(rows, width, k_heads, k_dim)
    v = randn(rows, width, v_heads, v_dim)
    a = randn(rows, width, v_heads)
    b = randn(rows, width, v_heads)
    a_log = randn(v_heads, dtype=torch.float32)
    dt_bias = randn(v_heads, dtype=torch.float32)
    rec_step = randn(rows, v_heads, k_dim, v_dim, dtype=torch.float32)
    rec_wide = rec_step.clone()

    want = torch.stack(
        [
            deltanet_fused.delta_rule_decode(
                (q[:, step], k[:, step]),
                v[:, step],
                (a[:, step], b[:, step]),
                rec_step,
                (a_log, dt_bias),
            )
            for step in range(width)
        ],
        dim=1,
    )
    got = deltanet_fused.delta_rule_verify_exact(
        (q, k), v, (a, b), rec_wide, (a_log, dt_bias)
    )

    torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(rec_wide, rec_step, atol=0, rtol=0)


def test_delta_rule_verify_exact_masks_shared_padding_lane() -> None:
    """Inactive graph rows may share one lane; only the active row may update persistent state."""
    rows, width, k_heads, v_heads, k_dim, v_dim = 4, 4, 2, 8, 16, 24
    gen = torch.Generator().manual_seed(919)

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(dtype)

    q = randn(rows, width, k_heads, k_dim)
    k = randn(rows, width, k_heads, k_dim)
    v = randn(rows, width, v_heads, v_dim)
    a = randn(rows, width, v_heads)
    b = randn(rows, width, v_heads)
    params = (randn(v_heads, dtype=torch.float32), randn(v_heads, dtype=torch.float32))
    state = randn(2, v_heads, k_dim, v_dim, dtype=torch.float32)
    before = state.clone()
    active = torch.tensor([True, False, False, False], device=DEVICE)
    lanes = torch.tensor([0, 1, 1, 1], device=DEVICE)

    deltanet_fused.delta_rule_verify_exact(
        (q, k), v, (a, b), state, params, active=active, lanes=lanes
    )

    assert not torch.equal(state[0], before[0])
    assert torch.equal(state[1], before[1]), "the shared padding lane must remain untouched"


@pytest.mark.parametrize("width", [2, 3, 4])
def test_delta_rule_verify_exact_applies_per_row_prefix_lengths(width: int) -> None:
    """Rollback advances each lane through only its committed prefix at every fixed width."""
    rows, k_heads, v_heads, k_dim, v_dim = width, 2, 8, 16, 24
    gen = torch.Generator().manual_seed(1231)

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(dtype)

    q = randn(rows, width, k_heads, k_dim)
    k = randn(rows, width, k_heads, k_dim)
    v = randn(rows, width, v_heads, v_dim)
    a = randn(rows, width, v_heads)
    b = randn(rows, width, v_heads)
    params = (randn(v_heads, dtype=torch.float32), randn(v_heads, dtype=torch.float32))
    lengths = torch.arange(1, width + 1, dtype=torch.int32, device=DEVICE)
    state = randn(rows, v_heads, k_dim, v_dim, dtype=torch.float32)
    want = state.clone()
    for row, length in enumerate(lengths.tolist()):
        for step in range(length):
            deltanet_fused.delta_rule_decode(
                (q[row : row + 1, step], k[row : row + 1, step]),
                v[row : row + 1, step],
                (a[row : row + 1, step], b[row : row + 1, step]),
                want[row : row + 1],
                params,
            )

    deltanet_fused.delta_rule_verify_exact(
        (q, k), v, (a, b), state, params, lengths=lengths
    )
    torch.testing.assert_close(state, want, atol=0, rtol=0)


# ---------------------------------------------------------------- the whole mixer


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-deltanet")
    write_checkpoint(build_hf(), out, mxfp4=False)
    return out


def test_fuse_in_proj_keeps_the_four_names_as_views(checkpoint: Path) -> None:
    """One weight, four views: no projection is stored twice and no reader has to change."""
    model = seed_model.Model(checkpoint, [str(DEVICE)], torch.float32, 32, 2)
    layer = next(
        w for w, t in zip(model.layers, model.cfg.layer_types, strict=True) if t != "full_attention"
    )
    fused = layer["in_proj_all"]
    assert fused.shape[0] == sum(seed_model.in_proj_sizes(model.cfg))
    at = 0
    for name, size in zip(
        seed_model.IN_PROJ_ORDER, seed_model.in_proj_sizes(model.cfg), strict=True
    ):
        assert layer[name].data_ptr() == fused[at].data_ptr(), f"{name} must be a view"
        assert layer[name].shape[0] == size
        at += size


def test_decode_matches_the_torch_path(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole `deltanet_decode` both ways, on the tiny checkpoint, state included.

    The two arms run on independent copies of the pool, so the comparison covers the layer
    output and everything the call wrote back into the recurrent and conv state. Forcing
    `available` is what lets this run on CPU under the Triton interpreter; on an accelerator
    it is already the shipping path.
    """
    torch.manual_seed(3)
    model = seed_model.Model(checkpoint, [str(DEVICE)], torch.float32, 32, 4)
    i = next(n for n, t in enumerate(model.cfg.layer_types) if t != "full_attention")
    slots = [0, 2, 3]
    x = torch.randn(len(slots), 1, model.cfg.hidden, device=DEVICE)
    for name, tensor in model.pool[i].items():
        model.pool[i][name] = torch.randn_like(tensor) * 0.1 if name == "rec" else tensor

    saved = {name: t.clone() for name, t in model.pool[i].items()}
    monkeypatch.setattr(deltanet_fused, "available", lambda _dev: False)
    want = model.deltanet_decode(i, x, slots)
    after_torch = {name: t.clone() for name, t in model.pool[i].items()}

    for name, tensor in saved.items():
        model.pool[i][name].copy_(tensor)
    monkeypatch.setattr(deltanet_fused, "available", lambda _dev: True)
    got = model.deltanet_decode(i, x, slots)

    assert relative_error(got, want) < FP32_TOL
    for name, tensor in after_torch.items():
        assert relative_error(model.pool[i][name], tensor) < FP32_TOL, f"{name} state diverged"
    untouched = [s for s in range(4) if s not in slots]
    assert torch.equal(model.pool[i]["rec"][untouched], saved["rec"][untouched])


# ---------------------------------------------------------------- masked_row_copy


@pytest.mark.parametrize("rows,channels,width", [(1, 1, 1), (4, 3, 1), (5, 8, 3), (6, 2, 5)])
def test_masked_row_copy_matches_torch_where(rows: int, channels: int, width: int) -> None:
    gen = torch.Generator().manual_seed(rows * 100 + channels * 10 + width)
    dst = torch.randn(rows, channels, width, generator=gen).to(DEVICE)
    src = torch.randn(rows, channels, width, generator=gen).to(DEVICE)
    active = torch.rand(rows, generator=gen) > 0.5
    if not active.any():
        active[0] = True
    if active.all() and rows > 1:
        active[0] = False

    before = dst.clone()
    want = torch.where(active[:, None, None], src, dst)
    deltanet_fused.masked_row_copy(dst, src, active)

    assert torch.equal(dst, want)
    assert torch.equal(dst[~active], before[~active]), "inactive rows must be untouched"


def test_masked_row_copy_handles_a_strided_src() -> None:
    """`causal_conv_static`'s real call: `src` is a suffix slice of `cat([state, x])`, whose
    channel stride is one wider than `width` -- the case `masked_row_copy`'s own docstring
    says a flat reshape would silently force a copy for."""
    rows, channels, width, conv_k = 4, 5, 3, 4
    gen = torch.Generator().manual_seed(11)
    full = torch.randn(rows, channels, conv_k, generator=gen).to(DEVICE)
    src = full[:, :, -width:]
    assert src.stride(1) == conv_k != width
    dst = torch.randn(rows, channels, width, generator=gen).to(DEVICE)
    active = torch.tensor([True, False, True, False])

    before = dst.clone()
    want = torch.where(active[:, None, None], src, dst)
    deltanet_fused.masked_row_copy(dst, src, active)

    assert torch.equal(dst, want)
    assert torch.equal(dst[~active], before[~active])


# ---------------------------------------------------------------- causal_conv_decode


def torch_causal_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_pool: torch.Tensor,
    lanes: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """`graph_decode.causal_conv_static`'s math, applied lane-by-lane against the pool directly
    -- the oracle for `causal_conv_decode`'s in-place read/advance."""
    state = state_pool[lanes].clone()
    full = torch.cat([state, x[:, :, None]], dim=-1)
    new_state = full[:, :, -state.shape[-1] :]
    out = F.silu(F.conv1d(full, weight, groups=full.shape[1]))[:, :, 0]
    for row in range(x.shape[0]):
        if active[row]:
            state_pool[lanes[row].item()] = new_state[row]
    return out


CONV_SHAPES = pytest.mark.parametrize(
    "rows,channels,conv_k",
    [
        (1, 1, 2),  # simplest case
        (4, 5, 4),  # the real model's conv kernel width
        (5, 8, 3),  # every row a distinct lane, no padding
        (3, 6, 2),  # conv_k = 2: single-tap state, no shift needed
    ],
)


@CONV_SHAPES
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv_decode_matches_torch(
    rows: int, channels: int, conv_k: int, dtype: torch.dtype
) -> None:
    gen = torch.Generator().manual_seed(rows * 97 + channels * 11 + conv_k)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(dtype)

    x = randn(rows, channels)
    weight = randn(channels, 1, conv_k)
    lane_count = rows + 1
    state_pool_a = randn(lane_count, channels, conv_k - 1)
    state_pool_b = state_pool_a.clone()
    active = torch.rand(rows, generator=gen) > 0.5
    if not active.any():
        active[0] = True
    if active.all() and rows > 1:
        active[0] = False
    # Lane 0 is the shared filler every inactive row points at; each active row gets its own
    # distinct lane (1..rows) -- the real invariant `pad_slot_for` keeps (a padding lane is
    # never one an active row also owns), which the kernel's read-before-write safety and the
    # "filler lane untouched" guarantee both depend on.
    own_lane = torch.arange(1, rows + 1, device=DEVICE)
    lanes = torch.where(active, own_lane, torch.zeros_like(own_lane))

    want = torch_causal_conv_decode(x, weight, state_pool_a, lanes, active)
    got = deltanet_fused.causal_conv_decode(x, weight, state_pool_b, lanes, active)

    tol = tolerance(dtype)
    assert relative_error(got, want) < tol
    assert relative_error(state_pool_b, state_pool_a) < tol


def test_causal_conv_decode_inactive_rows_leave_their_lane_untouched() -> None:
    rows, lane_count, channels, conv_k = 4, 2, 3, 4
    gen = torch.Generator().manual_seed(23)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE)

    x = randn(rows, channels)
    weight = randn(channels, 1, conv_k)
    state_pool = randn(lane_count, channels, conv_k - 1)
    before = state_pool.clone()
    # Lane 1 is the shared filler for every inactive row; lane 0 is a real, active row.
    lanes = torch.tensor([0, 1, 1, 1], device=DEVICE)
    active = torch.tensor([True, False, False, False], device=DEVICE)

    deltanet_fused.causal_conv_decode(x, weight, state_pool, lanes, active)

    assert torch.equal(state_pool[1], before[1]), "the shared filler lane must be untouched"
    assert not torch.equal(state_pool[0], before[0])


def test_causal_conv_decode_rejects_mismatched_state_pool_shape() -> None:
    x = torch.randn(2, 4, device=DEVICE)
    weight = torch.randn(4, 1, 3, device=DEVICE)
    bad_pool = torch.randn(3, 4, 5, device=DEVICE)  # width should be conv_k - 1 == 2
    lanes = torch.zeros(2, dtype=torch.long, device=DEVICE)
    active = torch.ones(2, dtype=torch.bool, device=DEVICE)
    with pytest.raises(ValueError, match="state_pool must be"):
        deltanet_fused.causal_conv_decode(x, weight, bad_pool, lanes, active)


@pytest.mark.parametrize("rows,width", [(1, 2), (4, 3), (4, 4)])
def test_causal_conv_verify_exact_matches_repeated_decode(rows: int, width: int) -> None:
    channels, conv_k = 8, 4
    gen = torch.Generator().manual_seed(rows * 97 + width)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(torch.bfloat16)

    x = randn(rows, width, channels)
    weight = randn(channels, 1, conv_k)
    initial = randn(rows, channels, conv_k - 1)
    lanes = torch.arange(rows, device=DEVICE)
    active = torch.ones(rows, dtype=torch.bool, device=DEVICE)

    reference_state = initial.clone()
    reference = torch.stack(
        [
            deltanet_fused.causal_conv_decode(
                x[:, step], weight, reference_state, lanes, active
            )
            for step in range(width)
        ],
        dim=1,
    )
    exact_state = initial.clone()
    exact = deltanet_fused.causal_conv_verify_exact(x, weight, exact_state)

    assert torch.equal(exact, reference)
    assert torch.equal(exact_state, reference_state)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_causal_conv_verify_exact_rolls_back_variable_prefixes(width: int) -> None:
    rows, channels, conv_k = width, 8, 4
    gen = torch.Generator().manual_seed(1000 + width)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(DEVICE).to(torch.bfloat16)

    x = randn(rows, width, channels)
    weight = randn(channels, 1, conv_k)
    initial = randn(rows, channels, conv_k - 1)
    lengths = torch.arange(1, rows + 1, dtype=torch.int32, device=DEVICE).clamp(max=width)
    lanes = torch.arange(rows, device=DEVICE)

    reference_state = initial.clone()
    for step in range(width):
        deltanet_fused.causal_conv_decode(
            x[:, step], weight, reference_state, lanes, step < lengths
        )
    exact_state = initial.clone()
    deltanet_fused.causal_conv_verify_exact(x, weight, exact_state, lengths=lengths)

    assert torch.equal(exact_state, reference_state)
