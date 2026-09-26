"""Hermetic CPU tests for the DeltaNet tensor-parallel sharding (`deltanet_tp`).

Two layers of checking, both single process:

1. The derivation itself, on the delta-rule kernels alone: the recurrence is block
   diagonal over value heads, so running head slices separately and concatenating must
   be *bit* identical to running all heads at once, with no communication anywhere.
   The same section pins the two alternatives the derivation rejects: a `v_dim` split is
   also communication free, and a `k_dim` split is not (it needs a collective per step).
2. The whole layer, hand sharded: build the tiny checkpoint once, construct one
   unsharded `Model` and `world_size` sharded ones, and check that summing the ranks'
   `deltanet` partial outputs reproduces the unsharded `deltanet` and that the ranks'
   recurrent and conv states reassemble into the unsharded ones.

`test_deltanet_tp_gloo.py` runs the same layer through real `torch.distributed`
collectives. Run with a python that has torch and safetensors:

    /tmp/torchenv/bin/python -m pytest \
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_deltanet_tp.py \
        -p no:cacheprovider --no-cov
"""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deltanet_tp  # noqa: E402
import deltanet_tp_fixture as fx  # noqa: E402
import model as seed_model  # noqa: E402
import tp as tp_mod  # noqa: E402


def group(ckpt: Path, rank: int, world: int) -> tp_mod.TP:
    """This rank's `tp.TP` on the fixture checkpoint, with no all-reduce.

    These handles are deliberately unconnected: every test below calls the component
    methods, which return row-parallel partial sums, and combines the ranks' partials by
    hand. `test_deltanet_tp_gloo.py` runs the same layer through the real collective, which
    `Model.layer` issues.
    """
    plan = tp_mod.plan(seed_model.load_cfg(ckpt), rank, world)
    return tp_mod.TP(plan, "cpu", None if world == 1 else (lambda _x: None))


# Real DeltaNet dims: 64 value heads over 16 key heads, k_dim = v_dim = 128.
HEADS, K_DIM, V_DIM = 64, 128, 128
# Spans a partial chunk, exact multiples of DELTA_CHUNK (64), and 2-3 chunk boundaries.
LENGTHS = [1, 2, 7, 63, 64, 65, 130]
WORLDS = [2, 4]


def _chunked_is_usable() -> bool:
    """Whether `delta_rule_chunked` runs at all in this build.

    It needs `torch.linalg.solve_triangular`, which the ROCm image this candidate deploys on
    does not implement for these shapes (`model.Model.delta_rule` documents why the chunked
    form is dead code on gfx942, and `seed_tests/test_delta_rule.py` fails there for the same
    reason). Sharding is orthogonal to which delta-rule spelling runs, so the value-head
    split is checked against whichever forms the platform can execute rather than made to
    fail on a missing op it does not depend on.
    """
    try:
        a = torch.eye(2).expand(1, 1, 2, 2)
        torch.linalg.solve_triangular(a, torch.zeros(1, 1, 2, 2), upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        return False
    return True


FORMS = [seed_model.delta_rule_recurrent]
if _chunked_is_usable():
    FORMS.append(seed_model.delta_rule_chunked)

# The all_reduce over out_proj is the one place the sharded layer reassociates a sum, so
# it is the only place a tolerance applies. fp32, values of order 1.
ATOL, RTOL = 1e-5, 1e-5


def rule_inputs(seed: int, t: int, heads: int = HEADS) -> dict[str, torch.Tensor]:
    """Delta-rule inputs in the ranges the model produces: beta in (0,1), g <= 0."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    a_log = torch.rand(heads, generator=gen) * 16.0
    return {
        "q": randn(1, t, heads, K_DIM),
        "k": randn(1, t, heads, K_DIM),
        "v": randn(1, t, heads, V_DIM),
        "g": -a_log * torch.nn.functional.softplus(randn(1, t, heads)),
        "beta": randn(1, t, heads).sigmoid(),
        "rec": randn(1, heads, K_DIM, V_DIM) * 0.1,
    }


def run_rule(fn, args: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    rec = args["rec"].clone()
    out = fn(args["q"], args["k"], args["v"], args["g"], args["beta"], rec)
    return out, rec


# ---------------------------------------------------------------- 1. the derivation


@pytest.mark.parametrize("fn", FORMS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("t", LENGTHS)
def test_recurrence_is_block_diagonal_over_heads(fn, world: int, t: int) -> None:  # noqa: ANN001
    """Head slices run independently reproduce the full run bit for bit.

    This is the whole argument for head-parallel DeltaNet: no term of the update or the
    readout couples two head indices, so a rank that owns a subset of heads needs
    nothing from the other ranks at any timestep or chunk boundary.
    """
    args = rule_inputs(seed=t, t=t)
    out, rec = run_rule(fn, args)

    per_head = HEADS // world
    outs, recs = [], []
    for r in range(world):
        sl = slice(r * per_head, (r + 1) * per_head)
        shard = {
            "q": args["q"][:, :, sl].contiguous(),
            "k": args["k"][:, :, sl].contiguous(),
            "v": args["v"][:, :, sl].contiguous(),
            "g": args["g"][:, :, sl].contiguous(),
            "beta": args["beta"][:, :, sl].contiguous(),
            "rec": args["rec"][:, sl].contiguous(),
        }
        o, s = run_rule(fn, shard)
        outs.append(o)
        recs.append(s)

    assert torch.equal(torch.cat(outs, dim=2), out)
    assert torch.equal(torch.cat(recs, dim=1), rec)


@pytest.mark.parametrize("fn", FORMS, ids=lambda f: f.__name__)
def test_value_dim_split_is_also_communication_free(fn) -> None:  # noqa: ANN001
    """`v_dim` is never contracted, so splitting it is exact too (the fallback scheme).

    Not bit exact only because the reductions that *do* run (over `k_dim`) see a
    different trailing extent and so a different vectorization, not different math.
    Heads are preferred at TP=4 because a `v_dim` split leaves q, k, beta, g and half
    the conv replicated on every rank.
    """
    args = rule_inputs(seed=11, t=48, heads=8)
    out, rec = run_rule(fn, args)
    outs, recs, part = [], [], V_DIM // 4
    for r in range(4):
        sl = slice(r * part, (r + 1) * part)
        o, s = run_rule(
            fn, args | {"v": args["v"][..., sl].contiguous(), "rec": args["rec"][..., sl].clone()}
        )
        outs.append(o)
        recs.append(s)
    assert torch.allclose(torch.cat(outs, dim=-1), out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.cat(recs, dim=-1), rec, atol=1e-6, rtol=1e-6)


def test_key_dim_split_needs_a_collective() -> None:
    """A communication-free `k_dim` split is wrong, which is why the scheme rejects it.

    Both `sum_c S[c,b] k[c]` in the update and `sum_a S[a,b] q[a]` in the readout
    contract `k_dim`. Summing the shards' outputs fixes the readout but not the update,
    so the states diverge from the first token that has a nonzero delta: being exact
    would take an all_reduce inside the loop, per token.
    """
    args = rule_inputs(seed=5, t=32, heads=8)
    out, _ = run_rule(seed_model.delta_rule_recurrent, args)
    total, part = torch.zeros_like(out), K_DIM // 4
    for r in range(4):
        sl = slice(r * part, (r + 1) * part)
        shard = args | {
            "q": args["q"][..., sl].contiguous(),
            "k": args["k"][..., sl].contiguous(),
            "rec": args["rec"][:, :, sl].clone(),
        }
        total += run_rule(seed_model.delta_rule_recurrent, shard)[0]
    assert not torch.allclose(total, out, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------- 2. weight sharding


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fx.write_checkpoint(tmp_path_factory.mktemp("deltanet-tp-ckpt"))


@pytest.fixture(scope="module")
def reference(ckpt: Path) -> seed_model.Model:
    return seed_model.Model(ckpt, ["cpu"], torch.float32, max_seq=256, max_batch=2)


def shards(ckpt: Path, world: int) -> list[seed_model.Model]:
    """One `Model` per rank, each holding only its own DeltaNet heads.

    These models are sharded but not connected: the component methods return row-parallel
    partial sums and the tests combine them by hand, which is what `Model.layer` does with
    a real all_reduce. `test_deltanet_tp_gloo.py` runs the same thing through that.
    """
    return [
        seed_model.Model(
            ckpt, ["cpu"], torch.float32, max_seq=256, max_batch=2, tp=group(ckpt, r, world)
        )
        for r in range(world)
    ]


def merge_segments(parts: list[torch.Tensor], dim: int, key_dim: int, val_dim: int) -> torch.Tensor:
    """Inverse of `shard_qkv_segments`: q, k and v each reassembled across ranks."""
    split = [p.split([key_dim, key_dim, val_dim], dim=dim) for p in parts]
    return torch.cat([torch.cat([s[j] for s in split], dim=dim) for j in range(3)], dim=dim)


@pytest.mark.parametrize("world", WORLDS)
def test_local_cfg_divides_only_the_deltanet_heads(ckpt: Path, world: int) -> None:
    full = seed_model.load_cfg(ckpt)
    local = deltanet_tp.local_cfg(full, group(ckpt, 1, world))
    assert (local.k_heads, local.v_heads) == (fx.K_HEADS // world, fx.V_HEADS // world)
    assert local.v_heads // local.k_heads == full.v_heads // full.k_heads  # rep is preserved
    for field in ("hidden", "heads", "kv_heads", "head_dim", "k_dim", "v_dim", "experts"):
        assert getattr(local, field) == getattr(full, field)


def test_local_cfg_rejects_an_indivisible_world(ckpt: Path) -> None:
    full = seed_model.load_cfg(ckpt)
    with pytest.raises(ValueError, match="linear_num_key_heads"):
        deltanet_tp.local_cfg(
            full, tp_mod.TP(tp_mod.Plan(0, 3, *([tp_mod.Shard(0, 1)] * 6)), "cpu", lambda _x: None)
        )
    assert deltanet_tp.local_cfg(full, tp_mod.TP.single(full, "cpu")) is full


@pytest.mark.parametrize("world", WORLDS)
def test_sharded_weights_reassemble(ckpt: Path, world: int) -> None:
    """Concatenating the ranks' slices, per segment where the axis is segmented, is a no-op."""
    cfg = seed_model.load_cfg(ckpt)
    whole = seed_model.load_layer(
        seed_model.Checkpoint(ckpt),
        0,
        cfg,
        tp_mod.TP.single(cfg, "cpu"),
        torch.device("cpu"),
        torch.float32,
    )
    parts = [
        deltanet_tp.shard_deltanet_weights(whole, cfg, group(ckpt, r, world)) for r in range(world)
    ]

    key_dim, val_dim = cfg.k_heads * cfg.k_dim // world, cfg.v_heads * cfg.v_dim // world
    for name in ("in_proj_qkv", "conv"):
        merged = merge_segments([p[name] for p in parts], 0, key_dim, val_dim)
        assert torch.equal(merged, whole[name]), name
    for name in ("in_proj_z", "in_proj_a", "in_proj_b", "A_log", "dt_bias"):
        assert torch.equal(torch.cat([p[name] for p in parts], dim=0), whole[name]), name
    assert torch.equal(torch.cat([p["out_proj"] for p in parts], dim=1), whole["out_proj"])
    for p in parts:
        assert p["dn_norm"] is whole["dn_norm"]  # per value head, replicated
        assert p["in_proj_qkv"].shape[0] == whole["in_proj_qkv"].shape[0] // world


def test_a_flat_cut_of_the_qkv_axis_would_be_wrong(ckpt: Path) -> None:
    """Guards the per-segment slicing: the naive `narrow` of the whole axis differs."""
    cfg = seed_model.load_cfg(ckpt)
    whole = seed_model.load_layer(
        seed_model.Checkpoint(ckpt),
        0,
        cfg,
        tp_mod.TP.single(cfg, "cpu"),
        torch.device("cpu"),
        torch.float32,
    )
    handle = group(ckpt, 1, 4)
    assert not torch.equal(
        deltanet_tp.shard_qkv_segments(whole["in_proj_qkv"], cfg, handle, dim=0),
        handle.shard(whole["in_proj_qkv"], 0, "flat"),
    )


def test_non_deltanet_layers_pass_through(ckpt: Path) -> None:
    cfg = seed_model.load_cfg(ckpt)
    attn = seed_model.load_layer(
        seed_model.Checkpoint(ckpt),
        1,
        cfg,
        tp_mod.TP.single(cfg, "cpu"),
        torch.device("cpu"),
        torch.float32,
    )
    assert cfg.layer_types[1] == "full_attention"
    assert deltanet_tp.shard_deltanet_weights(attn, cfg, group(ckpt, 2, 4)) is attn


# ---------------------------------------------------------------- 3. the whole layer


@pytest.mark.parametrize("world", WORLDS)
def test_state_buffers_are_sharded(ckpt: Path, world: int) -> None:
    ref = seed_model.Model(ckpt, ["cpu"], torch.float32, max_seq=64, max_batch=2)
    one = shards(ckpt, world)[0]
    for name in ("rec", "conv"):
        want = list(ref.pool[0][name].shape)
        got = list(one.pool[0][name].shape)
        assert got == [want[0], want[1] // world, *want[2:]], name


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("t", LENGTHS)
def test_sharded_layer_sums_to_the_unsharded_layer(
    ckpt: Path, reference: seed_model.Model, world: int, t: int
) -> None:
    """Sum of the ranks' partial `out_proj` products equals the single-device layer."""
    x = fx.hidden_states(t, seed=t)
    reference.begin(0)
    want = reference.deltanet(0, x)

    ranks = shards(ckpt, world)
    partials = []
    for m in ranks:
        m.begin(0)
        partials.append(m.deltanet(0, x))
    got = torch.stack(partials).sum(0)
    err = (got - want).abs().max().item()
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), f"t={t} world={world} err {err:.3e}"

    # The conv is depthwise and `F.linear`/`exp` are width invariant, so the conv state
    # is bit identical. The recurrent state is only within an ulp, because `softplus`
    # and `sigmoid` round the narrower `g`/`beta` projections differently; see the
    # numerics note in `deltanet_tp`.
    cfg = ranks[0].cfg
    merged_conv = merge_segments(
        [m.state[0]["conv"] for m in ranks], 1, cfg.k_heads * cfg.k_dim, cfg.v_heads * cfg.v_dim
    )
    assert torch.equal(merged_conv, reference.state[0]["conv"])
    merged_rec = torch.cat([m.state[0]["rec"] for m in ranks], dim=1)
    assert torch.allclose(merged_rec, reference.state[0]["rec"], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("world", WORLDS)
def test_prefill_then_decode_carries_state(
    ckpt: Path, reference: seed_model.Model, world: int
) -> None:
    """Chunked prefill followed by single-token steps, sharded, tracks the unsharded run."""
    lengths = [70, 1, 1, 1, 33, 1]
    xs = [fx.hidden_states(n, seed=100 + j) for j, n in enumerate(lengths)]
    ranks = shards(ckpt, world)
    reference.begin(0)
    for m in ranks:
        m.begin(0)
    for j, x in enumerate(xs):
        want = reference.deltanet(0, x)
        got = torch.stack([m.deltanet(0, x) for m in ranks]).sum(0)
        err = (got - want).abs().max().item()
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), f"step {j} err {err:.3e}"
    merged_rec = torch.cat([m.state[0]["rec"] for m in ranks], dim=1)
    assert torch.allclose(merged_rec, reference.state[0]["rec"], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("world", WORLDS)
def test_batched_decode_is_sharded_the_same_way(
    ckpt: Path, reference: seed_model.Model, world: int
) -> None:
    """`deltanet_decode` over two slots, with one reduce for the whole batch."""
    slots = [0, 1]
    ranks = shards(ckpt, world)
    for m in (reference, *ranks):
        for s in slots:
            m.begin(s)
    for step in range(3):
        x = fx.hidden_states(1, seed=200 + step, batch=len(slots))
        want = reference.deltanet_decode(0, x, slots)
        got = torch.stack([m.deltanet_decode(0, x, slots) for m in ranks]).sum(0)
        err = (got - want).abs().max().item()
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), f"step {step} err {err:.3e}"


def test_single_group_is_the_unchanged_path(ckpt: Path, reference: seed_model.Model) -> None:
    """`TP.single()` must produce exactly the model the seed had before."""
    same = seed_model.Model(
        ckpt, ["cpu"], torch.float32, max_seq=256, max_batch=2, tp=group(ckpt, 0, 1)
    )
    assert same.cfg == reference.cfg
    x = fx.hidden_states(9, seed=7)
    same.begin(0)
    reference.begin(0)
    assert torch.equal(same.deltanet(0, x), reference.deltanet(0, x))
