"""The fused MXFP4 grouped-GEMV kernels against the dequantize-then-`bmm` path they replace.

The oracle is `mxfp4_gemv.reference_moe`, which is the arithmetic `Model._routed_grouped`
does: `mxfp4.dequant_mxfp4` into a dense tensor, two `bmm`, then the weighted combine. The
fused kernels are a reformulation of exactly that, not an approximation, so the bar is tight:
an fp32 oracle should agree to fp32 reduction-order noise, and the bf16 oracle (what the
engine ships today) to a bf16 ulp.

Two ways to run it, because Triton compiles per target architecture:

    # real kernels, needs an accelerator
    python -m pytest .../seed_tests/test_mxfp4_fused_gemv.py -p no:cacheprovider --no-cov

    # logic only, no GPU: Triton's reference interpreter, on CPU tensors
    TRITON_INTERPRET=1 /tmp/torchenv/bin/python -m pytest ... -p no:cacheprovider --no-cov

The interpreter validates indexing, masking, the nibble/e8m0 decode and the control flow, but
not codegen or performance, so a passing interpreter run is necessary and not sufficient. The
real-dimension cases are marked `slow_dims` and skipped under the interpreter, where a
4096-deep reduction over 512 experts would take hours.
"""

import itertools
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mxfp4_gemv  # noqa: E402
from mxfp4 import BLOCK, dequant_mxfp4  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.skipif(
    not mxfp4_gemv.HAVE_TRITON or not (torch.cuda.is_available() or INTERPRET),
    reason="needs triton plus either an accelerator or TRITON_INTERPRET=1",
)

# reference/config.json, text_config: the shapes the kernels actually ship at.
HIDDEN = 4096
MOE_INTERMEDIATE = 1024
NUM_EXPERTS = 512
TOP_K = 10


def random_experts(
    experts: int, hidden: int, intermediate: int, *, seed: int
) -> dict[str, torch.Tensor]:
    """MXFP4 payloads in the `[E, N, K // 2]` + `[E, N, K // 32]` layout `load_experts` makes.

    Scales are drawn near 2**-7 rather than over the whole e8m0 range: that is the magnitude
    a real checkpoint's blocks land at, and it keeps a 4096-deep reduction inside a range
    where a bf16 comparison still says something. `test_dequant_matches_torch_bit_exactly`
    covers the full byte range separately.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    kw = {"dtype": torch.uint8, "generator": g, "device": DEVICE}

    def payload(rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed = torch.randint(0, 256, (experts, rows, k // 2), **kw)
        scale = torch.randint(118, 123, (experts, rows, k // BLOCK), **kw)
        return packed, scale

    gate_up, gate_up_scale = payload(2 * intermediate, hidden)
    down, down_scale = payload(hidden, intermediate)
    return {
        "gate_up": gate_up,
        "gate_up_scale": gate_up_scale,
        "down": down,
        "down_scale": down_scale,
    }


def random_routing(
    tokens: int, experts: int, top_k: int, *, seed: int, zero_rows: tuple[int, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(a_expert, a_weight)` for `a = t * top_k + j`, shaped like `Model.moe`'s flattening.

    `zero_rows` blanks whole tokens' weights, which is how a captured step spells a batch row
    that is not in flight: the kernel has to skip those assignments rather than write garbage.
    """
    g = torch.Generator().manual_seed(seed)
    # topk returns distinct experts per token, so mirror that rather than sampling with
    # replacement: a token routed to the same expert twice is not a state the router reaches.
    picks = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(tokens)])
    weight = torch.rand(tokens, top_k, generator=g)
    weight = weight / weight.sum(-1, keepdim=True)
    for row in zero_rows:
        weight[row] = 0.0
    return picks.to(torch.int32).reshape(-1).to(DEVICE), weight.reshape(-1).to(DEVICE)


def shard_experts(ex: dict[str, torch.Tensor], span: tuple[int, int]) -> dict[str, torch.Tensor]:
    """The `[E_local, ...]` payload a rank owning global ids `span` would have loaded.

    Expert parallelism gives each rank only its own experts, so the kernel's local index is
    `e - expert_lo`. Handing it the full pool with a narrowed range would make it read the
    wrong experts, which is what makes this slice part of the contract rather than a detail.
    """
    lo, hi = span
    return {name: tensor[lo:hi] for name, tensor in ex.items()}


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """Max elementwise difference relative to the largest magnitude in the oracle."""
    scale = want.float().abs().max().item()
    return ((got.float() - want.float()).abs().max() / max(scale, 1e-30)).item()


# ---------------------------------------------------------------- the dequant, in isolation

if mxfp4_gemv.HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant_probe(
        wq_ptr,
        ws_ptr,
        out_ptr,
        rows,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PERM: tl.constexpr,
    ):
        """Write `_dequant_tile`'s output to memory so it can be diffed against torch.

        Test-only: the point of the kernels under test is that they never do this.
        """
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < rows
        for k0 in range(0, K, BLOCK_K):
            values, scale = mxfp4_gemv._dequant_tile(
                wq_ptr, ws_ptr, 0, 0, n, n_ok, k0, K, BLOCK_K, PERM
            )
            off = (
                k0
                + tl.arange(0, BLOCK_K // 32)[None, :, None] * 32
                + tl.arange(0, 32)[None, None, :]
            )
            tl.store(
                out_ptr + n[:, None, None] * K + off,
                values * scale[:, :, None],
                mask=n_ok[:, None, None],
            )


DECODE_PATHS = pytest.mark.parametrize(
    "perm",
    [
        pytest.param(False, id="bit-pattern"),
        pytest.param(
            True,
            id="perm-lut",
            marks=pytest.mark.skipif(
                not mxfp4_gemv.perm_lut(DEVICE), reason="needs the gfx9 byte-permute path"
            ),
        ),
    ],
)
"""Both fp4 decodes, so the portable fallback never rots and the assembly never ships alone."""


@DECODE_PATHS
@pytest.mark.parametrize("k", [32, 64, 128])
def test_dequant_matches_torch_bit_exactly(k: int, perm: bool) -> None:
    """In-register dequant vs `mxfp4.dequant_mxfp4`, over every byte and every live scale.

    Bit-exact, not close: both the fp4 magnitudes and the e8m0 scale are exactly representable
    in fp32, so any difference at all is a decode bug. That bar is what makes the byte-permute
    path safe to ship: it is a different way of writing the same 16 constants, not a different
    approximation of them.
    """
    rows = 37  # deliberately not a multiple of BLOCK_N, to exercise the row mask
    g = torch.Generator().manual_seed(k)
    packed = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=g).to(DEVICE)
    # 0 is excluded: torch gives the subnormal 2**-127 there and the kernel gives +0.0, which
    # mxfp4_gemv._e8m0 documents. 255 (the format's NaN) is included; both give +inf.
    scale = torch.randint(1, 256, (rows, k // BLOCK), dtype=torch.uint8, generator=g).to(DEVICE)
    got = torch.zeros(rows, k, dtype=torch.float32, device=DEVICE)
    _dequant_probe[(triton.cdiv(rows, 8),)](
        packed, scale, got, rows, K=k, BLOCK_N=8, BLOCK_K=min(k, 64), PERM=perm
    )
    want = dequant_mxfp4(packed, scale, torch.float32)
    # Scale byte 255 is the format's NaN and makes a zero nibble come back NaN from both
    # implementations, so the NaN positions are compared as positions and the rest by value.
    assert torch.equal(got.isnan(), want.isnan())
    finite = ~want.isnan()
    assert torch.equal(got[finite], want[finite])
    # The sign of an exact zero used to be a documented difference here: the original
    # four-case decode folded away the negation and returned +0.0 for fp4 code 8, where the
    # torch LUT has -0.0. Both decodes in the tree now reproduce -0.0, so this is a guard
    # against reintroducing it rather than a pinned difference. Either way it is inert: the
    # accumulators start at +0.0 and `v + 0.0 == v + -0.0` for every finite v.
    signs_differ = torch.signbit(got[finite]) != torch.signbit(want[finite])
    assert torch.equal(
        want[finite][signs_differ].abs(), torch.zeros_like(want[finite][signs_differ])
    ), "signs may only disagree on an exact zero"


@DECODE_PATHS
def test_dequant_covers_every_packed_byte_value(perm: bool) -> None:
    """All 256 byte values, laid out exhaustively rather than sampled.

    The test above draws its bytes at random, which covers the range in practice but does not
    promise to. This one enumerates it, which is the bar an inline-assembly decode has to
    meet: a byte-permute selector that is wrong for one nibble value would be a handful of
    wrong weights in a 4096-deep reduction, well inside the tolerance every other test here
    uses. The scale is pinned to 127 (a factor of exactly 1) so a failure names the decode
    and not the scale.
    """
    rows, k = 8, 64  # 8 * 32 bytes = every value once, and the last row exercises no mask
    packed = torch.arange(256, dtype=torch.uint8, device=DEVICE).reshape(rows, k // 2)
    scale = torch.full((rows, k // 32), 127, dtype=torch.uint8, device=DEVICE)
    got = torch.zeros(rows, k, dtype=torch.float32, device=DEVICE)
    _dequant_probe[(1,)](packed, scale, got, rows, K=k, BLOCK_N=8, BLOCK_K=64, PERM=perm)
    want = dequant_mxfp4(packed, scale, torch.float32)
    assert torch.equal(got, want)
    # Including the sign of zero: code 8 is -0.0 in `mxfp4._FP4_VALUES`, and both decodes
    # reproduce it, so no caveat is needed about which zero the kernel produces.
    assert torch.equal(torch.signbit(got), torch.signbit(want))


# ---------------------------------------------------------------- the fused MoE

SMALL = pytest.mark.parametrize(
    "tokens,experts,hidden,intermediate,top_k",
    [
        (1, 8, 128, 64, 3),  # single token, the decode-latency shape
        (4, 8, 128, 64, 3),
        (6, 16, 96, 32, 4),  # hidden not a multiple of BLOCK_N, K not a power of two
        (3, 5, 64, 32, 5),  # top_k == experts: every token hits every expert
    ],
)


@SMALL
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_matches_dequant_bmm_small(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int, seed: int
) -> None:
    """Fused kernels vs dequantize-then-`bmm`, in fp32 so only the reduction order differs."""
    ex = random_experts(experts, hidden, intermediate, seed=seed)
    routing = random_routing(tokens, experts, top_k, seed=seed + 100)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(seed)).to(DEVICE)
    want = mxfp4_gemv.reference_moe(
        x, ex, routing, top_k, (0, experts), dequant_mxfp4, torch.float32
    )
    got = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    assert got.shape == (tokens, hidden)
    assert relative_error(got, want) < 1e-5


@SMALL
def test_fused_drops_experts_this_rank_does_not_own(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """Expert parallelism: the out-of-range assignments must vanish inside the kernel.

    Also the property the whole design rests on: narrowing the range changes the *result*
    without changing the launch shape, so no host-side filtering is needed to express it.
    """
    ex = random_experts(experts, hidden, intermediate, seed=7)
    routing = random_routing(tokens, experts, top_k, seed=8)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(9)).to(DEVICE)
    shard = (experts // 3, 2 * experts // 3)
    local = shard_experts(ex, shard)
    want = mxfp4_gemv.reference_moe(x, local, routing, top_k, shard, dequant_mxfp4, torch.float32)
    got = mxfp4_gemv.fused_moe(x, local, routing, top_k, shard)
    assert relative_error(got, want) < 1e-5
    assert not torch.equal(got, mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))), (
        "a narrowed range must actually drop work"
    )

    # Summing every rank's partial has to give the unsharded answer back: that is the whole
    # contract of dropping unowned assignments rather than clamping them into range.
    cuts = [0, experts // 3, 2 * experts // 3, experts]
    parts = [
        mxfp4_gemv.fused_moe(x, shard_experts(ex, span), routing, top_k, span)
        for span in itertools.pairwise(cuts)
    ]
    whole = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    assert relative_error(sum(parts[1:], parts[0]), whole) < 1e-5


@SMALL
def test_fused_skips_zero_weight_rows(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """A padded, not-in-flight batch row has zero combine weights and must come back zero.

    This is the other half of what replaces host-side boolean indexing: a captured step
    always launches `max_batch * top_k` programs, and the rows that are not in this step's
    batch are turned off by their weights alone.
    """
    if tokens < 2:
        pytest.skip("needs at least one live row beside the blanked one")
    ex = random_experts(experts, hidden, intermediate, seed=11)
    routing = random_routing(tokens, experts, top_k, seed=12, zero_rows=(0,))
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(13)).to(DEVICE)
    got = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    assert torch.count_nonzero(got[0]) == 0, "a zero-weight token must contribute nothing"

    # And the live rows must be untouched by the blanked one, i.e. the same answer they get
    # when it is not there at all.
    want = mxfp4_gemv.reference_moe(
        x, ex, routing, top_k, (0, experts), dequant_mxfp4, torch.float32
    )
    assert relative_error(got[1:], want[1:]) < 1e-5


@SMALL
def test_expert_sorted_order_does_not_change_the_result(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """The L2-locality permutation reorders programs only, so it must be bit-identical.

    It is a permutation of the program-to-assignment mapping, not of the output layout; a
    difference here would mean a program is writing somewhere it should not.
    """
    ex = random_experts(experts, hidden, intermediate, seed=21)
    a_expert, a_weight = random_routing(tokens, experts, top_k, seed=22)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(23)).to(DEVICE)
    plain = mxfp4_gemv.fused_moe(x, ex, (a_expert, a_weight), top_k, (0, experts))
    order = mxfp4_gemv.expert_sorted_order(a_expert)
    sorted_run = mxfp4_gemv.fused_moe(x, ex, (a_expert, a_weight), top_k, (0, experts), order=order)
    assert torch.equal(plain, sorted_run)


@SMALL
def test_fused_is_deterministic_across_repeat_launches(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """No atomics in the combine, so two launches on the same inputs are bit-identical."""
    ex = random_experts(experts, hidden, intermediate, seed=31)
    routing = random_routing(tokens, experts, top_k, seed=32)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(33)).to(DEVICE)
    first = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    second = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    assert torch.equal(first, second)


@SMALL
def test_fused_honors_caller_supplied_output_and_scratch(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """`out`/`inter` are the decode hot path's preallocated buffers; using them must not
    change the answer, and `out` must be written in place rather than replaced."""
    ex = random_experts(experts, hidden, intermediate, seed=41)
    routing = random_routing(tokens, experts, top_k, seed=42)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(43)).to(DEVICE)
    want = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    out = torch.full((tokens, hidden), float("nan"), device=DEVICE)
    inter = torch.full((tokens * top_k, intermediate), float("nan"), device=DEVICE)
    got = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts), out=out, inter=inter)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out, want)


@SMALL
@pytest.mark.skipif(not mxfp4_gemv.perm_lut(DEVICE), reason="needs the gfx9 byte-permute path")
def test_both_decode_paths_agree_to_reduction_order(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int, monkeypatch
) -> None:
    """The two fp4 decodes hold the same 16 constants, so only the reduction order may differ.

    Bit-identical is the bar for the decode itself, and the two tests above hold it to that
    over all 256 byte values. It is deliberately *not* the bar here, because `v_perm_b32`
    works on whole registers and so the byte-permute path loads the payload a word at a
    time. At the model's own dimensions Triton gives the word-wide tile and the byte-wide
    tile the same lane layout and the whole MoE comes back bit-identical, which
    `test_both_decode_paths_are_bit_identical_at_model_dimensions` pins. At these small
    shapes it does not, `tl.sum` folds the same 32 products in a different order, and the
    answers differ by ~6e-8 relative in fp32: four orders of magnitude inside a bf16 ulp,
    with neither path closer to the fp32 oracle. Asserting equality here would be asserting
    a layout choice, so the distinction is kept visible instead: the decode did not change,
    the summation order did.
    """
    ex = random_experts(experts, hidden, intermediate, seed=71)
    routing = random_routing(tokens, experts, top_k, seed=72)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(73)).to(DEVICE)
    fast = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    grouped_fast = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts), grouped=True)
    monkeypatch.setenv("SEED_MOE_PERM_LUT", "0")
    assert not mxfp4_gemv.perm_lut(DEVICE)
    plain = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts))
    grouped_plain = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts), grouped=True)
    assert relative_error(fast, plain) < 1e-6
    assert relative_error(grouped_fast, grouped_plain) < 1e-6
    # Neither ordering is the better one; both have to stay on the fp32 oracle.
    exact = mxfp4_gemv.reference_moe(
        x, ex, routing, top_k, (0, experts), dequant_mxfp4, torch.float32
    )
    assert relative_error(fast, exact) < 1e-5
    assert relative_error(plain, exact) < 1e-5


def test_fused_rejects_routing_of_the_wrong_length() -> None:
    """The routing length is `tokens * top_k` by construction; a mismatch is a caller bug."""
    ex = random_experts(4, 64, 32, seed=51)
    a_expert, a_weight = random_routing(2, 4, 3, seed=52)
    x = torch.randn(3, 64).to(DEVICE)  # 3 tokens against a 2-token routing
    with pytest.raises(ValueError, match="routing must be"):
        mxfp4_gemv.fused_moe(x, ex, (a_expert, a_weight), 3, (0, 4))


def test_fused_rejects_non_contiguous_inputs() -> None:
    """Flat row-major addressing: a strided input would be read at the wrong offsets.

    Caught rather than copied, because a silent `.contiguous()` on the decode hot path is a
    bandwidth bug that would never show up as a wrong answer.
    """
    ex = random_experts(4, 64, 32, seed=61)
    routing = random_routing(2, 4, 3, seed=62)
    wide = torch.randn(2, 128).to(DEVICE)
    with pytest.raises(ValueError, match="x must be contiguous"):
        mxfp4_gemv.fused_moe(wide[:, ::2], ex, routing, 3, (0, 4))


# ---------------------------------------------------------------- the de-duplicating path


@pytest.mark.parametrize("assignments,experts,block_m", [(30, 4, 16), (12, 3, 16), (7, 2, 32)])
def test_align_blocks_places_every_assignment_under_its_own_expert(
    assignments: int, experts: int, block_m: int
) -> None:
    """The grouping is only sound if each block holds one expert's assignments and no others.

    Pure torch, so it runs anywhere. This is the half of the de-duplicating path that does not
    need a GPU, and it is the half where an indexing mistake would silently compute the wrong
    expert's weights for a token rather than crash.
    """
    g = torch.Generator().manual_seed(assignments)
    a_expert = torch.randint(0, experts, (assignments,), dtype=torch.int32, generator=g)
    a_weight = torch.rand(assignments, generator=g)
    a_weight[::5] = 0.0  # dropped assignments must land in the sentinel bucket
    slot, block_expert = mxfp4_gemv.align_blocks(a_expert, a_weight, (0, experts), block_m)

    assert block_expert.numel() == mxfp4_gemv.grouped_block_count(assignments, experts, block_m)
    assert slot.numel() == block_expert.numel() * block_m
    placed = sorted(slot[slot >= 0].tolist())
    assert placed == list(range(assignments)), "every assignment must be placed exactly once"

    live = a_weight != 0
    for block in range(block_expert.numel()):
        expert = int(block_expert[block])
        rows = slot[block * block_m : (block + 1) * block_m]
        for row in rows[rows >= 0].tolist():
            if live[row]:
                assert int(a_expert[row]) == expert
            else:
                assert expert == experts, "a dropped assignment belongs to the sentinel bucket"


@SMALL
@pytest.mark.parametrize("seed", [0, 1])
def test_grouped_matches_the_per_assignment_kernels(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int, seed: int
) -> None:
    """Both spellings compute the same thing; only their weight traffic differs.

    Run against the fp32 oracle rather than against each other, so a shared bug in the two
    kernels cannot hide.
    """
    ex = random_experts(experts, hidden, intermediate, seed=seed + 80)
    routing = random_routing(tokens, experts, top_k, seed=seed + 90)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(seed)).to(DEVICE)
    want = mxfp4_gemv.reference_moe(
        x, ex, routing, top_k, (0, experts), dequant_mxfp4, torch.float32
    )
    got = mxfp4_gemv.fused_moe(x, ex, routing, top_k, (0, experts), grouped=True)
    assert relative_error(got, want) < 1e-4


@SMALL
def test_grouped_drops_unowned_experts_and_zero_weight_rows(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> None:
    """The sentinel bucket has to behave exactly like the per-assignment early return."""
    ex = random_experts(experts, hidden, intermediate, seed=81)
    routing = random_routing(tokens, experts, top_k, seed=91, zero_rows=(0,))
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(92)).to(DEVICE)
    shard = (experts // 3, 2 * experts // 3)
    local = shard_experts(ex, shard)
    want = mxfp4_gemv.reference_moe(x, local, routing, top_k, shard, dequant_mxfp4, torch.float32)
    got = mxfp4_gemv.fused_moe(x, local, routing, top_k, shard, grouped=True)
    assert relative_error(got, want) < 1e-4
    assert torch.count_nonzero(got[0]) == 0, "a zero-weight token must contribute nothing"


def test_use_grouped_picks_the_deduplicating_path_only_when_it_pays() -> None:
    """The selection rule is the whole reason both kernels exist; pin its two real cases."""
    # A 512-token prefill chunk: 5,120 assignments over 512 experts, ten deep.
    assert mxfp4_gemv.use_grouped(512 * TOP_K, NUM_EXPERTS)
    # A 48-token decode step over the same experts: almost no repeats to exploit.
    assert not mxfp4_gemv.use_grouped(48 * TOP_K, NUM_EXPERTS)
    # The same 480 assignments read against a TP=4 rank's 128-expert slice clear the
    # threshold, which is why the denominator has to be the whole axis and not the shard.
    assert mxfp4_gemv.use_grouped(48 * TOP_K, NUM_EXPERTS // 4)


def test_expert_parallel_shard_selects_the_same_kernels_as_one_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank owning a quarter of the experts must not read its shard as four times the depth.

    Expert parallelism hands every rank the *whole* assignment list and drops the out-of-range
    ones GPU-side, so `assignments / experts` is a density only when `experts` is the whole
    axis. Reading it against the rank's slice quadruples the apparent depth at TP=4 and
    selects the de-duplicating kernels for a decode step they lose on, which measured 23.5 ms
    of a 188 ms step (`TP_BYTELUT_BOTTLENECK_2026-09-22.md` section 3).
    """
    experts, hidden, intermediate, top_k, tokens = 16, 96, 32, 5, 8
    ex = random_experts(experts, hidden, intermediate, seed=11)
    routing = random_routing(tokens, experts, top_k, seed=12)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(13)).to(DEVICE)
    span = (experts // 4, experts // 2)  # this rank's quarter, as `load_experts` would slice it
    local = shard_experts(ex, span)
    # Decode-shaped over the whole axis, grouped-shaped over the shard: the case that differs.
    assert not mxfp4_gemv.use_grouped(tokens * top_k, experts)
    assert mxfp4_gemv.use_grouped(tokens * top_k, span[1] - span[0])

    monkeypatch.setattr(
        mxfp4_gemv,
        "_grouped_moe",
        lambda *a, **kw: pytest.fail("a decode-density step took the de-duplicating kernels"),
    )
    got = mxfp4_gemv.fused_moe(x, local, routing, top_k, span, total_experts=experts)
    want = mxfp4_gemv.reference_moe(x, local, routing, top_k, span, dequant_mxfp4, torch.float32)
    assert relative_error(got, want) < 1e-4


# ---------------------------------------------------------------- real model dimensions

slow_dims = pytest.mark.skipif(
    INTERPRET, reason="the Triton interpreter is far too slow for a 4096-deep reduction"
)


@pytest.fixture(scope="module")
def model_dim_experts() -> dict[str, torch.Tensor]:
    """The real expert payload, built once: 3.2 GB of MXFP4 at 512 experts."""
    return random_experts(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE, seed=60)


@slow_dims
@pytest.mark.parametrize("tokens,seed", [(1, 0), (8, 1), (48, 2)])
def test_fused_matches_dequant_bmm_at_model_dimensions(
    model_dim_experts: dict[str, torch.Tensor], tokens: int, seed: int
) -> None:
    """The shapes the engine actually runs: hidden 4096, intermediate 1024, top-10 of 512.

    Batch 1 is the single-stream decode shape and 48 is the benchmark's concurrency. The
    expert count is the real 512 even though only `tokens * top_k` of them are touched,
    because a smaller pool would not exercise the 2**31-byte offset the gate_up payload
    reaches and so would not catch a 32-bit overflow in the addressing.
    """
    ex = model_dim_experts
    routing = random_routing(tokens, NUM_EXPERTS, TOP_K, seed=seed + 300)
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(seed))
    x = x.to(DEVICE).to(torch.bfloat16)

    got = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS))
    exact = mxfp4_gemv.reference_moe(
        x, ex, routing, TOP_K, (0, NUM_EXPERTS), dequant_mxfp4, torch.float32
    )
    engine = mxfp4_gemv.reference_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS), dequant_mxfp4)
    # The fp32 oracle is the closest thing to a ground truth available here: the fused path
    # differs from it only by rounding its gated intermediate to bf16, which the `bmm` path
    # does too. A few bf16 ulp over a 4096-deep reduction.
    fused_error = relative_error(got, exact)
    assert fused_error < 1e-2
    # The bf16 `bmm` path the engine ships today rounds in more places (both GEMM outputs and
    # the combine), so the fused kernel must be at least as close to the fp32 answer as it is.
    # Asserting a tolerance between the two bf16 results instead would just be measuring which
    # of two roundings is noisier, and at 48 tokens that exceeds one ulp in both directions.
    assert fused_error <= relative_error(engine, exact)


@slow_dims
@pytest.mark.parametrize("tokens", [64, 128])
def test_grouped_matches_dequant_bmm_at_prefill_dimensions(
    model_dim_experts: dict[str, torch.Tensor], tokens: int
) -> None:
    """The de-duplicating path at the shapes it is chosen for, against the fp32 oracle.

    The small-dimension test above cannot reach the case this path exists for: several
    assignments landing in one `BLOCK_M` block and sharing a weight tile. At 64 tokens and
    top-10 over 512 experts that starts happening, and the block table stops being one
    assignment per block.
    """
    ex = model_dim_experts
    routing = random_routing(tokens, NUM_EXPERTS, TOP_K, seed=tokens + 400)
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    x = x.to(DEVICE).to(torch.bfloat16)

    exact = mxfp4_gemv.reference_moe(
        x, ex, routing, TOP_K, (0, NUM_EXPERTS), dequant_mxfp4, torch.float32
    )
    grouped = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS), grouped=True)
    single = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS), grouped=False)
    assert relative_error(grouped, exact) < 1e-2
    # Not bit-identical: the two spellings reduce in a different order, and the grouped one
    # combines in torch rather than in the kernel. They must agree to bf16 rounding, though.
    assert relative_error(grouped, single) < 1e-2


@slow_dims
@pytest.mark.skipif(not mxfp4_gemv.perm_lut(DEVICE), reason="needs the gfx9 byte-permute path")
@pytest.mark.parametrize("tokens", [1, 48])
def test_both_decode_paths_are_bit_identical_at_model_dimensions(
    model_dim_experts: dict[str, torch.Tensor], tokens: int, monkeypatch
) -> None:
    """At the shapes the engine ships, the byte-permute decode changes nothing at all.

    This is the property that matters for the accuracy gate: greedy decoding turns any logit
    perturbation into a token flip at the first near-tie, so a decode rewrite that moved the
    result even by a bf16 ulp would be indistinguishable from a regression. Here it does not
    move it, at batch 1 and at the benchmark's batch 48, over 4096-deep and 1024-deep
    reductions. The small-shape test above explains why this is a claim about these
    dimensions rather than about the kernel in general.
    """
    ex = model_dim_experts
    routing = random_routing(tokens, NUM_EXPERTS, TOP_K, seed=tokens + 500)
    x = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens))
    x = x.to(DEVICE).to(torch.bfloat16)
    fast = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS)).clone()
    monkeypatch.setenv("SEED_MOE_PERM_LUT", "0")
    plain = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, (0, NUM_EXPERTS))
    assert torch.equal(fast, plain)


@slow_dims
@pytest.mark.parametrize("shard", [(0, 128), (384, 512)])
def test_fused_expert_parallel_shards_at_model_dimensions(
    model_dim_experts: dict[str, torch.Tensor], shard: tuple[int, int]
) -> None:
    """One MI300A rank owns 128 of the 512 experts; its partial must be the oracle's."""
    ex = shard_experts(model_dim_experts, shard)
    routing = random_routing(16, NUM_EXPERTS, TOP_K, seed=400)
    x = torch.randn(16, HIDDEN, generator=torch.Generator().manual_seed(401))
    x = x.to(DEVICE).to(torch.bfloat16)
    got = mxfp4_gemv.fused_moe(x, ex, routing, TOP_K, shard)
    want = mxfp4_gemv.reference_moe(x, ex, routing, TOP_K, shard, dequant_mxfp4, torch.float32)
    assert relative_error(got, want) < 8e-3
    # Only the assignments this rank owns are in the answer, so it must differ from the
    # answer a rank owning everything would produce.
    assert got.abs().sum() > 0


@pytest.mark.parametrize(
    ("tokens", "hidden", "intermediate", "span", "pile_up"),
    [(12, 256, 128, (8, 16), False), (20, 512, 128, (0, 8), True), (5, 1024, 128, (4, 8), False)],
)
def test_fused_moe_bw_matches_reference(tokens, hidden, intermediate, span, pile_up):
    """`SEED_MOE_BW`: the folded-scale decode, the permuted K order, multi-unit experts (more
    than `BW_BLOCK_T` tokens on one expert), dropped and inactive rows, against the fp32 oracle.
    Its intermediate is rounded to bf16 like the shipped kernels', hence the same bar."""
    lo, hi = span
    ex = random_experts(hi - lo, hidden, intermediate, seed=7)
    a_expert, a_weight = random_routing(tokens, 2 * hi, TOP_K // 2, seed=3, zero_rows=(0,))
    if pile_up:
        a_expert[: tokens * TOP_K // 4] = lo + 1
    x = (
        torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(5))
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    routing = (a_expert.contiguous(), a_weight.to(torch.bfloat16).contiguous())
    got = mxfp4_gemv.fused_moe_bw(x, ex, routing, TOP_K // 2, span)
    want = mxfp4_gemv.reference_moe(x, ex, routing, TOP_K // 2, span, dequant_mxfp4, torch.float32)
    assert relative_error(got, want) < 2e-2
