"""Hermetic CPU tests for the tensor-parallel sharding, on the tiny random checkpoint.

What these can and cannot show. The risk in this change is the sharding arithmetic: which
rows of which weight a rank keeps, and whether summing or concatenating the ranks' outputs
reproduces the unsharded computation. That is what is tested here, in three layers:

1. `Plan` arithmetic, including the GQA case where there are fewer KV heads than ranks.
2. Weight slicing and output composition with no process group at all: build four real
   sharded `Model`s, run a component on each, and sum the partials by hand. Fast to run and
   sharp about which component is wrong.
3. The whole forward under a real `torch.distributed` process group on the gloo backend:
   four processes, real `init_process_group`, real `all_reduce` and `broadcast`, driven
   through `tp_driver` exactly as `server.py` drives them, against the same sequence decoded
   by the unsharded single-process model.

What they do not show: nothing here runs RCCL, a GPU, or more than one device. Layer 3
exercises the API surface and the protocol, so a rank/world-size or ordering mistake shows
up, but gloo and RCCL are different implementations and neither the ROCm build's backend
selection nor the Slurm/pyxis rendezvous is covered. Speed is not covered at all.

These run in fp32, where the row-parallel residual is about 1e-6. They therefore do *not*
bound the bf16 case the server deploys, which is why `tp.all_reduce` reduces in fp32 and why
the accuracy gate's greedy-token pins are the real check; see "Numerics" in tp.py.

    <python-with-torch> -m pytest seed_tests/test_tensor_parallel.py -o addopts= -p no:cacheprovider
"""

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402
import tp  # noqa: E402
from test_seed_parity import (  # noqa: E402
    VOCAB,
    build_hf,
    quantize_hf_in_place,
    tiny_config,
    write_checkpoint,  # noqa: E402
)
from tp_gloo_rank import MAX_SEQ, PREFILL_CHUNK, greedy  # noqa: E402

WORLD = 4

TP_AXES = {
    # The default tiny config has 2 key heads and 4 query heads, which do not divide by 4.
    # These do, and they keep the GQA ratio that forces KV replication (8 query, 2 KV).
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tp")
    write_checkpoint(build_hf(cfg=tiny_config(**TP_AXES)), out, mxfp4=False)
    return out


@pytest.fixture(scope="module")
def mxfp4_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tp_mxfp4")
    hf = build_hf(cfg=tiny_config(**TP_AXES))
    quantize_hf_in_place(hf)
    write_checkpoint(hf, out, mxfp4=True)
    return out


# ---------------------------------------------------------------- plan arithmetic


def test_query_heads_split_evenly_and_cover_the_axis() -> None:
    shards = [tp.even(32, r, WORLD, "heads") for r in range(WORLD)]
    assert [(s.start, s.count) for s in shards] == [(0, 8), (8, 8), (16, 8), (24, 8)]
    assert [s.rows(256) for s in shards] == [slice(i * 2048, (i + 1) * 2048) for i in range(WORLD)]


def test_kv_heads_are_replicated_when_there_are_fewer_than_ranks() -> None:
    """The real model: 32 query heads, 2 KV heads, TP=4. Ranks 0-1 read KV head 0, 2-3 head 1."""
    shards = [tp.kv_shard(32, 2, r, WORLD) for r in range(WORLD)]
    assert [(s.start, s.count) for s in shards] == [(0, 1), (0, 1), (1, 1), (1, 1)]
    # every local query head must map to the one local KV head under the global GQA grouping
    rep = 32 // 2
    for rank, shard in enumerate(shards):
        block = range(rank * 8, rank * 8 + 8)
        assert {j // rep for j in block} == {shard.start}


def test_kv_heads_split_evenly_when_there_are_at_least_as_many_as_ranks() -> None:
    shards = [tp.kv_shard(32, 8, r, WORLD) for r in range(WORLD)]
    assert [(s.start, s.count) for s in shards] == [(0, 2), (2, 2), (4, 2), (6, 2)]


def test_a_query_block_straddling_two_kv_groups_is_rejected() -> None:
    """4 query heads, 2 KV heads, 2 ranks is fine; 4 query heads and 4 ranks with 3 KV is not."""
    with pytest.raises(ValueError, match="across"):
        tp.kv_shard(4, 3, 0, WORLD)


def test_plan_names_the_config_field_that_does_not_divide(checkpoint: Path) -> None:
    """The real tiny config has 2 key heads, which do not divide by 4; the error says so."""
    cfg = seed_model.load_cfg(checkpoint)
    with pytest.raises(ValueError, match="linear_num_key_heads=4"):
        tp.plan(cfg, 0, 8)


def test_plan_rejects_an_indivisible_expert_count() -> None:
    with pytest.raises(ValueError, match="num_experts=10"):
        tp.even(10, 0, WORLD, "num_experts")


def test_plan_rejects_a_rank_outside_the_group(checkpoint: Path) -> None:
    cfg = seed_model.load_cfg(checkpoint)
    with pytest.raises(ValueError, match="rank 4"):
        tp.plan(cfg, 4, WORLD)


def test_single_is_a_world_of_one_with_no_reduce(checkpoint: Path) -> None:
    handle = tp.TP.single(seed_model.load_cfg(checkpoint), "cpu")
    assert (handle.world, handle.rank) == (1, 0)
    x = torch.ones(3)
    assert handle.all_reduce(x).tolist() == [1.0, 1.0, 1.0]  # untouched


def test_a_multi_rank_handle_without_a_reduce_is_rejected(checkpoint: Path) -> None:
    plan = tp.plan(seed_model.load_cfg(checkpoint), 0, WORLD)
    with pytest.raises(ValueError, match="needs an all-reduce"):
        tp.TP(plan, "cpu", None)


def test_the_generic_split_helpers_agree_with_the_plan(checkpoint: Path) -> None:
    """`split_size`/`shard` are the surface the DeltaNet sharding binds to; they must give
    the same slices the plan does, or the two halves would disagree."""
    cfg = seed_model.load_cfg(checkpoint)
    t = torch.arange(cfg.experts * 3).reshape(cfg.experts, 3)
    for rank in range(WORLD):
        handle = tp.TP(tp.plan(cfg, rank, WORLD), "cpu", lambda _x: None)
        assert (handle.world_size, handle.enabled) == (WORLD, True)
        assert handle.split_size(cfg.experts, "num_experts") == handle.plan.experts.count
        torch.testing.assert_close(
            handle.shard(t, 0, "num_experts"),
            t[handle.plan.experts.start : handle.plan.experts.stop],
        )
    single = tp.TP.single(cfg, "cpu")
    assert (single.world_size, single.enabled) == (1, False)


def test_the_deltanet_axes_of_the_plan_agree_with_the_generic_helpers(checkpoint: Path) -> None:
    """`deltanet_tp` shards through `split_size`/`shard`; the plan names the same two cuts.

    Two spellings of one fact, so they have to be pinned against each other: `Plan.dn_k` and
    `Plan.dn_v` document the DeltaNet split in `tp.py`'s table, while `deltanet_tp` takes it
    with the generic helpers.
    """
    cfg = seed_model.load_cfg(checkpoint)
    for rank in range(WORLD):
        handle = tp.TP(tp.plan(cfg, rank, WORLD), "cpu", lambda _x: None)
        assert handle.split_size(cfg.k_heads, "k") == handle.plan.dn_k.count
        assert handle.split_size(cfg.v_heads, "v") == handle.plan.dn_v.count
        # `shard` cuts a tensor at rank * count, which is where the plan's shard starts.
        t = torch.arange(cfg.v_heads)
        assert handle.shard(t, 0, "v")[0].item() == handle.plan.dn_v.start


def test_a_tensor_parallel_rank_may_not_also_split_its_layers_over_devices(
    checkpoint: Path,
) -> None:
    """The two parallelism axes are exclusive; combining them is rejected, not ignored.

    A layer split runs stages on separate threads, which would issue two ranks' all-reduces
    in different orders and hang the group.
    """
    cfg = seed_model.load_cfg(checkpoint)
    handle = tp.TP(tp.plan(cfg, 0, WORLD), "cpu", lambda _x: None)
    with pytest.raises(ValueError, match="a rank owns one device"):
        seed_model.Model(checkpoint, ["cpu", "cpu"], torch.float32, MAX_SEQ, 1, tp=handle)


def test_a_tensor_parallel_rank_has_nothing_to_pipeline(checkpoint: Path) -> None:
    """One device means one stage, so `_build_pipeline` declines and no stage thread exists.

    The pipeline is the thing TP replaces (DECODE_BOTTLENECK_2026-09-22.md section 3), and a
    stage thread issuing collectives is exactly what the exclusivity check above prevents; this
    pins that the remaining single-rank path does not start one by accident.
    """
    for model in ranks_of(checkpoint, max_batch=2):
        assert len(model.stages) == 1
        assert model.pipeline is None
        assert model.microbatches == 1


def test_the_all_reduce_of_a_bf16_partial_runs_in_fp32(checkpoint: Path) -> None:
    """The accuracy mitigation, pinned: a bf16 partial is summed exactly, then rounded once.

    The four values below each round to a different bf16, and their exact sum is not the sum
    of any pairwise-rounded ordering, so reducing in bf16 gives a different answer. Casting
    back is the only rounding the sharding is allowed to add on top of the partials
    themselves; see "Numerics" in tp.py for why the wire cost is worth it.
    """
    parts = [
        torch.tensor([v], dtype=torch.bfloat16) for v in (1.0, 0.0078125, 0.0078125, 0.0078125)
    ]
    exact = sum(p.double() for p in parts)

    def reduce(x: torch.Tensor) -> None:
        assert x.dtype is torch.float32, "the collective must see fp32, not the activation dtype"
        x.copy_(sum(p.to(x.dtype) for p in parts))

    handle = tp.TP(tp.plan(seed_model.load_cfg(checkpoint), 0, WORLD), "cpu", reduce)
    got = handle.all_reduce(parts[0].clone())
    assert got.dtype is torch.bfloat16
    assert got.item() == exact.to(torch.bfloat16).item()


def test_from_torch_distributed_without_a_group_is_the_single_handle(checkpoint: Path) -> None:
    handle = tp.TP.from_torch_distributed(seed_model.load_cfg(checkpoint), "cpu")
    assert (handle.rank, handle.world) == (0, 1)
    assert not handle.enabled


# ---------------------------------------------------------------- shards, by hand


def ranks_of(checkpoint: Path, max_batch: int = 1) -> list[seed_model.Model]:
    """The four sharded models, with no all-reduce: every component returns its raw partial."""
    cfg = seed_model.load_cfg(checkpoint)
    return [
        seed_model.Model(
            checkpoint,
            ["cpu"],
            torch.float32,
            MAX_SEQ,
            max_batch,
            tp=tp.TP(tp.plan(cfg, r, WORLD), "cpu", lambda _x: None),
        )
        for r in range(WORLD)
    ]


def unsharded(checkpoint: Path, max_batch: int = 1) -> seed_model.Model:
    return seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch)


def attention_layer(model: seed_model.Model) -> int:
    return model.cfg.layer_types.index("full_attention")


def test_query_projection_shards_concatenate_to_the_unsharded_weight(checkpoint: Path) -> None:
    """The load-time slice, checked directly: rank r keeps rows [r*2*2*head_dim, ...).

    q_proj emits the query and this model's output gate per head, so a wrong stride here
    would silently pair a head's query with another head's gate.
    """
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i = attention_layer(whole)
    got = torch.cat([m.layers[i]["q_proj"] for m in parts])
    torch.testing.assert_close(got, whole.layers[i]["q_proj"])
    torch.testing.assert_close(
        torch.cat([m.layers[i]["o_proj"] for m in parts], dim=1), whole.layers[i]["o_proj"]
    )


def test_kv_projection_shards_are_the_head_each_rank_reads(checkpoint: Path) -> None:
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i, d = attention_layer(whole), whole.cfg.head_dim
    for rank, m in enumerate(parts):
        head = rank // (WORLD // whole.cfg.kv_heads)
        for name in ("k_proj", "v_proj"):
            want = whole.layers[i][name][head * d : (head + 1) * d]
            torch.testing.assert_close(m.layers[i][name], want, msg=f"{name} rank {rank}")
        assert m.pool[i]["k"].shape[1] == 1  # one KV head resident, not two


def test_expert_shards_partition_the_expert_axis(checkpoint: Path) -> None:
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    per = whole.cfg.experts // WORLD
    for rank, m in enumerate(parts):
        ex, want = m.layers[0]["experts"], whole.layers[0]["experts"]
        assert ex["gate_up"].shape[0] == per
        assert m.expert_range == (rank * per, (rank + 1) * per)
        torch.testing.assert_close(ex["gate_up"], want["gate_up"][rank * per : (rank + 1) * per])
        torch.testing.assert_close(ex["down"], want["down"][rank * per : (rank + 1) * per])


def test_the_unsharded_expert_range_is_the_whole_axis(checkpoint: Path) -> None:
    """What `_routed_grouped` and `mxfp4_gemv.fused_moe` drop against; nothing at world 1."""
    whole = unsharded(checkpoint)
    assert whole.expert_range == (0, whole.cfg.experts)


def test_shared_expert_shards_split_the_intermediate_axis(checkpoint: Path) -> None:
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    per = whole.cfg.shared_inter // WORLD
    for rank, m in enumerate(parts):
        lo, hi = rank * per, (rank + 1) * per
        for name in ("shared_expert.gate_proj", "shared_expert.up_proj"):
            torch.testing.assert_close(m.layers[0][name], whole.layers[0][name][lo:hi])
        torch.testing.assert_close(
            m.layers[0]["shared_expert.down_proj"],
            whole.layers[0]["shared_expert.down_proj"][:, lo:hi],
        )
        torch.testing.assert_close(m.layers[0]["router"], whole.layers[0]["router"])


@pytest.mark.parametrize("tokens", [1, 7])
def test_attention_partials_sum_to_the_unsharded_output(checkpoint: Path, tokens: int) -> None:
    """Head-parallel attention: `o_proj` is row-parallel, so the four partials add up.

    This is the all-reduce done by hand. It says the shards are the right rows and that
    summing is the right composition; it says nothing about RCCL.
    """
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i = attention_layer(whole)
    x = torch.randn(1, tokens, whole.cfg.hidden, generator=torch.Generator().manual_seed(7)) * 0.1

    want = whole.full_attention(i, x, 0)
    got = sum(m.full_attention(i, x, 0) for m in parts)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_attention_partials_sum_when_resuming_after_a_cached_prefix(checkpoint: Path) -> None:
    """The `start > 0` branch, which takes `prefill_attention` rather than `is_causal`.

    That helper folds the query head group into the query length and repeats the mask to
    match; the fold is `heads // kv_heads` *of this rank's tensors*, 2 // 1 here against
    8 // 2 unsharded, so a sharded run exercises a different reshape than any unsharded test
    does. It feeds `p95_ttft_turn2plus_ms`, which is the objective's metric.
    """
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i = attention_layer(whole)
    gen = torch.Generator().manual_seed(21)
    prefix = torch.randn(1, 5, whole.cfg.hidden, generator=gen) * 0.1
    rest = torch.randn(1, 3, whole.cfg.hidden, generator=gen) * 0.1

    whole.full_attention(i, prefix, 0)
    want = whole.full_attention(i, rest, 5)
    for m in parts:
        m.full_attention(i, prefix, 0)
    got = sum(m.full_attention(i, rest, 5) for m in parts)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", [1, 7])
def test_attention_kv_caches_hold_the_same_entries_as_the_unsharded_cache(
    checkpoint: Path, tokens: int
) -> None:
    """Each rank's KV pool must hold exactly its own head's rows of the unsharded cache."""
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i, d = attention_layer(whole), whole.cfg.head_dim
    x = torch.randn(1, tokens, whole.cfg.hidden, generator=torch.Generator().manual_seed(8)) * 0.1
    whole.full_attention(i, x, 0)
    whole_rows = whole.block_tables[0].physical_rows(tokens, whole.block_size)
    whole_kv = {name: whole.pool[i][name][whole_rows] for name in ("k", "v")}
    for rank, m in enumerate(parts):
        m.full_attention(i, x, 0)
        head = rank // (WORLD // whole.cfg.kv_heads)
        rows = m.block_tables[0].physical_rows(tokens, m.block_size)
        for name in ("k", "v"):
            torch.testing.assert_close(
                m.pool[i][name][rows],
                whole_kv[name][:, head : head + 1],
                msg=f"{name} rank {rank}",
            )
        assert d == m.pool[i]["k"].shape[-1]


@pytest.mark.parametrize("uniform", [True, False])
def test_batched_decode_attention_partials_sum_to_the_unsharded_output(
    checkpoint: Path, uniform: bool
) -> None:
    """`attn_decode`, the batched decode path, which the old TP branch predates entirely.

    Both of its shapes: every slot at the same position (the uniform case, no mask, the
    SDPA call) and slots at different positions (the masked, written-out call). Each reads
    this rank's one KV head and folds its 2 query heads into the query length, so getting the
    head counts wrong here would not raise, it would silently attend with the wrong grouping.
    """
    whole, parts = unsharded(checkpoint, max_batch=3), ranks_of(checkpoint, max_batch=3)
    i = attention_layer(whole)
    gen = torch.Generator().manual_seed(23)
    slots, positions = [0, 1, 2], [4, 4, 4] if uniform else [4, 2, 7]

    for model in (whole, *parts):  # give every slot a prefix to attend over
        for step in range(8):
            prime = torch.randn(
                3, 1, whole.cfg.hidden, generator=torch.Generator().manual_seed(step)
            )
            model.attn_decode(i, prime * 0.1, slots, [step] * 3)

    x = torch.randn(3, 1, whole.cfg.hidden, generator=gen) * 0.1
    want = whole.attn_decode(i, x, slots, positions)
    got = sum(m.attn_decode(i, x, slots, positions) for m in parts)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", [1, 3, 19])
def test_moe_partials_sum_to_the_unsharded_output(checkpoint: Path, tokens: int) -> None:
    """Expert-parallel MoE: each rank contributes its own experts, so the partials add up.

    `tokens=1` with top_k 3 over 8 experts leaves at least one rank with no selected expert,
    which is the empty branch in `Model._routed_grouped`.
    """
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    x = torch.randn(1, tokens, whole.cfg.hidden, generator=torch.Generator().manual_seed(9)) * 0.1

    want = whole.moe(0, x)
    got = sum(m.moe(0, x) for m in parts)

    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_moe_partials_sum_under_mxfp4_experts(mxfp4_checkpoint: Path) -> None:
    """The expert axis is sliced, not the MXFP4-blocked K axis, so dequant is unaffected."""
    whole, parts = unsharded(mxfp4_checkpoint), ranks_of(mxfp4_checkpoint)
    x = torch.randn(1, 11, whole.cfg.hidden, generator=torch.Generator().manual_seed(10)) * 0.1
    torch.testing.assert_close(
        sum(m.moe(0, x) for m in parts), whole.moe(0, x), atol=1e-5, rtol=1e-5
    )


def test_at_least_one_rank_sees_no_expert_at_a_single_token(checkpoint: Path) -> None:
    """Guards the test above: if routing ever covered every rank, the empty path would be dead."""
    whole = unsharded(checkpoint)
    x = torch.randn(1, 1, whole.cfg.hidden, generator=torch.Generator().manual_seed(9)) * 0.1
    h = x.reshape(-1, whole.cfg.hidden)
    probs = torch.nn.functional.linear(h, whole.layers[0]["router"]).softmax(-1, dtype=torch.float)
    chosen = set(probs.topk(whole.cfg.top_k, dim=-1).indices.reshape(-1).tolist())
    per = whole.cfg.experts // WORLD
    assert any(not (chosen & set(range(r * per, (r + 1) * per))) for r in range(WORLD))


@pytest.mark.parametrize("tokens", [1, 5])
def test_deltanet_partials_sum_to_the_unsharded_output(checkpoint: Path, tokens: int) -> None:
    """Head-parallel DeltaNet: `out_proj` is row-parallel, so the four partials add up.

    The companion to the attention and MoE cases above, on the checkpoint this file uses
    (`test_deltanet_tp.py` covers the derivation and the sharding itself in depth). Its real
    job here is to pin that no rank returns the *whole* output, so a change that made the
    mixer reduction conditional would either double-count or drop three quarters of the layer.
    """
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i = whole.cfg.layer_types.index("linear_attention")
    x = torch.randn(1, tokens, whole.cfg.hidden, generator=torch.Generator().manual_seed(11)) * 0.1

    want = whole.deltanet(i, x)
    partials = [m.deltanet(i, x) for m in parts]
    torch.testing.assert_close(sum(partials), want, atol=1e-5, rtol=1e-5)

    for rank, partial in enumerate(partials):
        assert not torch.allclose(partial, want, atol=1e-3), f"rank {rank} returned a whole output"


def test_deltanet_state_is_sharded_over_value_heads(checkpoint: Path) -> None:
    """Each rank's recurrent state holds its own value heads, and they reassemble."""
    whole, parts = unsharded(checkpoint), ranks_of(checkpoint)
    i = whole.cfg.layer_types.index("linear_attention")
    x = torch.randn(1, 5, whole.cfg.hidden, generator=torch.Generator().manual_seed(15)) * 0.1
    whole.deltanet(i, x)
    for m in parts:
        m.deltanet(i, x)
        assert m.pool[i]["rec"].shape[1] == whole.cfg.v_heads // WORLD
    merged = torch.cat([m.state[i]["rec"] for m in parts], dim=1)
    torch.testing.assert_close(merged, whole.state[i]["rec"], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- threaded whole forward


class ThreadRing:
    """A barrier-synchronized all-reduce over `world` threads, standing in for RCCL.

    Partials are summed in rank order into one buffer, so every rank gets the same bytes
    whatever order the threads arrive in, which is the property `Model.moe` relies on when
    it routes without a collective. The second barrier is what makes the buffer reusable:
    no rank writes the next round's partial until every rank has read this one.
    """

    def __init__(self, world: int, timeout: float = 120.0) -> None:
        self.world, self.timeout = world, timeout
        self.barrier = threading.Barrier(world)
        self.slots: list[torch.Tensor | None] = [None] * world

    def reducer(self, rank: int):  # noqa: ANN201
        def reduce(x: torch.Tensor) -> None:
            self.slots[rank] = x.clone()
            self.barrier.wait(self.timeout)
            total = self.slots[0].clone()
            for r in range(1, self.world):
                total += self.slots[r]
            self.barrier.wait(self.timeout)
            x.copy_(total)

        return reduce


def in_lockstep(models: list[seed_model.Model], call) -> list:  # noqa: ANN001, ANN201
    """Run `call(model)` on every rank at once. Re-raises the first rank that failed."""
    out: list = [None] * len(models)
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            out[rank] = call(models[rank])
        except BaseException as exc:  # noqa: BLE001 -- surfaced below, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(len(models))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    if errors:
        raise errors[0]
    return out


def threaded_ranks(checkpoint: Path, ring: ThreadRing) -> list[seed_model.Model]:
    cfg = seed_model.load_cfg(checkpoint)
    return [
        seed_model.Model(
            checkpoint,
            ["cpu"],
            torch.float32,
            MAX_SEQ,
            max_batch=2,
            tp=tp.TP(tp.plan(cfg, r, WORLD), "cpu", ring.reducer(r)),
        )
        for r in range(WORLD)
    ]


def test_threaded_four_rank_forward_matches_the_unsharded_forward(checkpoint: Path) -> None:
    """The whole stack, every layer, with the all-reduce supplied by a thread barrier.

    Every rank must come out with the same logits, because the residual stream is replicated
    and only the shards differ, and those logits must be the unsharded model's.
    """
    ids = torch.randint(2, VOCAB, (1, 9), generator=torch.Generator().manual_seed(12))
    want = unsharded(checkpoint, max_batch=2).forward(ids, 0, all_logits=True)

    ring = ThreadRing(WORLD)
    models = threaded_ranks(checkpoint, ring)
    got = in_lockstep(models, lambda m: m.forward(ids, 0, all_logits=True))

    for rank, logits in enumerate(got):
        torch.testing.assert_close(logits, want, atol=2e-4, rtol=2e-4, msg=f"rank {rank}")


def test_threaded_four_rank_prefill_then_decode_matches_the_unsharded_run(
    checkpoint: Path,
) -> None:
    """Chunked prefill then decode steps: the KV cache, the recurrent state and the rope
    positions all have to line up across ranks, not just one forward."""
    prompt = torch.randint(2, VOCAB, (11,), generator=torch.Generator().manual_seed(13)).tolist()
    want, _ = greedy(unsharded(checkpoint, max_batch=2), prompt, 5)

    ring = ThreadRing(WORLD)
    models = threaded_ranks(checkpoint, ring)
    got = in_lockstep(models, lambda m: greedy(m, prompt, 5))

    for rank, (tokens, _logits) in enumerate(got):
        assert tokens == want, f"rank {rank}"


def test_threaded_four_rank_batched_decode_matches_the_unsharded_step(checkpoint: Path) -> None:
    """A real multi-slot `decode` step under the collectives, not one slot at a time.

    `greedy` above drives one slot, so it never puts two slots in one `decode` call; this is
    the shape the server actually runs and the one the batched attention/DeltaNet/MoE calls
    are written for. Two slots at *different* positions, so `attn_decode`'s masked
    (non-uniform-position) path is the one under test.
    """
    prompts = [
        torch.randint(2, VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()
        for n, seed in ((7, 17), (4, 18))
    ]

    def run(model: seed_model.Model):  # noqa: ANN202
        for slot, prompt in enumerate(prompts):
            model.begin(slot)
            for s in range(0, len(prompt), PREFILL_CHUNK):
                model.prefill(slot, prompt[s : s + PREFILL_CHUNK], s)
        slots = [0, 1]
        tokens = [prompt[-1] for prompt in prompts]
        positions = [len(prompt) for prompt in prompts]
        return model.decode(slots, tokens, positions)

    want = run(unsharded(checkpoint, max_batch=2))
    ring = ThreadRing(WORLD)
    got = in_lockstep(threaded_ranks(checkpoint, ring), run)

    for rank, logits in enumerate(got):
        assert logits.shape == want.shape
        torch.testing.assert_close(logits, want, atol=2e-4, rtol=2e-4, msg=f"rank {rank}")
        assert logits.argmax(-1).tolist() == want.argmax(-1).tolist(), f"rank {rank} greedy token"


# ---------------------------------------------------------------- real process group (gloo)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.parametrize("vocab_tp", ["0", "1"])
def test_real_process_group_matches_the_unsharded_run(
    checkpoint: Path, tmp_path: Path, vocab_tp: str
) -> None:
    """Four processes, real `init_process_group`, real `all_reduce` and `broadcast`.

    This is the highest-confidence check available without a GPU: `tp.init`, `tp_driver`'s
    command protocol and `Model`'s collectives are the production code, not a stand-in. The
    backend is gloo, so RCCL itself is still unverified. `vocab_tp="1"`: `SEED_LMHEAD_VOCAB_TP`
    (vocab-sharded head, logits all-gathered).
    """
    import os
    prompt = torch.randint(2, VOCAB, (11,), generator=torch.Generator().manual_seed(14)).tolist()
    want_tokens, want_logits = greedy(unsharded(checkpoint, max_batch=2), prompt, 5)

    out = tmp_path / "rank0.json"
    port, script = free_port(), Path(__file__).resolve().parent / "tp_gloo_rank.py"
    base = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--world", str(WORLD)]
    base += ["--port", str(port), "--prompt", ",".join(map(str, prompt)), "--new", "5"]
    env = {**os.environ, "SEED_LMHEAD_VOCAB_TP": vocab_tp}
    procs = [
        subprocess.Popen(  # noqa: S603
            [*base, "--rank", str(r), *(["--out", str(out)] if r == 0 else [])], env=env
        )
        for r in range(WORLD)
    ]
    try:
        codes = [p.wait(timeout=900) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert codes == [0] * WORLD, f"rank exit codes {codes}"

    got = json.loads(out.read_text())
    assert got["tokens"] == want_tokens
    torch.testing.assert_close(
        torch.tensor(got["logits"]), torch.tensor(want_logits), atol=2e-4, rtol=2e-4
    )


def test_real_process_group_score_matches_the_unsharded_run(
    checkpoint: Path, tmp_path: Path
) -> None:
    """`Op.SCORE`'s broadcast, under a real process group: regression coverage for the bug
    where `/v1/score` called `Model.forward` directly on rank 0, skipping the broadcast that
    tells ranks 1.. to join the forward's collectives -- which hangs every rank forever the
    first time `tp > 1` (see `server.score_continuation`, `Model.score`, `Broadcaster.score`).

    Scores `prompt[score_start:]` against the same prompt `greedy` above already decodes with,
    so one four-process boot covers both the decode path and the scoring path.
    """
    prompt = torch.randint(2, VOCAB, (11,), generator=torch.Generator().manual_seed(14)).tolist()
    score_start = 6
    want = unsharded(checkpoint, max_batch=2)
    want.begin(0)
    want_score = want.score(0, prompt, score_start)

    out = tmp_path / "rank0_score.json"
    port, script = free_port(), Path(__file__).resolve().parent / "tp_gloo_rank.py"
    base = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--world", str(WORLD)]
    base += ["--port", str(port), "--prompt", ",".join(map(str, prompt)), "--new", "5"]
    base += ["--score-start", str(score_start)]
    procs = [
        subprocess.Popen([*base, "--rank", str(r), *(["--out", str(out)] if r == 0 else [])])  # noqa: S603
        for r in range(WORLD)
    ]
    try:
        codes = [p.wait(timeout=900) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert codes == [0] * WORLD, f"rank exit codes {codes}"

    got = json.loads(out.read_text())
    got_score = torch.tensor(got["score_logits"]).reshape(got["score_shape"])
    torch.testing.assert_close(got_score, want_score, atol=2e-4, rtol=2e-4)


def test_real_process_group_scheduler_keeps_every_rank_on_one_block_allocator(
    checkpoint: Path, tmp_path: Path
) -> None:
    """The scheduler on rank 0, four real processes, a workload with copy-on-write, eviction
    and block reuse (`tp_gloo_sched_rank.run_workload`).

    Rank 0's allocator is the only authority: every block id reaches the workers inside a
    broadcast command (`ATTACH_BLOCKS`, `EXTEND_BLOCKS`, `COPY_BLOCK`). Before that, workers
    grew their tables from their own allocators, which never saw the cache's incref/decref,
    COW or eviction: after the first COW their tables named different blocks than rank 0's,
    so they read and wrote other sessions' KV, and they never freed, ending in a worker
    `MemoryError` and a rank-0 hang. Checks: every rank's block tables are identical at every
    command boundary, ranks holding the same KV head hold identical K/V in every live block,
    and the tokens match the same workload on the unsharded model.
    """
    from scheduler import Scheduler
    from session_cache import SessionCache
    from tp_driver import pool_handshake
    from tp_gloo_sched_rank import MAX_BATCH, run_workload

    ref = unsharded(checkpoint, max_batch=MAX_BATCH)
    ref.warmup()
    pool_handshake(ref)
    ref_cache = SessionCache(ref.block_allocator, ref.block_size, ref.num_snapshots)
    want = run_workload(Scheduler(ref, ref_cache), ref.cfg.vocab)

    port, script = free_port(), Path(__file__).resolve().parent / "tp_gloo_sched_rank.py"
    base = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--world", str(WORLD)]
    base += ["--port", str(port), "--out-dir", str(tmp_path)]
    procs = [subprocess.Popen([*base, "--rank", str(r)]) for r in range(WORLD)]  # noqa: S603
    try:
        codes = [p.wait(timeout=900) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert codes == [0] * WORLD, f"rank exit codes {codes}"

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["outputs"] == want
    logs = [json.loads((tmp_path / f"tables{r}.json").read_text()) for r in range(WORLD)]
    assert all(len(log) == len(logs[0]) for log in logs)
    for r in range(1, WORLD):
        for step, (mine, theirs) in enumerate(zip(logs[0], logs[r], strict=True)):
            assert mine == theirs, f"rank {r} block tables diverge at command {step}"
    live = result["live"]
    assert live, "setup: the run must leave cached entries holding blocks"
    rows = torch.tensor([b * ref.block_size + t for b in live for t in range(ref.block_size)])
    dumps = [torch.load(tmp_path / f"pool{r}.pt") for r in range(WORLD)]
    for r in range(WORLD):
        h = dumps[r]["kv_start"]
        for i, (k, v) in dumps[r]["pools"].items():
            torch.testing.assert_close(k[rows], ref.pool[i]["k"][rows][:, h : h + 1])
            torch.testing.assert_close(v[rows], ref.pool[i]["v"][rows][:, h : h + 1])


def test_pool_handshake_rejects_leaked_blocks(checkpoint: Path) -> None:
    """The boot self-check refuses to serve with blocks still allocated after warmup."""
    from tp_driver import PoolMismatchError, pool_handshake

    m = unsharded(checkpoint, max_batch=2)
    m.warmup()
    pool_handshake(m)
    assert m.scheduler_owns_blocks
    m.scheduler_owns_blocks = False
    m.block_allocator.alloc(1)
    with pytest.raises(PoolMismatchError, match="still allocated"):
        pool_handshake(m)


def test_the_gloo_rank_script_prefills_in_more_than_one_chunk() -> None:
    """A guard on the check above: an 11-token prompt has to cross a chunk boundary, or the
    prefill hand-off between chunks would go untested under a real process group."""
    assert 1 < PREFILL_CHUNK < 11


# ---------------------------------------------------------------- server wiring


def test_graph_capture_is_allowed_above_one_rank() -> None:
    """This used to be rejected, on the premise that `graph_decode` was a second, *unsharded*
    spelling of the decode step, so a rank replaying it would meet an eager rank inside a
    collective. The static step is sharded now and every rank captures its own graph; the
    sharding is checked against an unsharded reference in `test_graph_capture_tp.py`, and the
    ranks turning it on together is checked in `test_graph_capture.py`."""
    import server

    args = server.parse_args(["--model-path", "/x", "--tp", "4", "--enable-graph-capture"])
    assert (args.tp, args.enable_graph_capture) == (4, True)


def test_the_server_rejects_a_rank_outside_its_own_group() -> None:
    import server

    with pytest.raises(SystemExit):
        server.parse_args(["--model-path", "/x", "--tp", "4", "--rank", "4"])


def test_worker_argv_carries_every_flag_that_changes_a_shard() -> None:
    """A worker that loaded a different shape than rank 0 would hang on the first collective,
    so the re-exec has to carry every flag `build_model` reads."""
    import server

    args = server.parse_args(
        ["--model-path", "/m", "--tp", "4", "--tp-port", "31337", "--max-batch", "7"]
    )
    argvs = _dry_run_start_workers(server, args)
    assert len(argvs) == WORLD - 1
    for rank, argv in enumerate(argvs, start=1):
        assert argv[argv.index("--rank") + 1] == str(rank)
        for flag, value in (
            ("--model-path", "/m"),
            ("--tp", "4"),
            ("--tp-port", "31337"),
            ("--max-batch", "7"),
            ("--max-seq-len", str(args.max_seq_len)),
            ("--dtype", args.dtype),
        ):
            assert argv[argv.index(flag) + 1] == value, f"rank {rank} {flag}"


def _dry_run_start_workers(server, args) -> list[list[str]]:  # noqa: ANN001
    """`start_workers` with the process spawn replaced by recording each argv."""
    recorded: list[list[str]] = []

    def record(argv: list[str], *_a: object, **_k: object) -> None:
        recorded.append(argv)

    real = server.subprocess.Popen
    server.subprocess.Popen = record
    try:
        server.start_workers(args)
    finally:
        server.subprocess.Popen = real
    return recorded
