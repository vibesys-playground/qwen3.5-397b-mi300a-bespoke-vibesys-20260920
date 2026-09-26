"""Tensor-parallel sharding for the gated DeltaNet layers.

There is no published recipe for splitting this recurrence across devices, so the
scheme below is derived from the update itself. Per head, with state S [k_dim, v_dim]:

    S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t k_t v_t^T
    o_t = S_t^T q_t

Read as an index expression, for head h:

    S[h, a, b] <- exp(g[h]) * (S[h, a, b] - beta[h] k[h, a] sum_c S[h, c, b] k[h, c])
                  + beta[h] k[h, a] v[h, b]
    o[h, b]    =  sum_a S[h, a, b] q[h, a]

Three facts fall out:

1. **No term mixes two head indices.** `h` is a free index on every tensor in the
   recurrence, so the state is block diagonal over heads and the whole recurrence is
   `v_heads` independent recurrences that happen to be batched. Splitting heads across
   ranks therefore needs *no communication inside the recurrence at all*, at any
   timestep and at any chunk boundary. Heads are the dimension we shard.

2. **The only contracted state axis is `k_dim` (index `a`, `c`).** Both `sum_c` in the
   update and `sum_a` in the readout run over `k_dim`. Splitting `k_dim` across ranks
   would need an all_reduce of `[B, H, v_dim]` for the memory read and another of
   `[B, H, v_dim]` for the output, *per token* (per chunk in the chunked form, which is
   worse: the WY solve `(I + A)[U W]` contracts `k_dim` inside the triangular system).
   That is rejected.

3. **`v_dim` (index `b`) is never contracted**, so splitting `v_dim` is also exact and
   communication free. It is the fallback if `world_size` ever exceeds `v_heads`; it is
   not used at TP=4 because sharding heads also splits `q`, `k`, `beta`, `g`, the conv
   and every input projection, whereas a `v_dim` split leaves `q`, `k`, `beta`, `g` and
   the `q`/`k` half of the conv replicated on every rank. With `linear_num_value_heads`
   = 64 and `linear_num_key_heads` = 16 there is no shortage of heads at TP=4.

So the layer is Megatron's column-then-row pattern with value heads in the role of
attention heads, and the recurrence in the role of the attention kernel:

| tensor | shape | sharding |
| --- | --- | --- |
| `in_proj_qkv` | `[2*key_dim + value_dim, hidden]` | rows, **per segment** (see `_shard_qkv`) |
| `conv` | `[conv_dim, 1, conv_k]` | rows, per segment; depthwise, so per channel and exact |
| `in_proj_z` | `[value_dim, hidden]` | rows |
| `in_proj_a`, `in_proj_b` | `[v_heads, hidden]` | rows |
| `A_log`, `dt_bias` | `[v_heads]` | rows |
| `dn_norm` | `[v_dim]` | **replicated**: the output norm is over one head's channels |
| `out_proj` | `[hidden, value_dim]` | columns (the input dim) + one all_reduce |
| `rec` state | `[B, v_heads, k_dim, v_dim]` | heads; 1/world_size of the state per rank |
| `conv` state | `[B, conv_dim, conv_k - 1]` | channels, per segment |

Communication: **exactly one all_reduce of `[B, T, hidden]` per DeltaNet layer**, over
`out_proj`'s partial sum, and nothing else. `x` enters replicated (it is the residual
stream) and leaves replicated. The batched decode path reduces once for the whole batch
rather than once per slot. Following `tp.py`'s convention, `Model.deltanet` returns that
partial sum and `Model.layer` / `Model.decode_layer` issue the collective, exactly as
they do for full attention's `o_proj`; nothing here calls a collective itself.

Head alignment. `q`/`k` carry `k_heads` heads and are `repeat_interleave`d by
`rep = v_heads // k_heads` onto value heads. Rank `r` owns value heads
`[r*Vl, (r+1)*Vl)` and key heads `[r*Kl, (r+1)*Kl)` with `Vl = v_heads/P`,
`Kl = k_heads/P`. Value head `g` maps to key head `g // rep`, and
`[r*Vl, (r+1)*Vl) // rep = [r*Kl, (r+1)*Kl)` because `Vl / rep = Kl`, so the two cuts
land on the same group boundary for free whenever `P` divides both counts. That is the
only constraint (`local_cfg` enforces it); at TP=4 on this checkpoint it is 16/4 and
64/4.

Numerics. The scheme is exact in exact arithmetic, and no reduction in the recurrence
changes order, so the delta-rule kernels are *bit* identical on a rank's heads to the
corresponding slice of the unsharded run (`test_deltanet_tp.py` asserts `torch.equal`).
End to end the sharded layer agrees to ~1e-7 relative in fp32, from two places:

- `out_proj` is row parallel, so `world_size` partial products are each rounded and then
  summed instead of one accumulation over `value_dim`. This is the usual row-parallel
  tradeoff and the only one that is intrinsic to the sharding.
- torch's CPU/GPU elementwise kernels for `softplus` and `sigmoid` are not invariant to
  the tensor's width (the vectorized body and the scalar tail round differently), so the
  `beta` and `g` projections land within one ulp of the unsharded values rather than on
  them. `F.linear`, the depthwise conv and `exp` are width invariant, so the conv state
  and the q/k/v inputs to the recurrence *are* bit identical. This is a property of the
  kernels, not of the sharding, and it is why the layer-level tests use a tolerance.
"""

import dataclasses
from typing import TYPE_CHECKING

import torch
from tp import TP

if TYPE_CHECKING:
    from model import Cfg

# Row-sharded along dim 0 with one head (or one head's scalar) per row.
_ROW_SHARDED = ("in_proj_z", "in_proj_a", "in_proj_b", "A_log", "dt_bias")


def local_cfg(cfg: "Cfg", tp: TP) -> "Cfg":
    """`cfg` with the DeltaNet head counts cut to this rank's share.

    Every DeltaNet shape the model derives (`conv_dim`, the recurrent state, the views
    the projections are reshaped into) is a function of `k_heads` and `v_heads`, so
    dividing those two fields is the entire shape change; nothing else in `Cfg` is
    tensor-parallel. The full-attention and MoE fields are left alone: they are sharded
    (or not) by their own owners.
    """
    if not tp.enabled:
        return cfg
    return dataclasses.replace(
        cfg,
        k_heads=tp.split_size(cfg.k_heads, "linear_num_key_heads"),
        v_heads=tp.split_size(cfg.v_heads, "linear_num_value_heads"),
    )


def shard_deltanet_weights(layer: dict, cfg: "Cfg", tp: TP) -> dict:
    """This rank's slice of one loaded layer's DeltaNet tensors.

    `cfg` is the *global* config, because `layer` was loaded whole. Layers that are not
    DeltaNet layers, and the one-process group, pass straight through. Returns a new
    dict; the caller drops the full tensors by dropping the one it passed in.

    Loading whole and then slicing costs one layer's worth of transient memory per rank
    (the DeltaNet projections are about 100 M parameters per layer, ~200 MB in bf16,
    freed as soon as the caller drops the full dict). Slicing during the read instead
    would need a shard-aware `Checkpoint.load`, which is a change to the loader rather
    than to this scheme; it is worth doing only if load time becomes the constraint.
    """
    if not tp.enabled or "in_proj_qkv" not in layer:
        return layer
    out = dict(layer)
    out["in_proj_qkv"] = shard_qkv_segments(layer["in_proj_qkv"], cfg, tp, dim=0)
    out["conv"] = shard_qkv_segments(layer["conv"], cfg, tp, dim=0)
    for name in _ROW_SHARDED:
        out[name] = tp.shard(layer[name], 0, name)
    out["out_proj"] = tp.shard(layer["out_proj"], 1, "out_proj value_dim")
    return out  # dn_norm is [v_dim] and stays whole: the output norm is per value head


def shard_qkv_segments(t: torch.Tensor, cfg: "Cfg", tp: TP, dim: int) -> torch.Tensor:
    """Shard a `[q | k | v]`-concatenated axis, one segment at a time.

    `in_proj_qkv` and the depthwise conv are laid out as `key_dim, key_dim, value_dim`
    along one axis, and the layer splits them back out with exactly those sizes. The
    three segments have different head counts, so a single flat slice of the
    concatenation would hand a rank the wrong channels (at TP=4 it would give rank 1
    part of `q` and part of `k` instead of its quarter of each). Each segment is sharded
    on its own and the pieces are re-concatenated in the same order, which is what makes
    the local tensor splittable by the *local* `key_dim, key_dim, value_dim`.
    """
    key_dim, val_dim = cfg.k_heads * cfg.k_dim, cfg.v_heads * cfg.v_dim
    q, k, v = t.split([key_dim, key_dim, val_dim], dim=dim)
    return torch.cat(
        [
            tp.shard(q, dim, "key_dim"),
            tp.shard(k, dim, "key_dim"),
            tp.shard(v, dim, "value_dim"),
        ],
        dim=dim,
    )
