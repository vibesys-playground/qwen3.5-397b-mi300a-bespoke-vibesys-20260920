"""Shared fixture for the DeltaNet tensor-parallel tests: a tiny on-disk checkpoint.

`test_deltanet_tp.py` hand-shards in one process; `test_deltanet_tp_gloo.py` runs the
same model in real `torch.distributed` worker processes. Both need the same weights and
the same inputs, and the gloo workers are launched as plain subprocesses, so this lives
in an importable module rather than in either test file.

The dims mirror the real checkpoint's structure at toy size: `rep = v_heads // k_heads`
is 4 as in the real config (64 value heads over 16 key heads), and both head counts
divide 2 and 4 so the tests can run at TP=2 and TP=4.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREFIX = "model.language_model."

HIDDEN = 64
K_HEADS, V_HEADS = 4, 16  # rep = 4, as in the real config (16 key / 64 value heads)
K_DIM = V_DIM = 16
CONV_K = 4
LAYER_TYPES = ("linear_attention", "full_attention", "linear_attention")
VOCAB = 32
EXPERTS, TOP_K, INTER = 4, 2, 16
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 16

KEY_DIM, VALUE_DIM = K_HEADS * K_DIM, V_HEADS * V_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM

TEXT_CONFIG = {
    "hidden_size": HIDDEN,
    "vocab_size": VOCAB,
    "layer_types": list(LAYER_TYPES),
    "rms_norm_eps": 1e-6,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "linear_num_key_heads": K_HEADS,
    "linear_num_value_heads": V_HEADS,
    "linear_key_head_dim": K_DIM,
    "linear_value_head_dim": V_DIM,
    "linear_conv_kernel_dim": CONV_K,
    "num_experts": EXPERTS,
    "num_experts_per_tok": TOP_K,
    "shared_expert_intermediate_size": INTER,
    "eos_token_id": 1,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
    },
}


def _deltanet_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "linear_attn.in_proj_qkv.weight": (CONV_DIM, HIDDEN),
        "linear_attn.in_proj_z.weight": (VALUE_DIM, HIDDEN),
        "linear_attn.in_proj_a.weight": (V_HEADS, HIDDEN),
        "linear_attn.in_proj_b.weight": (V_HEADS, HIDDEN),
        "linear_attn.out_proj.weight": (HIDDEN, VALUE_DIM),
        "linear_attn.conv1d.weight": (CONV_DIM, 1, CONV_K),
        "linear_attn.A_log": (V_HEADS,),
        "linear_attn.dt_bias": (V_HEADS,),
        "linear_attn.norm.weight": (V_DIM,),
    }


def _attention_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "self_attn.q_proj.weight": (HEADS * 2 * HEAD_DIM, HIDDEN),  # output-gated q
        "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
        "self_attn.q_norm.weight": (HEAD_DIM,),
        "self_attn.k_norm.weight": (HEAD_DIM,),
    }


def _layer_shapes(i: int) -> dict[str, tuple[int, ...]]:
    shapes = {"input_layernorm.weight": (HIDDEN,), "post_attention_layernorm.weight": (HIDDEN,)}
    shapes |= _attention_shapes() if LAYER_TYPES[i] == "full_attention" else _deltanet_shapes()
    shapes |= {
        "mlp.shared_expert.gate_proj.weight": (INTER, HIDDEN),
        "mlp.shared_expert.up_proj.weight": (INTER, HIDDEN),
        "mlp.shared_expert.down_proj.weight": (HIDDEN, INTER),
        "mlp.gate.weight": (EXPERTS, HIDDEN),
        "mlp.shared_expert_gate.weight": (1, HIDDEN),
    }
    for e in range(EXPERTS):
        shapes |= {
            f"mlp.experts.{e}.gate_proj.weight": (INTER, HIDDEN),
            f"mlp.experts.{e}.up_proj.weight": (INTER, HIDDEN),
            f"mlp.experts.{e}.down_proj.weight": (HIDDEN, INTER),
        }
    return shapes


def write_checkpoint(out: Path, seed: int = 0) -> Path:
    """Write a random fp32 checkpoint the seed's loader accepts, and return `out`.

    Dense (not MXFP4) experts: the MoE quantization is orthogonal to DeltaNet sharding,
    and the dense path is the one `load_experts` takes when there is no `weight_scale`.
    """
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {
        f"{PREFIX}embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen) * 0.3,
        f"{PREFIX}norm.weight": torch.randn(HIDDEN, generator=gen) * 0.1,
        "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=gen) * 0.3,
    }
    for i in range(len(LAYER_TYPES)):
        for name, shape in _layer_shapes(i).items():
            # A_log feeds exp(); keep it in the checkpoint's (0, 16) range for exp(A_log).
            scale = 1.0 if name.endswith("A_log") else 0.3 if len(shape) > 1 else 0.1
            tensors[f"{PREFIX}layers.{i}.{name}"] = torch.randn(*shape, generator=gen) * scale
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({"text_config": TEXT_CONFIG}))
    return out


def hidden_states(t: int, seed: int, batch: int = 1) -> torch.Tensor:
    """Deterministic DeltaNet layer input [batch, t, hidden], identical on every rank."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, t, HIDDEN, generator=gen)
