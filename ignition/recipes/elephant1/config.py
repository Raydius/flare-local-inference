# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural checkpoint configuration for Elephant-1, a Qwen3.5-9B decision fine-tune."""

from dataclasses import dataclass
import json
from pathlib import Path

MODEL_ID = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
TEXT_PREFIX = "model.language_model"
PAGE_TOKENS = 128

# Weights-only NVFP4 storage: BF16 activations, group-16 scales. `ignition.preparation.quantization.linear_format`
# reads this entry grammar, and every packed projection keeps an ordinary block scale.
_NVFP4_A16 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

# Body projections packed at preparation. The embedding, vocabulary head, every normalization weight, the GDN
# convolution and its per-head scalars stay in checkpoint precision.
PACKED_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
PACKED_MLP = ("gate_proj", "up_proj", "down_proj")
PACKED_LINEAR_ATTENTION = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")

# The trained text length. Flare refuses longer requests instead of truncating them.
TRAINED_CONTEXT = 16384


def packed_plan(layers):
    """Body projections stored as NVFP4 for one checkpoint's layer kinds."""
    plan = {}
    for index, kind in enumerate(layers):
        if kind == "linear_attention":
            for field in PACKED_LINEAR_ATTENTION:
                plan[f"{TEXT_PREFIX}.layers.{index}.linear_attn.{field}"] = dict(_NVFP4_A16)
        else:
            for field in PACKED_ATTENTION:
                plan[f"{TEXT_PREFIX}.layers.{index}.self_attn.{field}"] = dict(_NVFP4_A16)
        for field in PACKED_MLP:
            plan[f"{TEXT_PREFIX}.layers.{index}.mlp.{field}"] = dict(_NVFP4_A16)
    return plan


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    draft: object = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("Elephant-1 preparation requires Qwen3_5ForConditionalGeneration")
        text = raw.get("text_config", {})
        required = {"model_type": "qwen3_5_text", "hidden_act": "silu", "attention_bias": False,
                    "attn_output_gate": True, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
                    "head_dim": 256, "tie_word_embeddings": False, "hidden_size": 4096,
                    "intermediate_size": 12288, "vocab_size": 248320, "num_attention_heads": 16,
                    "num_key_value_heads": 4, "num_hidden_layers": 32}
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"unsupported Elephant-1 {name}: {text.get(name)!r}")
        layers = tuple(text.get("layer_types", ()))
        if (len(layers) != text["num_hidden_layers"]
                or any(kind not in ("linear_attention", "full_attention") for kind in layers)
                or any((kind == "full_attention") != (index % 4 == 3) for index, kind in enumerate(layers))):
            raise ValueError("Elephant-1 layer types must be three linear-attention layers per full-attention layer")
        rope = text.get("rope_parameters", {})
        rotary_dim = int(text["head_dim"] * rope.get("partial_rotary_factor", 0))
        section = rope.get("mrope_section")
        if (rope.get("rope_type") != "default" or rope.get("partial_rotary_factor") != .25
                or rope.get("mrope_interleaved") is not True or tuple(section or ()) != (11, 11, 10)
                or sum(section) * 2 != rotary_dim):
            raise ValueError("Elephant-1 requires interleaved partial rotary embeddings covering the head dimension")
        if not 0 < float(text.get("rms_norm_eps", 0)) or not 0 < float(rope.get("rope_theta", 0)):
            raise ValueError("Elephant-1 rotary frequency base and normalization epsilon must be positive")
        if text.get("max_position_embeddings", 0) < TRAINED_CONTEXT:
            raise ValueError("Elephant-1 checkpoint must position-embed the trained text length")
        if raw.get("quantization_config") is not None:
            raise ValueError(
                "Elephant-1 preparation packs BF16 body weights itself;"
                " a checkpoint that declares its own quantization is served by its own recipe")
        return cls(text, packed_plan(layers))

    @property
    def layers(self):
        return tuple(self.text["layer_types"])

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def intermediate(self):
        return self.text["intermediate_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def rotary(self):
        rope = self.text["rope_parameters"]
        return int(self.text["head_dim"] * rope["partial_rotary_factor"]), float(rope["rope_theta"])

    @property
    def pack_gdn_inputs(self):
        """Elephant-1 keeps B and A input projections fused and QKV and Z separate; nothing is packed jointly."""
        return False

    def packed(self, prefix):
        entry = self.quantization.get(prefix)
        return entry is not None and entry.get("quant_algo") in ("NVFP4", "W4A16_NVFP4")
