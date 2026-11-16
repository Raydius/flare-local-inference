# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 canonical parameters, checkpoint mapping, and prepared projections."""

from dataclasses import dataclass

import torch

from ignition.preparation.persistent import allocated_tensor_bytes
from ignition.components.model import canonical_tree
from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader

from .config import ModelConfig, TEXT_PREFIX

VISION_PREFIX = "model.visual."


def checkpoint_parameters(config):
    """Every checkpoint tensor name the text recipe consumes."""
    names = {f"{TEXT_PREFIX}.embed_tokens.weight", f"{TEXT_PREFIX}.norm.weight", "lm_head.weight"}
    for index, kind in enumerate(config.layers):
        base = f"{TEXT_PREFIX}.layers.{index}"
        names |= {f"{base}.input_layernorm.weight", f"{base}.post_attention_layernorm.weight"}
        if kind == "linear_attention":
            names |= {f"{base}.linear_attn.{field}" for field in (
                "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
                "out_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight")}
        else:
            names |= {f"{base}.self_attn.{field}" for field in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "q_norm.weight", "k_norm.weight")}
        names |= {f"{base}.mlp.{field}" for field in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
    return names


def projection_plan(config):
    """Prepared projection name to the checkpoint prefixes fused into it, in fusion order.

    A fused query/gate projection arrives as per-head pairs, so query and gate rows are separated and the gate
    rows moved behind the query rows exactly as `ignition/recipes/qwen38_dense/weights.py` does. GDN B and A
    inputs fuse in checkpoint order.
    """
    plan = {}
    for index, kind in enumerate(config.layers):
        base = f"{TEXT_PREFIX}.layers.{index}"
        short = f"layers.{index}"
        if kind == "linear_attention":
            attention = f"{base}.linear_attn"
            plan[f"{short}.linear_attn.in_proj_qkv"] = (f"{attention}.in_proj_qkv",)
            plan[f"{short}.linear_attn.in_proj_z"] = (f"{attention}.in_proj_z",)
            plan[f"{short}.linear_attn.in_proj_ba"] = (f"{attention}.in_proj_b", f"{attention}.in_proj_a")
            plan[f"{short}.linear_attn.out_proj"] = (f"{attention}.out_proj",)
        else:
            attention = f"{base}.self_attn"
            plan[f"{short}.self_attn.qgkv_proj"] = (f"{attention}.q_proj", f"{attention}.k_proj",
                                                    f"{attention}.v_proj")
            plan[f"{short}.self_attn.o_proj"] = (f"{attention}.o_proj",)
        plan[f"{short}.mlp.gate_up"] = (f"{base}.mlp.gate_proj", f"{base}.mlp.up_proj")
        plan[f"{short}.mlp.down_proj"] = (f"{base}.mlp.down_proj",)
    return plan


def check_manifest(names, config):
    """Refuse a checkpoint that does not carry exactly the mapped text tensors; skip the vision tower."""
    expected = checkpoint_parameters(config)
    skipped = {name for name in names if name.startswith(VISION_PREFIX)}
    unknown = sorted(name for name in names if name not in expected and name not in skipped)
    if unknown:
        raise ValueError(f"Elephant-1 checkpoint carries unmapped tensors: {unknown[:8]}")
    missing = sorted(expected - set(names))
    if missing:
        raise ValueError(f"Elephant-1 checkpoint is missing mapped tensors: {missing[:8]}")
    return len(skipped)


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    tree: torch.nn.Module
    projections: dict
    head: object
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)
    skipped_vision_tensors: int = 0
    draft: object = None
    draft_head: object = None

    @classmethod
    def load(cls, manifest, device, *, progress=None):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        from ignition.components.weights import pack_body_nvfp4

        config = ModelConfig.load(manifest.snapshot)
        skipped = check_manifest(set(manifest.tensors), config)
        tree = canonical_tree(Qwen3_5TextModel, Qwen3_5TextConfig.from_dict(config.text))
        plan = projection_plan(config)
        grouped = {part + ".weight" for parts in plan.values() for part in parts}
        before = allocated_tensor_bytes(device)
        projections = {}
        with DirectWeightLoader(manifest, device, prefixes=(TEXT_PREFIX + ".", "lm_head."),
                                progress=progress) as loader:
            for name, _ in tuple(tree.named_parameters()):
                if name in grouped or name == "embed_tokens.weight":
                    continue
                part, _, field = name.rpartition(".")
                tensor = loader.load_tensor(f"{TEXT_PREFIX}.{name}")
                setattr(tree.get_submodule(part), field, torch.nn.Parameter(tensor, requires_grad=False))
            tree.embed_tokens.weight = torch.nn.Parameter(
                loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight"), requires_grad=False)
            for name, parts in plan.items():
                fields = tuple(f"{part}.weight" for part in parts)
                if len(fields) == 1:
                    combined = loader.load_linear(parts[0])
                else:
                    combined = loader.load_linear_group(parts)
                _validate_shape(name, combined, _expected_shape(config, name))
                if name.endswith(".self_attn.qgkv_proj"):
                    combined = _separate_query_gate(combined, config)
                projections[name] = (pack_body_nvfp4(combined) if any(config.packed(part) for part in parts)
                                     else combined)
            head = loader.load_linear("lm_head")
            if tuple(head.shape) != (config.vocabulary, config.hidden):
                raise ValueError("Elephant-1 vocabulary projection shape differs from the model")
        return cls(config, tree, projections, head, allocated_tensor_bytes(device) - before,
                   skipped_vision_tensors=skipped)


def _expected_shape(config, name):
    text = config.text
    query, kv = text["num_attention_heads"] * text["head_dim"], text["num_key_value_heads"] * text["head_dim"]
    value = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    shapes = {
        "self_attn.qgkv_proj": (2 * query + 2 * kv, config.hidden),
        "self_attn.o_proj": (config.hidden, query),
        "mlp.gate_up": (2 * config.intermediate, config.hidden),
        "mlp.down_proj": (config.hidden, config.intermediate),
        "linear_attn.in_proj_qkv": (
            2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] + value, config.hidden),
        "linear_attn.in_proj_z": (value, config.hidden),
        "linear_attn.in_proj_ba": (2 * text["linear_num_value_heads"], config.hidden),
        "linear_attn.out_proj": (config.hidden, value),
    }
    for suffix, shape in shapes.items():
        if name.endswith(suffix):
            return shape
    raise ValueError(f"unplanned projection {name}")


def _separate_query_gate(combined, config):
    """Move gate rows behind query rows: per-head [query, gate] pairs become contiguous blocks."""
    heads, dim, hidden = config.text["num_attention_heads"], config.text["head_dim"], config.hidden
    query, gate = combined[:2 * heads * dim].view(heads, 2, dim, hidden).unbind(1)
    separated = torch.cat((query.reshape(-1, hidden), gate.reshape(-1, hidden),
                           combined[2 * heads * dim:]))
    del query, gate
    return separated


def _validate_shape(name, weight, expected):
    shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else (weight.out_features, weight.in_features)
    if shape != expected:
        raise ValueError(f"{name}: checkpoint projection shape {shape} differs from the model's {expected}")
