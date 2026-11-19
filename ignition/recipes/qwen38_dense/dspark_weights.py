# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B DSpark draft layers, context projections and Markov transition weights."""

from dataclasses import dataclass

import torch

from ignition.components.dflash2_weights import DraftLayerWeights
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat

# The confidence head is not executed, so its tensors are checked for shape and never read.
_LOADED_PREFIXES = ("fc.", "hidden_norm.", "norm.", "layers.", "markov_head.")


@dataclass(frozen=True)
class DraftWeights:
    """One rank's BF16 draft weights.

    ``projection`` combines the concatenated target hidden states into one draft-width state.
    ``context_kv`` stacks every layer's key rows followed by its value rows, in layer order, so one
    projection produces the keys and values that all draft layers store for an accepted token.
    Each layer's ``qkv`` fuses its query, key and value projections and its convolution fields
    are ``None``. ``markov_embedding`` is replicated and ``markov_head`` is partitioned by
    vocabulary row. The draft embeds tokens and computes logits with the target's weights.
    """

    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None):
        config.validate_shard(shard)
        manifest = inspect_indexed_checkpoint(snapshot)
        validate_metadata(manifest, config)
        c, a = config.text, config.attention
        q_range = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        first, last = shard.replicated_partition(a.kv_heads)
        kv_range = first * a.key_dim, last * a.key_dim
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        layers, context_parts = [], []
        with DirectWeightLoader(manifest, device, prefixes=_LOADED_PREFIXES, progress=progress,
                                progress_label="Load draft weights") as loader:
            def matrix(prefix, rows=None, columns=None, *, temporary=False):
                return loader.load_tensor(prefix + ".weight", extent=(
                    slice(*(rows or (None, None))), slice(*(columns or (None, None)))), temporary=temporary)

            projection = matrix("fc", rows=shard.partition(c["hidden_size"], alignment=64))
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            for index in range(config.layers):
                p = f"layers.{index}"
                qkv = persistent_cat([matrix(p + ".self_attn." + name + "_proj", rows=rows, temporary=True)
                    for name, rows in zip(("q", "k", "v"), (q_range, kv_range, kv_range), strict=True)])
                context_parts.append(qkv[q_range[1] - q_range[0]:])
                output = matrix(p + ".self_attn.o_proj", columns=q_range)
                query_norm, key_norm, input_norm, post_norm = loader.load_tensors(tuple(p + name + ".weight"
                    for name in (".self_attn.q_norm", ".self_attn.k_norm", ".input_layernorm",
                                 ".post_attention_layernorm")))
                mlp = MLPWeights(persistent_cat([matrix(p + ".mlp." + name + "_proj", rows=intermediate,
                    temporary=True) for name in ("gate", "up")]), matrix(p + ".mlp.down_proj", columns=intermediate))
                layers.append(DraftLayerWeights(qkv, output, query_norm, key_norm, input_norm, post_norm, mlp,
                                                None, None))
            markov_embedding = matrix("markov_head.markov_w1")
            markov_head = matrix("markov_head.markov_w2", rows=shard.partition(c["vocab_size"], alignment=64))
        return cls(projection, hidden_norm, persistent_cat(context_parts), norm, markov_embedding, markov_head,
                   tuple(layers))


def tensor_shapes(config):
    """Shape of every tensor a DSpark checkpoint with this configuration stores, by tensor name."""
    c = config.text
    h, i, head = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    q, kv = c["num_attention_heads"] * head, c["num_key_value_heads"] * head
    shapes = {"fc.weight": (h, h * len(config.target_layers)), "hidden_norm.weight": (h,), "norm.weight": (h,),
              "markov_head.markov_w1.weight": (c["vocab_size"], config.markov_rank),
              "markov_head.markov_w2.weight": (c["vocab_size"], config.markov_rank),
              "confidence_head.proj.weight": (1, h + config.markov_rank), "confidence_head.proj.bias": (1,)}
    for layer in range(config.layers):
        for name, shape in {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
                "self_attn.o_proj": (h, q), "self_attn.q_norm": (head,), "self_attn.k_norm": (head,),
                "input_layernorm": (h,), "post_attention_layernorm": (h,), "mlp.gate_proj": (i, h),
                "mlp.up_proj": (i, h), "mlp.down_proj": (h, i)}.items():
            shapes[f"layers.{layer}.{name}.weight"] = shape
    return shapes


def validate_metadata(manifest, config):
    """Require exactly the DSpark tensor set, each stored as BF16 with its configured shape."""
    shapes = tensor_shapes(config)
    missing, unexpected = shapes.keys() - manifest.tensors.keys(), manifest.tensors.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError("Qwen3.8-27B DSpark checkpoint tensor names disagree with its configuration: "
                         f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, shape in shapes.items():
        tensor = manifest.tensors[name]
        if tensor.dtype != "BF16" or tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: Qwen3.8-27B DSpark requires BF16 storage with shape {shape}, "
                             f"found {tensor.dtype} {tuple(tensor.shape)}")
