# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 projections, grouped convolutions, and candidate selector."""

from __future__ import annotations


from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights, quantize_bf16_linear_as_mxfp8
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat


if TYPE_CHECKING:
    from b12x.gemm.blockscaled import Weight


@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: Weight | torch.Tensor
    output: Weight | torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    attention_conv: dict | None
    mlp_conv: dict | None


# Projector-less DFlash drafts name the context projection and its norm under ``encoder``.
_PROJECTION_NAMES = {"dflash2": ("fc", "hidden_norm"), "dflash": ("encoder.fc", "encoder.output_norm_enc"),
                     "dspark": ("fc", "hidden_norm")}
# A DSpark draft stores the layers without convolutions or a selector, a low-rank Markov transition,
# and a confidence head.
# The confidence head is not executed: its tensors are checked for shape and never read.
_LOADED_PREFIXES = {"dspark": ("fc.", "hidden_norm.", "norm.", "layers.", "markov_head.")}
_FAMILY = {"dflash2": "DFlash2", "dflash": "DFlash2", "dspark": "DSpark"}


@dataclass(frozen=True)
class DraftWeights:
    projection: Weight | torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: Weight | torch.Tensor
    norm: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]
    selector: dict | None
    # DSpark only: the transition's replicated embedding rows and its head, partitioned by vocabulary row.
    markov_embedding: torch.Tensor | None = None
    markov_head: torch.Tensor | None = None

    @classmethod
    def load(cls, snapshot, config, device, shard, *, progress=None, online_mxfp8=False, projector="dflash2"):
        config.validate_shard(shard)
        manifest = inspect_indexed_checkpoint(snapshot)
        validate_metadata(manifest, config, projector=projector)
        c, a = config.text, config.attention
        projection_name, hidden_norm_name = _PROJECTION_NAMES[projector]
        convolutions = projector == "dflash2"
        q_range = shard.partition(a.heads * a.key_dim, alignment=a.key_dim)
        first, last = shard.replicated_partition(a.kv_heads)
        kv_range = first * a.key_dim, last * a.key_dim
        intermediate = shard.partition(c["intermediate_size"], alignment=128)
        layers, context_parts = [], []

        def pack(weight):
            return quantize_bf16_linear_as_mxfp8(weight) if online_mxfp8 else weight

        with DirectWeightLoader(manifest, device, prefixes=_LOADED_PREFIXES.get(projector), progress=progress,
                                progress_label="Load draft weights") as loader:
            def matrix(prefix, rows=None, columns=None, *, temporary=None):
                return loader.load_tensor(prefix + ".weight", extent=(
                    slice(*(rows or (None, None))), slice(*(columns or (None, None)))),
                    temporary=online_mxfp8 if temporary is None else temporary)

            projection = pack(matrix(projection_name, rows=shard.partition(c["hidden_size"], alignment=64)))
            hidden_norm, norm = loader.load_tensors((hidden_norm_name + ".weight", "norm.weight"))
            for i in range(config.layers):
                p = f"layers.{i}"
                concatenate = torch.cat if online_mxfp8 else persistent_cat
                qkv = concatenate([matrix(p + ".self_attn." + n + "_proj", rows=r, temporary=True)
                    for n, r in zip(("q", "k", "v"), (q_range, kv_range, kv_range), strict=True)])
                context_kv = qkv[q_range[1] - q_range[0]:]
                context_parts.append(context_kv.clone() if online_mxfp8 else context_kv)
                del context_kv
                qkv = pack(qkv)
                output = pack(matrix(p + ".self_attn.o_proj", columns=q_range))
                qn, kn, inn, post = loader.load_tensors(tuple(p + n + ".weight" for n in
                    (".self_attn.q_norm", ".self_attn.k_norm", ".input_layernorm", ".post_attention_layernorm")))
                mlp = MLPWeights(pack(concatenate([matrix(p + ".mlp." + n + "_proj", rows=intermediate, temporary=True)
                    for n in ("gate", "up")])), pack(matrix(p + ".mlp.down_proj", columns=intermediate)))
                def conv(name):
                    if not convolutions:
                        return None
                    base, projection = loader.load_tensors((p + "." + name + ".base_kernel",
                                                           p + "." + name + ".kernel_projection.weight"))
                    return dict(base=base, projection=projection)
                layers.append(DraftLayerWeights(qkv, output, qn, kn, inn, post, mlp,
                                                conv("attention_conv"), conv("mlp_conv")))
            selector = None
            if convolutions:
                values = loader.load_tensors(("candidate_selector.hidden_projection.weight",
                    "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"))
                selector = dict(zip(("projection", "predecessor", "successor"), values, strict=True))
            markov_embedding = markov_head = None
            if projector == "dspark":
                markov_embedding = matrix("markov_head.markov_w1", temporary=False)
                markov_head = matrix("markov_head.markov_w2", temporary=False,
                                     rows=shard.partition(c["vocab_size"], alignment=64))
        context = pack((torch.cat if online_mxfp8 else persistent_cat)(context_parts))
        return cls(projection, hidden_norm, context, norm, tuple(layers), selector, markov_embedding, markov_head)


def tensor_shapes(config, projector="dflash2"):
    """Shape of every tensor a draft checkpoint with this configuration stores, by tensor name."""
    c = config.text
    h, i = c["hidden_size"], c["intermediate_size"]
    q, kv, head = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"], c["head_dim"]
    projection_name, hidden_norm_name = _PROJECTION_NAMES[projector]
    expected = {projection_name + ".weight": (h, h * len(config.target_layers)),
                hidden_norm_name + ".weight": (h,), "norm.weight": (h,)}
    if projector == "dspark":
        rank = c["markov_rank"]
        expected.update({"markov_head.markov_w1.weight": (c["vocab_size"], rank),
            "markov_head.markov_w2.weight": (c["vocab_size"], rank),
            "confidence_head.proj.weight": (1, h + rank), "confidence_head.proj.bias": (1,)})
    d = c["dflash_config"] if projector == "dflash2" else None
    if d is not None:
        expected.update({"candidate_selector.hidden_projection.weight": (d["selector_rank"], h),
            "candidate_selector.predecessor_codebook": (c["vocab_size"], d["selector_rank"]),
            "candidate_selector.successor_codebook": (c["vocab_size"], d["selector_rank"])})
    for layer in range(config.layers):
        p = f"layers.{layer}."
        for name, shape in {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h),
                "self_attn.v_proj": (kv, h), "self_attn.o_proj": (h, q), "self_attn.q_norm": (head,),
                "self_attn.k_norm": (head,), "input_layernorm": (h,), "post_attention_layernorm": (h,),
                "mlp.gate_proj": (i, h), "mlp.up_proj": (i, h), "mlp.down_proj": (h, i)}.items():
            expected[p + name + ".weight"] = shape
        if d is None:
            continue
        for name in ("attention_conv", "mlp_conv"):
            expected[p + name + ".base_kernel"] = (2, d["conv_kernel_size"], h)
            expected[p + name + ".kernel_projection.weight"] = (2 * d["conv_kernel_size"] * (h // d["conv_group_size"]), h)
    return expected


def validate_metadata(manifest, config, projector="dflash2"):
    expected, family = tensor_shapes(config, projector), _FAMILY[projector]
    if set(manifest.tensors) != set(expected):
        raise ValueError(f"{family} checkpoint tensor names disagree with its architecture")
    for name, shape in expected.items():
        tensor = manifest.tensors[name]
        if tensor.shape != shape or tensor.dtype != "BF16":
            raise ValueError(f"{name}: {family} requires BF16 storage with shape {shape}")
