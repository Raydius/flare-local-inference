# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pinned DSpark checkpoint, draft geometry and YaRN rotary parameters for Qwen3.8-27B."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from ignition.components.dflash2_geometry import DraftAttentionGeometry

REPOSITORY = "RadixArk/Qwen3.8-27B-DSpark"
REVISION = "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"
ROTARY = dict(rope_type="yarn", factor=32, original_max_position_embeddings=8192, beta_fast=32,
    beta_slow=1, rope_theta=10000000)
PROFILE = dict(architectures=["DSparkDraftModel"], layer_types=["full_attention"] * 5, num_hidden_layers=5,
    hidden_size=5120, intermediate_size=17408, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=248320, num_target_layers=64, target_layer_ids=[5, 19, 33, 47, 61], mask_token_id=248070,
    block_size=7, markov_rank=256, markov_head_type="vanilla", enable_confidence_head=True,
    confidence_head_with_markov=True, sliding_window=None, rms_norm_eps=1e-6, tie_word_embeddings=False)
# The published model reads the target layers and the mask token from this nested section.
NESTED = dict(target_layer_ids=PROFILE["target_layer_ids"], mask_token_id=PROFILE["mask_token_id"])
# Storage precision of the draft's paged key and value history, the precision of the target's pages.
PAGE_DTYPE = torch.float8_e4m3fn
_ABSENT = object()


def _require(section, name, value, label):
    actual = section.get(name, _ABSENT)
    # A boolean equals 0 or 1 under ``==``; the type comparison keeps the two apart.
    if actual is _ABSENT or isinstance(actual, bool) != isinstance(value, bool) or actual != value:
        found = "no value" if actual is _ABSENT else repr(actual)
        raise ValueError(f"Qwen3.8-27B DSpark requires {label} {value!r}, found {found}")


@dataclass(frozen=True)
class DSparkConfig:
    """Validated geometry of the five full-attention DSpark draft layers.

    The draft conditions on the target hidden states leaving ``target_layers`` and proposes
    ``block_size`` tokens per round. It has no embedding or vocabulary head of its own.
    """

    text: dict

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def target_layers(self):
        return tuple(self.text["target_layer_ids"])

    @property
    def block_size(self):
        return self.text["block_size"]

    @property
    def noise_token(self):
        return self.text["mask_token_id"]

    @property
    def markov_rank(self):
        return self.text["markov_rank"]

    @property
    def attention(self):
        """Grouped-query head geometry; ``window`` is ``None`` because every layer attends to the whole prefix."""
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], c["rope_parameters"]["rope_theta"], None)

    @property
    def rotary_gain(self):
        """YaRN attention factor ``0.1 * ln(factor) + 1``, multiplied into each rotated query and key."""
        return 0.1 * math.log(self.text["rope_parameters"]["factor"]) + 1

    def frequencies(self, device):
        """FP32 inverse frequencies of the ``head_dim // 2`` rotary pairs, interpolated for the YaRN factor."""
        from ignition.components.rotary import interpolated_frequencies

        rope = self.text["rope_parameters"]
        return interpolated_frequencies(rope["rope_theta"], self.text["head_dim"] // 2, rope["factor"],
            rope["original_max_position_embeddings"], rope["beta_fast"], rope["beta_slow"], device)

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)
        shard.partition(self.text["vocab_size"], alignment=64)


def from_dict(text, target):
    """Validate a DSpark ``config.json`` object against the pinned profile and the target model."""
    if not isinstance(text, dict):
        raise ValueError("Qwen3.8-27B DSpark requires a configuration object")
    for name, value in PROFILE.items():
        _require(text, name, value, name)
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("Qwen3.8-27B DSpark requires a rope_parameters object")
    for name, value in ROTARY.items():
        _require(rope, name, value, f"rope_parameters.{name}")
    nested = text.get("dflash_config")
    if not isinstance(nested, dict):
        raise ValueError("Qwen3.8-27B DSpark requires a dflash_config object")
    for name, value in NESTED.items():
        _require(nested, name, value, f"dflash_config.{name}")
    for name, value in (("hidden_size", target.hidden), ("vocab_size", target.vocabulary),
                        ("num_target_layers", len(target.layers))):
        if text[name] != value:
            raise ValueError(f"Qwen3.8-27B DSpark {name} {text[name]!r} differs from the target's {value!r}")
    return DSparkConfig(text)


def load(snapshot, target):
    return from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)
