# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer projector-less DFlash assistant geometry and target compatibility."""

from dataclasses import dataclass
import json
from pathlib import Path

from ignition.components.dflash2_geometry import DraftAttentionGeometry

REPOSITORY = "meta-models/Muse-Glimmer-30B-assistant"
REVISION = "c676cce91658f9568aaf264a416211e2782735b7"


@dataclass(frozen=True)
class DraftConfig:
    """The assistant configuration plus the target's ``vocab_size``, which the assistant omits.

    ``text`` carries no ``dflash_config``: the shared DFlash2 components then build no
    convolutions and no selector.
    """
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), target)

    @classmethod
    def from_dict(cls, text, target):
        required = dict(model_type="muse_glimmer_assistant", architectures=["MuseGlimmerAssistantModel"],
            hidden_size=6656, intermediate_size=19968, num_hidden_layers=5, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, hidden_act="silu", dtype="bfloat16", rms_norm_eps=1e-5,
            attention_dropout=0, sliding_window=2048, layer_types=["sliding_attention"] * 5, block_size=16,
            target_layer_ids=[1, 13, 25, 37, 49], mask_token_id=201818,
            rope_parameters={"rope_theta": 500000., "rope_type": "default"})
        if any(name not in text or text[name] != value for name, value in required.items()):
            raise ValueError("unsupported Muse Glimmer assistant geometry or arithmetic")
        if "dflash_config" in text or "conv_kernel_size" in text:
            raise ValueError("Muse Glimmer assistant drafts carry no convolution or selector geometry")
        if (text["hidden_size"] != target.hidden or text["intermediate_size"] != target.intermediate
                or text["rms_norm_eps"] != target.text["rms_norm_eps"]
                or text.get("vocab_size", target.vocabulary) != target.vocabulary
                or max(text["target_layer_ids"]) >= target.layers
                or text["mask_token_id"] >= target.vocabulary):
            raise ValueError("Muse Glimmer assistant draft does not match the target model")
        return cls({**text, "vocab_size": target.vocabulary})

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
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], c["rope_parameters"]["rope_theta"], c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)
