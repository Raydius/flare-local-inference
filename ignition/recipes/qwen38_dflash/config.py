"""Pinned DFlash2 draft repository and its validated configuration."""

from dataclasses import dataclass
import json
from pathlib import Path

DRAFT_REPOSITORY = "incoai/Qwen3.8-Flash-Next-DFlash2"
DRAFT_REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, c):
        # Declarations import this module's pins without torch; the reference needs it.
        from ignition.reference.qwen38_dflash2 import validate_config

        validate_config(c, strict=True)
        # The lifted draft component's rotary reads a top-level ``rope_theta``; the DFlash2
        # checkpoint stores it under ``rope_parameters``.
        return cls({**c, "rope_theta": float(c["rope_parameters"]["rope_theta"])})

    @property
    def block_size(self):
        return self.text["dflash_config"]["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def top_k(self):
        return self.text["dflash_config"]["selector_top_k"]

    @property
    def selector_rank(self):
        return self.text["dflash_config"]["selector_rank"]

    @property
    def window(self):
        return self.text["sliding_window"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def heads(self):
        return self.text["num_attention_heads"]

    @property
    def kv_heads(self):
        return self.text["num_key_value_heads"]

    @property
    def head_dim(self):
        return self.text["head_dim"]

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
    def eps(self):
        return self.text["rms_norm_eps"]

    @property
    def rope_theta(self):
        return self.text["rope_theta"]
