# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer text declaration geometry, NVFP4 declarations, and assistant compatibility."""

import copy
import json
import os
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.muse_glimmer_nvfp4.config import ModelConfig
from ignition.recipes.muse_glimmer_nvfp4.draft_config import DraftConfig
from ignition.recipes.muse_glimmer_nvfp4.recipe import RECIPE
from ignition.recipes.muse_spark.pool import continuation_layout, page_bytes

# original-el8/Muse-Glimmer-30B-NVFP4 @ b47e1e9f config.json, trimmed to the fields the config reads.
CONFIG = {
    "model_type": "muse_glimmer", "architectures": ["MuseGlimmerForConditionalGeneration"],
    "text_config": {
        "model_type": "muse_glimmer_text", "vocab_size": 202048, "hidden_size": 6656, "intermediate_size": 19968,
        "num_hidden_layers": 52, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "hidden_activation": "silu", "rms_norm_eps": 1e-05, "post_norm_eps": 1e-08, "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "attention_bias": False,
        "sliding_window": 2048, "final_logit_softcapping": 20.0, "qk_scale_factor": 3.87,
        "output_multiplier": 0.19611613513818404,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 13,
        "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0] * 13,
    },
    "quantization_config": {
        "config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}},
        "ignore": ["lm_head", "model.vision_adapter*", "model.vision_projection", "model.vision_tower*"],
        "quant_algo": "NVFP4", "quant_method": "modelopt",
    },
}

# meta-models/Muse-Glimmer-30B-assistant @ c676cce9 config.json.
ASSISTANT = {
    "architectures": ["MuseGlimmerAssistantModel"], "attention_dropout": 0, "block_size": 16, "dtype": "bfloat16",
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 6656, "intermediate_size": 19968,
    "layer_types": ["sliding_attention"] * 5, "mask_token_id": 201818, "model_type": "muse_glimmer_assistant",
    "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "sliding_window": 2048,
    "target_layer_ids": [1, 13, 25, 37, 49],
}


@pytest.fixture
def config():
    return ModelConfig.from_dict(copy.deepcopy(CONFIG))


def test_layer_kinds_rotary_and_windows(config):
    kinds = [config.attention(i).window for i in range(config.layers)]
    assert kinds.count(2048) == 39 and kinds.count(-1) == 13
    assert config.attention(0).rotary_dim == 128 and config.attention(0).theta == 500000.
    assert config.attention(3).rotary_dim == 0 and config.attention(3).theta == 0
    assert config.attention(0).local(TensorParallelShard(0, 1)) == (32, 2)
    assert (config.hidden, config.intermediate, config.vocabulary) == (6656, 19968, 202048)
    config.validate_shard(TensorParallelShard(0, 1))
    assert config.vocabulary_partition(TensorParallelShard(0, 1)) == (0, 202048, 202048)


def test_quantization_covers_the_eight_text_projections_per_layer_only(config):
    assert len(config.quantization) == 416
    assert all(entry == {"quant_algo": "NVFP4", "group_size": 16} for entry in config.quantization.values())
    assert "model.language_model.layers.51.self_attn.gate_proj" in config.quantization
    assert "model.language_model.layers.0.mlp.down_proj" in config.quantization
    assert "lm_head" not in config.quantization
    assert not any(name.startswith("model.vision") for name in config.quantization)


def test_declaration_matches_the_layout(config):
    shard = TensorParallelShard(0, 1)
    assert (RECIPE.context, RECIPE.concurrency, RECIPE.page_tokens, RECIPE.max_draft_tokens) == (131072, 4, 128, 15)
    assert RECIPE.max_images == 4 and RECIPE.language_model_only
    assert RECIPE.checkpoints[0].alias == "muse-glimmer-30b-nvfp4"
    assert continuation_layout(config, shard).row_bytes == RECIPE.costs(1, None).state_row_bytes
    assert page_bytes(config, shard) == RECIPE.costs(1, None).page_bytes
    drafting = ModelConfig(config.text, config.quantization, DraftConfig.from_dict(copy.deepcopy(ASSISTANT), config))
    assert continuation_layout(drafting, shard).row_bytes == RECIPE.costs(1, RECIPE.speculation).state_row_bytes


@pytest.mark.parametrize("fault", [
    dict(model_type="muse_spark"), dict(architectures=["MuseGlimmerModel"]),
    dict(text_config={**CONFIG["text_config"], "head_dim": 64}),
    dict(text_config={**CONFIG["text_config"], "layer_rope_theta": [500000.0] * 52}),
    dict(quantization_config={**CONFIG["quantization_config"], "ignore": []}),
])
def test_rejects_foreign_geometry(fault):
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**copy.deepcopy(CONFIG), **fault})


def test_assistant_config_is_accepted_and_carries_the_target_vocabulary(config):
    draft = DraftConfig.from_dict(copy.deepcopy(ASSISTANT), config)
    assert draft.layers == 5 and draft.block_size == 16 and draft.noise_token == 201818
    assert draft.target_layers == (1, 13, 25, 37, 49)
    assert draft.text["vocab_size"] == 202048 and "dflash_config" not in draft.text
    a = draft.attention
    assert (a.heads, a.kv_heads, a.key_dim, a.rotary_dim, a.theta, a.window) == (32, 8, 128, 128, 500000., 2048)
    draft.validate_shard(TensorParallelShard(0, 1))


@pytest.mark.parametrize("fault", [
    dict(mask_token_id=201817), dict(block_size=8), dict(target_layer_ids=[3, 19, 31, 47, 59]),
    dict(hidden_size=8192), dict(vocab_size=201088),
])
def test_assistant_config_rejects_a_mismatched_draft(config, fault):
    with pytest.raises(ValueError):
        DraftConfig.from_dict({**copy.deepcopy(ASSISTANT), **fault}, config)


SNAPSHOT = os.environ.get("FLARE_MUSE_GLIMMER_SNAPSHOT")


@pytest.mark.skipif(not SNAPSHOT, reason="FLARE_MUSE_GLIMMER_SNAPSHOT names no checkpoint")
def test_pinned_checkpoint_matches_the_fixture_and_declared_stored_bytes():
    snapshot = Path(SNAPSHOT)
    raw = json.loads((snapshot / "config.json").read_text())
    pinned, fixture = ModelConfig.from_dict(raw), ModelConfig.from_dict(copy.deepcopy(CONFIG))
    # The fixture omits Transformers' boilerplate keys; every key it carries must match the checkpoint.
    assert {key: pinned.text[key] for key in fixture.text} == fixture.text
    assert (pinned.layers, pinned.quantization, pinned.draft) == (fixture.layers, fixture.quantization, fixture.draft)
    assert [pinned.attention(i) for i in range(pinned.layers)] == [fixture.attention(i) for i in range(fixture.layers)]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == RECIPE.checkpoints[0].stored_bytes
