# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 geometry validation, NVFP4 body plan, and declaration costs."""

import copy

import pytest

from ignition.recipes.elephant1.config import TEXT_PREFIX, ModelConfig
from ignition.recipes.elephant1.pool import continuation_layout, page_bytes
from ignition.recipes.elephant1.recipe import RECIPE

# baselabs/elephant-1 @ b3f8a934 config.json, trimmed to the fields the recipe reads.
CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": False, "attn_output_gate": True, "hidden_act": "silu",
        "hidden_size": 4096, "head_dim": 256, "intermediate_size": 12288,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "max_position_embeddings": 262144, "model_type": "qwen3_5_text",
        "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06, "tie_word_embeddings": False, "vocab_size": 248320,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": .25, "rope_theta": 10000000,
                            "rope_type": "default"},
    },
}


@pytest.fixture
def config():
    return ModelConfig.from_dict(copy.deepcopy(CONFIG))


def test_geometry_and_rotary(config):
    assert config.layers.count("full_attention") == 8 and len(config.layers) == 32
    assert (config.hidden, config.intermediate, config.vocabulary) == (4096, 12288, 248320)
    rope = config.text["rope_parameters"]
    assert (config.text["head_dim"] * rope["partial_rotary_factor"], rope["rope_theta"]) == (64, 10000000)
    assert not config.pack_gdn_inputs


def test_body_plan_packs_every_wide_projection_only(config):
    assert len(config.quantization) == 248
    assert all(entry == {"quant_algo": "W4A16_NVFP4", "group_size": 16}
               for entry in config.quantization.values())
    for kept in (f"{TEXT_PREFIX}.embed_tokens", "lm_head",
                 f"{TEXT_PREFIX}.layers.0.linear_attn.conv1d",
                 f"{TEXT_PREFIX}.layers.3.self_attn.q_norm",
                 f"{TEXT_PREFIX}.layers.3.post_attention_layernorm"):
        assert kept not in config.quantization
    assert config.packed(f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_a")
    assert config.packed(f"{TEXT_PREFIX}.layers.31.self_attn.q_proj")


@pytest.mark.parametrize("fault", [
    dict(architectures=["Qwen3_5ForConditionalGenerationV2"]),
    dict(text_config={**CONFIG["text_config"], "linear_num_value_heads": 48}),
    dict(text_config={**CONFIG["text_config"], "head_dim": 128}),
    dict(text_config={**CONFIG["text_config"], "attn_output_gate": False}),
    dict(text_config={**CONFIG["text_config"], "layer_types": [
        "linear_attention", "full_attention", "linear_attention", "linear_attention"] * 8}),
    dict(text_config={**CONFIG["text_config"], "rope_parameters": {
        **CONFIG["text_config"]["rope_parameters"], "mrope_section": [11, 10, 11]}}),
    dict(text_config={**CONFIG["text_config"], "rope_parameters": {
        **CONFIG["text_config"]["rope_parameters"], "rope_theta": 10000}}),
])
def test_rejects_foreign_geometry(fault):
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**copy.deepcopy(CONFIG), **fault})


def test_declaration_matches_the_layout(config):
    assert (RECIPE.context, RECIPE.page_tokens, RECIPE.max_draft_tokens) == (16384, 128, 0)
    assert RECIPE.checkpoints[0].alias == "elephant-1-9b"
    assert continuation_layout(config).row_bytes == RECIPE.ranks[1].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[1].page_bytes
