# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 checkpoint mapping, fusion order, and manifest validation."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.elephant1.config import ModelConfig
from ignition.recipes.elephant1.weights import (
    _expected_shape, _separate_query_gate, check_manifest, checkpoint_parameters, projection_plan,
)

# baselabs/elephant-1 @ b3f8a934 config.json, trimmed to the fields the recipe reads.
TEXT = {
    "attention_bias": False, "attn_output_gate": True, "hidden_act": "silu",
    "hidden_size": 4096, "head_dim": 256, "intermediate_size": 12288,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    "max_position_embeddings": 262144, "model_type": "qwen3_5_text",
    "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": False, "vocab_size": 248320,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": .25, "rope_theta": 10000000, "rope_type": "default"},
}
CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
          "text_config": TEXT}


@pytest.fixture
def config():
    return ModelConfig.from_dict({**CONFIG, "text_config": dict(TEXT)})


def test_every_text_tensor_is_mapped(config):
    names = checkpoint_parameters(config)
    assert len(names) == 427
    assert "model.language_model.layers.0.linear_attn.A_log" in names
    assert "model.language_model.layers.3.self_attn.q_norm.weight" in names
    assert "lm_head.weight" in names
    assert not any(name.startswith("model.visual.") for name in names)


def test_manifest_accepts_the_text_tensors_and_counts_the_vision_tower(config):
    visual = {f"model.visual.blocks.{index}.attn.qkv.weight" for index in range(5)}
    skipped = check_manifest(checkpoint_parameters(config) | visual, config)
    assert skipped == 5


def test_manifest_refuses_missing_and_unmapped_tensors(config):
    names = checkpoint_parameters(config)
    with pytest.raises(ValueError, match="missing"):
        check_manifest(names - {"lm_head.weight"}, config)
    with pytest.raises(ValueError, match="unmapped"):
        check_manifest(names | {"model.language_model.layers.0.linear_attn.in_proj_qkv.bias"}, config)


def test_projection_plan_fuses_checkpoint_order(config):
    plan = projection_plan(config)
    assert len(plan) == 152
    assert plan["layers.0.linear_attn.in_proj_qkvz"] == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z")
    assert plan["layers.0.linear_attn.in_proj_ba"] == (
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a")
    assert plan["layers.3.self_attn.qgkv_proj"] == (
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj")
    assert plan["layers.31.mlp.gate_up"] == ("model.language_model.layers.31.mlp.gate_proj",
                                             "model.language_model.layers.31.mlp.up_proj")


def test_prepared_shapes_match_the_checkpoint(config):
    assert _expected_shape(config, "layers.3.self_attn.qgkv_proj") == (10240, 4096)
    assert _expected_shape(config, "layers.3.self_attn.o_proj") == (4096, 4096)
    assert _expected_shape(config, "layers.0.mlp.gate_up") == (24576, 4096)
    assert _expected_shape(config, "layers.0.mlp.down_proj") == (4096, 12288)
    assert _expected_shape(config, "layers.0.linear_attn.in_proj_qkvz") == (12288, 4096)
    assert _expected_shape(config, "layers.0.linear_attn.in_proj_ba") == (64, 4096)
    assert _expected_shape(config, "layers.0.linear_attn.out_proj") == (4096, 4096)


def test_query_and_gate_rows_become_contiguous_blocks():
    config = SimpleNamespace(text={"num_attention_heads": 2, "head_dim": 2}, hidden=1)
    combined = torch.arange(10, dtype=torch.bfloat16).view(10, 1)
    assert _separate_query_gate(combined, config).view(-1).tolist() == [
        0., 1., 4., 5., 2., 3., 6., 7., 8., 9.]
