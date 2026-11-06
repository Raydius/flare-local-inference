# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared ModelOpt NVFP4 quantization_config parser."""

import pytest

from ignition.preparation.quantization import matches_pattern, modelopt_nvfp4_layers

NVFP4_ENTRY = {"quant_algo": "NVFP4", "group_size": 16}
GLIMMER_LAYERS = 52
GLIMMER_LAYER_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.gate_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
GLIMMER_VISION_CANDIDATES = (
    "model.vision_tower.blocks.0.attn.qkv",
    "model.vision_tower.blocks.0.mlp.fc1",
    "model.vision_adapter.fc1",
    "model.vision_adapter.fc2",
    "model.vision_projection",
)


def _modelopt_group(*, input_activations=True):
    spec = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    group = {"weights": dict(spec), "targets": ["Linear"]}
    if input_activations:
        group["input_activations"] = dict(spec)
    return group


def _glimmer_raw(*, input_activations=True):
    # The quantization block of the Glimmer checkpoint's config.json, verbatim.
    return {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "quantization_config": {
            "config_groups": {"group_0": _modelopt_group(input_activations=input_activations)},
            "ignore": ["lm_head", "model.vision_adapter*", "model.vision_projection", "model.vision_tower*"],
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.47.0.dev48+g41b18a856"},
            "quant_method": "modelopt",
        },
    }


def _glimmer_candidates():
    names = ["lm_head", *GLIMMER_VISION_CANDIDATES]
    for index in range(GLIMMER_LAYERS):
        names.extend(f"model.language_model.layers.{index}.{proj}" for proj in GLIMMER_LAYER_PROJECTIONS)
    return names


def _gemma_dense_31b_raw():
    # tests/recipes/test_gemma4.py::_dense_31b_raw(nvfp4=True)'s quantization block plus
    # the input_activations spec the real ModelOpt checkpoints carry.
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {"group_0": _modelopt_group()},
            "ignore": ["lm_head", "*embed_vision*", "*self_attn*", "*vision_tower*"],
        },
    }


def _gemma_dense_31b_candidates():
    # gemma4/config.py:_candidate_linear_names over the dense 31B geometry: lm_head, then
    # q/o/gate/up/down per layer plus k_proj (attention_k_eq_v drops v_proj, no KV sharing).
    names = ["lm_head"]
    for index in range(60):
        prefix = f"model.language_model.layers.{index}"
        names.extend((
            f"{prefix}.self_attn.q_proj",
            f"{prefix}.self_attn.o_proj",
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
            f"{prefix}.mlp.down_proj",
            f"{prefix}.self_attn.k_proj",
        ))
    return names


def test_matches_pattern_glob_and_regex():
    assert matches_pattern("model.vision_tower.blocks.0.attn.qkv", "model.vision_tower*")
    assert not matches_pattern("model.language_model.layers.0.mlp.up_proj", "model.vision_tower*")
    assert matches_pattern("lm_head", "lm_head")
    assert not matches_pattern("lm_head.weight", "lm_head")
    assert matches_pattern("model.language_model.layers.12.self_attn.q_proj", r"re:layers\.1\d\.self_attn")
    assert not matches_pattern("model.language_model.layers.2.self_attn.q_proj", r"re:layers\.1\d\.self_attn")


def test_glimmer_config_keeps_every_text_projection_and_drops_ignored_names():
    candidates = _glimmer_candidates()
    layers = modelopt_nvfp4_layers(_glimmer_raw(), candidates)

    # The fixture passes lm_head and five vision names alongside 8 x 52 text projections; the
    # checkpoint's ignore globs remove exactly lm_head and the vision names.
    ignored = 1 + len(GLIMMER_VISION_CANDIDATES)
    assert len(candidates) == GLIMMER_LAYERS * len(GLIMMER_LAYER_PROJECTIONS) + ignored
    assert len(layers) == len(candidates) - ignored == 416
    assert "lm_head" not in layers
    assert not any(name.startswith("model.vision_") for name in layers)
    assert all(name.startswith("model.language_model.layers.") for name in layers)
    assert set(layers) == set(candidates[ignored:])
    assert all(entry == NVFP4_ENTRY for entry in layers.values())
    assert list(layers) == candidates[ignored:]


def test_gemma_dense_config_matches_the_former_recipe_branch():
    # Captured from gemma4/config.py:_parse_quantization before the branch was lifted:
    # 180 entries, the three MLP projections of each of the 60 layers, all NVFP4 group 16.
    layers = modelopt_nvfp4_layers(_gemma_dense_31b_raw(), _gemma_dense_31b_candidates())
    expected = {
        f"model.language_model.layers.{index}.mlp.{proj}": NVFP4_ENTRY
        for index in range(60)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    assert layers == expected
    assert len(layers) == 60 * 3 == 180
    assert "model.language_model.layers.0.self_attn.q_proj" not in layers
    assert "lm_head" not in layers


def test_weight_only_group_is_rejected():
    with pytest.raises(ValueError, match="input_activations"):
        modelopt_nvfp4_layers(_glimmer_raw(input_activations=False), _glimmer_candidates())


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda q: q.__setitem__("quant_algo", "FP8"), "expected modelopt NVFP4"),
        (lambda q: q.__setitem__("config_groups", {}), "config_groups.group_0"),
        (lambda q: q["config_groups"]["group_0"]["weights"].__setitem__("group_size", 32), "weights"),
        (lambda q: q["config_groups"]["group_0"]["input_activations"].__setitem__("num_bits", 8), "input_activations"),
        (lambda q: q.__setitem__("ignore", "lm_head"), "ignore list"),
        (lambda q: q.__setitem__("ignore", ["lm_head", 3]), "ignore list"),
    ],
)
def test_malformed_blocks_are_rejected(mutate, message):
    raw = _glimmer_raw()
    mutate(raw["quantization_config"])
    with pytest.raises(ValueError, match=message):
        modelopt_nvfp4_layers(raw, _glimmer_candidates())
