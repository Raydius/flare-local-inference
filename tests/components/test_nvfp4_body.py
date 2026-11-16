# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 body packing: reconstruction error, refusals, and the packed storage form."""

import pytest
import torch

from ignition.components.weights import nvfp4_body_parts, pack_body_nvfp4
from ignition.recipes.elephant1.config import ModelConfig
from ignition.recipes.elephant1.weights import projection_plan

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

# E2M1 magnitudes indexed by the three value bits the packer emits.
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])


def dequantize(values, scales, gain, columns):
    codes = torch.empty((values.shape[0], columns), dtype=torch.long)
    packed = values.to(torch.long)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    magnitude = E2M1[codes & 0x7]
    sign = torch.where((codes & 0x8) > 0, -1., 1.)
    return magnitude * sign * (scales.float() * gain).repeat_interleave(16, dim=1)


def test_reconstruction_stays_inside_the_declared_error_bound():
    generator = torch.Generator().manual_seed(7)
    weight = (torch.randn((256, 256), generator=generator) * .02).bfloat16()
    values, scales, gain = nvfp4_body_parts(weight, rows_per_chunk=64)
    rebuilt = dequantize(values, scales, gain, 256)
    source = weight.float()
    relative = ((rebuilt - source).norm() / source.norm()).item()
    largest = (rebuilt - source).abs().max().item() / source.abs().max().item()
    assert relative <= .12 and largest <= .2


def test_chunked_rows_match_a_single_chunk():
    generator = torch.Generator().manual_seed(11)
    weight = (torch.randn((129, 48), generator=generator) * .5).bfloat16()
    whole = dequantize(*nvfp4_body_parts(weight), 48)
    split = dequantize(*nvfp4_body_parts(weight, rows_per_chunk=40), 48)
    assert torch.equal(whole, split)


def test_final_storage_holds_codes_and_gain_but_not_row_major_scales():
    placed = []

    def allocate(shape, dtype, device):
        placed.append((tuple(shape), dtype))
        return torch.empty(shape, dtype=dtype, device=device)

    values, scales, gain = nvfp4_body_parts(torch.ones((64, 32), dtype=torch.bfloat16), allocate=allocate)
    assert placed == [((64, 16), torch.uint8), ((1,), torch.float32)]
    assert scales.shape == (64, 2) and scales.dtype is torch.float8_e4m3fn


@pytest.mark.parametrize("fault", [
    lambda: torch.randn((16, 32)),
    lambda: torch.zeros((16, 24), dtype=torch.bfloat16),
    lambda: torch.zeros((16, 32), dtype=torch.bfloat16).t(),
])
def test_refuses_matrices_it_cannot_pack(fault):
    with pytest.raises(ValueError):
        nvfp4_body_parts(fault())


def test_packing_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        pack_body_nvfp4(torch.zeros((64, 128), dtype=torch.bfloat16))


def test_recipe_packs_every_planned_projection_and_keeps_the_rest_in_checkpoint_precision():
    config = ModelConfig.from_dict({**CONFIG, "text_config": dict(TEXT)})
    parts = {part for prefixes in projection_plan(config).values() for part in prefixes}
    assert parts and all(config.packed(part) for part in parts)
    for kept in (f"model.language_model.embed_tokens", "lm_head",
                 "model.language_model.layers.0.linear_attn.conv1d",
                 "model.language_model.layers.0.linear_attn.A_log",
                 "model.language_model.layers.0.linear_attn.dt_bias",
                 "model.language_model.layers.3.self_attn.k_norm",
                 "model.language_model.layers.0.input_layernorm"):
        assert not config.packed(kept)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 packing requires CUDA")
def test_packed_storage_matches_the_projection_geometry():
    weight = torch.randn((128, 4096), device="cuda").bfloat16()
    packed = pack_body_nvfp4(weight)
    assert (packed.out_features, packed.in_features) == (128, 4096)
    assert packed.values.dtype is torch.uint8 and packed.values.shape == (128, 2048)
    assert packed.global_scale_kind == "multiplier" and packed.activation_global_scale is None
