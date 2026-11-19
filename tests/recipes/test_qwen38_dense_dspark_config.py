# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B DSpark configuration validation, rotary parameters and checkpoint tensor coverage."""

import copy
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from ignition.preparation.checkpoint import IndexedCheckpoint, TensorMetadata
from ignition.recipes.qwen38_dense import dspark_config, dspark_weights

# config.json of RadixArk/Qwen3.8-27B-DSpark at revision b9a5dbdf03bc999c6c73c426b19c2d9041cea393.
_NESTED = {'attention_mode': 'gqa',
           'confidence_head_alpha': 1.0,
           'confidence_head_with_markov': True,
           'enable_confidence_head': True,
           'markov_head_type': 'vanilla',
           'markov_rank': 256,
           'mask_token_id': 248070,
           'projector_type': 'dspark',
           'target_layer_ids': [5, 19, 33, 47, 61]}
DSPARK_CONFIG = {'architectures': ['DSparkDraftModel'],
 'attention_bias': False,
 'attention_dropout': 0.0,
 'attention_mode': 'gqa',
 'auto_map': {'AutoModel': 'dspark.DSparkDraftModel'},
 'block_size': 7,
 'bos_token_id': None,
 'confidence_head_alpha': 1.0,
 'confidence_head_with_markov': True,
 'dflash_config': copy.deepcopy(_NESTED),
 'draft_vocab_size': 248320,
 'dspark_config': copy.deepcopy(_NESTED),
 'dtype': 'bfloat16',
 'enable_confidence_head': True,
 'eos_token_id': 248044,
 'head_dim': 128,
 'hidden_act': 'silu',
 'hidden_size': 5120,
 'initializer_range': 0.02,
 'intermediate_size': 17408,
 'layer_types': ['full_attention'] * 5,
 'markov_head_type': 'vanilla',
 'markov_rank': 256,
 'mask_token_id': 248070,
 'max_position_embeddings': 262144,
 'max_window_layers': 5,
 'model_type': 'qwen3',
 'num_attention_heads': 32,
 'num_hidden_layers': 5,
 'num_key_value_heads': 8,
 'num_target_layers': 64,
 'pad_token_id': 248044,
 'projector_type': 'dspark',
 'rms_norm_eps': 1e-06,
 'rope_parameters': {'beta_fast': 32.0,
                     'beta_slow': 1.0,
                     'factor': 32.0,
                     'original_max_position_embeddings': 8192,
                     'rope_theta': 10000000,
                     'rope_type': 'yarn'},
 'rope_scaling': {'beta_fast': 32.0,
                  'beta_slow': 1.0,
                  'factor': 32.0,
                  'original_max_position_embeddings': 8192,
                  'rope_type': 'yarn'},
 'rope_theta': 10000000,
 'sliding_window': None,
 'target_hidden_size': None,
 'target_layer_ids': [5, 19, 33, 47, 61],
 'tie_word_embeddings': False,
 'training_block_size': 16,
 'transformers_version': '5.8.1',
 'use_cache': True,
 'use_sliding_window': False,
 'vocab_size': 248320}

# Tensor names and shapes of the published model.safetensors header at the same revision; all BF16.
_LAYER_SHAPES = {"input_layernorm.weight": (5120,), "mlp.down_proj.weight": (5120, 17408),
    "mlp.gate_proj.weight": (17408, 5120), "mlp.up_proj.weight": (17408, 5120),
    "post_attention_layernorm.weight": (5120,), "self_attn.k_norm.weight": (128,),
    "self_attn.k_proj.weight": (1024, 5120), "self_attn.o_proj.weight": (5120, 4096),
    "self_attn.q_norm.weight": (128,), "self_attn.q_proj.weight": (4096, 5120),
    "self_attn.v_proj.weight": (1024, 5120)}
PUBLISHED_SHAPES = {"confidence_head.proj.bias": (1,), "confidence_head.proj.weight": (1, 5376),
    "fc.weight": (5120, 25600), "hidden_norm.weight": (5120,), "norm.weight": (5120,),
    "markov_head.markov_w1.weight": (248320, 256), "markov_head.markov_w2.weight": (248320, 256),
    **{f"layers.{layer}.{name}": shape for layer in range(5) for name, shape in _LAYER_SHAPES.items()}}


class Target:
    """The target geometry the draft configuration is validated against."""
    hidden = 5120
    vocabulary = 248320
    layers = ("full_attention",) * 64


def published():
    return copy.deepcopy(DSPARK_CONFIG)


def header(shapes=PUBLISHED_SHAPES, dtype="BF16"):
    """Checkpoint metadata with the given tensors and no payload."""
    tensors = {name: TensorMetadata(name, "model.safetensors", 0, 0, shape, dtype) for name, shape in shapes.items()}
    return IndexedCheckpoint(Path("draft"), tensors, ())


def test_published_configuration_loads_with_the_pinned_geometry(tmp_path):
    import json

    assert (dspark_config.REPOSITORY, dspark_config.REVISION) == (
        "RadixArk/Qwen3.8-27B-DSpark", "b9a5dbdf03bc999c6c73c426b19c2d9041cea393")
    (tmp_path / "config.json").write_text(json.dumps(DSPARK_CONFIG))
    draft = dspark_config.load(tmp_path, Target)
    assert draft == dspark_config.from_dict(published(), Target)
    assert (draft.layers, draft.block_size, draft.noise_token, draft.markov_rank) == (5, 7, 248070, 256)
    assert draft.target_layers == (5, 19, 33, 47, 61)
    a = draft.attention
    assert (a.heads, a.kv_heads, a.key_dim, a.value_dim, a.rotary_dim, a.theta, a.window) == (
        32, 8, 128, 128, 128, 10000000, None)


@pytest.mark.parametrize("name,value", [
    ("architectures", ["DFlash2DraftModel"]),
    ("layer_types", ["sliding_attention"] * 5),
    ("layer_types", ["full_attention"] * 4),
    ("num_hidden_layers", 4),
    ("hidden_size", 4096),
    ("intermediate_size", 16384),
    ("num_attention_heads", 40),
    ("num_key_value_heads", 4),
    ("head_dim", 160),
    ("vocab_size", 248321),
    ("num_target_layers", 63),
    ("target_layer_ids", [5, 19, 33, 47, 62]),
    ("mask_token_id", 248069),
    ("block_size", 8),
    ("markov_rank", 128),
    ("markov_head_type", "gated"),
    ("enable_confidence_head", False),
    ("confidence_head_with_markov", False),
    ("confidence_head_with_markov", 1),
    ("sliding_window", 2048),
    ("rms_norm_eps", 1e-5),
    ("tie_word_embeddings", True),
    ("rope_parameters.rope_type", "default"),
    ("rope_parameters.factor", 16.0),
    ("rope_parameters.original_max_position_embeddings", 32768),
    ("rope_parameters.beta_fast", 16.0),
    ("rope_parameters.beta_slow", 2.0),
    ("rope_parameters.rope_theta", 1000000),
    ("dflash_config.target_layer_ids", [1, 16, 31, 46, 61]),
    ("dflash_config.mask_token_id", 248069),
])
def test_each_validated_field_is_named_when_it_differs(name, value):
    text = published()
    section, _, key = name.rpartition(".")
    (text[section] if section else text)[key] = value
    with pytest.raises(ValueError, match=rf"DSpark requires {name} ") as error:
        dspark_config.from_dict(text, Target)
    assert repr(value) in str(error.value)


@pytest.mark.parametrize("name", [*dspark_config.PROFILE, "rope_parameters", "dflash_config",
    *(f"rope_parameters.{key}" for key in dspark_config.ROTARY),
    *(f"dflash_config.{key}" for key in dspark_config.NESTED)])
def test_each_validated_field_is_named_when_it_is_absent(name):
    text = published()
    section, _, key = name.rpartition(".")
    del (text[section] if section else text)[key]
    with pytest.raises(ValueError, match=rf"DSpark requires (a )?{name} "):
        dspark_config.from_dict(text, Target)


@pytest.mark.parametrize("name,attribute,value", [("hidden_size", "hidden", 4096),
    ("vocab_size", "vocabulary", 151936), ("num_target_layers", "layers", ("full_attention",) * 48)])
def test_target_geometry_mismatch_names_the_field(name, attribute, value):
    target = type("OtherTarget", (Target,), {attribute: value})
    with pytest.raises(ValueError, match=rf"DSpark {name} .* differs from the target's"):
        dspark_config.from_dict(published(), target)


def test_rotary_gain_and_frequencies_follow_yarn():
    from transformers import PretrainedConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    draft = dspark_config.from_dict(published(), Target)
    assert draft.rotary_gain == 1.3465735902799727
    frequencies = draft.frequencies("cpu")
    assert frequencies.shape == (64,) and frequencies.dtype == torch.float32
    # Pairs completing at least beta_fast rotations over 8192 positions keep their frequency;
    # pairs completing at most beta_slow rotations are divided by the factor.
    plain = 1 / 10000000 ** (torch.arange(64, dtype=torch.float32) / 64)
    assert frequencies[0] == plain[0] == 1
    assert torch.equal(frequencies[-1], plain[-1] / 32)
    reference = PretrainedConfig(rope_parameters=dict(DSPARK_CONFIG["rope_parameters"]), hidden_size=5120,
        num_attention_heads=32, head_dim=128, max_position_embeddings=262144)
    inverse, factor = ROPE_INIT_FUNCTIONS["yarn"](reference, device="cpu")
    torch.testing.assert_close(frequencies, inverse, rtol=1e-6, atol=0)
    assert factor == pytest.approx(draft.rotary_gain, rel=1e-15)


def test_published_header_maps_completely():
    draft = dspark_config.from_dict(published(), Target)
    assert len(PUBLISHED_SHAPES) == 62
    assert dspark_weights.tensor_shapes(draft) == PUBLISHED_SHAPES
    dspark_weights.validate_metadata(header(), draft)
    # Every tensor outside the confidence head is read by the loader's prefix selection.
    unread = {name for name in PUBLISHED_SHAPES if not name.startswith(dspark_weights._LOADED_PREFIXES)}
    assert unread == {"confidence_head.proj.weight", "confidence_head.proj.bias"}


@pytest.mark.parametrize("name", ["lm_head.weight", "embed_tokens.weight", "layers.5.input_layernorm.weight",
    "layers.0.attention_conv.base_kernel", "candidate_selector.predecessor_codebook", "confidence_head.proj.scale"])
def test_extra_tensor_is_rejected_by_name(name):
    draft = dspark_config.from_dict(published(), Target)
    with pytest.raises(ValueError, match=rf"tensor names disagree.*unexpected=\['{name}'\]"):
        dspark_weights.validate_metadata(header({**PUBLISHED_SHAPES, name: (5120,)}), draft)


@pytest.mark.parametrize("name", ["fc.weight", "layers.3.self_attn.k_norm.weight", "markov_head.markov_w2.weight",
                                  "confidence_head.proj.bias"])
def test_missing_tensor_is_rejected_by_name(name):
    draft = dspark_config.from_dict(published(), Target)
    shapes = {key: shape for key, shape in PUBLISHED_SHAPES.items() if key != name}
    with pytest.raises(ValueError, match=rf"tensor names disagree.*missing=\['{name}'\]"):
        dspark_weights.validate_metadata(header(shapes), draft)


@pytest.mark.parametrize("name,shape", [("fc.weight", (5120, 20480)), ("layers.0.self_attn.o_proj.weight", (5120, 5120)),
    ("markov_head.markov_w1.weight", (248320, 512)), ("confidence_head.proj.weight", (1, 5120)),
    ("confidence_head.proj.bias", (2,))])
def test_misshaped_tensor_is_rejected_by_name(name, shape):
    draft = dspark_config.from_dict(published(), Target)
    with pytest.raises(ValueError, match=rf"^{name}: .*BF16 storage with shape"):
        dspark_weights.validate_metadata(header({**PUBLISHED_SHAPES, name: shape}), draft)


def test_non_bf16_tensor_is_rejected():
    draft = dspark_config.from_dict(published(), Target)
    manifest = header()
    tensors = dict(manifest.tensors)
    tensors["norm.weight"] = replace(tensors["norm.weight"], dtype="F32")
    with pytest.raises(ValueError, match="^norm.weight: .*found F32"):
        dspark_weights.validate_metadata(replace(manifest, tensors=tensors), draft)


def test_rebuild_manifest_names_every_draft_weight():
    from ignition.components.dflash2_weights import DraftLayerWeights
    from ignition.components.weights import MLPWeights
    from ignition.preparation.rebuild_manifest import declare_weights, declared_views

    def tensor():
        return torch.zeros(2, 2)

    layer = DraftLayerWeights(tensor(), tensor(), tensor(), tensor(), tensor(), tensor(),
        MLPWeights(tensor(), tensor()), None, None)
    draft = dspark_weights.DraftWeights(tensor(), tensor(), tensor(), tensor(), tensor(), tensor(), (layer,))
    names = [name for name, *_ in declared_views(declare_weights({"draft": draft}, components=("draft",)))]
    assert sorted(names) == sorted(["draft.projection", "draft.hidden_norm", "draft.context_kv", "draft.norm",
        "draft.markov_embedding", "draft.markov_head", "draft.layers[0].qkv", "draft.layers[0].output",
        "draft.layers[0].query_norm", "draft.layers[0].key_norm", "draft.layers[0].input_norm",
        "draft.layers[0].post_attention_norm", "draft.layers[0].mlp.gate_up", "draft.layers[0].mlp.down"])


def test_published_checkpoint_header_and_configuration_validate():
    """Reads the downloaded checkpoint's ``config.json`` and safetensors header, and no tensor payload."""
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    snapshot = os.environ.get("FLARE_QWEN38_DSPARK_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires FLARE_QWEN38_DSPARK_SNAPSHOT naming the RadixArk/Qwen3.8-27B-DSpark snapshot")
    import json

    assert json.loads((Path(snapshot) / "config.json").read_text()) == DSPARK_CONFIG
    draft = dspark_config.load(snapshot, Target)
    manifest = inspect_indexed_checkpoint(snapshot)
    assert {name: tuple(tensor.shape) for name, tensor in manifest.tensors.items()} == PUBLISHED_SHAPES
    dspark_weights.validate_metadata(manifest, draft)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA_VISIBLE_DEVICES")
def test_load_fuses_projections_and_stacks_context_rows(tmp_path):
    from safetensors.torch import save_file

    from ignition.components.parallel import TensorParallelShard

    small = dspark_config.DSparkConfig({**published(), "hidden_size": 128, "intermediate_size": 256,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 256, "markov_rank": 16})
    generator = torch.Generator().manual_seed(0)
    source = {name: torch.randn(shape, generator=generator).to(torch.bfloat16)
              for name, shape in dspark_weights.tensor_shapes(small).items()}
    save_file(source, str(tmp_path / "model.safetensors"))
    weights = dspark_weights.DraftWeights.load(tmp_path, small, torch.device("cuda:0"), TensorParallelShard(0, 1))

    def check(actual, name):
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.equal(actual.cpu(), source[name]), name

    check(weights.projection, "fc.weight")
    check(weights.hidden_norm, "hidden_norm.weight")
    check(weights.norm, "norm.weight")
    check(weights.markov_embedding, "markov_head.markov_w1.weight")
    check(weights.markov_head, "markov_head.markov_w2.weight")
    assert len(weights.layers) == 5
    for index, layer in enumerate(weights.layers):
        p = f"layers.{index}."
        check(layer.qkv[:128], p + "self_attn.q_proj.weight")
        check(layer.qkv[128:192], p + "self_attn.k_proj.weight")
        check(layer.qkv[192:], p + "self_attn.v_proj.weight")
        assert torch.equal(weights.context_kv[index * 128:(index + 1) * 128], layer.qkv[128:])
        check(layer.output, p + "self_attn.o_proj.weight")
        check(layer.query_norm, p + "self_attn.q_norm.weight")
        check(layer.key_norm, p + "self_attn.k_norm.weight")
        check(layer.input_norm, p + "input_layernorm.weight")
        check(layer.post_attention_norm, p + "post_attention_layernorm.weight")
        check(layer.mlp.gate_up[:256], p + "mlp.gate_proj.weight")
        check(layer.mlp.gate_up[256:], p + "mlp.up_proj.weight")
        check(layer.mlp.down, p + "mlp.down_proj.weight")
        assert layer.attention_conv is None and layer.mlp_conv is None
