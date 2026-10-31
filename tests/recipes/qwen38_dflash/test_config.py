"""The pinned DFlash2 draft configuration and the checkpoint tensor check in front of weight loading."""

from copy import deepcopy
import json

import pytest
import torch

from ignition.reference.qwen38_dflash2 import PINNED, DFlash2Reference
from ignition.recipes.qwen38_dflash.config import DRAFT_REPOSITORY, DRAFT_REVISION, DraftConfig
from ignition.recipes.qwen38_dflash.draft_weights import DraftWeights, check_tensors

# Small geometry that keeps the reference's structure; expected_shapes reads only these fields.
SMALL = {**PINNED, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 1,
         "layer_types": ["sliding_attention"], "num_attention_heads": 2, "num_key_value_heads": 1,
         "head_dim": 16, "sliding_window": 8, "vocab_size": 64,
         "dflash_config": {**PINNED["dflash_config"], "selector_rank": 8, "target_layer_ids": [0]}}


def test_pinned_config_is_accepted_and_exposes_the_draft_geometry():
    config = DraftConfig.from_dict(deepcopy(PINNED))
    assert (DRAFT_REPOSITORY, len(DRAFT_REVISION)) == ("incoai/Qwen3.8-Flash-Next-DFlash2", 40)
    assert (config.block_size, config.noise_token, config.target_layers) == (8, 248077, (5, 15, 25, 35, 45))
    assert (config.window, config.layers, config.kv_heads, config.head_dim) == (2048, 5, 8, 128)
    assert (config.hidden, config.intermediate, config.heads, config.vocabulary) == (2560, 7680, 32, 248320)
    assert (config.top_k, config.selector_rank, config.eps) == (16, 256, 1e-6)
    # The lifted draft component reads rope_theta and head_dim from the top level of `text`.
    assert config.rope_theta == config.text["rope_theta"] == 10000. and config.text["head_dim"] == 128
    assert config.text["rope_parameters"] == PINNED["rope_parameters"]


@pytest.mark.parametrize("path,value,message", [
    (("sliding_window",), 1024, "sliding_window"),
    (("dflash_config", "target_layer_ids"), [5, 15, 25, 35, 46], "target_layer_ids"),
    (("dflash_config", "mask_token_id"), 248078, "mask_token_id"),
    (("dflash_config", "block_size"), 4, "block_size"),
    (("num_hidden_layers",), 4, "num_hidden_layers"),
])
def test_rejects_a_draft_whose_geometry_differs_from_the_pinned_one(path, value, message):
    text = deepcopy(PINNED)
    node = text
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        DraftConfig.from_dict(text)


def test_load_reads_config_json(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PINNED))
    assert DraftConfig.load(tmp_path).target_layers == (5, 15, 25, 35, 45)
    (tmp_path / "config.json").write_text(json.dumps({**PINNED, "sliding_window": 4096}))
    with pytest.raises(ValueError, match="sliding_window"):
        DraftConfig.load(tmp_path)


def fabricate(path, config, *, drop=None, reshape=None, dtype=torch.bfloat16, extra=None):
    from safetensors.torch import save_file

    tensors = {name: torch.zeros(shape, dtype=dtype) for name, shape in DFlash2Reference.expected_shapes(config).items()}
    if drop is not None:
        del tensors[drop]
    if reshape is not None:
        name, shape = reshape
        tensors[name] = torch.zeros(shape, dtype=dtype)
    if extra is not None:
        tensors[extra] = torch.zeros((1,), dtype=dtype)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(config))


def inspected(path):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    return inspect_indexed_checkpoint(path).tensors


def test_single_file_checkpoint_passes_the_tensor_check(tmp_path):
    fabricate(tmp_path, SMALL)
    tensors = inspected(tmp_path)
    assert set(tensors) == set(DFlash2Reference.expected_shapes(SMALL)) and len(tensors) == 6 + 15
    check_tensors(tensors, SMALL)


@pytest.mark.parametrize("fault,message", [
    (dict(drop="layers.0.self_attn.k_proj.weight"), "missing draft tensor layers.0.self_attn.k_proj.weight"),
    (dict(reshape=("fc.weight", (32, 16))), r"fc.weight: expected BF16 \(32, 32\), got BF16 \(32, 16\)"),
    (dict(extra="layers.1.input_layernorm.weight"), "unexpected draft tensor layers.1"),
    (dict(dtype=torch.float32), "got F32"),
])
def test_tensor_check_rejects_missing_misshaped_unexpected_or_non_bf16_tensors(tmp_path, fault, message):
    fabricate(tmp_path, SMALL, **fault)
    with pytest.raises(ValueError, match=message):
        check_tensors(inspected(tmp_path), SMALL)


def test_draft_weights_check_the_checkpoint_before_opening_a_loader(tmp_path):
    # The pinned geometry is 1.2 GB, so a complete fabricated checkpoint is impractical; a file of the
    # small geometry exercises the check `DraftWeights.load` runs before any CUDA loader opens.
    fabricate(tmp_path, SMALL)
    config = DraftConfig.from_dict(deepcopy(PINNED))
    with pytest.raises(ValueError, match="draft tensor"):
        DraftWeights.load(tmp_path, config, "cpu")
    assert DraftWeights.__dataclass_fields__.keys() == {"projection", "hidden_norm", "norm", "context_kv", "layers", "selector"}
