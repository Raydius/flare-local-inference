# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projector-less DFlash drafter checkpoint checks and block-16 scratch through the shared DFlash2 components."""

from dataclasses import MISSING, dataclass, fields, replace
import os

import pytest
import torch

from ignition.components.dflash2 import DraftScratch
from ignition.components.dflash2_geometry import DraftAttentionGeometry
from ignition.components.dflash2_weights import DraftWeights, validate_metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import GraphBufferPool

# meta-models/Muse-Glimmer-30B-assistant @ c676cce9: hidden 6656, intermediate 19968, 32/8 heads x 128,
# five sliding layers tapping target layers [1, 13, 25, 37, 49], block 16, mask token 201818.
ASSISTANT = dict(hidden_size=6656, intermediate_size=19968, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, num_hidden_layers=5, rms_norm_eps=1e-5, sliding_window=2048, vocab_size=202048,
    block_size=16, target_layer_ids=[1, 13, 25, 37, 49], mask_token_id=201818)
SMALL = {**ASSISTANT, "hidden_size": 128, "intermediate_size": 256, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 256}


@dataclass(frozen=True)
class StubDraftConfig:
    """The slice of a draft configuration the shared DFlash2 components read."""
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
    def attention(self):
        c = self.text
        return DraftAttentionGeometry(c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
            c["head_dim"], c["head_dim"], 500000., c["sliding_window"])

    def validate_shard(self, shard):
        self.attention.local(shard)
        shard.partition(self.text["hidden_size"], alignment=64)
        shard.partition(self.text["intermediate_size"], alignment=128)


def expected_shapes(c):
    """The assistant's state_dict: encoder projection + norm, final norm, and 11 tensors per layer."""
    h, i, head = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    q, kv = c["num_attention_heads"] * head, c["num_key_value_heads"] * head
    shapes = {"encoder.fc.weight": (h, h * len(c["target_layer_ids"])), "encoder.output_norm_enc.weight": (h,),
              "norm.weight": (h,)}
    for layer in range(c["num_hidden_layers"]):
        p = f"layers.{layer}."
        shapes.update({p + "self_attn.q_proj.weight": (q, h), p + "self_attn.k_proj.weight": (kv, h),
            p + "self_attn.v_proj.weight": (kv, h), p + "self_attn.o_proj.weight": (h, q),
            p + "self_attn.q_norm.weight": (head,), p + "self_attn.k_norm.weight": (head,),
            p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
            p + "mlp.gate_proj.weight": (i, h), p + "mlp.up_proj.weight": (i, h), p + "mlp.down_proj.weight": (h, i)})
    return shapes


def fabricate(path, c, *, dtype=torch.bfloat16, random=False):
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(0)
    tensors = {name: (torch.randn(shape, generator=generator) if random else torch.zeros(shape)).to(dtype)
               for name, shape in expected_shapes(c).items()}
    save_file(tensors, str(path / "model.safetensors"))
    return tensors


def test_assistant_tensor_set_is_58_bf16_tensors_with_the_published_shapes(tmp_path):
    fabricate(tmp_path, SMALL)
    manifest = inspect_indexed_checkpoint(tmp_path)
    assert len(manifest.tensors) == 58 and set(manifest.tensors) == set(expected_shapes(SMALL))
    validate_metadata(manifest, StubDraftConfig(SMALL), projector="dflash")
    # The pinned geometry is 5.1 GB, so the published header is reproduced as metadata only.
    template = manifest.tensors["norm.weight"]
    full = {name: replace(template, name=name, shape=shape) for name, shape in expected_shapes(ASSISTANT).items()}
    assert full["encoder.fc.weight"].shape == (6656, 33280) and full["layers.4.mlp.down_proj.weight"].shape == (6656, 19968)
    assert full["layers.0.self_attn.q_proj.weight"].shape == (4096, 6656)
    validate_metadata(replace(manifest, tensors=full), StubDraftConfig(ASSISTANT), projector="dflash")


@pytest.mark.parametrize("fault,message", [
    (dict(drop="layers.2.self_attn.k_norm.weight"), "tensor names disagree"),
    (dict(rename=("encoder.fc.weight", "fc.weight")), "tensor names disagree"),
    (dict(extra="candidate_selector.predecessor_codebook"), "tensor names disagree"),
    (dict(reshape=("encoder.fc.weight", (128, 128))), r"encoder.fc.weight: DFlash2 requires BF16 storage with shape \(128, 640\)"),
    (dict(dtype=torch.float32), "requires BF16 storage"),
])
def test_projector_less_check_rejects_missing_renamed_extra_misshaped_or_non_bf16_tensors(tmp_path, fault, message):
    fabricate(tmp_path, SMALL, dtype=fault.get("dtype", torch.bfloat16))
    manifest = inspect_indexed_checkpoint(tmp_path)
    tensors = dict(manifest.tensors)
    if "drop" in fault:
        del tensors[fault["drop"]]
    if "rename" in fault:
        old, new = fault["rename"]
        tensors[new] = replace(tensors.pop(old), name=new)
    if "extra" in fault:
        tensors[fault["extra"]] = replace(tensors["norm.weight"], name=fault["extra"])
    if "reshape" in fault:
        name, shape = fault["reshape"]
        tensors[name] = replace(tensors[name], shape=shape)
    with pytest.raises(ValueError, match=message):
        validate_metadata(replace(manifest, tensors=tensors), StubDraftConfig(SMALL), projector="dflash")


def test_projector_switch_keeps_the_dflash2_tensor_set_apart(tmp_path):
    fabricate(tmp_path, SMALL)
    manifest = inspect_indexed_checkpoint(tmp_path)
    dflash2 = StubDraftConfig({**SMALL, "dflash_config": dict(selector_rank=16, conv_kernel_size=2, conv_group_size=16)})
    with pytest.raises(ValueError, match="tensor names disagree"):
        validate_metadata(manifest, dflash2)
    with pytest.raises(ValueError, match="tensor names disagree"):
        DraftWeights.load(tmp_path, dflash2, "cpu", TensorParallelShard(0, 1))


def test_block_16_scratch_skips_the_convolution_buffers_and_has_no_default_block():
    assert next(f for f in fields(DraftScratch) if f.name == "block_size").default is MISSING
    config, shard = StubDraftConfig(SMALL), TensorParallelShard(0, 1)
    scratch = DraftScratch.allocate(config, shard, 32, "meta", buffers=GraphBufferPool("meta").cursor())
    assert scratch.block_size == 16 and scratch.convolution is None
    assert scratch.embedding.shape == (32, 128) and scratch.query.shape == (32, 4, 32)
    assert scratch.window_key.shape == (2 * 17, 128, 2, 32)
    rows = scratch.rows(16, 16)
    assert rows.block_size == 16 and rows.convolution is None and rows.embedding.shape == (16, 128)
    assert all(not isinstance(t, dict) for t in scratch.scratch_tensors())
    with pytest.raises(ValueError, match="complete proposal blocks"):
        DraftScratch.allocate(config, shard, 24, "meta", buffers=GraphBufferPool("meta").cursor())
    with_conv = StubDraftConfig({**SMALL, "dflash_config": dict(conv_kernel_size=2, conv_group_size=16)})
    assert DraftScratch.allocate(with_conv, shard, 32, "meta",
        buffers=GraphBufferPool("meta").cursor()).convolution["coefficients"].shape == (32, 32)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires assigned CUDA_VISIBLE_DEVICES")
def test_projector_less_load_reads_the_encoder_tensors_without_convolutions_or_selector(tmp_path):
    source = fabricate(tmp_path, SMALL, random=True)
    weights = DraftWeights.load(tmp_path, StubDraftConfig(SMALL), torch.device("cuda:0"), TensorParallelShard(0, 1),
                                projector="dflash")
    def check(actual, name):
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.equal(actual.cpu(), source[name]), name

    check(weights.projection, "encoder.fc.weight")
    check(weights.hidden_norm, "encoder.output_norm_enc.weight")
    check(weights.norm, "norm.weight")
    assert weights.selector is None and len(weights.layers) == 5
    for i, layer in enumerate(weights.layers):
        p = f"layers.{i}."
        assert layer.attention_conv is None and layer.mlp_conv is None
        check(layer.qkv[:128], p + "self_attn.q_proj.weight")
        check(layer.qkv[128:192], p + "self_attn.k_proj.weight")
        check(layer.qkv[192:], p + "self_attn.v_proj.weight")
        assert torch.equal(weights.context_kv[i * 128:(i + 1) * 128], layer.qkv[128:])
        check(layer.output, p + "self_attn.o_proj.weight")
        check(layer.query_norm, p + "self_attn.q_norm.weight")
        check(layer.key_norm, p + "self_attn.k_norm.weight")
        check(layer.input_norm, p + "input_layernorm.weight")
        check(layer.post_attention_norm, p + "post_attention_layernorm.weight")
        check(layer.mlp.gate_up[:256], p + "mlp.gate_proj.weight")
        check(layer.mlp.gate_up[256:], p + "mlp.up_proj.weight")
        check(layer.mlp.down, p + "mlp.down_proj.weight")
