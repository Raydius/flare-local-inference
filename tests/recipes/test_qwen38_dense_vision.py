# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen image input: the plan, the vision configuration checks, the contract and the upload generation."""

import copy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.qwen38_dense import config as dense_config
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.contract import prepare
from ignition.recipes.qwen38_dense.pool import ContinuationPool

TEXT = dict(hidden_size=5120, vocab_size=248320, num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
    layer_types=["full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)],
    rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=.25,
                         mrope_interleaved=True, mrope_section=[11, 11, 10]))
# The `vision_config` and image token `RadixArk/DaisyRose-27B-NVFP4` publishes.
PUBLISHED = dict(image_token_id=248056, vision_config=dict(depth=27, hidden_size=1152, num_heads=16,
    intermediate_size=4304, patch_size=16, temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
    num_position_embeddings=2304, out_hidden_size=5120, deepstack_visual_indexes=[], hidden_act="gelu_pytorch_tanh",
    initializer_range=.02, model_type="qwen3_5_vision"))
TEXT_SIGNATURES = {
    "prefill256": GraphSignature("prefill", 1, 256, ("request_state_rows", "token_ids", "page_tables")),
    "decode4": GraphSignature("decode", 4, 1, ("request_state_rows", "row_slot_ids", "page_tables")),
    "init4": GraphSignature("init", 4, 1, ("request_state_rows", "state_num_ops"))}
# SHA-256 of the contract `prepare` returned for TEXT_SIGNATURES before the recipe declared image input.
TEXT_CONTRACT = "7ba69fd1c953d836d436fc581fa93f3c3b169457d60c00c9b80cd28b2f25952e"


def snapshot(tmp_path, **changes):
    raw = copy.deepcopy(PUBLISHED)
    for name, value in changes.items():
        section = raw["vision_config"] if name in raw["vision_config"] else raw
        section[name] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path


def test_published_vision_configuration_validates(tmp_path):
    assert dense_config.vision_config(snapshot(tmp_path), TEXT) == PUBLISHED["vision_config"]


@pytest.mark.parametrize("name,value", [("depth", 24), ("hidden_size", 1024), ("num_heads", 12),
    ("intermediate_size", 4096), ("patch_size", 14), ("temporal_patch_size", 1), ("spatial_merge_size", 3),
    ("in_channels", 1), ("num_position_embeddings", 1024), ("out_hidden_size", 2560),
    ("deepstack_visual_indexes", [8, 16]), ("image_token_id", 151655)])
def test_foreign_vision_geometry_is_refused_by_field(tmp_path, name, value):
    with pytest.raises(ValueError, match=rf"unsupported Qwen dense (vision )?{name}: "):
        dense_config.vision_config(snapshot(tmp_path, **{name: value}), TEXT)


@pytest.mark.parametrize("name,value", [("mrope_interleaved", False), ("mrope_section", [16, 24, 24])])
def test_foreign_text_position_axes_are_refused_by_field(tmp_path, name, value):
    text = {**TEXT, "rope_parameters": {**TEXT["rope_parameters"], name: value}}
    with pytest.raises(ValueError, match=f"unsupported Qwen dense {name}: "):
        dense_config.vision_config(snapshot(tmp_path), text)


def test_a_checkpoint_without_a_vision_encoder_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": 248056}))
    with pytest.raises(ValueError, match="requires a vision_config object"):
        dense_config.vision_config(tmp_path, TEXT)


def test_plan_reserves_image_slots_unless_language_only(tmp_path):
    from ignition import planning
    from ignition.cli.config import Configuration, Setting
    from ignition.cli.preparation import Host, derive
    from ignition.errors import UsageError
    from ignition.recipes import resolve

    model = resolve("daisyrose-27b")
    assert (model.recipe.max_images, model.recipe.image_bytes, model.recipe.language_model_only) == (4, 32 << 20, True)
    host = Host((planning.Gpu(0, "Blackwell", "GPU-0", 256 << 30, 250 << 30, (12, 0)),), None)
    configuration = Configuration(tmp_path / "config.toml", {}, exists=True)

    def plan(**options):
        return derive(configuration, model, host, gpus="0", concurrency=Setting(16, "test"),
                      speculation=model.recipe.speculation, **options)

    images, text = plan(), plan(language_model_only=True)
    assert (images.image_concurrency, images.max_images) == (4, 4)
    assert (text.image_concurrency, text.max_images) == (0, 0)
    assert plan(image_concurrency=2).image_concurrency == 2
    with pytest.raises(UsageError):
        plan(language_model_only=True, image_concurrency=2)


def contract(signatures, **options):
    config = ModelConfig(TEXT, {})
    pool = ContinuationPool.for_model(config, 3, 2, "meta", vision=bool(options))
    return prepare(signatures, {}, pool, vocabulary=248320, **options)


def test_text_contract_is_the_contract_before_image_input():
    digest = hashlib.sha256(json.dumps(contract(TEXT_SIGNATURES), sort_keys=True, default=str).encode()).hexdigest()
    assert digest == TEXT_CONTRACT


def test_image_contract_declares_vision_inputs_identifiers_and_three_position_rows():
    uploads = ("vision_pixel_values", "vision_position_ids", "vision_slot", "vision_generation", "vision_feature_slot")
    signatures = {**TEXT_SIGNATURES,
        "prefill256": GraphSignature("prefill", 1, 256, ("request_state_rows", "token_ids", "page_tables",
            "vision_token_indices", "vision_feature_slot", "prefill_positions")),
        "vision256": GraphSignature("vision", 1, 256, uploads)}
    result = contract(signatures, image_slots=4)
    assert result["vision"] == dict(feature_slots=4, features_per_request=16384, max_images=4,
        patches_per_feature=4, patch_capacities=[256, 1024, 4096, 16384])
    frontend = result["frontend"]
    assert frontend["image_tokens"] == dict(feature_token=248056, span=None)
    assert (frontend["image_preparation"], frontend["image_layout"], frontend["position_dimensions"]) == (
        p.QWEN_IMAGE_PREPARATION, p.QWEN_IMAGE_LAYOUT, 3)
    graphs = {graph["graph"]: graph for graph in result["graphs"]}
    vision = {entry["buffer"]: entry["source"] for entry in graphs["vision256"]["inputs"]}
    assert vision == {"vision_pixel_values": p.IMAGE_TENSOR, "vision_position_ids": p.IMAGE_TENSOR,
        "vision_slot": p.REQUEST_SLOT, "vision_generation": p.REQUEST_GENERATION,
        "vision_feature_slot": p.IMAGE_FEATURE_SLOT}
    assert graphs["vision256"]["state"] == [] and graphs["vision256"]["pending_token"] is None
    prefill = {entry["buffer"]: entry["source"] for entry in graphs["prefill256"]["inputs"]}
    assert prefill["prefill_positions"] == p.PROMPT_POSITION
    assert prefill["vision_token_indices"] == p.IMAGE_FEATURE_INDEX
    text = contract(TEXT_SIGNATURES)
    assert text["frontend"]["position_dimensions"] == 0 and text["vision"]["max_images"] == 0
    assert {key: value for key, value in frontend.items()
            if key not in ("image_tokens", "image_preparation", "image_layout", "position_dimensions")} == {
        key: value for key, value in text["frontend"].items()
        if key not in ("image_tokens", "image_preparation", "image_layout", "position_dimensions")}


def test_qwen_image_profile_accepts_the_dense_model_type():
    from ignition.preparation.frontend import image_settings

    processor = dict(processor_class="Qwen3VLProcessor", image_processor_type="Qwen2VLImageProcessorFast",
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5] * 3, image_std=[.5] * 3,
        size=dict(shortest_edge=65536, longest_edge=16777216))
    model = dict(PUBLISHED, video_token_id=248057)
    for model_type in ("qwen3_5", "qwen4_exp"):
        settings = image_settings(processor, dict(model, model_type=model_type), "qwen")["qwen"]
        assert settings["image_token_id"] == 248056 and settings["position_side"] == 48
    with pytest.raises(ValueError, match="unsupported model configuration model_type"):
        image_settings(processor, dict(model, model_type="qwen2_vl"), "qwen")
    with pytest.raises(ValueError, match="unsupported model configuration model_type"):
        image_settings(dict(processor, processor_class="Qwen2_5_VLProcessor",
            image_processor_type="Qwen2VLImageProcessor"), dict(model, model_type="qwen3_5"), "mimo")


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"),
                    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")
@torch.no_grad()
def test_generation_publishes_a_valid_upload_and_rejects_malformed_ones():
    """The upload generation with a stand-in encoder, compiled and captured as a CUDA graph."""
    from b12x.preparation import PreparationSession

    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.qwen38_dense.vision_state import PreparedVisionGeneration, VisionInputs

    device, capacity, width, generation = torch.device("cuda:0"), 16, 64, 7

    class Encoder:
        """Feature row `r` holds the pixel sum of its four patches plus the live patch count."""

        def __init__(self):
            self.capacity = capacity

        def __call__(self, pixels, metadata):
            merged = pixels.float().view(capacity // 4, -1).sum(1, keepdim=True) + metadata.num_patches.float()
            return merged.expand(-1, width).bfloat16()

    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=248320, capacity=2)
    batch = SimpleNamespace(token_ids=torch.zeros(3, dtype=torch.int64, device=device),
        row_slot_ids=torch.ones(1, dtype=torch.int32, device=device),
        row_generations=torch.full((1,), generation, dtype=torch.int64, device=device))
    state = VisionState(batch, PrefillVisionInputs.allocate(3, device), requests, image_token=248056,
                        features=8, hidden=width, image_slots=1)
    completion = CompletionOutputs.allocate((2, 2, 32), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    operation = PreparedVisionGeneration(Encoder(), inputs, state, completion, frame, sequence)
    errors = torch.zeros(1, dtype=torch.int32, device=device)

    def body():
        errors.copy_(operation.forward()[1])

    with PreparationSession(device="cuda:0", autotune=False) as session:
        forward = compile_forward(body)
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()

        def upload(patches, offset, *, pixel=1.0, upload_generation=generation):
            requests.generation[1], requests.status[1], requests.error_code[1] = generation, 1, 0
            inputs.vision_slot.fill_(1)
            inputs.vision_generation.fill_(upload_generation)
            inputs.vision_num_patches.fill_(patches)
            inputs.vision_cu_seqlens.copy_(torch.tensor([0, patches]))
            inputs.vision_feature_offset.fill_(offset)
            inputs.vision_pixel_values.fill_(float("nan"))
            inputs.vision_pixel_values[:max(patches, 0)] = pixel
            inputs.vision_interpolation_weights.fill_(.25)
            state.feature_slot.zero_()
            graph.replay()
            torch.cuda.synchronize()
            return int(errors), int(requests.status[1]), int(requests.error_code[1])

        state.arena.fill_(-1)
        state.generation.zero_()
        state.count.zero_()
        assert upload(8, 0) == (0, 1, 0)
        assert (int(state.generation[1]), int(state.count[1])) == (generation, 2)
        expected = 4 * 1536 + 8
        assert torch.equal(state.arena[0, :2].float().cpu(), torch.full((2, width), float(expected)).bfloat16().float())
        assert (state.arena[0, 2:] == -1).all()
        assert upload(4, 2, pixel=2.0) == (0, 1, 0) and int(state.count[1]) == 3
        published = state.arena.clone()
        for arguments, options in (((6, 3), {}), ((0, 3), {}), ((20, 3), {}), ((4, 1), {}), ((4, 8), {}),
                                   ((4, 3), dict(pixel=float("inf"))), ((4, 3), dict(upload_generation=6))):
            error, status, code = upload(*arguments, **options)
            assert error == 1 and code in (0, 129), (arguments, options)
            assert status == (1 if options.get("upload_generation") else 3), (arguments, options)
            assert torch.equal(state.arena, published) and int(state.count[1]) == 3, (arguments, options)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES")
                    or not os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT"),
                    reason="requires FLARE_RUN_CUDA_TESTS=1, an assigned CUDA_VISIBLE_DEVICES and "
                           "FLARE_QWEN38_DENSE_SNAPSHOT naming a dense Qwen checkpoint")
@torch.no_grad()
def test_checkpoint_vision_graph_loads_encodes_and_publishes():
    """The vision graph as the bootstrap builds it: the checkpoint loader, the encoder and the publication."""
    from pathlib import Path

    from b12x.preparation import PreparationSession
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding

    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights
    from ignition.recipes.qwen38_dense.vision_state import FEATURES, PreparedVisionGeneration, VisionInputs

    snapshot = Path(os.environ["FLARE_QWEN38_DENSE_SNAPSHOT"]).expanduser()
    device, capacity, generation = torch.device("cuda:0"), 256, 5
    text = json.loads((snapshot / "config.json").read_text())["text_config"]
    vision = Qwen3_5VisionConfig.from_dict(dense_config.vision_config(snapshot, text))
    with torch.device("meta"):
        module = Qwen3_5VisionModel(vision)
    with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(module, loader)
    assert len(weights.projections) == 27 * 4 + 2
    assert all(isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16
               for weight in weights.projections.values())
    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=text["vocab_size"], capacity=2)
    batch = SimpleNamespace(token_ids=torch.zeros(3, dtype=torch.int64, device=device),
        row_slot_ids=torch.ones(1, dtype=torch.int32, device=device),
        row_generations=torch.full((1,), generation, dtype=torch.int64, device=device))
    state = VisionState(batch, PrefillVisionInputs.allocate(3, device), requests, image_token=dense_config.IMAGE_TOKEN,
                        features=FEATURES, hidden=text["hidden_size"], image_slots=1)
    completion = CompletionOutputs.allocate((2, 2, 32), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    errors = torch.zeros(1, dtype=torch.int32, device=device)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena(device)
        with torch.device("meta"):
            module = Qwen3_5VisionModel(vision)
        module.eval().requires_grad_(False)
        encoder = PreparedVision(module, VisionScratch.allocate(capacity, device, width=text["hidden_size"]),
            VisionMetadata.allocate(capacity, 1, device), weights, session=session, name="vision256",
            projection_workspace=arena, attention_workspace=arena, rotary=Qwen3_5VisionRotaryEmbedding)
        encoder.functional(session=session)
        operation = PreparedVisionGeneration(encoder, inputs, state, completion, frame, sequence)
        arena.finalize()
        forward = compile_forward(lambda: errors.copy_(operation.forward()[1]))
        forward()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()

        def upload(rows, columns, offset, *, pixel_scale=1.0):
            patches = rows * columns
            requests.generation[1], requests.status[1], requests.error_code[1] = generation, 1, 0
            inputs.vision_slot.fill_(1)
            inputs.vision_generation.fill_(generation)
            inputs.vision_num_patches.fill_(patches)
            inputs.vision_cu_seqlens.copy_(torch.tensor([0, patches]))
            inputs.vision_feature_offset.fill_(offset)
            inputs.vision_pixel_values.fill_(float("nan"))
            pixels = torch.randn((patches, 1536), generator=torch.Generator().manual_seed(patches)) * pixel_scale
            inputs.vision_pixel_values[:patches] = pixels.to(device)
            grid = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij"), -1)
            inputs.vision_position_ids[:patches] = grid.view(-1, 2).to(device)
            inputs.vision_interpolation_indices[:patches] = torch.arange(patches, device=device)[:, None] % 2304
            inputs.vision_interpolation_weights.fill_(.25)
            state.feature_slot.zero_()
            graph.replay()
            torch.cuda.synchronize()
            return int(errors), int(requests.status[1])

        state.arena.fill_(0)
        assert upload(8, 12, 0) == (0, 1)
        assert int(state.count[1]) == 24
        published = state.arena[0, :24].float()
        assert torch.isfinite(published).all() and (published.abs().amax(1) > 0).all()
        assert (state.arena[0, 24:] == 0).all()
        assert upload(8, 8, 24) == (0, 1) and int(state.count[1]) == 40
        assert torch.equal(state.arena[0, :24].float(), published)
        before = state.arena.clone()
        assert upload(8, 8, 40, pixel_scale=float("inf")) == (1, 3)
        assert torch.equal(state.arena, before) and int(state.count[1]) == 40
