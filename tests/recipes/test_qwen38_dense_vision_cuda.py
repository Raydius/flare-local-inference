# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen vision encoder as a captured CUDA graph against the canonical encoder.

The dense Qwen3.8 checkpoint stores every vision tensor in BF16. The prepared encoder of
``ignition.recipes.qwen38.vision`` binds those projections unquantized and is compared with
``transformers``' ``Qwen3_5VisionModel`` evaluated in FP32 on the same BF16 weights.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")

IMAGES = 2
# The prepared encoder rounds to BF16 after each of 11 operations per block (two normalizations,
# the query-key-value projection, the rotary rotation, attention, the output projection, two
# residual additions, two MLP projections and the activation) and after 9 operations outside the
# 27 blocks; the FP32 reference rounds none of them. One rounding changes a tensor by at most
# 2**-9 of its L2 norm, and 306 of them in quadrature give 0.034 only if every error reaches the
# merged features with unit gain. The encoder's normalizations and attention do not have unit gain:
# the canonical encoder evaluated in BF16 differs from its FP32 evaluation by a relative L2 error of
# 0.027 to 0.047 on these uploads. Each case therefore measures that difference on its own input.
# The prepared and the canonical BF16 evaluations round the same operations with different kernels,
# so their errors are two samples of one rounding process; the prepared error may reach twice the
# canonical one.
BF16_ROUNDING_FACTOR = 2.0
# One image and two images in one upload, as (frames, patch rows, patch columns) per image.
# Every case except the full 256-patch image leaves padding rows.
UPLOADS = {256: ([(1, 16, 16)], [(1, 8, 8), (1, 8, 12)]),
           1024: ([(1, 24, 32)], [(1, 16, 16), (1, 16, 32)])}


def replay(graph):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[name] == after[name], name


@pytest.fixture(scope="module")
def checkpoint():
    """Vision configuration and the BF16 ``model.visual.*`` tensors of the dense checkpoint, on the GPU."""
    from safetensors import safe_open

    value = os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT")
    if not value or not (Path(value).expanduser() / "config.json").is_file():
        pytest.skip("FLARE_QWEN38_DENSE_SNAPSHOT does not name a dense Qwen3.8 checkpoint folder")
    snapshot = Path(value).expanduser()
    tensors = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cuda:0") as file:
            for key in file.keys():
                if key.startswith("model.visual."):
                    tensors[key.removeprefix("model.visual.")] = file.get_tensor(key)
    assert tensors and all(tensor.dtype == torch.bfloat16 for tensor in tensors.values())
    return SimpleNamespace(config=json.loads((snapshot / "config.json").read_text())["vision_config"], tensors=tensors)


@pytest.fixture(scope="module")
def reference(checkpoint):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

    """The canonical encoder on the checkpoint's BF16 weights, evaluated in FP32 and in BF16."""
    def module(dtype):
        value = Qwen3_5VisionModel(Qwen3_5VisionConfig(**checkpoint.config)).eval().requires_grad_(False)
        value.load_state_dict(checkpoint.tensors)
        return value.to("cuda:0", dtype)

    return SimpleNamespace(fp32=module(torch.float32), bf16=module(torch.bfloat16))


@pytest.fixture(scope="module")
def prepared(checkpoint):
    """One captured encoder graph per patch capacity, with its pixel, metadata and output storage."""
    from b12x.preparation import PreparationSession
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionConfig, Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding,
    )

    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights

    device = torch.device("cuda", 0)
    config = Qwen3_5VisionConfig(**checkpoint.config)
    # The split VisionWeights.load makes: block and merger projection weights apart from other parameters.
    parameters, projections = {}, {}
    for path, tensor in checkpoint.tensors.items():
        if (path.endswith(".weight") and not path.startswith("patch_embed.")
                and path.split(".")[-2] in ("qkv", "proj", "linear_fc1", "linear_fc2")):
            projections[path.removesuffix(".weight")] = tensor
        else:
            parameters[path] = tensor
    weights = VisionWeights(parameters, projections)
    entries = {}
    with torch.no_grad(), PreparationSession(device=device, autotune=False) as session:
        arena = ScratchArena(device)
        for capacity in UPLOADS:
            scratch = VisionScratch.allocate(capacity, device, config.out_hidden_size)
            metadata = VisionMetadata.allocate(capacity, IMAGES, device)
            pixels = torch.zeros((capacity, 1536), dtype=torch.float32, device=device)
            with torch.device("meta"):
                module = Qwen3_5VisionModel(config)
            module.eval().requires_grad_(False)
            encoder = PreparedVision(module, scratch, metadata, weights, session=session, name=f"vision{capacity}",
                projection_workspace=arena, attention_workspace=arena, rotary=Qwen3_5VisionRotaryEmbedding)
            entries[capacity] = SimpleNamespace(scratch=scratch, metadata=metadata, pixels=pixels, encoder=encoder)
        arena.finalize()
        for entry in entries.values():
            entry.encoder.functional(session=session)
            forward = compile_forward(lambda entry=entry: entry.scratch.output.copy_(
                entry.encoder(entry.pixels, entry.metadata)))
            forward()
            entry.graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(entry.graph):
                forward()
            entry.forward = forward
        yield entries


@pytest.mark.parametrize("images", [1, 2])
@pytest.mark.parametrize("capacity", list(UPLOADS))
@torch.no_grad()
def test_captured_dense_vision_encoder_matches_the_canonical_encoder(prepared, reference, capacity, images,
                                                                     record_property):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    entry, device = prepared[capacity], torch.device("cuda", 0)
    grid = torch.tensor(UPLOADS[capacity][images - 1], device=device)
    lengths = grid.prod(1)
    patches = int(lengths.sum())
    assert patches <= capacity and patches % 4 == 0
    indices, factors = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
    metadata = entry.metadata
    for tensor in vars(metadata).values():
        tensor.zero_()
    metadata.interpolation_indices[:patches].copy_(indices)
    metadata.interpolation_weights[:patches].copy_(factors)
    metadata.position_ids[:patches].copy_(get_vision_position_ids(grid, 2))
    # Images the upload does not use have zero length: their cumulative length repeats the total.
    metadata.cu_seqlens.fill_(patches)
    metadata.cu_seqlens[0] = 0
    metadata.cu_seqlens[1:images + 1].copy_(lengths.cumsum(0))
    metadata.num_patches.fill_(patches)
    # Padding rows hold pixel values too; attention lengths and the output mask must exclude them.
    entry.pixels.normal_(generator=torch.Generator(device=device).manual_seed(capacity + images))
    # Both reference evaluations read the BF16 pixel values the prepared encoder's first cast produces.
    rounded = entry.pixels[:patches].bfloat16()
    expected = reference.fp32(rounded.float(), grid).pooler_output
    norm = float(expected.norm())
    rounding = float((reference.bf16(rounded, grid).pooler_output.float() - expected).norm()) / norm
    for tensor in vars(entry.scratch).values():
        tensor.fill_(float("nan"))
    replay(entry.graph)
    actual = entry.scratch.output.float()
    assert expected.shape == (patches // 4, actual.shape[1])
    assert torch.count_nonzero(actual[patches // 4:]) == 0
    error = actual[:patches // 4] - expected
    relative, bound = float(error.norm()) / norm, BF16_ROUNDING_FACTOR * rounding
    record_property("numerical_error", json.dumps({"relative_l2": relative, "bound": bound,
        "canonical_bf16_relative_l2": rounding, "max_absolute": float(error.abs().max())}))
    assert relative <= bound, (relative, bound)
