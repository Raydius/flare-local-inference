# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen rotary positions with image input against the canonical three-axis positions."""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding

from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.decoder import rotary_axes, rotary_positions
from ignition.recipes.qwen38_dense.pool import continuation_layout

ROTARY = dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=.25, mrope_interleaved=True,
              mrope_section=[11, 11, 10])
# Five text tokens, an image of 8 x 12 patches merged two by two into a 4 x 6 grid, seven text tokens.
TOKEN_TYPES = [0] * 5 + [1] * 24 + [0] * 7
ROW = 3


def canonical():
    """Positions (tokens, 3) and the offset `transformers` assigns the prompt."""
    owner = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    owner.get_vision_position_ids = partial(Qwen3_5Model.get_vision_position_ids, owner)
    types = torch.tensor([TOKEN_TYPES])
    positions, delta = Qwen3_5Model.get_rope_index(owner, torch.zeros_like(types), types,
                                                   image_grid_thw=torch.tensor([[1, 8, 12]]))
    return positions[:, 0].T.contiguous(), int(delta)


def chunk(start, length, capacity):
    """Packed metadata of one prefill chunk of `length` tokens at `start` in a graph of `capacity` tokens."""
    offsets = torch.arange(capacity)
    return SimpleNamespace(state_rows=torch.tensor([ROW]), requests=torch.zeros(capacity, dtype=torch.int64),
        starts=torch.tensor([start]), lengths=torch.tensor([length], dtype=torch.int32),
        positions=torch.where(offsets < length, start + offsets, -1))


def prefill(positions, boundaries, stale=0, capacity=40):
    """Run the prompt through prefill chunks that end at `boundaries`; return the stored offsets and positions."""
    rope_delta = torch.full((5, 1), 99, dtype=torch.int64)
    rope_delta[ROW] = stale
    produced, start = [], 0
    for end in (*boundaries, positions.shape[0]):
        upload = torch.full((capacity, 3), 7777, dtype=torch.int64)
        upload[:end - start] = positions[start:end]
        result = rotary_positions(chunk(start, end - start, capacity), rope_delta, upload)
        assert torch.equal(result[end - start:], torch.full((capacity - end + start, 3), -1))
        produced.append(result[:end - start])
        start = end
    return rope_delta, torch.cat(produced)


def test_rotary_pairs_read_the_axes_of_the_canonical_recomposition():
    positions, _ = canonical()
    config = Qwen3_5TextConfig(hidden_size=64, num_attention_heads=1, num_key_value_heads=1, head_dim=256,
                               rope_parameters=dict(ROTARY))
    reference = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = reference(torch.zeros(1, dtype=torch.float32), positions.T[:, None])
    axes = rotary_axes(ROTARY["mrope_section"], 32)
    assert axes.tolist().count(1) == 11 and axes.tolist().count(2) == 10
    frequency = positions[:, axes].float() * reference.inv_freq[None]
    embedding = torch.cat((frequency, frequency), -1)
    assert torch.equal(embedding.cos(), cos[0]) and torch.equal(embedding.sin(), sin[0])


def test_equal_axes_give_the_one_axis_angles():
    positions = torch.arange(300)
    inverse = 1.0 / (ROTARY["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    axes = rotary_axes(ROTARY["mrope_section"], 32)
    three = positions[:, None].expand(-1, 3)[:, axes].float() * inverse[None]
    assert torch.equal(three, positions.float()[:, None] * inverse[None])


@pytest.mark.parametrize("boundaries", [(), (3,), (5,), (12,), (29,), (30,), (4, 17, 33)])
def test_prefill_stores_the_canonical_offset_for_every_chunking(boundaries):
    positions, delta = canonical()
    assert delta == 5 + 6 + 7 - len(TOKEN_TYPES)
    stored, produced = prefill(positions, boundaries, stale=-40)
    assert torch.equal(produced, positions)
    assert stored[ROW].item() == delta
    assert (stored[[0, 1, 2, 4]] == 99).all()


def test_decode_continues_from_the_canonical_next_position():
    positions, delta = canonical()
    stored, _ = prefill(positions, (12,))
    tokens = len(TOKEN_TYPES)
    m = SimpleNamespace(state_rows=torch.tensor([0, ROW]), requests=torch.tensor([1, 1, 0]),
                        positions=torch.tensor([tokens, tokens + 1, -1]))
    stored[0] = 0
    decoded = rotary_positions(m, stored)
    following = int(positions.max()) + 1
    assert decoded.tolist() == [[following] * 3, [following + 1] * 3, [-1] * 3]
    assert stored[ROW].item() == delta


def test_text_prompt_keeps_token_positions_and_a_zero_offset():
    positions = torch.arange(36)[:, None].expand(-1, 3).contiguous()
    stored, produced = prefill(positions, (10, 20), stale=-17)
    assert torch.equal(produced, positions) and stored[ROW].item() == 0


def test_an_inactive_chunk_stores_nothing():
    rope_delta = torch.full((5, 1), -6, dtype=torch.int64)
    m = chunk(0, 0, 8)
    m.state_rows = torch.tensor([0])
    rotary_positions(m, rope_delta, torch.zeros((8, 3), dtype=torch.int64))
    assert (rope_delta == -6).all()


def test_only_a_layout_with_vision_stores_the_offset():
    text = dict(hidden_size=5120, vocab_size=248320, num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
        layer_types=["full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)])
    config = ModelConfig(text, {})
    plain, vision = continuation_layout(config), continuation_layout(config, vision=True)
    assert [region.name for region in vision.regions] == [*(region.name for region in plain.regions), "rope_delta"]
    region = vision.regions[-1]
    assert (tuple(region.shape), region.dtype) == ((1,), torch.int64)
    assert vision.regions[:-1] == plain.regions


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"),
                    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")
@torch.no_grad()
def test_captured_graphs_store_and_apply_the_offset():
    """Prefill and decode position graphs, compiled and captured, replay with the canonical positions."""
    from b12x.preparation import PreparationSession

    from ignition.preparation.compilation import compile_forward

    positions, delta = canonical()
    capacity, device = 40, "cuda"
    axes = rotary_axes(ROTARY["mrope_section"], 32).to(device)
    inverse = (1.0 / (ROTARY["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(device)
    # Rows of a wider backing, as the state pool exposes a region of its rows.
    backing = torch.zeros((5, 64), dtype=torch.uint8, device=device)
    rope_delta = backing[:, 16:24].view(torch.int64).view(5, 1)
    upload = torch.zeros((capacity, 3), dtype=torch.int64, device=device)
    m = SimpleNamespace(**{name: value.to(device) for name, value in vars(chunk(0, 0, capacity)).items()})
    decode = SimpleNamespace(state_rows=torch.tensor([ROW, 0], device=device),
        requests=torch.tensor([0, 0], device=device), positions=torch.tensor([-1, -1], device=device))
    angles = {name: torch.zeros((rows, 32), dtype=torch.float32, device=device)
              for name, rows in (("prefill", capacity), ("decode", 2))}

    def graph(session, body):
        forward = compile_forward(body)
        forward()
        captured = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(captured):
            forward()
        return captured

    with PreparationSession(device="cuda:0", autotune=False) as session:
        prefill_graph = graph(session, lambda: angles["prefill"].copy_(
            rotary_positions(m, rope_delta, upload)[:, axes].float() * inverse[None]))
        decode_graph = graph(session, lambda: angles["decode"].copy_(
            rotary_positions(decode, rope_delta)[:, axes].float() * inverse[None]))
        backing.fill_(0)
        rope_delta[ROW] = -40
        start = 0
        for end in (12, 30, len(TOKEN_TYPES)):
            reference = chunk(start, end - start, capacity)
            for name in ("starts", "lengths", "positions"):
                getattr(m, name).copy_(getattr(reference, name))
            upload.fill_(7777)
            upload[:end - start].copy_(positions[start:end])
            prefill_graph.replay()
            expected = positions[start:end][:, axes.cpu()].float() * inverse.cpu()[None]
            assert torch.equal(angles["prefill"][:end - start].cpu(), expected)
            start = end
        assert rope_delta[ROW].item() == delta
        assert int(backing.view(torch.int64).ne(0).sum()) == 1
        decode.positions.copy_(torch.tensor([start, start + 1]))
        decode_graph.replay()
        following = (int(positions.max()) + 1 + torch.arange(2)).float()[:, None] * inverse.cpu()[None]
        assert torch.equal(angles["decode"].cpu(), following)
