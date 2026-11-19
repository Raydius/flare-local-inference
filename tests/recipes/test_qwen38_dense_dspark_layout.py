# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen state rows and page regions under the DFlash2 and DSpark proposal methods."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ignition.recipes import resolve
from ignition.recipes.qwen38_dense import draft_config, dspark_config
from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig
from ignition.recipes.qwen38_dense.config import ModelConfig
from ignition.recipes.qwen38_dense.pool import ContinuationPool, continuation_layout, page_bytes, page_specs

from .test_qwen38_dense_dflash import DRAFT_CONFIG
from .test_qwen38_dense_dspark_config import DSPARK_CONFIG

LINEAR = [index for index in range(64) if index % 4 != 3]
FULL = [index for index in range(64) if index % 4 == 3]
TARGET_PAGES = [(f"layer{index}.{field}", (128 * 4 * 256,)) for index in FULL for field in ("key", "value")]


@pytest.fixture
def target():
    text = dict(hidden_size=5120, vocab_size=248320, num_hidden_layers=64, num_key_value_heads=4, head_dim=256,
        layer_types=["full_attention" if index in FULL else "linear_attention" for index in range(64)])
    return ModelConfig(text, {})


def regions(layout):
    return [(region.name, tuple(region.shape), region.dtype) for region in layout.regions]


def recurrent(width):
    return [entry for index in LINEAR for entry in (
        (f"layer{index}.convolution", (10240, width), torch.bfloat16),
        (f"layer{index}.recurrent", (48, 128, 128), torch.float32))]


TAIL = [("last_hidden", (5120,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)]


def test_dflash2_keeps_its_state_row_page_regions_and_costs(target):
    """The DFlash2 names, shapes, dtypes and byte counts are those of the recipe before DSpark was declared."""
    config = replace(target, draft=draft_config.from_dict(copy.deepcopy(DRAFT_CONFIG), target))
    partial_pages = [(name, shape, torch.uint8) for name, shape in TARGET_PAGES]
    rings = [(f"draft{index}.{field}", (2048, 8, 128), torch.float8_e4m3fn)
             for index in range(5) for field in ("key", "value")]
    assert list(page_specs(config, mtp=True)) == list(page_specs(config)) == TARGET_PAGES
    assert page_bytes(config, mtp=True) == page_bytes(config) == 4_194_304
    drafting = continuation_layout(config, mtp=True)
    assert regions(drafting) == recurrent(10) + partial_pages + rings + TAIL
    assert drafting.row_bytes == 186_001_664
    assert continuation_layout(config, mtp=True, verification_width=16).row_bytes == 193_865_984
    plain = continuation_layout(config)
    assert regions(plain) == recurrent(3) + partial_pages + TAIL
    assert plain.row_bytes == 158_148_864
    recipe = resolve("qwen3.8-27b").recipe
    assert recipe.costs(1, recipe.speculation) == recipe.ranks[1]
    assert (recipe.ranks[1].state_row_bytes, recipe.ranks[1].page_bytes) == (193_865_984, 4_194_304)


def test_dspark_stores_draft_history_in_ten_page_regions_and_no_ring(target):
    config = replace(target, draft=dspark_config.from_dict(copy.deepcopy(DSPARK_CONFIG), target))
    assert dspark_config.PAGE_DTYPE == torch.float8_e4m3fn
    draft_pages = [(f"draft{index}.{field}", (128 * 8 * 128,)) for index in range(5) for field in ("key", "value")]
    assert list(page_specs(config, mtp=True)) == TARGET_PAGES + draft_pages
    assert list(page_specs(config)) == TARGET_PAGES
    layout = continuation_layout(config, mtp=True)
    # Each page region has one partial-page copy in the state row; the row holds no draft ring.
    assert regions(layout) == recurrent(10) + [(name, shape, torch.uint8)
                                              for name, shape in TARGET_PAGES + draft_pages] + TAIL
    costs = resolve("daisyrose-27b").recipe.speculative_methods["dspark"].ranks[1]
    assert layout.row_bytes == costs.state_row_bytes == 166_340_864
    assert page_bytes(config, mtp=True) == costs.page_bytes == 5_505_024
    pool = ContinuationPool.for_model(config, 3, 2, "meta", mtp=True)
    assert [name for name in pool.pages if name.startswith("draft")] == [name for name, _ in draft_pages]
    assert len(pool.pages) == 42
    plain = ContinuationPool.for_model(config, 3, 2, "meta")
    assert len(plain.pages) == 32 and plain.layout.row_bytes == 158_148_864


def test_dspark_preparation_refuses_more_than_seven_draft_tokens():
    options = dict(draft_snapshot=Path("draft"), speculative_method="dspark", max_active=8)
    accepted = PreparationConfig(speculative_widths=tuple(range(1, 9)), **options)
    assert accepted.dspark and accepted.verification_width == 8
    with pytest.raises(ValueError, match="DSpark supports at most seven draft tokens"):
        PreparationConfig(speculative_widths=tuple(range(1, 10)), **options)
    with pytest.raises(ValueError, match="the DSpark draft uses the target's vocabulary head"):
        PreparationConfig(speculative_widths=(1, 2), draft_head_quantization="mxfp8", **options)
    with pytest.raises(ValueError, match="proposal method must be dflash2 or dspark"):
        PreparationConfig(speculative_widths=(1, 2), **{**options, "speculative_method": "mtp"})
    assert not PreparationConfig(draft_snapshot=Path("draft"), speculative_widths=tuple(range(1, 17))).dspark


def test_feature_capture_keeps_every_prefill_position_for_an_unwindowed_draft():
    from types import SimpleNamespace

    from ignition.components.dflash2 import PreparedContextCapture

    metadata = SimpleNamespace(positions=torch.empty(4096, dtype=torch.int64, device="meta"),
                               starts=torch.empty(1, dtype=torch.int64, device="meta"))
    windowed = PreparedContextCapture(metadata, 25600, 2048, compact_prefill=True)
    assert windowed.compact and windowed.shape == (2048, 25600)
    unwindowed = PreparedContextCapture(metadata, 25600, None, compact_prefill=False)
    assert not unwindowed.compact and unwindowed.shape == (4096, 25600)


def test_dspark_owner_sizes_a_seven_position_block_scored_from_the_pending_token(target):
    from types import SimpleNamespace

    from ignition.components.parallel import TensorParallelShard
    from ignition.components.sampling import DeviceRequestState
    from ignition.recipes.kimi3.dspark import PreparedMarkov
    from ignition.recipes.qwen38_dense import dspark

    config = replace(target, draft=dspark_config.from_dict(copy.deepcopy(DSPARK_CONFIG), target))
    pool = ContinuationPool.for_model(config, 3, 2, "meta", mtp=True)
    requests = DeviceRequestState.allocate("meta", vocab_size=config.vocabulary, capacity=2)
    embedding = SimpleNamespace(weight=torch.empty((config.vocabulary, config.hidden), device="meta"))
    weights = SimpleNamespace(config=config, shard=TensorParallelShard(0, 1),
        tree=SimpleNamespace(embed_tokens=embedding), head=None, draft_head=None)
    owner = dspark.SpeculativePreparation(weights, pool, requests, None, None, completion=None, frame=None,
        sequence=None, session=None, workspace=None, verification_width=8)
    assert (owner.block_size, owner.logit_offset) == (7, 0)
    assert (owner.draft_scratch.tokens, owner.draft_scratch.block_size) == (2 * 7, 7)
    assert owner.draft_scratch.logit_shape == (14, config.vocabulary)
    assert owner.forwards.markov is PreparedMarkov
    assert owner.forwards.context.func is dspark.prepared_context and owner.forwards.draft.func is dspark.PreparedDraft
    assert owner.records.delta.shape == (48, 2, 48, 8, 128)
    assert config.draft.noise_token == 248070
    assert dspark.tail_pages(7) == 2
    key, value = pool.draft_attention(4, config)
    assert key.shape == value.shape == (2, 128, 8, 128) and key.dtype == torch.float8_e4m3fn


def test_rope_scaling_raises_the_preparation_context_ceiling():
    from ignition.recipes.declaration import RopeScaling

    options = dict(draft_snapshot=Path("draft"), speculative_method="dspark", max_active=8,
                   speculative_widths=tuple(range(1, 9)))
    scaled = PreparationConfig(context=1_048_576, rope_scaling=RopeScaling(4., 262144), **options)
    assert scaled.context == 1_048_576
    with pytest.raises(ValueError, match=r"128-aligned in \[128,262144\]"):
        PreparationConfig(context=1_048_576, **options)
    with pytest.raises(ValueError, match=r"128-aligned in \[128,1048576\]"):
        PreparationConfig(context=1_048_576 + 128, rope_scaling=RopeScaling(4., 262144), **options)
    with pytest.raises(ValueError, match="must extend the trained 262,144-token context"):
        PreparationConfig(context=262144, rope_scaling=RopeScaling(4., 131072), **options)


def test_scaled_target_frequencies_interpolate_only_the_slow_pairs():
    from ignition.components.rotary import interpolated_frequencies

    native = 1.0 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    scaled = interpolated_frequencies(10000000, 32, 4., 262144, 32., 1., "cpu")
    assert scaled.shape == native.shape and torch.equal(scaled[0], native[0])
    assert torch.allclose(scaled[-1], native[-1] / 4)
    assert (scaled <= native).all() and (scaled >= native / 4 - 1e-12).all()
