# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint catalog coverage and dense native-to-role packing for mixed Qwen graphs."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from ignition.components.capacity import batch_capacities, mixed_shapes, prefill_capacities
from ignition.recipes.qwen38.bootstrap import PreparationConfig
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
from ignition.recipes.qwen38.mixed_state import MixedPacking, role_inputs
from ignition.recipes.qwen38.recipe import RECIPE
from ignition.recipes.declaration import Speculation


@pytest.mark.parametrize("concurrency", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("widths", [(1,), (1, 3, 6), tuple(range(1, 9))])
def test_joint_catalog_covers_both_axes_without_cross_product(concurrency, widths):
    prompts = prefill_capacities(4096)
    shapes = mixed_shapes(concurrency, 4096, widths)
    batches = batch_capacities(concurrency)[1:]
    budget = max(batch_capacities(concurrency)[-1], max(widths))
    assert len(shapes) == sum(len(prompts) + sum(w > 1 and b * w <= budget for w in widths) for b in batches)
    assert len(shapes) == len(set(shapes))
    for batch in batches:
        assert {(b, 1, p) for b in (batch,) for p in prompts} <= set(shapes)
        for width in widths:
            assert any(b == batch and w == width for b, w, _ in shapes) == (width == 1 or batch * width <= budget)
    config = PreparationConfig(max_active=concurrency, mtp=True, speculative_widths=widths, mixed_prefill=True)
    assert all(p + (b - 1) * w <= config.token_capacity for _, b, w, p in config.mixed_shapes)
    assert all(w == 1 for enabled, _, w, _ in config.mixed_shapes if not enabled)


@pytest.mark.parametrize("mixed", [False, True])
def test_plan_and_preparation_use_identical_catalogs(mixed, monkeypatch):
    from ignition import planning
    from ignition.recipes.qwen38 import bootstrap

    gpu = planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-test", 96 << 30, 95 << 30, (12, 0))
    monkeypatch.setattr(bootstrap, "prepare_model", lambda snapshot, config, **kwargs: config)
    for speculation in (None, Speculation(0), Speculation(5), Speculation(3, adaptive=False)):
        plan = planning.plan(RECIPE, RECIPE.checkpoints[0], (gpu,), selected=(0,),
            concurrency=4, reserve_bytes=0, speculation=speculation, mixed_prefill=mixed)
        config = RECIPE.prepare(Path("/checkpoint"), plan, compile_forwards=True, kernel_cache=None, progress=None)
        expected = RECIPE.mixed_shapes(4, 4096, speculation) if mixed else ()
        assert config.mixed_prefill == plan.to_dict()["mixed_prefill"] == mixed
        assert config.mixed_shapes == plan.mixed_shapes == expected
        assert plan.to_dict()["mixed_shapes"] == expected
        assert config.mtp == (speculation is not None)
    assert len(PreparationConfig(mtp=True, mixed_prefill=True).mixed_shapes) == 42
    assert PreparationConfig(max_active=1, mtp=True, mixed_prefill=True).mixed_shapes == ()


def test_mixed_capture_defaults_off_without_changing_speculation():
    for mtp in (False, True):
        config = PreparationConfig(mtp=mtp)
        assert not config.mixed_prefill and config.mixed_shapes == ()
        assert config.token_capacity == config.prefill_chunk_size


def test_prompt_sampling_precedes_draft_scratch_reuse():
    from ignition.recipes.qwen38.mixed_model import DecodeTarget

    logits = torch.empty((3, 4))
    scratch = SimpleNamespace(logits=logits[:2])
    sampled = []

    def target():
        logits.fill_(0)
        logits[-1, 2] = 1

    decoder = DecodeTarget(scratch, target, lambda: sampled.append(int(logits[-1].argmax())))
    decoder()
    logits.fill_(float("nan"))
    assert sampled == [2]
    assert decoder.scratch.logits.shape == (2, 4)


@pytest.mark.parametrize("prompt_row", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("width", [1, 3, 6])
def test_packing_preserves_native_row_order_and_physical_slots(prompt_row, width):
    batch, prefill, length = 4, 128, 65
    tokens = prefill + (batch - 1) * width
    inputs = DecoderInputs.allocate("cpu", context=256, tokens=tokens, capacity=batch)
    p, d = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
    packing = MixedPacking(inputs, p, d, batch, width, prefill)
    inputs.live_rows.fill_(batch)
    inputs.prefill_row.fill_(prompt_row)
    inputs.row_slot_ids.copy_(torch.tensor([2, 0, 3, 1]))
    inputs.row_generations.copy_(torch.tensor([21, 32, 43, 54]))
    inputs.query_lengths.fill_(width)
    if prompt_row >= 0:
        inputs.query_lengths[prompt_row] = length
    inputs.token_ids.copy_(torch.arange(tokens) + 1000)
    packing.upload_roles()
    assert d.row_slot_ids.tolist() == [(-1 if r == prompt_row else s) for r, s in enumerate([2, 0, 3, 1])]
    prompt_ids = torch.full((prefill,), -1)
    if prompt_row >= 0:
        prompt_ids[:length] = 0
        start = int(inputs.query_lengths[:prompt_row].sum())
        assert p.token_ids[:length].tolist() == list(range(1000 + start, 1000 + start + length))
    else:
        assert p.live_rows.item() == 0
    decode_ids = torch.arange(batch).repeat_interleave(width)
    decode_ids[decode_ids == prompt_row] = -1
    prompt = SimpleNamespace(qsa=SimpleNamespace(request_ids=prompt_ids))
    decode = SimpleNamespace(qsa=SimpleNamespace(request_ids=decode_ids),
        gdn=SimpleNamespace(query_start_loc=torch.arange(batch + 1) * width))
    packing.bind_metadata(prompt, decode)
    packed = torch.zeros(tokens, dtype=torch.int64)
    packing.combine((p.token_ids, inputs.token_ids[packing.decode_indices]), packed)
    live = int(inputs.query_lengths.sum())
    assert packed[:live].tolist() == inputs.token_ids[:live].tolist()
    assert not packing.prompt_valid.any() if prompt_row == -1 else packing.prompt_valid.sum() == length


@pytest.mark.parametrize("drafting", [False, True])
def test_residual_checkpoint_follows_the_target_residual_region(drafting, monkeypatch):
    """The checkpoint feeds the MTP draft: an MTP layout compiles it, the DFlash layout has no region for it."""
    from ignition.recipes.qwen38 import decoder_state
    from ignition.recipes.qwen38.state import HybridStateLayout
    from ignition.recipes.qwen38_dflash.bootstrap import state_layout

    compiled = []
    monkeypatch.setattr(decoder_state, "compile_kernel", lambda kernel, args, grid: compiled.append(kernel) or kernel)
    for layout, expected in ((HybridStateLayout.build(speculative_tokens=7), True),
                             (HybridStateLayout.build(), False), (state_layout(True), False)):
        pool = SimpleNamespace(layout=layout, region=lambda name: torch.zeros((2, 4)))
        state = PreparedDecoderState.__new__(PreparedDecoderState)
        state.pool, state.maintain_draft_state, state.rows = pool, drafting, 1
        state.requests = SimpleNamespace(capacity=2)
        state.inputs = SimpleNamespace(row_slot_ids=None, request_state_rows=None)
        state.qsa, state.gdn = SimpleNamespace(rope_positions=None), SimpleNamespace(query_start_loc=None)
        state.token_state = SimpleNamespace(active=None)
        state._residual = None
        state.prepare_residual(torch.zeros(1))
        assert (state._residual is not None) == (drafting and expected)
    assert layout.speculative_tokens == 7 and compiled == ([decoder_state._residual_checkpoint] if drafting else [])


@pytest.mark.parametrize("prompt_row", [-1, 1])
def test_packed_features_are_gathered_per_role(prompt_row):
    """The packed forward writes one tapped-feature buffer; finish() splits it the way hidden and residual split."""
    from ignition.components.recurrent import DeltaMetadata
    from ignition.recipes.qwen38.mixed_state import MixedScratch, PreparedMixedState
    from ignition.recipes.qwen38.ple import PLEMetadata

    batch, width, prefill, length, feature_width = 4, 3, 16, 5, 8
    tokens = prefill + (batch - 1) * width
    inputs = DecoderInputs.allocate("cpu", context=256, tokens=tokens, capacity=batch)
    p, d = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
    packing = MixedPacking(inputs, p, d, batch, width, prefill)
    scratch = MixedScratch.allocate(prefill, batch * width, "cpu", world_size=1)

    def role(n, rows):
        gdn = DeltaMetadata.allocate(rows, "cpu")
        ple = PLEMetadata(torch.zeros(n, dtype=torch.int64), gdn.query_start_loc, torch.zeros((rows, 2), dtype=torch.int64),
                          gdn.convolution_slots, gdn.fresh, gdn.accepted_tokens, gdn.num_seqs, gdn.num_tokens)
        return SimpleNamespace(tokens=n, gdn=gdn, ple=ple, requests=None, pool=None, inputs=inputs,
                               prepare_hidden=lambda *a: None, prepare_residual=lambda *a: None, finish=lambda: None)

    prompt, decode = role(prefill, 1), role(batch * width, batch)
    features = torch.arange(tokens, dtype=torch.bfloat16)[:, None].expand(-1, feature_width).contiguous()
    prompt_features = torch.full((prefill, feature_width), -1, dtype=torch.bfloat16)
    decode_features = torch.full((batch * width, feature_width), -1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="packed buffer and both role buffers"):
        PreparedMixedState(packing, prompt, decode, scratch, prompt_residual=scratch.prompt_residual,
                           decode_residual=scratch.decode_residual, features=features)
    state = PreparedMixedState(packing, prompt, decode, scratch, prompt_residual=scratch.prompt_residual,
        decode_residual=scratch.decode_residual, features=features, prompt_features=prompt_features,
        decode_features=decode_features)
    state.prepare_hidden(torch.zeros((tokens, 2560), dtype=torch.bfloat16), torch.zeros((batch * width + 1, 2560), dtype=torch.bfloat16))
    state.prepare_residual(torch.zeros((tokens, 10240), dtype=torch.bfloat16))
    inputs.live_rows.fill_(batch)
    inputs.prefill_row.fill_(prompt_row)
    inputs.row_slot_ids.copy_(torch.tensor([2, 0, 3, 1]))
    inputs.query_lengths.fill_(width)
    if prompt_row >= 0:
        inputs.query_lengths[prompt_row] = length
    packing.upload_roles()
    prompt_ids = torch.full((prefill,), -1)
    if prompt_row >= 0:
        prompt_ids[:length] = 0
    decode_ids = torch.arange(batch).repeat_interleave(width)
    decode_ids[decode_ids == prompt_row] = -1
    packing.bind_metadata(SimpleNamespace(qsa=SimpleNamespace(request_ids=prompt_ids)),
        SimpleNamespace(qsa=SimpleNamespace(request_ids=decode_ids),
                        gdn=SimpleNamespace(query_start_loc=torch.arange(batch + 1) * width)))
    state.finish()
    lengths = inputs.query_lengths.tolist()
    starts = [sum(lengths[:r]) for r in range(batch)]
    expected_prompt = torch.zeros_like(prompt_features)
    if prompt_row >= 0:
        expected_prompt[:length] = features[starts[prompt_row]:starts[prompt_row] + length]
    expected_decode = torch.zeros_like(decode_features)
    for row in range(batch):
        if row != prompt_row:
            expected_decode[row * width:(row + 1) * width] = features[starts[row]:starts[row] + width]
    torch.testing.assert_close(prompt_features, expected_prompt, rtol=0, atol=0)
    torch.testing.assert_close(decode_features, expected_decode, rtol=0, atol=0)
    assert PreparedMixedState(packing, prompt, decode, scratch, prompt_residual=scratch.prompt_residual,
                              decode_residual=scratch.decode_residual).features is None


def _mixed_round(monkeypatch, *, taps):
    """A mixed round on CPU with its Triton-backed pieces stubbed: what it allocates and hands each half."""
    from ignition.components.completion import CompletionOutputs
    from ignition.components.recurrent import DeltaMetadata
    from ignition.preparation.scratch import GraphBufferPool
    from ignition.recipes.qwen38 import mixed_model
    from ignition.recipes.qwen38.decoder import DecoderScratch
    from ignition.recipes.qwen38.mixed_state import MixedScratch
    from ignition.recipes.qwen38.ple import PLEMetadata

    batch, width, prefill = 2, 3, 16
    tokens = prefill + (batch - 1) * width
    seen = SimpleNamespace(decoders=[], halves=[])

    class State:
        def __init__(self, inputs, requests, pool, *, tokens, rows, mode, maintain_draft_state):
            self.inputs, self.requests, self.pool, self.tokens, self.rows, self.mode = inputs, requests, pool, tokens, rows, mode
            self.gdn = DeltaMetadata.allocate(rows, "cpu")
            self.ple = PLEMetadata(torch.zeros(tokens, dtype=torch.int64), self.gdn.query_start_loc,
                torch.zeros((rows, 2), dtype=torch.int64), self.gdn.convolution_slots, self.gdn.fresh,
                self.gdn.accepted_tokens, self.gdn.num_seqs, self.gdn.num_tokens)

    class Decoder:
        def __init__(self, weights, scratch, state, *args, **kwargs):
            self.scratch, self.state, self.kwargs = scratch, state, kwargs
            seen.decoders.append(self)

    class Sampler:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            pass

        prepare_broadcast = prepare

    class Half:
        def __init__(self, owner, *args, **kwargs):
            self.owner, self.args, self.kwargs = owner, args, kwargs
            seen.halves.append(self)
            factory = kwargs.get("decoder_factory")
            if factory is None:
                args[0].residual_output = kwargs["residual_output"]
            else:
                b, w = args[:2]
                extra = {}
                if getattr(owner, "feature_taps", ()):
                    extra = dict(feature_taps=owner.feature_taps, auxiliary=owner.auxiliary(b * w, kwargs["buffers"]))
                self.decoder = factory(owner.weights, owner.scratch.rows(b * w, b * w), State(owner.inputs, None, None,
                    tokens=b * w, rows=b, mode="decode", maintain_draft_state=False), None, None, None, None, None, None,
                    None, session=None, name="round.target", max_cache_pages=None,
                    residual_output=kwargs["residual_output"], verification=True, **extra, **owner.collectives)

        def prepare(self):
            pass

    monkeypatch.setattr(mixed_model, "PreparedDecoderState", State)
    monkeypatch.setattr(mixed_model, "PreparedDecoder", Decoder)
    monkeypatch.setattr(mixed_model, "PreparedSampling", Sampler)
    monkeypatch.setattr(mixed_model, "MixedEmbeddingInjection", lambda *args: None)
    sampling_pool, decoder_pool = GraphBufferPool("cpu", chunk_bytes=1 << 20), GraphBufferPool("cpu", chunk_bytes=1 << 20)
    owner = SimpleNamespace(weights=None, inputs=DecoderInputs.allocate("cpu", context=256, tokens=tokens, capacity=batch),
        requests=None, pool=None, scratch=DecoderScratch.allocate(tokens, "cpu", sample_rows=batch * width + 1,
                                                                   buffers=decoder_pool.cursor()),
        completion=CompletionOutputs.allocate((2, batch, 24), device="cpu"), session=None, owners=[],
        collectives=dict(vocabulary_runtime=None, coordinator=None), sampling_workspace=None, frame=None, sequence=None,
        cursor=sampling_pool.cursor, auxiliary=lambda n, buffers: buffers.zeros((n, 2 * 2560), torch.bfloat16))
    if taps:
        owner.feature_taps = (5, 15)
    scratch = MixedScratch.allocate(prefill, batch * width, "cpu", world_size=1)
    round = mixed_model.PreparedMixedRound(owner, batch, width, prefill, "mixed", scratch=scratch, features=None,
                                           enabled=True, absorption=Half, speculative_round=Half)
    return round, seen, owner, scratch, sampling_pool


def test_mixed_round_feature_buffers_come_from_one_drafter_cursor(monkeypatch):
    from ignition.preparation.disposable import storage_ranges

    round, seen, owner, scratch, sampling_pool = _mixed_round(monkeypatch, taps=True)
    prompt, decode = seen.halves
    assert prompt.kwargs["buffers"] is decode.kwargs["buffers"] and prompt.kwargs["buffers"].pool is sampling_pool
    assert round.prompt_decoder.auxiliary is round.prompt_features and round.prompt_features.shape == (16, 5120)
    (decoder,), = (seen.decoders,)
    assert decoder.kwargs["feature_taps"] == (5, 15) and decoder.kwargs["auxiliary"] is round.features
    assert round.features.shape == (round.packing.tokens, 5120) and decoder.state.decode_features.shape == (6, 5120)
    assert decoder.state.features is round.features and decoder.state.prompt_features is round.prompt_features

    def overlaps(a, b):
        return any(x < y + n and y < x + m for x, m in a for y, n in b)

    features = storage_ranges([round.features, round.prompt_features, decoder.state.decode_features])
    assert not overlaps(features, storage_ranges([owner.scratch])) and not overlaps(features, storage_ranges([scratch]))
    assert overlaps(features, storage_ranges(sampling_pool.chunks))


def test_mixed_round_without_taps_keeps_the_mtp_call_shape(monkeypatch):
    import inspect
    from ignition.recipes.qwen38.mixed_model import PreparedMixedRound
    from ignition.recipes.qwen38.speculative_model import PreparedPrefillAbsorption, PreparedSpeculativeRound

    parameters = inspect.signature(PreparedMixedRound).parameters
    assert parameters["absorption"].default is PreparedPrefillAbsorption
    assert parameters["speculative_round"].default is PreparedSpeculativeRound
    round, seen, owner, scratch, sampling_pool = _mixed_round(monkeypatch, taps=False)
    prompt, decode = seen.halves
    assert round.features is None and round.prompt_features is None and sampling_pool.chunks == {}
    assert prompt.args == (round.prompt_decoder, "mixed.prompt") and set(prompt.kwargs) == {"residual_output"}
    assert decode.args == (2, 3, "mixed") and set(decode.kwargs) == {"decoder_factory", "residual_output"}
    (decoder,), = (seen.decoders,)
    assert "auxiliary" not in decoder.kwargs and "feature_taps" not in decoder.kwargs
    assert decoder.state.features is None
