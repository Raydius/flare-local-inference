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

    logits = torch.zeros((5, 4))
    logits[-1, 2] = 1
    sampled = []
    role = SimpleNamespace(row_slot_ids=torch.tensor([0]), row_generations=torch.tensor([1]),
                           live_rows=torch.tensor([1]), sample_enabled=torch.tensor([1]))
    packing = SimpleNamespace(batch=2, width=2, prompt=role)
    runtime = SimpleNamespace(gdn_partitions=SimpleNamespace(parts=(None, SimpleNamespace(gather=lambda x: x))))
    state = SimpleNamespace(prompt=SimpleNamespace(begin_runtime=lambda _: None),
                            combine_runtime=lambda *_: runtime)
    forward = SimpleNamespace(state=state, embedding_injection=SimpleNamespace(validate_runtime=lambda _: None))
    class Forward:
        def __init__(self):
            self.state = forward.state
            self.embedding_injection = forward.embedding_injection

        def __call__(self, _):
            return logits[:4], logits, SimpleNamespace(residual=logits), None

    class Sampler:
        def run(self, values, **_):
            sampled.append(int(values[0].argmax()))

    decoder = DecodeTarget(SimpleNamespace(logits=logits[:4]), Forward(), Sampler())
    _, target_logits, _, _ = decoder(None, packing)
    assert target_logits.shape == (4, 4)
    logits.fill_(float("nan"))
    assert sampled == [2]


@pytest.mark.parametrize("prompt_row", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("width", [1, 3, 6])
@pytest.mark.parametrize("functional", [False, True])
def test_packing_preserves_native_row_order_and_physical_slots(prompt_row, width, functional):
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
    if functional:
        packing.functional()
        packing = packing.runtime()
        assert packing.prompt.row_slot_ids is not p.row_slot_ids
        assert packing.decode.row_slot_ids is not d.row_slot_ids
        p, d = packing.prompt, packing.decode
    else:
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

@pytest.mark.parametrize("speculative", [False, True])
def test_image_validation_precedes_prompt_state_gather(speculative):
    from ignition.recipes.qwen38.mixed_model import DecodeTarget, PreparedMixedRound

    class ForwardComplete(Exception):
        pass

    status = torch.tensor([0])
    errors = torch.tensor([1])
    packing = SimpleNamespace(prompt=object(), decode=object())
    observed = []

    def validate(value):
        assert value is packing
        status.fill_(1)
        return errors

    def begin(value):
        assert value is packing.prompt
        observed.append(status.item())
        return "prompt-state"

    def combine(value, prompt, decode):
        assert value is packing and prompt == "prompt-state"
        return SimpleNamespace()

    class Forward:
        embedding_injection = SimpleNamespace(validate_runtime=validate)
        state = SimpleNamespace(prompt=SimpleNamespace(begin_runtime=begin),
                                decode=SimpleNamespace(begin_runtime=lambda _: "decode-state"),
                                combine_runtime=combine)

        def __call__(self, runtime):
            assert runtime.injection_errors == (errors, packing)
            raise ForwardComplete

    forward = Forward()
    with pytest.raises(ForwardComplete):
        if speculative:
            DecodeTarget(None, forward, None)("decode-state", packing)
        else:
            round_ = PreparedMixedRound.__new__(PreparedMixedRound)
            round_.owner = SimpleNamespace()
            round_.packing = SimpleNamespace(runtime=lambda: packing)
            round_.absorb = None
            round_.enabled = False
            round_.prompt_state = forward.state.prompt
            round_.decoder = forward
            round_()
    assert observed == [1]


@pytest.mark.parametrize("drafting", [False, True])
def test_residual_checkpoint_follows_the_target_residual_region(drafting, monkeypatch):
    """The checkpoint feeds the MTP draft: an MTP layout compiles it, the DFlash layout has no region for it."""
    from ignition.recipes.qwen38 import decoder_state
    from ignition.recipes.qwen38.state import HybridStateLayout

    # The DFlash2 row: seven speculative histories and five draft windows, no MTP regions.
    dflash = HybridStateLayout.build(speculative_tokens=7, world_size=1, mtp=False, draft_windows=(5, 2048, 8, 128))
    compiled = []
    monkeypatch.setattr(decoder_state, "compile_kernel", lambda kernel, args, grid: compiled.append(kernel) or kernel)
    for layout, expected in ((HybridStateLayout.build(speculative_tokens=7), True),
                             (HybridStateLayout.build(), False), (dflash, False)):
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
    """The packed forward writes one tapped-feature buffer; finish_runtime() splits it the way hidden and residual split."""
    from ignition.components.recurrent import DeltaMetadata
    from ignition.recipes.qwen38.mixed_state import FunctionalStatePartition, MixedScratch, PreparedMixedState
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
                               prepare_hidden=lambda *a: None, prepare_residual=lambda *a: None,
                               finish_runtime=lambda runtime, hidden, residual: hidden[:0])

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
    parts = tuple(FunctionalStatePartition(mode, None, indices, valid) for mode, indices, valid in (
        ("prefill", packing.prompt_indices, packing.prompt_valid), ("decode", packing.decode_indices, packing.decode_valid)))
    state.finish_runtime(SimpleNamespace(gdn_partitions=SimpleNamespace(parts=parts), prompt=None, decode=None),
                         torch.zeros((tokens, 2560), dtype=torch.bfloat16), torch.zeros((tokens, 10240), dtype=torch.bfloat16))
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
