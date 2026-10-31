# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed target metadata and state endpoints through native CUDA graph replay."""

import os

import pytest
import torch

from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState
from ignition.recipes.qwen38.mixed_state import MixedPacking, MixedScratch, PreparedMixedState, role_inputs
from ignition.recipes.qwen38.mixed_model import MixedEmbeddingInjection
from ignition.recipes.qwen38.sampling import DeviceRequestState
from ignition.recipes.qwen38.speculative_state import PreparedVerificationState
from ignition.recipes.qwen38.state import HybridStatePool
from ignition.recipes.qwen38.vision_state import VisionFeatureState

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


@pytest.mark.parametrize("enabled,width", [(False, 1), (True, 1), (True, 3), (True, 6)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("live", [1, 3, 4])
@torch.no_grad()
def test_mixed_metadata_and_committed_endpoints(enabled, width, compiled, live):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    batch, prefill = 4, 128
    tokens = prefill + (batch - 1) * width
    pool = HybridStatePool.allocate(5, device, speculative_tokens=7 if enabled else 0)
    requests = DeviceRequestState.allocate(device, capacity=batch)
    inputs = DecoderInputs.allocate(device, context=256, tokens=tokens, capacity=batch)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=device)
    inputs.page_tables.copy_(torch.arange(16, device=device).view(4, 4))
    pi, di = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
    p = PreparedDecoderState(pi, requests, pool, tokens=prefill, rows=1, mode="prefill",
                             maintain_draft_state=enabled)
    if enabled:
        candidates = torch.arange(32, device=device).view(4, 8).long() + 100
        d = PreparedVerificationState(di, requests, pool, candidates,
            torch.ones(batch, dtype=torch.int32, device=device), width=width, rows=batch,
            token_limits=di.query_lengths[:batch])
    else:
        d = PreparedDecoderState(di, requests, pool, tokens=batch, rows=batch, mode="decode",
                                 maintain_draft_state=False)
    packing = MixedPacking(inputs, pi, di, batch, width, prefill)
    scratch = MixedScratch.allocate(prefill, batch * width, device, world_size=1)
    state = PreparedMixedState(packing, p, d, scratch,
        prompt_residual=scratch.prompt_residual, decode_residual=scratch.decode_residual)
    hidden = (torch.arange(tokens, device=device) + 100).bfloat16()[:, None].expand(-1, 2560).contiguous()
    residual = (torch.arange(tokens, device=device) + 10).bfloat16()[:, None].expand(-1, 10240).contiguous()
    sample = torch.zeros((batch * width + 1, 2560), dtype=torch.bfloat16, device=device)
    state.prepare_hidden(hidden, sample)
    state.prepare_residual(residual)
    features = VisionFeatureState.allocate(device, capacity=batch)
    features.generation.fill_(7)
    features.count.fill_(1)
    features.arena[:, 0].fill_(77)
    embedding = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
    injection = MixedEmbeddingInjection(packing, features, p, embedding, scratch.prompt_gdn.output)

    def forward():
        packing.upload_roles()
        injection.validate()
        state.begin()
        injection.inject()
        state.finish()

    function = forward
    if compiled:
        from ignition.preparation.compilation import compile_forward
        function = compile_forward(forward)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        function()
    worker = PreparedGraphBundle(device=0, graphs={"mixed": graph}, buffers={"sample": sample},
        resources=(pool, requests, inputs, p, d, packing, scratch, state, hidden, residual, function)).start_worker()
    try:
        for prompt_row in (-1, 0, 1, 3):
            if prompt_row >= live:
                continue
            for stale in (False, True):
                pool.backing.zero_()
                requests.generation.fill_(7)
                requests.status.fill_(1)
                requests.consumed_tokens.fill_(63)
                requests.pending_token.fill_(99)
                inputs.row_slot_ids.copy_(torch.tensor([2, 0, 3, 1], device=device))
                inputs.row_generations.fill_(7)
                inputs.live_rows.fill_(live)
                inputs.query_lengths.fill_(width)
                inputs.query_lengths[2] = 1
                inputs.query_lengths[live:].zero_()
                inputs.prefill_row.fill_(prompt_row)
                inputs.token_ids.copy_(torch.arange(tokens, device=device) + 1000)
                inputs.vision_token_indices.fill_(-1)
                features.feature_slot.fill_(-1)
                embedding.fill_(-4)
                inputs.prefill_positions.copy_((torch.arange(tokens, device=device) + 20)[:, None].expand(-1, 3))
                stale_row = (prompt_row + 1) % 4
                if stale:
                    inputs.row_generations[stale_row] = 6
                if prompt_row >= 0:
                    prompt_slot = int(inputs.row_slot_ids[prompt_row])
                    requests.pending_token[prompt_slot] = -1
                    requests.consumed_tokens[prompt_slot] = 0
                    inputs.query_lengths[prompt_row] = 65
                lengths = inputs.query_lengths.tolist()
                starts = [sum(lengths[:r]) for r in range(4)]
                if prompt_row >= 0:
                    inputs.vision_token_indices[starts[prompt_row] + 17] = 0
                    features.feature_slot.fill_(0)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(30000)
                assert completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                for row, slot in enumerate([2, 0, 3, 1]):
                    start, length = starts[row], lengths[row]
                    active = row < live and not (stale and row == stale_row)
                    assert state.qsa.request_ids[start:start + length].tolist() == [row if active else -1] * length
                    assert state.qsa.is_prefilling[row].item() == (row == prompt_row)
                    consumed = int(requests.consumed_tokens[slot])
                    expected = 65 if row == prompt_row else 63 + int(active and not enabled)
                    assert consumed == expected
                    if row == prompt_row or active and not enabled:
                        assert pool.last_hidden[slot + 1, 0] == hidden[start + length - 1, 0]
                    else:
                        assert not pool.last_hidden[slot + 1].any()
                assert (state.qsa.request_ids[sum(lengths):] == -1).all()
                assert state.qsa.query_start_loc.tolist() == [0, *[sum(lengths[:r + 1]) for r in range(4)]]
                expected_embedding = torch.full_like(embedding, -4)
                if prompt_row >= 0:
                    expected_embedding[starts[prompt_row] + 17] = 77
                torch.testing.assert_close(embedding, expected_embedding, rtol=0, atol=0)
    finally:
        worker.close()
        graph.reset()


@torch.no_grad()
def test_dflash_mixed_state_writes_the_rings_a_prefill_and_a_round_would(tmp_path):
    """A DFlash mixed round's prompt and decode ring writes equal a prefill's and a verification round's.

    The packed target forward is stood in for by features that are a function of each
    token's request slot and position, so both paths see the same per-token features; the
    draft context projections and ring writers are the pinned draft's, run eagerly. The prompt
    chunk is longer than the 2048-position window. Ring agreement is bounded by the E4M3
    contract's measured repeatability (``tests/recipes/qwen38_dflash/test_draft_cuda.py``:
    rings 2e-2 relative per row), since the two paths project different token counts.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from safetensors import safe_open

    from b12x.preparation import PreparationSession
    from ignition.components.dflash_draft import PreparedContext, Rotary
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.qwen38_dflash.config import DraftConfig
    from ignition.recipes.qwen38_dflash.draft_weights import DraftWeights
    from ignition.recipes.qwen38_dflash.recipe import DRAFT_WINDOWS
    from ignition.recipes.qwen38_dflash.speculative_model import (
        PreparedMixedPromptContext, prefill_feature_rows, rings, verification_feature_rows,
    )

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    draft = Path(os.environ.get("FLARE_QWEN38_DFLASH2_CHECKPOINT", "~/models/incoai/Qwen3.8-Flash-Next-DFlash2")).expanduser()
    if not draft.is_dir():
        pytest.skip("requires the pinned DFlash2 draft")
    device = torch.device("cuda", 0)
    config = DraftConfig.load(draft)
    with safe_open(str(draft / "model.safetensors"), "pt", device=str(device)) as file:
        draft_weights = DraftWeights.assemble(config, file.get_tensor)
    window, feature_width = config.window, len(config.target_layers) * 2560
    batch, width, prefill, length = 4, 3, window + 64, window + 52
    prompt_row, prompt_slot, consumed = 1, 0, 63
    counts = [3, 0, 2, 1]
    tokens = prefill + (batch - 1) * width
    slots = torch.tensor([2, 0, 3, 1], device=device)

    def features_of(positions, slot_of_token):
        # One deterministic feature row per (slot, position), in the E4M3-representable range.
        channel = torch.arange(feature_width, device=device)
        value = ((slot_of_token[:, None] * 4099 + positions[:, None] * 17 + channel[None]) % 89 - 44) / 64
        return torch.where(positions[:, None] >= 0, value, 0).to(torch.bfloat16)

    def fresh():
        pool = HybridStatePool.allocate(5, device, speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS)
        requests = DeviceRequestState.allocate(device, capacity=batch)
        requests.generation.fill_(7)
        requests.status.fill_(1)
        requests.consumed_tokens.fill_(consumed)
        requests.pending_token.fill_(99)
        requests.pending_token[prompt_slot] = -1
        requests.consumed_tokens[prompt_slot] = 0
        return pool, requests

    def inputs_for(capacity_tokens, rows):
        inputs = DecoderInputs.allocate(device, context=4096, tokens=capacity_tokens, capacity=batch)
        inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=device)
        inputs.page_tables.copy_(torch.arange(4 * 64, device=device).view(4, 64))
        inputs.row_generations.fill_(7)
        inputs.live_rows.fill_(rows)
        inputs.vision_token_indices.fill_(-1)
        return inputs

    def context_for(auxiliary, name, buffers):
        return PreparedContext(draft_weights, auxiliary, rings(config, pool), window=window, epsilon=config.eps,
            rotary=Rotary(config, device), session=session, name=name, workspace=arena, buffers=buffers)

    def ring_rows(pool, slot):
        return [ring[inputs_rows[slot]].float() for pair in rings(config, pool) for ring in pair]

    inputs_rows = [1, 2, 3, 4]
    results = {}
    with PreparationSession(device=str(device), autotune=False) as session:
        arena = ScratchArena(device)
        cursor = GraphBufferPool(device).cursor()

        # The mixed state: one prompt chunk and a verification batch packed into one forward.
        pool, requests = fresh()
        inputs = inputs_for(tokens, batch)
        inputs.row_slot_ids.copy_(slots.int())
        inputs.query_lengths.fill_(width)
        inputs.query_lengths[prompt_row] = length
        inputs.prefill_row.fill_(prompt_row)
        inputs.token_ids.copy_(torch.arange(tokens, device=device) % 1000 + 1000)
        starts = [sum(inputs.query_lengths[:r].tolist()) for r in range(batch)]
        inputs.prefill_positions[starts[prompt_row]:starts[prompt_row] + length].copy_(
            torch.arange(length, device=device)[:, None].expand(-1, 3))
        pi, di = role_inputs(inputs, prefill), role_inputs(inputs, batch * width)
        p = PreparedDecoderState(pi, requests, pool, tokens=prefill, rows=1, mode="prefill", maintain_draft_state=True)
        candidates = torch.arange(32, device=device).view(4, 8).long() + 100
        d = PreparedVerificationState(di, requests, pool, candidates, torch.ones(batch, dtype=torch.int32, device=device),
                                      width=width, rows=batch, token_limits=di.query_lengths[:batch])
        packing = MixedPacking(inputs, pi, di, batch, width, prefill)
        scratch = MixedScratch.allocate(prefill, batch * width, device, world_size=1)
        packed = cursor.zeros((tokens, feature_width), torch.bfloat16)
        prompt_features = cursor.zeros((prefill, feature_width), torch.bfloat16)
        decode_features = cursor.zeros((batch * width, feature_width), torch.bfloat16)
        state = PreparedMixedState(packing, p, d, scratch, prompt_residual=scratch.prompt_residual,
            decode_residual=scratch.decode_residual, features=packed, prompt_features=prompt_features,
            decode_features=decode_features)
        state.prepare_hidden(torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device),
                             torch.zeros((batch * width + 1, 2560), dtype=torch.bfloat16, device=device))
        state.prepare_residual(torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device))
        # The DFlash layout holds no target_residual region, so the prompt role compiles no checkpoint.
        assert p._residual is None
        prompt_owner = SimpleNamespace(inputs=pi, requests=requests, window=window, context=context_for)
        prompt_decoder = SimpleNamespace(state=p, auxiliary=prompt_features)
        prompt_context = PreparedMixedPromptContext(prompt_owner, prompt_decoder, "mixed.prompt",
                                                    residual_output=scratch.prompt_residual, buffers=cursor)
        assert prompt_decoder.residual_output is scratch.prompt_residual
        decode_context = context_for(decode_features, "mixed.decode", cursor)
        arena.finalize()
        packing.upload_roles()
        prompt_context.begin()
        state.begin()
        slot_of_token = slots[packing.row_ids]
        packed.copy_(features_of(state.qsa.query_positions, slot_of_token))
        state.finish()
        decode_owner = SimpleNamespace(inputs=di, requests=requests)
        emitted = torch.tensor(counts, dtype=torch.int32, device=device)
        decode_context(decode_features, *verification_feature_rows(decode_owner, d.qsa, batch, width, emitted))
        prompt_context.finish()
        torch.cuda.synchronize()
        assert int(requests.consumed_tokens[prompt_slot]) == length
        assert state.qsa.query_positions[starts[prompt_row]:starts[prompt_row] + length].tolist() == list(range(length))
        results["mixed"] = {slot: ring_rows(pool, slot) for slot in range(batch)}
        mixed_pool = pool

        # The same prompt through a prefill state and the same batch through a verification state.
        pool, requests = fresh()
        inputs = inputs_for(prefill, 1)
        inputs.row_slot_ids.fill_(-1)
        inputs.row_slot_ids[0] = prompt_slot
        inputs.query_lengths.zero_()
        inputs.query_lengths[0] = length
        inputs.token_ids.copy_(torch.arange(prefill, device=device) % 1000 + 1000)
        inputs.prefill_positions[:length].copy_(torch.arange(length, device=device)[:, None].expand(-1, 3))
        p2 = PreparedDecoderState(inputs, requests, pool, tokens=prefill, rows=1, mode="prefill", maintain_draft_state=True)
        features2 = cursor.zeros((prefill, feature_width), torch.bfloat16)
        context2 = context_for(features2, "prefill", cursor)
        owner2 = SimpleNamespace(inputs=inputs, requests=requests)
        p2.prepare_hidden(torch.zeros((prefill, 2560), dtype=torch.bfloat16, device=device),
                          torch.zeros((1, 2560), dtype=torch.bfloat16, device=device))
        p2.prepare_residual(torch.zeros((prefill, 10240), dtype=torch.bfloat16, device=device))
        assert p2._residual is None
        p2.begin()
        features2.copy_(features_of(p2.qsa.query_positions, torch.full((prefill,), prompt_slot, device=device)))
        p2.finish()
        context2(features2, *prefill_feature_rows(owner2, p2.qsa, window))
        decode_inputs = inputs_for(batch * width, batch)
        decode_slots = torch.where(torch.arange(batch, device=device) == prompt_row, -1, slots)
        decode_inputs.row_slot_ids.copy_(decode_slots.int())
        decode_inputs.query_lengths.fill_(width)
        decode_inputs.query_lengths[prompt_row] = 0
        d2 = PreparedVerificationState(decode_inputs, requests, pool, candidates,
            torch.ones(batch, dtype=torch.int32, device=device), width=width, rows=batch,
            token_limits=decode_inputs.query_lengths[:batch])
        features3 = cursor.zeros((batch * width, feature_width), torch.bfloat16)
        context3 = context_for(features3, "round", cursor)
        d2.begin()
        features3.copy_(features_of(d2.qsa.query_positions, decode_slots.clamp_min(0).repeat_interleave(width)))
        context3(features3, *verification_feature_rows(SimpleNamespace(inputs=decode_inputs, requests=requests),
                                                       d2.qsa, batch, width, emitted))
        torch.cuda.synchronize()
        assert int(requests.consumed_tokens[prompt_slot]) == length
        results["separate"] = {slot: ring_rows(pool, slot) for slot in range(batch)}
        separate_pool = pool

    # Every slot's rings agree between the paths within the contract's repeatability, and the
    # written slots are exactly the prompt's last `window` positions and each row's emitted columns.
    for slot in range(batch):
        for mixed, separate in zip(results["mixed"][slot], results["separate"][slot], strict=True):
            written = separate.flatten(1).abs().sum(1) > 0
            assert (mixed.flatten(1).abs().sum(1) > 0).tolist() == written.tolist()
            if written.any():
                error = (mixed[written] - separate[written]).norm() / separate[written].norm()
                assert error.item() <= 2e-2, (slot, error.item())
        rows_written = int((results["separate"][slot][0].flatten(1).abs().sum(1) > 0).sum())
        if slot == prompt_slot:
            assert rows_written == window
        else:
            assert rows_written == counts[slots.tolist().index(slot)]
    del mixed_pool, separate_pool
