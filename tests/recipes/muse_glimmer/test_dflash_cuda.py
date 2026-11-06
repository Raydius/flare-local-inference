# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One width-16 speculative round of the prepared Muse Glimmer drafter against the installed assistant.

The prepared prefill of the traced prompt supplies the target taps (layers 1, 13, 25, 37, 49)
and fills the draft rings through ``PreparedContext``; the round runs through
``PreparedParallelDraftRound`` exactly as the catalog's ``speculative1x16`` graph does. The
oracle is ``MuseGlimmerAssistantModel`` over the same taps and the block
``[pending token, 201818 x 15]`` embedded raw (no embedding norm), with the inclusive
bidirectional sliding mask Transformers declares for the config and keys and values rounded
to E4M3 at the cache seam (``Fp8History``) as the rings and staged pages store them.

Tolerance (BF16 accumulation order through five layers): relative L2 at most 2e-2 on the
block's normalized hidden states and on its softcapped FP32 logits, over the fifteen proposal
rows and over the anchor row.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tests.recipes.muse_glimmer.test_reference_cuda import (
    CHECKPOINT, CUDA, SNAPSHOT, WINDOW, device, prefill, relative_l2, softcapped, traced_prompt,
)

ASSISTANT_SNAPSHOT = os.environ.get("FLARE_MUSE_GLIMMER_ASSISTANT")
ASSISTANT = pytest.mark.skipif(not ASSISTANT_SNAPSHOT, reason="FLARE_MUSE_GLIMMER_ASSISTANT names no assistant checkpoint")
RELATIVE_L2 = .02
BLOCK = 16


@CUDA
@CHECKPOINT
@ASSISTANT
@torch.no_grad()
def test_width_16_round_proposes_the_assistant_block():
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_glimmer.decoder import Scratch
    from ignition.recipes.muse_glimmer.dflash import SpeculativePreparation
    from ignition.recipes.muse_glimmer.weights import ModelWeights
    from ignition.recipes.muse_spark.pool import PAGE_TOKENS, ContinuationPool
    from ignition.reference.muse_glimmer import Fp8History
    from transformers.models.muse_glimmer_assistant import MuseGlimmerAssistantModel

    snapshot, assistant_snapshot, selected = Path(SNAPSHOT), Path(ASSISTANT_SNAPSHOT), device()
    ids = traced_prompt(snapshot)
    tokens = len(ids)
    assert tokens > WINDOW
    shard = TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(snapshot), selected, shard=shard,
                                draft_snapshot=assistant_snapshot)
    c, draft = weights.config.text, weights.config.draft
    assert draft.block_size == BLOCK and draft.target_layers == (1, 13, 25, 37, 49)
    blocks = (tokens + BLOCK + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool = ContinuationPool.allocate(weights.config, shard, 3, blocks, selected, dflash=True)
    requests = DeviceRequestState.allocate(selected, vocab_size=c["vocab_size"], capacity=1)
    inputs = BatchInputs.allocate(BLOCK, blocks, selected, capacity=1)
    completion = CompletionOutputs.allocate((2, 1, VECTOR_TOKENS_FIELD + BLOCK), device=selected)
    frame = torch.zeros(1, dtype=torch.int32, device=selected)
    sequence = torch.zeros(1, dtype=torch.int64, device=selected)
    cursor = lambda: GraphBufferPool(selected, chunk_bytes=2 << 20).cursor()
    with PreparationSession(device=selected, autotune=False) as session:
        arena = ScratchArena(selected)
        decoder, metadata = prefill(weights, pool, ids, selected, blocks=blocks, session=session, workspace=arena,
                                    name="glimmer.dflash.prefill")
        context = PreparedContext(weights, decoder.auxiliary, metadata, pool, session=session,
                                  name="glimmer.dflash.prefill.context", workspace=arena, runtime=None,
                                  coordinator=None, buffers=cursor())
        owner = SpeculativePreparation(weights, pool, requests, inputs, Scratch.allocate(weights.config, shard, BLOCK, BLOCK, selected),
            completion=completion, frame=frame, sequence=sequence, session=session, workspace=arena,
            buffers=cursor(), decoder_buffers=cursor(), context_buffers=cursor())
        operation = PreparedParallelDraftRound(owner, 1, BLOCK, "glimmer.dflash.speculative1x16", buffers=cursor(),
                                               target_token_capacity=BLOCK)
        operation.prepare()
        arena.finalize()
        pool.clear_for_preparation()
        decoder()
        context(decoder.auxiliary, metadata.lengths)
        torch.cuda.synchronize()
        pending = int(decoder.scratch.logits[0].argmax())
        taps = decoder.auxiliary[:tokens].clone()
        requests.admit(0, RequestAdmission(generation=1, seed=2718, max_new_tokens=64, top_k=20, top_p=.95), ids,
                       consumed_tokens=tokens)
        requests.pending_token[0] = pending
        inputs.live_rows.fill_(1)
        inputs.row_slot_ids.zero_()
        inputs.row_generations.fill_(1)
        inputs.request_state_rows[0].fill_(1)
        inputs.page_tables[0].copy_(torch.arange(blocks, dtype=torch.int32, device=selected))
        inputs.query_lengths.fill_(BLOCK)
        operation()
        torch.cuda.synchronize()
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        emitted = int(operation.operations[-1].emitted_count[0])
        assert 1 <= emitted <= BLOCK
        assert int(requests.consumed_tokens[0]) == tokens + emitted
        assert torch.equal(operation.draft_metadata.token_ids[1:], torch.full((BLOCK - 1,), draft.noise_token, device=selected))
        assert int(operation.draft_metadata.token_ids[0]) == pending
        proposed_hidden = operation.draft.scratch.hidden.clone()
        proposed_logits = operation.draft.scratch.logits.clone()
        ring = pool.reserved_region("draft0.key")[1]
        for position in range(tokens, tokens + emitted):
            assert ring[position % draft.attention.window].view(torch.uint8).count_nonzero(), position

    assistant = MuseGlimmerAssistantModel.from_pretrained(assistant_snapshot, dtype=torch.bfloat16,
                                                          attn_implementation="eager").to(selected).eval()
    block_ids = torch.full((BLOCK,), draft.noise_token, dtype=torch.int64, device=selected)
    block_ids[0] = pending
    noise = F.embedding(block_ids, weights.embedding)[None]
    positions = torch.arange(tokens + BLOCK, device=selected)[None]
    # transformers.masking_utils.sliding_window_bidirectional_overlay: |q - k| <= sliding_window.
    visible = (positions[0, tokens:, None] - positions[0, None, :]).abs() <= draft.text["sliding_window"]
    mask = torch.zeros((1, 1, BLOCK, tokens + BLOCK), dtype=torch.float32, device=selected).masked_fill(~visible, -torch.inf)
    expected_hidden = assistant(noise_embeds=noise, context_hidden_states=taps[None],
                                attention_mask={"sliding_attention": mask, "full_attention": mask},
                                position_ids=positions, past_key_values=Fp8History()).last_hidden_state[0]
    expected_logits = softcapped(expected_hidden, weights.head, c)
    report = {name: dict(hidden=relative_l2(proposed_hidden[span], expected_hidden[span]),
                         logits=relative_l2(proposed_logits[span], expected_logits[span]),
                         top1=float((proposed_logits[span].argmax(-1) == expected_logits[span].argmax(-1)).float().mean()))
              for name, span in (("anchor", slice(0, 1)), ("proposals", slice(1, BLOCK)))}
    print(dict(tokens=tokens, pending=pending, emitted=emitted, width=BLOCK, relative_l2_bound=RELATIVE_L2, **report), flush=True)
    for name, errors in report.items():
        assert errors["hidden"] <= RELATIVE_L2 and errors["logits"] <= RELATIVE_L2, (name, errors)
