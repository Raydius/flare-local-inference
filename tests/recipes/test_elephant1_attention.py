# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 block attention: segment bounds on the CPU and the captured layer against a reference."""

import os

import pytest
import torch

from ignition.recipes.elephant1.attention import SEGMENTS, BlockAttention, block_segments, key_slots
from ignition.recipes.elephant1.config import PAGE_TOKENS


def _distances(capacity, start, spans):
    """Per-token distances as Helios packs them for `capacity` tokens starting at `start`."""
    left, right = (torch.zeros(capacity, dtype=torch.int64) for _ in range(2))
    for first, end in spans:
        for position in range(first, end):
            left[position - start], right[position - start] = position - first, end - position - 1
    return left, right


@pytest.mark.parametrize("start,count,spans,queries,keys", [
    (0, 250, ((0, 173), (173, 250)), [0, 173, 250], [0, 173, 423]),   # both blocks in one submission
    (0, 173, ((0, 173),), [0, 173, 173], [0, 173, 173]),               # the prefix block alone
    (173, 77, ((173, 250),), [0, 77, 77], [0, 250, 250]),              # a suffix resumed at the prefix end
    (100, 60, (), [0, 0, 0], [0, 0, 0]),                               # a causal submission
    (0, 2, ((0, 1), (1, 2)), [0, 0, 0], [0, 0, 0]),                    # one-token blocks are causal
])
def test_block_segments_bound_each_block_by_its_end(start, count, spans, queries, keys):
    left, right = _distances(256, start, spans)
    blocks, actual_queries, actual_keys = block_segments(torch.tensor(start), torch.tensor(count, dtype=torch.int32),
                                                         left, right)
    assert bool(blocks) == any(end - first > 1 for first, end in spans)
    assert actual_queries.tolist() == queries
    assert actual_keys.tolist() == keys


def test_key_slots_pack_both_segments_from_their_pages():
    pages = torch.tensor([5, 2, -1, -1], dtype=torch.int32)
    slots = key_slots(torch.tensor([0, 130, 330], dtype=torch.int32), pages, 400)
    position = torch.cat((torch.arange(130), torch.arange(200)))
    expected = torch.where(position < PAGE_TOKENS, 5 * PAGE_TOKENS + position, 2 * PAGE_TOKENS + position % PAGE_TOKENS)
    assert torch.equal(slots[:330], expected)
    # Rows past both segments name a valid slot and are never read.
    assert torch.equal(slots[330:], torch.full((70,), 5 * PAGE_TOKENS))
    assert (key_slots(torch.zeros(3, dtype=torch.int32), torch.full((4,), -1, dtype=torch.int32), 8) == 0).all()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")
@torch.no_grad()
def test_captured_block_attention_matches_the_block_mask():
    from ignition.components.paged_attention import PreparedPagedAttention
    from ignition.components.paged_batch import Metadata
    from ignition.components.varlen_attention import PreparedVarlenAttention
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    torch.manual_seed(31)
    tokens, context, heads, kv_heads, dim = 256, 512, 16, 4, 256
    blocks = context // PAGE_TOKENS
    key_cache, value_cache = (torch.randn(8, PAGE_TOKENS, kv_heads, dim, device=device).bfloat16() for _ in range(2))
    page_row = torch.tensor([5, 2, 7, 0], device=device)
    slots = (page_row[:, None] * PAGE_TOKENS + torch.arange(PAGE_TOKENS, device=device)).flatten()
    metadata = Metadata.allocate(tokens, 1, blocks, device)
    metadata.page_table.copy_(page_row[None])
    query = torch.randn(tokens, heads, dim, device=device).bfloat16()
    gate = torch.randn(tokens, heads, dim, device=device).bfloat16()
    left, right = (torch.zeros(tokens, dtype=torch.int64, device=device) for _ in range(2))
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    attended, observed = torch.zeros_like(query), torch.zeros_like(query)

    def submit(start, count, spans):
        distances = _distances(tokens, start, spans)
        left.copy_(distances[0])
        right.copy_(distances[1])
        metadata.starts.fill_(start)
        metadata.lengths.fill_(count)
        metadata.query_start_loc.copy_(torch.tensor([0, count]))
        lengths.fill_(start + count)

    def expected(start, count, spans):
        keys, values = (cache.flatten(0, 1)[slots].float() for cache in (key_cache, value_cache))
        result = torch.zeros(count, heads, dim, device=device)
        for row in range(count):
            position = start + row
            visible = next((end for first, end in spans if first <= position < end), position + 1)
            for group in range(kv_heads):
                head = slice(group * heads // kv_heads, (group + 1) * heads // kv_heads)
                scores = query[row, head].float() @ keys[:visible, group].T / dim ** .5
                result[row, head] = scores.softmax(-1) @ values[:visible, group]
        return result * gate[:count].float().sigmoid()

    with RecordedPreparationSession(device=device, checkpoint="elephant1-block", workload_schema="elephant1-block",
                                    numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        layer = object.__new__(BlockAttention)
        torch.nn.Module.__init__(layer)
        layer.tokens, layer.heads, layer.kv_heads, layer.dim, layer.context = tokens, heads, kv_heads, dim, context
        layer.left, layer.right, layer.key_cache, layer.value_cache = left, right, key_cache, value_cache
        layer.attention = PreparedPagedAttention(
            query, key_cache, value_cache, metadata.page_table, lengths, metadata.query_start_loc, attended,
            output_gate=gate, tuning_page_count=8, mode="prefill", width=1, session=session, workspace=arena,
            name="causal", caller_owned_placeholders=True,
        ).functional(session=session, cache_seqlens=None, page_table=None, cu_seqlens_q=None)
        segment_lengths = torch.zeros(SEGMENTS + 1, dtype=torch.int32, device=device)
        layer.block = PreparedVarlenAttention(
            query, *(torch.zeros((SEGMENTS * context, kv_heads, dim), dtype=torch.bfloat16, device=device)
                     for _ in range(2)),
            segment_lengths, attended, cu_k=torch.zeros_like(segment_lengths), max_seqlen_k=context,
            session=session, workspace=arena, name="block").functional(session=session)

        def forward():
            observed.copy_(layer.attend(query, gate, lengths, metadata))

        submit(0, 250, ((0, 173), (173, 250)))
        session.tune()
        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            cases = (
                (0, 250, ((0, 173), (173, 250))),   # both blocks in one submission
                (173, 77, ((173, 250),)),           # a suffix resumed at a prefix end that is not page-aligned
                (0, 173, ((0, 173),)),              # the prefix block alone
                (100, 60, ()),                      # a causal submission
            )
            for start, count, spans in cases:
                submit(start, count, spans)
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                torch.testing.assert_close(observed[:count].float(), expected(start, count, spans), atol=.004, rtol=.02)
            # The prefix block's rows do not depend on the suffix: change the suffix's queries and keys.
            submit(0, 250, ((0, 173), (173, 250)))
            graph.replay()
            prefix = observed[:173].clone()
            suffix = slots[173:250]
            key_cache.flatten(0, 1)[suffix] = torch.randn(77, kv_heads, dim, device=device).bfloat16()
            value_cache.flatten(0, 1)[suffix] = torch.randn(77, kv_heads, dim, device=device).bfloat16()
            query[173:250].normal_()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(observed[:173], prefix, atol=0, rtol=0)
        finally:
            graph.reset()


def test_contract_declares_the_block_end_marker(monkeypatch):
    from google.protobuf import json_format

    from ignition.preparation import runtime_core_pb2 as p
    from ignition.recipes.elephant1 import contract
    from ignition.recipes.qwen38_dense.contract import FRONTEND

    seen = {}
    monkeypatch.setattr(contract, "_prepare", lambda *args, **kwargs: seen.update(kwargs))
    contract.prepare({}, {}, None, vocabulary=7, block_end_marker=(510, 2374, 29, 271))
    assert seen["frontend"]["block_end_marker"] == [510, 2374, 29, 271]
    assert seen["frontend"]["assistant_marker"] == FRONTEND["assistant_marker"]
    # The dense Qwen contract this one extends declares no marker, and the wire message carries one.
    assert FRONTEND["block_end_marker"] == []
    message = json_format.ParseDict({"block_end_marker": seen["frontend"]["block_end_marker"]}, p.FrontendContract())
    assert list(message.block_end_marker) == [510, 2374, 29, 271]
