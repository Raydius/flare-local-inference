# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU dataflow checks for Muse Glimmer forwards that take packed metadata and return local values.

Projections, cache writers and the paged attention kernel are replaced by recording callables, so
these cases cover the Python dataflow around them: which metadata reaches each operator, what the
forward returns, and what a verification commit writes. Kernel arithmetic is covered by the CUDA
tests in this directory.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.muse_glimmer_nvfp4.decoder import PreparedDecoder
from ignition.recipes.muse_glimmer_nvfp4.attention import PAGE_TOKENS, PreparedAttention

HEADS, KV_HEADS, DIM = 2, 1, 4
WIDTHS = (HEADS * DIM, KV_HEADS * DIM, KV_HEADS * DIM, HEADS * DIM)


def pack(starts, lengths, states, *, width, pages=None):
    rows = len(starts)
    pages = torch.arange(rows * 4, dtype=torch.int32).view(rows, 4) + 10 if pages is None else pages
    return Metadata.pack(torch.tensor(starts), torch.tensor(lengths, dtype=torch.int32), torch.tensor(states),
        pages, torch.zeros((rows, width), dtype=torch.int64), torch.tensor([rows], dtype=torch.int32),
        token_capacity=rows * width, sample_count=rows * width, max_query_tokens=width)


def attention_fixture(window, mode, tokens, *, states=4):
    attention = PreparedAttention.__new__(PreparedAttention)
    torch.nn.Module.__init__(attention)
    attention.geometry = SimpleNamespace(window=window, rotary_dim=0)
    attention.mode = mode
    attention.query_shape = (tokens, HEADS, DIM)
    attention.key_shape = attention.value_shape = (tokens, KV_HEADS, DIM)
    attention.widths, attention.inverse = WIDTHS, None
    attention.eps, attention.query_factor = 1e-5, 1.
    generator = torch.Generator().manual_seed(4127)
    weight = torch.randn((sum(WIDTHS), HEADS * DIM), generator=generator).bfloat16()
    attention.projection = lambda rows: rows @ weight[:sum(WIDTHS[:3])].T
    attention.gate_projection = lambda rows: rows @ weight[sum(WIDTHS[:3]):].T
    attention.output_projection = attention.reduction = lambda rows: rows
    calls = SimpleNamespace(written=[], committed=[], attended=[])
    attention.writer = lambda key, value, slots: calls.written.append((key, value, slots))
    attention.commit_writer = None
    if window > 0:
        history = [torch.randn((states, window, KV_HEADS, DIM), generator=generator).bfloat16() for _ in range(2)]
        attention.key_cache, attention.value_cache = history
        if mode == "decode":
            attention.window_row_stride = 5
            attention.window_columns = torch.arange(window // PAGE_TOKENS, dtype=torch.int32)
        else:
            attention.writer = None
    if mode == "verify":
        attention.commit_writer = lambda key, value, slots: calls.committed.append((key, value, slots))

    def attend(query, **operands):
        calls.attended.append(operands)
        return query

    attention.attention = attend
    source = torch.randn((tokens, HEADS * DIM), generator=generator).bfloat16()
    return attention, calls, source


def test_global_attention_writes_and_reads_through_the_forward_metadata():
    attention, calls, source = attention_fixture(-1, "decode", 2)
    for starts in ([127, 300], [128, 5]):
        m = pack(starts, [1, 1], [1, 2], width=1)
        output, _ = attention(source, metadata=m)
        assert output.shape == source.shape
        _, _, slots = calls.written[-1]
        expected = [int(m.page_table[row, start // PAGE_TOKENS]) * PAGE_TOKENS + start % PAGE_TOKENS
                    for row, start in enumerate(starts)]
        assert slots.tolist() == expected
        operands = calls.attended[-1]
        assert operands["page_table"] is m.page_table and operands["cu_seqlens_q"] is m.query_start_loc
        assert operands["cache_seqlens"].tolist() == [start + 1 for start in starts]
        assert operands["cache_seqlens"].dtype == torch.int32


def test_global_attention_leaves_inactive_rows_unwritten():
    attention, calls, source = attention_fixture(-1, "decode", 2)
    attention(source, metadata=pack([127, 300], [1, 0], [1, 2], width=1))
    assert calls.written[-1][2].tolist()[1] == -1


def test_sliding_decode_addresses_the_circular_row_of_each_request():
    window = 2 * PAGE_TOKENS
    attention, calls, source = attention_fixture(window, "decode", 2)
    m = pack([3, 700], [1, 1], [1, 3], width=1)
    attention(source, metadata=m)
    operands, (_, _, slots) = calls.attended[-1], calls.written[-1]
    assert operands["page_table"].dtype == torch.int32
    assert operands["page_table"].tolist() == [[5, 6], [15, 16]]
    assert operands["cache_seqlens"].tolist() == [4, window]
    assert slots.tolist() == [1 * 5 * PAGE_TOKENS + 3, 3 * 5 * PAGE_TOKENS + 700 % window]


def test_sliding_prefill_stages_history_before_candidates_and_retains_the_window():
    window, tokens = 2 * PAGE_TOKENS, 4
    attention, calls, source = attention_fixture(window, "prefill", tokens)
    pages = (tokens + window + PAGE_TOKENS - 1) // PAGE_TOKENS
    attention.staged_shapes = ((pages, PAGE_TOKENS, KV_HEADS, DIM),) * 2
    attention.pages = torch.arange(pages, dtype=torch.int32).view(1, -1)
    before = attention.key_cache.clone()
    records = []
    m = pack([3], [tokens], [2], width=tokens)
    attention(source, metadata=m, records=records)
    (key, value), = records
    operands = calls.attended[-1]
    staged = operands["k_cache"].view(-1, KV_HEADS, DIM).bfloat16()
    fp8 = lambda rows: rows.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
    assert torch.equal(staged[:3], fp8(before[2, :3]))
    assert torch.equal(staged[3:3 + tokens], fp8(key))
    assert staged[3 + tokens:].count_nonzero() == 0
    assert "page_table" not in operands and operands["cache_seqlens"].tolist() == [3 + tokens]
    assert torch.equal(attention.key_cache[2, 3:3 + tokens], key)
    assert torch.equal(attention.value_cache[2, 3:3 + tokens], value)
    assert torch.equal(attention.key_cache[2, :3], before[2, :3])
    assert torch.equal(attention.key_cache[1], before[1])


def test_verification_commits_only_the_accepted_prefix_of_each_request():
    window, width = 2 * PAGE_TOKENS, 4
    attention, calls, source = attention_fixture(window, "verify", 2 * width)
    pages = (window + width + PAGE_TOKENS - 1) // PAGE_TOKENS
    attention.staged_shapes = ((2 * pages, PAGE_TOKENS, KV_HEADS, DIM),) * 2
    attention.pages = torch.arange(2 * pages, dtype=torch.int32).view(2, -1)
    before = attention.key_cache.clone()
    records = []
    m = pack([10, 600], [width, width], [1, 3], width=width)
    attention(source, metadata=m, records=records)
    assert torch.equal(attention.key_cache, before) and not calls.committed
    attention.commit(torch.tensor([3, 1]), records[0], m)
    key, value, slots = calls.committed[-1]
    assert key is records[0][0] and value is records[0][1]
    assert slots.tolist() == [1 * window + 10, 1 * window + 11, 1 * window + 12, -1,
                              3 * window + 600 % window, -1, -1, -1]


def decoder_fixture(*, target_layers=(), verify=False, vision_state=None, scoring=None):
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.embedding = lambda *, token_ids, num_rows: token_ids[:, None].expand(-1, 8).to(torch.bfloat16)
    decoder.embedding_norm = lambda hidden: hidden * .5
    commits = []

    def layer(step):
        def forward(hidden, *, metadata, records):
            if records is not None:
                records.append(step)
            return hidden + step
        forward.self_attn = SimpleNamespace(commit=lambda counts, record, metadata: commits.append((record, counts)))
        return forward

    decoder.layers, decoder.target_layers, decoder.verify = [layer(1), layer(2), layer(3)], target_layers, verify
    decoder.norm = lambda hidden: hidden * 2
    decoder.scoring, decoder.vision_state = scoring, vision_state
    decoder.head = lambda hidden: hidden.clone()
    decoder.output_multiplier, decoder.softcap = 3., 30.
    return decoder, commits


def softcapped(logits):
    return torch.tanh(logits * 3. / 30.) * 30.


def prefill_metadata(ids):
    ids = torch.tensor(ids)
    return Metadata.pack(torch.tensor([0]), torch.tensor([ids.numel()], dtype=torch.int32), torch.tensor([1]),
        torch.zeros((1, 1), dtype=torch.int32), ids[None], torch.tensor([1], dtype=torch.int32),
        token_capacity=ids.numel(), sample_count=1)


def test_decoder_returns_local_outputs_for_the_forward_token_ids():
    decoder, _ = decoder_fixture()
    for ids in ([1, 2], [3, 5]):
        hidden, logits, auxiliary, records = decoder(prefill_metadata(ids))
        expected = (torch.tensor(ids[:1]).to(torch.bfloat16)[:, None].expand(-1, 8) * .5 + 6) * 2
        assert torch.equal(hidden, expected) and torch.equal(logits, softcapped(expected))
        assert auxiliary is None and records is None


def test_decoder_concatenates_draft_taps_in_target_layer_order():
    decoder, _ = decoder_fixture(target_layers=(2, 0))
    _, _, auxiliary, _ = decoder(prefill_metadata([4, 4]))
    assert auxiliary.shape == (2, 16)
    assert torch.equal(auxiliary[:, :8], torch.full((2, 8), 8.)) and torch.equal(auxiliary[:, 8:], torch.full((2, 8), 3.))


def test_verification_records_reach_each_layer_commit():
    decoder, commits = decoder_fixture(verify=True)
    m = prefill_metadata([4, 4])
    _, _, _, records = decoder(m)
    assert records == [1, 2, 3]
    counts = torch.tensor([1])
    decoder.commit(counts, records, m)
    assert [record for record, _ in commits] == [1, 2, 3] and all(seen is counts for _, seen in commits)


def test_scored_prefill_softcaps_prompt_logits_like_completion_logits():
    scored = []
    scoring = SimpleNamespace(parts=((0, 2, lambda source: source.clone(), scored.append),))
    decoder, _ = decoder_fixture(scoring=scoring)
    hidden, logits, _, _ = decoder(prefill_metadata([1, 2]))
    normalized = (torch.tensor([1, 2]).to(torch.bfloat16)[:, None].expand(-1, 8) * .5 + 6) * 2
    assert torch.equal(scored[0], softcapped(normalized))
    assert torch.equal(hidden, normalized[:1]) and torch.equal(logits, softcapped(normalized[:1]))


def test_prefill_decoder_injects_image_features_after_the_embedding_norm():
    class Vision:
        calls = []

        def inject(self, metadata, embedding):
            self.calls.append((metadata, embedding.clone()))
            embedding[0].fill_(7.)

    vision = Vision()
    decoder, _ = decoder_fixture(vision_state=vision)
    m = prefill_metadata([3, 3])
    hidden, _, _, _ = decoder(m)
    (metadata, normed), = vision.calls
    assert metadata is m and torch.equal(normed, torch.full((2, 8), 1.5))
    assert torch.equal(hidden[0], torch.full((8,), (7. + 6) * 2))


def test_text_only_decoder_skips_the_vision_hook():
    decoder, _ = decoder_fixture()
    hidden, _, _, _ = decoder(prefill_metadata([3, 3]))
    assert torch.equal(hidden, torch.full((1, 8), (1.5 + 6) * 2))


def test_decoder_constructor_keeps_the_vision_hook_optional():
    import inspect

    assert inspect.signature(PreparedDecoder.__init__).parameters["vision_state"].default is None
