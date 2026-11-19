# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen DSpark draft history and proposal blocks on fabricated small-geometry weights.

The draft's context writer and proposal block run as captured CUDA graphs over FP8 draft pages.
Stored history must depend only on the accepted tokens, and a proposal block must not change it.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")

HIDDEN, INTERMEDIATE, VOCABULARY, LAYERS, TAPS, RANK = 512, 1024, 1024, 2, 2, 64
HEADS, KV_HEADS, HEAD = 4, 1, 128
PAGES, BLOCKS, BLOCK = 8, 4, 7
POISON = 0x55


def small_config():
    from ignition.recipes.qwen38_dense.dspark_config import DSparkConfig

    draft = DSparkConfig(dict(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, vocab_size=VOCABULARY,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD,
        target_layer_ids=list(range(TAPS)), mask_token_id=VOCABULARY - 1, block_size=BLOCK, markov_rank=RANK,
        rms_norm_eps=1e-6, rope_parameters=dict(rope_type="yarn", factor=32.0,
            original_max_position_embeddings=8192, beta_fast=32.0, beta_slow=1.0, rope_theta=10000000)))
    return SimpleNamespace(hidden=HIDDEN, vocabulary=VOCABULARY, text=dict(hidden_size=HIDDEN), draft=draft)


@pytest.fixture(scope="module")
def weights():
    from ignition.components.dflash2_weights import DraftLayerWeights, DraftWeights
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import MLPWeights

    generator = torch.Generator(device="cuda").manual_seed(8191)

    def matrix(rows, columns):
        return (torch.randn((rows, columns), device="cuda", generator=generator) * columns ** -.5).bfloat16()

    def scale(width):
        return (1 + .1 * torch.randn(width, device="cuda", generator=generator)).bfloat16()

    q, kv = HEADS * HEAD, KV_HEADS * HEAD
    layers = tuple(DraftLayerWeights(matrix(q + 2 * kv, HIDDEN), matrix(HIDDEN, q), scale(HEAD), scale(HEAD),
        scale(HIDDEN), scale(HIDDEN), MLPWeights(matrix(2 * INTERMEDIATE, HIDDEN), matrix(HIDDEN, INTERMEDIATE)),
        None, None) for _ in range(LAYERS))
    draft = DraftWeights(matrix(HIDDEN, TAPS * HIDDEN), scale(HIDDEN),
        torch.cat([layer.qkv[q:] for layer in layers]).contiguous(), scale(HIDDEN), layers, None,
        markov_embedding=matrix(VOCABULARY, RANK), markov_head=matrix(VOCABULARY, RANK))
    return SimpleNamespace(config=small_config(), shard=TensorParallelShard(0, 1), draft=draft,
        embedding=matrix(VOCABULARY, HIDDEN), head=matrix(VOCABULARY, HIDDEN))


class Pages:
    """Draft page regions as the dense pool exposes them, filled with a byte no writer produces by default."""

    def __init__(self):
        from ignition.recipes.qwen38_dense.dspark_config import PAGE_DTYPE

        self.dtype = PAGE_DTYPE
        self.regions = [tuple(torch.full((PAGES, 128, KV_HEADS, HEAD), POISON, dtype=torch.uint8, device="cuda")
                              for _ in range(2)) for _ in range(LAYERS)]

    def draft_attention(self, layer, config):
        return tuple(region.view(self.dtype) for region in self.regions[layer])

    def bytes(self):
        return torch.stack([torch.stack(pair) for pair in self.regions]).clone()

    def rows(self, positions):
        """Stored bytes of the given token positions under the identity page table, per layer and role."""
        flat = self.bytes().view(LAYERS, 2, PAGES * 128, KV_HEADS, HEAD)
        return flat[:, :, positions]


def metadata(tokens, rows=1, samples=None):
    from ignition.components.paged_batch import Metadata

    return Metadata.allocate(tokens, rows, BLOCKS, "cuda", samples=samples)


def place(batch, starts, lengths, token_ids=None):
    """Bind requests to consecutive page-table rows: request ``r`` owns pages ``r * BLOCKS`` onward."""
    rows = len(starts)
    width = max(max(lengths), 1)
    tokens = torch.zeros((rows, width), dtype=torch.int64, device="cuda") if token_ids is None else token_ids
    pages = torch.arange(rows * BLOCKS, dtype=torch.int32, device="cuda").view(rows, BLOCKS)
    batch.set(torch.tensor(starts, device="cuda"), torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        torch.arange(1, rows + 1, device="cuda"), pages, tokens, torch.tensor([rows], dtype=torch.int32, device="cuda"))


def replay(graph):
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
        assert before[name] == after[name], name


class Captured:
    """One prepared forward compiled and captured as a CUDA graph."""

    def __init__(self, session, body):
        from ignition.preparation.compilation import compile_forward

        forward = compile_forward(body)
        forward()
        self.graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(self.graph):
            forward()

    def __call__(self):
        replay(self.graph)


class Harness:
    """Context writers for a prefill chunk, an eight-token round and a one-token round, and a proposal block."""

    def __init__(self, weights, session, arena, *, chunk):
        from ignition.components.dflash2 import DraftScratch
        from ignition.recipes.kimi3.dspark import PreparedMarkov
        from ignition.recipes.qwen38_dense.dspark import PreparedDraft, prepared_context

        self.pages = Pages()
        self.contexts = {}
        for name, tokens in (("chunk", chunk), ("round8", 8), ("round1", 1)):
            source = torch.zeros((tokens, TAPS * HIDDEN), dtype=torch.bfloat16, device="cuda")
            batch, counts = metadata(tokens), torch.zeros(1, dtype=torch.int32, device="cuda")
            context = prepared_context(weights, source, batch, self.pages, session=session,
                                      name=f"dspark{chunk}.{name}.context", workspace=arena)
            self.contexts[name] = SimpleNamespace(source=source, batch=batch, counts=counts, context=context)
        self.block = metadata(2 * BLOCK, rows=2, samples=2 * BLOCK)
        scratch = DraftScratch.allocate(weights.config.draft, weights.shard, 2 * BLOCK, "cuda", block_size=BLOCK)
        draft = PreparedDraft(weights, scratch, self.block, self.pages, embedding=weights.embedding,
            head=weights.head, session=session, name=f"dspark{chunk}.draft", workspace=arena)
        assert draft.logit_shape == (2 * BLOCK, VOCABULARY) and draft.logit_dtype == torch.float32
        self.previous = torch.zeros(2, dtype=torch.int64, device="cuda")
        markov = PreparedMarkov(weights, self.previous, runtime=None, coordinator=None, session=session,
                                name=f"dspark{chunk}.markov")
        self.logits = torch.zeros((2 * BLOCK, VOCABULARY), dtype=torch.float32, device="cuda")
        self.bias = torch.zeros((2, VOCABULARY), dtype=torch.float32, device="cuda")
        arena.finalize()
        for entry in self.contexts.values():
            entry.graph = Captured(session, lambda entry=entry: entry.context(entry.source, entry.counts, entry.batch))
        self.draft = Captured(session, lambda: self.logits.copy_(draft(self.block)[0]))
        self.markov = Captured(session, lambda: self.bias.copy_(markov(self.previous)))

    def write(self, name, features, start, accepted=None):
        """Offer ``features`` for the positions from ``start`` and accept the first ``accepted`` of them."""
        entry = self.contexts[name]
        count = features.shape[0]
        entry.source.zero_()
        entry.source[:count].copy_(features)
        place(entry.batch, [start], [count])
        entry.counts.fill_(count if accepted is None else accepted)
        entry.graph()

    def propose(self, start, pending, *, second=None):
        """Draft logits of a block at ``start``; a second request is inactive unless given its own start."""
        tokens = torch.full((2, BLOCK), VOCABULARY - 1, dtype=torch.int64, device="cuda")
        tokens[:, 0] = pending
        place(self.block, [start, 0 if second is None else second], [BLOCK, 0 if second is None else BLOCK], tokens)
        self.logits.fill_(float("nan"))
        self.draft()
        return self.logits.clone()


@pytest.fixture(scope="module")
def prepared(weights):
    from b12x.preparation import PreparationSession
    from ignition.preparation.scratch import ScratchArena

    with torch.no_grad(), PreparationSession(device="cuda:0", autotune=False) as session:
        yield (Harness(weights, session, ScratchArena("cuda"), chunk=128),
               Harness(weights, session, ScratchArena("cuda"), chunk=384))


def features(count, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((count, TAPS * HIDDEN), device="cuda", generator=generator).bfloat16()


@torch.no_grad()
def test_chunked_prefill_stores_the_history_of_one_prefill(prepared):
    chunked, single = prepared
    prompt = features(300, 17)
    single.write("chunk", prompt, 0)
    for start in range(0, 300, 128):
        chunked.write("chunk", prompt[start:start + 128], start)
    positions = torch.arange(300, device="cuda")
    assert torch.equal(chunked.pages.bytes(), single.pages.bytes())
    assert not single.pages.rows(positions).eq(POISON).all(-1).all(-1).any()
    assert single.pages.bytes().view(LAYERS, 2, -1, KV_HEADS, HEAD)[:, :, 300:].eq(POISON).all()
    assert single.pages.dtype == torch.float8_e4m3fn


@torch.no_grad()
def test_a_rejecting_round_stores_only_accepted_positions_and_the_next_block_matches(prepared, weights):
    rejecting, direct = prepared
    prompt, offered = features(300, 17), features(8, 29)
    for harness in prepared:
        for region in (tensor for pair in harness.pages.regions for tensor in pair):
            region.fill_(POISON)
    for start in range(0, 300, 128):
        rejecting.write("chunk", prompt[start:start + 128], start)
    # The round offers eight positions and accepts three; a one-token round then accepts one more.
    rejecting.write("round8", offered, 300, accepted=3)
    stored = rejecting.pages.bytes().view(LAYERS, 2, -1, KV_HEADS, HEAD)
    assert stored[:, :, 303:].eq(POISON).all()
    assert not stored[:, :, 300:303].eq(POISON).all(-1).all(-1).any()
    rejecting.write("round1", offered[3:4], 303)
    direct.write("chunk", torch.cat((prompt, offered[:4])), 0)
    assert torch.equal(rejecting.pages.bytes(), direct.pages.bytes())

    before = rejecting.pages.bytes()
    pending = 41
    after_rejection, uninterrupted = rejecting.propose(304, pending), direct.propose(304, pending)
    assert torch.isfinite(after_rejection[:BLOCK]).all()
    assert torch.equal(after_rejection[:BLOCK], uninterrupted[:BLOCK])
    assert torch.equal(rejecting.pages.bytes(), before)
    # A second active request with its own pages leaves the first request's block unchanged.
    paired = rejecting.propose(304, pending, second=0)
    assert torch.equal(paired[:BLOCK], after_rejection[:BLOCK])
    assert torch.equal(rejecting.pages.bytes(), before)
    # History reaches the block: other accepted features change the proposals.
    direct.write("chunk", torch.cat((prompt, features(4, 31))), 0)
    assert not torch.equal(direct.propose(304, pending)[:BLOCK], uninterrupted[:BLOCK])

    rejecting.previous.copy_(torch.tensor([pending, 7], device="cuda"))
    rejecting.markov()
    expected = weights.draft.markov_embedding[rejecting.previous].float() @ weights.draft.markov_head.float().T
    assert torch.allclose(rejecting.bias, expected, rtol=2e-2, atol=2e-2)
