# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen DSpark proposals over paged draft history.

Accepted target positions project their five captured hidden states once into every draft layer's
keys and values, stored in that layer's page regions at the token's page slot. A proposal block
holds the pending token followed by mask tokens; five noncausal grouped-query layers attend every
accepted position and the block itself, and block position ``i`` scores draft ``i + 1``. A low-rank
transition from the previously sampled token biases each position's logits while drafts are sampled
left to right. The draft embeds tokens and computes logits with the target's weights; its
confidence head is not executed.
"""

from functools import partial
from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half

from ignition.components.dflash2 import DraftScratch
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import prepare_projection
from ignition.components.mlp import PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.recipes.inkling2.dspark_state import context_slots, stage_paged_tail
from ignition.recipes.kimi3.dspark import PreparedMarkov, canonical_norm

from . import dflash
from .config import PAGE_TOKENS


def rotate(values, positions, inverse, gain):
    """Qwen3's rotate-half rotary embedding with the YaRN gain multiplied into the FP32 cosine and sine."""
    frequencies = positions.clamp_min(0).float()[:, None] * inverse[None]
    angles = torch.cat((frequencies, frequencies), -1)
    cos, sin = (angles.cos() * gain).to(values.dtype), (angles.sin() * gain).to(values.dtype)
    return values * cos[:, None] + rotate_half(values) * sin[:, None]


def tail_pages(block):
    """Private pages that hold a committed partial page followed by one proposal block."""
    return (PAGE_TOKENS - 1 + block + PAGE_TOKENS - 1) // PAGE_TOKENS


def tail_slots(metadata, pages):
    """Each block row's slot in its request's private tail pages, after the committed partial page."""
    m = metadata
    slot = m.requests * (pages * PAGE_TOKENS) + m.positions - m.starts[m.requests] // PAGE_TOKENS * PAGE_TOKENS
    return torch.where((m.positions >= 0) & (m.lengths[m.requests] > 0), slot, -1)


class PreparedContext:
    """Write accepted target features into every draft layer's key and value pages.

    The stored keys and values depend only on a token's position and target features, so one
    prefill, chunked prefill and verification rounds leave the same history for the same tokens.
    Positions a round does not accept receive no slot.
    """

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, examples=None):
        config, draft = weights.config, weights.draft
        c, a = config.draft, config.draft.attention
        eps, device, tokens = c.text["rms_norm_eps"], auxiliary.device, auxiliary.shape[0]
        self.inverse, self.gain = persistent_copy(c.frequencies(device)), c.rotary_gain
        self.key_shape = (tokens, a.kv_heads, a.key_dim)
        self.value_shape = (tokens, a.kv_heads, a.value_dim)

        def empty(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "dspark.context." + role, shape, dtype, device)

        projected = empty("projected", (tokens, config.hidden))
        self.project = prepare_projection(auxiliary, draft.projection, projected, session=session,
            name=name + ".project", workspace=workspace).functional(session=session)
        self.norm = canonical_norm(draft.hidden_norm, eps)
        kv = empty("kv", (tokens, c.layers * a.kv_heads * (a.key_dim + a.value_dim)))
        self.project_kv = prepare_projection(projected, draft.context_kv, kv, session=session,
            name=name + ".kv", workspace=workspace).functional(session=session)
        self.layers = []
        for index, layer in enumerate(draft.layers):
            slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
            writer = PreparedPagedKVWriter(empty("key", self.key_shape), empty("value", self.value_shape),
                *pool.draft_attention(index, config), slots, session=session, name=f"{name}.{index}.write",
                caller_owned_placeholders=True).functional()
            self.layers.append((canonical_norm(layer.key_norm, eps), writer))

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        slots = context_slots(m, counts, page_tokens=PAGE_TOKENS)
        kv = self.project_kv(self.norm(self.project(auxiliary)))
        width = self.key_shape[1] * self.key_shape[2]
        chunks = kv.view(kv.shape[0], len(self.layers), -1)
        for index, (norm, writer) in enumerate(self.layers):
            key = rotate(norm(chunks[:, index, :width].view(self.key_shape)), m.positions, self.inverse, self.gain)
            writer(key, chunks[:, index, width:].view(self.value_shape).contiguous(), slots)

    def scratch_tensors(self):
        return ()


class DraftAttention:
    """Noncausal grouped-query attention over a request's draft pages and its proposal block.

    The block's keys and values exist only for the round: they are written, in the page dtype,
    into private tail pages after a copy of the request's committed partial page.
    """

    def __init__(self, index, weights, source, config, scratch, metadata, pool, *, session, name, workspace,
                 examples=None):
        c, a = config.draft, config.draft.attention
        eps, device, tokens = c.text["rms_norm_eps"], source.device, source.shape[0]
        rows, block = metadata.lengths.numel(), scratch.block_size
        self.query_shape = (tokens, a.heads, a.key_dim)
        self.key_shape, self.value_shape = (tokens, a.kv_heads, a.key_dim), (tokens, a.kv_heads, a.value_dim)

        def empty(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "dspark.draft." + role, shape, dtype, device)

        self.qkv = prepare_projection(source, weights.qkv, empty("qkv", (tokens, weights.qkv.shape[0])),
            session=session, name=name + ".qkv", workspace=workspace).functional(session=session)
        self.q_norm, self.k_norm = canonical_norm(weights.query_norm, eps), canonical_norm(weights.key_norm, eps)
        self.inverse, self.gain = persistent_copy(c.frequencies(device)), c.rotary_gain
        self.key_pages, self.value_pages = pool.draft_attention(index, config)
        self.tail_pages = tail_pages(block)
        tail = {role: torch.empty((rows * self.tail_pages, *pages.shape[1:]), dtype=pages.dtype, device=device)
                for role, pages in (("tail_key", self.key_pages), ("tail_value", self.value_pages))}
        self.stage = PreparedPagedKVWriter(empty("key", self.key_shape), empty("value", self.value_shape),
            tail["tail_key"], tail["tail_value"], torch.full((tokens,), -1, dtype=torch.int64, device=device),
            session=session, name=name + ".stage", caller_owned_placeholders=True).functional(
                k_cache=None, v_cache=None)
        attended = empty("attended", (tokens, a.heads, a.value_dim))
        self.attention = PreparedPagedAttention(empty("query", self.query_shape), self.key_pages, self.value_pages,
            metadata.page_table, metadata.lengths, metadata.query_start_loc, attended, mode="verify",
            width=block, window=-1, causal=False, session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True, **tail).functional(session=session, k_cache=None, v_cache=None,
                cache_seqlens=None, cu_seqlens_q=None, page_table=None, tail_key=None, tail_value=None)
        self.output = prepare_projection(attended.flatten(1), weights.output, empty("output", (tokens, config.hidden)),
            session=session, name=name + ".output", workspace=workspace).functional(session=session)

    def stage_tail(self, key, value, metadata):
        """Copy each request's committed partial page into its tail pages, then write the block after it."""
        m = metadata
        staged = []
        for pages, rows in ((self.key_pages, key), (self.value_pages, value)):
            vacant = torch.zeros(rows.shape, dtype=torch.uint8, device=rows.device)
            staged.append(stage_paged_tail(pages.view(torch.uint8), vacant, m, self.tail_pages).view(pages.dtype))
        self.stage(key, value, tail_slots(m, self.tail_pages), k_cache=staged[0], v_cache=staged[1])
        return staged

    def __call__(self, source, metadata):
        m = metadata
        q, k, v = self.qkv(source).split(tuple(shape[1] * shape[2] for shape in
                                                (self.query_shape, self.key_shape, self.value_shape)), -1)
        query = rotate(self.q_norm(q.view(self.query_shape)), m.positions, self.inverse, self.gain)
        key = rotate(self.k_norm(k.view(self.key_shape)), m.positions, self.inverse, self.gain)
        tail_key, tail_value = self.stage_tail(key, v.view(self.value_shape).contiguous(), m)
        lengths = torch.where(m.lengths > 0, m.starts.int() + m.lengths, 0)
        attended = self.attention(query, k_cache=self.key_pages, v_cache=self.value_pages, cache_seqlens=lengths,
            cu_seqlens_q=m.query_start_loc, page_table=m.page_table, tail_key=tail_key, tail_value=tail_value)
        return self.output(attended.flatten(1))


class PreparedDraft:
    """Score every position of a proposal block with the target's embedding and vocabulary head."""

    def __init__(self, weights, scratch, metadata, pool, *, embedding, head, session, name, workspace,
                 runtime=None, vocabulary_runtime=None, coordinator=None, examples=None):
        config, draft = weights.config, weights.draft
        eps, device, tokens = config.draft.text["rms_norm_eps"], embedding.device, scratch.tokens
        self.logit_shape, self.logit_dtype = scratch.logit_shape, torch.float32
        self.vocabulary = scratch.vocabulary_size

        def empty(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "dspark.draft." + role, shape, dtype, device)

        hidden = empty("hidden", (tokens, config.hidden))
        self.embedding = PreparedShardedEmbedding(embedding, metadata.token_ids,
            empty("local_embedding", (tokens, config.hidden)), hidden, runtime=None, coordinator=coordinator,
            session=session, name=name + ".embedding", num_rows=metadata.live_counts[:1]).functional(
                session=session, dynamic_inputs=True)
        intermediate = config.draft.text["intermediate_size"]
        self.layers = []
        for index, layer in enumerate(draft.layers):
            prefix = f"{name}.layer{index}"
            attention = DraftAttention(index, layer, hidden, config, scratch, metadata, pool, session=session,
                name=prefix + ".attention", workspace=workspace, examples=examples)
            mlp = PreparedMLP(hidden, layer.mlp, empty("mlp.gate_up", (tokens, 2 * intermediate)),
                empty("mlp.intermediate", (tokens, intermediate)), empty("mlp.output", (tokens, config.hidden)),
                session=session, name=prefix + ".mlp", workspace=workspace).functional(session=session)
            self.layers.append((canonical_norm(layer.input_norm, eps), attention,
                                canonical_norm(layer.post_attention_norm, eps), mlp))
        self.norm = canonical_norm(draft.norm, eps)
        logits = empty("logits", (tokens, scratch.local_vocabulary_size))
        self.head = PreparedVocabularyProjection(hidden, head, logits, logits, runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, name=name + ".head", activation_mode="a16",
            workspace=workspace).functional(session=session)

    def release_outputs(self):
        return self

    def forward(self, metadata):
        m = metadata
        hidden = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        for norm, attention, post_norm, mlp in self.layers:
            hidden = hidden + attention(norm(hidden), m)
            hidden = hidden + mlp(post_norm(hidden))
        hidden = self.norm(hidden[m.sample_indices])
        return hidden, self.head(hidden)[:, :self.vocabulary].float()

    def __call__(self, metadata):
        return self.forward(metadata)[1], None

    def scratch_tensors(self):
        return ()


class SpeculativePreparation(dflash.SpeculativePreparation):
    """Target verification with DSpark proposals: the DFlash2 owner's records, decoder and round driver.

    The block holds one position per draft, so the pending token's position scores the first draft.
    """

    def __init__(self, weights, pool, requests, inputs, scratch, **options):
        super().__init__(weights, pool, requests, inputs, scratch, **options)
        block = weights.config.draft.block_size
        self.block_size, self.logit_offset = block, 0
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * block, pool.backing.device, block_size=block)
        self.forwards = SimpleNamespace(decoder=self.forwards.decoder,
            context=partial(PreparedContext, examples=self.examples),
            draft=partial(PreparedDraft, embedding=weights.tree.embed_tokens.weight, head=weights.head,
                          examples=self.examples),
            markov=PreparedMarkov)
