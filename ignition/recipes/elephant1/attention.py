# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 full attention under its trained block mask.

The checkpoint's eight full-attention layers were trained with a block mask: a token attends to every
token of its own block and of every earlier block. Helios derives a prompt's blocks from the declared
block-end marker, keeps each inside one prefill submission, and uploads for every packed token its
distance to the block's first and last token. A submission therefore holds one block or two adjacent
ones, and a prompt without the marker is causal.

A prefill graph serves both cases without a branch. The causal paged operator and the dense
variable-length operator each receive zero lengths when the submission is not theirs, and the output is
selected by the submission's kind. The dense operator reads contiguous keys, so each block's keys, every
key of the request up to the block's end, are gathered from the paged cache after the submission's own
keys are written.
"""

import torch

from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.recipes.qwen38_dense.decoder import FullAttention

from .config import PAGE_TOKENS

SEGMENTS = 2


def block_segments(start, count, left, right):
    """Query and key segment bounds of one prefill submission.

    `start` and `count` are the submission's first position and token count; `left` and `right` hold each
    packed token's distance to its block's first and last token and are zero outside a block. Returns
    whether the submission holds blocks, the cumulative query lengths of its two segments, and their
    cumulative key lengths. A segment's keys are the request's positions before the block's end. A
    submission without blocks has empty segments.
    """
    index = torch.arange(left.numel(), device=left.device)
    blocks = ((left + right) * (index < count)).sum() > 0
    first = (right[0] + 1).clamp(max=count)
    second_right = right.gather(0, first.clamp(max=left.numel() - 1).reshape(1)).squeeze(0)
    first_keys = start + first
    second_keys = torch.where(first < count, first_keys + second_right + 1, 0)
    zero = torch.zeros((), dtype=torch.int64, device=left.device)
    queries = torch.stack((zero, first, count.to(torch.int64) + zero))
    keys = torch.stack((zero, first_keys, first_keys + second_keys))
    return blocks, (queries * blocks).to(torch.int32), (keys * blocks).to(torch.int32)


def key_slots(keys, pages, rows):
    """Paged-cache token slots of both segments' keys, packed in segment order into `rows` rows."""
    index = torch.arange(rows, device=keys.device)
    position = torch.where(index < keys[1], index, index - keys[1])
    position = torch.where(index < keys[2], position, 0)
    return pages[position // PAGE_TOKENS].long().clamp(min=0) * PAGE_TOKENS + position % PAGE_TOKENS


class BlockAttention(FullAttention):
    """Prefill attention that runs declared blocks bidirectionally and everything else causally."""

    def __init__(self, *args, spans, context, **kwargs):
        super().__init__(*args, **kwargs)
        _, weights, _, metadata, pool, layer = args
        if kwargs["mode"] != "prefill" or metadata.page_table.shape[0] != 1 or self.query_dtype != torch.bfloat16:
            raise ValueError("block attention prepares one BF16 prefill row")
        self.left, self.right = spans
        self.context = context
        self.key_cache, self.value_cache = pool.attention(layer, weights.config)
        buffers, device = kwargs["buffers"], self.left.device
        query = buffers["q"][:self.tokens].view(self.tokens, self.heads, self.dim)
        for field in ("block_key", "block_value"):
            if field not in buffers:
                buffers[field] = torch.zeros((SEGMENTS * context, self.kv_heads, self.dim), dtype=torch.bfloat16,
                                             device=device)
        lengths = torch.zeros(SEGMENTS + 1, dtype=torch.int32, device=device)
        self.block = PreparedVarlenAttention(
            query, buffers["block_key"], buffers["block_value"], lengths,
            buffers["attended"][:self.tokens].view_as(query), cu_k=torch.zeros_like(lengths), max_seqlen_k=context,
            session=kwargs["session"], workspace=kwargs["workspace"], name=kwargs["name"] + ".block",
        ).functional(session=kwargs["session"])

    def attend(self, query, gate, lengths, metadata):
        blocks, queries, keys = block_segments(metadata.starts[0], metadata.lengths[0],
                                               self.left[:self.tokens], self.right[:self.tokens])
        gated = self.attention.gated
        causal = self.attention(query, output_gate=gate if gated else None, page_table=metadata.page_table,
                                cu_seqlens_q=torch.where(blocks, 0, metadata.query_start_loc),
                                cache_seqlens=torch.where(blocks, 0, lengths))
        slots = key_slots(keys, metadata.page_table[0], SEGMENTS * self.context)
        block = self.block(query, self.key_cache.flatten(0, 1)[slots], self.value_cache.flatten(0, 1)[slots],
                           queries, keys)
        opened = gate.reshape_as(block).sigmoid()
        return torch.where(blocks, block * opened, causal) if gated else torch.where(blocks, block, causal) * opened

    def forward(self, hidden_states, position_embeddings, *, metadata, **kwargs):
        query, gate, lengths = self.write_cache(hidden_states, position_embeddings, metadata)
        return self.projections["o_proj"](self.attend(query, gate, lengths, metadata).flatten(1)), None
