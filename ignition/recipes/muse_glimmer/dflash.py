# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer projector-less DFlash composition and single-GPU speculative preparation."""

from dataclasses import dataclass
from functools import partial

import torch

from ignition.components.dflash2 import DraftScratch, PreparedContext, PreparedDraft as _PreparedDraft
from ignition.components.speculation import SpeculativeRngState
from ignition.recipes.muse_spark.pool import PAGE_TOKENS


class PreparedDraft(_PreparedDraft):
    """The assistant's block rows through the target's raw mask embedding and softcapped head.

    The assistant carries no convolutions and no selector (``DraftConfig`` admits no
    ``dflash_config``), so the shared draft runs ``norm -> attention -> add; post-norm
    -> MLP -> add`` and sets ``selector = None`` for the round driver. Proposal logits
    are the bare ``lm_head`` readout, as Transformers' DFlash candidate generator reads
    them: the target's ``output_multiplier`` and softcap would flatten the proposal
    distribution (top probability ~4e-4) and stochastic verification would reject
    every draft.
    """

    def __init__(self, weights, *args, **kwargs):
        super().__init__(weights, *args, embedding=weights.embedding, head=weights.head,
            output_multiplier=1., final_logit_softcapping=None, **kwargs)


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


def _target_decoder(weights, tables, scratch, metadata, pool, **kwargs):
    from .decoder import PreparedDecoder
    kwargs["buffers"] = kwargs["buffers"].fork()
    return PreparedDecoder(weights, scratch, metadata, pool, **kwargs)


class SpeculativePreparation:
    """What ``PreparedParallelDraftRound`` reads to build one round at TP1: every runtime is ``None``."""

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, buffers, decoder_buffers, context_buffers):
        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=None)
        self.markov_runtime = None
        self.forwards = DraftForwards(
            partial(_target_decoder, embedding_runtime=None, buffers=decoder_buffers),
            partial(PreparedContext, runtime=None, coordinator=None, buffers=context_buffers),
            partial(PreparedDraft, embedding_runtime=None, buffers=buffers))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device, buffers=buffers, block_size=self.block_size)
