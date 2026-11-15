# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer DFlash2 composition and single-GPU speculative preparation."""

from functools import partial

from ignition.components.dflash2_preparation import DFlash2Preparation
from ignition.recipes.muse_spark.pool import PAGE_TOKENS

from .decoder import PreparedDecoder


class SpeculativePreparation(DFlash2Preparation):
    """The shared DFlash2 owner at TP1, over the target's raw mask embedding and ``lm_head``.

    Each draft layer runs a two-tap grouped convolution before its attention and
    another before its MLP; the rank-256 top-16 candidate selector traces the block
    once the layers are done. The draft head applies the target's ``output_multiplier``
    and ``final_logit_softcapping`` from ``dflash_config``, so proposal logits sit on
    the target's own scale. The proposal block is the drafter's sixteen tokens at
    every verification width.
    """

    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, examples):
        def decoder(weights, tables, scratch, metadata, pool, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, embedding_runtime=None, examples=examples,
                                   **kwargs)

        super().__init__(weights, pool, requests, inputs, scratch, decoder=decoder,
            embedding=weights.embedding, head=weights.head, page_tokens=PAGE_TOKENS,
            completion=completion, frame=frame, sequence=sequence, session=session, workspace=workspace,
            verification_width=weights.config.draft.block_size, examples=examples)
        options = weights.config.draft.text["dflash_config"]
        self.forwards.draft = partial(self.forwards.draft, output_multiplier=options["output_multiplier"],
                                      final_logit_softcapping=options["final_logit_softcapping"])
