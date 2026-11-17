# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 text contract: continuation state, paged BF16 KV, and generation logprob outputs.

The Qwen3.5 chat-template syntax and prefix boundaries are shared with the dense Qwen text recipe, so the contract
construction is the same one. Elephant-1 adds its block-end marker: the tokens that close a rendered state
block, which split a decision prompt into the shared prefix and the question suffix of the trained mask.
"""

from ignition.recipes.qwen38_dense.contract import FRONTEND, prepare as _prepare

# The end of the vendor prompt's state block (`qev.prompt.state_block`).
BLOCK_END_TEXT = "</state>\n\n"


def prepare(signatures, buffers, pool, *, vocabulary, block_end_marker):
    return _prepare(signatures, buffers, pool, vocabulary=vocabulary,
                    frontend=dict(FRONTEND, block_end_marker=list(block_end_marker)))
