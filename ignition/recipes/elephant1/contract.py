# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Elephant-1 text contract: continuation state, paged BF16 KV, and generation logprob outputs.

The Qwen3.5 chat-template syntax and prefix boundaries are shared with the dense Qwen text recipe, so the contract
construction is the same one; this module exists so the recipe owns its identity and can diverge without touching
another recipe's prepared cores.
"""

from ignition.recipes.qwen38_dense.contract import prepare as _prepare


def prepare(signatures, buffers, pool, *, vocabulary):
    return _prepare(signatures, buffers, pool, vocabulary=vocabulary)
