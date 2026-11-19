# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B DSpark draft weights: the shared draft loader and shape table with the DSpark tensor set.

Each layer's ``qkv`` fuses its query, key and value projections and its convolution fields are
``None``; ``context_kv`` stacks every layer's key rows followed by its value rows. The draft embeds
tokens and computes logits with the target's weights, and its confidence head is validated and not
loaded.
"""

from functools import partial

from ignition.components import dflash2_weights

load = partial(dflash2_weights.DraftWeights.load, projector="dspark")
tensor_shapes = partial(dflash2_weights.tensor_shapes, projector="dspark")
validate_metadata = partial(dflash2_weights.validate_metadata, projector="dspark")
