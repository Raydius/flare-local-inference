# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen patch uploads and generation-scoped image feature publication."""

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.recipes.qwen38.vision import VisionMetadata
from ignition.recipes.qwen38.vision_state import FEATURES, PATCHES, VisionInputs

from .config import VISION

__all__ = ["FEATURES", "PATCH_CAPACITIES", "PreparedVisionGeneration", "VisionInputs"]

PATCH_CAPACITIES = (256, 1024, 4096, 16384)


class PreparedVisionGeneration:
    """Validate one patch upload, encode it, and publish its features to the request's arena slot.

    An upload is valid for a running request of the uploaded generation, a patch count that is a
    positive multiple of four within the graph's capacity, features that append to the request's
    published features within the arena, and finite pixels with in-range interpolation and position
    metadata. An invalid upload reaches the encoder as zeros and ends the request with an error.
    """

    def __init__(self, encoder, inputs, state, completion, frame, sequence):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        self.forward()

    def forward(self):
        e, i, s = self.encoder, self.inputs, self.state
        r, capacity = s.requests, e.capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, offset = i.vision_num_patches, i.vision_feature_offset
        features = count // 4
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        valid &= (i.vision_generation > 0) & (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= capacity) & (count % 4 == 0)
        valid &= (i.vision_cu_seqlens[:1] == 0) & (i.vision_cu_seqlens[1:] == count)
        valid &= (offset >= 0) & (offset + features <= s.arena.shape[1])
        valid &= offset == torch.where(s.generation[slot] == i.vision_generation, s.count[slot], 0)
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        live = (torch.arange(capacity, device=count.device) < count)[:, None]
        pixels, indices = i.vision_pixel_values[:capacity], i.vision_interpolation_indices[:capacity]
        weights, positions = i.vision_interpolation_weights[:capacity], i.vision_position_ids[:capacity]
        valid &= (~live | torch.isfinite(pixels)).all()
        valid &= (~live | ((indices >= 0) & (indices < VISION["num_position_embeddings"]))).all()
        valid &= (~live | ((weights >= 0) & (weights <= 1))).all()
        valid &= (~live | ((positions >= 0) & (positions < PATCHES))).all()
        kept = valid & live
        m = VisionMetadata(torch.where(kept, indices, 0), torch.where(kept, weights, 0),
            torch.where(kept, positions, 0), torch.where(valid, i.vision_cu_seqlens, 0),
            torch.where(valid, count, 0))
        source = pixels.to(torch.bfloat16)
        source.masked_fill_(~kept, 0)
        output = e(source, m)
        feature_live = (torch.arange(capacity // 4, device=count.device) < features)[:, None]
        finite = (~(valid & feature_live) | torch.isfinite(output)).all()
        errors = (~valid).to(torch.int32) | (~finite).to(torch.int32) * 64
        self.publish(output, torch.where(valid, features, 0), errors)
        return output, errors
