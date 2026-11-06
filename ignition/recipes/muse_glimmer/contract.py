# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark's text contract with Glimmer's frontend and, when prepared, its vision graphs."""

from ignition.preparation import contract as c
from ignition.recipes.muse_spark.contract import prepare as _prepare

from .frontend import FRONTEND

IMAGE_KEYS = ("image_tokens", "image_preparation", "image_layout")


def vision_block(image_slots):
    from .vision import PATCH_CAPACITIES
    from .vision_state import FEATURES

    return dict(feature_slots=image_slots, features_per_request=FEATURES, max_images=4,
                patches_per_feature=4, patch_capacities=list(PATCH_CAPACITIES))


def prepare(signatures, buffers, pool, *, vocabulary, draft_lag=None, image_slots=None):
    """``image_slots`` names the prepared feature slots; ``None`` is the text-only package."""
    kinds = {s.kind for s in signatures.values()}
    text = {name: s for name, s in signatures.items() if s.kind != "vision"}
    if (image_slots is None) != ("vision" not in kinds):
        raise ValueError("vision graphs and image slots are prepared together")
    frontend = dict(FRONTEND)
    vision = None
    if image_slots is not None:
        vision = vision_block(image_slots)
    else:
        for key in IMAGE_KEYS:
            frontend.pop(key, None)
    contract = _prepare(text, buffers, pool, vocabulary=vocabulary, draft_lag=draft_lag, vision=vision,
                        frontend=frontend)
    if image_slots is not None:
        # Spark's text inputs name no image tensors; vision graphs bind theirs by field name.
        inputs = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
        contract["graphs"].extend(c.graph_contracts(
            {name: s for name, s in signatures.items() if s.kind == "vision"}, {"vision": inputs},
            continuation=None, pending=None))
    return contract
