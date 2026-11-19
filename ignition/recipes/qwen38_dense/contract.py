# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen text inputs, recurrent continuations, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import IMAGE_TOKEN, PAGE_TOKENS


_SYNTAX = dict(tools=p.QWEN_CODER, wrapping_newlines=1, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"))
FRONTEND = c.frontend(**_SYNTAX)
# Image prompts use the Qwen patch preparation and layout, with three rotary positions per token.
IMAGE_FRONTEND = c.frontend(**_SYNTAX, image_tokens=dict(feature_token=IMAGE_TOKEN, span=None),
    image_preparation=p.QWEN_IMAGE_PREPARATION, image_layout=p.QWEN_IMAGE_LAYOUT, position_dimensions=3)


def prepare(signatures, buffers, pool, *, vocabulary, frontend=FRONTEND, image_slots=0):
    """Declare the catalog's uploads and storage; `image_slots` above zero declares image input."""
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    vision = None
    if image_slots:
        from .vision_state import FEATURES, PATCH_CAPACITIES

        inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision"
                                            for field in s.inputs})
        vision = dict(feature_slots=image_slots, features_per_request=FEATURES, max_images=4,
                      patches_per_feature=4, patch_capacities=list(PATCH_CAPACITIES))
        frontend = IMAGE_FRONTEND
    if "decode" in inputs:
        inputs["decode"].pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    mtp = any(s.speculation_enabled for s in signatures.values())
    consumed = pool.layout.region("consumed_tokens") if mtp else None
    cursor = dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=0) if mtp else None
    pending = c.state_view("request_pending_token", 8)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, vision=vision, page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
                 *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        frontend=frontend)
