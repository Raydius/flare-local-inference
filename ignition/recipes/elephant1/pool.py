# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident Gated DeltaNet histories and paged BF16 full-attention KV for Elephant-1."""

import torch

from ignition.components.gated_delta import GDNState
from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS


def convolution_width(config):
    """Channels carried by one GDN convolution state: concatenated Q/K/V inputs."""
    text = config.text
    return (2 * text["linear_num_key_heads"] * text["linear_key_head_dim"]
            + text["linear_num_value_heads"] * text["linear_value_head_dim"])


def recurrent_shape(config):
    text = config.text
    return (text["linear_num_value_heads"], text["linear_key_head_dim"], text["linear_value_head_dim"])


def page_specs(config):
    size = PAGE_TOKENS * config.text["num_key_value_heads"] * config.text["head_dim"]
    for index, kind in enumerate(config.layers):
        if kind == "full_attention":
            for field in ("key", "value"):
                yield f"layer{index}.{field}", (2 * size,)


def page_bytes(config):
    return sum(shape[0] for _, shape in page_specs(config))


def continuation_layout(config, *, verification_width=1):
    if type(verification_width) is not int or not 1 <= verification_width <= 16:
        raise ValueError("Elephant-1 verification capacity must be an integer in [1,16]")
    regions = []
    for index, kind in enumerate(config.layers):
        if kind == "linear_attention":
            regions.extend(((f"layer{index}.convolution", (convolution_width(config), 2 + verification_width),
                             torch.bfloat16),
                            (f"layer{index}.recurrent", recurrent_shape(config), torch.float32)))
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config))
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16),
                    ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, verification_width=1, **kwargs):
        return cls.allocate(continuation_layout(config, verification_width=verification_width),
                            page_specs(config), slots, pages, device, **kwargs)

    def partial_pages(self):
        return tuple(self.pages.items())

    def recurrent(self, layer):
        return GDNState(self.reserved_region(f"layer{layer}.convolution"),
                        self.reserved_region(f"layer{layer}.recurrent"))

    def attention(self, layer, config):
        shape = (-1, PAGE_TOKENS, config.text["num_key_value_heads"], config.text["head_dim"])
        prefix = f"layer{layer}"
        return tuple(self.pages[f"{prefix}.{field}"].view(torch.bfloat16).view(shape)
                     for field in ("key", "value"))
