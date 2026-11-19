# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch reference of the Qwen3.8-27B DSpark draft for comparison with the prepared draft.

The arithmetic follows the draft repository's `dflash.py` and `dspark.py` at the revision
`ignition/recipes/qwen38_dense/dspark_config.py` pins: target hidden states pass through `fc` and
`hidden_norm` and enter every layer as keys and values beside the block's own; the block is
noncausal; rotary embedding is Hugging Face YaRN with its default attention factor; the vanilla
Markov head adds the bias of the previously sampled token. Weights are the checkpoint's tensors
under their published names, and every product is computed in FP32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def yarn_frequencies(rope: dict, dim: int) -> tuple[torch.Tensor, float]:
    """Inverse frequencies of `dim // 2` rotary pairs and the attention factor of a YaRN configuration."""
    base, factor = float(rope["rope_theta"]), float(rope["factor"])
    original = rope["original_max_position_embeddings"]
    attention_factor = rope.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    def correction_dim(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(float(rope["beta_fast"]))), 0)
    high = min(math.ceil(correction_dim(float(rope["beta_slow"]))), dim - 1)
    if low == high:
        high += 0.001
    position_frequencies = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation = 1.0 - ramp
    inverse = (1.0 / (factor * position_frequencies)) * (1 - extrapolation) \
        + (1.0 / position_frequencies) * extrapolation
    return inverse, attention_factor


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, -1)
    return torch.cat((-right, left), -1)


class DSparkReference:
    """The draft's context projection, noncausal block and Markov bias on one request."""

    def __init__(self, config: dict, tensors: dict[str, torch.Tensor]):
        self.config, self.w = config, tensors
        self.layers, self.eps = config["num_hidden_layers"], config["rms_norm_eps"]
        self.heads, self.kv_heads, self.head_dim = (config["num_attention_heads"],
            config["num_key_value_heads"], config["head_dim"])
        self.block_size, self.mask_id = config["block_size"], config["mask_token_id"]
        self.inverse, self.attention_factor = yarn_frequencies(config["rope_parameters"], self.head_dim)
        self.device = tensors["fc.weight"].device

    def rotary_tables(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine and sine of each position's angles, scaled by the attention factor."""
        frequencies = positions.float()[:, None] * self.inverse.to(positions.device)[None]
        angles = torch.cat((frequencies, frequencies), -1)
        return angles.cos() * self.attention_factor, angles.sin() * self.attention_factor

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate rows `[tokens, heads, head_dim]` at their positions."""
        cos, sin = self.rotary_tables(positions)
        return x * cos[:, None] + rotate_half(x) * sin[:, None]

    def context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Concatenated target hidden states `[positions, taps * hidden]` as the draft's context states."""
        return rms_norm(F.linear(target_hidden.float(), self.w["fc.weight"].float()),
                        self.w["hidden_norm.weight"], self.eps)

    def keys_values(self, layer: int, states: torch.Tensor, positions: torch.Tensor):
        """One layer's rotated keys and its values for context states or normalized block states."""
        p = f"layers.{layer}.self_attn."
        keys = F.linear(states, self.w[p + "k_proj.weight"].float()).view(-1, self.kv_heads, self.head_dim)
        values = F.linear(states, self.w[p + "v_proj.weight"].float()).view(-1, self.kv_heads, self.head_dim)
        return self.rotate(rms_norm(keys, self.w[p + "k_norm.weight"], self.eps), positions), values

    def block_positions(self, start: int) -> torch.Tensor:
        """Positions of the block's inputs: the pending token at `start`, then the mask tokens."""
        return torch.arange(start, start + self.block_size, device=self.device)

    def forward(self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor, *, storage=None,
                history=None) -> torch.Tensor:
        """Normalized hidden rows of one block that follows `target_hidden.shape[0]` accepted positions.

        `storage`, when given, maps each key or value tensor to the values a history store returns for it,
        which models a prepared history's storage precision. `history`, when given, holds each layer's
        stored context keys and values, which the block attends in place of the computed ones.
        """
        store = (lambda x: x) if storage is None else storage
        start = target_hidden.shape[0]
        context = self.context(target_hidden)
        context_positions = torch.arange(start, device=self.device)
        positions = self.block_positions(start)
        hidden = noise_embedding.float()
        groups = self.heads // self.kv_heads
        for layer in range(self.layers):
            p = f"layers.{layer}."
            normalized = rms_norm(hidden, self.w[p + "input_layernorm.weight"], self.eps)
            query = F.linear(normalized, self.w[p + "self_attn.q_proj.weight"].float())
            query = rms_norm(query.view(-1, self.heads, self.head_dim), self.w[p + "self_attn.q_norm.weight"], self.eps)
            query = self.rotate(query, positions)
            if history is None:
                context_keys, context_values = (store(x) for x in self.keys_values(layer, context, context_positions))
            else:
                context_keys, context_values = history[layer]
            block_keys, block_values = (store(x) for x in self.keys_values(layer, normalized, positions))
            keys = torch.cat((context_keys, block_keys)).repeat_interleave(groups, 1)
            values = torch.cat((context_values, block_values)).repeat_interleave(groups, 1)
            scores = torch.einsum("qhd,khd->hqk", query, keys) * self.head_dim ** -0.5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).reshape(hidden.shape[0], -1)
            hidden = hidden + F.linear(attended, self.w[p + "self_attn.o_proj.weight"].float())
            normalized = rms_norm(hidden, self.w[p + "post_attention_layernorm.weight"], self.eps)
            gate = F.linear(normalized, self.w[p + "mlp.gate_proj.weight"].float())
            up = F.linear(normalized, self.w[p + "mlp.up_proj.weight"].float())
            hidden = hidden + F.linear(F.silu(gate) * up, self.w[p + "mlp.down_proj.weight"].float())
        return rms_norm(hidden, self.w["norm.weight"], self.eps)

    def markov_bias(self, previous: torch.Tensor) -> torch.Tensor:
        """The vanilla Markov head's logit bias for the tokens that follow `previous`."""
        embedding = self.w["markov_head.markov_w1.weight"].float()[previous.long()]
        return F.linear(embedding, self.w["markov_head.markov_w2.weight"].float())

    def propose(self, logits: torch.Tensor, pending: int) -> torch.Tensor:
        """Greedy drafts from block logits: row `i` scores the token at block position `i + 1`.

        The Markov bias of row `i` comes from the token at block position `i`, the pending token for
        the first row and the previous draft after it.
        """
        previous = torch.tensor(pending, device=logits.device)
        drafts = []
        for row in logits.float():
            previous = (row + self.markov_bias(previous)).argmax(-1)
            drafts.append(previous)
        return torch.stack(drafts)
