# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer gated attention over paged FP8 and circular sliding histories, with forward-local values."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.linear import prepare_projection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.window_attention import stage_windows, commit_window
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy

from ignition.recipes.muse_spark.pool import KV_DTYPE, PAGE_TOKENS


@dataclass(frozen=True)
class AttentionScratch:
    """Operand declarations for one attention forward; the forward allocates its own values."""
    projected: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    gated: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, requests=1, examples=None, max_verify_width=8):
        geometries = [config.attention(i) for i in range(config.layers)]
        if len({(a.heads, a.kv_heads, a.key_dim, a.value_dim) for a in geometries}) != 1:
            raise ValueError("Muse Glimmer attention scratch requires matching global and sliding head geometry")
        geometry = geometries[0]
        heads, kv_heads = geometry.local(shard)
        def zeros(role, *shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "muse_glimmer.attention." + role, shape, dtype, device)
        window = max(a.window for a in geometries)
        pages = max((tokens + max(window, 0) + PAGE_TOKENS - 1) // PAGE_TOKENS,
                    requests * ((max(window, 0) + max_verify_width + PAGE_TOKENS - 1) // PAGE_TOKENS))
        return cls(zeros("projected", tokens,
                         2 * heads * geometry.key_dim + kv_heads * (geometry.key_dim + geometry.value_dim)),
            zeros("query", tokens, heads, geometry.key_dim), zeros("key", tokens, kv_heads, geometry.key_dim),
            zeros("value", tokens, kv_heads, geometry.value_dim), zeros("attended", tokens, heads, geometry.value_dim),
            zeros("gated", tokens, heads, geometry.value_dim),
            zeros("partial", tokens, config.text["hidden_size"]), zeros("output", tokens, config.text["hidden_size"]),
            zeros("window_key", pages, PAGE_TOKENS, kv_heads, geometry.key_dim, dtype=KV_DTYPE),
            zeros("window_value", pages, PAGE_TOKENS, kv_heads, geometry.value_dim, dtype=KV_DTYPE))

    def rows(self, tokens):
        return type(self)(*(getattr(self, name) if name.startswith("window_") else getattr(self, name)[:tokens]
                            for name in self.__dataclass_fields__))


def head_norm(values, eps):
    value = values.float()
    return (value * torch.pow(value.square().mean(-1, keepdim=True) + eps, -0.5)).to(values.dtype)


class PreparedAttention(torch.nn.Module):
    """Gated attention whose forward takes its packed metadata and returns local values.

    ``weights.qkv`` rows are ordered ``[q | k | v]`` (``heads × key_dim``, ``kv_heads × key_dim``,
    ``kv_heads × value_dim``) and ``weights.gate`` projects the ``heads × value_dim`` gate. The sigmoid
    gate multiplies the raw attended value; Glimmer applies no per-head norm before it.

    ``scratch`` and ``metadata`` declare operand shapes for preparation and are not retained. A
    sliding layer stages its circular history and the forward's candidates into forward-local FP8
    pages for prefill and verification; decode reads and writes the circular rows in place. A caller
    that passes ``records`` receives each layer's projected key and value rows, which ``commit``
    writes for the accepted prefix of a verification forward.
    """

    def __init__(self, layer, weights, config, shard, source, scratch, metadata, pool, *, mode,
                 runtime, coordinator, session, name, workspace):
        super().__init__()
        self.geometry = geometry = config.attention(layer)
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported Muse Glimmer attention mode: {mode}")
        self.mode = mode
        heads, kv_heads = scratch.query.shape[1], scratch.key.shape[1]
        self.query_shape, self.key_shape, self.value_shape = (
            tuple(t.shape) for t in (scratch.query, scratch.key, scratch.value))
        self.widths = (heads * geometry.key_dim, kv_heads * geometry.key_dim,
                       kv_heads * geometry.value_dim, heads * geometry.value_dim)
        if sum(self.widths) != scratch.projected.shape[1]:
            raise ValueError("Muse attention projection widths must fill the projected scratch rows")
        tokens, qkv_width, gate_width = scratch.projected.shape[0], sum(self.widths[:3]), self.widths[3]
        qkv_region, gate_region = scratch.projected.view(-1).split((tokens * qkv_width, tokens * gate_width))
        self.projection = prepare_projection(source, weights.qkv, qkv_region.view(tokens, qkv_width),
            session=session, name=name + ".qkv", workspace=workspace).functional(session=session)
        self.gate_projection = prepare_projection(source, weights.gate, gate_region.view(tokens, gate_width),
            session=session, name=name + ".gate", workspace=workspace).functional(session=session)
        self.output_projection = prepare_projection(scratch.gated.flatten(1), weights.output, scratch.partial,
            session=session, name=name + ".output", workspace=workspace).functional(session=session)
        self.reduction = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)
        self.inverse = None
        if geometry.rotary_dim:
            self.inverse = persistent_copy(1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
                dtype=torch.float32, device=source.device) / geometry.rotary_dim)))
        self.eps = config.text["rms_norm_eps"]
        self.query_factor = config.text["qk_scale_factor"]
        slots = torch.full_like(metadata.positions, -1)
        self.key_cache, self.value_cache = pool.cache(layer, "key"), pool.cache(layer, "value")
        self.commit_writer = self.writer = self.pages = self.staged_shapes = None
        window, rows = -1, metadata.lengths.numel()
        # Operands bound to None are supplied by each forward.
        runtime_operands = dict(page_table=None, cache_seqlens=None, cu_seqlens_q=None)
        if geometry.window > 0:
            if mode in ("prefill", "verify"):
                key, value = scratch.window_key, scratch.window_value
                if mode == "verify":
                    pages = (geometry.window + metadata.max_query_tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
                    key, value = key[:rows * pages], value[:rows * pages]
                    self.commit_writer = PreparedPagedKVWriter(scratch.key, scratch.value,
                        self.key_cache, self.value_cache, slots, session=session, name=name + ".commit",
                        caller_owned_placeholders=True).functional()
                self.staged_shapes = tuple(key.shape), tuple(value.shape)
                self.pages = pages = persistent_copy(
                    torch.arange(key.shape[0], dtype=torch.int32, device=source.device).view(rows, -1))
                runtime_operands = dict(k_cache=None, v_cache=None, cache_seqlens=None, cu_seqlens_q=None)
                window = geometry.window - 1
            else:
                key, value = pool.window_pages(layer, "key"), pool.window_pages(layer, "value")
                self.window_row_stride = pool.window_page_stride(layer)
                self.window_columns = persistent_copy(torch.arange(geometry.window // PAGE_TOKENS,
                    dtype=torch.int32, device=source.device))
                pages = torch.zeros((rows, self.window_columns.numel()), dtype=torch.int32, device=source.device)
        else:
            key, value, pages = self.key_cache, self.value_cache, metadata.page_table
        if geometry.window < 0 or mode == "decode":
            self.writer = PreparedPagedKVWriter(scratch.key, scratch.value, key, value, slots,
                session=session, name=name + ".cache_write", caller_owned_placeholders=True).functional()
        if geometry.window > 0:
            tuning_page_count = (key.shape[0] if mode in ("prefill", "verify") else
                                 (int(pool.memory_capacity[0].item()) - 1) * self.window_row_stride + geometry.window // PAGE_TOKENS)
        else:
            tuning_page_count = int(pool.memory_capacity[1].item())
        self.attention = PreparedPagedAttention(scratch.query, key, value, pages, torch.zeros_like(metadata.lengths),
            metadata.query_start_loc, scratch.attended, mode=mode, window=window,
            tuning_page_count=tuning_page_count,
            width=1 if mode == "prefill" else metadata.max_query_tokens,
            session=session, workspace=workspace, name=name + ".attention",
            caller_owned_placeholders=True).functional(session=session, **runtime_operands)

    def _rotary(self, values, cos, sin):
        rotary, tail = values.split((self.geometry.rotary_dim, values.shape[-1] - self.geometry.rotary_dim), -1)
        left, right = rotary.chunk(2, -1)
        rotated = torch.cat((-right, left), -1)
        return torch.cat((rotary * cos[:, None] + rotated * sin[:, None], tail), -1)

    def _stage(self, key, value, metadata):
        staged = tuple(torch.empty(shape, dtype=KV_DTYPE, device=key.device) for shape in self.staged_shapes)
        stage_windows(self.key_cache, key, staged[0], metadata)
        stage_windows(self.value_cache, value, staged[1], metadata)
        return staged

    def forward(self, hidden_states, *, metadata, records=None, **kwargs):
        m, a = metadata, self.geometry
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        q, k, v = self.projection(rows).split(self.widths[:3], -1)
        gate = self.gate_projection(rows)
        query = head_norm(q.view(self.query_shape), self.eps) * self.query_factor
        key = head_norm(k.view(self.key_shape), self.eps)
        if a.rotary_dim:
            frequencies = m.positions.clamp_min(0).float()[:, None] * self.inverse[None]
            angles = torch.cat((frequencies, frequencies), -1)
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            query, key = self._rotary(query, cos, sin), self._rotary(key, cos, sin)
        value = v.view(self.value_shape).contiguous()
        if records is not None:
            records.append((key, value))
        operands = dict(cu_seqlens_q=m.query_start_loc)
        if a.window < 0:
            columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
            tokens = torch.arange(m.positions.numel(), device=m.positions.device)
            pages = m.token_page_table[tokens, columns]
            slots = torch.where(m.positions >= 0, pages.long() * PAGE_TOKENS
                                + m.positions.remainder(PAGE_TOKENS), -1)
            operands.update(page_table=m.page_table, cache_seqlens=(m.starts + m.lengths).int())
        elif self.mode == "decode":
            pages = (m.state_rows[:, None] * self.window_row_stride + self.window_columns[None]).int()
            slots = torch.where(m.positions >= 0,
                m.state_rows[m.requests] * self.window_row_stride * PAGE_TOKENS + m.positions.remainder(a.window), -1)
            operands.update(page_table=pages, cache_seqlens=(m.starts + m.lengths).clamp(max=a.window).int())
        else:
            staged_key, staged_value = self._stage(key, value, m)
            operands.update(k_cache=staged_key, v_cache=staged_value,
                            cache_seqlens=m.starts.clamp(max=a.window).int() + m.lengths)
        if self.writer is not None:
            self.writer(key, value, slots)
        attended = self.attention(query, **operands)
        if a.window > 0 and self.mode == "prefill":
            commit_window(self.key_cache, key, m.state_rows[0], m.starts[0], m.lengths[0])
            commit_window(self.value_cache, value, m.state_rows[0], m.starts[0], m.lengths[0])
        gated = attended * torch.sigmoid(gate.view(attended.shape))
        return self.reduction(self.output_projection(gated.flatten(1))).view_as(hidden_states), None

    def commit(self, counts, record, metadata):
        """Write the accepted prefix of a verification forward's rows into the circular history."""
        if self.commit_writer is not None:
            m, window = metadata, self.geometry.window
            key, value = record
            offset = m.positions - m.starts[m.requests]
            slots = torch.where((m.positions >= 0) & (offset >= 0) & (offset < counts[m.requests]),
                m.state_rows[m.requests] * window + m.positions.remainder(window), -1)
            self.commit_writer(key, value, slots)
