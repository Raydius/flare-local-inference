# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer attention forwards match canonical arithmetic and FP8 histories under graph replay.

Synthetic weights at a 128-wide hidden size exercise a sliding prefill forward and decode forwards
over global and sliding layers, including inactive rows and the circular wrap at 2048 tokens.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.recipes.muse_glimmer_nvfp4.attention import AttentionScratch, PreparedAttention
from ignition.recipes.muse_spark.pool import ContinuationPool


class Geometry:
    heads, kv_heads, key_dim, value_dim, theta = 16, 2, 64, 64, 500000

    def __init__(self, window):
        self.window, self.rotary_dim = window, 64 if window > 0 else 0

    def local(self, shard):
        return self.heads, self.kv_heads


def configuration(window, hidden=128):
    return SimpleNamespace(layers=1, attention=lambda layer: Geometry(window),
        text=dict(hidden_size=hidden, rms_norm_eps=1e-5, qk_scale_factor=5.4730064863875))


def compiled_readback(attention, source, metadata, scratch):
    """Compile one attention forward that copies its output, key and value rows into retained tensors.

    The forward returns forward-local values, so a replayed graph is observed through these copies.
    """
    from ignition.preparation.compilation import compile_forward

    readback = SimpleNamespace(output=torch.zeros_like(scratch.output), key=torch.zeros_like(scratch.key),
                               value=torch.zeros_like(scratch.value))

    def forward():
        records = []
        output, _ = attention(source, metadata=metadata, records=records)
        (key, value), = records
        readback.output.copy_(output)
        readback.key.copy_(key)
        readback.value.copy_(value)

    return compile_forward(forward), readback



@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')
def test_sliding_prefill_gates_the_raw_attended_value():
    """A sliding prefill forward matches the canonical arithmetic: separate gate projection, no output norm."""
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm, apply_rotary_pos_emb

    device, shard = torch.device('cuda', 0), TensorParallelShard(0, 1)
    torch.manual_seed(2382)
    window, capacity, blocks = 2048, 129, 20
    config = configuration(window)
    pool = ContinuationPool.allocate(config, shard, 3, blocks, device)
    fused = torch.randn((2304, 128), device=device).bfloat16() * .025
    output = torch.randn((128, 1024), device=device).bfloat16() * .025
    weights = SimpleNamespace(qkv=fused[:1280].contiguous(), gate=fused[1280:].contiguous(), output=output)
    source = torch.randn((capacity, 128), device=device).bfloat16()
    metadata = Metadata.allocate(capacity, 1, blocks, device)
    metadata.set(torch.tensor([0], device=device), torch.tensor([capacity], dtype=torch.int32, device=device),
        torch.tensor([1], device=device), torch.arange(blocks, dtype=torch.int32, device=device)[None],
        torch.zeros((1, capacity), device=device, dtype=torch.int64), torch.tensor([1], device=device))
    with RecordedPreparationSession(device=device, checkpoint='muse-glimmer-attention-component',
            workload_schema='muse-glimmer-attention', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, shard, capacity, device)
        attention = PreparedAttention(0, weights, config, shard, source, scratch, metadata, pool,
            mode='prefill', runtime=None, coordinator=None, session=session, name='prefill', workspace=arena)
        session.tune()
        compiled, readback = compiled_readback(attention, source, metadata, scratch)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        pool.clear_for_preparation()
        norm = MuseGlimmerRMSNorm(eps=1e-5, with_scale=False)
        source.normal_()
        q, k, v, gate = (source @ fused.T).split((1024, 128, 128, 1024), -1)
        q = norm(q.view(capacity, 16, 64)) * config.text['qk_scale_factor']
        k, v = norm(k.view(capacity, 2, 64)), v.view(capacity, 2, 64)
        inv = 1 / (500000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
        freq = torch.arange(capacity, device=device).float()[:, None] * inv
        cos, sin = torch.cat((freq, freq), -1).cos().bfloat16(), torch.cat((freq, freq), -1).sin().bfloat16()
        q_ref, k_ref = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos[None], sin[None])
        q, k = q_ref[0].transpose(0, 1), k_ref[0].transpose(0, 1)
        stored_k = k.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
        stored_v = v.clamp(-448, 448).to(torch.float8_e4m3fn).bfloat16()
        ks, vs = stored_k.repeat_interleave(8, 1).float(), stored_v.repeat_interleave(8, 1).float()
        scores = torch.einsum('qhd,thd->hqt', q.float(), ks) / 8
        positions = torch.arange(capacity, device=device)
        valid = positions[None] <= positions[:, None]
        prob = scores.masked_fill(~valid[None], float('-inf')).softmax(-1)
        attended = torch.einsum('hqt,thd->qhd', prob, vs).bfloat16()
        expected = (attended.flatten(1) * gate.sigmoid()) @ output.T
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(readback.output, expected, atol=.02, rtol=.04)
        torch.testing.assert_close(readback.key, k, rtol=0, atol=.03125)
        graph.reset()



@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA GPU')
@pytest.mark.parametrize('window', [-1, 2048])
def test_decode_wrap_and_inactive_lanes(window):
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm

    device, shard = torch.device('cuda', 0), TensorParallelShard(0, 1)
    config = configuration(window)
    pool = ContinuationPool.allocate(config, shard, 5, 80, device)
    fused = torch.randn((2304, 128), device=device).bfloat16() * .025
    weights = SimpleNamespace(qkv=fused[:1280].contiguous(), gate=fused[1280:].contiguous(),
                              output=torch.randn((128, 1024), device=device).bfloat16() * .025)
    source = torch.randn((4, 128), device=device).bfloat16()
    metadata = Metadata.allocate(4, 4, 20, device)
    pages = torch.arange(80, device=device, dtype=torch.int32).view(4, 20)
    starts = torch.tensor([127, 2047, 2048, 2049], device=device)
    states = torch.arange(1, 5, device=device)
    def set_counts(counts):
        metadata.set(starts, torch.tensor(counts, dtype=torch.int32, device=device), states, pages,
            torch.zeros((4, 1), device=device, dtype=torch.int64), torch.tensor([4], device=device))
    set_counts([1, 0, 1, 1])
    with RecordedPreparationSession(device=device, checkpoint='muse-decode-component',
            workload_schema='muse-decode', numerical_policy='default', compile_workers=1) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        scratch = AttentionScratch.allocate(config, shard, 4, device, requests=4)
        attention = PreparedAttention(0, weights, config, shard, source, scratch, metadata, pool,
            mode='decode', runtime=None, coordinator=None, session=session, name='decode', workspace=arena)
        session.tune()
        compiled, readback = compiled_readback(attention, source, metadata, scratch)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for kind in ('key', 'value'):
            cache = pool.cache(0, kind)
            cache.copy_(torch.randn(cache.shape, device=device) * .3)
        norm = MuseGlimmerRMSNorm(eps=1e-5, with_scale=False)
        for counts in ([1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]):
            set_counts(counts)
            before = {kind: pool.cache(0, kind).clone() for kind in ('key', 'value')}
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated(device) == allocated
            projected = source @ fused.T
            q, k, v, gate = projected.split((1024, 128, 128, 1024), -1)
            torch.testing.assert_close(readback.value[:sum(counts)], v[:sum(counts)].view(-1, 2, 64), atol=.002, rtol=.02)
            packed = 0
            for row, count in enumerate(counts):
                if not count:
                    for kind in ('key', 'value'):
                        cache = pool.cache(0, kind)
                        actual = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20]
                        previous = before[kind][row + 1] if window > 0 else before[kind][row * 20:(row + 1) * 20]
                        torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                    continue
                length = min(int(starts[row]) + 1, window) if window > 0 else int(starts[row]) + 1
                caches = []
                for kind, produced in (('key', readback.key), ('value', readback.value)):
                    cache = pool.cache(0, kind)
                    history = cache[row + 1] if window > 0 else cache[row * 20:(row + 1) * 20].flatten(0, 1)
                    slot = int(starts[row]) % window if window > 0 else int(starts[row])
                    quantized = produced[packed].clamp(-448, 448).to(torch.float8_e4m3fn)
                    assert torch.equal(history[slot].view(torch.uint8), quantized.view(torch.uint8))
                    caches.append(history[:length].repeat_interleave(8, 1).float())
                ks, vs = caches
                # The forward does not return its query; the reference applies the canonical head norm,
                # query scale and, on a sliding layer, the rotation at the row's position.
                query = norm(q[packed].view(16, 64)) * config.text['qk_scale_factor']
                if window > 0:
                    inverse = 1 / (500000 ** (torch.arange(0, 64, 2, device=device).float() / 64))
                    angle = (starts[row].float() * inverse).repeat(2)
                    left, right = query.chunk(2, -1)
                    query = query * angle.cos().bfloat16() + torch.cat((-right, left), -1) * angle.sin().bfloat16()
                scores = torch.einsum('hd,thd->ht', query.float(), ks) / 8
                attended = torch.einsum('ht,thd->hd', scores.softmax(-1), vs).bfloat16()
                gated = attended.flatten() * gate[packed].sigmoid()
                expected = gated @ weights.output.T
                torch.testing.assert_close(readback.output[packed], expected, atol=.02, rtol=.04)
                packed += 1
            starts.add_(torch.tensor(counts, device=device))
        graph.reset()
