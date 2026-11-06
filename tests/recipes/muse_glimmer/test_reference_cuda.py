# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Glimmer prefill and decode against the layer-streamed NVFP4 text reference.

The traced prompt is a chat-templated conversation tokenized by the checkpoint tokenizer and
long enough to cross the 2048-token sliding window, so sliding (rope) and full (NoPE) layers
are compared on both sides of the boundary. Prefill rows come from a forward hook on the last
prepared layer through the decoder's norm and head arithmetic; the sampled row and the decode
row come from the decoder's own head. The reference rounds keys and values to E4M3 at the
cache seam (``Fp8History``) as the prepared rings and pages do, so what remains is NVFP4 kernel
versus FP32-decode arithmetic and BF16 reduction order over 52 layers.

Tolerance: relative L2 (reference norm as denominator, docs/core-design.md) at most 5e-2 on the
final normalized hidden states and the softcapped logits of every compared row group, and
top-1 agreement of at least 0.9 over the prefill rows. Spark's checkpoint tests bound single
stages at 4e-3 to 8e-3 (tests/recipes/muse_spark/test_checkpoint_moe_cuda.py); this is the
whole-model bound over 52 layers.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")
SNAPSHOT = os.environ.get("FLARE_MUSE_GLIMMER_SNAPSHOT")
CHECKPOINT = pytest.mark.skipif(not SNAPSHOT, reason="FLARE_MUSE_GLIMMER_SNAPSHOT names no checkpoint")
WINDOW = 2048
PROMPT_MINIMUM = WINDOW + 200
RELATIVE_L2 = .05
TOP1_AGREEMENT = .9


def device():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    selected = torch.device("cuda", 0)
    torch.cuda.set_device(selected)
    torch.backends.cuda.matmul.allow_tf32 = False
    return selected


def traced_prompt(snapshot, minimum=PROMPT_MINIMUM):
    """Token ids of a chat-templated prompt with at least ``minimum`` tokens, from the checkpoint tokenizer."""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot)
    template = (Path(snapshot) / "chat_template.jinja").read_text()
    lines = 64
    while True:
        passage = "\n".join(f"Entry {i}: the reading of gauge {i} was {i * 7 % 101} units." for i in range(lines))
        messages = [{"role": "system", "content": "You are a careful assistant."},
                    {"role": "user", "content": passage + "\n\nWhich entry has the largest reading?"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=template,
                                             add_generation_prompt=True, current_date="2026-11-06")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= minimum:
            return ids
        lines += 16


def relative_l2(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def softcapped(hidden, head, config):
    logits = F.linear(hidden, head).float() * config["output_multiplier"]
    return torch.tanh(logits / config["final_logit_softcapping"]) * config["final_logit_softcapping"]


def prefill(weights, pool, ids, selected, *, blocks, session, workspace, name):
    """Construct the prepared prefill of ``ids`` on state row 1 over pages ``0..blocks-1``."""
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.muse_glimmer.decoder import PreparedDecoder, Scratch

    tokens = len(ids)
    metadata = Metadata.allocate(tokens, 1, blocks, selected)
    metadata.set(torch.tensor([0], device=selected), torch.tensor([tokens], dtype=torch.int32, device=selected),
                 torch.tensor([1], device=selected), torch.arange(blocks, dtype=torch.int32, device=selected)[None],
                 torch.tensor(ids, device=selected).view(1, -1), torch.tensor([1], dtype=torch.int32, device=selected))
    scratch = Scratch.allocate(weights.config, weights.shard, tokens, 1, selected)
    decoder = PreparedDecoder(weights, scratch, metadata, pool, mode="prefill", runtime=None, embedding_runtime=None,
                              vocabulary_runtime=None, coordinator=None, session=session, name=name, workspace=workspace)
    return decoder, metadata


@CUDA
@CHECKPOINT
@torch.no_grad()
def test_prepared_prefill_and_decode_match_the_reference_across_the_window():
    from b12x.preparation import PreparationSession
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.muse_glimmer.decoder import PreparedDecoder, Scratch
    from ignition.recipes.muse_glimmer.weights import ModelWeights
    from ignition.recipes.muse_spark.pool import PAGE_TOKENS, ContinuationPool
    from ignition.reference.muse_glimmer import Fp8History, text_logits

    snapshot, selected = Path(SNAPSHOT), device()
    ids = traced_prompt(snapshot)
    tokens = len(ids)
    assert tokens > WINDOW
    shard = TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(snapshot), selected, shard=shard)
    c = weights.config.text
    blocks = (tokens + 1 + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool = ContinuationPool.allocate(weights.config, shard, 3, blocks, selected, dflash=False)
    with PreparationSession(device=selected, autotune=False) as session:
        arena = ScratchArena(selected)
        decoder, _ = prefill(weights, pool, ids, selected, blocks=blocks, session=session, workspace=arena,
                             name="glimmer.reference.prefill")
        decode_metadata = Metadata.allocate(1, 1, blocks, selected)
        decode = PreparedDecoder(weights, Scratch.allocate(weights.config, shard, 1, 1, selected), decode_metadata, pool,
                                 mode="decode", runtime=None, embedding_runtime=None, vocabulary_runtime=None,
                                 coordinator=None, session=session, name="glimmer.reference.decode", workspace=arena)
        arena.finalize()
        pool.clear_for_preparation()
        rows = {}
        handle = decoder.layers[-1].register_forward_hook(
            lambda module, args, output: rows.__setitem__("last", output.clone()))
        decoder()
        torch.cuda.synchronize()
        handle.remove()
        prefill_hidden = decoder.norm(rows["last"])
        prefill_logits = softcapped(prefill_hidden, weights.head, c)
        sampled_hidden, sampled_logits = decoder.scratch.hidden[0].clone(), decoder.scratch.logits[0].clone()
        next_token = int(sampled_logits.argmax())
        decode_metadata.set(torch.tensor([tokens], device=selected), torch.tensor([1], dtype=torch.int32, device=selected),
                            torch.tensor([1], device=selected), torch.arange(blocks, dtype=torch.int32, device=selected)[None],
                            torch.tensor([[next_token]], device=selected), torch.tensor([1], dtype=torch.int32, device=selected))
        decode()
        torch.cuda.synchronize()
        decode_hidden, decode_logits = decode.scratch.hidden[0].clone(), decode.scratch.logits[0].clone()
    assert torch.isfinite(prefill_hidden).all() and torch.isfinite(prefill_logits).all()
    assert prefill_logits.abs().max() <= c["final_logit_softcapping"]

    hidden, logits = text_logits(snapshot, ids + [next_token], device=str(selected), history=Fp8History())
    hidden, logits = hidden.to(selected), logits.to(selected)
    assert hidden.shape == (tokens + 1, c["hidden_size"]) and logits.shape == (tokens + 1, c["vocab_size"])
    report = {}
    for name, span in (("within_window", slice(0, WINDOW)), ("past_window", slice(WINDOW, tokens))):
        report[name] = dict(rows=span.stop - span.start,
            hidden=relative_l2(prefill_hidden[span], hidden[span]), logits=relative_l2(prefill_logits[span], logits[span]),
            top1=float((prefill_logits[span].argmax(-1) == logits[span].argmax(-1)).float().mean()))
    report["sampled"] = dict(hidden=relative_l2(sampled_hidden, hidden[tokens - 1]),
                             logits=relative_l2(sampled_logits, logits[tokens - 1]))
    decode_scores = decode_logits.float()
    report["decode"] = dict(hidden=relative_l2(decode_hidden, hidden[tokens]), logits=relative_l2(decode_logits, logits[tokens]),
                            reference_argmax_rank=int((decode_scores > decode_scores[logits[tokens].argmax()]).sum()))
    print(dict(tokens=tokens, next_token=next_token, relative_l2_bound=RELATIVE_L2, top1_bound=TOP1_AGREEMENT, **report),
          flush=True)
    for name, errors in report.items():
        assert errors["hidden"] <= RELATIVE_L2 and errors["logits"] <= RELATIVE_L2, (name, errors)
    assert report["within_window"]["top1"] >= TOP1_AGREEMENT and report["past_window"]["top1"] >= TOP1_AGREEMENT, report
    assert report["decode"]["reference_argmax_rank"] < 5, report["decode"]
