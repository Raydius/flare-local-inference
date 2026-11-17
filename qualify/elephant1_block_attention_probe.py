# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe dense noncausal attention and prefill storage at Elephant-1's block sizes.

Elephant-1's eight full-attention layers were trained with a two-block mask: a token attends to its
whole block and to every earlier block. This probe answers two questions before anything is built.

  operator  whether b12x variable-length attention, the operator behind
            `ignition.components.varlen_attention`, runs noncausally at the model's geometry
            (16 query heads, 4 key-value heads, head dimension 256, BF16) for a question suffix
            attending to the prefix and itself, and for a whole prefix attending to itself. Each
            output is compared with PyTorch `scaled_dot_product_attention` in FP32 over the same
            BF16-rounded inputs, which are seeded standard-normal values, not model activations.
  storage   the bytes the decoder's declared scratch, the batch inputs, the prefill pages of the
            paged key-value pool, the prompt-logprob outputs and the gathered block keys and values
            take for one prefill row at each prefill capacity. This is
            allocation arithmetic on the CPU: it leaves out every operator's private scratch in a
            captured graph and is not a graph measurement.

    python -m qualify.elephant1_block_attention_probe --checkpoint DIR --device cuda:0 \\
        --output qualify/results/elephant1-mask-feasibility/operator.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

QUERY_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 256
# (name, description, query tokens, key tokens)
CASES = (("suffix_over_prefix", "256 suffix queries over 4,096 prefix keys and the suffix's own 256 keys", 256, 4352),
         ("whole_prefix", "one 16,384-token block attending to itself", 16384, 16384))
PREFILL_CAPACITIES = (2048, 4096, 8192, 16384)
ENVELOPE_BYTES, SERVING_PROCESS_BYTES = 12 << 30, 9608 << 20
REFERENCE_QUERY_ROWS = 512


def _reference(q, k, v):
    """FP32 attention over the BF16-rounded inputs, a slice of queries at a time."""
    keys, values = (tensor.float().transpose(0, 1).repeat_interleave(QUERY_HEADS // KV_HEADS, 0) for tensor in (k, v))
    return torch.cat([
        torch.nn.functional.scaled_dot_product_attention(rows.float().transpose(0, 1), keys, values).transpose(0, 1)
        for rows in q.split(REFERENCE_QUERY_ROWS)])


def _attend(device, queries, keys):
    from b12x.attention import varlen

    generator = torch.Generator(device=device).manual_seed(queries + keys)

    def normal(rows, heads):
        return torch.randn((rows, heads, HEAD_DIM), generator=generator, device=device).bfloat16()

    q, k, v = normal(queries, QUERY_HEADS), normal(keys, KV_HEADS), normal(keys, KV_HEADS)
    cu_q, cu_k = (torch.tensor([0, rows], dtype=torch.int32, device=device) for rows in (queries, keys))
    out = torch.zeros_like(q)
    plan = varlen.plan(q, k, v, cu_q, cu_k, max_seqlen_q=queries, max_seqlen_k=keys, causal=False)
    specs = plan.scratch_specs()
    scratch, = (torch.zeros(tuple(spec.shape), dtype=spec.dtype, device=device) for spec in specs)
    varlen.run_out(q, k, v, cu_q, out=out, scratch=scratch, plan=plan, cu_seqlens_k=cu_k, max_seqlen_q=queries,
                   max_seqlen_k=keys, softmax_scale=HEAD_DIM ** -0.5)
    torch.cuda.synchronize(device)
    reference = _reference(q, k, v)
    return {"ran": True, "maximum_absolute_difference": (out.float() - reference).abs().max().item(),
            "reference_maximum_magnitude": reference.abs().max().item(),
            "scratch_bytes": sum(tensor.numel() * tensor.element_size() for tensor in (scratch,)),
            "operand_bytes": sum(tensor.numel() * tensor.element_size() for tensor in (q, k, v, out))}


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    return sum(_tensor_bytes(getattr(value, field.name)) for field in dataclasses.fields(value))


def _storage(checkpoint):
    from ignition.components.paged_batch import BatchInputs
    from ignition.components.prompt_logprobs import PromptOutputs
    from ignition.recipes.elephant1.attention import SEGMENTS
    from ignition.recipes.elephant1.config import PAGE_TOKENS, TRAINED_CONTEXT, ModelConfig
    from ignition.recipes.elephant1.pool import page_bytes
    from ignition.recipes.qwen38_dense.decoder import Scratch

    config, cpu = ModelConfig.load(checkpoint), torch.device("cpu")
    headroom = ENVELOPE_BYTES - SERVING_PROCESS_BYTES
    # Keys and values of both segments, gathered from pages for the dense operator; today's causal core has none.
    gathered = 2 * SEGMENTS * TRAINED_CONTEXT * KV_HEADS * HEAD_DIM * torch.finfo(torch.bfloat16).bits // 8
    rows = []
    for tokens in PREFILL_CAPACITIES:
        scratch = _tensor_bytes(Scratch.allocate(tokens, 1, config, cpu))
        inputs = _tensor_bytes(BatchInputs.allocate(tokens, TRAINED_CONTEXT // PAGE_TOKENS, cpu, capacity=2))
        scoring = sum(tensor.numel() * tensor.element_size() for tensor in PromptOutputs(tokens, cpu).buffers().values())
        rows.append({"prefill_tokens": tokens, "decoder_scratch_bytes": scratch, "batch_input_bytes": inputs,
                     "prefill_page_bytes": tokens // PAGE_TOKENS * page_bytes(config),
                     "prompt_output_bytes": scoring})
    for row in rows:
        row["increase_over_2048_bytes"] = gathered + sum(value - rows[0][field] for field, value in row.items()
                                                         if field.endswith("_bytes"))
        row["fits_headroom"] = row["increase_over_2048_bytes"] <= headroom
    return {"method": "allocation arithmetic on the CPU from Scratch.allocate, BatchInputs.allocate, the page pool's"
                      " prefill pages (chunk / 128 pages), the prompt-logprob outputs and the gathered block keys"
                      " and values for one prefill row; not a captured-graph measurement, and it leaves out each"
                      " operator's private scratch",
            "envelope_bytes": ENVELOPE_BYTES, "serving_process_bytes": SERVING_PROCESS_BYTES,
            "headroom_bytes": headroom, "gathered_block_key_value_bytes": gathered,
            "headroom_basis": "the measured serving process already holds the 2,048-token prefill storage, so a"
                              " capacity fits when its increase over 2,048 tokens is within the headroom",
            "capacities": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="baselabs/elephant-1 snapshot directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    cases = {}
    for name, description, queries, keys in CASES:
        try:
            with torch.no_grad():
                result = _attend(device, queries, keys)
        except Exception as error:  # the probe's finding when the operator rejects a geometry
            result = {"ran": False, "error": f"{type(error).__name__}: {error}"}
        cases[name] = {"description": description, "query_tokens": queries, "key_tokens": keys, **result}
        torch.cuda.empty_cache()
        print(f"{name}: {result}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "operator": {"name": "b12x.attention.varlen, causal=False, flash arithmetic", "query_heads": QUERY_HEADS,
                     "kv_heads": KV_HEADS, "head_dimension": HEAD_DIM, "dtype": "bfloat16",
                     "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
                     "reference": "torch scaled_dot_product_attention in FP32 over the same BF16-rounded seeded"
                                  " standard-normal inputs",
                     "cases": cases},
        "prefill_storage": _storage(args.checkpoint)}, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
