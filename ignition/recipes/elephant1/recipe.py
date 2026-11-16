# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU text-only Elephant-1 preparation for calibrated decision reads."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults

from .config import MODEL_ID, PAGE_TOKENS, REVISION, TRAINED_CONTEXT


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Elephant-1 preparation supports one local GPU")
    if plan.speculation is not None:
        raise ValueError("Elephant-1 serves single-position decisions without learned speculation")
    if plan.benchmark_selection is not None:
        raise ValueError("Elephant-1 has no benchmark selection; it serves single-position decisions")
    config = PreparationConfig(prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort,
        tuning_replica=plan.tuning_replica,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        max_active=plan.concurrency, context=plan.context, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="elephant1", served_model_name="Elephant-1", prepare=prepare, usage=usage,
    # A decision read is one prefill and one scored position, so the served defaults are deterministic.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=1, seed=0),
    checkpoints=(Checkpoint(alias="elephant-1-9b", repository=MODEL_ID, revision=REVISION,
        architecture="Qwen3_5ForConditionalGeneration", stored_bytes=18_819_722_392,
        # Text tensors only: 6.92B body parameters at NVFP4 W4A16, the BF16 embedding and vocabulary head.
        device_bytes=7_963_437_024),
    ),
    context=TRAINED_CONTEXT, page_tokens=PAGE_TOKENS, concurrency=2, max_concurrency=2,
    max_draft_tokens=0, prefill_chunk=2048, language_model_only=True,
    # Continuation and KV geometry, and the graph allowance. At concurrency 2 and the full context, with the KV
    # cache capped at two contexts, the serving process held 9,608 MiB on an RTX 5060 Ti.
    ranks=MappingProxyType({1: RankCosts(55_714_048, 4_194_304, 2 << 30)}),
)
