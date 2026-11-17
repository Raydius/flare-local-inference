# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a single-GPU Elephant-1 text graph catalog with device-resident state."""

import contextlib
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.prompt_logprobs import PreparedScoredPrefill, PromptScoring, prefill_modes

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase
from ignition.recipes.qwen38_dense.bootstrap import _declared_weight_roots
from ignition.recipes.qwen38_dense.decoder import PreparedDecoder, Scratch

from .attention import BlockAttention

from .config import PAGE_TOKENS, TRAINED_CONTEXT
from .contract import BLOCK_END_TEXT, prepare as prepare_contract
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    device: int = 0
    context: int = TRAINED_CONTEXT
    prefill_chunk_size: int = TRAINED_CONTEXT
    max_active: int = 2
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None

    def __post_init__(self):
        if not PAGE_TOKENS <= self.context <= TRAINED_CONTEXT or self.context % PAGE_TOKENS:
            raise ValueError(f"Elephant-1 context must be 128-aligned in [128,{TRAINED_CONTEXT}]")
        if not PAGE_TOKENS <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % PAGE_TOKENS:
            raise ValueError("Elephant-1 prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        request_capacity(self.max_active)


@torch.no_grad()
@with_autotune_effort
def prepare(snapshot, config, *, progress=None, core_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress)
    release_unused()
    layout = continuation_layout(weights.config)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Elephant-1 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    decode_tokens = slots
    state_rows = slots + snapshots
    pages = config.prefill_chunk_size // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(weights.config, state_rows + 1, pages, device,
            storage=core_storage, maximum_pages=total_memory // page_bytes(weights.config),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, slots), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
        coordinator = None
        identity = hashlib.sha256(json.dumps({
            "config": weights.config.text, "headers": [s.header_sha256 for s in manifest.shards],
            "contract": "elephant-1.graph-inputs.v2", "full_attention": "block_bidirectional"},
            sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="elephant-1", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, weights.config, device)
        decode_storage = Scratch.allocate(decode_tokens, slots, weights.config, device)
        # Each packed prefill token's distance to its block's first and last token, zero outside a block.
        spans = tuple(torch.zeros(config.prefill_chunk_size, dtype=torch.int64, device=device) for _ in range(2))
        buffers = dict(inputs.buffers(), **commands.buffers(), vision_left=spans[0], vision_right=spans[1],
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame)
    forwards, signatures = {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=False)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for kind, mode, sizes in prefill_modes(prefill_sizes, tuple(reversed(batch_sizes)), scoring):
            for capacity in sizes:
                rows = 1 if mode == "prefill" else capacity
                name = f"{kind}{capacity}"
                m = Metadata.allocate(capacity, rows, blocks, device)
                scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                decoder = PreparedDecoder(weights, scratch, m, pool, mode=mode,
                    scoring=scoring if kind == "scored_prefill" else None,
                    vocabulary_runtime=None, coordinator=coordinator, session=session, name=name, workspace=arena,
                    attention=(partial(BlockAttention, spans=spans, context=config.context)
                               if mode == "prefill" else None))
                batch = PreparedBatch(inputs, requests, m, mode=mode, shard=weights.shard, pool=pool)
                forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=None,
                    coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                    context_factory=None)
                add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                    (*forward_inputs, "vision_left", "vision_right") if mode == "prefill" else forward_inputs)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for capacity in reversed(batch_sizes):
            name = f"resume{capacity}"
            forward = PreparedResume(weights, requests, pool, inputs, completion, frame, sequence,
                rows=capacity, runtime=None, vocabulary_runtime=None, coordinator=coordinator,
                session=session, name=name, activation_mode="auto", workspace=arena)
            add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                    runtime=None, coordinator=coordinator, session=session, name=name, capacity=capacity,
                    admission_buffers=admission, draft_state=False)
                fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
    del prefill_storage, decode_storage, m, scratch
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    arena.release_preparation()
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
        stream=stream, pool=graph_pool, collective=None, coordinator=coordinator,
        compile_decoder=config.compile_decoder, progress=progress, cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs, materialize=pool.clear_for_preparation, warmup_storage=core_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "init", "restore", "snapshot"))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {role: {"path": str(snapshot / file)} for role, file in (("tokenizer", "tokenizer.json"),)}
    assets["chat_template"] = annotated_template_asset(
        (snapshot / "chat_template.jinja").read_bytes(), Path(__file__).parent / "assets" / "chat_template.patch.json")
    eos = weights.config.text["eos_token_id"]
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text())
    eos_text = tokenizer["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    from tokenizers import Tokenizer
    vocabulary_tokens = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer_eos = vocabulary_tokens.token_to_id(eos_text)
    block_end_marker = vocabulary_tokens.encode(BLOCK_END_TEXT, add_special_tokens=False).ids
    eos_ids = list(dict.fromkeys([*(eos if isinstance(eos, list) else [eos]), tokenizer_eos]))
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=eos_ids),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
            block_end_marker=block_end_marker))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config), completion_output_bytes=completion.nbytes,
        operator_scratch_bytes=arena.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder,
        speculative_shapes=(), speculative_round_shapes=(), speculative_graph_budget=0,
        graph_buffer_bytes=0, private_buffer_bytes=0, skipped_vision_tensors=weights.skipped_vision_tensors)
    if scoring is not None:
        memory.update(scoring.memory_report())
    roots, components = _declared_weight_roots(weights)
    prepared = PreparedModel(rebuild_weights=declare_weights(roots, components=components, layers=()),
        device=config.device, graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        benchmark_state=None, benchmark_pool=None, benchmark_requests=None, benchmark_observers=None,
        benchmark_rounds=None,
        close_resources=(() if scoring is None else (scoring,)),
        graph_resources=(),
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                            *storage_ranges([arena.tensor,
                                             [f.scratch_tensors() for f in forwards.values() if hasattr(f, "scratch_tensors")]]),
                            *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, compiled,
                   stream, graph_pool, session, coordinator))
    return prepared
