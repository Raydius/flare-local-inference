"""Prepare the Qwen graph catalog with DFlash2 drafts on one GPU before starting native serving."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.capacity import speculative_token_capacity
from ignition.components.completion import CompletionOutputs
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)
from ignition.components.sampling import AdmissionBuffers
from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation import runtime_package_pb2 as contract_proto
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.contract import history_region
from ignition.preparation.deferred import deferred_mode
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.host_memory import device_memory_info
from ignition.preparation.model import PreparedModel
from ignition.preparation.qwen38 import FRONTEND_SHA256, MODEL_ID, REVISION
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import GraphBufferPool, ScratchArena, bind_scratch
from ignition.preparation.tuning import with_autotune_effort
from ignition.progress import phase
from ignition.recipes.qwen38.attention import QSAState, RotaryTables
from ignition.recipes.qwen38.bootstrap import PreparationConfig as _QwenPreparationConfig, _identity as _qwen_identity
from ignition.recipes.qwen38.contract import prepare as prepare_contract
from ignition.recipes.qwen38.decoder import DecoderScratch, ModelWeights, PreparedDecoder
from ignition.recipes.qwen38.decoder_state import DecoderInputs, PreparedDecoderState, PreparedResume
from ignition.recipes.qwen38.maintenance import PreparedMaintenance
from ignition.recipes.qwen38.ple import PLETable
from ignition.recipes.qwen38.sampling import DeviceRequestState
from ignition.recipes.qwen38.state import HybridStateLayout, HybridStatePool, StateTransferMetadata
from ignition.recipes.qwen38.tuning import preparation_session
from ignition.recipes.qwen38.vision import PreparedVision, VisionScratch

from .config import DRAFT_REVISION, DraftConfig
from .draft_weights import DraftWeights
from .export import disposable_ranges
from .recipe import DRAFT_WINDOWS, RECIPE

SPECULATIVE_TOKENS = 7


def state_layout(drafting):
    """The hybrid row every DFlash preparation allocates: speculative histories, rings, no MTP regions."""
    return HybridStateLayout.build(speculative_tokens=SPECULATIVE_TOKENS if drafting else 0, world_size=1,
                                   mtp=False, draft_windows=DRAFT_WINDOWS if drafting else None)


@dataclass(frozen=True)
class PreparationConfig(_QwenPreparationConfig):
    """qwen38's preparation settings on one GPU, speculating through ``draft_checkpoint`` instead of ``mtp``.

    The inherited ``mtp`` field is qwen38's speculation switch; here it mirrors
    ``draft_checkpoint`` so the inherited capacity properties describe the DFlash catalog.
    """

    draft_checkpoint: Path | None = None

    def __post_init__(self):
        if self.draft_checkpoint is not None and not isinstance(self.draft_checkpoint, Path):
            raise TypeError("draft_checkpoint must be a Path")
        if (self.hosts is not None or self.roce_ring or self.roce_chain or self.shard_hyperconnections
                or (self.devices is not None and len(self.devices) != 1)):
            raise ValueError("qwen3.8-flash-dflash2 prepares on a single GPU: the DFlash2 draft is not sharded")
        if self.mixed_prefill:
            raise ValueError("qwen3.8-flash-dflash2 prepares no mixed prefill rounds")
        object.__setattr__(self, "mtp", self.drafting)
        super().__post_init__()
        if self.snapshot_budget_bytes < state_layout(self.drafting).row_bytes:
            raise ValueError("snapshot budget requires at least one state row; headroom must be nonnegative")

    @property
    def drafting(self):
        return self.draft_checkpoint is not None


def _identity(manifest, draft, layout, rope_scaling=None):
    """Helios seeds prefix-cache digests from this: the target identity plus what this recipe adds to a row."""
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.glob("*.py"))}
    value = {"target": _qwen_identity(manifest, 1, rope_scaling), "recipe": RECIPE.name, "sources": sources,
             "draft": None if draft is None else {"revision": DRAFT_REVISION,
                                                  "headers": [shard.header_sha256 for shard in draft.shards]},
             "row_bytes": layout.row_bytes}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def prepare_model(snapshot: Path, config: PreparationConfig | None = None, *, progress=None,
                  compile_decoder: bool = True, package_storage=None) -> PreparedModel:
    config = PreparationConfig() if config is None else config
    transient = contextlib.ExitStack()
    try:
        prepared = _prepare_model(snapshot, config, progress=progress, compile_decoder=compile_decoder,
                                  package_storage=package_storage, transient=transient)
    except BaseException as error:
        try:
            transient.close()
        except BaseException as cleanup_error:
            error.add_note(f"transient allocation scope cleanup failed: {cleanup_error!r}")
        raise
    else:
        transient.close()
        return prepared


@with_autotune_effort
def _prepare_model(snapshot: Path, config: PreparationConfig, *, transient, progress=None,
                   compile_decoder: bool = True, package_storage=None) -> PreparedModel:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    from ignition.recipes.qwen38.vision_state import (
        PreparedEmbeddingInjection,
        PreparedVisionGeneration,
        PreparedVisionHandoff,
        VisionFeatureState,
        VisionInputs,
    )

    def report(message):
        if progress is not None:
            progress(message)

    drafting = config.drafting
    shard = TensorParallelShard(0, 1)
    device = torch.device("cuda", config.device if config.devices is None else config.devices[0])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_qwen38_checkpoint(snapshot)
        draft_manifest = draft_config = None
        if drafting:
            draft_manifest = inspect_indexed_checkpoint(config.draft_checkpoint)
            draft_config = DraftConfig.load(config.draft_checkpoint)
    # Loading, construction and tuning free most of what they allocate; a transient package pool returns
    # that memory to the device once tuning completes, before compiled forwards are bound.
    if package_storage is not None:
        transient.enter_context(package_storage.transient())
    weights = ModelWeights.load(manifest, device, progress=progress, shard=shard)
    if torch.cuda.get_device_properties(device).is_integrated:
        from ignition.preparation.host_memory import release_page_cache

        release_page_cache(manifest.shard_paths)
    draft_weights = None
    if drafting:
        draft_weights = DraftWeights.load(config.draft_checkpoint, draft_config, device, progress=progress)
    host_storage = {} if package_storage is None else {"host_allocator": package_storage.host_allocate}
    table = PLETable.load(manifest, device, tokens=config.token_capacity, progress=progress, **host_storage)
    sample_rows = max(batch * width for batch, width in config.speculative_shapes) if drafting else config.request_capacity
    collectives = dict(runtime=None, vocabulary_runtime=None, coordinator=None)
    session = preparation_session(
        compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
        native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
        device=device, checkpoint=REVISION, coordinator=None,
        namespace={"config": manifest.metadata_sha256["config.json"], "headers": manifest.headers_sha256, "world_size": 1,
                   "shard_hyperconnections": False},
        headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
    session.__enter__()
    graphs = {}
    try:
        with phase(progress, "Allocate model state"):
            speculative_tokens = SPECULATIVE_TOKENS if drafting else 0
            layout = state_layout(drafting)
            snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
            slots = config.request_capacity
            # Verification commits into each request's live row, so a request holds one row.
            state_rows = snapshots + slots
            state = HybridStatePool.allocate(state_rows + 1, device,
                speculative_tokens=speculative_tokens, world_size=1, mtp=False,
                draft_windows=DRAFT_WINDOWS if drafting else None, storage=package_storage,
                backed_slots=(min(state_rows + 1, config.preparation_state_rows) if package_storage is not None else None))
            if state.layout != layout:
                raise ValueError("the allocated hybrid row differs from the configured layout")
            register_benchmark_state(
                session, next(iter(state.gdn.values())).recurrent,
                storage=state.backing, restore=state.reset_for_preparation,
            )
            requests = DeviceRequestState.allocate(device, capacity=slots)
            from ignition.components.sampling import PreparedSampling

            sampling_workspace = bind_scratch(PreparedSampling.scratch_layout(slots, requests.vocab_size), device=device)
            inputs = DecoderInputs.allocate(device, context=config.context, tokens=config.token_capacity, capacity=slots)
            graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            scratch = DecoderScratch.allocate(config.token_capacity, device, sample_rows=sample_rows,
                world_size=1, buffers=graph_buffers.cursor(), hyperconnection_lowrank=320)
            workspaces = ScratchArena(device, overlap=config.stream_overlap)
            rotary = RotaryTables.allocate(weights.tree.config.text_config, config.context, device,
                scaling=config.rope_scaling)
            cache = {index: QSAState.allocate(pages=1, slots=slots + 1, device=device, draft_tokens=speculative_tokens, kv_heads=2)
                     for index in range(3, 48, 4)}
            page_bytes = sum(sum(getattr(value, name).nbytes for name in
                ("main_k_cache", "main_v_cache", "compressed_k_cache")) for value in cache.values())
            planned_pages = torch.cuda.mem_get_info(device)[1] // page_bytes
            admission = AdmissionBuffers.allocate(device, vocab_size=248320, capacity=slots)
            transfer = StateTransferMetadata.allocate(slots, device)
            frame = torch.zeros(1, dtype=torch.int32, device=device)
            sequence = torch.zeros(1, dtype=torch.int64, device=device)
            table.sequence = sequence
            from ignition.components.readiness import ReadinessTimings
            table.timings = ReadinessTimings.allocate(frame, 1)
            completion = CompletionOutputs.allocate((2, slots, 24), device=device)
            features = VisionFeatureState.allocate(device, capacity=slots, image_slots=config.image_concurrency,
                feature_slot=inputs.vision_feature_slot)
            vision_inputs = VisionInputs.allocate(device)
            vision_scratch = {}
            buffers = {**inputs.buffers(), **table.buffers(), **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
                **{f"state_{name}": getattr(transfer, name) for name in transfer.__dataclass_fields__},
                **{name: getattr(vision_inputs, name) for name in vision_inputs.__dataclass_fields__},
                "frame_index": frame, "submission_sequence": sequence, **completion.buffers(),
                "sampling_grammar_mask": requests.grammar_mask, "state_pool": state.backing, "memory_capacity": state.memory_capacity}
            functions, signatures, decoders, owners = {}, {}, [], []
            speculation = None
            if drafting:
                from .speculative_model import SpeculativePreparation

                speculation = SpeculativePreparation(target_weights=weights, draft_weights=draft_weights,
                    draft_config=draft_config, pool=state, requests=requests, inputs=inputs, scratch=scratch,
                    workspace=workspaces, rotary=rotary, cache=cache, table=table, session=session,
                    completion=completion, frame=frame, sequence=sequence, max_cache_pages=planned_pages,
                    verification_width=max(config.speculative_widths), **collectives)
                buffers.update(speculation.buffers())
        scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=None, ipc=None,
            runtime=None, coordinator=None)
        serving_modes = (True, False) if drafting else (False,)
        graph_count = (4 + len(serving_modes) * (prefill_graph_count(config.prefill_capacities, scoring)
                                              + 4 * len(config.batch_capacities))
                       + len(config.batch_capacities) + (len(config.speculative_shapes) if drafting else 0))
        with phase(progress, "Construct graph catalog", total=graph_count) as stage:
            def add(name, function, kind, batch, tokens, fields, *, enabled=None):
                if kind == "scored_prefill":
                    function = PreparedScoredPrefill(function, requests, inputs)
                    fields = (*fields, "prompt_target_tokens")
                functions[name] = function
                signatures[name] = GraphSignature(kind, batch, tokens, tuple(fields), speculation_enabled=enabled)
                stage.update(advance=1)

            vision_fields = (*vision_inputs.__dataclass_fields__, "vision_feature_slot")
            for capacity in (16384, 4096, 1024, 256):
                cursor = graph_buffers.cursor()
                s = vision_scratch[capacity] = VisionScratch.allocate(capacity, device, buffers=cursor)
                handoff = PreparedVisionHandoff(vision_inputs, features, requests, s.output, completion, frame, sequence,
                                               pixels=s.patches, buffers=cursor)
                handoff.prepare()
                with torch.device("meta"):
                    module = Qwen4ExpVisionModel(weights.tree.config.vision_config)
                module.eval().requires_grad_(False)
                encoder = PreparedVision(module, s, handoff.metadata, weights.vision, session=session,
                    name=f"vision{capacity}", projection_workspace=workspaces,
                    attention_workspace=workspaces)
                add(f"vision{capacity}", PreparedVisionGeneration(handoff, encoder), "vision", 1, capacity, vision_fields)
                owners.extend((handoff, encoder))

            forward_fields = (*inputs.buffers(), "sampling_grammar_mask")
            for enabled in serving_modes:
                suffix = "-no-mtp" if drafting and not enabled else ""
                modes = prefill_modes(config.prefill_capacities,
                    () if enabled else reversed(config.batch_capacities), scoring)
                for kind, mode, capacities in modes:
                    for capacity in capacities:
                        rows = 1 if mode == "prefill" else capacity
                        name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                        metadata = PreparedDecoderState(inputs, requests, state, tokens=capacity, rows=rows, mode=mode,
                            maintain_draft_state=False)
                        s = scratch.rows(capacity, rows)
                        injection = None
                        if mode == "prefill":
                            injection = PreparedEmbeddingInjection(features, requests, inputs.vision_token_indices[:capacity],
                                inputs.row_slot_ids, inputs.row_generations, inputs.query_lengths, inputs.live_rows,
                                s.embedding, rows=rows)
                            injection.prepare()
                        # The tapped feature buffer and the context's scratch share one of the owner's
                        # cursors: both live from the decoder's layers through the ring writes, and the
                        # owner's pool never aliases the decoder scratch drawn from ``graph_buffers``.
                        cursor = speculation.cursor() if enabled else None
                        taps = dict(feature_taps=speculation.feature_taps,
                                    auxiliary=speculation.auxiliary(capacity, cursor)) if enabled else {}
                        decoder = PreparedDecoder(weights, s, metadata, cache, rotary, table, workspaces,
                            completion, frame, sequence, session=session, name=name,
                            embedding_injection=injection, max_cache_pages=planned_pages,
                            sampling_workspace=sampling_workspace,
                            scoring=scoring if kind == "scored_prefill" else None, **taps, **collectives)
                        forward = speculation.prefill(decoder, name, buffers=cursor) if enabled else decoder
                        add(name, forward, kind, rows,
                            capacity if mode == "prefill" else 1,
                            forward_fields, enabled=enabled)
                        decoders.append(decoder)

            if speculation is not None:
                fields = (*forward_fields, *speculation.buffers())
                for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                    name = f"speculative{batch}x{width}"
                    operation = speculation.round(batch, width, name)
                    operation.prepare()
                    add(name, operation, "speculative", batch, width, fields, enabled=True)
                decoders.extend(speculation.decoders)
                owners.append(speculation)

            state_fields = tuple(f"state_{name}" for name in transfer.__dataclass_fields__ if name != "errors")
            admission_fields = tuple(admission.__dataclass_fields__)
            for enabled in serving_modes:
                suffix = "-no-mtp" if drafting and not enabled else ""
                for capacity in reversed(config.batch_capacities):
                    name = f"resume{capacity}" + suffix
                    resume = PreparedResume(inputs, requests, state, scratch.sample_hidden[:capacity],
                        scratch.logits[:capacity], weights.tree.lm_head.weight, completion, frame, sequence,
                        session=session, name=name, vocabulary_runtime=None,
                        local_logits=None if scratch.local_logits is None else scratch.local_logits[:capacity],
                        coordinator=None, sampling_workspace=sampling_workspace)
                    add(name, resume, "resume", capacity, 1, forward_fields, enabled=enabled)
                    owners.append(resume)
                    for mode in ("init", "restore", "snapshot"):
                        maintenance = PreparedMaintenance(mode, requests, state, transfer, completion, frame, sequence,
                            admission_buffers=admission, forward_inputs=inputs,
                            qsa_states=tuple(cache.values()), capacity=capacity,
                            speculation=speculation if enabled else None)
                        maintenance.prepare()
                        fields = state_fields + (forward_fields if mode == "snapshot" else ("request_state_rows", *admission_fields))
                        if mode == "restore":
                            fields += ("page_tables",)
                        add(f"{mode}{capacity}" + suffix, maintenance, mode, capacity, 1, fields, enabled=enabled)
                        owners.append(maintenance)
        session.register_scratch(workspaces)
        with phase(progress, "Prepare and tune kernels") as stage, transient:
            tuning = session.tune(show_progress=progress is not None, console=stage.console)
        report(f"selected prepared kernels: {tuning['selection_counts']}")
        stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle() if package_storage is None else package_storage.graph_pool.id
        compiled_functions = {}
        def capture_catalog(*, compile_decoder, materialize=None, label="Prepare serving graphs"):
            captured, compiled = capture_forwards(functions, signatures, session=session, device=device,
                stream=stream, pool=graph_pool, compile_decoder=compile_decoder, materialize=materialize,
                progress=progress, label=label, cache_root=config.kernel_cache,
                collective=None, coordinator=None, retain_graphs=config.retain_graphs,
                warmup_storage=package_storage)
            graphs.update(captured)
            compiled_functions.clear()
            compiled_functions.update(compiled)

        page_capacity = config.preparation_pages

        def rebind_cache(pages):
            for decoder in decoders:
                for index in range(3, 48, 4):
                    decoder.layers[index][1].rebind_pages_for_preparation(pages[index])
                decoder.qsa_states = pages
            if speculation is not None:
                speculation.rebind_pages_for_preparation(pages)

        if compile_decoder:
            from b12x._lib.compile_plan import compile_only_launches

            mode = deferred_mode()
            with compile_only_launches():
                rebind_cache({index: value.resize_pages_for_preparation(planned_pages if package_storage is not None else page_capacity, metadata_mode=mode)
                              for index, value in cache.items()})

        final_cache = None
        def materialize_cache():
            nonlocal final_cache
            with phase(progress, "Allocate preparation KV cache"):
                free = device_memory_info(device)[0]
                if page_capacity * page_bytes + config.headroom_bytes > free:
                    raise MemoryError("insufficient memory for preparation KV pages and configured headroom")
                if package_storage is not None:
                    final_cache = {index: value.resize_pages_for_preparation(planned_pages, storage=package_storage,
                        prefix=f"kv-{index}-", backed_pages=page_capacity) for index, value in cache.items()}
                else:
                    final_cache = {index: value.resize_pages_for_preparation(page_capacity)
                                   for index, value in cache.items()}
                report(f"{page_capacity * page_bytes / (1 << 20):.1f} MiB per GPU · {page_capacity:,} pages")
                rebind_cache(final_cache)
                for owner in owners:
                    if isinstance(owner, PreparedMaintenance):
                        owner.rebind_pages_for_preparation(final_cache)
                state.memory_capacity[1] = page_capacity

        capture_catalog(compile_decoder=compile_decoder, materialize=materialize_cache)
        session.freeze()
        free, total = device_memory_info(device)
        memory = {"weights_bytes": weights.allocated_bytes, "ngram_request_bytes": table.mailbox.host_view.nbytes,
            "hybrid_state_bytes": state.backing.nbytes, "state_rows": state_rows,
            "preparation_state_bytes": state.backed_slots * layout.row_bytes,
            "preparation_state_rows": state.backed_slots,
            "pages": page_capacity, "page_bytes": page_bytes, "kv_bytes": page_capacity * page_bytes,
            "cached_token_capacity": page_capacity * 64, "headroom_bytes": config.headroom_bytes, "page_tokens": 64,
            "free_device_bytes": free, "total_device_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": workspaces.nbytes, "graphs": len(graphs), "tuning": tuning,
            "prefill_chunk_size": config.prefill_chunk_size, "prefill_capacities": config.prefill_capacities,
            "mixed_shapes": (), "mixed_prefill": False,
            "compiled_decoder": compile_decoder, "rank": 0, "world_size": 1, "shard_hyperconnections": False}
        if scoring is not None:
            memory.update(scoring.memory_report())
        if speculation is not None:
            memory.update(draft_weights_bytes=sum(tensor.nbytes for tensor in draft_manifest.tensors.values()),
                speculative_widths=config.speculative_widths,
                speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
                speculative_shapes=config.speculative_shapes,
                speculative_state_columns=8, draft_raw_ring_capacity=layout.draft_ring_capacity,
                draft_window_tokens=draft_config.window,
                speculative_fixed_width=config.speculative_fixed_width)
        hashes = {"tokenizer": ("tokenizer.json", FRONTEND_SHA256["tokenizer.json"])}
        cache_regions = [history_region(buffers, f"cache.{layer}.{name}", getattr(value, name), contract_proto.PAGED_KEY_VALUE,
            tokens_per_record=4 if name == "compressed_k_cache" else 1)
            for layer, value in (final_cache or cache).items()
            for name in ("main_k_cache", "main_v_cache", "compressed_k_cache")]
        buffers["request_pending_token"] = requests.pending_token
        server_config = {"model_id": MODEL_ID, "identity": _identity(manifest, draft_manifest, layout, config.rope_scaling),
            "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
            "pages": page_capacity, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": 64,
            "frontend": prepare_frontend(manifest.snapshot, eos_token_ids=[248046, 248044], image_profile="qwen"),
            "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
            "model_contract": prepare_contract(signatures, buffers, state, image_slots=config.image_concurrency, page_tokens=64, draft_lag=0 if drafting else None, embedding_tables=(0,), regions=cache_regions),
            "assets": {name: {"path": str((manifest.snapshot / filename).resolve()), "sha256": digest}
                       for name, (filename, digest) in hashes.items()}}
        server_config["assets"]["chat_template"] = annotated_template_asset(
            (manifest.snapshot / "chat_template.jinja").read_bytes(),
            Path(__file__).parents[1] / "qwen38" / "assets" / "chat_template.jinja")
        if speculation is not None:
            server_config.update(speculative_widths=config.speculative_widths,
                speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
                speculative_fixed_width=config.speculative_fixed_width)
        resources = (scoring, weights, table, session, state, requests, inputs, scratch, workspaces, rotary,
            cache, final_cache, admission, transfer, features, vision_inputs, vision_scratch, graph_buffers,
            functions, compiled_functions, decoders, owners, stream, graph_pool, speculation, draft_weights)
        report(f"prepared {len(graphs)} graphs with {page_capacity} preparation KV pages")
        export_zero_ranges = (*disposable_ranges(cache=final_cache, scratch=scratch, workspaces=workspaces,
                                                 vision_scratch=vision_scratch, decoders=decoders, owners=owners,
                                                 speculation=speculation),
                              *storage_ranges(graph_buffers.chunks))
        return PreparedModel(rebuild_weights=declare_weights({'weights': weights, 'draft': draft_weights}, components={
        'tree.model.language_model.embed_tokens': 'embedding',
        'tree.model.language_model.hyper_connection_mixer': 'hyper-connection',
        'tree.lm_head': 'head',
        'vision': 'vision',
        'draft': 'draft',
    }, layers=('layers', 'tree.model.language_model.layers')), device=device.index, max_active=config.max_active, graphs=graphs, buffers=buffers, signatures=signatures,
            export_zero_ranges=(*export_zero_ranges, *(() if scoring is None else scoring.zero_ranges()), *completion.zero_ranges(),
                                *((t.data_ptr(), t.nbytes) for t in table.buffers().values())),
            export_events=table.events(), package_data=table,
            resources=resources, server_config=server_config, memory_report=memory, session=session,
            close_resources=(table, *((scoring,) if scoring is not None else ())))
    except BaseException:
        torch.cuda.synchronize(device)
        for graph in graphs.values():
            graph.reset()
        session.close()
        table.close()
        raise

