"""Qualified checkpoint, serving range, and memory costs of the Qwen 3.8 DFlash2 recipe."""

from pathlib import Path
from types import MappingProxyType

from ignition.errors import UsageError
from ignition.preparation.qwen38 import ARCHITECTURE
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation
from ignition.recipes.qwen38.recipe import _GRAPH_BYTES, _costs

from .config import DRAFT_REPOSITORY, DRAFT_REVISION

# The DFlash2 draft's stored tensors, loaded in place of the target's MTP module.
_DRAFT_BYTES = 1_206_131_200
# The target the acceptance spike measured: the same Qwen3.8-Flash-Next geometry and tensor
# inventory as qwen38's pin, with QAD-calibrated expert block scales. Bytes come from its
# model.safetensors.index.json; the declaration test checks the stored total against it.
_CHECKPOINT = Checkpoint(
    alias="qwen3.8-flash-dflash2", repository="local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD",
    revision="dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2", architecture=ARCHITECTURE,
    stored_bytes=102_138_950_916,
    # Decoder and vision tensors plus the DFlash2 draft: the stored total less the 28,800,138,244
    # PLE n-gram bytes delivered through compact inputs and the 1,596,720,640 MTP bytes this
    # recipe never loads, plus the draft.
    device_bytes=72_948_223_232,
    draft_bytes=_DRAFT_BYTES,
    # Routed-expert block scales expand at load from 3,965,911,336 CSF bytes to
    # 48 layers x 512 experts x 307,200 bytes.
    expansion_bytes=3_583_835_864)

# Five draft layers each keep a 2048-position window of 8 E4M3 key/value heads of 128 per state row:
# `HybridStateLayout.build(speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS).row_bytes`,
# which the declaration test checks; declarations import without torch.
DRAFT_WINDOWS = (5, 2048, 8, 128)
_STATE_ROW_BYTES = 142_821_120
# The [4096, 12800] BF16 feature buffer the prefill decoder taps for the draft context.
_AUXILIARY_BYTES = 4096 * 12800 * 2


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, package_storage=None,
            ranks=None):
    """Prepare the plan's catalog with DFlash2 drafts on local GPU 0; the draft is unsharded, so TP1 only."""
    if plan.world_size != 1 or ranks is not None:
        raise UsageError("qwen3.8-flash-dflash2 prepares on a single GPU: the DFlash2 draft is not sharded")
    from .bootstrap import PreparationConfig, prepare_model

    speculation = plan.speculation
    drafting = {}
    if speculation is not None:
        from huggingface_hub import snapshot_download

        draft = snapshot_download(DRAFT_REPOSITORY, revision=DRAFT_REVISION, cache_dir=plan.hub_cache,
                                  allow_patterns=("config.json", "model.safetensors"))
        drafting = dict(draft_checkpoint=Path(draft), speculative_widths=speculation.widths,
            speculative_token_capacity=plan.speculative_token_capacity, speculative_fixed_width=speculation.fixed_width)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=(0,), compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency, stream_overlap=plan.stream_overlap,
        context=plan.declared_context, prefill_chunk_size=plan.prefill_chunk, mixed_prefill=plan.mixed_prefill,
        rope_scaling=plan.checkpoint.rope_scaling,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache, **drafting)
    storage = {} if package_storage is None else {"package_storage": package_storage}
    return prepare_model(snapshot, config, progress=progress, compile_decoder=compile_forwards, **storage)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="qwen38_dflash", served_model_name="Qwen3.8-Flash-Next", prepare=prepare, usage=usage,
    # Thinking defaults; bounds span thinking/instruct presets: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=20,
        ranges={"temperature": SamplingRange(.7, 1.), "top_p": SamplingRange(.8, .95),
                "presence_penalty": SamplingRange(0., 2.)}),
    checkpoints=(_CHECKPOINT,),
    context=262144, page_tokens=64, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    # Rounds adapt among zero through seven DFlash2 drafts; verification commits accepted tokens
    # into the live state row, so it keeps no extra rows.
    speculation=Speculation(draft_tokens=7, token_capacity=128), max_draft_tokens=7, verification_rows=0,
    stream_overlap=True,
    ranks=MappingProxyType({
        1: _costs(_STATE_ROW_BYTES, 69_632, _GRAPH_BYTES[1] + _AUXILIARY_BYTES, speculating=False),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(116_505_344, 69_632, _GRAPH_BYTES[1], speculating=False),
    }),
    max_images=4, image_bytes=32 << 20, mixed_prefill=False,
)
