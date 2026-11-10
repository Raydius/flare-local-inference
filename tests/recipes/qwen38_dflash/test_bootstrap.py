"""The DFlash2 preparation config, its identity, and the contract its catalog declares."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.qwen38.bootstrap import PreparationConfig as QwenPreparationConfig, _identity as qwen_identity
from ignition.recipes.qwen38.contract import prepare as prepare_contract
from ignition.recipes.qwen38.state import HybridStateLayout
from ignition.recipes.qwen38_dflash.bootstrap import PreparationConfig, _identity, state_layout
from ignition.recipes.qwen38_dflash.recipe import DRAFT_WINDOWS, RECIPE

DRAFT = Path("/draft")


def test_snapshot_floor_is_the_allocated_dflash_layout():
    speculative = HybridStateLayout.build(speculative_tokens=7, mtp=False, draft_windows=DRAFT_WINDOWS)
    assert state_layout(True) == speculative and state_layout(False) == HybridStateLayout.build()
    assert speculative.row_bytes == RECIPE.ranks[1].state_row_bytes
    # qwen38's MTP floor accepts a budget the DFlash row does not fit.
    QwenPreparationConfig(mtp=True, snapshot_budget_bytes=speculative.row_bytes - 1)
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(draft_checkpoint=DRAFT, snapshot_budget_bytes=speculative.row_bytes - 1)
    config = PreparationConfig(draft_checkpoint=DRAFT, snapshot_budget_bytes=speculative.row_bytes)
    assert config.drafting and config.mtp and config.speculative_shapes
    assert config.token_capacity == max(config.prefill_chunk_size, max(b * w for b, w in config.speculative_shapes))
    plain = PreparationConfig(snapshot_budget_bytes=state_layout(False).row_bytes)
    assert not plain.drafting and not plain.mtp and plain.mixed_shapes == ()
    with pytest.raises(ValueError, match="at least one state row"):
        PreparationConfig(snapshot_budget_bytes=state_layout(False).row_bytes - 1)


@pytest.mark.parametrize("arguments,message", [
    (dict(devices=(0, 1)), "single GPU"), (dict(hosts=("a",), devices=(0,)), "single GPU"),
    (dict(shard_hyperconnections=True, devices=(0, 1)), "single GPU"), (dict(draft_checkpoint="/draft"), "Path"),
])
def test_config_rejects_sharding_and_untyped_drafts(arguments, message):
    with pytest.raises((ValueError, TypeError), match=message):
        PreparationConfig(**arguments)


@pytest.mark.parametrize("mixed", [False, True])
def test_plan_and_preparation_use_identical_mixed_catalogs(mixed, monkeypatch):
    """The DFlash plan's mixed shapes are qwen38's, in draft-enabled and draft-disabled modes."""
    from ignition import planning
    from ignition.recipes.declaration import Speculation
    from ignition.recipes.qwen38.recipe import RECIPE as QWEN
    from ignition.recipes.qwen38_dflash import bootstrap, recipe

    gpu = planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-test", 96 << 30, 95 << 30, (12, 0))
    monkeypatch.setattr(bootstrap, "prepare_model", lambda snapshot, config, **kwargs: config)
    monkeypatch.setattr(recipe, "snapshot_download", lambda *args, **kwargs: "/draft", raising=False)
    for speculation in (None, Speculation(7, token_capacity=128), Speculation(3, token_capacity=128, adaptive=False)):
        plan = planning.plan(RECIPE, RECIPE.checkpoints[0], (gpu,), selected=(0,),
            concurrency=4, reserve_bytes=0, speculation=speculation, mixed_prefill=mixed)
        config = RECIPE.prepare(Path("/checkpoint"), plan, compile_forwards=True, kernel_cache=None, progress=None)
        expected = RECIPE.mixed_shapes(4, 4096, speculation) if mixed else ()
        assert config.mixed_prefill == plan.to_dict()["mixed_prefill"] == mixed
        assert config.mixed_shapes == plan.mixed_shapes == expected
        assert config.drafting == config.mtp == (speculation is not None)
        assert expected == QWEN.mixed_shapes(4, 4096, speculation if mixed else None) if mixed else expected == ()
        assert all(p + (b - 1) * w <= config.token_capacity for _, b, w, p in config.mixed_shapes)
    assert RECIPE.mixed_prefill and PreparationConfig(draft_checkpoint=DRAFT, mixed_prefill=True).mixed_shapes
    assert PreparationConfig(max_active=1, draft_checkpoint=DRAFT, mixed_prefill=True).mixed_shapes == ()


def test_identity_differs_from_qwen38_and_tracks_the_draft():
    from ignition.recipes.declaration import RopeScaling

    manifest = SimpleNamespace(headers_sha256="headers", metadata_sha256={"config.json": "config"})
    draft = SimpleNamespace(shards=(SimpleNamespace(header_sha256="draft-a"), SimpleNamespace(header_sha256="draft-b")))
    plain, speculative = state_layout(False), state_layout(True)
    native, scaled = PreparationConfig(), PreparationConfig(context=1_048_576, rope_scaling=RopeScaling(4., 262144))
    identities = {
        "qwen": qwen_identity(manifest),
        "plain": _identity(manifest, None, plain, native),
        "draft": _identity(manifest, draft, speculative, native),
        "other-draft": _identity(manifest, SimpleNamespace(shards=(SimpleNamespace(header_sha256="x"),)), speculative, native),
        "scaled": _identity(manifest, draft, speculative, scaled),
    }
    assert len(set(identities.values())) == len(identities)
    assert _identity(manifest, draft, speculative, native) == identities["draft"]


def test_contract_declares_copy_inputs_and_lag_zero_without_draft_page_tables():
    from ignition.components.parallel_draft import allocate_copy_buffers
    from ignition.preparation import contract as c

    copies = allocate_copy_buffers(2, "cpu")
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32, region=lambda name: SimpleNamespace(offset=24, nbytes=8)))
    # The catalog's fields: the decoder inputs plus the grammar mask; speculative rounds add the four copies.
    from ignition.recipes.qwen38.decoder_state import DecoderInputs

    forward = (*DecoderInputs.__dataclass_fields__, "sampling_grammar_mask")
    signatures = {"speculative2x4": GraphSignature("speculative", 2, 4, (*forward, *copies), speculation_enabled=True),
                  "decode2": GraphSignature("decode", 2, 1, forward)}
    assert "draft_page_tables" not in forward and "draft_page_tables" in c.text_inputs("decode")
    contract = prepare_contract(signatures, {"page_tables": torch.zeros(1), **copies}, pool, page_tokens=64,
                                draft_lag=0, embedding_tables=(0,))
    graphs = {graph["graph"]: graph for graph in contract["graphs"]}
    bound = {b["buffer"]: b["source"] for b in graphs["speculative2x4"]["inputs"]}
    assert {name: bound[name] for name in copies} == {
        "copy_token_ids": p.COPY_TOKEN_IDS, "copy_token_count": p.COPY_TOKEN_COUNT,
        "copy_expected_consumed": p.COPY_CONSUMED_TOKENS, "copy_generation": p.COPY_GENERATION}
    assert graphs["speculative2x4"]["draft_cursor"]["lag_tokens"] == 0
    assert not any(b["buffer"] in copies for b in graphs["decode2"]["inputs"])
    assert [table["buffer"] for table in contract["cache"]["page_tables"]] == ["page_tables"]
