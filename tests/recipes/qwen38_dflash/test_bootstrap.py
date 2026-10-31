"""The DFlash2 preparation config, its identity, and the contract its catalog declares."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation import runtime_package_pb2 as p
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
    (dict(shard_hyperconnections=True, devices=(0, 1)), "single GPU"),
    (dict(mixed_prefill=True), "no mixed prefill"), (dict(draft_checkpoint="/draft"), "Path"),
])
def test_config_rejects_sharding_mixed_rounds_and_untyped_drafts(arguments, message):
    with pytest.raises((ValueError, TypeError), match=message):
        PreparationConfig(**arguments)


def test_identity_differs_from_qwen38_and_tracks_the_draft():
    from ignition.recipes.declaration import RopeScaling

    manifest = SimpleNamespace(headers_sha256="headers", metadata_sha256={"config.json": "config"})
    draft = SimpleNamespace(shards=(SimpleNamespace(header_sha256="draft-a"), SimpleNamespace(header_sha256="draft-b")))
    plain, speculative = state_layout(False), state_layout(True)
    identities = {
        "qwen": qwen_identity(manifest),
        "plain": _identity(manifest, None, plain),
        "draft": _identity(manifest, draft, speculative),
        "other-draft": _identity(manifest, SimpleNamespace(shards=(SimpleNamespace(header_sha256="x"),)), speculative),
        "scaled": _identity(manifest, draft, speculative, rope_scaling=RopeScaling(4., 262144)),
    }
    assert len(set(identities.values())) == len(identities)
    assert _identity(manifest, draft, speculative) == identities["draft"]


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
