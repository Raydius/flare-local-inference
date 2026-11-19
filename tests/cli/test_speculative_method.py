# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned proposal selection, configuration precedence and physical planning."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.cli import _parser, display, help as cli_help, preparation
from ignition.cli.config import Configuration
from ignition.errors import UsageError
from ignition.recipes import resolve
from ignition.recipes.declaration import SpeculativeMethod, Speculation
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto


def selected(*flags, model="qwen3.8-flash", configuration=None):
    args = _parser().parse_args(["export", model, *flags])
    return preparation.speculation(args, resolve(model), configuration)


def test_flash_default_mtp_and_explicit_dflash_preserve_requested_capacities():
    default = selected()
    assert default.method == "mtp" and default.draft_tokens == 5
    dflash = selected("--speculative-method", "dflash2", "--mtp-draft-tokens", "7",
                      "--speculative-token-capacity", "128")
    assert dflash.method == "dflash2"
    assert dflash.widths == tuple(range(1, 9)) and dflash.token_capacity == 128
    assert selected("--speculative-method", "mtp") == default
    assert selected("--speculative-method", "dflash2", "--mtp-draft-tokens", "0").widths == (1,)


def test_method_configuration_and_explicit_flags(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[defaults]\nspeculative_method = "dflash2"\n')
    configuration = Configuration.load(path)
    assert selected(configuration=configuration).method == "dflash2"
    assert selected("--speculative-method", "mtp", configuration=configuration).method == "mtp"
    assert selected("--no-mtp", configuration=configuration) is None
    path.write_text('[defaults]\nspeculative_method = "dflash2"\n[models."qwen3.8-flash"]\nspeculative_method = "mtp"\n')
    assert selected(configuration=Configuration.load(path)).method == "mtp"
    with pytest.raises(UsageError, match="cannot accompany"):
        selected("--no-mtp", "--speculative-method", "dflash2")
    with pytest.raises(UsageError, match="does not prepare"):
        selected("--speculative-method", "mtp", model="qwen3.8-27b")


def test_method_costs_and_rebuild_metadata(monkeypatch):
    model = resolve("qwen3.8-flash")
    gpu = planning.Gpu(0, "GB10", "GPU-test", 128 << 30, 127 << 30, (12, 1))
    def plan(method):
        return planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,),
            concurrency=4, reserve_bytes=0, speculation=selected("--speculative-method", method))
    mtp, dflash = plan("mtp"), plan("dflash2")
    assert dflash.costs.state_row_bytes == 142_821_120
    assert dflash.costs.page_bytes == 12 * 69_632
    assert mtp.costs.page_bytes == 13 * 69_632
    assert dflash.weights_bytes - mtp.weights_bytes == 1_206_140_008 - model.checkpoint.draft_bytes
    assert replace(dflash, force_moe_decode_a16=True).force_moe_decode_a16
    assert dflash.to_dict()["speculation"]["method"] == "dflash2"
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), dflash, compiled_forwards=True)
    encoded = json_format.ParseDict(manifest, proto.RebuildManifest())
    decoded = proto.RebuildManifest.FromString(encoded.SerializeToString())
    assert decoded.speculation.method == "dflash2"
    assert decoded.speculation.draft_tokens == 5 and decoded.speculation.token_capacity == 128
    with pytest.raises(UsageError, match="separate prefill"):
        planning.plan(model.recipe, model.checkpoint, (gpu,), selected=(0,), concurrency=4,
                      reserve_bytes=0, speculation=dflash.speculation, mixed_prefill=True)


def with_dspark(model="qwen3.8-27b"):
    """A resolved model whose recipe also declares a `dspark` method limited to seven draft tokens."""
    resolved = resolve(model)
    recipe = resolved.recipe
    method = SpeculativeMethod(ranks=recipe.ranks, max_draft_tokens=7)
    return replace(resolved, recipe=replace(recipe, speculative_methods={**recipe.speculative_methods, "dspark": method}))


def test_a_method_limits_its_own_draft_tokens_beside_the_recipe_limit(monkeypatch):
    model = with_dspark()
    assert model.recipe.max_draft_tokens == 15
    def chosen(*flags):
        return preparation.speculation(_parser().parse_args(["export", "qwen3.8-27b", *flags]), model)
    dspark = chosen("--speculative-method", "dspark", "--mtp-draft-tokens", "7")
    assert dspark.method == "dspark" and dspark.widths == tuple(range(1, 9))
    with pytest.raises(UsageError, match="--mtp-draft-tokens must be 0 through 7; --no-mtp disables speculation"):
        chosen("--speculative-method", "dspark", "--mtp-draft-tokens", "8")
    assert chosen("--mtp-draft-tokens", "15").method == "dflash2"
    with pytest.raises(UsageError, match="--mtp-draft-tokens must be 0 through 15; --no-mtp disables speculation"):
        chosen("--mtp-draft-tokens", "16")
    assert model.recipe.draft_token_limit("dspark") == 7
    assert model.recipe.draft_token_limit("dflash2") == 15
    monkeypatch.setattr(cli_help, "recipes", lambda: (model.recipe,))
    assert "--mtp-draft-tokens: 0..7; default 7" in cli_help.recipe_help("qwen3.8-27b", "dspark")
    assert "--mtp-draft-tokens: 0..15; default 7" in cli_help.recipe_help("qwen3.8-27b")
    assert "--speculative-method: dflash2, dspark; default dflash2" in cli_help.recipe_help("qwen3.8-27b")


def test_a_recipe_without_the_method_refuses_dspark():
    with pytest.raises(UsageError, match="--speculative-method dspark: qwen3.8-flash does not prepare this proposal method"):
        selected("--speculative-method", "dspark")
    with pytest.raises(ValueError, match="mtp, dflash2 or dspark"):
        Speculation(draft_tokens=1, method="dsparkle")


def test_dspark_is_named_in_the_plan_report_the_startup_line_and_the_configuration_template(tmp_path):
    dspark = Speculation(draft_tokens=7, method="dspark")
    assert display.speculation(dspark) == "up to 7 draft tokens per round · DSpark"
    assert display.proposals(dspark) == " · DSpark proposals"
    assert display.proposals(replace(dspark, method="dflash2")) == " · DFlash2 proposals"
    assert display.speculation(replace(dspark, method="dflash2")) == "up to 7 draft tokens per round · DFlash2"
    assert display.proposals(replace(dspark, method="mtp")) == "" and display.proposals(None) == ""
    # Kimi K3.1 Pro's default method has no name and stays the MTP module.
    assert display.speculation(replace(dspark, method=None)) == "up to 7 draft tokens per round · MTP module"
    configuration = Configuration(tmp_path / "config.toml", {}, exists=False)
    configuration.write_initial("qwen3.8-27b", 8)
    assert ('# speculative_method = "mtp"    # proposal method; Qwen Flash also supports dflash2,'
            " Qwen3.8 27B also supports dspark") in (tmp_path / "config.toml").read_text()
