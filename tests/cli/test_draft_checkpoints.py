# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External draft models share the configured target-checkpoint hub cache."""

from importlib import import_module
import sys
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli.config import Configuration, Setting
from ignition.cli.preparation import Host, derive
from ignition.recipes import ResolvedModel, recipes


@pytest.mark.parametrize("name,embedded", [("nemotron_h", False), ("kimi3", False), ("kimi3", True), ("muse_spark", False), ("qwen38_dense", False), ("muse_glimmer", False)])
@pytest.mark.parametrize("custom_cache", [False, True])
@pytest.mark.parametrize("drafting", [False, True])
def test_draft_preparation_selects_local_or_hub_snapshot(monkeypatch, tmp_path, name, custom_cache, drafting, embedded):
    import huggingface_hub

    monkeypatch.setenv("HOME", str(tmp_path))
    values = {"paths": {"hub_cache": str(tmp_path / "custom-hub")}} if custom_cache else {}
    configuration = Configuration(tmp_path / "config.toml", values, exists=True)
    hub_cache = configuration.directory("hub_cache")
    recipe = next(recipe for recipe in recipes() if recipe.name == name)
    model = ResolvedModel(recipe, recipe.checkpoints[0])
    world = recipe.tensor_parallel[0]
    gpus = tuple(planning.Gpu(i, "Blackwell", f"GPU-{i}", 256 << 30, 250 << 30, (12, 0))
                 for i in range(world))
    plan = derive(configuration, model, Host(gpus, None), gpus=f"0-{world - 1}",
                  concurrency=Setting(4, "test"), speculation=recipe.speculation if drafting else None)
    assert plan.hub_cache == hub_cache

    draft = import_module(f"ignition.recipes.{name}." + ("config" if name == "kimi3" else "draft_config"))
    repository = draft.DSPARK_REPOSITORY if name == "kimi3" else draft.REPOSITORY
    revision = draft.DSPARK_REVISION if name == "kimi3" else draft.REVISION
    snapshot = hub_cache / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    if drafting:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").touch()

    # Use the real hub reader offline: it must find this exact cached revision,
    # even when the library was imported before Flare's cache configuration.
    download = huggingface_hub.snapshot_download
    downloads = []

    def cached_download(repository, **options):
        downloads.append(repository)
        assert options["allow_patterns"] == ("config.json", "model.safetensors")
        return download(repository, **options, local_files_only=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached_download)

    def prepare_model(snapshot, config, **options):
        return config

    def preparation_config(**options):
        return SimpleNamespace(**(dict(draft_checkpoint=None) | options))

    monkeypatch.setitem(sys.modules, f"ignition.recipes.{name}.bootstrap", SimpleNamespace(
        PreparationConfig=preparation_config, prepare=prepare_model, prepare_rank=prepare_model))
    monkeypatch.setitem(sys.modules, "ignition.preparation.rank_group", SimpleNamespace(
        prepare_group=lambda snapshot, config, prepare, **options: config))
    target = tmp_path / "target"
    if embedded:
        snapshot = target / "dspark"
        snapshot.mkdir(parents=True)
    config = recipe.prepare(target, plan, compile_forwards=True, kernel_cache=None, progress=None)
    if name in ("muse_spark", "qwen38_dense", "muse_glimmer"):
        if name == "qwen38_dense":
            assert config.mtp is drafting
        else:
            assert config.dflash is drafting
        assert getattr(config, "draft_snapshot", None) == (snapshot if drafting else None)
        if drafting:
            assert config.speculative_widths == plan.speculation.widths
            assert config.speculative_fixed_width == plan.speculation.fixed_width
            assert config.speculative_token_capacity == {"qwen38_dense": 128, "muse_spark": 32,
                                                         "muse_glimmer": recipe.speculation.token_capacity}[name]
            assert config.speculative_graph_budget == 64
    else:
        assert config.draft_checkpoint == (str(snapshot) if drafting else None)
    assert downloads == ([repository] if drafting and not embedded else [])


def _dflash_plan(tmp_path, draft_model, *, speculation=True, name="qwen38_dflash"):
    configuration = Configuration(tmp_path / "config.toml", {}, exists=True)
    recipe = next(recipe for recipe in recipes() if recipe.name == name)
    gpus = (planning.Gpu(0, "Blackwell", "GPU-0", 256 << 30, 250 << 30, (12, 0)),)
    return recipe, derive(configuration, ResolvedModel(recipe, recipe.checkpoints[0]), Host(gpus, None), gpus="0",
                          concurrency=Setting(4, "test"), speculation=recipe.speculation if speculation else None,
                          draft_model=draft_model)


def _draft_folder(tmp_path, *names):
    folder = tmp_path / "draft"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"{}")
    return folder


@pytest.mark.parametrize("name,field", [("qwen38_dflash", "draft_checkpoint"), ("muse_glimmer", "draft_snapshot")])
def test_a_draft_folder_replaces_the_pinned_download(monkeypatch, tmp_path, name, field):
    import huggingface_hub

    monkeypatch.setenv("HOME", str(tmp_path))
    folder = _draft_folder(tmp_path, "config.json", "model.safetensors")
    recipe, plan = _dflash_plan(tmp_path, folder, name=name)
    assert plan.draft_model == folder

    def download(*args, **options):
        raise AssertionError("a draft folder must not be downloaded")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepare = lambda snapshot, config, **options: config
    monkeypatch.setitem(sys.modules, f"ignition.recipes.{name}.bootstrap", SimpleNamespace(
        PreparationConfig=lambda **options: SimpleNamespace(**options), prepare_model=prepare, prepare=prepare))
    config = recipe.prepare(tmp_path / "target", plan, compile_forwards=True, kernel_cache=None, progress=None)
    assert getattr(config, field) == folder


@pytest.mark.parametrize("name,files,speculation,message", [
    ("qwen38", ("config.json", "model.safetensors"), True, "qwen3.8-flash loads no external draft"),
    ("qwen38_dflash", ("config.json",), True, "lacks model.safetensors"),
    ("qwen38_dflash", ("config.json", "model.safetensors"), False, "requires speculation"),
    ("muse_glimmer", ("config.json",), True, "lacks model.safetensors"),
    ("muse_glimmer", ("config.json", "model.safetensors"), False, "requires speculation"),
])
def test_a_draft_folder_is_validated(monkeypatch, tmp_path, name, files, speculation, message):
    from ignition.errors import UsageError

    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(UsageError, match=message):
        _dflash_plan(tmp_path, _draft_folder(tmp_path, *files), speculation=speculation, name=name)


def test_recipe_and_draft_model_are_export_flags():
    from pathlib import Path

    from ignition.cli import _parser

    args = _parser().parse_args(["export", "/ckpt", "--recipe", "qwen38_dflash", "--draft-model", "/draft"])
    assert args.recipe_name == "qwen38_dflash" and args.draft_model == Path("/draft")
    with pytest.raises(SystemExit):
        _parser().parse_args(["debug-serve", "/ckpt", "--recipe", "qwen38_dflash"])


def test_multi_host_export_rejects_a_draft_folder(monkeypatch, tmp_path):
    from ignition.cli import _parser
    from ignition.cli.serve import export
    from ignition.errors import UsageError

    monkeypatch.setenv("HOME", str(tmp_path))
    args = _parser().parse_args(["export", str(tmp_path), "--ranks", "2", "--draft-model", str(tmp_path)])
    with pytest.raises(UsageError, match="--draft-model applies to single-host export"):
        export(args)
