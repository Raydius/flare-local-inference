# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify that a rebuild record's reproduction arguments carry every `ignition export` flag that changes the core."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from ignition import planning
from ignition.cli import _parser
from ignition.cli.preparation import rope_scaled
from ignition.declaration import Checkpoint, RankCosts, Recipe, RopeScaling, SamplingDefaults
from ignition.preparation import rebuild_manifest
from ignition.registry import ResolvedModel, resolve

GIB, MIB = 1 << 30, 1 << 20

_CHECKPOINT = dict(repository="lab/toy", revision="ab12cd34" + "0" * 32, architecture="ToyForCausalLM",
                   stored_bytes=GIB, device_bytes=GIB)
# Two aliases of one repository revision: the checkpoint's native context, and that context extended by rope scaling.
RECIPE = Recipe(
    name="toy", served_model_name="Toy",
    checkpoints=(Checkpoint(alias="toy", **_CHECKPOINT),
                 Checkpoint(alias="toy-yarn4", **_CHECKPOINT, context=1_048_576, rope_scaling=RopeScaling(4., 262144))),
    context=262144, page_tokens=64, concurrency=4, max_concurrency=8, max_draft_tokens=0, prefill_chunk=4096,
    ranks=MappingProxyType({1: RankCosts(MIB, MIB, GIB)}), prepare=None, usage=None,
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1))

# `ignition export` options that reproduction arguments omit. Each selects where the export reads or
# writes, who signs it, how many host resources preparation uses, or what the terminal shows; none
# changes the prepared model. `--collective-ladder` names a host-local file whose content the rebuild
# record stores in `collective_ladder_json`.
NOT_REPRODUCED = frozenset({
    "--help", "--config", "--verbose", "--no-weights", "--reuse-weights", "--publisher-key", "--repo-id",
    "--official-source", "--compile-workers", "--tuning-gpus", "--transfer-streams", "--rendezvous",
    "--collective-ladder"})


def _plan(checkpoint):
    gpu = planning.Gpu(0, "NVIDIA RTX PRO 6000 Blackwell", "GPU-0000", 96 * GIB, 95 * GIB, (12, 0))
    return planning.plan(RECIPE, checkpoint, (gpu,), selected=(0,), concurrency=4, reserve_bytes=0, speculation=None)


def _reproduction(model, plan=None):
    return rebuild_manifest.reproduction_arguments(model, plan or _plan(model.checkpoint), compiled_forwards=True)


def _reparsed(arguments):
    """Parse reproduction arguments as `ignition export` receives them."""
    assert arguments[:2] == ["ignition", "export"]
    return _parser(recipes=(RECIPE,)).parse_args(arguments[1:])


def test_rope_parameters_flag_is_recorded_with_the_value_given():
    value = '{"factor": 4.0, "original_max_position_embeddings": 262144}'
    given = _parser(recipes=(RECIPE,)).parse_args(["export", "toy", "--rope-parameters", value])
    native = resolve("toy", recipes=(RECIPE,))
    model = replace(native, checkpoint=rope_scaled(native, given.rope_parameters))
    arguments = _reproduction(model)
    assert arguments[arguments.index("--rope-parameters") + 1] == value

    shorter = replace(native, checkpoint=rope_scaled(native, {"factor": 4, "max_position_embeddings": 524_288,
                                                              "beta_fast": 16, "beta_slow": 2}))
    recorded = _reparsed(_reproduction(shorter)).rope_parameters
    assert rope_scaled(native, recorded) == shorter.checkpoint


def test_alias_declared_rope_scaling_is_recorded_so_reproduction_prepares_the_same_context():
    model = resolve("toy-yarn4", recipes=(RECIPE,))
    plan = _plan(model.checkpoint)
    assert plan.declared_context == 1_048_576
    reproduced = _reparsed(_reproduction(model, plan))
    # The recorded repository and revision name the unscaled alias, which serves the native context.
    target = resolve(reproduced.model, recipes=(RECIPE,))
    assert target.checkpoint.alias == "toy" and target.checkpoint.context is None
    scaled = rope_scaled(target, reproduced.rope_parameters)
    assert _plan(scaled).declared_context == plan.declared_context
    assert scaled.rope_scaling == model.checkpoint.rope_scaling


def test_unscaled_export_records_no_rope_parameters():
    arguments = _reproduction(resolve("toy", recipes=(RECIPE,)))
    assert "--rope-parameters" not in arguments and "--draft-model" not in arguments and "--recipe" not in arguments


def test_draft_folder_is_recorded_by_name_without_its_host_path():
    model = resolve("toy", recipes=(RECIPE,))
    plan = replace(_plan(model.checkpoint), draft_model=Path("/host/drafts/toy-draft"))
    arguments = _reproduction(model, plan)
    assert arguments[arguments.index("--draft-model") + 1] == "toy-draft"
    assert not any("/host" in argument for argument in arguments)
    assert _reparsed(arguments).draft_model == Path("toy-draft")


def test_checkpoint_directory_records_the_recipe_that_prepared_it():
    model = ResolvedModel(RECIPE, RECIPE.checkpoints[0], Path("/host/checkpoints/toy-finetune"))
    arguments = _reproduction(model)
    assert arguments[2] == "toy-finetune"
    assert _reparsed(arguments).recipe_name == "toy"


def _export_actions():
    parser = _parser(recipes=(RECIPE,))
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return [action for action in commands.choices["export"]._actions if action.option_strings]


def _recorded_flags(reproduction_arguments):
    """Every flag `reproduction_arguments` emits over plans that together set each recorded setting."""
    costs = SimpleNamespace(page_tokens=64)
    speculation = SimpleNamespace(
        method="mtp", draft_tokens=3, mtp_only=True, adaptive=False, packed_verification=True, graph_budget=8,
        draft_body_quantization="nvfp4", draft_head_quantization="nvfp4", markov_head_quantization="nvfp4",
        vocabulary_parallel_sampling=True)
    single = SimpleNamespace(
        recipe=SimpleNamespace(name="toy", ranks={2: costs}, max_images=4, speculation=speculation, stream_overlap=True),
        checkpoint=RECIPE.checkpoints[1], world_size=2, roce_ring=True, roce_chain=True,
        context_parallel=True, context_parallel_size=1, concurrency=4, language_model_only=False, image_concurrency=2,
        prefill_chunk=4096, expert_scale_compression="rank1", prompt_logprobs=True, mixed_prefill=True,
        speculation=speculation, speculative_token_capacity=16, online_mxfp8=True, online_mxfp8_scope="all",
        mla_compute_mode="dense", force_moe_decode_a16=True, native_dense_activations=True,
        shard_hyperconnections=True, joint_tuning=True, stream_overlap=False, collective_tuning="static",
        autotune_effort="default", lock_gpu_clocks=True, draft_model=Path("/host/drafts/toy-draft"))
    multi = SimpleNamespace(**{**vars(single), "context_parallel": False, "language_model_only": True,
                               "speculation": None})
    local = SimpleNamespace(directory=Path("/host/checkpoints/toy-finetune"))
    emitted = (reproduction_arguments(local, single, compiled_forwards=False)
               + reproduction_arguments(local, multi, compiled_forwards=False, multi_host=True))
    return {argument for argument in emitted if argument.startswith("--")}


def _unaccounted(reproduction_arguments):
    recorded = _recorded_flags(reproduction_arguments)
    return sorted(action.option_strings[-1] for action in _export_actions()
                  if not recorded & set(action.option_strings) and not NOT_REPRODUCED & set(action.option_strings))


def test_every_export_flag_is_reproduced_or_listed_as_not_changing_the_core():
    assert _unaccounted(rebuild_manifest.reproduction_arguments) == []
    options = {option for action in _export_actions() for option in action.option_strings}
    assert NOT_REPRODUCED <= options
    assert not NOT_REPRODUCED & _recorded_flags(rebuild_manifest.reproduction_arguments)


def test_flag_enumeration_names_a_flag_the_record_omits():
    def without_rope(*args, **kwargs):
        arguments = rebuild_manifest.reproduction_arguments(*args, **kwargs)
        at = arguments.index("--rope-parameters")
        return arguments[:at] + arguments[at + 2:]

    assert _unaccounted(without_rope) == ["--rope-parameters"]


def test_reproduction_arguments_parse_as_an_export_command():
    model = resolve("toy-yarn4", recipes=(RECIPE,))
    plan = replace(_plan(model.checkpoint), draft_model=Path("/host/drafts/toy-draft"))
    parsed = _reparsed(_reproduction(model, plan))
    assert parsed.rope_parameters == json.loads('{"factor": 4.0, "original_max_position_embeddings": 262144}')
