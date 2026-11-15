# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import TRANSFORMERS_RELEASE, TRANSFORMERS_REVISION, supported_transformers
from ignition.recipes.qwen38 import create_model_tree


def test_exact_checkpoint_constructs_canonical_meta_tree_without_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    snapshot = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    if not snapshot.exists():
        pytest.skip("exact Qwen3.8 checkpoint is not cached")
    initialized = torch.cuda.is_initialized()
    model = create_model_tree(inspect_qwen38_checkpoint(snapshot))
    assert type(model).__name__ == "Qwen4ExpForConditionalGeneration"
    assert len(model.model.language_model.layers) == 48
    assert len(model.model.visual.blocks) == 27
    assert model.model.language_model.layers[1].ple is not None
    assert model.model.language_model.layers[3].self_attn.head_dim == 256
    assert sum(p.numel() for p in model.parameters()) == 177392830576
    assert all(p.device.type == "meta" and not p.requires_grad for p in model.parameters())
    assert all(p.device.type == "meta" for p in model.buffers())
    assert not model.training
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("version,direct_url,accepted", [
    (TRANSFORMERS_RELEASE, None, True),
    ("5.21.0.dev0", '{"vcs_info": {"commit_id": "%s"}}' % TRANSFORMERS_REVISION, True),
    ("5.21.0.dev0", '{"vcs_info": {"commit_id": "0000000000000000000000000000000000000000"}}', False),
    ("5.22.0", None, False),
])
def test_qwen_preparation_accepts_the_locked_release_and_the_pinned_revision(version, direct_url, accepted):
    from types import SimpleNamespace

    distribution = SimpleNamespace(version=version, read_text=lambda name: direct_url)
    assert supported_transformers(distribution) is accepted


def test_locked_transformers_release_is_the_accepted_release():
    import tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    pins = [d for d in project["project"]["optional-dependencies"]["prepare"] if d.startswith("transformers")]
    assert pins == [f"transformers[torch,vision,chat_template]=={TRANSFORMERS_RELEASE}"]
