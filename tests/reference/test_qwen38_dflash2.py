"""CPU checks of the DFlash2 draft reference on fabricated small-geometry weights."""

from __future__ import annotations

import copy

import pytest
import torch

from ignition.reference.qwen38_dflash2 import (
    PINNED,
    DFlash2Reference,
    grouped_convolve,
    select_step,
)


def _small_config():
    config = copy.deepcopy(PINNED)
    config.update(hidden_size=64, intermediate_size=96, num_hidden_layers=2, layer_types=["sliding_attention"] * 2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=16, vocab_size=512)
    config["dflash_config"].update(conv_group_size=8, mask_token_id=500, selector_rank=8, selector_top_k=4,
                                   target_layer_ids=[5, 15])
    return config


def _fabricate(config, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in DFlash2Reference.expected_shapes(config).items():
        if name.endswith("norm.weight"):
            tensors[name] = (1 + 0.1 * torch.randn(shape, generator=generator)).bfloat16()
        else:
            tensors[name] = (0.2 * torch.randn(shape, generator=generator)).bfloat16()
    return tensors


@pytest.fixture(scope="module")
def draft():
    config = _small_config()
    return DFlash2Reference(config, _fabricate(config), strict=False)


@pytest.fixture(scope="module")
def target(draft):
    generator = torch.Generator().manual_seed(1)
    vocab, hidden = draft.config["vocab_size"], draft.hidden
    return (torch.randn(vocab, hidden, generator=generator).bfloat16(),
            (0.3 * torch.randn(vocab, hidden, generator=generator)).bfloat16())


def _features(draft, count, seed=2):
    generator = torch.Generator().manual_seed(seed)
    width = draft.hidden * len(draft.config["dflash_config"]["target_layer_ids"])
    return torch.randn(count, width, generator=generator).bfloat16(), torch.arange(count)


@pytest.mark.parametrize("field, value", [("block_size", 7), ("target_layer_ids", [4, 15, 25, 35, 45])])
def test_strict_geometry_rejection_names_the_field(field, value):
    config = copy.deepcopy(PINNED)
    config["dflash_config"][field] = value
    with pytest.raises(ValueError, match=field):
        DFlash2Reference(config, {})


def test_strict_rejects_small_geometry():
    config = _small_config()
    with pytest.raises(ValueError, match="hidden_size"):
        DFlash2Reference(config, _fabricate(config))


def test_tensor_shape_rejection():
    config = _small_config()
    tensors = _fabricate(config)
    tensors["fc.weight"] = tensors["fc.weight"][:, :-1]
    with pytest.raises(ValueError, match="fc.weight"):
        DFlash2Reference(config, tensors, strict=False)
    tensors = _fabricate(config)
    del tensors["norm.weight"]
    with pytest.raises(ValueError, match="norm.weight"):
        DFlash2Reference(config, tensors, strict=False)


def test_shapes_and_finite_values(draft, target):
    embedding, lm_head = target
    context = draft.context(*_features(draft, 12))
    assert len(context.layers) == draft.layers
    for k, v in context.layers:
        assert k.shape == (12, draft.kv_heads, draft.head_dim) and v.shape == k.shape
        assert k.dtype == v.dtype == torch.bfloat16
    hidden = draft.block(7, 12, context, embedding)
    assert hidden.shape == (8, draft.hidden) and hidden.dtype == torch.bfloat16 and hidden.isfinite().all()
    ids, unary = draft.candidates(hidden, lm_head)
    assert ids.shape == (7, 4) and ids.dtype == torch.int64
    assert unary.shape == (7, 4) and unary.dtype == torch.float32 and unary.isfinite().all()
    proposals = draft.walk(hidden, 7, ids, unary)
    assert proposals.shape == (7,) and proposals.dtype == torch.int64
    assert all(int(proposals[p]) in ids[p].tolist() for p in range(7))
    torch.testing.assert_close(draft.propose(7, 12, context, embedding, lm_head), proposals, rtol=0, atol=0)


def test_block_inputs_substitute_mask_tokens(draft, target):
    embedding, _ = target
    inputs = draft.block_inputs(7, embedding)
    assert inputs.shape == (8, draft.hidden)
    torch.testing.assert_close(inputs[0], embedding[7], rtol=0, atol=0)
    for p in range(1, 8):
        torch.testing.assert_close(inputs[p], embedding[draft.mask_id], rtol=0, atol=0)


def test_context_window_and_noncausal_block(draft, target):
    embedding, _ = target
    features, positions = _features(draft, 20)
    full = draft.block(3, 20, draft.context(features, positions), embedding)
    window = draft.context(features[-draft.window:], positions[-draft.window:])
    torch.testing.assert_close(draft.block(3, 20, window, embedding), full, rtol=0, atol=0)
    short = draft.context(features[-draft.window + 1:], positions[-draft.window + 1:])
    assert not torch.equal(draft.block(3, 20, short, embedding), full)
    inputs = draft.block_inputs(3, embedding)
    changed = inputs.clone()
    changed[7] = embedding[11]
    assert not torch.equal(draft.forward(changed, 20, window)[0], draft.forward(inputs, 20, window)[0])


def _convolve_loop(x, coefficients, base, side, group):
    taps, hidden = base.shape[1], x.shape[1]
    out = torch.zeros(x.shape, dtype=torch.float32)
    coeff = coefficients.float().view(x.shape[0], 2, taps, hidden // group)
    for row in range(x.shape[0]):
        for tap in range(taps):
            if row % 8 < tap:
                continue
            for column in range(hidden):
                weight = base[side, tap, column].float() + coeff[row, side, tap, column // group]
                out[row, column] += weight * x[row - tap, column].float()
    return out.bfloat16()


def test_grouped_convolution_matches_loop_and_stays_block_local():
    generator = torch.Generator().manual_seed(3)
    hidden, group = 32, 8
    x = torch.randn(16, hidden, generator=generator).bfloat16()
    coefficients = torch.randn(16, 2 * 2 * hidden // group, generator=generator).bfloat16()
    base = torch.randn(2, 2, hidden, generator=generator).bfloat16()
    for side in (0, 1):
        torch.testing.assert_close(grouped_convolve(x, coefficients, base, side),
                                   _convolve_loop(x, coefficients, base, side, group), rtol=0, atol=0)
    perturbed = x.clone()
    perturbed[7] += 1
    torch.testing.assert_close(grouped_convolve(perturbed, coefficients, base, 0)[8:],
                               grouped_convolve(x, coefficients, base, 0)[8:], rtol=0, atol=0)
    assert not torch.equal(grouped_convolve(perturbed, coefficients, base, 0)[7], grouped_convolve(x, coefficients, base, 0)[7])


def test_walk_matches_brute_force(draft, target):
    embedding, lm_head = target
    hidden = draft.block(5, 12, draft.context(*_features(draft, 12)), embedding)[:4]
    ids, unary = draft.candidates(hidden, lm_head)
    assert ids.shape == (3, 4)
    projected = torch.nn.functional.linear(hidden[1:], draft.w["candidate_selector.hidden_projection.weight"])
    a, b = draft.w["candidate_selector.predecessor_codebook"], draft.w["candidate_selector.successor_codebook"]
    expected, predecessor = [], 5
    for p in range(3):
        best, best_score = None, None
        for c in range(4):
            score = float(unary[p, c])
            for r in range(a.shape[1]):
                score += float((a[predecessor, r] * projected[p, r]).float() * b[ids[p, c], r].float())
            if best_score is None or score > best_score:
                best, best_score = c, score
        einsum = unary[p] + torch.einsum("r,cr->c", (a[predecessor] * projected[p]).float(), b[ids[p]].float())
        torch.testing.assert_close(select_step(unary[p], projected[p], a[predecessor], b[ids[p]]), einsum)
        predecessor = int(ids[p, best])
        expected.append(predecessor)
    torch.testing.assert_close(draft.walk(hidden, 5, ids, unary), torch.tensor(expected), rtol=0, atol=0)


def test_walk_breaks_ties_to_the_lowest_id(draft, target):
    embedding, _ = target
    config = draft.config
    tensors = dict(draft.w)
    tensors["candidate_selector.predecessor_codebook"] = torch.zeros_like(tensors["candidate_selector.predecessor_codebook"])
    flat = DFlash2Reference(config, tensors, strict=False)
    hidden = flat.block(5, 12, flat.context(*_features(flat, 12)), embedding)[:4]
    ids = torch.tensor([[9, 3, 7, 1], [40, 2, 2, 8], [6, 300, 11, 11]])
    unary = torch.zeros(3, 4)
    assert flat.walk(hidden, 5, ids, unary).tolist() == [1, 2, 6]


# CUDA: parity with the shared Triton components, and the measurement on one real trace.

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")


@cuda
@torch.no_grad()
def test_cuda_convolution_matches_triton_component():
    from ignition.components.grouped_convolution import _convolve

    torch.manual_seed(7)
    rows, hidden, group, block = 16, 2560, 16, 8
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device="cuda")
    coefficients = (0.1 * torch.randn((rows, 2 * 2 * hidden // group), device="cuda")).bfloat16()
    base = torch.randn((2, 2, hidden), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(x)
    for side in (0, 1):
        torch.library.wrap_triton(_convolve)[((x.numel() + 255) // 256,)](
            x, coefficients, base, output, rows, hidden, group, block, 2, coefficients.stride(0), side, 256)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, grouped_convolve(x, coefficients, base, side, block), rtol=2e-2, atol=2e-2)


@cuda
@torch.no_grad()
def test_cuda_walk_matches_candidate_selector_component():
    from b12x.preparation import PreparationSession
    from ignition.components.candidate_selector import PreparedCandidateSelector

    config = _small_config()
    config.update(hidden_size=256, vocab_size=1024)
    config["dflash_config"].update(selector_rank=64, selector_top_k=16, conv_group_size=16)
    tensors = {name: value.cuda() for name, value in _fabricate(config, seed=3).items()}
    draft = DFlash2Reference(config, tensors, strict=False)
    rows, block, vocab = 2, draft.block_size, config["vocab_size"]
    torch.manual_seed(11)
    hidden = torch.randn((rows, block, draft.hidden), dtype=torch.bfloat16, device="cuda")
    head = (0.3 * torch.randn((vocab, draft.hidden), device="cuda")).bfloat16()
    pending = torch.tensor([17, 400], device="cuda")
    weights = {"projection": tensors["candidate_selector.hidden_projection.weight"],
               "predecessor": tensors["candidate_selector.predecessor_codebook"],
               "successor": tensors["candidate_selector.successor_codebook"]}
    with PreparationSession(device="cuda:0", autotune=False) as session:
        selector = PreparedCandidateSelector(hidden, weights, top_k=draft.top_k, session=session, name="selector")
        selector.prepare(hidden)
        predecessor, component = pending.clone(), []
        for position in range(1, block):
            logits = torch.nn.functional.linear(hidden[:, position], head).float()
            selector(logits, predecessor, position)
            predecessor = logits.argmax(-1)
            component.append(predecessor)
    component = torch.stack(component, 1)
    for row in range(rows):
        ids, unary = draft.candidates(hidden[row], head)
        reference = draft.walk(hidden[row], int(pending[row]), ids, unary)
        assert reference.tolist() == component[row].tolist()


@cuda
@torch.no_grad()
def test_cuda_measure_real_trace_beats_shuffled_control(tmp_path):
    from qualify import dflash2_acceptance as da

    traces = Path(os.environ.get("FLARE_DFLASH2_TRACES", ""))
    snapshot = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", "/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"))
    draft = Path(os.environ.get("FLARE_DFLASH2_DRAFT", "~/models/incoai/Qwen3.8-Flash-Next-DFlash2")).expanduser()
    chat = sorted(traces.glob("chat-*.pt")) if traces.is_dir() else []
    if not chat or not snapshot.is_dir() or not draft.is_dir():
        pytest.skip("requires FLARE_DFLASH2_TRACES with a chat-* trace, the exact Qwen checkpoint, and the DFlash2 draft")
    folder = tmp_path / "traces"
    folder.mkdir()
    shutil.copy(chat[0], folder)

    def run(shuffle):
        args = SimpleNamespace(traces=folder, snapshot=snapshot, draft=draft, widths="4,8", arms="dflash2", e4m3_kv=False,
                               shuffle_features=shuffle, seed=0)
        return da.measure(args)["results"]["dflash2"]

    real, control = run(False), run(True)
    assert real["4"]["overall"]["rounds"] > 0 and real["8"]["overall"]["rounds"] > 0
    assert real["4"]["overall"]["acceptance_length"] >= 2.0
    assert control["4"]["overall"]["acceptance_length"] < real["4"]["overall"]["acceptance_length"]
