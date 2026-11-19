# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The DSpark draft reference on fabricated weights, and the prepared draft compared against it."""

import os

import pytest
import torch

from ignition.reference.qwen38_dense_dspark import DSparkReference, yarn_frequencies

CUDA = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1" and bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
ROPE = dict(rope_type="yarn", factor=32.0, original_max_position_embeddings=8192, beta_fast=32.0, beta_slow=1.0,
            rope_theta=10000000)


def small_config(**changes):
    return dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=96, block_size=7, mask_token_id=95, markov_rank=8,
        rms_norm_eps=1e-6, target_layer_ids=[1, 3], rope_parameters=dict(ROPE)) | changes


def fabricate(config, seed=0):
    generator = torch.Generator().manual_seed(seed)
    h, i, head = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * head, config["num_key_value_heads"] * head
    shapes = {"fc.weight": (h, h * len(config["target_layer_ids"])), "hidden_norm.weight": (h,), "norm.weight": (h,),
        "markov_head.markov_w1.weight": (config["vocab_size"], config["markov_rank"]),
        "markov_head.markov_w2.weight": (config["vocab_size"], config["markov_rank"])}
    for layer in range(config["num_hidden_layers"]):
        for name, shape in {"self_attn.q_proj": (q, h), "self_attn.k_proj": (kv, h), "self_attn.v_proj": (kv, h),
                "self_attn.o_proj": (h, q), "self_attn.q_norm": (head,), "self_attn.k_norm": (head,),
                "input_layernorm": (h,), "post_attention_layernorm": (h,), "mlp.gate_proj": (i, h),
                "mlp.up_proj": (i, h), "mlp.down_proj": (h, i)}.items():
            shapes[f"layers.{layer}.{name}.weight"] = shape
    return {name: ((1 + .1 * torch.randn(shape, generator=generator)) if len(shape) == 1
                   else torch.randn(shape, generator=generator) * shape[-1] ** -.5).bfloat16()
            for name, shape in shapes.items()}


@pytest.mark.parametrize("dim,factor", [(16, 32.0), (128, 32.0), (128, 4.0)])
def test_rotary_tables_equal_the_transformers_yarn_computation(dim, factor):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    rope = dict(ROPE, factor=factor)
    config = Qwen3Config(hidden_size=4 * dim, num_attention_heads=4, head_dim=dim, max_position_embeddings=262144,
                         rope_parameters=dict(rope))
    expected, expected_factor = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
    inverse, attention_factor = yarn_frequencies(rope, dim)
    assert torch.equal(inverse, expected) and attention_factor == pytest.approx(expected_factor, rel=0, abs=1e-12)
    reference = DSparkReference(small_config(head_dim=dim, rope_parameters=rope), {"fc.weight": torch.empty(0)})
    positions = torch.tensor([0, 1, 127, 128, 8191, 262143])
    angles = positions.float()[:, None] * expected[None]
    cos, sin = reference.rotary_tables(positions)
    assert torch.equal(cos, torch.cat((angles, angles), -1).cos() * expected_factor)
    assert torch.equal(sin, torch.cat((angles, angles), -1).sin() * expected_factor)
    if factor == 32.0:
        assert attention_factor == 1.3465735902799727


def test_markov_bias_is_the_low_rank_transition_of_the_previous_token():
    config = small_config()
    tensors = fabricate(config)
    reference = DSparkReference(config, tensors)
    previous = torch.tensor([0, 7, 95])
    w1, w2 = (tensors[f"markov_head.markov_w{n}.weight"].float() for n in (1, 2))
    assert torch.equal(reference.markov_bias(previous), w1[previous] @ w2.T)
    assert reference.markov_bias(previous).shape == (3, config["vocab_size"])


def test_block_row_scores_the_next_position_from_the_token_before_it():
    """With flat draft logits, a transition that favors `previous + 1` drafts the pending token's successors."""
    config = small_config()
    tensors = fabricate(config)
    vocabulary, rank = config["vocab_size"], config["markov_rank"]
    codes = torch.nn.functional.normalize(torch.randn((vocabulary, rank), generator=torch.Generator().manual_seed(3)))
    tensors["markov_head.markov_w1.weight"] = codes.bfloat16()
    tensors["markov_head.markov_w2.weight"] = (8 * codes.roll(1, 0)).bfloat16()
    reference = DSparkReference(config, tensors)
    assert reference.block_positions(20).tolist() == list(range(20, 27))
    drafts = reference.propose(torch.zeros((7, vocabulary)), pending=40)
    assert drafts.tolist() == [41, 42, 43, 44, 45, 46, 47]
    # A row's own logits decide when they outweigh the transition.
    logits = torch.zeros((7, vocabulary))
    logits[2, 5] = 1e4
    assert reference.propose(logits, pending=40).tolist() == [41, 42, 5, 6, 7, 8, 9]


def test_block_is_noncausal_and_reads_every_context_position():
    config = small_config()
    reference = DSparkReference(config, fabricate(config))
    generator = torch.Generator().manual_seed(5)
    target = torch.randn((9, 2 * config["hidden_size"]), generator=generator)
    noise = torch.randn((7, config["hidden_size"]), generator=generator)
    base = reference.forward(noise, target)
    assert base.shape == (7, config["hidden_size"]) and torch.isfinite(base).all()
    later = noise.clone()
    later[6] += 1
    assert not torch.allclose(reference.forward(later, target)[0], base[0])
    first = target.clone()
    first[0] += 1
    assert not torch.allclose(reference.forward(noise, first), base)


def published_names(weights):
    """A prepared `DraftWeights` as the checkpoint's tensors, undoing the fused and stacked layouts."""
    a = weights.config.draft.attention
    q, kv = a.heads * a.key_dim, a.kv_heads * a.key_dim
    intermediate = weights.config.draft.text["intermediate_size"]
    draft = weights.draft
    tensors = {"fc.weight": draft.projection, "hidden_norm.weight": draft.hidden_norm, "norm.weight": draft.norm,
        "markov_head.markov_w1.weight": draft.markov_embedding, "markov_head.markov_w2.weight": draft.markov_head}
    for index, layer in enumerate(draft.layers):
        p = f"layers.{index}."
        context = draft.context_kv[index * 2 * kv:(index + 1) * 2 * kv]
        assert torch.equal(context, layer.qkv[q:])
        tensors.update({p + "self_attn.q_proj.weight": layer.qkv[:q], p + "self_attn.k_proj.weight": context[:kv],
            p + "self_attn.v_proj.weight": context[kv:], p + "self_attn.o_proj.weight": layer.output,
            p + "self_attn.q_norm.weight": layer.query_norm, p + "self_attn.k_norm.weight": layer.key_norm,
            p + "input_layernorm.weight": layer.input_norm,
            p + "post_attention_layernorm.weight": layer.post_attention_norm,
            p + "mlp.gate_proj.weight": layer.mlp.gate_up[:intermediate],
            p + "mlp.up_proj.weight": layer.mlp.gate_up[intermediate:], p + "mlp.down_proj.weight": layer.mlp.down})
    return tensors


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


# Each BF16 result rounds to 8 significant bits, a relative error of at most 2**-9. A draft layer chains
# about ten rounded results (two norms, the fused projection, the rotation, the attention output and its
# projection, the gated MLP's three stages and two residual sums), and the final norm and vocabulary
# projection add two more.
BF16_STEP = 2.0 ** -9


def bf16_bound(layers):
    return (10 * layers + 2) * BF16_STEP


@pytest.mark.skipif(not CUDA, reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned CUDA_VISIBLE_DEVICES")
@torch.no_grad()
def test_prepared_draft_matches_the_reference_on_fabricated_weights():
    """Prepared logits before and after the Markov bias follow the reference at every block position.

    The stored context keys and values are first compared with the reference's at FP8 E4M3 precision.
    The block then attends the stored history in both, and the reference stores the block's own keys and
    values at FP8, so the remaining difference is BF16 rounding of the prepared draft's intermediate results.
    """
    from b12x.preparation import PreparationSession
    from ignition.preparation.scratch import ScratchArena
    from tests.recipes import test_qwen38_dense_dspark_cuda as prepared

    weights = prepared.weights.__wrapped__()
    config = dict(weights.config.draft.text)
    reference = DSparkReference(config, published_names(weights))
    start, pending = 300, 41
    target = prepared.features(start, 17)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        harness = prepared.Harness(weights, session, ScratchArena("cuda"), chunk=384)
        harness.write("chunk", target, 0)
        logits = harness.propose(start, pending)[:prepared.BLOCK]
        harness.previous.copy_(torch.tensor([pending, 7], device="cuda"))
        harness.markov()
        bias = harness.bias[0].clone()
        history = [tuple(pages.float().view(-1, prepared.KV_HEADS, prepared.HEAD)[:start].clone()
                         for pages in harness.pages.draft_attention(layer, None)) for layer in range(reference.layers)]

    def fp8(values):
        return values.clamp(-448, 448).to(torch.float8_e4m3fn).float()

    tokens = torch.full((prepared.BLOCK,), prepared.VOCABULARY - 1, device="cuda")
    tokens[0] = pending
    assert reference.block_positions(start).tolist() == list(range(start, start + prepared.BLOCK))
    noise = weights.embedding[tokens]
    context, positions = reference.context(target), torch.arange(start, device="cuda")
    history_errors = [relative_error(stored, fp8(computed)) for layer in range(reference.layers)
                      for stored, computed in zip(history[layer], reference.keys_values(layer, context, positions))]
    expected = reference.forward(noise, target, storage=fp8, history=history) @ weights.head.float().T
    computed = reference.forward(noise, target, storage=fp8) @ weights.head.float().T
    unstored = reference.forward(noise, target) @ weights.head.float().T
    expected_bias = reference.markov_bias(torch.tensor(pending, device="cuda"))
    bias_error = relative_error(bias, expected_bias)
    biased_error = relative_error(logits[0] + bias, expected[0] + expected_bias)
    rows = [relative_error(logits[row], expected[row]) for row in range(prepared.BLOCK)]
    print(dict(history_errors=history_errors, row_errors=rows,
               logit_error_with_computed_fp8_history=relative_error(logits, computed),
               logit_error_with_unrounded_history=relative_error(logits, unstored),
               markov_bias_error=bias_error, biased_logit_error=biased_error, bound=bf16_bound(reference.layers)))
    # A stored key or value differs from the reference's FP8 rounding where BF16 rounding of the context
    # projection, norm and rotation moved it across an FP8 boundary, by one FP8 step of 2**-4 relative.
    assert max(history_errors) < 2.0 ** -4
    assert max(rows) < bf16_bound(reference.layers)
    assert bias_error < 2 * BF16_STEP
    assert biased_error < bf16_bound(reference.layers)
    assert logits.argmax(-1).tolist() == expected.argmax(-1).tolist()


@pytest.mark.skipif(not CUDA or not os.environ.get("FLARE_QWEN38_DSPARK_SNAPSHOT"),
    reason="requires FLARE_RUN_CUDA_TESTS=1, an assigned CUDA_VISIBLE_DEVICES with about 12 GiB free, and"
           " FLARE_QWEN38_DSPARK_SNAPSHOT and FLARE_QWEN38_DENSE_SNAPSHOT naming the pinned draft and a target")
@torch.no_grad()
def test_prepared_draft_matches_the_reference_on_the_pinned_weights():
    """The same comparison on `RadixArk/Qwen3.8-27B-DSpark` with a dense Qwen target's embedding and head."""
    from pathlib import Path
    from types import SimpleNamespace

    from b12x.preparation import PreparationSession
    from safetensors import safe_open

    from ignition.components.dflash2_weights import DraftLayerWeights
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import MLPWeights
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38_dense import dspark_config
    from ignition.recipes.qwen38_dense.config import ModelConfig
    from ignition.recipes.qwen38_dense.dspark_weights import DraftWeights
    from tests.recipes import test_qwen38_dense_dspark_cuda as prepared

    snapshot = Path(os.environ["FLARE_QWEN38_DSPARK_SNAPSHOT"])
    target_snapshot = Path(os.environ.get("FLARE_QWEN38_DENSE_SNAPSHOT", ""))
    if not (target_snapshot / "config.json").is_file():
        pytest.skip("requires FLARE_QWEN38_DENSE_SNAPSHOT naming a dense Qwen target checkpoint")
    target = ModelConfig.load(target_snapshot)
    draft_config = dspark_config.load(snapshot, target)
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cuda") as source:
        tensors = {name: source.get_tensor(name) for name in source.keys() if not name.startswith("confidence_head.")}
    embedding_name, head_name = "model.language_model.embed_tokens.weight", "lm_head.weight"
    import json

    shards = json.loads((target_snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(target_snapshot / shards[embedding_name], framework="pt", device="cuda") as source:
        embedding = source.get_tensor(embedding_name)
    if head_name not in shards:
        pytest.skip("the target checkpoint stores no BF16 lm_head.weight tensor")
    with safe_open(target_snapshot / shards[head_name], framework="pt", device="cuda") as source:
        head = source.get_tensor(head_name)
    layers = []
    for index in range(draft_config.layers):
        p = f"layers.{index}."
        layers.append(DraftLayerWeights(
            torch.cat([tensors[p + f"self_attn.{name}_proj.weight"] for name in "qkv"]),
            tensors[p + "self_attn.o_proj.weight"], tensors[p + "self_attn.q_norm.weight"],
            tensors[p + "self_attn.k_norm.weight"], tensors[p + "input_layernorm.weight"],
            tensors[p + "post_attention_layernorm.weight"],
            MLPWeights(torch.cat([tensors[p + f"mlp.{name}_proj.weight"] for name in ("gate", "up")]),
                       tensors[p + "mlp.down_proj.weight"]), None, None))
    q = draft_config.attention.heads * draft_config.attention.key_dim
    draft = DraftWeights(tensors["fc.weight"], tensors["hidden_norm.weight"],
        torch.cat([layer.qkv[q:] for layer in layers]), tensors["norm.weight"],
        tensors["markov_head.markov_w1.weight"], tensors["markov_head.markov_w2.weight"], tuple(layers))
    weights = SimpleNamespace(config=SimpleNamespace(hidden=target.hidden, vocabulary=target.vocabulary,
        text=target.text, draft=draft_config), shard=TensorParallelShard(0, 1), draft=draft,
        embedding=embedding, head=head)
    reference = DSparkReference(dict(draft_config.text), tensors)
    start, pending = 300, 41
    generator = torch.Generator(device="cuda").manual_seed(17)
    features = torch.randn((start, len(draft_config.target_layers) * target.hidden), device="cuda",
                           generator=generator).bfloat16()
    geometry = dict(HIDDEN=target.hidden, VOCABULARY=target.vocabulary, LAYERS=draft_config.layers,
        TAPS=len(draft_config.target_layers), KV_HEADS=draft_config.attention.kv_heads)
    saved = {name: getattr(prepared, name) for name in geometry}
    try:
        for name, value in geometry.items():
            setattr(prepared, name, value)
        with PreparationSession(device="cuda:0", autotune=False) as session:
            harness = prepared.Harness(weights, session, ScratchArena("cuda"), chunk=384)
            harness.write("chunk", features, 0)
            logits = harness.propose(start, pending)[:prepared.BLOCK]
            harness.previous.copy_(torch.tensor([pending, 7], device="cuda"))
            harness.markov()
            bias = harness.bias[0].clone()
    finally:
        for name, value in saved.items():
            setattr(prepared, name, value)
    tokens = torch.full((prepared.BLOCK,), draft_config.noise_token, device="cuda")
    tokens[0] = pending

    def fp8(values):
        return values.clamp(-448, 448).to(torch.float8_e4m3fn).float()

    expected = reference.forward(embedding[tokens], features, storage=fp8) @ head.float().T
    expected_bias = reference.markov_bias(torch.tensor(pending, device="cuda"))
    rows = [relative_error(logits[row], expected[row]) for row in range(prepared.BLOCK)]
    print(dict(row_errors=rows, markov_bias_error=relative_error(bias, expected_bias)))
    assert max(rows) < bf16_bound(reference.layers)
    assert relative_error(bias, expected_bias) < 2 * BF16_STEP
    assert relative_error(logits[0] + bias, expected[0] + expected_bias) < bf16_bound(reference.layers)
