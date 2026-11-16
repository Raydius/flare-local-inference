# Package Elephant-1 as a text decision core that fits 12GB

Tier: 2
Type: feature

## Stated Requirements

- "We need to create a recipe that will allow Elephant-1 to be packaged and fit in 12GB"
- "Yes the target is GPU 1, the 5060ti."
- "Let's start without vision"

Assumed: the package is text-only — no vision tower, no image tokens, image inputs refused.
Assumed: the lane is causal-only; the checkpoint's `block_bidirectional` full-attention mask is out of scope and the OneJev parity probe measures the deviation.
Assumed: decisions are served as client-rendered prompts through Completions with 1-token, top-20 generation logprobs; no new API surface.
Assumed: packaging quantizes the BF16 checkpoint to NVFP4 at preparation; the embedding stays BF16.
Assumed: declared context is the trained 16,384 tokens; longer requests are refused.
Assumed: the single MTP layer named by `mtp_num_hidden_layers` has no shipped tensors and is not served.

## Why

`baselabs/elephant-1` (MIT, 9.41B parameters, one 18.8GB BF16 `model.safetensors`) is a Qwen3.5-9B fine-tune specialized for one shared-prefix prefill that returns calibrated probabilities per option instead of generating text. Its text stack is the architecture this repo already serves — 24 GatedDeltaNet layers plus 8 full-attention layers, `head_dim` 256, GQA 16/4, partial rotary 0.25, fused query/gate projection — but no recipe accepts it: the dense Qwen recipe pins `linear_num_value_heads` 48 and requires a ModelOpt-quantized checkpoint, and the shipped weights are BF16 at 18.8GB, which cannot fit the 12GB budget on GPU 1.

The readout costs nothing extra: `qev/readout.py` in the checkpoint's `serve.patch` shows `slot_lm` is the base LM head applied at the final suffix token over the slot letters, so the decision interface is a standard prefill plus one position's logits. What the package must add is a recipe for the geometry, a BF16→NVFP4 preparation path the repo lacks, and capacities sized to 12GB.

The one intended deviation from training is the attention mask. The reference runs full-attention layers with a two-block `block_bidirectional` mask; this package exports ordinary causal graphs and measures the difference against the reference runtime instead of assuming it is negligible.

## What Changes

- New recipe `ignition/recipes/elephant1/`: geometry validation, weight mapping, decoder (GDN + gated attention + dense MLP), text contract, export entry, and recipe declaration (alias `elephant-1-9b`, context 16,384, page tokens 128, 12GB-targeted concurrency).
- New preparation path that packs a BF16 checkpoint's body projections to NVFP4 (weights-only, W4A16) by generalizing the packing already used by `components/vocabulary.nvfp4_draft_head`, with the embedding left BF16 and the vocabulary head packed separately.
- Registration in `ignition/recipes/__init__.py`, a preparation section in `docs/model-preparation.md`, declaration tests, and one qualification driver that compares package probabilities against the reference `DecisionEngine` on a fixed prompt set.
- Generation logprob outputs are enabled so a caller reads the answer position's top-20 logprobs and maps slot letters to probabilities.

Non-goals: vision and image tokens; the `block_bidirectional` mask; a native decide endpoint or any new HTTP route; speculative decoding and the MTP layer; multi-GPU or Orbit serving; publishing an NVFP4 checkpoint to the Hub.

## Capabilities

### New Capabilities

`elephant-1-serving`: the prepared text-only Elephant-1 core, its declared capacities, its decision readout through generation logprobs, its refusal rules, and the qualification statements for the causal mask and NVFP4 precision.

### Modified Capabilities

None.

## Impact

- New files only under `ignition/recipes/elephant1/`, `ignition/components/` (quantization helper), `tests/recipes/`, and `qualify/`; plus edits to `ignition/recipes/__init__.py`, `docs/model-preparation.md`, and `tests/recipes/test_declarations.py`.
- No `helios/`, `selene/`, `selene-manager/`, `flare/` or `flare-cli/` change; no contract additions.
- No change to existing recipes or their checkpoints; `qwen38_dense` keeps its 48-value-head geometry and quantized-checkpoint requirement.
- Package footprint target: under 12GB of device memory for weights, KV pages, GDN state, graphs and workspace at the declared context and concurrency.

## Blast Radius

1. `ignition/recipes/__init__.py:24` — `recipes()` imports and returns every recipe; the new module joins both lists.
2. `ignition/recipes/qwen38_dense/config.py:32` — the dense Qwen geometry pins `linear_num_value_heads` 48; elephant-1 declares 32, so this config rejects the checkpoint.
3. `ignition/recipes/qwen38_dense/config.py:54` — `quantized_layers(raw)` requires a ModelOpt `MIXED_PRECISION` `quantization_config`; the elephant-1 `config.json` has none.
4. `ignition/preparation/quantization.py:20` — `modelopt_nvfp4_layers` only reads published NVFP4 layers; nothing in preparation produces them from BF16 body weights.
5. `ignition/components/vocabulary.py:114` — `nvfp4_draft_head` is the in-repo BF16→NVFP4 packer (b12x `pack_weight`, `swizzle_block_scale`, E2M1 codes, amax global gain); the body plan generalizes it.
6. `ignition/components/linear.py:243` — `NVFP4LinearWeight` A16 accepts an optional activation scale; a W4A16 body runs with BF16 activations and no activation scale.
7. `ignition/components/gated_delta.py:146` — `PreparedGDN` executes the 24 linear-attention layers; `GDNState` at `:38` allocates their fp32 recurrence.
8. `ignition/recipes/qwen38/decoder.py:307` — the existing `PreparedGDN` wiring the new decoder follows.
9. `ignition/recipes/qwen38_dense/weights.py:152` — existing de-interleave of the fused query/gate projection; `components/head_gate.PreparedHeadGate` (imported at `ignition/recipes/qwen38_dense/decoder.py:16`) applies the output gate.
10. `ignition/components/qk_norm_rope.py:47` — rotary parameter set; text-only elephant-1 needs 64 of 256 rotary dims at theta 1e7 with no mRoPE axes.
11. `docs/model-preparation.md:356` — the per-recipe preparation section pattern a new section follows.
12. `tests/recipes/test_declarations.py:27` — recipe declarations enumerate aliases and capacities; a new alias adds cases.
13. grepped `elephant` across the repository — no sites; the recipe, tests, docs section, and qualification driver are all new.
