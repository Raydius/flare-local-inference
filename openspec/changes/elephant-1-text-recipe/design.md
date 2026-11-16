# Design — elephant-1-text-recipe

## Approach

A new recipe directory `ignition/recipes/elephant1/` mirrors `ignition/recipes/qwen38_dense/` — same HF architecture (`Qwen3_5ForConditionalGeneration`), same fused query/gate projection, same partial rotary, same GDN hybrid — with elephant-1's geometry and a BF16 weight source instead of a ModelOpt-quantized one. Nothing in `helios/`, `selene/`, `flare/` changes: the core uses existing contract fields (text inputs, paged KV regions, continuation state, generation logprob outputs) and existing components.

1. **`config.py` — geometry validation and storage plan selection.** Validate `architectures == ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5_text`, 32 `layer_types` with `full_attention` at indices 3, 7, …, 31, hidden 4096, intermediate 12288, vocabulary 248320, 16 attention heads, 4 kv heads, `head_dim` 256, GDN key heads 16, value heads 32, key/value head dim 128, `linear_conv_kernel_dim` 4, `attn_output_gate: true`, `partial_rotary_factor` 0.25, `rope_type` default, theta 1e7, untied embeddings, and no vision/audio inputs in scope. The recipe selects its own NVFP4 packing plan and refuses a checkpoint that declares its own `quantization_config`: this change serves the published BF16 artifact, and a pre-quantized variant would be served by a recipe that reads that declaration.
2. **`weights.py` — mapping.** Read `model.language_model.*` (layers, final norm, embedding) and `lm_head.weight`; de-interleave the fused query/gate rows the way `ignition/recipes/qwen38_dense/weights.py:152` does; skip `model.visual.*` as out-of-scope payload and record the skip; refuse any tensor that maps nowhere so a checkpoint with a changed layout fails loudly.
3. **`quantize.py` — BF16 to NVFP4 body plan (new preparation code).** A general packer added to `ignition/components/weights.py`, generalized from `ignition/components/vocabulary.py:114` (`nvfp4_draft_head`): per-tensor amax global gain, group-16 E4M3 scales, E2M1 codes, b12x `pack_weight` and `swizzle_block_scale`. Packed: `q_proj` (both row groups), `k_proj`, `v_proj`, `o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `linear_attn.out_proj`. Kept BF16: `embed_tokens`, `lm_head` (first stage), all norms, `linear_attn.conv1d`, `A_log`, `dt_bias`, `q_norm`, `k_norm`. W4A16 needs no activation global scale (`ignition/components/linear.py:243`). Packing runs chunked on the preparation GPU; export memory is planned from the packed sizes before allocation.
4. **`decoder.py` — composition.** 24 `PreparedGDN` blocks (`ignition/components/gated_delta.py:146`, wired as in `ignition/recipes/qwen38/decoder.py:307`) and 8 attention blocks: `qk_norm_rope` with 64 rotary dims of 256 at theta 1e7 and no mRoPE axes (`ignition/components/qk_norm_rope.py:47`), paged attention, `PreparedHeadGate` for the sigmoid output gate, then the output projection; dense MLP through the existing gate/up/down projections.
5. **`contract.py` — prepared contract.** Text inputs only; paged KV regions for the 8 attention layers at 128-token pages; `CONTINUATION_STATE` regions for the 24 GDN layers; generation logprob outputs enabled so a caller can request `logprobs: 20` at the generated position; no image or video input slots, so an image-token request has no graph to bind and is refused.
6. **`bootstrap.py` / `recipe.py` — declaration.** Alias `elephant-1-9b`, served name `Elephant-1`, context 16,384, `page_tokens` 128, concurrency and prefill chunk sized from the measured device cost, checkpoint entry pinned to `baselabs/elephant-1` at its downloaded revision with `stored_bytes` 18,819,722,392, `usage()` reporting prepared pages and free device bytes. Sampling defaults are a single deterministic distribution read: temperature 1, top_p 1, top_k 1, and the served path never samples.
7. **`docs/model-preparation.md`** gains a section with the export and run commands and the measured 12GB footprint; `tests/recipes/test_declarations.py` gains the new alias's declaration cases; a CPU-runnable test covers geometry validation, weight mapping, and unseen-tensor refusal; a quantization test bounds packed-projection error against BF16 over a fixed distribution.
8. **`qualify/elephant1_decision_parity.py` — acceptance probe.** Fixed states and question sets rendered by the vendor prompt rules. Stage A: the recipe's own BF16 canonical forward through `ignition debug-serve --direct --eager` (the authorized diagnostic execution path) versus the vendor `DecisionEngine` (OneJev at the pinned commit) isolates the mask deviation without a second export format. Stage B: the NVFP4 core versus that BF16 canonical forward isolates the precision deviation, including GDN recurrence. Report per option: argmax agreement, mean and max absolute probability delta, and rank agreement for score questions; refusal on prompt length comes from both runtimes for comparison.
9. **Capacity.** Concurrency is chosen from `flare inspect` on GPU 1 so weights, KV pages, GDN state, graphs, and workspace fit 12GB; the declared value and the measured peak are recorded, not extrapolated.

Serving stays a client-rendered prompt through `/v1/completions` with `max_tokens: 1` and `logprobs: 20`: the top-20 entries contain the slot-letter tokens, and the client maps them to probabilities. No route, tokenizer contract, or frontend metadata changes.

Residual risks the probe must expose: NVFP4 error accumulating through 24 recurrent GDN layers; low-margin option flips where the reference sits near a decision boundary; the mask deviation being confounded with quantization error (stage A prevents that); and the vision tower's 0.456B parameters being absent from the package by scope, so any benchmark that exercises images is out of reach for this core.

### Explicit Rejections

- Slicing `lm_head` to the option-letter rows — rejected: it replaces the standard vocabulary output with a decision-specific head, the model-specific native specialization the repository forbids, and it saves ≤0.6GB inside a 4.5GB headroom.
- Widening `qwen38_dense` to accept 32 GDN value heads — rejected: it would change a qualified recipe's identity and capacity contract for a checkpoint it does not serve.
- Refactoring `nvfp4_draft_head` to delegate to the new general packer — rejected: it would change prepared bytes for existing speculative recipes with no benefit to this change.
- The vendor `block_bidirectional` mask in this change — rejected by operator scope; measured instead, with the result recorded against this core rather than assumed.
- Vision tower and image inputs — rejected by operator scope.
- Any new HTTP route or decide endpoint — rejected: nine lines of client arithmetic over `logprobs` need no server contract.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.825
Caller Agreement: 0.5
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.85
- google-gemini-rpa (models/gemini-flash-lite-latest) — ok, agreement 0.45
- deepseek-flash (deepseek-flash) — ok, agreement 0.8
- local-ai-qllm (qLLM) — cancelled, quorum met before it responded

Certainty and caller agreement sit under the configured 0.9 floors. The gap is the lm_head-slicing point below, which the design rejects, plus the review's request to keep the fused query/gate handling in higher precision.

### Review Feedback

Consensus feedback recorded verbatim:
- "Packaging as a new declared recipe (ignition/recipes/<name>/) with a recipe-local decoder and contract, and no model-identity branches in the shared runtime, is the right boundary." — ACCEPTED: no Helios, contract, or shared-runtime change.
- "Quantize only the large body projections, and keep GDN parameters (A_log, dt_bias, conv1d), norms, q/k norms, and the fused query/gate handling in higher precision." — ACCEPTED for the listed exclusions (they are BF16). ACCEPTED in part for the fused query/gate projection: it is packed as one NVFP4 weight because a row-group split would break the shared fused projection path, and stage B of the probe measures it; if the gate rows prove sensitive the packing plan keeps `q_proj` BF16 as a declared one-line change.
- "Slice the lm_head to the option-letter vocabulary rows and keep embeddings high precision or host-resident." — ACCEPTED for the embedding (stays BF16). REJECTED for head slicing: it converts the standard vocabulary output into a decision-specific head, which the repository's native-specialization rule forbids, for ≤0.6GB of savings.
- "Either implement the vendor two-block full-attention mask as a declared contract or measure the causal deviation against the BF16 vendor runtime, separating mask effects from quantization effects." — ACCEPTED: stage A compares the recipe's BF16 canonical forward (`ignition debug-serve --direct --eager`) against the vendor runtime, so mask and precision deltas are never confounded.
- "Accept on option-probability and ranking agreement (argmax flips, per-option probability error, calibration) against the vendor reference, not perplexity." — ACCEPTED: the probe reports argmax agreement, mean/max probability delta, and score-question rank agreement.
- "Declare capacities and CUDA graph variants, and measure peak device memory under the 12GB ceiling including graphs, workspace, KV/GDN state, and fragmentation." — ACCEPTED: capacity is measured through `flare inspect` on GPU 1 at the declared context and concurrency.
- "Blast radius extends beyond the listed files to loader key-coverage checks, cache and graph planning, capacity tests, recipe registry snapshots, and the shared quantization and draft-head code, so run their existing tests." — ACCEPTED: `tests/recipes/test_qwen38_exact_checkpoints.py`, `tests/recipes/test_qwen38_kv_budget.py`, `tests/test_prefill_capacities.py`, `tests/preparation/test_checkpoint.py`, `tests/components/test_vocabulary_cuda.py`, `tests/cli/test_models.py`, and `tests/cli/test_commands.py` are added to the regression set.
- "Main risks: quantization error accumulating through recurrent GDN layers, low-margin option flips, mask deviation confounded with quantization error, and the parameter-count discrepancy from the vision tower." — ACCEPTED: stated in the approach and measured by the two-stage probe; the vision tower's absence is recorded as a scope limit of this core.
