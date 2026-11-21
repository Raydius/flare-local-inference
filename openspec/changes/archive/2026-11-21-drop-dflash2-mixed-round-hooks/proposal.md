# Drop the DFlash2 mixed-round hooks from the Qwen3.8 Flash recipe

Tier: 3
Type: bug

## Stated Requirements

- "My implementation of DFlash2 for Qwen3.8-Flash-Next was rejected for the following reason: \"DFlash2 drafts inside mixed prefill rounds will not be added. They would make mixed rounds carry a second speculative mode, and nothing here measures what that gains over the separate prefill and verification graphs main requires (ignition/planning.py:574-575, ignition/recipes/qwen38/bootstrap.py:130).\""
- "So to be clear, the mixed arm is effectively dead weight in the current fork anyway, correct? We're not even using it?"
- "What is the optimal method?" — answered and recorded: arm Q (upstream `qwen38` with `--speculative-method dflash2`, the MXFP8 draft, `--mixed-prefill` off) stays the serving configuration.
- "First update the documentation in the vault at `Personal Projects/AI Control Plane/HyperFlare/` -- captrue the feedback, our testing results, and our planned response." — done in the vault (`DFlash2 Recipe Progress` section 21) before this change.
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."
- Not included: the two guards and their documentation — "leave both guards and their documentation untouched" (the operator's accepted scope: DFlash2 mixed hooks out, guards and docs in place).

## Root Cause

wrong-abstraction: the fork's retired `qwen38_dflash` recipe parameterized shared Qwen3.8 Flash preparation with a second draft program it can no longer run. `HybridStateLayout.build` still takes `mtp=` and `draft_windows=` to lay out E4M3 draft sliding-window rows for a DFlash2 draw inside a mixed round (`ignition/recipes/qwen38/state.py:37`), the target decoder still carries per-layer feature taps (`ignition/recipes/qwen38/decoder.py:224`), `PreparedMixedState` still carries packed/prompt/decode feature buffers (`ignition/recipes/qwen38/mixed_state.py:201`), and `prepare_commit` still accepts a private `feedback=` scratch pair (`ignition/recipes/qwen38/speculative_state.py:250`). `README.md:95` already records them as "Left without a production caller, unchanged"; every remaining caller is a test of the retired arm, so the parameters are fork-specific policy carried in shared code that upstream's separate-graph DFlash2 path never exercises.

## Blast Radius

1. `ignition/recipes/qwen38/state.py:37` — `HybridStateLayout.build(..., mtp=True, draft_windows=None)`; the only caller passing either is `tests/recipes/test_qwen38_mixed.py:201`.
2. `ignition/recipes/qwen38/state.py:81-85` — the `draft{layer}.key`/`value` E4M3 window regions no production layout requests.
3. `ignition/recipes/qwen38/state.py:87` — `if speculative_tokens and mtp and draft_config is None` gates the MTP feedback and draft-ring regions on the flag no caller sets.
4. `ignition/recipes/qwen38/state.py:134-135` and `:144-145` — `HybridStatePool.allocate` forwards both parameters; the production call at `ignition/recipes/qwen38/bootstrap.py:412-415` passes neither, as do `:182` and `:407`.
5. `ignition/recipes/qwen38/decoder.py:224-231`, `:255`, `:283-284`, `:451-455` — `validate_feature_taps`, the `feature_taps`/`auxiliary` parameters and the per-layer tap write; no caller in `ignition/` or `tests/` passes them.
6. `ignition/recipes/qwen38/mixed_state.py:201-209` and `:298-300` — `features`/`prompt_features`/`decode_features` and their gather-copies; reached only by `tests/recipes/test_qwen38_mixed.py:243-281`, and `ignition/recipes/qwen38/mixed_model.py:79-80` constructs the state without them.
7. `ignition/recipes/qwen38/speculative_state.py:250-262` — `prepare_commit(..., feedback=None)`; both callers pass four positional arguments (`ignition/recipes/qwen38/speculative_model.py:244`, `tests/recipes/test_qwen38_speculative_state.py:82`).
8. `ignition/components/dflash2_weights.py:37` and `:43` — the `"dflash"` projector entry (`encoder.fc`, `encoder.output_norm_enc`) and its family label; every recipe loads through the default `"dflash2"` (`ignition/recipes/qwen38/bootstrap.py:352`, `ignition/recipes/muse_glimmer_nvfp4/weights.py:75`, `ignition/recipes/qwen38_dense/dspark_weights.py:16` for `"dspark"`), and only `tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py:123-129` names `"dflash"`.
9. `README.md:95` — the Retired-patches entry states these hooks were left in place; it becomes stale with this change.
10. grepped `feature_taps`, `validate_feature_taps`, `draft_windows`, `prompt_features`, `feedback=`, `projector` and `HybridStatePool.allocate` across `ignition/`, `tests/` and `docs/` (excluding `__pycache__` and `openspec/changes/archive`) — the sites above are all of them. `docs/model-preparation.md:1359` (checkpoint-declared AttnRes feature taps) and `docs/mtp-design.md:188` (MTP draft windows) name other mechanisms and stay.
11. grepped `mixed_prefill` and `dflash2` across `ignition/` — the guards at `ignition/planning.py:589-590` and `ignition/recipes/qwen38/bootstrap.py:132-133`, their doc lines `docs/cli-design.md:447` and `docs/model-preparation.md:630`, and upstream's mixed path (`ignition/recipes/qwen38/recipe.py:141`, `mixed_model.py`, `mixed_shapes` in `ignition/components/capacity.py`) stay untouched; the MTP and DFlash2 serving paths keep their coverage in `tests/recipes/test_qwen38_mixed.py`, `test_qwen38_mixed_cuda.py`, `test_qwen38_dflash_state.py` and `test_qwen38_dflash_preparation.py`.
