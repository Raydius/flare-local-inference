# Design — drop-dflash2-mixed-round-hooks

## Approach

Delete the five hooks the retired `qwen38_dflash` arm added to shared Qwen3.8 Flash preparation, plus the tests that reach only them. No serving path changes: each parameter is reached today only by a test of the retired arm, and none appears in a production call site. The regions and attributes those hooks share with live paths stay, which is the one place this change can go wrong.

- `ignition/recipes/qwen38/state.py` — drop `mtp=` and `draft_windows=` from `HybridStateLayout.build` (`:37`) and from `HybridStatePool.allocate` (`:134-135`, forwarded at `:144-145`), their docstring sentences (`:40-46`) and the window validation (`:50-52`), and delete the `draft{layer}.key`/`value` E4M3 block (`:81-85`). Keep the MTP feedback and draft-ring block (`:87-100`) under `if speculative_tokens and draft_config is None:` — `mtp` defaults to `True` and no production caller sets it, so that block runs for every MTP export and feeds `decoder_state.py:288-291`, `speculative_state.py:260-261`, `speculative_model.py:144-145,303-304`, `mixed_model.py:160-161` and `bootstrap.py:700`. Keep `draft_config=` and its `draft_context_specs` block (`:101-104`): the live DFlash2 draft rows come from there (`ignition/recipes/qwen38/bootstrap.py:407,412`), and they use the same `draft{layer}.key`/`value` names, so deleting the wrong block silently shrinks the DFlash2 state row.
- `ignition/recipes/qwen38/decoder.py` — delete `validate_feature_taps` (`:224-232`), the `feature_taps`/`auxiliary` parameters (`:255`), their validation and assignment (`:283-284`), and the per-layer tap block (`:451-455`), which sits directly below the live `context_layers` block (`:448-450`). Keep `self.auxiliary = None`: `capture_context` reassigns it to the DFlash2 context-capture buffer (`:366-373`), `release_outputs` clears it (`:406`), and `ignition/recipes/qwen38/dflash.py:154` passes it to `PreparedContext` (`:247`). The checkpoint-declared AttnRes taps the live draft consumes are a separate mechanism in `ignition/components/dflash2.py` (`docs/model-preparation.md:1359`) and stay.
- `ignition/recipes/qwen38/mixed_state.py` — delete `features`/`prompt_features`/`decode_features`, their shape validation (`:201-209`) and the two gather-copies in `finish_runtime` (`:298-300`); `PreparedMixedRound` builds the state without them (`ignition/recipes/qwen38/mixed_model.py:79-80`). The `features=` at `bootstrap.py:573` is the separate `VisionFeatureState` (`:442`) consumed by `MixedEmbeddingInjection`, and stays.
- `ignition/recipes/qwen38/speculative_state.py` — delete `feedback=` from `prepare_commit` (`:250-262`) so the commit binds the pool's `target_residual` and `target_residual_position` regions again; both callers pass four positional arguments (`ignition/recipes/qwen38/speculative_model.py:244`, `tests/recipes/test_qwen38_speculative_state.py:82`). Standalone DFlash2 rounds go through `PreparedParallelDraftRound` (`ignition/components/dflash2_preparation.py:48-55`) and never reach this method.
- `ignition/components/dflash2_weights.py` — drop the `"dflash"` entries from `_PROJECTION_NAMES` and `_FAMILY` (`:37`, `:43`); `_FAMILY` only formats error text (`:154-160`). Loaders use `"dflash2"` (`ignition/recipes/qwen38/bootstrap.py:352`, `ignition/recipes/muse_glimmer_nvfp4/weights.py:75`) or bind `"dspark"` (`ignition/recipes/qwen38_dense/dspark_weights.py:16`).
- Tests — delete `test_packed_features_are_gathered_per_role` (`tests/recipes/test_qwen38_mixed.py:219-284`) and `test_projector_switch_keeps_the_dflash2_tensor_set_apart` (`tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py:123-129`); in `test_residual_checkpoint_follows_the_target_residual_region` (`:194-216`) delete the `dflash` layout (`:201`), its comment clause (`:198`) and the `(dflash, False)` tuple entry (`:205`), keeping the MTP and no-draft cases as the CPU coverage of the state gate.
- `README.md:95` — the entry states these hooks were "Left without a production caller, unchanged"; rewrite it as present-state fact: the hooks are removed, mixed rounds are MTP-only, DFlash2 serves with separate prefill and verification graphs. Keep the entry's second clause about the hooks removed with the `6b68b9b4` move.
- Cleanup — delete the stale `ignition/recipes/qwen38_dflash/` directory, which holds only `__pycache__` of the deleted recipe and pollutes greps; nothing imports it and it is outside the `_identity` hash globs.

Specs: `skip_specs: true` in `.openspec.yaml` — no capability spec documents these hooks (the `qwen38-dflash2-speculation` capability was retired with the recipe on 2026-11-10), so the change adds, changes and removes no requirement; a delta would invent one.

Carried behavior kept: both guards (`ignition/planning.py:589-590`, `ignition/recipes/qwen38/bootstrap.py:132-133`), their documentation (`docs/cli-design.md:447`, `docs/model-preparation.md:630`), and upstream's MTP mixed path (`ignition/recipes/qwen38/recipe.py:141`, `mixed_model.py`, `mixed_shapes` in `ignition/components/capacity.py`).

Core compatibility and weight reuse: no production call passed `mtp` or `draft_windows`, so the region set, `row_bytes` (`state.py:103`) and `snapshot_capacity` (`bootstrap.py:408`, `planning.py:526`) are byte-identical — pinned by `tests/recipes/test_declarations.py:67-71`, `tests/recipes/test_qwen38_state.py:31` (exact 116,505,344) and `test_qwen38_dflash_preparation.py:22,38-40`, all four of which stay in the must-pass list. The `_identity` digest (`bootstrap.py:247-252`, which hashes every `.py` under `ignition/recipes/qwen38/` and `ignition/components/`) changes, but it is written once into the exported core's server config (`:708`) and read only for display and legacy summaries (`helios/src/core/cache.rs:96-101`); the only load-time comparison is the model id (`heliopause/src/export/weight_reuse.rs:108`), so previously exported cores load without re-export and `--reuse-weights` entries stay valid (`ignition/cli/serve.py:145-153`, `weight_reuse.rs:88-91,120-146`).

### Explicit Rejections

- Keeping the hooks behind a flag or a recipe capability — rejected because AGENTS.md prohibits toggle-selected paths and the arm they serve is the one upstream rejected; the peer quorum made the same point.
- Removing `mixed_model.py`, `mixed_state.py`'s mixed machinery or `mixed_shapes` along with the hooks — rejected because upstream's MTP mixed path owns them and `tests/recipes/test_qwen38_mixed_cuda.py` covers them.
- Deleting `test_packed_features_are_gathered_per_role` outright — rejected in favour of stripping only its feature-buffer parts, because it is the only default-enabled CPU test driving `PreparedMixedState.prepare_hidden`/`prepare_residual`/`finish_runtime` including the `prompt_row=-1` lane; the CUDA coverage is gated behind `FLARE_RUN_CUDA_TESTS`.
- Removing `projector=` itself rather than the `"dflash"` entry — rejected because `"dspark"` uses the same mechanism (`ignition/recipes/qwen38_dense/dspark_weights.py:16`).

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.89
Caller Agreement: 0.85
Arbiter: ok
Sources:
- anthropic (claude-haiku-5-5) — ok, agreement 0.92
- local-ai (qLLM) — ok, agreement 0.90
- google (models/gemini-flash-lite-latest) — ok, agreement 0.85
- deepseek (deepseek-flash) — cancelled, agreement null (quorum met before it responded)

Certainty 0.89 and caller agreement 0.85 sit under the 0.9 floors in `radflow/openspec-config.yaml`; recorded as warns, the quorum stands.

### Review Feedback

Consensus feedback recorded verbatim:
- "Delete tests that exercise only the retired arm, converting any that remain into guard or absence tests." — ACCEPTED: `test_residual_checkpoint_follows_the_target_residual_region` keeps its MTP and no-draft cases as the absence check for the draft-window rows, and `test_packed_features_are_gathered_per_role` is converted rather than deleted.
- "Rewrite README.md:95 to record the removal and the policy (DFlash2 uses separate graphs; mixed rounds are MTP-only)." — ACCEPTED: the entry is rewritten with both statements.
- "Recipe hash changes will invalidate preparation weight reuse; accept this as expected." — ACCEPTED with a correction: the digest changes but gates nothing at load or reuse time (evidence in Approach); recorded rather than assumed.
- "Verify by grepping for each symbol, checking that no production caller remains, running the remaining CPU suites, and smoke-testing the active paths at concurrency 1 and 8." — ACCEPTED: tasks carry the per-symbol sweeps, the CPU suites, and the GPU 1 CUDA suites for the mixed and DFlash2 paths.
- "Removal is the correct and efficient choice; do not keep the arm behind a flag." — ACCEPTED: no flag, capability flag or opt-in added.

## Bomb-squad findings

- HIGH — `state.py:37` and `:135` default `mtp=True`, so the gate at `:87` is live for every MTP export and the blast radius's phrasing invited deleting the region block with the parameter — fixed: task 1 keeps `:87-100` verbatim and must pass `tests/recipes/test_qwen38_state.py:92-102` and `test_qwen38_dflash_preparation.py:38-40`.
- HIGH — the deleted `draft_windows` block (`state.py:81-85`) and the live DFlash2 draft rows (`:101-104`, via `draft_context_specs`) emit the same `draft{layer}.key`/`value` names — fixed: task 1 deletes only `:81-85` and pins `test_qwen38_dflash_preparation.py:23-27` as must-pass.
- MEDIUM — `PreparedDecoder.auxiliary` is overloaded: the taps parameter sets it, `capture_context` (`decoder.py:366-373`) reassigns it to the live DFlash2 context buffer, and `ignition/recipes/qwen38/dflash.py:154,247` reads it — fixed: task 3 keeps `self.auxiliary = None` and deletes only `:451-455`.
- MEDIUM — the stated test ranges were wrong: `test_packed_features_are_gathered_per_role` spans `tests/recipes/test_qwen38_mixed.py:219-284`, and deleting only `:201` orphans `:198`, `:204-205` — fixed: tasks 2 and 4 carry the corrected ranges.
- Coverage shrink from deleting the packed-features test (the only un-gated test of `PreparedMixedState`) — fixed: task 4 converts that test instead of deleting it.
- Loss of the Glimmer projector-separation test — declined: the guardrail that matters, rejecting an `encoder.fc`-named checkpoint under the default projector, survives in the fault table at `tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py:96-99`.
- Stale `ignition/recipes/qwen38_dflash/__pycache__` of the deleted recipe pollutes greps — fixed: task 6 removes the directory; outside the `_identity` globs, so no identity impact.
- Hidden-reachability sweep (subclasses, `**kwargs`, `getattr`, `functools.partial`, cross-recipe imports, `ignition/reference/`) found no further caller of any candidate hook — hunt completed, no latent defects of this class beyond the items above.
