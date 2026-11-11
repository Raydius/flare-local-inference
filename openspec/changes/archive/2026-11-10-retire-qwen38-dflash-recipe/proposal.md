# Retire the fork's qwen38_dflash recipe in favor of upstream's qwen38 DFlash2 path

Tier: 3
Type: feature

## Stated Requirements

- "it does not.  let's retire it." — in answer to: decode throughput while long prompts prefill concurrently is the one case the comparison did not measure; if that workload matters it is the open question before retiring `qwen38_dflash`.
- "Arm Q (Recommended)" — in answer to: which upstream-path package should serve in place of Rebase3.

Assumed: retiring removes the recipe package `ignition/recipes/qwen38_dflash/`, its registration and aliases (`qwen3.8-flash-dflash2`, `qwen3.8-flash-dflash2-yarn4`), its tests, the two qualify tools that import it, and its documentation; README moves its entry to Retired patches.
Assumed: shared code stays untouched: `ignition/components/` (including the `admission` and `draft_page_blocks` parameters of `PreparedParallelDraftRound`, whose only caller is the recipe), `ignition/reference/qwen38_dflash2.py` (used by Muse Glimmer's tests and `qualify/dflash2_acceptance.py`), history drafts, the two mixed speculative export fixes outside the recipe (`ecb6c1dc`, `388d3d21`; the third, `a088b0b6`, lives in the recipe and goes with it), and every Helios file.
Assumed: the fork's hooks inside upstream-owned `ignition/recipes/qwen38/` files (feature taps, the DFlash state layout, mixed-round absorption, verification feedback) also stay in this change, although the recipe was their only production caller; README lists them as left without a caller, and restoring those files to upstream's text is a later change.
Assumed: history drafts on Qwen 3.8 DFlash2 rounds end with the recipe (upstream's path declares no copy inputs); the served profiles run with history drafts off, and the 2026-11-06 A/B measured them slower on this draft, so nothing in use is lost.
Assumed: recorded results under `qualify/results/` and archived changes stay as history.
Assumed: packages already exported with the recipe keep loading (serving reads the package, not the recipe); they can no longer be re-exported on an image built after this change.
Assumed: arm Q's profile *2026-11-10 Flare DFlash2 Q c16* (entry `gb202-tp1-202611110138`, image `local/flare-ignition:f83d3120`) becomes the served profile; Rebase3's profile and package stay on disk as the rollback; the arm A and arm M packages, which no recommendation uses, are removed to free disk.
Assumed: upstream issue #210 (whether mixed prefill rounds may embed a DFlash2 draw) is left as it is; closing or commenting on it is the operator's.

## Why

The fork carries a second Qwen 3.8 DFlash2 recipe (about 1,100 lines plus tests) whose one addition over upstream's `--recipe qwen38` with speculative method `dflash2` is mixed prefill. Measured at 16 sequences and a 1M context (`qualify/results/dflash2-paths-c16/README.md`), it does not beat upstream's path on `code_edit` tokens per second, cannot hold a 1M prompt (909,312 KV tokens), and exports in 92 minutes against 28; the operator does not need the one unmeasured case. Each upstream rebase has cost carry commits to keep it working.

## What Changes

- **BREAKING**: `ignition export` no longer knows `--recipe qwen38_dflash` or the aliases `qwen3.8-flash-dflash2` and `qwen3.8-flash-dflash2-yarn4`. Qwen 3.8 DFlash2 exports use `--recipe qwen38` with speculative method `dflash2` (and `--rope-parameters` for 1M).
- **BREAKING**: `--history-drafts` has no Qwen 3.8 DFlash2 package to act on: upstream's path declares no copy inputs. History drafts remain for MTP rounds and for other recipes' parallel-draft rounds.
- **BREAKING**: no alias pins `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`; that repository id resolves to `qwen38` with MTP proposals unless speculative method `dflash2` is given.
- The recipe package, its tests, `qualify/dflash2_tap_check.py`, `qualify/mixed_prefill_check.py` and their tests are deleted; tests that list aliases or parametrize over recipes lose the recipe's entries; `--draft-model` validation cases move to `qwen38` DFlash2 and Muse Glimmer.
- Documentation: README's Active-patches entry moves to Retired patches with the measured reason; `docs/model-preparation.md` drops the "Qwen3.8 Flash DFlash2" section in favor of upstream's path with the fork's flags; `docs/cli-design.md` and `docs/TODO.md` drop the recipe's entries.
- Serving: the console serves arm Q's profile; arm A and arm M packages are removed; the vault tracker records the retirement.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `qwen38-dflash2-speculation`: every requirement is removed; the capability described the retired recipe.
- `history-drafts`: the requirement "A proposal-only round may take its admission from the owner" keeps its behavior; its scenario stops naming the retired recipe.

## Impact

- Flare source under `ignition/recipes/`, `tests/`, `qualify/`, `docs/`, `README.md`. No Helios, component, kernel, or package-format change.
- Console (`~/LocalApps/vllm-env`): no code change; one serving switch, two package removals.
- Console profiles whose `export:--recipe` is `qwen38_dflash` (Rebase3, *2026-11-04 Flare DFlash2*, arm M) keep serving their existing packages on their existing images and cannot export on a newer image.
- GPU 0: one stop of the served instance to switch from Rebase3 to arm Q; no export.

## Blast Radius

1. ignition/recipes/__init__.py:29 — imports the recipe; line 40 registers it in `recipes()`.
2. grepped 'git ls-files ignition/recipes/qwen38_dflash tests/recipes/qwen38_dflash' — 12 tracked files (recipe.py, config.py, bootstrap.py, speculative_model.py, export.py, __init__.py; six test files), all deleted.
3. tests/recipes/test_declarations.py:463 — `DFLASH = resolve("qwen3.8-flash-dflash2")` and the tests through line 625 that use it.
4. tests/cli/test_models.py:25 — alias tuple lists both aliases; lines 113 and 340-357 resolve the recipe by name.
5. tests/cli/test_commands.py:57 — "supported models" message lists both aliases.
6. tests/cli/test_draft_checkpoints.py:90 — `_dflash_plan` defaults to the recipe; lines 107, 129-130, 147-150 parametrize over it.
7. tests/cli/test_mtp_only.py:109 — names the recipe among those with a `draft_checkpoint` field.
8. tests/preparation/test_frontend_metadata.py:23 — names the recipe among Qwen recipes.
9. tests/recipes/test_qwen38_mixed.py:126 — imports the recipe's `state_layout` for one DFlash layout case.
10. tests/recipes/test_qwen38_mixed_cuda.py:151 — one DFlash mixed-state test importing the recipe (lines 168-170).
11. qualify/dflash2_tap_check.py:93 — imports the recipe's config, bootstrap and speculative model; tests/reference/test_dflash2_tap_check.py:6 tests it.
12. qualify/mixed_prefill_check.py:53 — imports the recipe's speculative model and bootstrap.
13. docs/model-preparation.md:523 — "Qwen3.8 Flash DFlash2" section through line 563; lines 551-563 document the surviving `dflash2_long_context` and `dflash2_draft_sweep` tools and are kept.
14. docs/cli-design.md:396 — `--recipe NAME` row gives `qwen38_dflash` as its example; line 397's `--draft-model` row names it too.
15. docs/TODO.md:7 — open items about the recipe at lines 7 and 23; line 30's item on draft identity assumes the recipe declares its own.
16. README.md:29 — Active-patches entry for the recipe; line 6 says the fork recipe stays separate; line 36 cites the retired recipe's package; line 41's entry lists `a088b0b6`, which lives in the recipe; line 44's `--recipe` verdict; line 53's qualification-tools entry.
17. openspec/specs/qwen38-dflash2-speculation/spec.md:9 — twelve requirements, all about the recipe.
18. openspec/specs/history-drafts/spec.md:61 — a scenario names the qwen38_dflash round.
19. ignition/components/parallel_draft.py:111 — `admission` and `draft_page_blocks` parameters; grepped 'proposal_only=\|admission=' across ignition/ — the recipe is the only caller passing `admission` or `draft_page_blocks`; minimax_m3 and deepseek41 pass `proposal_only` alone. Left in place.
20. grepped 'qwen38_dflash2' across ignition/ qualify/ tests/ — `ignition/reference/qwen38_dflash2.py` is imported by tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py:18, qualify/dflash2_acceptance.py:233 and tests/reference/test_qwen38_dflash2.py:10; it is the DFlash2 reference model, not the recipe, and stays.
21. grepped 'qwen38_dflash\|qwen3.8-flash-dflash2' across ignition/ helios/ flare-cli/ outside ignition/recipes/qwen38_dflash/ — only ignition/recipes/__init__.py; no Helios or CLI source names the recipe.
22. grepped 'dflash2_tap_check\|mixed_prefill_check' across README.md docs/ qualify/ tests/ — tests/reference/test_dflash2_tap_check.py only.
23. grepped 'ls openspec/lessons.md' — no lessons file in this repository.
24. ignition/recipes/qwen38/dflash.py:169 — upstream's DFlash2 preparation returns no copy buffers, so its packages declare no copy inputs and history drafts do not apply to them.
25. ignition/recipes/qwen38/state.py:37 — fork hooks `mtp=` and `draft_windows=` (also ignition/recipes/qwen38/decoder.py:204 feature taps, ignition/recipes/qwen38/mixed_model.py:42 absorption and speculative round, ignition/recipes/qwen38/mixed_state.py:138 features, ignition/recipes/qwen38/speculative_state.py:185 feedback); the recipe was their only production caller. Left in place.
26. ignition/recipes/qwen38_dflash/recipe.py:20 — the only `Checkpoint` pinning the QAD target; ignition/recipes/__init__.py:254 then resolves that repository id through hub metadata to `qwen38`.
27. tests/cli/test_models.py:339 — `test_a_named_recipe_prepares_a_directory_of_its_architecture` needs two recipes of one architecture; none remain after the retirement.
28. qualify/speculative_benchmark.py:376 — `validate_arm` requires eight slots and a mixed graph, so `dflash2_served.py` and `dflash2_draft_sweep.py` accept only packages of the retired recipe; `qualify/dflash2_console_probe.py` measures the surviving path. Not changed here; the live change `reuse-validate-arm-in-console-probe` edits the same validator.
29. ignition/recipes/qwen38/bootstrap.py:263 — the surviving path writes the pinned draft repository and revision into the package identity even when `--draft-model` supplied a folder; the retired recipe hashed the draft's shard headers. Recorded in docs/TODO.md, not changed here.
30. ignition/preparation/rebuild_manifest.py:216 — existing cores record `recipe = "qwen38_dflash"`; helios/src/core/conversion.rs:2728 compares it as a string only, so loading is unaffected.
31. tests/components/test_parallel_draft_admission.py:111 — the remaining caller of `admission` and `draft_page_blocks`.
