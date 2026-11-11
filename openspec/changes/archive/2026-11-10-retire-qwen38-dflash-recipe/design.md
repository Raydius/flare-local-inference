# Design — retire-qwen38-dflash-recipe

## Approach

The change deletes one recipe and everything that names it, and switches the served profile. It adds no code path. See proposal.md for why.

1. Serving first, no export. Arm Q's package (`gb202-tp1-202611110138`, image `local/flare-ignition:f83d3120`, profile *2026-11-10 Flare DFlash2 Q c16*) already exists and passed the 1M passkey probe. Before the switch, check off-GPU that the profile passes no `--history-drafts` and that `flare inspect` reads the entry (16 slots, 1,048,576-token context). One stop of the Rebase3 instance, one start of arm Q, then a short served check (a chat completion and `/health`). Rebase3's profile, package and image `7ebef730` stay on disk as the rollback: starting that profile again needs no export.
2. Source removal in one commit, so no intermediate tip imports a missing module: delete `ignition/recipes/qwen38_dflash/` and its two lines in `ignition/recipes/__init__.py`; delete `tests/recipes/qwen38_dflash/`, `qualify/dflash2_tap_check.py`, `qualify/mixed_prefill_check.py`, `tests/reference/test_dflash2_tap_check.py`; in the eight other test files remove or retarget exactly the entries the latent-defect hunt tabulated (test_declarations.py lines 463-626 deleted; test_models.py alias tuple, message and `test_a_named_recipe_never_falls_back` retargeted to `qwen38`, `test_a_named_recipe_prepares_a_directory_of_its_architecture` deleted because no two recipes share an architecture; test_commands.py message; test_draft_checkpoints.py rows and default name; name tuples in test_mtp_only.py and test_frontend_metadata.py; the DFlash layout case in test_qwen38_mixed.py rebuilt on `HybridStateLayout.build(..., mtp=False, draft_windows=…)` with the window tuple inlined; the DFlash test in test_qwen38_mixed_cuda.py deleted).
3. Documentation in a second commit: README (entry moved to Retired patches with the measured reason; line 6; the `a088b0b6` hash dropped from the mixed-export entry; the `--recipe` verdict reworded, since the flag now only validates; the qualification-tools entry says `dflash2_served.py` and `dflash2_draft_sweep.py` measure packages of the retired recipe only; a "left without a caller" list: `PreparedParallelDraftRound`'s `admission` and `draft_page_blocks`, the hooks in `ignition/recipes/qwen38/`, `projector="dflash"` in `dflash2_weights.py`), `docs/model-preparation.md` (section rewritten for `ignition export MODEL --speculative-method dflash2` with `--rope-parameters` and `--draft-model`; the tool paragraphs kept), `docs/cli-design.md` (two rows), `docs/TODO.md` (recipe items removed with their result pointers moved to README's tools entry; the identity item rewritten to state that the surviving path records the pinned draft's repository and revision even for a `--draft-model` folder).
4. Specs: the two deltas under `specs/` are synced by `/rad-sync` before archive.
5. Housekeeping after the switch has served: remove the arm A and arm M packages with `flare models remove`; record the retirement in the vault tracker.

Verification is baseline-relative on CPU: the failing node IDs of `tests/cli`, `tests/recipes/test_declarations.py`, `tests/recipes/test_qwen38_mixed.py`, `tests/preparation/test_frontend_metadata.py`, `tests/reference` and `tests/components/test_parallel_draft_admission.py` on the tip are a subset of those on the pre-change tip, and every node ID that disappears belongs to a deleted or retargeted test listed above. `git grep` for the recipe name and aliases returns only `openspec/`, `qualify/results/` and README's Retired entry. `ignition export --help` output and the "supported models" message no longer list the aliases. No GPU test runs: the change removes code and the served package was qualified by the comparison study.

No image is built. The served image predates the removal, and a later image is built when a change needs one.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 4/3, escalated to tier 1)
Certainty: 0.867
Caller Agreement: 0.85
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.95
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.85
- local-ai-qllm (qLLM) — ok, agreement 0.8

Certainty and caller agreement are under the configured floor of 0.9. The differences are the two REJECTED points below and the preflight and soak points, which the approach now carries.

### Review Feedback

Consensus feedback recorded verbatim:
- "Preflight the replacement package off-GPU (signature, manifest, image ID, KV capacity for 1M tokens at 16 sequences) and ensure no operational config passes the retired recipe alias." — ACCEPTED: step 1 inspects the entry and the profile before the stop; the proposal lists the profiles that still name the recipe and what happens to them.
- "Switch the served production package to the already-exported upstream path package in a single production stop, keeping the old package and image available for rollback." — ACCEPTED: step 1.
- "In the code repository, retarget sibling/shared tests and qualification tools that depend on surviving code paths." — ACCEPTED for tests (step 2). REJECTED for the two qualify tools: both exist to check the retired recipe's bootstrap and have no surviving subject.
- "Unregister and delete the recipe files, tests, documentation, TODOs, README entries, and capability specs in an atomic step." — ACCEPTED in part: source and tests are one commit; documentation is a second commit so each message explains one thing, and no tip between them imports a missing module.
- "Verify via test-ID diffs (ensuring no silent test skips), CLI checks, and smoke tests (1M-prompt, 16-sequence, concurrency 1/8 throughput)." — ACCEPTED for test-ID diffs and CLI checks. REJECTED for repeating the throughput and 1M runs: `qualify/results/dflash2-paths-c16/` measured this exact package on this image hours earlier.
- "Delete old packages only after a soak/rollback window closes." — ACCEPTED: Rebase3 stays until the operator says otherwise; only the arm A and arm M packages, which nothing serves, are removed.
- "Leave optional constructor parameters in shared components if they are part of upstream APIs, used by other recipes, or if removing them requires GPU validation of surviving recipes. Remove them only if proven fork-only/dead and safe, using a separate commit/change." — ACCEPTED: components and the `qwen38/` hooks stay; README lists them for a later change.
- "Mitigated because packages are self-describing and Helios does not import Python recipes; keep old Helios support for mixed-prefill graph/schema features." — ACCEPTED: no Helios file changes.
- "Mitigated by annotating recorded measurement results with their provenance instead of deleting them, and keeping the reference model module (`ignition/reference/qwen38_dflash2.py`) intact." — ACCEPTED: results and the reference module stay; README's Retired entry points at the results.
- "Compare pre- and post-change coverage of `parallel_draft.py` and ensure surviving tests cover the shared draft component." — ACCEPTED as a named check, not a coverage run: `tests/components/test_parallel_draft_admission.py` and upstream's `tests/recipes/test_qwen38_dflash_preparation.py` and `test_qwen38_dflash_state.py` stay and are in the baseline-relative set. Not replaced: the CUDA parity check of the shared component against the reference on the pinned Flash-Next draft (`tests/recipes/qwen38_dflash/test_draft_cuda.py`); README records the gap.
- "Keep the old production package and its image/serving config in the operations console for a defined soak window (e.g., one representative traffic cycle or 30 days) to allow a zero-export rollback via an extra stop." — ACCEPTED: no date is set here; removal of Rebase3 is the operator's.
- Performant, Correct, Efficient — ACCEPTED: the peers' three-heading summary restates the points above (one stop, no re-export; test-ID diff and compatibility by self-describing packages; delete instead of carrying fork code).

### Bomb-Squad Findings

Root-philosophical cause: the recipe was retired as a directory, but it was also the only production caller of hooks the fork added to upstream-owned files, the only pin of one checkpoint, and the only path for one feature.

1. History drafts on Qwen 3.8 DFlash2 are lost and no stated requirement covers it (ignition/recipes/qwen38/dflash.py:169) — ACCEPTED: proposal gains a BREAKING line and an Assumed line; step 1 checks the profile passes no `--history-drafts`.
2. Fork hooks in `ignition/recipes/qwen38/` lose their only production caller (ignition/recipes/qwen38/state.py:37) — ACCEPTED as disclosure: listed in the proposal and README; removal is a later change.
3. One of the three mixed-export commits lives in the recipe (`a088b0b6`) — ACCEPTED: Assumed line and README entry corrected.
4. The only pin of the QAD target disappears and the repository id resolves to MTP (ignition/recipes/qwen38_dflash/recipe.py:20) — ACCEPTED: BREAKING line; the spec delta's Migration text gives the MODEL forms.
5. `--recipe NAME` can no longer select between recipes (tests/cli/test_models.py:339) — ACCEPTED: test deleted, README verdict reworded.
6. `dflash2_served.py` and `dflash2_draft_sweep.py` reject every package of the surviving path (qualify/speculative_benchmark.py:376) — ACCEPTED as disclosure in README; the tools are not changed here.
7. Documentation lines the radius missed (README.md:6, docs/cli-design.md:397, docs/TODO.md:30) — ACCEPTED: radius and step 3 extended.
8. The surviving path's package identity ignores a `--draft-model` folder (ignition/recipes/qwen38/bootstrap.py:263) — ACCEPTED as a rewritten TODO item; not changed here.
9. Existing cores record the recipe name (ignition/preparation/rebuild_manifest.py:216) — ACCEPTED: one radius line; loading is unaffected.
10. `admission` and `draft_page_blocks` keep a test caller (tests/components/test_parallel_draft_admission.py:111) — ACCEPTED: in the baseline-relative set.
11. `projector="dflash"` has no production caller — ACCEPTED: added to README's left-without-a-caller list.

## Risks

- A profile that names the retired recipe is started on a newer image: the export fails with an unknown recipe. Mitigation: BREAKING line; those profiles keep their images.
- Arm Q misbehaves in service. Mitigation: start the Rebase3 profile again; no export.
- A deleted test was the only cover for surviving code. Mitigation: the test-ID diff lists every disappearing ID; the one known gap is recorded.
