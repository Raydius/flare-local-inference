# Rebase PR #229 onto upstream main's runtime-core layout and port Muse Glimmer onto per-call-metadata forwards

Tier: 3
Type: bug

## Stated Requirements

- "Help me understand what is pending for me on this PR https://github.com/NVIDIA/hyperflare-inference/pull/229#issuecomment-6717222575" — the exploration found one item pending on the author: the maintainer's 2026-11-13 request (comment 6861799827) "Please reconcile the branch with main and rerun the checks affected by the resolution." — conflict: the CPU checks were rerun on the rebased branch and on its base; the CUDA checks of the recipe were not, because the operator, asked once during apply, answered "Not free" for GPU 0 and "Skip GPU 1 tests" for GPU 1 (tasks.md task 1.5 `correction:`), and the PR description and comment name them as not run.
- "Should I start it as a change named `rebase-pr229-runtime-core`?" — answered by invoking `/rad-propose`; this change is that rebase.
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."

Assumed: the target is the latest `upstream/main` at apply time (`abd56461` at planning; retarget if it moves, per the operator's earlier "Retarget to e774a468" choice).
Assumed: Muse Glimmer gets its own attention module on the PR branch and `ignition/recipes/muse_spark/attention.py` stays byte-identical to upstream, as on `deploy` ("Glimmer needs to have first-class support without negatively affecting Spark development").
Assumed: commits are rewritten and re-acknowledged at the new `docs/design-review.md` digest by the model credited with the work, and the PR description and comment are drafted for the operator to read before anything is pushed or posted.
Assumed: the exact-branch GPU tests run in a container only on a GPU the operator releases; neither GPU was released during apply, so every CUDA test, export, and serving of the branch stay unrun and are disclosed as such.

## Why

PR #229 (`pr/muse-glimmer`, head `5264a595`, three commits on `d2b5c5f3`) is `CONFLICTING` against `upstream/main` `abd56461`, 128 commits ahead of its base. The maintainer asked for a reconciliation and a rerun of the affected checks. Since the 2026-11-13 request, `main` has renamed the runtime package to the runtime core, given image number 10 to Inkling 2's padded-grid profile, and moved model forwards to per-call metadata with Inductor-planned temporaries. A textual rebase would therefore leave a recipe that conflicts on Protobuf numbering and no longer prepares on `main`.

## What Changes

- Rebase the three PR commits onto the latest `upstream/main`. Resolve the 14 conflicting paths on `main`'s structure, carrying each side's code as it runs today.
- Renumber Muse Glimmer's image profile from 10 to 11 (`MUSE_GLIMMER_IMAGE_PREPARATION = 11`, `ImageSettings.muse_glimmer = 11`, `PaddedGrid = 10, MuseGlimmer = 11` in `helios/src/model/contract.rs`). Move the profile from `runtime_package.proto` to `runtime_core.proto`, and regenerate `runtime_core_pb2.py` with libprotoc 35.1.
- Port the Glimmer recipe onto per-call-metadata forwards. The ported Glimmer-owned files come from the port `deploy` already carries (commit `b5db759a`, plus the Glimmer hunks of `63808d0b` and `6645fa2e`). Fork-only features are removed from them (`--draft-model`, `Recipe.draft_files`), and the PR's own content that `deploy` lacks is kept (the W4A4 input-activation check).
- Drop the PR's widening of `ignition/recipes/muse_spark/attention.py` and its tests. Glimmer uses its own `attention.py`, and a `docs/TODO.md` entry records folding it back onto Spark's attention once Spark accepts per-call metadata.
- Rerun the affected CPU checks on the branch and on pristine `main`, and the GPU-1 CUDA tests of the ported Glimmer modules on the exact branch.
- Rewrite the three commit messages and the PR description in the four-part order with `main`'s vocabulary ("core", not "package"). Re-acknowledge each commit at the current digest (`ed22` at planning). Draft one PR comment. Publish after the operator reads the drafts.

## Capabilities

### New Capabilities
- none

### Modified Capabilities
- `muse-glimmer-serving`: the "Spark unchanged" scenario changes from "Spark attention unchanged after the widening" to "Spark's attention file is upstream's text and Glimmer owns its attention". This corrects the spec to the code `deploy` already runs.

## Impact

PR branch `pr/muse-glimmer` on `origin` (force-push with lease), the PR #229 description, and one PR comment. The `deploy` checkout changes only under this change directory. No export or serving runs.

## Root Cause

order-of-operations: three upstream changes landed after the PR's base `d2b5c5f3`, while the PR still assumed the earlier layout:
- The runtime package was renamed to the runtime core: `runtime_package_pb2.py` was deleted on `main`, and `helios/proto/runtime_core.proto` replaced the package proto.
- Inkling 2 Small's `8628b7c5` took number 10 for `PADDED_GRID_IMAGE_PREPARATION` and `Empty padded_grid` (`upstream/main:helios/proto/runtime_core.proto:374`, `:509`), the numbers the PR's Muse Glimmer profile uses.
- `2b38bf9d`/`4a24d78a` ("Let Inductor plan model temporary storage") moved decoders to `release_outputs()` and `__call__(metadata)`. The PR's decoder and DFlash2 preparation still bind buffers at construction.

The fork met the same three changes on 2026-11-15 when `deploy` moved to `6b68b9b4`. Its resolution (`helios/proto/runtime_core.proto:375`, `ignition/recipes/muse_glimmer_nvfp4/decoder.py:117`) is the port this change reuses. `main` has changed none of the Glimmer interfaces since `6b68b9b4` (`git log 6b68b9b4..upstream/main -- ignition/recipes/muse_spark ignition/components/dflash2.py helios/src/frontend/images helios/proto` is empty).

## Blast Radius

1. helios/proto/runtime_core.proto:375 — Glimmer image enum and oneof member move to 11; on `main`, value 10 is `PADDED_GRID_IMAGE_PREPARATION` (line 374 there) and `padded_grid` (line 509 there).
2. ignition/preparation/runtime_core_pb2.py:1 — regenerated from the merged proto. The PR's `runtime_package_pb2.py` is deleted on `main` (modify/delete conflict).
3. helios/src/model/contract.rs:68 — `ImagePreparationKind` gains `MuseGlimmer = 11` beside `PaddedGrid = 10` (content conflict).
4. helios/src/frontend/images/conditioned.rs:191 — exhaustive profile match gains `MuseGlimmer` beside `PaddedGrid` (content conflict); `mod.rs` and `packing.rs` conflict in the same profile lists.
5. grepped `git merge-tree --write-tree --name-only upstream/main origin/pr/muse-glimmer` — also `helios/src/runtime_core/describe.rs`, `helios/src/runtime_core/protobuf.rs` (the PR's `runtime_package/` readers renamed), `ignition/components/dflash2.py`, `ignition/recipes/__init__.py:42`, `tests/cli/test_models.py`, `tests/cli/test_planning.py`, `tests/preparation/test_checkpoint_source.py`, `tests/preparation/test_frontend_metadata.py`: 14 conflicting paths in total.
6. ignition/recipes/muse_glimmer_nvfp4/decoder.py:117 — `deploy`'s per-call `__call__(metadata)` port; the PR head's decoder binds buffers at construction and does not run on `main`'s forward protocol.
7. ignition/recipes/muse_glimmer_nvfp4/dflash.py:14 — `deploy` prepares the drafter through `DFlash2Preparation` with the Inductor `DraftScratch`; the PR head uses the pooled classes, which lack `release_outputs()` on `main`.
8. ignition/recipes/muse_glimmer_nvfp4/recipe.py:21 — `plan.draft_model` and `recipe.draft_files` (line 27, 76) are fork-only (`git grep -c draft_files upstream/main -- ignition` is 0) and stay off the branch; `mtp_only` (line 33) is upstream's (`upstream/main:ignition/recipes/declaration.py:123`) and is carried.
9. ignition/recipes/muse_spark/attention.py:70 — `main`'s Spark attention still binds buffers at construction; the PR's widening (prepare_projection, `output_norm`) is dropped and `tests/recipes/test_muse_spark_attention.py` returns to `main`'s text.
10. docs/TODO.md:40 — the fold-back entry for Glimmer's attention is carried onto the branch.
11. ignition/preparation/checkpoint.py:401 — the PR's `total_size` acceptance of the shard file-byte sum; `main`'s `949966ba` changed `checkpoint_source.py` and its tests, hence the `test_checkpoint_source.py` conflict.
12. ignition/recipes/muse_glimmer_nvfp4/config.py:62 — the PR's W4A4 input-activation check, absent on `deploy`, is kept on the branch (`git show origin/pr/muse-glimmer:ignition/recipes/muse_glimmer_nvfp4/config.py`, lines 63-69).
13. .githooks/commit-msg:62 — rejects a `design-review=` digest other than the staged document's first four SHA-256 hex characters; `upstream/main:docs/design-review.md` digests to `ed22`, so the three `3ad6` acknowledgments go stale.
14. openspec/specs/muse-glimmer-serving/spec.md:54 — the "Spark unchanged" scenario describes the dropped widening; the delta corrects it.
15. openspec/specs/upstream-contribution-branch/spec.md:13 — the "carries only the patch" requirement that excludes the fork-only hunks in item 8; unchanged.
