# Rebase pr/muse-glimmer onto upstream main d2b5c5f3 and bring its messages to the four-part rule

Tier: 2 — items 2 and 5-7 cite files only to show they are untouched; the edit is one regenerated file plus three commit messages and the PR body
Type: bug

## Stated Requirements

- "Help me understand the conflicts identified in `ignition/preparation/runtime_package_pub2.py` (https://github.com/NVIDIA/hyperflare-inference/pull/229) -- 1) is this something that can easily be resolved, and 2) will it also get resolved if I rebase with the latest upstream?"
- "Does that follow the contribution rules?" — asked of the exploration's plan (rebase, regenerate `runtime_package_pb2.py`, force-push); the answer added three obligations from upstream `fc2c914c`: fresh `design-review=3ad6` acknowledgments, four-part commit messages and PR description, and validation re-run on the new base.
- "It is inaccurate to call this authored by Claude Opus 5.5 when the actual work was done by Fable 5.1 -- this is just messaging."
- "They need to coexist because #192 is for an entirely different checkpoint that has entirely different (incompatible) topology." — conflict: the two checkpoints' `config.json` architecture fields are identical (52 layers, 39 sliding / 13 full, hidden 6656, 32 heads / 2 KV heads, vocab 202048); what differs and is incompatible is the quantization layout (`original-el8`: NVFP4 W4A4 group-16 on every decoder Linear, BF16 `lm_head`; `nvidia` at `47818374`: 180 FP8 and 213 NVFP4 W4A16 targets including `lm_head`), so the posted argument names the quantization layout
- "#192 renames" — chosen for the alias question: #229 keeps `muse-glimmer-30b-nvfp4` and the comment asks #192 to take a different alias
- "PR #229 is based on a checkpoint that is more appropriately tagged as NVFP4 https://huggingface.co/original-el8/Muse-Glimmer-30B-NVFP4" — the reason #229 keeps the `-nvfp4` alias: the `original-el8` card states "NVIDIA NVFP4 W4A4" on "416 decoder linear projections across 52 layers", while the `nvidia` card calls its checkpoint "a mixed precision checkpoint with W4A16 NVFP4 weights and FP8 weights and activations" (AutoQuantize, 6.6 effective bits per parameter, BF16 fallback allowed)

## Why

PR #229 (`pr/muse-glimmer`, head `8d3c0e81`, three commits on `91cfb369`) is `CONFLICTING` against `upstream/main` `d2b5c5f3`, ten commits ahead of its base. Rebasing pulls in `fc2c914c`, which changes `docs/design-review.md` (digest `4d3a` → `3ad6`) and requires every commit message and PR description to answer why, what changed, claimed benefit, and evidence, in that order. The three existing messages and the PR body do not.

## What Changes

- Rebase the three PR commits onto `upstream/main` `d2b5c5f3` in the worktree `/tmp/glimmer-pr`. Resolve the only conflict, `ignition/preparation/runtime_package_pb2.py`, by regenerating it from the merged `helios/proto/runtime_package.proto` with libprotoc 35.1 (`/tmp/protoc-bin/protoc`). This libprotoc reproduces both sides' committed files byte for byte. No hand-merging.
- Re-run the PR's CPU validation (pytest suites and the Helios `cargo test` filters) on the rebased head and on `d2b5c5f3`, comparing the failure sets.
- Rewrite the three commit message bodies in the four-part order and replace their validation figures with the re-run's. Review each against the staged `3ad6` checklist and prose skill, and carry one `Design-Signed-Off-By: …; design-review=3ad6` acknowledgment, accepted by `.githooks/commit-msg`.
- Rewrite the PR #229 description in the same order. Correct "Two commits" to three and "four concurrent requests" to the declared eight-default and sixteen-ceiling. Drop the closing review assurance, which the new rule classes as a process assurance.
- Force-push with lease to `origin pr/muse-glimmer` and post one PR comment stating the resolution and the re-run results.
- Answer the `triage:design-review` overlap with a coexistence argument in the PR #229 description and comment. The two checkpoints share one architecture but have incompatible quantization layouts, and each recipe rejects the other's checkpoint. #229 keeps `muse-glimmer-30b-nvfp4`, and the comment asks PR #192 to take a different alias.
- Out of scope: renaming #192 or editing its branch, the maintainer's final ruling, and GPU export or serving of the branch.

## Root Cause

order-of-operations: upstream's Kimi K3.1 change `68bbd6dd` added `RebuildManifest.online_mxfp8_scope = 30` (`helios/proto/runtime_package.proto:118` on `upstream/main`). The PR's first commit `ee24d5fe` adds `MUSE_GLIMMER_IMAGE_PREPARATION = 10` and `MuseGlimmerImageSettings` (after `helios/proto/runtime_package.proto:365` and `:496`). Both sides regenerated `runtime_package_pb2.py`, whose serialized descriptor string and `_serialized_start`/`_serialized_end` offset table (`ignition/preparation/runtime_package_pb2.py:36` onward) change on any schema edit. Two independent schema additions therefore always collide textually. Separately, `fc2c914c` landed after the PR's acknowledgments were computed, so they name a superseded checklist.

## Blast Radius

1. `ignition/preparation/runtime_package_pb2.py:36` — the only file `git merge-tree --write-tree --name-only origin/pr/muse-glimmer upstream/main` reports as conflicting. It is regenerated by the command at `helios/manifest.md:379`. A `/tmp/pb2-dryrun.PPoe` dry run of the merged proto loads with field 30, enum value 10 and oneof tag 10 present.
2. `helios/proto/runtime_package.proto:118` — upstream's field 30 in `RebuildManifest`. It auto-merges and sits in a different message from the PR's additions, so no number collides. It is not edited.
3. `docs/design-review.md:134` (principle 16) and `docs/design-review.md:162` (review step 3), with `AGENTS.md:43` (Required PR descriptions) on `upstream/main` — the four-part rule the three messages and the PR body must meet.
4. `.githooks/commit-msg:63` — rejects an acknowledgment whose digest differs from the staged checklist. That digest is `3ad6` after the rebase, so all three `design-review=4d3a` trailers go stale.
5. `ignition/recipes/declaration.py:240` on `upstream/main` — upstream's new `online_mxfp8_scope` and `online_mxfp8_scopes` `Recipe` fields default to `"all"`, so the PR's `muse_glimmer_nvfp4` declaration needs no edit.
6. `tests/cli/test_planning.py:1` and `docs/model-preparation.md:1` — touched by both sides and auto-merged. They are not edited, but `tests/cli/test_planning.py` is in the re-run set.
7. `ignition/components/dflash2.py:1`, `ignition/recipes/muse_spark/attention.py:1`, `ignition/recipes/gemma4/config.py:1`, `ignition/preparation/quantization.py:1`, `ignition/preparation/checkpoint.py:1`, `ignition/recipes/__init__.py:1`, `tests/cli/test_models.py:1`, `helios/src/frontend/images/mod.rs:1` — `git diff --stat 91cfb369 upstream/main` over these paths (and `tests/cli/test_commands.py`, `ignition/recipes/muse_spark`, `helios/src/frontend/images`, and upstream's prose skill `.agents/skills/repo-prose/SKILL.md`, absent from this checkout) is empty. Upstream left every shared module the PR widens untouched, and the prose skill is unchanged.
8. `openspec/specs/upstream-contribution-branch/spec.md:23` — the main spec still requires messages to state "problem, resulting behavior, compatibility impact, validation, and authorship", which conflicts with upstream's four-part order; the delta under `specs/upstream-contribution-branch/` corrects it.
9. `ignition/recipes/muse_glimmer_nvfp4/config.py:64` — #229's checkpoint check rejects any checkpoint without NVFP4 on every decoder projection and a BF16 head, which rules out `nvidia/Muse-Glimmer-30B-NVFP4`. #192's `ignition/recipes/muse_glimmer/config.py` (fork `YouNeedCryDear/flare-local-inference` at `d8a9a05b`, lines 50-53) requires a `W4A16_NVFP4` `lm_head`, which rules out #229's checkpoint. The alias at `ignition/recipes/muse_glimmer_nvfp4/recipe.py:59` stays.
