<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Prepare the Muse Glimmer 30B NVFP4 recipe as an upstream PR branch on NVIDIA main

Tier: 3
Type: feature

## Stated Requirements

- "Prepare the entire Muse Glimmer implementation for upstream PR"
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."

Assumed: "upstream" is `NVIDIA/spark-local-inference` (remote `upstream`, GitHub redirect of `NVIDIA/flare-local-inference`), the target of PRs #211 and #212 (FORK.md:16: `pr/*` branches cut on `upstream/main`).
Assumed: "the entire implementation" is every Glimmer change archived on `deploy` — the recipe with images and the z-lab DFlash2 drafter (`muse-glimmer-text-recipe`, `rename-muse-glimmer-nvfp4`, `glimmer-zlab-dflash2-drafter`), the 1,048,576-token declared context (`glimmer-1m-context`), and the parity-bound fix (`glimmer-dflash2-hidden-row-parity`) — plus the shared-module widenings those commits made (muse_spark attention, DFlash2 components, the modelopt NVFP4 parser lift, Helios image packing, the package proto) and nothing else of the fork.
Assumed: the PR branch is `pr/muse-glimmer` on `upstream/main` (`972e28e4` at fetch time; re-fetched and recorded when the branch is cut), built in a fresh worktree; `deploy` gains only this change's openspec record. Fork-only material stays out: `openspec/**`, `.claude/handoff.md`, `docs/TODO.md`, `qualify/results/**`, the `qualify/long_context_passkey.py` probe, the fork's `--recipe`/`--draft-model` flags and their `qwen3.8-flash-dflash2*` aliases.
Assumed: the branch is squashed to two DCO-signed commits (recipe; 1M context), each re-acknowledged against upstream's `docs/design-review.md` (`4d3a`) and prose skill, as #212 was built (change `upstream-rope-pr`).
Assumed: opening the PR is the operator's leg (`gh pr create --head Raydius:pr/muse-glimmer --base main`); the branch, the pushed ref, and the audited PR body are this change's deliverables. GPU re-validation of the rebased branch is not run here — the body states which evidence came from the fork's packages and which checks were not run on the branch.
Assumed: the PR body leads with the served and passkey evidence and names the raw metric paths — PR #212's reviewer triaged it to design-review until first-party retrieval numbers were posted.

## Blast Radius

1. ignition/recipes/muse_glimmer_nvfp4/recipe.py:21 — `draft = plan.draft_model` reads a field only the fork's `--draft-model` flag sets (`ignition/planning.py:196`, `ignition/cli/preparation.py:100`; `git grep draft_model upstream/main -- ignition` is empty); the PR copy keeps only the pinned `z-lab/Muse-Glimmer-30B-DFlash2` download (ignition/recipes/muse_glimmer_nvfp4/draft_config.py:12).
2. tests/cli/test_draft_checkpoints.py:18,107 — the Glimmer parametrizations carry; `test_a_draft_folder_replaces_the_pinned_download[muse_glimmer_nvfp4]` and the `_dflash_plan(name=)` helper are the fork's flag and stay out (upstream's file at `upstream/main:tests/cli/test_draft_checkpoints.py:18` has neither).
3. tests/cli/test_models.py:25 — the alias expectation carries fork-only `qwen3.8-flash-yarn4`, `qwen3.8-flash-dflash2*` and the fork's `--recipe` tests; the PR hunk appends `muse-glimmer-30b-nvfp4` to upstream's list only (same two sites PR #212 edits — conflict named in the body).
4. ignition/recipes/__init__.py:37,40 — registration of `muse_glimmer_nvfp4`; the fork's `qwen38_dflash` import and `_supporting(name=)`/`resolve(recipe_name=)` widening (`2c8ba959`) stay out.
5. ignition/recipes/muse_spark/attention.py:66-68 — the projection factory writes q/k/v and gate into disjoint regions of `scratch.projected`; Python view/split only, but projection-packing territory under upstream `AGENTS.md:304` — authorship (Fable 5.1 design sign-off on every Glimmer commit, `git log --format=%(trailers:key=Design-Signed-Off-By)` 9834def8..d0093957) is disclosed in the body.
6. helios/proto/runtime_package.proto:360 — `MUSE_GLIMMER_IMAGE_PREPARATION = 10` and `MuseGlimmerImageSettings muse_glimmer = 10` (upstream's enum ends at `PATCH_MERGE_IMAGE_PREPARATION = 9`, oneof at `patch_merge = 9`; tag 10 free in both); an additive package-format change under `AGENTS.md` "Model package compatibility" — readers at `helios/src/runtime_package/protobuf.rs:161` and `ignition/preparation/frontend.py:117-130`; existing packages keep loading (no field removed, no default changed).
7. ignition/preparation/runtime_package_pb2.py:1 — the only file the 3-way dry run of the Glimmer diff (merge-base `585159c9`..`deploy`, 58 files / +5131 −282) fails to apply on `upstream/main`; it is generated and is regenerated from the proto with `/tmp/protoc-bin/protoc` on the branch.
8. ignition/preparation/quantization.py:18 and ignition/recipes/gemma4/config.py:10,524 — the modelopt NVFP4 parser lifted out of Gemma 4 and shared; Gemma's results are the carried behavior (`tests/recipes/test_gemma4.py`).
9. docs/model-preparation.md:1098 — the Glimmer section carries; upstream's own edits since the fork base (startup-race storage paragraph) are not touched.
10. CONTRIBUTING.md:9-13 — DCO `Signed-off-by` on every commit; none of the 19 Glimmer commits carries one (`git log --format=%(trailers:key=Signed-off-by) 9834def8^..d0093957` empty), so the squashed commits are created with `-s`.
11. upstream `docs/design-review.md:170` — the `Design-Signed-Off-By` trailer's digest must be `4d3a` (sha256 of upstream's staged document; the fork's commits carry `a38f`), and since upstream `972e28e4` (#214) the acknowledgment also covers `.agents/skills/repo-prose/SKILL.md`, absent from the fork.
12. grepped `git log --oneline 585159c9..deploy -- <shared files>` minus Glimmer/passkey commits — no other fork commit touches `components/dflash2*.py`, `preparation/{checkpoint,frontend,quantization}.py`, `muse_spark/{attention,contract,frontend}.py`, `gemma4/config.py`, `helios/src/frontend/images/**`, `helios/src/model/**`, `helios/src/runtime_package/**`, or the proto: the fork's versions are upstream + Glimmer.
13. grepped `git log --oneline 585159c9..upstream/main -- <same paths>` — 6 of upstream's 16 new commits touch them (`d68ed8f5` shared DFlash2 preparation, `87cc8c1e` format-4 packages, `fa0eadad`, `78b18b1c`, `b4196955`, `534537c8`); the dry run above applies cleanly over all of them except the generated `_pb2.py`.
14. grepped `from ignition` imports under `ignition/recipes/muse_glimmer_nvfp4/` and `ignition/reference/muse_glimmer.py` against `upstream/main` — every imported module exists upstream; the recipe depends on no fork-only module once item 1 is removed.
15. qualify/results/glimmer-passkey/summary.json:1, qualify/long_context_passkey.py:1, tests/reference/test_long_context_passkey.py:1 — fork-only (the `yarn4-mtp-passkey-evidence` probe; `qualify/results/` absent upstream); cited in the PR body as evidence, not carried.
16. grepped `git status --short` on `deploy` — clean; no tracked file in this checkout changes except `openspec/changes/upstream-muse-glimmer-pr/**`; the code lives on `pr/muse-glimmer` in its own worktree, as `pr/rope-scaling` lives in `/tmp/rope-pr`.
17. openspec/changes/archive/2026-11-05-upstream-rope-pr/tasks.md:5 — precedent: branch in a worktree, CPU suites on the branch and on pristine `/tmp/upstream-main2` (now re-pointed to `972e28e4`), `--force-with-lease` push to `origin` only, PR text audited before the operator opens it.

## Why

Muse Glimmer serves on one GPU with images, the z-lab DFlash2 drafter and a 1,048,576-token context, verified across seven archived changes; it lives only on the fork's `deploy` stack. Upstream has no Glimmer recipe, and the widenings it needed (NVFP4 projections in the Spark attention, the shared modelopt parser, Glimmer image packing) are general. FORK.md's rule — patches that prove worth it go upstream from `pr/*` branches — now applies.

## What Changes

- A branch `pr/muse-glimmer` on `upstream/main` with two DCO-signed, re-acknowledged commits: (1) the `muse_glimmer_nvfp4` recipe — preparation, decoder, vision frontend, DFlash2 drafter, package proto settings, Helios image preparation, docs and tests; (2) the 1,048,576-token declared context.
- The fork-only `plan.draft_model` branch, `--recipe`/`--draft-model` test cases, and the `qwen3.8-flash-dflash2*` alias expectations removed from the PR copy; `runtime_package_pb2.py` regenerated.
- CPU validation on the branch and on pristine `972e28e4`; the branch pushed to `origin`; the PR body (problem, behavior, package-format impact, validation with fork-package evidence and unrun checks, authorship) written and audited against upstream's prose skill.
- The PR opened by the operator from `Raydius:pr/muse-glimmer` against `main`.

## Capabilities

### New Capabilities

- `upstream-contribution-branch` — a fork patch set prepared as an upstream PR branch: cut on `upstream/main`, fork-only code excluded, DCO and design-review acknowledgments against upstream's documents, validation on the branch and on the pristine base, evidence disclosed by package.

### Modified Capabilities

- none — `muse-glimmer-serving` describes behavior the PR carries unchanged.
