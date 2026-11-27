# Move the core and recipe forks onto upstream v0.3.2 and release them as v0.3.2-r2

Tier: 3
Type: feature

## Stated Requirements

- "Help me merge in the updates from main at tag v0.3.2."
- "it is expected that the repo has now split since the previous rebase, so that will migrate some of the work from this repo into the recipes repo"
- "after the rebase, the patchwork will have to be evaluated to determine if they are still necessary and/or if they need to be adapted to the new underlying code"
- "the expectation is that we will arrive at tag v0.3.2-r2 on the deploy branch as well as the recipe repo so we know they are aligned"
- "the goal is to end up with a sharable version of the post-patch fork flare run binary built with docker and uploaded to ghcr with tag v0.3.2-r2"
- "the flare-ignition binary only needs to be built for local use, and also needs the same tag"
- "the followup will be to export a new package with the updated flare-ignition from the console, optimizing for a C8 cap"
- "I will go through the full radflow process with this agent and start over to ensure the job is done correctly"
- "afterward, I want a comparison to be made, both in spec and in work, to determine exactly how and why the previous agent fell short"
- "update the deploy-branch-specific agent instructions to ensure that future rebasing always follows the optimal path" (added 2026-11-25, during apply)
- "skip the squash." (2026-11-25, on whether to squash the stack to one commit per patch after this move)
- "It sounds like a lot more stable upstream is available, let's move the goal post instead of doing extra work" (2026-11-26, after the device error of task 7.2 was traced to FlashInfer and upstream's commits past `v0.3.2` were listed)
- "Move now, no v0.3.2 export" (2026-11-26, chosen over one Qwen export from the fixed `v0.3.2` image first)
- "Untagged main, now" (2026-11-26, chosen over waiting for upstream's next tag, after "Does the latest upstream have a tag version number?" was answered no)
- "Be sure to name the upstream commit this is based on since it’s no longer v0.3.2 exactly" (2026-11-26, during the review of the `deploy` README's patch notes)

Assumed: "merge in" is a replay of the fork's 153 commits from `deploy` at `09169e7a` onto upstream `c0c4f798` (tag `v0.3.2`), with no merge commit, as every base move recorded in `FORK.md` did.
Assumed: the recipe fork's `deploy` is rebuilt on upstream recipes `07d7bb3`, the last upstream recipes commit dated before the `v0.3.2` tag; the upstream recipes repository has no tags and floats its core dependency on `branch = "main"`.
Assumed: after the move onto `v0.3.2`, both forks move once more, to the tips of upstream's `main` branches as they stand when that replay starts, each named by commit (on 2026-11-26: core `c915b5f0`, 27 commits past `v0.3.2`; recipes `7d70e230`, 10 commits past `07d7bb3`; FlashInfer `0d48af53`, the revision that recipes commit locks). No export and no release happens on the `v0.3.2` base.
Assumed: the release keeps the name `v0.3.2-r2`, because `v0.3.2` is still the newest upstream tag and is an ancestor of the new base; `FORK.md` records the exact base commits.
Assumed: the work runs on branch `wt/rebase-forks-onto-upstream-v0-3-2` in each fork and reaches `deploy` when the change closes.
Assumed: a patch verdict is one of kept, adapted, or retired as superseded, and "adapted" includes wiring a patch into an upstream mechanism that produced no textual conflict.
Assumed: the git tag is `v0.3.2-r2` in both forks; the image tags drop the leading `v` as `v0.3.2-r1` did (`ghcr.io/raydius/hyperflare-inference:0.3.2-r2`, `local/flare-ignition:0.3.2-r2`).
Assumed: the release tooling for `-rN` tags is re-derived in this change, because `deploy` at `09169e7a` accepts only `vX.Y.Z` tags; `v0.3.2-r1`, its image and its GitHub Release stay published and untouched.
Assumed: the C8 console export and a correct served answer from it come before the release steps, each GPU 0 run started on the operator's go; the release publishes only a stack that has served.
Assumed: Muse Glimmer and Elephant-1 each get one export that serves on the new stack, on the operator's go, before archive.
Assumed: the comparison against `attempt/v0.3.2-rebase-20261124` is this change's closing task and consults that branch only after this change's own verdicts are recorded.
Assumed: the deploy-branch-specific agent instructions are `FORK.md`, the only instruction file the fork adds to upstream's `AGENTS.md`; the procedure is written there as a present-state section.
Assumed: the untracked change `audit-v0-3-2-rebase` is superseded by this one and is retired when this change archives; its open debug ledger is an input to the comparison.

## Why

Upstream released `v0.3.2` and split model recipes out of the core repository into `hyperflare-recipes`. The fork's first move onto that base produced a preparation image whose first export fails every request, so the fork has no trusted `v0.3.2` stack to serve from or share.

## What Changes

- Replay the fork onto upstream's own split commit `da037284` first, where upstream renames `ignition/recipes` to `hyperflare_recipes/recipes` inside core, so git carries the fork's recipe work through the rename with a real common ancestor.
- Divide that replayed tree between the two forks and prove the division conserves every changed path: core continues to `c0c4f798`, recipes continue from `4541fcb` to `07d7bb3`.
- Give every active fork patch a recorded verdict against `v0.3.2`, traced through each upstream mechanism added in `ba42481a..c0c4f798` that touches the same concern.
- Re-derive `-rN` release tags, the serving image workflow, and the two-ref preparation image at a version tag.
- Export the operator's Qwen profile at eight sequences on the new stack and serve it; export and serve Muse Glimmer and Elephant-1.
- Record the base move in both forks' `README.md` and `FORK.md`.
- Compare this change with the previous attempt, in specification and in work.
- Release steps, run after the change closes and never as tasks: push tag `v0.3.2-r2` to both forks; the container workflow publishes `ghcr.io/raydius/hyperflare-inference:0.3.2-r2`; pull it and check `flare --version`. The local image `local/flare-ignition:0.3.2-r2` is built inside the change from the commits those tags name.

## Capabilities

### New Capabilities

- `fork-base-move`: how the two forks move to a new upstream base, what is proven at the split, and what must serve before a release.
- `release-versioning`: the tag forms that release the fork, the version every artifact reports, and the pairing of the two forks at one tag.
- `rope-scaled-export`: an export prepared with `--rope-parameters` records them in its rebuild record.

### Modified Capabilities

- `muse-glimmer-serving`: the scenario that pins upstream's Spark attention names the files at their post-split paths.

## Impact

- Core fork `Raydius/flare-local-inference` and recipe fork `Raydius/hyperflare-recipes`: `deploy` is rewritten in both at close.
- FlashInfer fork `Raydius/flashinfer`: the eviction commit and the Gated DeltaNet head-ratio patch are re-cut on `0aa01681` or retired.
- The image recipe `~/LocalApps/vllm-env/build/flare-ignition/` gains the version tag.
- GPU 0 time for three exports, each on the operator's go.

## Blast Radius

1. scripts/fork/rebase.sh:13 — the fork's replay mechanism (`git rebase --onto … --update-refs`), which assumes one repository.
2. grepped `git rev-list --count ba42481a..deploy` = 153 fork commits; by touched paths 46 core only, 22 recipe side only, 52 both, 33 `openspec/` only.
3. grepped `git log ba42481a..v0.3.2` = 64 upstream commits; `da037284` (382 renames into `hyperflare_recipes/`) and `17bb3c4a` (950 files deleted from core) are the split.
4. grepped `git rev-parse da037284:hyperflare_recipes` in core against `4541fcb:hyperflare_recipes` in the recipes clone — both `c34613db65a8`; `qualify` is identical too (`f01e26344bf6`).
5. grepped `git diff --diff-filter=A --name-only ba42481a deploy` under the extracted roots — 200 fork-added files, which a replay past `17bb3c4a` leaves in core without a conflict.
6. grepped `uv.lock` of the 14 newest upstream recipes commits — each locks `ignition` at `425ab2d4`, seven commits behind `v0.3.2`; no upstream recipes commit names `v0.3.2`.
7. README.md:14 — the active patch list, 23 entries, each judged against `ba42481a`.
8. flare-cli/src/serving_overrides.rs:1 — `patch/03` (`--max-context-tokens`, `--chat-template`), core side.
9. ignition/cli/preparation.py:53 — `patch/07` `rope_scaled`; grepped `git show v0.3.2:ignition/preparation/rebuild_manifest.py` — `reproduction_arguments` at line 231 and `from_plan` at 321 arrive inside the range and name no rope field.
10. helios/src/model/contract.rs:311 — `block_end_marker`, the Elephant-1 block mask's contract field; upstream `a2bf7860` and `1e4c0593` change marker and decision handling in the range.
11. helios/src/runtime_core/mod.rs:77 — the persisting L2 cache limit patch.
12. ignition/components/paged_tma.py:1 — the `kv_tile_ranges` patch; upstream `db4ba3d3` changes the same kernel call in the range.
13. grepped `git log ba42481a..v0.3.2 --oneline` for patches upstream may now cover — `173eb256` (locked Transformers for Qwen3.8), `0d45b17a` (image input for Qwen3.8 27B), `1bf68649` (inverted recipe registries, consolidated `PreparationConfig`), `1eb61e81` (cuBLAS warmup workspaces isolated from capture).
14. pyproject.toml:25 — FlashInfer pin `Raydius/flashinfer@cf1ac676`, on `598c8547`; upstream recipes `07d7bb3` locks `0aa01681`.
15. .github/workflows/container.yml:9 — the tag filter admits `vX.Y.Z` only; .github/workflows/release.yml:5 the same.
16. scripts/set_release_version.py:16 — `STABLE_VERSION` refuses `0.3.2-r2`.
17. Dockerfile:24 — the serving image copies `ignition/recipes/qwen38/assets/`, a path upstream removed; `git diff ba42481a v0.3.2 -- Dockerfile` deletes three lines.
18. grepped `build.sh` in the image recipe (outside this repository) — it tags `local/flare-ignition:<core8>-<recipes8>` and refuses a recipe lock that names another core commit; its `Dockerfile` writes the `ignition` launcher by hand because both packages declare that script.
19. openspec/debug/new-stack-package-nan-logits.md:6 — the previous stack's package failed 1314 of 1314 requests; the cause is not confirmed.
20. openspec/specs/muse-glimmer-serving/spec.md:56 — cites `ignition/recipes/muse_spark/attention.py`, a path that moves to the recipes repository.
21. grepped `git fetch upstream` on 2026-11-26 — `upstream/main` is 27 commits past `v0.3.2` (`c915b5f0`) and upstream recipes `main` is 10 commits past `07d7bb3` (`7d70e230`, locking FlashInfer `0d48af53`); the operator brought both into this change on 2026-11-26. In that range `27716185` replaces `RebuildManifest` with `PreparationManifest`, `35059e30`, `eb2b087b` and `a2746f49` rework `qwen38` and `qwen38_dense` attention and prefill, `1e78f080` and `e30d55db` change acceptance forecasts, and FlashInfer `c3ab1d04` fixes the grouped RMSNorm defect that failed task 7.2.

Lessons consulted: `openspec/lessons.md` entries `serve-from-the-operators-running-profile` and `a-replacement-core-keeps-the-replaced-profiles-capabilities`.
