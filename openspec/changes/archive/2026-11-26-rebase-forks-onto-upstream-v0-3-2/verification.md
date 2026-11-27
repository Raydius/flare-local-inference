# Verification — rebase-forks-onto-upstream-v0-3-2

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-27
Tier: 3
Basis: full
Depth: full

The one finding, a lint violation in the record, was fixed in this audit; the code needs no change.

## Dimensions

- Implemented: every one of the 32 checked boxes is backed by an artifact that says what its evidence claims; each was opened and quoted below. The code the fit, record and release commits claim exists on core `wt/` 86029f80 and recipes `wt/` 15ba64a9, for example `helios/src/runtime_core/mod.rs:100` (the persisting L2 bound) and `ignition/preparation/rebuild_manifest.py:254` (the rope parameters in the reproduction arguments). Three cited outputs were never saved: the `tools/prove.py` summary (3.2, 3.3) and the `/metrics` count of 7.3. The commit map, the empty `mismatches.txt` and the served checks support the same results. `gate/stage5-core/cargo-helios.log` (5.1) was overwritten by a later verify run (it now reads 1711 passed); `cargo-helios-5-1.log` holds the 5.1-era 1660/0/75. Commits cited in 5.1 to 7.1 survive as ancestors of `backup/pre-rebase-20261126-wt` after the second move rewrote them. No QA goal: the change names no release-QA item.
- Reachable: every new path reaches an entry point. `apply_context_limits` (`helios/src/runtime_core/mod.rs:111`) is reached from created contexts (:486) and from `apply_context_settings` (:1632) for retained primary and persistent-worker contexts (:817, :831, :832). `speculation::copied` is called on the completion path (`helios/src/scheduler.rs:2755`). `reproduction_arguments` and the `rope_scaling.*` parameters run inside `from_plan`, which `ignition export` calls (`ignition/cli/serve.py:375`, :242). The guard steps run on tag pushes (`.github/workflows/release.yml:26`, `container.yml:26`). The fork recipes register in `recipes()` (recipes `hyperflare_recipes/recipes/__init__.py:40-41`), and `check_ignition_launcher.py` runs in the image build (vllm-env `Dockerfile:73`). No caller of a removed symbol survives (`STABLE_VERSION`, the fork recipes' `collective_tuning` and `collective_ladder`, the three-argument `apply_context_settings`).
- Tested: two boxes carry `Verify:` commands (5.1, 5.5); one `--run-verify` call reused both passes on this exact code. The other 30 boxes have no covering command. The task gates compared full suites against upstream baselines with no fork code (core `gate/second-core-run3`: 0 added ids, cargo 3104 passed; recipes `gate/final-recipes`: 1 added id, which passes with its missing core support module). The three exports served on GPU 0 from image 6a735b098533. Qwen answered "Paris", completed three turns and answered an image request at concurrency 8 and 1,048,576 tokens. Muse Glimmer did the same. Elephant-1 returned 158 of 158 finite decisions. The core gate of 9.5 ran on f1dc9dbb plus the uncommitted fixes later committed as b7fdb892, so equality of trees is not proven. The Helios verify reuse on the current tree and the exported Qwen core's recorded `rope_scaling.*` parameters (7.2 discovery) exercise that code.
- Optimal: advisory; nothing admissible. No AGENTS.md rule is broken in Flare, Helios, recipe or component code: the DSpark head refusal now reads declared data (`ignition/planning.py:491-495`), and Helios carries no model identity or environment read. The small items found are listed as low in the code review.
- Reused: every reuse claim holds by import. `from ignition.registry import` (recipes `hyperflare_recipes/recipes/__init__.py:10`); `from ignition.preparation.bootstrap import BasePreparationConfig` (`elephant1/bootstrap.py:25`, `muse_glimmer_nvfp4/bootstrap.py:32`); `from ignition.preparation.bootstrap import common_preparation_kwargs` (`elephant1/recipe.py:23`, `muse_glimmer_nvfp4/recipe.py:33`); `from hyperflare_recipes.recipes.muse_spark.contract import prepare` (`muse_glimmer_nvfp4/contract.py:7`); `from hyperflare_recipes.recipes.muse_spark.frontend import FRONTEND, runtime_assets` (`muse_glimmer_nvfp4/frontend.py:10,12`); `from hyperflare_recipes.components.parallel_draft import allocate_copy_buffers` (`nemotron_h/bootstrap.py:333`). One bounded L2 loop serves all contexts (`helios/src/runtime_core/mod.rs:111`, `cuCtxSetLimit` only at :148). Muse Glimmer's attention module is its own by the task's "own modules" rule and predates this diff.
- Faithful: every Stated Requirements bullet is covered and every behavior change traces to the patch-evaluation bullet. The split and replay are tasks 2.1 to 4.3 and 9.3 to 9.4, the patch verdicts 5.1 to 5.5 and 9.5, the local `local/flare-ignition` image at tag `0.3.2-r2` tasks 7.1, 9.6 and 10.1, and the C8 console export task 7.2. The comparison is 8.1 (`comparison.md`), the base-move instructions 5.7 (`FORK.md:268`), the move to upstream `main` 9.1 to 9.6, and the named upstream commit 10.1 (`README.md:10`). The tags `v0.3.2-r2` and the ghcr publication: the tooling and the stamped serving image are delivered (6.1, 6.3, 9.6, 10.1). Pushing the tags, which publishes the image, follows `/rad-close`, as the proposal's What Changes states, because the tag names the commit `deploy` reaches at close. The changes to existing behavior trace to "the patchwork will have to be evaluated … adapted": the declared context in shared preparation, the declared DSpark head formats, the rope check at planning, `block_end_marker` omitted when empty, and upstream's checkpoint size check taken (`f533065e`, `b7fdb892`, 9.3 discovery). No task touched production: the served checks call `127.0.0.1:8100` on the operator's console instance, and nothing is pushed beyond the FlashInfer pin branch the operator approved.
- Reviewed: change code review over the change's own commits (core 2c587539..86029f80, recipes c3b36f5b..15ba64a9, vllm-env 4f1620f~1..1c8f0c7). There are no high or medium findings and 11 lows. Peer coverage: complete (8 calls, 65 file parts, all returned).
- Environment: doctor exit 1: omp trailer and tool-call hooks absent in this checkout and 16 other worktrees (repair `adopt.sh --harness omp --ensure-omp-hook`); engine-stamp leg cannot check the dlx-resolved engine version (stamp 2.2.0) offline. changed_files 0 (the change's commits live on the `wt/` worktrees of both forks, not on this checkout). suite runs: 0 in 240 min, 0 deliberate re-runs, budget 3. verify runs: 0 ran, 2 reused, 0 covered. Peer coverage complete. Tier-check warns that three evidence lines (7.2, 7.3, 7.4) parse to no path token. web search skipped — unavailable: no UX convention question in a change with no UI surface.

## Evidence audited

All 32 boxes audited; none skipped. The change's code and artifacts live outside this checkout (the `wt/` worktrees of both forks, the image recipe, and the scratch directory `~/models/.cache/rebase-v0.3.2/`), which the linter cannot resolve. Each line therefore cites the box's own `evidence:` line in tasks.md and names the artifact the audit opened, its line, and the text read there.

- [1.1 Create `wt/rebase-forks-onto-upstream-v0-3-2` from `deploy` in bot] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:19 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference/.git/packed-refs line 242 reads "09169e7ac3a5edfd41b5bbaf731a8aeb39dabb14 refs/tags/backup/pre-rebase-20261125-deploy"; Tested: no covering command)
- [1.2 Record upstream baselines with no fork code in worktrees under the] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:23 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/baseline/S/pytest-summary.txt line 1 reads "94 failed, 6053 passed, 3579 skipped"; Tested: no covering command)
- [2.1 Replay the fork onto `S` in core with `scripts/fork/rebase.sh da03] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:30 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/replay/stage1-conflicts.log line 30 reads "c32bb51b | Accept the locked Transformers release in Qwen3.8 preparation | DROPPED"; Tested: no covering command)
- [2.2 Gate the stage-1 tip: run the same commands as the `S` baseline an] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:39 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/stage1/pytest-summary.txt line 1 reads "87 failed, 6421 passed, 3619 skipped"; Tested: no covering command)
- [3.1 Divide, core side: replay `S..stage-1 tip` onto `D`; in the extrac] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:45 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/divide/core-conflicts.log line 1 reads "21a57ea7 (stage-1 590714c6) | Complete Elephant-1 preparation through core export"; Tested: no covering command)
- [3.2 Divide, recipes side: fetch the stage-1 tip into the recipes fork ] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:53 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/divide/classify.tsv line 776 reads "fork-added, imports hyperflare_recipes"; Tested: no covering command)
- [3.3 Prove conservation and write the commit map to `commit-map.tsv` in] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:58 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/commit-map.tsv line 153 reads "Specify the base-move procedure in FORK.md"; Tested: no covering command)
- [4.1 Replay core from `D` onto `c0c4f798` with `scripts/fork/rebase.sh ] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:63 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/replay/stage3-core-conflicts.log line 1 reads "b240b1bd | flare run --generation-defaults: server-side request defaults"; Tested: no covering command)
- [4.2 Replay recipes from `S'` onto `R`; in `pyproject.toml` name `ignit] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:70 — "evidence:" (artifact opened outside this checkout: ~/dev/hyperflare-recipes.worktrees/rebase-forks-onto-upstream-v0-3-2/docs/model-preparation.md line 1966 reads "## Muse Glimmer 30B"; Tested: no covering command)
- [4.3 Gate both results against the `v0.3.2` and `R` baselines, then swe] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:79 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/stage3-core/pytest-summary.txt line 1 reads "12 failed, 2406 passed, 460 skipped"; Tested: no covering command)
- [5.1 Verdicts for the serving patches (`patch/01`, `patch/03`, the pers] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:87 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/helios/src/runtime_core/mod.rs line 100 reads "fn persisting_l2_limit(recorded: usize, device_maximum: i32) -> usize {"; Tested: reused)
- [5.2 Verdicts for the preparation flags (`patch/06`, `patch/07`, `--rec] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:95 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/ignition/preparation/rebuild_manifest.py line 254 reads "args.extend(['--rope-parameters', json.dumps(parameters)])"; Tested: no covering command)
- [5.3 Verdicts for the recipe-side patches, in the recipes fork (Muse Gl] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:103 — "evidence:" (artifact opened outside this checkout: ~/dev/hyperflare-recipes.worktrees/rebase-forks-onto-upstream-v0-3-2/hyperflare_recipes/recipes/__init__.py line 40 reads "from .muse_glimmer_nvfp4.recipe import RECIPE as muse_glimmer_nvfp4"; Tested: no covering command)
- [5.4 Verdict for the FlashInfer pin and the Gated DeltaNet head-ratio p] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:111 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/verdicts/5.4/patch-dry-run.log line 9 reads "dry-run exit=0 on 0aa016814e2ad70f4208389030cdb56abafc40e1"; Tested: no covering command)
- [5.5 Verdicts for `patch/00-fork-tooling` and the qualification tools: ] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:118 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/scripts/fork/sync.sh line 34 reads "Kept the fork's tag(s) above"; Tested: reused)
- [5.6 Write the records: `README.md` Active and Retired patches judged a] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:126 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/README.md line 80 reads "**Image input for the dense Qwen recipe"; Tested: no covering command)
- [5.7 Write the base-move procedure into `FORK.md` as the section "Movin] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:132 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/FORK.md line 268 reads "## Moving to a newer upstream"; Tested: no covering command)
- [6.1 Release tooling per the `release-versioning` delta: tag filters an] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:138 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/scripts/set_release_version.py line 18 reads "RELEASE_VERSION = re.compile("; Tested: no covering command)
- [6.2 Image recipe: `build.sh` takes an optional third argument `<versio] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:144 — "evidence:" (artifact opened outside this checkout: ~/LocalApps/vllm-env/build/flare-ignition/build.sh line 39 reads "org.opencontainers.image.version="; Tested: no covering command)
- [6.3 Build the serving image locally from the core `wt/` tip stamped `0] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:149 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/stage5-core/serving-image-build.log line 127 reads "Compiling flare-cli v0.3.2-r2 (/src/flare-cli)"; Tested: no covering command)
- [7.1 Freeze the candidates: set the recipes lock to the final core `wt/] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:154 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/stage7-core/ignition-image-build.log line 727 reads "built local/flare-ignition"; Tested: no covering command)
- [9.1 Fix the targets of the second move and record baselines: fetch `up] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:162 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/baseline/T/pytest-summary.txt line 1 reads "9 failed, 2385 passed, 457 skipped"; Tested: no covering command)
- [9.2 Re-cut the FlashInfer pin on `F`: confirm `c3ab1d04` is an ancesto] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:167 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/verdicts/9.2/pin-675984cd-gpu1.log line 6 reads "3 passed, 20 deselected"; Tested: no covering command)
- [9.3 Replay core from `c0c4f798` onto `T` with a copy of `scripts/fork/] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:172 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/pyproject.toml line 47 reads "rev = "675984cda42c2e8dbd62d9117cb131c2e232ea59""; Tested: no covering command)
- [9.4 Replay recipes from `R` onto `R2`; drop `cb9b93e5` and the earlier] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:179 — "evidence:" (artifact opened outside this checkout: ~/dev/hyperflare-recipes.worktrees/rebase-forks-onto-upstream-v0-3-2/pyproject.toml line 32 reads "rev = "675984cda42c2e8dbd62d9117cb131c2e232ea59""; Tested: no covering command)
- [9.5 Gate both results against the 9.1 baselines and refresh the record] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:187 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/ignition/preparation/rebuild_manifest.py line 415 reads "add('rope_scaling.context', SCOPE_DEPLOYMENT, int(plan.checkpoint.context))"; Tested: no covering command)
- [9.6 Rebuild the images from the new tips with host memory watched: loc] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:198 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/final-core/ignition-image-build.log line 735 reads "built local/flare-ignition"; Tested: no covering command)
- [7.2 Export and serve the operator's Qwen profile at eight sequences: r] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:204 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/qwen-r2-final/served-checks.json line 86 reads "Paris"; Tested: no covering command)
- [7.3 On the operator's go, export Muse Glimmer (`muse-glimmer-30b-nvfp4] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:213 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/glimmer-r2-final/served-checks.json line 77 reads "Paris"; Tested: no covering command)
- [7.4 On the operator's go, export Elephant-1 (`elephant-1-9b`, concurre] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:219 — "evidence:" (artifact opened outside this checkout: ~/models/.cache/rebase-v0.3.2/gate/elephant-r2-final/decision-probes.json line 67 reads "158,"; Tested: no covering command)
- [8.1 Write `comparison.md` in this change directory from this change's ] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/comparison.md:65 — "## Defects of the 2026-11-24 move" (Tested: no covering command)
- [10.1 Record the exports and name the base in every release-facing text] — openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:230 — "evidence:" (artifact opened outside this checkout: ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2/README.md line 10 reads "Revision `v0.3.2-r2` is built on upstream `c915b5f0`"; Tested: no covering command)

## Findings

1. Lint exit 2 at verify time, fixed in this audit (was blocking, now cleared). The evidence of task 10.1 quoted the build output `local/flare-ignition:86029f80-15ba64a9 and local/flare-ignition:0.3.2-r2`. The linter read the image tags as file citations `local/flare-ignition:86029` and `local/flare-ignition:0` that never existed, which is a structural violation. The line now names the image and its two tags separately (`openspec/changes/rebase-forks-onto-upstream-v0-3-2/tasks.md:230`), as the 9.5 discovery did for tasks 6.3 and 7.1. `radflow-lint --tier-check` now exits 1 with warnings only.
   Dimension: Implemented

## Code review

Review standard: 001ce8ddb3cf5aa072e28e2cc5bbab28ff39cdc0
Peer part: path=core:.github/workflows/container.yml part=1/1 blob=cc0c55f68ff7 status=returned reason=quorum achieved (call p00)
Peer part: path=core:.github/workflows/release.yml part=1/1 blob=b2df3c2e38c5 status=returned reason=quorum achieved (call p00)
Peer part: path=core:FORK.md part=1/1 blob=f69a580cf96a status=returned reason=quorum achieved (call p00)
Peer part: path=core:README.md part=1/1 blob=c3d718db561f status=returned reason=quorum achieved (call p02)
Peer part: path=core:docs/TODO.md part=1/1 blob=d8da28ff46ac status=returned reason=quorum achieved (call p03)
Peer part: path=core:docs/container.md part=1/1 blob=54d376d99f89 status=returned reason=quorum achieved (call p03)
Peer part: path=core:docs/model-preparation.md part=1/1 blob=53c87bfb125c status=returned reason=quorum achieved (call p03)
Peer part: path=core:docs/release.md part=1/1 blob=70c89dfb1fe7 status=returned reason=quorum achieved (call p03)
Peer part: path=core:helios/src/runtime_core/mod.rs part=1/1 blob=deaa77ecae6e status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/src/scheduler.rs part=1/1 blob=f71fec6c13c1 status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/src/scheduler/policy_tests.rs part=1/1 blob=68effd0e3e9b status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/src/scheduler/speculation.rs part=1/1 blob=b95d5dd17a23 status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/fixtures/templates/elephant1/chat_template.patch.json part=1/1 blob=1a92f8609963 status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/fixtures/templates/muse_glimmer_nvfp4/chat_template.patch.json part=1/1 blob=9cd2ffde611f status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/support/contracts/deepseek4.json part=1/1 blob=af8fc8484b2e status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/support/contracts/deepseek41.json part=1/1 blob=c17e3e888cfc status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/support/contracts/glm53.json part=1/1 blob=45fac72ec4d3 status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/support/contracts/mimo_v2.json part=1/1 blob=8b47536bbd8c status=returned reason=quorum achieved (call p04)
Peer part: path=core:helios/tests/support/contracts/qwen38.json part=1/1 blob=85e8e08545d0 status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/cli/bench.py part=1/1 blob=e4f372aee2eb status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/cli/preparation.py part=1/1 blob=5bc7dcda3c3e status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/cli/serve.py part=1/1 blob=43241bfc6d28 status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/declaration.py part=1/1 blob=8ce4674a78ee status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/planning.py part=1/1 blob=77c3fa8ae668 status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/preparation/bootstrap.py part=1/1 blob=7581f2e78214 status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/preparation/contract.py part=1/1 blob=d5dae483f6b5 status=returned reason=quorum achieved (call p04)
Peer part: path=core:ignition/preparation/rebuild_manifest.py part=1/1 blob=1d1521f1bc1a status=returned reason=quorum achieved (call p04)
Peer part: path=core:pyproject.toml part=1/1 blob=0b1f39b68b96 status=returned reason=quorum achieved (call p04)
Peer part: path=core:scripts/fork/sync.sh part=1/1 blob=7684558dcb2c status=returned reason=quorum achieved (call p04)
Peer part: path=core:scripts/set_release_version.py part=1/1 blob=c3ae544778b2 status=returned reason=quorum achieved (call p04)
Peer part: path=core:tests/cli/test_reproduction_arguments.py part=1/1 blob=64cdec2fc530 status=returned reason=quorum achieved (call p05)
Peer part: path=core:tests/preparation/test_compile_pool.py part=1/1 blob=21e375f9c77c status=returned reason=quorum achieved (call p05)
Peer part: path=core:tests/test_fork_sync.py part=1/1 blob=38038b20e7b6 status=returned reason=quorum achieved (call p05)
Peer part: path=core:tests/test_set_release_version.py part=1/1 blob=292e6ee83fc6 status=returned reason=quorum achieved (call p05)
Peer part: path=core:openspec/specs/muse-glimmer-serving/spec.md part=1/1 blob=df5f78cdcf9a status=returned reason=quorum achieved (call p05)
Peer part: path=core:openspec/specs/upstream-contribution-branch/spec.md part=1/1 blob=35bdf7b882df status=returned reason=quorum achieved (call p05)
Peer part: path=recipes:README.md part=1/1 blob=f6164acd5842 status=returned reason=quorum achieved (call p05)
Peer part: path=recipes:docs/TODO.txt part=1/1 blob=f13f7acff4c4 status=returned reason=quorum achieved (call p06)
Peer part: path=recipes:docs/model-preparation.md part=1/1 blob=adcf7c1495e0 status=returned reason=quorum achieved (call p06)
Peer part: path=recipes:hyperflare_recipes/components/dflash2.py part=1/1 blob=598be1ad1184 status=returned reason=quorum achieved (call p06)
Peer part: path=recipes:hyperflare_recipes/components/parallel_draft.py part=1/1 blob=c715238cca0f status=returned reason=quorum achieved (call p06)
Peer part: path=recipes:hyperflare_recipes/recipes/elephant1/bootstrap.py part=1/1 blob=e4639849f356 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/elephant1/recipe.py part=1/1 blob=3d67c4d6c239 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/elephant1/weights.py part=1/1 blob=23023ffbdeac status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/muse_glimmer_nvfp4/bootstrap.py part=1/1 blob=25629ee1cc86 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/muse_glimmer_nvfp4/recipe.py part=1/1 blob=15b4a3127e93 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/qwen38_dense/contract.py part=1/1 blob=3db4158b48ee status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/qwen38_dense/decoder.py part=1/1 blob=edc170262b6b status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/recipes/qwen38_dense/recipe.py part=1/1 blob=581c888762ef status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:hyperflare_recipes/reference/qwen38_dense_dspark.py part=1/1 blob=c12986124a24 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:pyproject.toml part=1/1 blob=64723a695ac5 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/cli/test_draft_checkpoints.py part=1/1 blob=b2ad0b4b3fbe status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/cli/test_models.py part=1/1 blob=eeff0954c694 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/cli/test_speculative_method.py part=1/1 blob=1936e0f7009a status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/components/test_gated_norm_compile_pool.py part=1/1 blob=10d0dd9b088d status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/muse_glimmer_nvfp4/test_preparation.py part=1/1 blob=6e6275492f01 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/test_declarations.py part=1/1 blob=2f841f7adbc7 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/test_elephant1_attention.py part=1/1 blob=f0f1a4529eca status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/test_qwen38_binding_finalization.py part=1/1 blob=3f05dc83d518 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/test_qwen38_dense_text_only.py part=1/1 blob=10e016b0f693 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:tests/recipes/test_qwen38_dense_vision.py part=1/1 blob=296d6fc3bb38 status=returned reason=quorum achieved (call p07)
Peer part: path=recipes:uv.lock part=1/1 blob=1d817109734d status=returned reason=quorum achieved (call p07)
Peer part: path=vllm:build/flare-ignition/Dockerfile part=1/1 blob=cc29ab41e080 status=returned reason=quorum achieved (call p08)
Peer part: path=vllm:build/flare-ignition/build.sh part=1/1 blob=567b21ce71d7 status=returned reason=quorum achieved (call p08)
Peer part: path=vllm:build/flare-ignition/check_ignition_launcher.py part=1/1 blob=237ecf3fbf15 status=returned reason=quorum achieved (call p08)
Peer coverage: complete

The diff carried no secret candidate. Every call ran with no `tier` argument, reached tier 2 with five sources (anthropic-claude-haiku-5-5, local-ai-qllm, google-gemini-rpa-models-gemini-flash-lite-latest, deepseek-deepseek-flash, anthropic-claude-sonnet-5-5), all `ok`. No response carried an arbiter field. Caller Agreement is low because the caller draft sent was a placeholder.

- p00 (workflows, FORK.md): Quorum achieved (3/3); Certainty 0.7; Caller Agreement 0.1; Arbiter not reported.
- p02 (core README.md): Quorum achieved (3/3); Certainty 0.7; Caller Agreement 0; Arbiter not reported.
- p03 (core docs): Quorum achieved (5/3); Certainty 0.7833; Caller Agreement 0.1; Arbiter not reported.
- p04 (Helios, Ignition, scripts): Quorum achieved (4/3); Certainty 0.725; Caller Agreement 0.1; Arbiter not reported.
- p05 (core tests, planning specs, recipes README): Quorum achieved (5/3); Certainty 0.75; Caller Agreement 0.1; Arbiter not reported.
- p06 (recipes docs, components): Quorum achieved (3/3); Certainty 0.7; Caller Agreement 0.05; Arbiter not reported.
- p07 (recipes code, tests, lock): Quorum achieved (3/3); Certainty 0.85; Caller Agreement 0.05; Arbiter not reported.
- p08 (vllm-env image recipe): Quorum achieved (4/3); Certainty 0.85; Caller Agreement 0.1; Arbiter not reported.

Low findings (no action asked):

1. `scripts/fork/sync.sh:33` — a failed tag fetch whose output holds any "would clobber" line counts as success, so another error in the same output is dropped. CONFIRMED; it is reachable only when a clobber and a second error coincide, and later steps fail on a broken fetch.
2. vllm-env `build/flare-ignition/build.sh:28` — `locked_url` is not checked for emptiness. An empty `CORE_URL` would make `insteadOf ""` match every git URL (`Dockerfile:43`). This is unreachable while the lock check at :30 passes, because both values are read from the same lock block.
3. vllm-env `build/flare-ignition/build.sh:16` — the version argument is not validated before use as a tag and label; Docker refuses an invalid tag.
4. Recipes `hyperflare_recipes/recipes/elephant1/bootstrap.py:37` and `muse_glimmer_nvfp4/bootstrap.py:42` — `TuningReplica` is imported and no longer used, because the field moved to `BasePreparationConfig`. Not changed: the release labels name 15ba64a9, and AGENTS.md rejects lint-only churn.
5. `helios/src/server/metrics.rs:455-456` — the copy-depth predicate stays inline beside the new `speculation::copied` (`helios/src/scheduler/speculation.rs:169`), which is `pub(super)`.
6. `ignition/preparation/rebuild_manifest.py:307` — the recorded `--draft-model` keeps the folder name only, so recorded arguments need the folder supplied again. This is by design (5.2 discovery), and `docs/TODO.md` lists the replay as unrun.
7. `tests/test_set_release_version.py` — the lockfile-stamping case skips without `cargo`; task 6.1 ran `cargo metadata` on the stamped manifests separately.
8. Core `README.md:10` — it states the published image for `v0.3.2-r2` before either workflow has run from a revision tag. `docs/TODO.md` lists that run as unchecked.
9. Recipes `docs/model-preparation.md` (history drafting) — it lists MiMo V2 among the `--history-drafts` recipes, while recipes `docs/TODO.txt` records upstream's `buffers=` call defect in `mimo_v2/bootstrap.py`. The operator chose to record and not fix it (9.5 discovery).
10. `.github/workflows/release.yml:6` — a `workflow_dispatch` run skips the tag-name guard. This matches the default-branch check above it, and a manual run resolves no release version.
11. `hyperflare_recipes/components/dflash2.py:118-119` and `parallel_draft.py:88` — class defaults serve only tests that build instances without the constructor. Production `__init__` sets them (`dflash2.py:135`, `parallel_draft.py:197`).

NOT A DEFECT items checked:

- Muse Glimmer loses `mtp_only` (p07): `common_preparation_kwargs` passes it (core `ignition/preparation/bootstrap.py:47`).
- DSpark head and body refusal removed (p04): the dense DSpark method declares no head formats (recipes `qwen38_dense/recipe.py:124`), so `ignition/planning.py:491-495` refuses them. Upstream's body check stays (`ignition/planning.py:501-502`).
- `block_end_marker` omitted from five fixtures breaks a Rust reader (p04): the field is `serde(default)` (`helios/src/model/contract.rs:341-342`).
- Revision `-r0` and leading zeros admitted by the trigger glob (p00, p03): the guard step refuses them (`.github/workflows/release.yml:33`, `scripts/set_release_version.py:18`).
- Deleted compile-pool test (p05): moved to recipes `tests/components/test_gated_norm_compile_pool.py` (commit a6391591).
- Muse Glimmer profile number 11 contradicts the schema (p02): upstream's enums end at 10, so 11 is free (9.3 evidence, `c915b5f0:helios/proto/runtime_core.proto`).

## Advisory rules judged

- Stage decay: no. Tasks 9.1 to 10.1 were added during apply on operator decisions dated 2026-11-26, before any `verification.md` existed, and tasks.md has no `## Follow-ups` heading.
- Declined discoveries: n/a — no `declined:` discovery line.
- Split-offs: no. `debug-v0-3-2-device-error-1` quotes the operator's "Begin the debug" and was proposed at the operator's invocation. `audit-v0-3-2-rebase` predates this change and is retired at archive as the proposal states.

## Live UI pass

n/a — no UI surface

