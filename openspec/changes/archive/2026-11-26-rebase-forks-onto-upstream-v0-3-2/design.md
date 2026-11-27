# Design — rebase-forks-onto-upstream-v0-3-2

## Approach

### Names used below

- `B` = `ba42481a`, the upstream commit `deploy` sits on.
- `S` = `da037284`, upstream's core commit that renames `ignition/recipes` and `ignition/components` into `hyperflare_recipes/` while both trees are still in core.
- `D` = `17bb3c4a`, the next core commit, which deletes the extracted trees.
- `S'` = `4541fcb`, the recipes repository commit whose `hyperflare_recipes/` and `qualify/` trees equal core's at `S`.
- `R` = `07d7bb3`, the recipes base. Core's base is `c0c4f798` (`v0.3.2`).
- Scratch directory: `~/models/.cache/rebase-v0.3.2/` (worktrees, baselines, logs; never `/tmp`, which Docker cannot mount).

### Why replay onto S first

- The recipes repository has rewritten history with one root, so no fork commit has an ancestor there.
- At `S` upstream performs the rename inside core. A replay onto `S` gives every fork commit a three-way merge whose base is its own parent, and git's rename detection moves fork edits and fork-added files to the `hyperflare_recipes/` paths.
- Because the two subtrees are identical at `S` and `S'`, the replayed tree divides between the repositories with no content merge, and the division can be proven by blob equality.

### Sequence

1. Refs. Branch `wt/rebase-forks-onto-upstream-v0-3-2` from `deploy` in each fork. Tag `backup/pre-rebase-20261125-*`. The `patch/NN` branches and the `base/current` tag sit on `c0c4f798` from the earlier attempt; tag those positions `backup/attempt-20261124-*` and return them to the `backup/pre-rebase-20261124-*` positions.
2. Upstream baselines with no fork code, at `S`, at `c0c4f798`, and at `R` installed against core `c0c4f798`: the CPU Python suite and the per-crate Rust tests from `AGENTS.md`. The failing sets are the reference for every later gate.
3. Stage 1, core: `scripts/fork/rebase.sh` onto `S`, with `merge.renameLimit` raised and `rerere` on. Conflicts keep both sides on upstream's structure. Gate: no fork file remains at a pre-rename path, and the suites add no failure to the `S` baseline.
4. Division. Replay the stage-1 commits onto `D` in core, resolving a modify/delete in an extracted root by deletion and removing fork-added files under those roots in the commit that added them. Cherry-pick the same commits onto `S'` in the recipes fork, dropping core paths. A commit left empty on one side is dropped there.
5. Conservation proof. For every path changed in `S..stage-1 tip`: its blob at that tip equals its blob in exactly one of the two results. Core's result changes no extracted path against `D`; the recipes result changes only extracted paths against `S'`. A per-commit map records the core commit, the recipes commit, or "emptied".
6. Stage 3. Ordinary replays: core `D` to `c0c4f798` (13 upstream commits), recipes `S'` to `R` (10 upstream commits). The recipes `pyproject.toml` sources name the core fork by exact commit and the FlashInfer revision the pin verdict selects; `uv lock` follows.
7. Gates on both results against the baselines, plus two sweeps: no import of a moved module path (`ignition.recipes`, extracted `ignition.components.*`), and no file in core under an extracted root.
8. Patch verdicts (next section), then the release tooling, the images, the exports, the records, and the comparison.

### Second move: `v0.3.2` to upstream `main`

The operator moved the goal on 2026-11-26. The export of task 7.2 failed on a FlashInfer defect (`openspec/changes/debug-v0-3-2-device-error-1`, ledger H10) that upstream FlashInfer `c3ab1d04` removes, and upstream's branches past `v0.3.2` carry work on the operator's serving path. No export runs on the `v0.3.2` base; the forks move again before any export.

- Targets are fixed by commit when the replay starts and named `T` (core), `R2` (recipes) and `F` (the FlashInfer revision `R2`'s lock names). On 2026-11-26 they are `c915b5f0`, `7d70e230` and `0d48af53`. Upstream has no tag past `v0.3.2`, so this departs from `FORK.md`'s rule of moving to release tags, by the operator's choice; the target is still never the moving branch name.
- The range holds no directory move between repositories, so this is an ordinary replay per `FORK.md` "Moving to a newer upstream": backup tags, upstream baselines at `T` and `R2`, `scripts/fork/rebase.sh`, conflicts resolved to both sides on upstream's structure, gates against the baselines.
- FlashInfer: the pin is re-cut on `F` with the fork's one commit (`d73da5c4`'s stub rejection and `compile_compress` program cache), if its probes still fail on unmodified `F`. `F` contains `c3ab1d04`, so the cherry-pick `49efc285` and the two pin commits made for it (core `b14a8bce`, recipes `cb9b93e5`) are dropped in the replay.
- Verdicts are refreshed only for patches whose concern upstream touched in `c0c4f798..T` or `R..R2`: the rebuild record of `--rope-parameters`, `--draft-model` and `--recipe` against `PreparationManifest` (`27716185`); the fork's `qwen38` and `qwen38_dense` changes against `35059e30`, `eb2b087b` and `a2746f49`; copy-depth acceptance attribution against `1e78f080` and `e30d55db`; the persisting L2 bound against `c915b5f0`; and the FlashInfer pin. Every other verdict of tasks 5.1 to 5.6 stands when its tests pass on the new base.
- The release name stays `v0.3.2-r2` (`release-versioning` delta); `FORK.md`'s record of bases names `T` and `R2`.
- The images of tasks 6.3 and 7.1 are rebuilt from the new tips; tasks 7.2 to 7.4 export from the rebuilt preparation image. A passing task 7.2 is also the passing half of ledger check C1.
- The added tasks get their own peer review before they start, recorded under External Review.

### Patch verdicts

Each of the 23 active patches in `README.md` gets one row: need, integration, tests, verdict.

- Need: the patch's own test or a behaviour probe fails on unpatched upstream. A patch upstream now covers is retired with the upstream commit named.
- Integration: `git log B..c0c4f798` over every file the patch touches, and a search for each upstream enumeration, added in the range, of the thing the patch extends.
- Enumerations to search: preparation flags (rebuild record, `BasePreparationConfig`, help, tuning replicas), recipe registration, weight-container field declarations, declared costs, tuning identities, context-limit replay sites, acceptance estimates.
- Flag conservation: every fork flag in `flare run --help` and `ignition export --help` at `09169e7a` appears in the same help on the result or carries a retire verdict.
- A test asserts that each preparation flag `ignition export` accepts is either emitted by `reproduction_arguments` or named in an explicit list of flags that do not affect the prepared model.

### Release tooling

- `.github/workflows/container.yml` and `release.yml`: tag filter and a guard step for `vX.Y.Z` and `vX.Y.Z-rN`.
- `scripts/set_release_version.py`: accepts `X.Y.Z-rN`; `tests/test_set_release_version.py` covers accepted and refused forms. Cargo reads `0.3.2-r2` as a pre-release version; the Debian packaging path is checked with `dpkg --validate-version`.
- The serving image is verified before the change closes by a local `docker build` of `Dockerfile` stamped `0.3.2-r2`: `flare --version` and the fork flags in `flare run --help`.
- `build.sh` in the image recipe takes an optional version and adds the tag `local/flare-ignition:<version>` and an image version label. Its lock check stays. The image build fails unless exactly one installed distribution's `ignition` script is the entry point and importing it as `__mp_main__` does not reach the argument parser.
- The recipes fork has no workflow; its release is the tag alone, on the commit whose lock names the core commit tagged `v0.3.2-r2`.

### Exports

- The preparation image is built from the final commits of both `wt/` branches, the same commits the release tags.
- The console profile is a copy of the operator's profile "2026-11-24 Flare DFlash2 MXFP8 A16 - v0.3.2-r1" (`max_num_seqs` 8, 1,048,576-token context) with only the image changed; the field diff against the running instance is shown before the export starts.
- Served checks: a correct one-word answer, a multi-turn chat, an image request, and the `flare` section of `/v1/models` compared with the replaced core's.
- A failing export or served check stops all work. Diagnosis starts with an export from the unpatched upstream stack at the same base as the discriminator between upstream and fork, unless the operator directs otherwise, as on 2026-11-26.
- Muse Glimmer and Elephant-1 follow, one export and served check each.

### Comparison

- Written last as `comparison.md` in this change directory; the earlier attempt's branches and archived change are read only after this change's verdict table exists.
- Specification side: scope, gates, and assumptions of the two proposals, designs, and task lists.
- Work side: per repository, the tree diff between `attempt/v0.3.2-rebase-20261124` and this change's result; each verdict that differs; each known defect of the attempt attributed to a named difference or recorded as unattributed with what was tried.
- Durable lessons go to `openspec/lessons.md`.

### Explicit rejections

- Path-rewritten patches applied to the recipes repository — rejected because they merge against content that already diverged, with no common ancestor.
- Squashing the 153 commits into 23 patch commits — rejected because the per-commit history and acknowledgments are the fork's record, and the stage-1 replay needs no squash to be provable.
- Rewriting upstream's `reproduction_arguments` to enumerate flags by reflection — rejected as churn in upstream code; the enumeration test gives the same guarantee.
- Moving to `upstream/main` (24 commits past `v0.3.2`) — rejected: the request names the tag.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.85
Caller Agreement: 0.6
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.8
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.6
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.85

### Review Feedback

Consensus feedback recorded verbatim:
- "Do not replay the 153 commits as one monorepo rebase. Build a ledger with one row per commit, bucketed core, recipe, both, or docs, and reduce to the 23 active patches." — REJECTED in part: the replay onto `S` is a monorepo rebase only up to upstream's own rename commit, where true ancestry exists; the per-commit ledger is ACCEPTED as the commit map of step 5; reducing to 23 commits is rejected (see Explicit rejections). This line reconciles the 0.85 certainty and the 0.6 caller agreement: the disagreement is the replay method, and the conservation proof is what makes the commit-preserving method checkable.
- "Anchor the two histories by tree identity: hyperflare_recipes/ and qualify/ at core S match S' byte for byte, so record an explicit path map" — ACCEPTED: the division step uses that identity, and the extracted roots are listed in the task.
- "Split mixed commits by path or hunk. Build two sibling stacks: core on v0.3.2 and recipes on R." — ACCEPTED: steps 4 and 6.
- "Pin the recipes' core dependency to the exact core commit that becomes v0.3.2-r2, not floating main, and re-lock." — ACCEPTED: step 6 and the `release-versioning` pairing requirement.
- "the rebuild record must enumerate every preparation flag, including rope/YaRN, ideally generated by reflection" — ACCEPTED as the enumeration test; generation by reflection REJECTED (see Explicit rejections).
- "protobuf enum values must not be renumbered and wire compatibility must be checked" — ACCEPTED: the verdict task for the contract field and the image profile number checks the numbers against `c0c4f798:helios/proto/runtime_core.proto` and loads a core exported before the move.
- "each console script has exactly one owner; spawned compiler workers must not re-enter the CLI" — ACCEPTED: the image build checks under Release tooling.
- "Update the release tag filter and version stamping to accept v0.3.2-r2 (PEP 440 normalization must match image tags)." — ACCEPTED for the tag filter and stamping; the Python packages keep their own version, so no PEP 440 string is derived from the tag, which the release task states.
- "Run a real model export with the prep image, serve it at 8 concurrent sequences with finite logits and baseline-matching output, and root-cause the non-finite logits rather than leaving them unconfirmed." — ACCEPTED: the export gates the release; attribution of the earlier failure is a comparison task with a recorded outcome.
- "Tag both repos v0.3.2-r2 only on the tested commits, then smoke-test the tagged artifacts." — ACCEPTED: the image is built from the commits that get tagged, and the release steps pull the published image and check its version.
- "Blast radius extends to ignition.* imports, CI and path filters, CODEOWNERS, golden fixtures, old rebuild records, protobuf consumers, Cargo and Python packaging, and docs." — ACCEPTED: the import sweep, the workflow review in the release task, and the package-data check; neither fork has a CODEOWNERS file.
- "Speed comes from working on the 23 patches rather than 153 commits, caching builds, running cheap gates first, and parallelizing core, recipe, and image builds." — ACCEPTED in part: gates run cheapest first and the task graph runs the two repositories' work in parallel; working on 23 squashed patches is rejected as above.

### Peer Review Verdict — second move (2026-11-26)

Quorum: achieved (weight 3/3, tier 1)
Certainty: 0.83
Caller Agreement: 0.3
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.8
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.5
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.8

### Review Feedback — second move

Consensus feedback recorded verbatim:
- "Define the FlashInfer pin decision as explicit criteria. Replace the 'if two still fail' rule with a table of outcomes, confirm the upstream fix is an ancestor first, and run the probes on the final candidate pin, including any cherry-picked commit. Publish any cherry-picked FlashInfer commit to a stable fork ref, and keep core's and recipes' pins identical." — ACCEPTED: task 9.2 states each outcome, checks ancestry first, runs the probes on the pin itself, and requires the pin to be on the fork remote, pushed on the operator's go, because the image build fetches FlashInfer from there; tasks 9.3 and 9.4 name the same pin.
- "Do not lock recipes to the 'final core commit' in 9.4 before the core verdicts, re-judging and gate (9.5) are settled. Lock only after core is final, and re-lock and re-gate if core changes. Make sure the final core commit is pushed to a stable ref so the lock is fetchable." — ACCEPTED in part: task 9.4 locks the replayed core tip so the recipes gates can install it, and task 9.6 locks again at the final core commit and reruns the gates the last edits touched; the core commit is not pushed, because `build.sh` serves the locked core commit to the build from the local clone.
- "Record old base hashes and old pins in 9.1, and use explicit --onto <new> <old> --update-refs with range-diff and conflict-resolution logs. Apply the existing kept/adapted/retired verdicts during replay." — ACCEPTED: task 9.1 records them; tasks 9.3 and 9.4 keep a `git range-diff` against the pre-move stack beside the conflict log.
- "Baselines must match the final dependency set: record them at the chosen FlashInfer pin, include GPU/kernel tests (the defect was GPU-only), and capture a pre-move interface snapshot (CLI help, config and env keys, manifest fields)." — ACCEPTED in part: the baselines install upstream's own lock, as `FORK.md` specifies, so a difference names the fork; the pre-move help texts are the snapshot task 9.5 compares; the GPU check is FlashInfer's RMSNorm test on GPU 1 in tasks 9.2 and 9.6. The recipes' GPU suites need GPU 0 and a checkpoint and stay unrun before the export.
- "Derive the re-judge set from path and symbol overlap between the full upstream range and every fork patch, not a fixed list. Include the KV compression feature, and check that its default is off unless configured." — ACCEPTED: task 9.5 derives the set from the paths both the fork and the upstream range change, and adds the listed patches when the overlap misses them; it records the default of upstream's KV compression setting.
- "Cover the preparation-manifest migration: backward compatibility, old on-disk artifacts, a round-trip test, and the three moved flags." — ACCEPTED: task 9.5's verdict for the record of `--rope-parameters`, `--draft-model` and `--recipe` requires the fork's record test on `PreparationManifest` and upstream's legacy-core loading tests passing in the gate.
- "Add a rollback path (backup tags, retained prior images), target-freeze policy, pass criteria and reference outputs for served checks, a GPU-memory watch during export and serve, and a fix for the inconsistent '7.2-7.4' numbering." — ACCEPTED in part: task 9.1's backup tags and the retained `v0.3.2` images are the rollback; targets are frozen by commit in task 9.1; the served checks of task 7.2 already state their pass criteria. The export runs through the console, which reports GPU memory; task numbers keep their history.

## Bomb-squad findings

- History-draft copies reach upstream's host-wide acceptance estimate with no copy mask (`helios/src/scheduler.rs:2564`; upstream `c0c4f798:helios/src/scheduler.rs:2709`) — fixed: 5.1
- The persisting L2 bound misses upstream's second context-limit site for persistent workers (`helios/src/runtime_core/mod.rs:384`; upstream `c0c4f798:helios/src/runtime_core/mod.rs:1485`) — fixed: 5.1
- The DSpark Markov tensors drop out of the draft container's declared weight fields (`ignition/components/weights.py:47`; upstream recipes `07d7bb3:hyperflare_recipes/components/dflash2_weights.py:38`) — fixed: 5.3
- The rebuild record loses alias-declared rope scaling and the `--draft-model` folder (`ignition/cli/__init__.py:51`; upstream `c0c4f798:ignition/preparation/rebuild_manifest.py:238`) — fixed: 5.2
- The dense Qwen recipe counts the vision encoder in `device_bytes` and again through upstream's `vision_bytes` (`ignition/recipes/qwen38_dense/recipe.py:74`; upstream `c0c4f798:ignition/declaration.py:184`) — fixed: 5.3
- The dense Qwen tuning identity has no input for rope scaling (`ignition/recipes/qwen38_dense/bootstrap.py:286`; upstream recipes `07d7bb3:hyperflare_recipes/recipes/qwen38_dense/bootstrap.py:164`) — fixed: 5.2
- Two imports inside function bodies name moved modules and fail only under rope scaling (`ignition/cli/preparation.py:61`, `ignition/recipes/qwen38_dense/decoder.py:446`) — fixed: 5.2
- The recipes package installs core Ignition and FlashInfer from upstream sources (`pyproject.toml:25`; upstream recipes `07d7bb3:pyproject.toml:31`) — fixed: 5.4
- The recipes package data omits the fork recipes' template assets (`pyproject.toml:55`) — fixed: 5.3
- The fork recipes register in a function upstream emptied (`ignition/recipes/__init__.py:26`) — fixed: 5.3
- The fork-only recipes forward the page-limited `plan.context` and do not inherit `BasePreparationConfig` (`ignition/recipes/elephant1/recipe.py:28`, `ignition/recipes/muse_glimmer_nvfp4/recipe.py:39`) — fixed: 5.3
- Upstream's `common_preparation_kwargs` passes the native context, not a rope-scaled declared context (`ignition/planning.py:354`; upstream `c0c4f798:ignition/preparation/bootstrap.py:62`) — fixed: 5.2
