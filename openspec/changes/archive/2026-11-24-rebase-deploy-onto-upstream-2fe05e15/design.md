# Design — rebase-deploy-onto-upstream-2fe05e15

## Approach

Scope and Assumed lines: proposal.md. The work runs in three phases: **resolve once** (one scratch replay holds core and recipes together, so every stop is resolved against both sides and tested as a whole), **split by path** (mechanical, from the resolved stack), and **wire and prove** (pins, image, gates, publish). Paths: core checkout `~/dev/flare-local-inference`, recipe clone `~/dev/hyperflare-recipes`, scratch and evidence `~/models/.cache/rebase-2fe05e15/`.

### Phase A — prepare

0. **Pre-flight.** `git fetch upstream` in core and in the recipe clone. If core `upstream/main` moved past `2fe05e15` or recipe `main` past `07d7bb3`, retarget, rerun `probe2.sh`, and record a Revisions entry. Commit `.claude/handoff.md` so `deploy` is clean.
1. **Backups.** `git tag backup/pre-rebase-20261124-<ref>` at `deploy`, `base/current`, `main`, `patch/00`, `01`, `03`, `06`, `07`; record each sha.
2. **Core mirror.** `scripts/fork/sync.sh` fast-forwards `main` to `2fe05e15` and pushes without force.
3. **Recipe fork.** `gh repo fork local-inference-lab/hyperflare-recipes --clone=false` creates `Raydius/hyperflare-recipes` (asked once at this task — it publishes a public repo). Clone it to `~/dev/hyperflare-recipes` with `origin` = the fork and `upstream` = `local-inference-lab/hyperflare-recipes` (push disabled, as in core); `main` mirrors upstream `main`.
4. **FlashInfer, before the stack moves.** In `~/dev/flashinfer`: scratch worktree at `0aa01681` (the recipe lock), copy the three eviction cases from `cf1ac676a` with no fix code, run them in a container from the current `flare-ignition` image. All pass → retire the pin. Any fail → branch `evict-planned-jits-0aa01681`, port only the failing hunks, rerun to green, push, pin that sha.

### Phase B — resolve once

5. **Scratch base `R`.** Detached `2fe05e15` + `git revert --no-commit 17bb3c4a`. In the same commit: drop `.github/workflows/track-flashinfer-main.yml`; fix the revert's `tests/preparation/test_packaging.py` to `2fe05e15`'s `packages` key and to assert `hyperflare_recipes.cli:main`, so the packaging test is not vacuous. `R`'s recipe subtrees equal recipe `4541fcb` by tree id (Blast Radius 21). `R` is never published.
6. **Replay.** In a scratch branch `split/unified`: `git rebase --onto R ba42481a <deploy copy> --update-refs` over copies of the five patch branches (`split/patch-NN`), so the real refs stay on their backups until Phase D. Continue with `git -c core.hooksPath=/dev/null rebase --continue`. Rules:
   - **Moved recipe files** and `UA` adds under `hyperflare_recipes/reference/`: port the fork's hunk onto upstream's moved file.
   - **`DU` stops** (e.g. `ignition/recipes/qwen38_dense/contract.py` at `f9ec666a`): put the fork's changes onto upstream's file at its new path — never re-create the old path. This keeps git's directory-rename detection on, so later fork additions follow into `hyperflare_recipes/recipes/`.
   - **Dense-Qwen image input** (`1f486f63`, `751eeefc`): drop both commits (`git rebase` `drop` via the todo list) — upstream #386 supersedes them; resolve the `AA` tests to upstream's files; later DSpark and rope-scaling hunks that touched the fork's vision code port onto #386's `vision_state.py`.
   - **Slim core** (`ignition/{registry,declaration,planning}.py`, `ignition/cli/*`, `ignition/preparation/*`): port onto upstream's structure; a recipe reference inside core moves to the recipe side.
   - **Rust and proto**: union; field numbers unchanged; regenerate `runtime_core_pb2.py` from the merged proto.
   - **`43207689`** (`pyproject.toml`, `uv.lock`): keep upstream's `hyperflare_recipes.cli:main` script and `include = ["ignition*", "hyperflare_recipes*"]`; the FlashInfer pin goes in `[tool.uv.sources]` as `{ git = …, rev = <sha> }` with a bare dependency line (no second source); regenerate `uv.lock` in the image.
   - **`docs/TODO.txt`**: upstream's entries fold into the fork's `docs/TODO.md`.
   - A hunk that needs new mechanism stops the task and becomes a discovery for the operator.
   Stop count is expected above probe B's 21, because the probe kept the fork's file at the `DU` stop.
7. **Move and rename.** One commit on `split/unified`: `git mv` whatever remains under `ignition/recipes/` (expected `elephant1/`, `muse_glimmer_nvfp4/`) into `hyperflare_recipes/recipes/`, register both in `recipes()`, add their `assets/*` to package-data. Then rewrite every old module path, code and string forms, across the tree except `openspec/` and history docs. Done when `git ls-tree -r HEAD -- ignition/recipes ignition/reference` is empty and `git grep -nE "ignition\.(recipes|reference)|ignition\.components\.(<moved modules>)" -- ':!openspec'` returns nothing. Also fix the two `python -m ignition` invocations (`docs/model-preparation.md`, the muse_glimmer reference docstring) to `ignition` / `python -m hyperflare_recipes.cli`.
8. **Unified gate.** On `split/unified` vs `R`, CPU only (step 12's file list), compared **by test id**; fork-only test files must pass outright. This proves the resolution before it is split.

### Phase C — split

9. **Path rule** (proposal Assumed): recipe side = `hyperflare_recipes/`, `tests/recipes/`, `tests/reference/`, `qualify/`, `tools/`, `scripts/` files present in recipe `4541fcb`, `uv.lock`, and fork-new tests importing `hyperflare_recipes`; core = the rest. Generated as `recipe.paths` (one pathspec per line) from `fork.class` plus the import check; reviewed before use.
10. **Recipe fork `deploy`.** `git format-patch --binary R..split/unified -- $(cat recipe.paths)` → `git am -3` onto recipe `4541fcb` in `~/dev/hyperflare-recipes` (identical tree, so it applies without conflict), then `git rebase main` onto `07d7bb3` (10 upstream commits, 41 files; resolve stops by the step 6 rules). Then one fork commit: `pyproject.toml` `[tool.uv.sources] ignition = { git = "https://github.com/Raydius/flare-local-inference.git", rev = "<core deploy sha>" }` (the sha is fetchable because step 11's result is first pushed to the staging branch `split/deploy` on the core fork, deleted after step 17), the FlashInfer outcome, `elephant1` and `muse_glimmer_nvfp4` package-data, `uv lock`; and a README "Fork notes" section (base, the core pin, patch list).
11. **Core.** On copies of the stack: `FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --prune-empty --index-filter 'git rm -r -q --cached --ignore-unmatch --pathspec-from-file=<recipe.paths>' -- split/patch-00 … split/unified ^R`, then `git rebase --onto 2fe05e15 R split/unified --update-refs` (drops `R`, carries the five patch copies). Name every commit that came out empty (recipe-only) in the record commit. Both sides carry a `Split-From: <split/unified sha>` trailer (`--msg-filter` here, `git interpret-trailers` over the patches in step 10), and `git range-diff` of each side against `R..split/unified` is read before publish. Core `pyproject.toml` ends at upstream's text — no `uv.lock`, no recipe package — and upstream's `tests/preparation/test_core_boundary.py` passes unedited.
12. **Record commit** on core, hook on, digest `eb2f`: README stack base `2fe05e15`, a 2026-11-24 History line (the split, the retired dense image input, FlashInfer outcome, `eb2f`), the "judged against" sentence re-judged with every "Still needed" row checked against upstream commit subjects as well as flag names, the recipe rows moved to a pointer at the recipe fork, and a FORK.md section for the two-repo layout and the base move.

### Phase D — wire, prove, publish

13. **Image.** vllm-env `build/flare-ignition`: `build.sh <core-ref> <recipe-ref>` (both default `deploy`) archives core to the context root and the recipe fork under `recipes/`, fails when the recipe `uv.lock` `ignition` sha differs from `<core-ref>`, and labels both revisions. `Dockerfile`: `uv sync --locked --extra prepare` runs in `recipes/` with `UV_PROJECT_ENVIRONMENT=/opt/flare/.venv`; `flare-cli` still builds from the core root. Console `_RECIPES_SCRIPT`: `try: from hyperflare_recipes.recipes import recipes` / `except ImportError: from ignition.recipes import recipes`.
14. **CPU gates**, under a 14 GiB watchdog on the real PID, compared by test id. Core vs pristine `2fe05e15`: `tests/cli`, `tests/preparation`, `tests/frontend`. Recipes vs pristine recipe `main`: `tests/recipes/test_qwen38_*`, `tests/recipes/test_qwen38_dense_*`, `tests/recipes/muse_glimmer_nvfp4`, `tests/recipes/test_elephant1_attention.py`, `test_elephant1_config.py`, `test_elephant1_weights.py`, `tests/recipes/test_declarations.py`, `tests/components`, `tests/reference`. Every named path is checked to exist first.
15. **Image and Rust gate.** Build the image from both `deploy`s; inside it: `uv lock --check` (recipes), `ignition --help`, the console probe lists `qwen38`, `qwen38_dense`, `muse_glimmer_nvfp4`, `elephant1`; `cargo test --locked --release` for `flare-cli`; `cargo check --locked` at the core `patch/01`, `03`, `06`, `07` tips; Helios `--features runtime-core,server` with memlock raised. No failure pristine does not also have.
16. **GPU 1 gate.** `--gpus device=1`, `FLARE_RUN_CUDA_TESTS=1`: `tests/components/test_packed_speculation_cuda.py`, the CUDA cases in `tests/recipes/muse_glimmer_nvfp4` and the Elephant-1 files, `tests/recipes/test_qwen38_mixed_cuda.py`, `tests/components/test_dflash2.py`, against pristine. Then `ignition` dry-run planning of Elephant-1 within its 12.5 GB envelope (#386 grew dense-Qwen rows), `flare inspect` an existing Elephant-1 core, and serve it for one masked decision request.
17. **Publish.** Core: `git branch -f` each real ref to its `split/` copy, then `git push --force-with-lease=<ref>:<backup sha> origin <ref>` for `patch/00`, `01`, `03`, `06`, `07`, `deploy`, then `git push -f origin base/current`. Order: core first, then the recipe fork, then vllm-env. Recipe fork: push `main` and `deploy` (new branches, no force); delete the core `split/deploy` staging branch. Then vllm-env commit, vault rebase paragraph, and `~/LocalApps/vllm-env/build/README.md` image row naming both shas.

## Revisions

- 2026-11-24, task 1.1 pre-flight: core `upstream/main` moved `2fe05e15` → `c0c4f798` (one commit, "Remove redundant RegistryDwords EnableResizableBar check in flare doctor": `flare-cli/src/doctor.rs`, `docs/cli-design.md`; no recipe path). The core base is `c0c4f798` wherever this design and tasks.md say `2fe05e15`; `R` = `c0c4f798` + revert of `17bb3c4a`. Recipe `main` is still `07d7bb3`. The fork edits `docs/cli-design.md`, so one docs stop may appear; probe B's other findings are unaffected.

## Decisions

- **Resolve once, then split by path.** Resolving in one tree keeps core and recipe halves of a "both" commit (49 of them) consistent and lets one CPU gate prove them together. Rejected: rebasing core onto bare `2fe05e15` and replaying recipe commits onto the recipe repo separately, which resolves the 49 shared commits twice, each half blind to the other.
- **`format-patch`/`am` for recipes, `filter-branch` for core.** The recipe side has no branch structure to keep and its target tree is byte-identical, so patches apply cleanly. Core must keep `patch/NN` refs, which `filter-branch` rewrites in place. Rejected: `git filter-repo` (not installed; would be a new dependency for one run).
- **Recipe fork pins core by exact `rev`, and the build checks it matches the core ref.** An unpublished core sha must be lockable before publish, which `branch = "deploy"` cannot do (it resolves to the old tip). Rejected: a path source to `../core`, which makes the lock unusable outside the image; a pin per core patch boundary, which no consumer builds.
- **The recipe fork has one `deploy` stack, no patch branches.** No consumer builds from a recipe patch branch today. Rejected: mirroring `patch/06`/`07` there.
- **Retire the fork's dense image input for #386.** Upstream ships the same feature; the fork's convention retires superseded patches (`dd62eb13`). Rejected: carrying both, which leaves two `vision_state.py` in one recipe.
- **Rebase, not merge; rerere records only reviewed resolutions; replayed commits keep historical trailers.** As on 2026-11-18 (FORK.md).

## Risks / Trade-offs

- [The path rule misfiles a file] → the unified gate passed before the split; after the split each side's gate and the core boundary test catch an import across the line; `recipe.paths` is reviewed before use.
- [Recipe rebase from `4541fcb` to `07d7bb3` conflicts with fork work] → same resolution rules; 41 upstream files, mostly DeepSeek.
- [The image build needs network access to two public GitHub repos] → it already fetches FlashInfer by git; both forks are public.
- [Every future core `deploy` move needs a recipe lock bump] → the build check fails loudly; FORK.md documents `uv lock --upgrade-package ignition` as part of each core move.
- [Core prefix checks (`rebuild_manifest.py:400`, `aot.py:137`, `forward_metadata.py:147`) match only `ignition.`/`b12x.` modules] → upstream ships the same behaviour with its recipes outside core; carried as upstream behaviour, re-checked by the GPU gate's planning run.

## Migration Plan

Nothing published moves before step 17. Rollback before publish: delete `split/*` and the scratch worktrees. After publish: force-with-lease each core ref back to its backup tag; the recipe fork can be archived on GitHub; `main` stays.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.85
Caller Agreement: 0.8
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.85
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.8
- deepseek-flash (deepseek-flash) — ok, agreement 0.3

An earlier call on the superseded in-tree design (quorum achieved, certainty 0.78, caller agreement 0.35) argued against resurrecting the deleted tree; the operator's "Split in this change" adopted that direction, and this verdict is on the split design.

### Review Feedback

Consensus feedback recorded verbatim:
- "splitting the 49 mixed commits by path while keeping author, date, and message plus a traceability trailer" — ACCEPTED: step 11 and step 10 add a `Split-From:` trailer on both sides.
- "preserving namespaced patch/NN refs in each repo with a mapping manifest" — REJECTED: no consumer builds from a recipe patch branch; core keeps its five refs and the `Split-From:` trailers are the mapping.
- "Pin the recipes uv git source to an exact migrated core SHA (updated per patch boundary)" — ACCEPTED in part: step 10 pins `rev = <core deploy sha>`, staged on `split/deploy` so it is fetchable before publish; REJECTED per-patch-boundary pins, which no consumer builds.
- "Adopt upstream's image-input feature and drop the fork's duplicate unless it has extra behavior" — ACCEPTED: step 6 drops `1f486f63`/`751eeefc`; extra behaviour, if any, surfaces as a port onto #386 at the later DSpark stops.
- "update module paths in imports, sys.modules keys, job specs, and packaging metadata, and smoke-test the console's ignition.recipes.recipes() import inside the image" — ACCEPTED: steps 7, 13 and 15 already cover each.
- "Validate with git range-diff against the original stack" — ACCEPTED: step 11 reads `range-diff` per side before publish.
- "buf breaking checks" — REJECTED: `buf` is not in the toolchain; wire compatibility is proved by the field-number audit (bomb-squad) and by loading and serving an existing core (step 16).
- "land core first, then recipes, then image/console, keeping archive refs until green" — ACCEPTED: step 17 orders the pushes; backup tags stay.
- "Speed-ups: script classification and filtering, reuse conflict resolutions (rerere), cache uv and cargo" — ACCEPTED in part: classification is scripted (`fork.class`, `recipe.paths`) and rerere records reviewed resolutions; REJECTED new uv/cargo caching, which the image build does not need for one run.
- Certainty 0.85 and caller agreement 0.8 sit below the 0.9 floors — reconciled: the divergent points are the rejected ones above (patch refs in recipes, per-boundary pins, buf), each rejected for a stated repository fact; every accepted point is now in the design.

## Bomb-squad findings

- Duplicate dense-Qwen image input (fork `1f486f63` vs upstream #386 `0d45b17a`) — fixed: step 6 drops the fork's commits; proposal Assumed line.
- `DU` contract stop rule kept the fork's stale file — fixed: step 6 `DU` rule ports onto upstream's file.
- 28-file move list is a probe artifact — fixed: step 7 moves what remains and checks emptiness.
- Stale `ignition.recipes`/moved-component imports in code, tests and qualify — fixed: step 7 tree-wide rewrite with an empty-grep done-check.
- Module paths inside strings (`sys.modules` keys, `CompileJob.create`, declaration tests) — fixed: step 7 grep covers string forms.
- `python -m ignition` loads no recipes — fixed: step 7 rewrites the two invocations.
- Core prefix checks skip `hyperflare_recipes.*` — declared: upstream behaviour, carried (Risks).
- Muse Glimmer assets missing from package-data — fixed: step 7 and step 10.
- Probe kept `ignition.__main__:main` at `43207689` — fixed: step 6 pyproject rule.
- Vacuous packaging test after revert — fixed: step 5.
- Boundary test weaker in the revert than claimed — fixed: core runs upstream's stricter `2fe05e15` version after the split (step 11); step 7's grep is the in-scratch guard.
- FlashInfer pin as a second source — fixed: step 6 puts it in `[tool.uv.sources]`.
- Proto field numbers — hunt completed — no collision found.
- Elephant-1 gate paths did not exist; #386 may grow its plan past 12.5 GB — fixed: steps 14 and 16.
- File-level baseline comparison masks fork-only failures — fixed: steps 8 and 14 compare by test id.
- `.gitattributes` eol rule misses moved `.jinja` assets — declined: upstream's own line, and the recipe repo owns those files after the split.
