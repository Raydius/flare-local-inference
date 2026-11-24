# Split the fork into core and recipes, and rebase both onto upstream

Tier: 3
Type: feature

## Stated Requirements

- "Help me update main from upstream and rebase deploy onto main"
- "Help me pull upstream for main and rebase deploy onto main"
- "Split in this change" — the operator's answer to how the fork adopts upstream's core/recipes split

Assumed: "update main from upstream" is the mirror move — `scripts/fork/sync.sh` fast-forwards `main` `ba42481a` → `2fe05e15` and pushes it to `origin` without force; `main` never carries fork commits.
Assumed: the fork becomes two forks, matching upstream: this repo (`Raydius/flare-local-inference`) carries only core patches on `2fe05e15`, and a new GitHub fork `Raydius/hyperflare-recipes` of `local-inference-lab/hyperflare-recipes` carries the fork's recipe, component, reference, model-test and `qualify` work on recipe `main`.
Assumed: all conflict resolution happens once, in one scratch replay of `deploy` onto `2fe05e15` + a revert of `17bb3c4a` (probe B's base); the resolved stack is then split by path into the two forks; the scratch base is never published.
Assumed: a path belongs to the recipe side when it lives under `hyperflare_recipes/`, `tests/recipes/`, `tests/reference/`, `qualify/` or `tools/`, is `uv.lock`, or is a fork-new test that imports `hyperflare_recipes`; everything else is core.
Assumed: the fork's own recipes (`elephant1`, `muse_glimmer_nvfp4`, the `qwen38_dense` additions) move to `hyperflare_recipes/recipes/` in the recipe fork and register there.
Assumed: core `patch/00`, `01`, `03`, `06`, `07` keep their core halves on `2fe05e15`; the recipe halves of `patch/06` and `patch/07` become commits in the recipe fork's single `deploy` stack, which has no patch branches.
Assumed: the recipe fork's `pyproject.toml` sources `ignition` from `Raydius/flare-local-inference` branch `deploy` and carries the fork's FlashInfer pin; its `uv.lock` pins both shas; core takes upstream's `pyproject.toml` FlashInfer line unedited.
Assumed: the fork's FlashInfer pin follows the recipe repo's lock: the three planned-JIT eviction cases run against `0aa01681`; all pass → the recipe fork takes upstream's line and the pin retires; any fail → the failing hunks of `cf1ac676a` are ported onto `0aa01681` as branch `evict-planned-jits-0aa01681` and pinned. `gdn-head-ratio.patch` is re-checked at the new base.
Assumed: the vllm-env image recipe builds from a core ref and a recipe ref (`build.sh <core-ref> <recipe-ref>`, both defaulting to `deploy`) and fails when the recipe lock's `ignition` sha differs from the core ref; the console's recipe probe imports `hyperflare_recipes.recipes` and falls back to `ignition.recipes` for older images.
Assumed: core's README and FORK.md describe the two-repo fork and the recipe fork's README gains a fork notes section; `scripts/fork/sync.sh` and `rebase.sh` run in either checkout unchanged (their refs are environment-configurable).
Assumed: the proof is the standing rebase gate set per repo, baseline-relative (core against pristine `2fe05e15`, recipes against pristine recipe `main`), plus the image built from both forks, the GPU 1 CUDA suites, `flare inspect`, and serving an existing core; evidence under `~/models/.cache/rebase-2fe05e15/`.
Assumed: backup tags `backup/pre-rebase-20261124-*` mark the eight core refs before anything moves; core refs publish with `--force-with-lease` against them, `base/current` last; the recipe fork's `main` and `deploy` are new branches, pushed without force.
Assumed: the upstream PR branches (`pr/muse-glimmer`, `pr/rope-scaling`), detached worktrees and `~/dev/flashinfer` are untouched; retargeting recipe PRs at the recipe repo is not part of this change.
Assumed: replayed commits keep their historical `Design-Signed-Off-By` trailers; commits this change writes in core are acknowledged against digest `eb2f`; the recipe fork follows its own repo's commit hooks.
Assumed: the fork's dense-Qwen image input (`1f486f63`, `751eeefc`) retires in favour of upstream's #386 (`0d45b17a`, same feature), as the fork retired superseded patches before (`dd62eb13`); the DaisyRose DSpark and rope-scaling work carries on top of #386, and the README's image-input row moves to Retired.
Assumed: `skip_specs: true` — the move carries every requirement of the existing capabilities.
Not included: moving the fork's upstream PR branches to the recipe repo — "the upstream PR branches … are untouched" was the standing scope, and the operator's choice names only the fork's own split.

## Why

`main` sits at `ba42481a`, 63 commits behind `upstream/main` `2fe05e15`, and `deploy` carries a 152-commit patch stack. In that range upstream split core from recipes: `da037284` moved every recipe, 109 non-Helios components, the reference implementations and the model tests into `hyperflare_recipes/`, and `17bb3c4a` moved them with history into `local-inference-lab/hyperflare-recipes`, along with `qualify/`, `tools/` and `uv.lock`. The fork's work falls on both sides (44 core-only commits, 26 recipe-only, 49 both), and the recipe repo has already moved on (10 commits since the split). Following upstream's layout keeps each fork a thin patch stack on its own upstream, so later recipe fixes arrive by rebase instead of by hand.

## What Changes

- `main` fast-forwards `ba42481a` → `2fe05e15` and is pushed to `origin` (no force).
- A scratch replay of `deploy` onto `2fe05e15` + revert of `17bb3c4a` resolves the 21 stops probe B found, with every resolution on upstream's current files, then moves the fork's own recipes into `hyperflare_recipes/recipes/`.
- The resolved stack splits by path. Core: `base/current`, `patch/00`, `01`, `03`, `06`, `07` and `deploy` are rewritten to their core halves on `2fe05e15`; commits left empty are dropped and named. Recipes: the recipe halves apply to recipe `4541fcb` (identical tree) and are rebased onto recipe `main` `07d7bb3` as the recipe fork's `deploy`.
- The recipe fork's `pyproject.toml` points `ignition` at the core fork's `deploy` and carries the FlashInfer decision; `uv lock` pins both.
- The vllm-env image builds from both forks; the console's recipe probe learns the new module path.
- Core README and FORK.md, and the recipe fork README, record the two-repo fork; the design-checklist digest moves `1af4` → `eb2f`.
- No behaviour change to any recipe, Helios policy, contract, or core format beyond what upstream brings.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. The move carries every requirement (`skip_specs: true`).

## Impact

- Core branches: `main` (fast-forward), `base/current`, `patch/00`, `01`, `03`, `06`, `07`, `deploy`; backups `backup/pre-rebase-20261124-*`.
- New repo: `Raydius/hyperflare-recipes` (GitHub fork, public like its upstream) with `main` (mirror) and `deploy`; local clone `~/dev/hyperflare-recipes`.
- Cores: existing cores keep loading — upstream's proto additions are new field numbers; the fork's `= 11` fields are untouched.
- Build: vllm-env `build/flare-ignition/build.sh` and `Dockerfile` take two refs; console `_RECIPES_SCRIPT` import.
- Out of scope: console re-exports and served re-measurement; the upstream PR branches and detached worktrees.

## Blast Radius

1. scripts/fork/sync.sh:42 — the in-place fast-forward of `main`; `git rev-list --count main..upstream/main` = 63, `upstream/main` = `2fe05e15`, and `main` = `origin/main` = `base/current` = `ba42481a` with `git rev-list --count upstream/main..main` = 0, so the mirror move is a pure fast-forward.
2. scripts/fork/rebase.sh:77 — `git rebase --onto <base> base/current deploy --update-refs`, the command that moves the stack and retags `base/current`; `git rev-list --count main..deploy` = 152, `git rev-list --merges --count main..deploy` = 0.
3. grepped `git diff --diff-filter=D --name-only ba42481a upstream/main` — 889 upstream-deleted files; grepped the `comm -12` against `git diff --name-only main deploy` — the fork edits 68 of them: `ignition/recipes/**` (qwen38, qwen38_dense, muse_spark, gemma4, mimo_v2, nemotron_h), 12 `ignition/components/**`, `qualify/speculative_benchmark.py`, `qualify/speculative_workload.py`, 31 extracted `tests/**`, `uv.lock`.
4. grepped `comm -12` of both changed-file lists — 109 files changed on both sides, including `helios/src/scheduler.rs`, `helios/src/scheduler/speculation.rs`, `helios/src/scheduler/tests.rs`, `helios/src/scheduler/policy_tests.rs`, `helios/src/server/request.rs`, `helios/src/server/runtime.rs`, `helios/src/server/runtime/phase_tests.rs`, `helios/src/model/prompt.rs`, `helios/src/model/packing.rs`, `flare-cli/src/cli.rs`, `flare-cli/src/main.rs`, `ignition/planning.py`, `pyproject.toml`, `uv.lock`.
5. grepped `'^STOP'` in `/home/rjdollete/models/.cache/rebase-2fe05e15/probe.stops` — 27 stop entries (`ca0291d2`…`1f486f63`); grepped `'^[0-9]+:.* !'` in `range-diff.txt` — 20 altered pairs of 152; probe artifacts `rebase.out`, `probe.stops`, `range-diff.txt`, probe tip `8747b8f4`.
6. grepped `git ls-tree -r --name-only` at `ba42481a`, `da037284`, `17bb3c4a^`, `17bb3c4a` — recipes 243 / 0 (+386 under `hyperflare_recipes/`) / same / 0: upstream moved the tree in `da037284` and updated it through 1bf68649 (registry inversion, `ignition/recipes/__init__.py` → `ignition/registry.py`) before deleting it, so a restore from `ba42481a` would bring back pre-inversion code; `git revert 17bb3c4a` on `2fe05e15` applies with no conflict (probe B, `25f1e676`).
7. ignition/cli/__init__.py:273 (upstream) — `def main(argv=None, *, recipes=(), prog="ignition", source_roots=())`; the recipe repo's `hyperflare_recipes/cli.py` passes `recipes=recipes()` and its `pyproject.toml` binds `ignition = "hyperflare_recipes.cli:main"` — the recipe fork inherits the CLI binding, no fork edit needed.
8. grepped `git show upstream/main:tests/preparation/test_core_boundary.py` — lines 41-42 assert `hyperflare_recipes` and `ignition/recipes` do not exist; the core fork after the split satisfies upstream's version unedited, and probe B's tip keeps `ignition/components` at 14 files, the same as `2fe05e15`.
9. pyproject.toml:25 — the fork pins `flashinfer-python[b12x] @ git+https://github.com/Raydius/flashinfer.git@cf1ac676a18544780afe525782a8f9610b017d58` (README.md:74, branch `evict-planned-jits-598c8547`); upstream `pyproject.toml:47` sources `local-inference-lab/flashinfer` branch `main` and upstream deleted `uv.lock` — the fork regenerates its lock at the new base.
10. helios/proto/runtime_core.proto:375, helios/proto/runtime_core.proto:436, helios/proto/runtime_core.proto:515 — the fork's `MUSE_GLIMMER_IMAGE_PREPARATION = 11`, `block_end_marker = 11`, `muse_glimmer = 11`; upstream's diff touches `RebuildManifest` 32–36, `GraphOperation` 14, `Execution` 8 — no collision, and the hunks are in disjoint regions, so the proto merges cleanly.
11. grepped `git diff --stat ba42481a upstream/main -- helios/manifest.md` — +52/−3 upstream; `git diff --name-only ba42481a deploy -- helios/manifest.md` names the fork's manifest edits — both sides edit the manifest spec, read both sides for the ABI/field tables.
12. grepped `git diff ba42481a upstream/main -- docs/TODO.txt` — 26 added lines; the fork renamed `docs/TODO.txt` → `docs/TODO.md` (stop-class carry, same as `b85c54d8` on 2026-11-18); AGENTS.md and docs/core-format-design.md:5 link the fork's target.
13. grepped `sha256sum` of the two `docs/design-review.md` revisions — upstream/main digest `eb2f…` vs deploy digest `1af4…`: the checklist digest the commit-msg hook checks moves, so every commit this change writes acknowledges `eb2f`.
14. README.md:6 and README.md:12 — the stack-base line and the "judged against `ba42481a`" sentence, both rewritten for `2fe05e15`; README.md:74 carries the FlashInfer pin row; FORK.md:150 gains the 2026-11-24 base-move entry.
15. grepped `heliopause` in the fork's changed-file list — none; upstream deleted the `heliopause/ignition` symlink in `da037284` and grepped `ignition` in both `heliopause/build.rs` revisions — zero hits, so the Rust extension build is unaffected.
16. `git status --porcelain` on `deploy` — ` M .claude/handoff.md` is the only tracked change; scripts/fork/sync.sh:23 refuses to run with tracked changes, so it is committed or stashed before the move.
17. grepped `'^U:'` in the probe's `probe.stops` — the stops split 50 `ignition/**`, 31 `tests/**`, 3 `helios/**` (`runtime_core/mod.rs`, `scheduler.rs`, `scheduler/speculation.rs`), 2 `qualify/**`, 2 `docs/**`, `pyproject.toml`, `uv.lock`, `flare-cli/src/cli.rs`; the Rust auto-merges in the 109-file overlap still take the read-both-sides pass.
18. grepped each active patch's feature on `upstream/main` (`git grep -l`) — `rope-parameters` 0 files, `muse_glimmer_nvfp4` 0, `elephant1` 0, `block_end_marker` 0, `generation_defaults` only as a rejected key — so `patch/01`, `03`, `06`, `07` and the fork recipes stay "Still needed"; each README row is re-verified at apply.
19. grepped `~/models/.cache/rebase-2fe05e15/probe2/probe.stops` — 21 stops; conflicts on `hyperflare_recipes/recipes/qwen38/*`, `qwen38_dense/*`, `muse_spark/attention.py`, `hyperflare_recipes/components/{parallel_draft,weights,paged_tma,dflash2}.py`, 3 `UA` adds under `hyperflare_recipes/reference/`, `DU ignition/recipes/qwen38_dense/contract.py`, 2 `AA` tests, core `ignition/{registry,declaration,planning}.py`, `ignition/cli/{__init__,preparation,bench,help}.py`, `ignition/preparation/{capture,frontend,runtime_core_pb2}.py`, `helios/src/{scheduler.rs,scheduler/speculation.rs,runtime_core/mod.rs}`, `flare-cli/src/cli.rs`, `pyproject.toml`, `uv.lock`; the probe tip `29946583` leaves 28 fork files under `ignition/recipes/{elephant1,muse_glimmer_nvfp4,qwen38_dense}`.
20. grepped `_RECIPES_SCRIPT` in the vllm-env console's flare_params module (line 558) — it runs `from ignition.recipes import recipes` inside the image; after the move that import fails and `_probe_recipes` returns `None`, so the console loses the image's recipe list unless the import learns `hyperflare_recipes.recipes`.
21. grepped `git rev-parse <commit>:<dir>` for `hyperflare_recipes`, `tests/recipes`, `qualify` at `25f1e676` (core + revert) and at `local-inference-lab/hyperflare-recipes` `4541fcb` — identical tree ids (`c34613db`, `a759b93c`, `f01e2634`): the recipe repo starts from exactly the tree upstream deleted, so the fork's recipe commits resolved on probe B apply to `4541fcb` unchanged; recipe `main` (`07d7bb3`) is 10 commits and 41 files past `4541fcb`.
22. grepped the recipe repo's `pyproject.toml` and `uv.lock` (clone `~/models/.cache/rebase-2fe05e15/hyperflare-recipes`) — `ignition = { git = "https://github.com/NVIDIA/hyperflare-inference.git", branch = "main" }` locked at core `425ab2d4`, `flashinfer-python` on `local-inference-lab/flashinfer` main locked at `0aa01681` (19 commits past the fork pin's base `598c8547`); `uv.lock` lives only in the recipe repo, so the fork's FlashInfer pin and its core pin move there.
23. grepped `~/models/.cache/rebase-2fe05e15/fork.class` (fork-changed paths on probe B's tip, `openspec/` excluded, classified by which upstream tree holds them) — 74 core, 70 recipe, 218 fork-new (139 `qualify/results`, 29 `ignition/recipes`, 16 `tests/recipes`, 8 `tests/reference`, 3 `hyperflare_recipes/reference`, 5 under `tests/{components,preparation,frontend}` that need an import check); 0 paths in both trees.
24. grepped patch ranges — `patch/00`, `01`, `03`: 0 recipe-side files; `patch/06`: 8 of 15; `patch/07`: 1 of 6; `deploy`'s 146 commits: 252 of 348.
25. scripts/fork/rebase.sh:77 — the stack mover whose `git rebase --onto … --update-refs` the core half runs directly (design step 11), on the `split/*` copies rather than through the script; grepped `gh repo view` — `Raydius/flare-local-inference` is PUBLIC and `Raydius/hyperflare-recipes` does not exist; `git filter-repo` is not installed, `git filter-branch` is.
26. grepped the vllm-env image recipe (`build/flare-ignition`) — its build script archives one fork ref into the build context (lines 9-24); its Dockerfile runs `uv sync --locked --extra prepare` in that tree (line 35) and builds `flare-cli` from it (line 47); both change to take a core ref and a recipe ref.
27. grepped `git log --oneline ba42481a..upstream/main` — `0d45b17a` "Add image input for Qwen3.8 27B (#386)" adds `hyperflare_recipes/recipes/qwen38_dense/vision_state.py` and rewrites `contract.py` (+22/−11); the fork's `1f486f63` adds its own `ignition/recipes/qwen38_dense/vision_state.py`, and upstream's `tests/recipes/test_qwen38_dense_{positions,vision}.py` (1128 lines) collide `AA` with the fork's (511) — two implementations of one feature that rename detection does not flag.
28. grepped probe tip `29946583` for `ignition\.(recipes|reference)` and moved-component module names in code and strings — lazy imports in the moved `mimo_v2` and `nemotron_h` bootstraps (ignition/recipes/mimo_v2/bootstrap.py, ignition/recipes/nemotron_h/bootstrap.py on `deploy`), `sys.modules` keys at `tests/cli/test_draft_checkpoints.py` line 121, `CompileJob.create("ignition.components.gated_norm_preparation:…")` at `tests/preparation/test_compile_pool.py` line 385, `tests/recipes/test_declarations.py` line 43, fork tests and `qualify/dflash2_acceptance.py` line 144 (both trees now hold it in `Raydius/hyperflare-recipes`) — rename detection moves files but never rewrites the module paths inside them.
