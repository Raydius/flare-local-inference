# Comparison — the 2026-11-24 move and this change

The 2026-11-24 move (change `rebase-deploy-onto-upstream-2fe05e15`, branch `attempt/v0.3.2-rebase-20261124` in both forks) and this change produced nearly the same code on upstream `v0.3.2`. The earlier move fell short in what it required as proof, not in how it replayed commits: it excluded an export and a served request from its scope, and it judged each patch by whether its feature was absent upstream, not by whether upstream's added mechanisms reached it.

The decisive result is that this change, on the same `v0.3.2` base and the same FlashInfer pin, exported a package that failed in the same way. The package that serves comes from the later base, upstream core `c915b5f0` (27 commits past tag `v0.3.2`), recipes `7d70e230` and FlashInfer `0d48af53`.

## What was compared

| | 2026-11-24 move | This change |
|---|---|---|
| Change | `rebase-deploy-onto-upstream-2fe05e15`, archived on the attempt branch | `rebase-forks-onto-upstream-v0-3-2` |
| Core result on `c0c4f798` | `attempt/v0.3.2-rebase-20261124` at `65d8759b`, 148 commits | `backup/pre-rebase-20261126-wt` at `b14a8bce`, 153 commits |
| Recipes result on `07d7bb39` | same branch name at `d8cd2098`, 74 commits | same tag name at `cb9b93e5`, 83 commits |
| FlashInfer pin on that base | `Raydius/flashinfer` `d73da5c4` (on `0aa01681`) | the same commit |
| Final base | `c0c4f798` (`v0.3.2`) | core `c915b5f0`, recipes `7d70e230`, FlashInfer `675984cd` (on `0d48af53`) |
| Served export | none in scope | `gb202-tp1-202611261556`, served on 2026-11-26 (task 7.2) |

The tree diffs below compare the two results on the shared `v0.3.2` base, so they show the two moves and not upstream's later commits.

## Specification side

| Aspect | 2026-11-24 move | This change | Consequence |
|---|---|---|---|
| Proof of the move | "the standing rebase gate set per repo, baseline-relative", an image build, GPU 1 component suites, `flare inspect`, serving an existing core | the same gates, then an export of the operator's profile and served requests before any release | The earlier gates compare the fork with unpatched upstream, so a defect upstream's own stack has is invisible to them. |
| Exports | "Out of scope: console re-exports and served re-measurement" | tasks 7.2 to 7.4, each on the operator's go; "the release publishes only a stack that has served" | The first export on the earlier stack ran after the change was archived and its tag published. |
| Patch verdicts | each README row "checked against upstream commit subjects and flag names" (design step 12) | need shown on unpatched upstream, plus a search of every upstream enumeration the patch extends (design "Patch verdicts") | A patch whose feature upstream lacks still got "Yes" while an upstream mechanism added in the range bypassed it. |
| Specs | `skip_specs: true`: "The move carries every requirement" | three new capabilities and one modified, including `rope-scaled-export` | The earlier change had no requirement that a rope-scaled core records its scaling. |
| Split method | revert upstream's deletion commit `17bb3c4a` on a scratch base, replay once, split by a path list with `filter-branch` and `format-patch` | replay onto upstream's rename commit `da037284`, divide at `17bb3c4a`, prove by blob equality that each changed path is in exactly one fork | Both methods conserved the fork's content; the tree diff finds no file lost by either. |
| Image launcher | a hand-written `ignition` script in the image, with no check | the image build fails unless the launcher is the recipes entry point and stops before the argument parser when imported as `__mp_main__` (task 6.2) | See defect D3. |
| Dependency revision | the three eviction cases run on `0aa01681`; no kernel test of that revision on a GPU | the same on `v0.3.2`; after the failure, FlashInfer's RMSNorm test on GPU 1 gates the pin (tasks 9.2, 9.6) | Neither plan tested the pinned FlashInfer revision's kernels before the first export. |
| Assumptions the two share | the base is the tag; the pin follows the revision upstream recipes lock | the same, until the operator moved the base on 2026-11-26 | The shared assumption that upstream's locked FlashInfer revision is sound was false on `0aa01681`. |

## Work side

### Core, `65d8759b` against `b14a8bce`

`git diff --stat` outside `openspec/` and `.claude/`: 30 files, 900 insertions, 838 deletions. Fourteen are code or tests; the rest are records and release documents.

| Path | Difference | Verdict that differs |
|---|---|---|
| `helios/src/runtime_core/mod.rs` | this change bounds the recorded persisting L2 limit on the retained primary context and on both persistent-worker contexts; the earlier move bounded only the context Helios creates | persisting L2 limit: "Yes" there, adapted here |
| `helios/src/scheduler.rs`, `scheduler/speculation.rs`, `scheduler/policy_tests.rs` | this change leaves history-copy depths out of upstream's host-wide acceptance population, with a test | copy-depth attribution: "Yes" there, adapted here |
| `ignition/preparation/rebuild_manifest.py`, `tests/cli/test_reproduction_arguments.py` | this change emits `--rope-parameters`, `--recipe` and `--draft-model` in the reproduction arguments and tests that every export flag is recorded or listed as not prepared | `patch/06`, `patch/07`, `--recipe` / `--draft-model`: "Yes" there, adapted here |
| `ignition/preparation/bootstrap.py` | `common_preparation_kwargs` passes `plan.declared_context` here and `plan.recipe.context` there | `patch/06`: adapted here |
| `ignition/planning.py`, `ignition/declaration.py` | the earlier move refused a draft head format by `recipe.name == "qwen38_dense"` in core planning; this change reads a declared `draft_head_quantizations` field of the method | DSpark in the dense recipe: adapted here; the name test contradicts the rule that shared code never selects by recipe identity |
| `helios/src/scheduler/policy_tests.rs` | this change adds a test that terminal prefill chunks never divide an atomic span | block-end marker: adapted here |
| `scripts/fork/sync.sh`, `tests/test_fork_sync.py` | 72 lines removed, 14 added: this change carries only the tag handling the sync fails without | `patch/00`: adapted here |
| `scripts/set_release_version.py`, both workflows, `docs/release.md`, `docs/container.md` | both implement `-rN` tags; the refused forms and their tests differ | none |
| `README.md`, `FORK.md`, `docs/TODO.md`, `docs/model-preparation.md` | records; this change adds the base-move procedure to `FORK.md` and moves the per-model sections to the recipes fork | none |

### Recipes, `d8cd2098` against `cb9b93e5`

`git diff --stat`: 36 files, 401 insertions, 177 deletions.

| Path | Difference | Verdict that differs |
|---|---|---|
| `recipes/elephant1/recipe.py`, `recipes/muse_glimmer_nvfp4/recipe.py` | this change builds both configurations from upstream's `common_preparation_kwargs`; the earlier move listed each setting by hand | Elephant-1 and Muse Glimmer: fitted to upstream's shared configuration here |
| `recipes/qwen38_dense/contract.py`, `recipes/elephant1/contract.py` | the frontend argument is named `frontend` here and `frontend_override` there | none |
| `recipes/qwen38_dense/bootstrap.py`, `components/dflash2.py` | ordering and comments; one duplicated assignment removed | none |
| `recipes/qwen38/` | no difference | none |
| `tests/` (12 files), `docs/TODO.txt`, `README.md`, `pyproject.toml`, `uv.lock` | tests for the declarations above, the fork's follow-up list, and the lock | none |

The Qwen3.8 Flash Next recipe is byte-identical in the two results. The package failure on the earlier stack therefore has no cause in a difference between the two replays.

## Defects of the 2026-11-24 move

Sources: `openspec/debug/new-stack-package-nan-logits.md`, the superseded proposal `openspec/changes/audit-v0-3-2-rebase/proposal.md`, and task 8.1 of this change.

**D1. The first package prepared on the stack failed every request with "prepared model reported device error 1"** (ledger Symptoms; audit Blast Radius 6).

- Attribution: FlashInfer `0aa01681`, the revision upstream recipes `07d7bb39` locks and both moves pinned plus one commit. Its plan compiler builds the prepared grouped RMSNorm kernel without the FP32-gain setting (ledger H10, confirmed on GPU 1); the defect arrives with FlashInfer `f975f6c2e`, an ancestor of `0aa01681` and not of the previous pin's base `598c8547`. Upstream FlashInfer `c3ab1d04` removes it.
- Named difference: none between the two replays. This change's own export on that base and pin, `gb202-tp1-202611260714`, returned the same error. The package `gb202-tp1-202611261556`, exported after the move to FlashInfer `675984cd`, answers the same prompt with "Paris".
- What separates the two changes is scope: the earlier one archived and published `v0.3.2-r1` with no export, and this one stopped at the failed export before any release.
- Limit: no export ran from the unpatched upstream stack on `v0.3.2`, so the served failure is attributed to upstream's locked FlashInfer revision by the kernel test and the later passing export, not by a served upstream package.

**D2. A rope-scaled core's rebuild record named no rope scaling** (audit Root Cause and Blast Radius 1 to 3).

- Attribution: the earlier move's verdict rule. Upstream added the rebuild record inside the replayed range (`da037284`); it merged with the fork's `--rope-parameters` without a textual conflict, and the README row was judged "Yes" because upstream has no such flag.
- Named difference: `ignition/preparation/rebuild_manifest.py` (15 lines) and `tests/cli/test_reproduction_arguments.py` in this change. The record of `gb202-tp1-202611261556` holds `--rope-parameters '{"factor": 4.0, "original_max_position_embeddings": 262144}'`, `--recipe qwen38` and `--draft-model flare-draft`.

**D3. The preparation image's launcher re-ran the export in every compiler worker** (task 8.1; `vllm-env` commit `df77012`).

- Attribution: the image recipe written in the earlier move's task 4.1. Its hand-written `ignition` script called `main()` at import, and FlashInfer's compiler workers start by `multiprocessing` spawn, which imports the parent's main script as `__mp_main__`. No gate of that change ran an export, the only path that starts those workers.
- Named difference: the guard in `df77012` (2026-11-25, before this change's image work) and the build check of task 6.2, which fails the image build when the launcher reaches the argument parser on such an import.

**D4. `--draft-model` is declared on `ignition export` only, so `ignition debug-serve` cannot load a local draft folder** (audit Blast Radius 4).

- Attribution: carried behavior of the fork's patch, present before both moves; not introduced by either.
- Status: unchanged in this change (`ignition/cli/__init__.py` declares the flag on the export parser). It blocked the ledger's direct-execution diagnostic H7, which the confirmed cause made unnecessary.

**D5. The capture rewrite in `ignition/preparation/capture.py` as a cause of D1** (audit Blast Radius 5; ledger H9).

- Attribution: none; it was a candidate, not a defect. The same capture code is in the stack that serves, on both bases.

**D6. No step ran the patched stack end to end** (audit Root Cause).

- Attribution: the earlier proposal's scope line and its `skip_specs: true`. This change's `fork-base-move` capability requires a served export before a release.

## What this change did not do better

- Its first image on `v0.3.2` carried the same FlashInfer pin and failed the same way; its CPU gates and patch verdicts did not predict D1.
- It planned a hand-written fix for D1 before fetching FlashInfer's `main`, where the fix already existed (`openspec/lessons.md`, `check-upstream-for-the-fix-before-writing-it`).
- Muse Glimmer and Elephant-1 have not been exported on the final base when this comparison is written.

## Lessons recorded

- `a-base-move-is-proven-by-a-served-export` — D1, D6.
- `judge-a-patch-against-what-upstream-added-not-what-it-lacks` — D2 and the three adapted Helios and planning verdicts.
- `a-console-script-runs-again-in-spawned-workers` — D3.
