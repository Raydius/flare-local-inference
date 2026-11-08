# Rebase pr/muse-glimmer onto upstream main, resolving ignition/recipes/__init__.py

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "/rad-hotfix" (invoked on the exploration of NVIDIA/spark-local-inference#229, which concluded: rebase `pr/muse-glimmer` onto `upstream/main`, resolve `ignition/recipes/__init__.py`, add `kimi-k3.1-pro` to the alias tuple, rerun the registry tests, push)
- (from the PR thread, NVIDIA/spark-local-inference#229, original-el8 2026-11-08) "GitHub reports merge conflicts with `main`; local `git merge-tree` confirms them for PR head `67860fba4562` against `main` at `91cfb3695f10`. Conflicting files: `ignition/recipes/__init__.py`. Please reconcile these files with main and rerun the checks affected by the resolution."

## Why

PR #229 (`pr/muse-glimmer`, three commits `615a908e`, `3914db38`, `67860fba` on base `972e28e4`) is `CONFLICTING` against `upstream/main` at `91cfb369`, three commits ahead of the base. GitHub will not merge it and the triage note asks for the reconciliation.

## What Changes

Rebase the three PR commits onto `upstream/main` in the existing worktree `/tmp/glimmer-pr`, resolve the single conflicting region in `ignition/recipes/__init__.py` as the union of both sides (`kimi31` after `kimi3`, `muse_glimmer_nvfp4` last), insert `kimi-k3.1-pro` into the PR's hardcoded `aliases()` tuple in `tests/cli/test_models.py`, rerun the recipe-registry tests, and force-push with lease to `origin pr/muse-glimmer`. No re-acknowledgment is needed: `docs/design-review.md` is unchanged between the base and main, and all three commits already carry the matching digest.

## Root Cause

order-of-operations: main's Kimi K3.1 landing (`af0907d9`) inserted `kimi31` into the `recipes()` import block and return tuple in `ignition/recipes/__init__.py:32-38`, and the PR's first commit appends `muse_glimmer_nvfp4` to the same two lines; both sides changed the same region.

## Blast Radius

1. `ignition/recipes/__init__.py:35` and `ignition/recipes/__init__.py:38` — the import line and return tuple both sides edit; the resolution keeps both insertions in recipe order.
2. `tests/cli/test_models.py:24` — the PR's hardcoded `aliases()` tuple, which must gain `kimi-k3.1-pro` after `kimi-k3-dspark` once `kimi31` is registered.
3. `helios/src/runtime_package/protobuf.rs:1`, `tests/cli/test_planning.py:1`, `docs/model-preparation.md:1` — touched by both sides but auto-merged (`git merge-tree --write-tree --name-only upstream/main origin/pr/muse-glimmer` names only `ignition/recipes/__init__.py` as a conflict); the Protobuf additions use distinct field numbers (PR: `MuseGlimmerImageSettings` as field 10; main: `CompressedPreparedWeightCatalog` as field 17); no edit.
4. `.githooks/commit-msg:62` — `git diff --stat 972e28e4 upstream/main -- docs/design-review.md` is empty, so the existing `design-review=4d3a` acknowledgments remain valid through the replay.
