# Rebase pr/rope-scaling onto upstream main, resolving only ignition/cli/bench.py

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Rebase only the identified file"
- (from the PR thread, NVIDIA/spark-local-inference#212, original-el8 2026-11-08) "GitHub reports merge conflicts with `main`; local `git merge-tree` confirms them for PR head `2ca523b2aff6` against `main` at `91cfb3695f10`. Conflicting files: `ignition/cli/bench.py`. Please reconcile these files with main and rerun the checks affected by the resolution."

## Why

PR #212 (`pr/rope-scaling`, two commits `74f08a4b` and `2ca523b2` on base `0009b89d`) is `CONFLICTING` against `upstream/main` at `91cfb369`, twelve commits ahead of the base. GitHub will not merge it and the triage note asks for the reconciliation. Nothing else on the PR is blocked on the submitter.

## What Changes

Rebase the two PR commits onto `upstream/main` in the existing worktree `/tmp/rope-pr`, resolve the single conflicting hunk in `ignition/cli/bench.py` as the union of both sides, re-acknowledge both commits against main's `docs/design-review.md` (the hook digest changed on main), rerun the tests that exercise `bench.py`, and force-push with lease to `origin pr/rope-scaling`. The only other edit is the `kimi-k3.1-pro` alias that main added to `aliases()`, inserted into the PR's hardcoded alias tuple in `tests/cli/test_models.py:25` (operator correction "Add kimi", recorded on task 1.1) so that test passes on the rebased branch.

## Root Cause

order-of-operations: main's `run()` in `ignition/cli/bench.py` was rewritten to admit Kimi K3.1 (`upstream/main:ignition/cli/bench.py:134-137`) after the PR branched, and the PR's commit `2ca523b2` edits the same `resolve_model(...)` call on line 134 to pass `rope_parameters`; both sides changed one line.

## Blast Radius

1. `ignition/cli/bench.py:134` — the only conflicting line: the PR passes `getattr(args, "rope_parameters", None)` as a third `resolve_model` argument, main rewrote the same call and added the `("qwen38_dense", "kimi31")` recipe guard and the `--speculation-study` guard on the following lines (`upstream/main` lines 134-137); the resolution keeps main's lines and appends the PR's argument.
2. `.githooks/commit-msg:62` — compares each commit's `design-review=` digest to the staged `docs/design-review.md:1`, which changed on main since the base (`git diff --stat 0009b89d upstream/main -- docs/design-review.md`: 15 insertions, 3 deletions), so both replayed commits need a fresh acknowledgment.
3. `ignition/planning.py:1`, `tests/recipes/test_declarations.py:1`, and the PR's `docs/TODO.txt` (absent from this checkout) — touched by both sides but auto-merged (`git merge-tree --write-tree --name-only upstream/main origin/pr/rope-scaling` names only `ignition/cli/bench.py` as a conflict); no edit.
