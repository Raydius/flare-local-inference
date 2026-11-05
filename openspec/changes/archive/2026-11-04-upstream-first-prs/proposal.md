# Open the first upstream PR set in order: owner issue, FlashInfer #67 refresh, Flare mixed-export PR

Tier: 2
Type: feature

## Stated Requirements

- "open the first upstream PR set in order: (1) owner issue to Luke Alonso on Flash-Next DFlash2 (shared component + YaRN; whether a mixed-prefill DFlash round is a variant of the existing graph family), gating every Flare PR; (2) FlashInfer: update open PR #67 (pr/evict-planned-jits 925ac3ab) and open the MoE direct-SFB guard PR (pr/moe-direct-sfb-flat-fragment 5b830a60) on local-inference-lab/flashinfer; (3) Flare: open the mixed-speculative-export PR (pr/mixed-speculative-export 9acb56f2 on 585159c9) on NVIDIA/flare-local-inference." — conflict: the MoE guard PR is superseded (FlashInfer merged Mark Yong's #71 `3c2729e7` on 2026-11-04 guarding both sites identically, and upstream Flare `5a1640c8` now pins FlashInfer `1f8ed8a3`, which contains it); the issue's premise moved (upstream `d68ed8f5` added Flash Next DFlash2 on the shared component, rejecting mixed prefill and without YaRN), so the issue is re-scoped to what the fork still carries.
- "FlashInfer tasks live in this repo's change (precedent: rebase change task 1.2)."
- "Opening each issue/PR is an operator leg."
- "Audit, don't trust, the branch drafts and vault section 18 / \"Upstream PR Drafts.md\"."
- "Read first: vault \"Control Plane/Flare/DFlash2 Recipe Progress.md\" Part A and sections 13, 17, 18; \"Upstream PR Drafts.md\"; archived rebase change tasks 1.2 and 2.2 discoveries."

Assumed: the change produces audited branches and texts; the operator opens each issue and PR (operator legs), and the drafting of (3) does not wait for the owner's answer to (1).
Assumed: the owner issue is filed on NVIDIA/flare-local-inference and re-scoped to the fork's remaining deltas over upstream's own Flash Next DFlash2: mixed-prefill DFlash rounds (upstream rejects them: "DFlash2 requires separate prefill and verification graphs"), YaRN serving (patch/06 + patch/07), and the fork's served measurements.
Assumed: "update PR #67" means force-pushing the audited `pr/evict-planned-jits` commit onto the PR's head ref `Raydius/flashinfer:fix/evict-planned-jits` (a PR's head branch cannot be renamed), rebased onto FlashInfer `origin/main` `1f8ed8a3` where it merges clean.
Assumed: the Flare PR branch is rebased onto upstream `main` `0009b89d` (merges clean; upstream still builds the mixed decoder in verification mode and forbids `draft_tokens` on mixed kinds), its CPU tests re-run there, and its commits gain the DCO `Signed-off-by` line CONTRIBUTING.md requires (both commits lack it today).

## Blast Radius

1. grepped `git diff --stat 42ba2f16 pr/evict-planned-jits` in ~/dev/flashinfer — branch `pr/evict-planned-jits` `925ac3ab` touches 4 files (144/8): `git merge-tree --write-tree origin/main pr/evict-planned-jits` merges clean onto `1f8ed8a3`; PR #67 head is `fix/evict-planned-jits` `da45a722` (stale, on `f5f5341d`); no `Signed-off-by` on the commit (FlashInfer's recent main commits carry none either).
2. grepped `git merge-tree --write-tree origin/main pr/moe-direct-sfb-flat-fragment` in ~/dev/flashinfer — CONFLICT in the dynamic MoE kernel: `3c2729e7` "Fix moe sfb double reshape (#71)" guards both the main and the swapped-FC1 flatten sites with the same `cute.rank(...) == 2` test and adds live-graph oracle tests); dropped, nothing to open; only the fork pin `pyproject.toml:25` (`Raydius/flashinfer@49fe5449`) still carries it until the next rebase re-pins.
3. pyproject.toml:25 — the fork's FlashInfer pin `49fe5449` = `42ba2f16` + #67 + the guard; upstream `main` now pins `1f8ed8a3` (`git show upstream/main:pyproject.toml` line 25), so the "pin lags code" note in the drafts is stale and is not sent.
4. grepped `git diff --stat 585159c9 pr/mixed-speculative-export` — branch `pr/mixed-speculative-export` `9acb56f2` (local + `origin`, worktree /tmp/review-defects) touches 8 files (126/13); `git merge-tree --write-tree upstream/main pr/mixed-speculative-export` merges clean onto `0009b89d`; upstream `d68ed8f5` changed `ignition/recipes/qwen38/decoder.py` (28 lines) and `0009b89d` changed `helios/model-contract.md`, so the branch is re-based and its CPU suites re-run; `git log --format=%(trailers:key=Signed-off-by)` on both commits is empty.
5. ignition/recipes/qwen38/mixed_model.py:74 at `upstream/main` still passes `verification=True`, and `ignition/preparation/execution.py:86` at `upstream/main` still rejects `draft_tokens` on non-speculative kinds — the defect the PR fixes is live upstream.
6. ignition/recipes/qwen38_dflash/bootstrap.py:315 — the fork's mixed-prefill DFlash rounds, the capability upstream `d68ed8f5` rejects (`git show upstream/main:ignition/recipes/qwen38/bootstrap.py` line 126) and the issue asks about.
7. ignition/recipes/qwen38_dflash/recipe.py:93 — the fork's `qwen3.8-flash-dflash2-yarn4` declaration; upstream `recipe.py` at `upstream/main` has no `rope_scaling` (grepped 'rope\|yarn' — none), so YaRN stays a fork delta the issue names.
8. CONTRIBUTING.md:9 — DCO sign-off required on every Flare commit; `git log -5 upstream/main` shows the owner's commits unsigned but the external contributor's (`0009b89d`) signed.
9. docs/design-review.md:159 — the `Design-Signed-Off-By` trailer and `.githooks/commit-msg` hook every re-created commit must satisfy.
10. grepped 'Owner issue\|FlashInfer PRs\|Defect PR' in the vault note "DFlash2 Recipe Progress.md" (Control Plane/Flare) — Part A Next items 2–4 and 8 and the Later list are rewritten for the three findings above; section 18's "dropped" list gains the MoE PR.
11. grepped '^## ' in the vault note "Upstream PR Drafts.md" (Control Plane/Flare) — three draft sections plus Dropped; the texts the operator pastes; items 2 and the pin-lag note are retired, item 1 and 3 are re-audited, the issue text is added.
12. grepped 'Signed-off-by' across `.githooks/commit-msg` — the hook does not add or require it; sign-off is added with `git commit -s` / `git rebase --exec 'git commit --amend --no-edit -s'`.
13. No file in this repository's tracked tree changes except `openspec/changes/upstream-first-prs/**`; every code edit lives on the PR branches of the two repositories (grepped `git status --short` — clean).
