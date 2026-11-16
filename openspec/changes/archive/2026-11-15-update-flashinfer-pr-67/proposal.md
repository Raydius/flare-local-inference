# Move FlashInfer PR #67 onto upstream's rewritten main

Tier: 1
Type: feature

External Review: Waived — Tier 1 (one README line in this repo; the PR's commit already exists and was tested under `rebase-deploy-onto-upstream-29192ef7` task 1.2)

## Stated Requirements

- "Update PR #67 appropriately"

Assumed: PR #67 is `local-inference-lab/flashinfer#67` (author Raydius, open). `NVIDIA/hyperflare-inference#67` is Mark Yong's merged PR and the fork has no PR #67.
Assumed: "appropriately" means the PR shows only what upstream still lacks: its head becomes the one commit already carried in the fork pin, and its title and body describe that commit.
Assumed: the PR stays open on its existing head ref; it is not closed and reopened.
Assumed: pushing the head ref and editing the title and body are operator-approved steps, as in `upstream-first-prs` task 1.2.

## Why

FlashInfer's `main` history was rewritten after the 2026-11-04 refresh, so PR #67 (head `95334c3e` on `1f8ed8a3`, which is no longer in `main`) now reports 100 commits, 880 files and `CONFLICTING`. Upstream `810c0849f` also landed its own planned-JIT registry (`_PLANNED_TRITON_JITS`), which replaces the PR's main hunk; two gaps remain and the fork carries them as one commit, `7245b1e0`, on upstream's tip `4fd836dc`.

## What Changes

- PR head ref `Raydius/flashinfer:fix/evict-planned-jits` moves `95334c3e` → `7245b1e0` (or that commit rebased, if `origin/main` has moved past `4fd836dc` at apply time), pushed with lease. `pr/evict-planned-jits` follows it.
- PR title becomes the commit's subject: "Reject unresolved planning stubs in compiler workers and sweep the MLA compress memo".
- PR body is rewritten for the remaining change: what `810c0849f` already covers, the two gaps, and validation on `4fd836dc`.
- README's FlashInfer pin row drops "predates the re-cut" and names the PR head.
- Vault notes "Upstream PR Drafts" section 1 and the DFlash2 tracker record the new head, title and body.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. `upstream-contribution-branch` governs Flare PR branches (sign-off and design-review trailers); FlashInfer carries neither.

## Impact

- `local-inference-lab/flashinfer#67`: head, title, body. Thread and number are kept.
- `Raydius/flashinfer`: `fix/evict-planned-jits` and `pr/evict-planned-jits` force-updated with lease; `evict-planned-jits-4fd836dc` (the Flare pin's branch) is untouched.
- This repo: `README.md` one line. `pyproject.toml` and `uv.lock` are unchanged, because the pin already names `7245b1e0`.
- The commit's `Co-Authored-By: Claude Opus 5.5` trailer is kept byte-identical; that session wrote the re-cut.

## Blast Radius

1. README.md:50 — "([FlashInfer PR #67](…), open, predates the re-cut)" becomes stale once the head moves.
2. README.md:48 — pin row heading `Raydius/flashinfer@7245b1e0`; read only, already correct.
3. pyproject.toml:25 — pin `Raydius/flashinfer.git@7245b1e0…`; read only, the same commit the PR head takes.
4. grepped 'mergeable|commits|changedFiles|headRefOid' in `gh pr view 67 -R local-inference-lab/flashinfer --json …` (outside this repo) — `CONFLICTING`, 100 commits, 880 files, head `95334c3e`.
5. grepped '1f8ed8a3' in `git -C ~/dev/flashinfer log origin/main` (outside this repo) — not an ancestor; `origin/main` = `4fd836dc`, merge base with the PR head is `8f4ee770`, 2430 commits behind.
6. grepped `_PLANNED_JITS|_NATIVE_JITS|_PLANNED_TRITON_JITS` in `origin/main:flashinfer/experimental/b12x/_lib` — upstream has its own registry; the fork's `_PLANNED_JITS` is absent and no longer needed.
7. grepped 'changed' in `git -C ~/dev/flashinfer show --stat 7245b1e0` (outside this repo) — three files (`_lib/compile_pool.py` +5, `attention/mla_compress/_cute.py` +2/−3, `test_first_use_compile.py` +100), parent `4fd836dc`.
8. grepped 'evict' in `git -C ~/dev/flashinfer ls-remote fork` (outside this repo) — `fix/evict-planned-jits` and `pr/evict-planned-jits` at `95334c3e`; `evict-planned-jits-4fd836dc` at `7245b1e0`.
9. grepped 'error' in `gh run view 41957794022 -R local-inference-lab/flashinfer --log-failed` (outside this repo) — PR check `label` fails with "Parameter token or opts.auth is required": the workflow has no token on fork PRs; not caused by the branch and not fixable from it.
10. grepped `#67` across `README.md`, `FORK.md`, `docs/` — FORK.md:67 refers to the unrelated Flare PR #67; no further sites.
11. grepped 'PR' across openspec/lessons.md — the file is absent; no recorded scenario to consult.
