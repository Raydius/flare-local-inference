# Rebase the patch stack and the DFlash2 branch onto upstream 585159c9

Tier: 3
Type: feature

## Stated Requirements

- "Based on the TODO list document in the repo as well as the relevant vault documentation, what is next?"
- Chosen in explore (2026-11-04): "Rebase onto upstream 585159c9" — "Second rebase of the stack and the DFlash branch; settles ABI 6, the dflash2.py drift, and whether c468f0d6 retires under #154. Re-export follows."
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."

Assumed: `main` stays the upstream mirror and moves to `585159c9` by fast-forward only; the stack (`deploy`, `patch/*`) and `wt/dflash-s-e2vk` move through `scripts/fork/rebase.sh` and `git rebase --onto`, as the 2026-11-02 rebase did.
Assumed: whether `c468f0d6` (warmup retention scoped to integrated GPUs) retires under upstream #154 is settled inside this change by one export from a tip without it; the decision, either way, is recorded for the upstream defect-PR step.
Assumed: the GPU re-export and served probe run on the yarn4 console profile (`qwen3.8-flash-dflash2-yarn4`, 1,048,576 context), the entry the sweep and the past-262k probe used; the fold entry is not re-exported.
Assumed: FlashInfer PR #67 stays pinned as a `Raydius/flashinfer` sha carrying the eviction fix on top of upstream's new pin; refreshing the PR itself upstream is queue item 3, not this change.

## Why

Upstream has moved 60 commits past the stack's base (`6fa01a49` → `585159c9`): it rewrote the DFlash2 component the fold's adapter sits on, bumped the package execution ABI to 6 with package-defined completion capacity, replaced the warmup-retention mechanism our `c468f0d6` works around, and moved its FlashInfer pin. Every upstream PR in the queue (owner issue, defect PRs, rope PR, recipe PR) rebases on upstream `main` at its time, so none can be cut from the current tree, and the defect-PR list cannot be fixed until #154 is tested against our export.

## What Changes

- `base/current` moves from `6fa01a49` to `585159c9`; `patch/00`…`patch/07` and `deploy` replay (one conflicting file, `flare/src/main.rs`); `main` fast-forwards to `585159c9`; the deploy README and `FORK.md` record the move.
- `wt/dflash-s-e2vk` replays its 24 non-merge commits onto the new `deploy`; five conflicting files resolve keeping both sides' behavior: `docs/TODO.txt` (modify/delete: upstream's six new entries go to `docs/TODO.md`), `flare/src/main.rs`, `ignition/components/speculation.py`, `pyproject.toml`, `uv.lock`.
- `ignition/recipes/qwen38_dflash/bootstrap.py` carries the parent hunk upstream applied to `qwen38`: `CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + completion_token_capacity))` replaces the literal 24 (identical bytes at width 8; the literal would fail validation if the draft count ever moved).
- The adapter (`ignition/recipes/qwen38_dflash/speculative_model.py`) is re-qualified against upstream's changed `components/dflash2.py` (`prepare_projection`, `PreparedWindowStaging`, `compact_prefill`, `DraftScratch.block_size`): parity suite on GPU 1, CPU suites, then the export.
- FlashInfer: upstream now pins `local-inference-lab/flashinfer@560fef16`; PR #67 (`1c5b88e2`, on `f5f5341d`) is rebased onto `560fef16` (zero conflicts probed) on `Raydius/flashinfer`, and that sha replaces `1c5b88e2` in `pyproject.toml` and `uv.lock`.
- `c468f0d6` decision: one export from a tip without the integrated-GPU gate on upstream's #154 pool. If it completes, `c468f0d6` is reverted on the branch and the defect-PR item is struck; if it OOMs, the gate stays and the incident evidence extends to #154.
- The rebased tip is built as `local/flare-ignition:<sha8>`, the yarn4 package re-exported on GPU 0 (ABI 6 image: a fresh entry regardless of upstream's "no re-export on engine upgrade" guarantee, since the image changes prepared code) and probed served at concurrency 1 and 8; results land in the vault tracker and the Fork Patch Notes.
- No behavior change to any recipe, Helios, contract, or package format: every resolution carries the code as it runs on the side that wrote it.

## Capabilities

### New Capabilities

- none

### Modified Capabilities

- none — a rebase carries every requirement; `qwen38-dflash2-speculation`, `history-drafts`, `dflash2-served-comparison`, `dflash2-draft-sweep`, `dflash2-acceptance-measurement`, and `kernel-preparation` keep their text (`skip_specs: true`).

## Impact

- Git: `base/current`, `patch/00`–`patch/07`, `deploy`, `wt/dflash-s-e2vk` rewritten and force-pushed with lease; `backup/pre-rebase-20261104-*` tags keep the old refs; `main` fast-forwarded.
- `Raydius/flashinfer`: one new branch (`evict-planned-jits-560fef16`) pushed; nothing merged anywhere.
- Images: one `local/flare-ignition:<sha8>` build (~15 min) plus a kernel-folder link; a second throwaway image for the `c468f0d6` probe. GPU 0 for two yarn4 exports (~48 min each) while the owner's instance is stopped; GPU 1 for the CUDA parity and mixed suites.
- Console: a new checkpoint profile duplicating *2026-11-04 Flare DFlash2 YaRN4* on the new image; existing entries keep serving on their images.

## Blast Radius

1. scripts/fork/rebase.sh:13 — `git rebase --onto <resolved> base/current <tip> --update-refs`; `base/current` is `6fa01a49`, `deploy` is `0fe9ece5` (14 commits above base), upstream `585159c9` is 60 commits past `6fa01a49` (`git rev-list --count 6fa01a49..upstream/main`).
2. grepped 'CONFLICT' across `git merge-tree --write-tree upstream/main deploy` — the stack's one conflict: `flare/src/main.rs`.
3. grepped 'CONFLICT' across `git merge-tree --write-tree upstream/main HEAD` — the branch's conflicts: `docs/TODO.txt` (modify/delete), `flare/src/main.rs`, `ignition/components/speculation.py`, `pyproject.toml`, `uv.lock`; 19 other files both sides touched auto-merge (`AGENTS.md`, `docs/cli-design.md`, `ignition/components/parallel_draft.py`, `ignition/recipes/qwen38/bootstrap.py`, `tests/recipes/test_qwen38_bootstrap.py`, …).
4. flare/src/main.rs:156,:161,:300 — patch/01 (`generation_defaults::`) and patch/03 (`serving_overrides::`) hunks against upstream's `cluster::require_transport()` / `CacheAgreement` load rewrite and `thinking_token_budget` (902c88e2, f6b73106); disjoint statements in the same function.
5. ignition/components/speculation.py:533 — our `copy_offered_mask[row] & ((1 << offered.clamp_min(0)) - 1)` (6cfd075b) in the record line upstream moved into `PreparedSpeculativeStateCommit` (`upstream/main:…speculation.py:546`, unmasked); the mask keeps ours, the class split keeps upstream's.
6. pyproject.toml:25 and uv.lock:403 — `Raydius/flashinfer@1c5b88e2` against upstream `local-inference-lab/flashinfer@560fef16` (3 commits past `f5f5341d`, on `origin/astra/clustered-pcie-allreduce-main`, not on their `main`); `git merge-tree --write-tree 560fef16 1c5b88e2` in `~/dev/flashinfer`: zero conflicts; PR #67's four files untouched between `f5f5341d` and `560fef16`.
7. ignition/recipes/qwen38_dflash/bootstrap.py:215 — `CompletionOutputs.allocate((2, slots, 24), …)`; upstream `5be93f8f` changed the parent to `VECTOR_TOKENS_FIELD + config.completion_token_capacity` (`upstream/main:ignition/recipes/qwen38/bootstrap.py:351`; ignition/components/completion.py:22 `VECTOR_TOKENS_FIELD = 16`); the derivative has no conflict and would silently keep the literal.
8. ignition/recipes/qwen38_dflash/speculative_model.py:126 (`PreparedComponentContext(...)`), :213 (`DraftScratch.allocate(c, …, batch * c.block_size, …)`), :331 (`PreparedSpeculativeCommit(...)`) — callers of upstream-changed signatures; every new parameter defaults (`compact_prefill=False`, `block_size=None`, `PreparedSpeculativeCommit` now subclasses `PreparedSpeculativeStateCommit` at `upstream/main:…speculation.py:609`), so the adapter compiles unchanged and parity decides equivalence.
9. ignition/components/dflash2.py:197 — `stage_windows` replaced upstream by `PreparedWindowStaging` (`window_staging.py`), `UnquantizedProjection` by `prepare_projection(..., workspace=)`; our BF16 draft resolves to the unquantized projection, checked by `tests/recipes/qwen38_dflash/test_draft_cuda.py` and `tests/components/test_dflash2.py`.
10. ignition/preparation/capture.py:67 — `c468f0d6`'s integrated-GPU gate on `retain_transient_backing()`; upstream #154 (`680be69a`, `heliopause/src/allocator.rs` +744 lines; ignition/preparation/package_storage.py:80 gains `flare_package_transient_allocate`) pools retained mappings instead of releasing; `upstream/main:ignition/preparation/capture.py:64` still retains unconditionally; auto-merge, decision by export.
11. ignition/components/parallel_draft.py:50 (`allocate_copy_buffers`), :110 (`admission=None, draft_page_blocks=None`) — upstream added `shortlisted_selector` (5639881c) and the private `PreparedSpeculativeStateCommit` path; ours added `allocate_copy_buffers`, `substitute_copies`, `admission`, `draft_page_blocks`; auto-merge, exercised by `tests/components/test_parallel_draft*.py` and `tests/recipes/qwen38_dflash/test_round.py`.
12. docs/TODO.md:229 — the last entry; `docs/TODO.txt` is deleted here, six entries added upstream (`git diff 6fa01a49 upstream/main -- docs/TODO.txt`): they append to `docs/TODO.md`.
13. AGENTS.md:312 — our only hunk (the TODO link); upstream rewrote the package-compatibility section (`27fc7ed2`: "Execution ABI labels do not gate loading"; "an absent or zero completion token capacity means eight") and added `docs/verification.md`; ours changed only the TODO link; auto-merge, the text becomes the branch's binding guidance.
14. FORK.md:131 (patch inventory) and `deploy:README.md:6-7` — the dated base-move entry; grepped `rope\|yarn\|generation-defaults\|max-context-tokens\|chat-template` across `6fa01a49..upstream/main` subjects: no upstream equivalent of patch/01, 03, 06, 07; no patch retires.
15. grepped 'git archive\|tag=' across ~/LocalApps/vllm-env/build/flare-ignition/build.sh — the image is built from `git archive` of a committed ref (`local/flare-ignition:<sha8>`); a new image id needs its kernel folder under `~/models/.cache/flare/kernels/images/` linked to `d8911aa8032e`'s (vault tracker section 12).
16. qualify/dflash2_served.py:48 `container_command` and `qualify/dflash2_draft_sweep.py` — the served probe drivers; unchanged by the rebase, run against the new entry.
17. tests/recipes/qwen38_dflash/test_draft_cuda.py:284 (`test_block_matches_reference`), tests/components/test_dflash2.py:18, tests/recipes/qwen38_dflash/test_round.py:15, tests/recipes/test_qwen38_mixed_cuda.py:165 — the requalification set; upstream also changed `tests/components/test_packed_speculation_cuda.py` and `tests/recipes/test_qwen38_bootstrap.py` (auto-merge).
18. grepped 'Now (2026-11-04)\|Current stack' across the vault notes `DFlash2 Recipe Progress.md` and `Flare Fork Patch Notes.md` (`/data/CloudVault/Home Lab/Physical Workstations/ProArt/Local AI/Control Plane/Flare/`) — the Now block and the Current stack section are rewritten at the end; grepped 'lessons' across openspec/ — no `openspec/lessons.md` to consult.
