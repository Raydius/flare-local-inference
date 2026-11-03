# Rebase the patch stack and the DFlash2 branch onto upstream 6fa01a49

Tier: 3
Type: feature

## Stated Requirements

- "Pull the upstream NVIDIA changes into `main` branch and begin comparing their implementation of dflash to what we've done so far here."
- "main wasn't supposed to diverge, that's why we have the dflash branch."

Assumed: `main` is the mirror and already sits at upstream `6fa01a49` (reset 2026-11-02); the pull lands in the patch stack (`deploy`, `patch/*`) and in `wt/dflash-s-e2vk`, through the fork's `scripts/fork/rebase.sh`, never as a merge into `main`.
Assumed: the comparison with upstream's DFlash2 (`ignition/components/dflash2.py`, `qwen38_dense`) ends in this change as a recorded finding; folding `ignition/components/dflash_draft.py` into upstream's component is the following change.
Assumed: the existing DFlash2 packages keep loading (upstream still declares execution ABI 5 / model contract 3), so re-validation is an export on the rebased image plus a served probe, not a package-format migration.

## Why

Upstream has moved 61 commits past the stack's base (`56f5702a` → `6fa01a49`) with its own DFlash2 components, invocation-view fixes in the compiler, vocabulary-parallel sampling in the shared draft round, a collective ladder, and a floated FlashInfer pin; every further DFlash2 change here would otherwise be written against code upstream has already changed, and the component fold cannot start until both DFlash2 implementations are in one tree.

## What Changes

- `base/current` moves from `56f5702a` to `6fa01a49`; `patch/00`…`patch/07` and `deploy` replay (two conflicting files), the deploy README records the rebase, `FORK.md` gets its dated entry.
- `wt/dflash-s-e2vk` replays its 15 DFlash2 commits onto the new `deploy` (the `c977814b` merge of old `main` is dropped: its upstream side is in `6fa01a49`); nine conflicting files are resolved keeping both sides' behavior.
- `ignition/preparation/compilation.py`: upstream's invocation-view base offsets and "source reads the destination region" gather stay beside our `opaque_written` tracking; the three reproducers in `tests/preparation/test_aot.py` decide the merged predicate.
- `ignition/components/parallel_draft.py`: upstream's `vocabulary_parallel` path and removal of `draft_argmax` are kept with our proposal-only metadata, admission, `draft_page_blocks`, and history-copy buffers; the fork's `draft_argmax` block goes, as upstream deleted it.
- FlashInfer: upstream now pins `local-inference-lab/flashinfer@main` (locked `f5f5341d`); open PR #67 (`da45a722`, the planned-Triton-kernel eviction every Ignition Triton family needs) is rebased onto `f5f5341d` on `Raydius/flashinfer` and that commit replaces `da45a722` in `pyproject.toml` and `uv.lock`.
- `ignition/recipes/qwen38_dflash/{bootstrap,recipe}.py` carry the two hunks upstream applied to their parent `qwen38` since the base: the chat-template asset becomes `qwen38/assets/chat_template.patch.json` (upstream deleted the `.jinja`; without this every export fails after capture) and `collective_ladder` is passed through.
- `docs/TODO.txt` (deleted here, edited upstream): upstream's two new entries move into `docs/TODO.md`.
- `ignition/recipes/__init__.py` and `tests/cli/test_models.py`: `qwen38_dflash` joins upstream's recipe tuple beside `muse_spark`; the alias list gains `muse-spark-open-nvfp4`.
- The rebased branch is built as `local/flare-ignition:<sha8>`, the CPU suites and the GPU 1 CUDA mixed tests run in it, the DFlash2 package is re-exported on GPU 0 and probed served; the result and the component comparison land in the vault progress note.
- No behavior change to any recipe, Helios, contract, or package format: every resolution carries the code as it runs on both sides.

## Capabilities

### New Capabilities

- none

### Modified Capabilities

- none — a rebase carries every requirement; `qwen38-dflash2-speculation`, `dflash-draft-block`, `history-drafts`, `dflash2-served-comparison`, and `kernel-preparation` keep their text.

## Impact

- Git: `base/current`, `patch/00`–`patch/07`, `deploy`, `wt/dflash-s-e2vk` rewritten and force-pushed with lease; `backup/pre-rebase-20261102-*` tags keep the old refs; `main` untouched.
- `Raydius/flashinfer`: one rebased branch for PR #67 pushed, nothing merged there.
- Images: one `local/flare-ignition:<sha8>` build (~15 min) and a cold kernel folder (~7 min tune); GPU 0 for one DFlash2 export (~40 min) while the owner's instance is stopped; GPU 1 for the CUDA mixed tests.
- Console: the running panel profile on `local/flare-ignition:b91e06d7` keeps serving; nothing in `~/LocalApps/vllm-env` changes.

## Blast Radius

1. scripts/fork/rebase.sh:13 — `git rebase --onto <resolved> base/current <tip> --update-refs`; `base/current` is `56f5702a`, the stack tip `deploy` is `4718693a`, upstream `6fa01a49` is 61 commits past `acbf2858`.
2. grepped 'CONFLICT' across `git merge-tree --write-tree upstream/main deploy` — the stack's conflicts: `ignition/planning.py`, `tests/cli/test_models.py`; everything else auto-merges.
3. grepped 'CONFLICT' across `git merge-tree --write-tree upstream/main wt/dflash-s-e2vk` — the branch's conflicts: `docs/TODO.txt` (modify/delete), `ignition/cli/preparation.py`, `ignition/components/parallel_draft.py`, `ignition/planning.py`, `ignition/preparation/compilation.py`, `ignition/recipes/__init__.py`, `pyproject.toml`, `tests/cli/test_models.py`, `uv.lock`.
4. ignition/preparation/compilation.py:486 — our `opaque_written` set beside upstream's `source = node.args[1].meta['val']` gather predicate (cbb7d9d9); the comment line both sides rewrote is the textual conflict.
5. ignition/components/parallel_draft.py:122 — `draft_page_blocks`, proposal-only `metadata`, `admission` (ours) against `vocabulary_parallel` scratch and `draft_positions` (upstream #135); :208 — our `argmax` block, deleted upstream.
6. ignition/planning.py:496 — our `declared` context (patch/06) against upstream's `world_sizes` filter for `--vocabulary-parallel-sampling`.
7. ignition/recipes/__init__.py:33 — the recipe tuple with `qwen38_dflash`; upstream appends `muse_spark`.
8. ignition/cli/preparation.py:99 — `_check_draft_model` (b91e06d7) against upstream's `collective_ladder` resolution in the same function.
9. tests/cli/test_models.py:21 and :109 — the alias tuple and the supported-models message pin both sides' aliases.
10. pyproject.toml:25 and uv.lock:403 — `Raydius/flashinfer@da45a722` (PR #67, still OPEN, 4 files) against upstream `local-inference-lab/flashinfer@main` locked at `f5f5341d`, 26 flashinfer commits later; `f5f5341d` does not contain `da45a722`.
11. tests/preparation/test_compile_pool.py:298 — `test_worker_publishes_the_ignition_triton_program_it_planned` fails on a FlashInfer without PR #67; it decides whether the rebased pin still carries the fix.
12. tests/preparation/test_aot.py:682–842 — the five opaque-copy reproducers that pin our compiler fix; upstream's `tests/preparation/test_invocation_views.py` (192 lines, cbb7d9d9) pins theirs.
13. ignition/recipes/qwen38/decoder_state.py:270 — `prepare_residual` (our mixed-prefill region keying, 53729a21) in a file both sides edited (ours: mixed-prefill feature taps 53729a21; upstream: license headers and `collective_ladder`); auto-merge, re-run `tests/recipes/test_qwen38_mixed.py`.
14. ignition/recipes/nemotron_h/dflash.py:5 — imports `components/dflash_draft.py`, where ours lifted it into `components/dflash_draft.py` (−181 lines); upstream only added the license header; auto-merge.
15. FORK.md:78 and `deploy:README.md:17` — the dated base-move entry and the patch table; no patch is retired by this move (grepped `rope\|yarn\|generation-defaults\|max-context-tokens\|chat-template` across `acbf2858..upstream/main`: no upstream equivalent of patch/01, 03, 06, 07).
16. FORK.md:1 — the image is built from the committed tip, never the working tree (the console's `build/flare-ignition/build.sh` archives `git rev-parse --verify REF^{commit}`); the image tag follows the rebased branch tip.
17. ignition/recipes/qwen38_dflash/bootstrap.py:459 — `annotated_template_asset((manifest.snapshot / "chat_template.jinja").read_bytes(), … "qwen38" / "assets" / "chat_template.jinja")`; upstream `28f0833b` deleted that asset and `annotated_template_asset` now parses a `chat_template.patch.json` (`upstream/main:ignition/recipes/qwen38_dense/bootstrap.py:302`).
18. ignition/recipes/qwen38_dflash/recipe.py:60 and ignition/recipes/qwen38_dflash/bootstrap.py:163 — `PreparationConfig(...)` and `preparation_session(...)` without the `collective_ladder` their parent `qwen38` now passes.
19. grepped `dflash` across `upstream/main:ignition/recipes/qwen38/` — no DFlash2 for Qwen3.8-Flash-Next upstream; `components/dflash2.py`, `dflash2_geometry.py`, `dflash2_weights.py` (Muse-derived, TP-sharded, packed `speculative{b}x{w}p{packed}`) are the fold target of the following change, not touched here.
