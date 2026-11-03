# Fold the DFlash draft component into upstream's shared DFlash2 component

Tier: 3
Type: feature

## Stated Requirements

- "Move ignition/recipes/qwen38_dflash off our ignition/components/dflash_draft.py (230 lines) onto upstream's ignition/components/dflash2.py (+ dflash2_geometry.py, dflash2_weights.py), behind a thin recipe adapter."
- "1. Console export from a committed-tip image"
- "2. qualify.dflash2_served --dflash2-model-id ckpt, trials 0:1 and 0:8, vs baselines above" — baselines: "Served code_edit: 329.1 tok/s @ conc 1, 516.5 @ conc 8 (requal baselines)"
- "3. Delete dflash_draft.py"
- "Fold = step 1 of upstreaming order; prerequisite for an upstream recipe PR."
- "Must carry: Flash-Next checkpoint contract | 64-token ring geometry w/ 5x2048 window | PreparedMixedPromptContext (mixed prefill) | admission validator | proposal-only execution | history copies (construction-time ValueError when combined w/ upstream's vocabulary-parallel sampling)." — the staging page size is the one 64 the component owns (`ignition/components/dflash2.py:21` fixes it at 128); the rings (5 × 2048 × 8 × 128 E4M3) and the recipe's 64-token KV pages are unchanged, so the window geometry is carried and the staging page size follows the component (see Assumed in design.md)
- "Upstream brings: TP-sharded embedding/head | packed verification speculative{b}x{w}p{packed}."
- "Compare the two rotaries: dflash2.py:31-37 vs dflash_draft.py:60-70. Rope scaling = later item, NOT the fold."
- "Planning only this session (propose's boundary)"
- "main stays the upstream mirror"; "Never land on deploy. Never rad-close this worktree — offer a commit instead"; "Commit/push only when asked"
- "Exports and serves run containerized from a committed ref (build.sh), never host venv"; "Owner launches console runs"
- "Update vault tracker Part A when state changes"

## Blast Radius

1. ignition/components/dflash_draft.py:1 — the component being deleted (`DraftScratch`, `Rotary`, `PreparedContext`, `DraftAttention`, `DraftMLP`, `PreparedDraftBlock`, `DraftForwards`, `ring_slots`)
2. ignition/components/dflash2.py:21 — upstream's component the recipe moves onto: `PAGE_TOKENS = 128` staging pages; `PreparedContext(weights, auxiliary, metadata, pool, …)` reads `metadata.{positions,starts,requests,state_rows}` and `pool.reserved_region(f"draft{i}.key")` (:51–:97); `DraftScratch.allocate(config, shard, tokens, device, buffers)` (:127); `PreparedDraft(…, embedding, head, output_multiplier, final_logit_softcapping)` (:204–:257) carries the embedding, head, and selector; component unchanged by this change
3. ignition/components/dflash2_geometry.py:11 — `DraftAttentionGeometry` the recipe's `DraftConfig.attention` must produce; `local(shard)` partitions heads
4. ignition/components/dflash2_weights.py:37 — `DraftWeights.load(snapshot, config, device, shard)` with `validate_metadata` (:76) replaces `qwen38_dflash/draft_weights.py`; same tensor inventory and BF16 check as `ignition/reference/qwen38_dflash2.py:129 expected_shapes`
5. ignition/recipes/qwen38_dflash/speculative_model.py:17 — imports the deleted component; `rings()` (:63) reads `pool.region("dflash.{i}.key")`; `PreparedDraft(PreparedDraftBlock)` (:114) appends the selector upstream's draft already owns; `SpeculativePreparation.begin/context/draft` (:195–:208) allocate per-round scratch and build the old context; `PreparedPrefillContext`, `PreparedMixedPromptContext`, `PreparedDFlashRound` (:224–:348) call `context(auxiliary, positions, rows, live)`
6. ignition/recipes/qwen38_dflash/config.py:22 — `from_dict` injects a top-level `rope_theta` for the old `Rotary`; needs `attention` (geometry) and `validate_shard` for the upstream loader/scratch
7. ignition/recipes/qwen38_dflash/draft_weights.py:1 — deleted; its `check_tensors` fault matrix moves to the upstream `validate_metadata` on the Flash-Next config
8. ignition/recipes/qwen38_dflash/bootstrap.py:157 — `DraftWeights.load(... )` gains the shard argument; :182 `HybridStatePool.allocate(draft_windows=…)`; :230 `SpeculativePreparation(...)` keeps its keyword surface
9. ignition/recipes/qwen38_dflash/export.py:17 — disposable ranges read `sampling_buffers`, `layer_buffers`, and the round's scratch; the component's `scratch_tensors()` replaces the per-round draft scratch
10. ignition/recipes/qwen38_dflash/recipe.py:34 — `DRAFT_WINDOWS = (5, 2048, 8, 128)` and `_AUXILIARY_BYTES` declarations; read, unchanged unless the export's memory report moves
11. ignition/recipes/qwen38/state.py:83 — the ring regions are named `dflash.{layer}.key/value`; the component reads `draft{layer}.key/value` through `reserved_region`, which `HybridStatePool` (:105) lacks (`PagedStateStorage.reserved_region`, ignition/components/paged_state.py:52, is the component's model)
12. ignition/recipes/nemotron_h/dflash.py:9 — imports `DraftForwards, PreparedContext, PreparedDraftBlock, Rotary` from the deleted component (lifted by c1498eca); upstream's self-contained file (`git show upstream/main:ignition/recipes/nemotron_h/dflash.py`, 236 lines) is the restore target; `nemotron_h/bootstrap.py` keeps its history-copy hunk from 6cfd075b
13. ignition/components/parallel_draft.py:164 — the shared round builds `o.forwards.context/draft` and reads `o.draft_scratch.rows(b*block, b*block)` (:180), `getattr(self.draft, "selector")` (:190), `o.copy_buffers` (:217, the construction-time `ValueError` with `vocabulary_parallel` at :219); unchanged
14. ignition/components/candidate_selector.py:26 — the fork's fused dot product (c1498eca); upstream's `PreparedDraft` calls the same `prepare`/`__call__`; unchanged
15. tests/components/test_dflash_draft.py:5 — `ring_slots` test; deleted with the component
16. tests/recipes/qwen38_dflash/test_draft_cuda.py:144 — parity test constructs the old component with `page_tokens=64`; rewritten against the upstream component with the recorded bounds (:1–:60) kept
17. tests/recipes/qwen38_dflash/test_round.py:15 — `DraftForwards`, `ring_slots`, `o.begin(BATCH, cursor)` (:78), `o.page_tokens == 64` (:92), `o.draft_scratch.logits.shape` (:104)
18. tests/recipes/qwen38_dflash/test_config.py:11 — `DraftWeights`, `check_tensors` from the deleted module; :27 asserts the injected `rope_theta`
19. tests/recipes/qwen38_dflash/test_layout.py:37 — region names `dflash.{layer}.{field}`
20. tests/recipes/test_qwen38_mixed_cuda.py:167 — builds the old `PreparedContext`/`Rotary` and `DraftWeights.assemble`
21. tests/recipes/test_qwen38_dense_dflash_cuda.py:57 — upstream's construction pattern (`Metadata`, `SimpleNamespace(reserved_region=…)`) the rewritten parity test follows; unchanged
22. tests/recipes/nemotron_h/test_dflash_cuda.py:17 — proves the restored Nemotron file (GPU 1, in the image; bit-identical error values recorded in `openspec/changes/archive/2026-10-31-qwen38-dflash2-recipe/tasks.md:10`)
23. qualify/dflash2_tap_check.py:156 and qualify/mixed_prefill_check.py:53 — import `PreparedPrefillContext` by name only; unchanged
24. openspec/specs/dflash-draft-block/spec.md:9 — requires the deleted module; retired by a REMOVED delta
25. openspec/specs/qwen38-dflash2-speculation/spec.md:20 — gains the component binding, liveness, and parity requirements the retired spec carried
26. docs/TODO.md:7 and :227 — the TP-sharding and fold entries
27. grepped 'dflash_draft' across ignition/ tests/ qualify/ docs/ — sites 1, 5, 12, 15–18, 20, 26 only; grepped 'dflash\.' region names — sites 5, 11, 19 only; grepped 'page_tokens=64|PAGE_TOKENS' across ignition/recipes/qwen38_dflash/ tests/recipes/qwen38_dflash/ — ignition/recipes/qwen38_dflash/speculative_model.py:32 (`PAGE_TOKENS = 64`), tests/recipes/qwen38_dflash/test_round.py:92, tests/recipes/qwen38_dflash/test_draft_cuda.py:174 only
28. openspec/changes/archive/2026-11-02-rebase-dflash-onto-upstream-6fa01a49/design.md:59 — bomb-squad traps consulted (derivative-recipe drift applies: the recipe stays a derivative of `qwen38`; compiler-rule conjunction and `draft_argmax` are untouched here); grepped 'lessons.md' across openspec/ — no lessons file exists in this repo

## Why

`qwen38_dflash` runs on a 230-line copy of the DFlash draft block while upstream `6fa01a49` ships the same block as `components/dflash2.py` for Qwen3.8-27B and Muse Spark; an upstream recipe PR built on the copy would fail the reuse rule, so the fold is step 1 of the upstreaming order (vault DFlash2 Recipe Progress §13).

## What Changes

- `qwen38_dflash` composes upstream's `PreparedContext`, `DraftScratch`, `PreparedDraft`, `DraftAttentionGeometry`, and `DraftWeights`; the recipe keeps only what the component does not own: the Flash-Next checkpoint pin and geometry, the hybrid-state ring regions, the feature-row adapter that feeds the component's metadata contract from Qwen's prefill/mixed/verification states, the proposal-only round with its admission validator, history copies, GDN commit, and the mixed prompt context.
- Ring regions are renamed `draft{layer}.key/value` (upstream's name for the same storage, `ignition/recipes/qwen38_dense/pool.py:37`) and `HybridStatePool` exposes them through `reserved_region`; the component's 128-token staging pages replace the recipe's 64; the draft scratch is allocated once at request capacity as upstream does. **BREAKING** for packages: the state layout's region names and the recipe identity change, so the package is re-exported (the plan already requires it).
- `nemotron_h/dflash.py` returns to upstream's self-contained file; `ignition/components/dflash_draft.py`, `qwen38_dflash/draft_weights.py`, and `tests/components/test_dflash_draft.py` are deleted.
- The parity, round, config, layout, and mixed CUDA tests move to the component's construction; bounds and the shortlist/walk equality are kept as recorded.
- Requalification: console export from the committed tip's image, then `qualify.dflash2_served` trials 0:1 and 0:8 against 329.1 / 516.5 tok/s.

## Capabilities

### New Capabilities

- none

### Modified Capabilities

- `dflash-draft-block` — all three requirements REMOVED (the module is deleted; Nemotron is back on its own code; parity and liveness move to the recipe spec)
- `qwen38-dflash2-speculation` — ADDED: the draft runs on the shared DFlash2 component with the Flash-Next contract; context writes respect liveness and recency; the prepared draft reproduces the DFlash2 reference under its storage contract

## Impact

- Code: `ignition/components/dflash_draft.py` (−230), `ignition/recipes/qwen38_dflash/{config,speculative_model,bootstrap,export,draft_weights}.py`, `ignition/recipes/qwen38/state.py`, `ignition/recipes/nemotron_h/dflash.py` (restore), six test files, `docs/TODO.md`, two spec deltas. Upstream's three component files are not edited.
- Packages: `gb202-tp1-202611021657` stays loadable but is superseded; the fold's export is a new entry under the console profile.
- Risk: upstream's rotary and norms compute in BF16 (`dflash2.py:37`, `Qwen3RMSNorm`) where the deleted block computed in FP32; the parity bounds (`test_draft_cuda.py:46`, rings 2e-2, hidden/logits 5e-2, exact shortlist and walk) and the served medians decide; no bound is relaxed.
- Hardware: GPU 1 in the `local/flare-ignition` image for CUDA suites; GPU 0 for the owner-launched console export and the served probe.
