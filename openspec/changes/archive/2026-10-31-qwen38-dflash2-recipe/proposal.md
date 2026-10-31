# Add the qwen38_dflash recipe: DFlash2 block drafts for Qwen3.8-Flash-Next

Tier: 3
Type: feature

## Stated Requirements

- "The previous Dflash development session crashed.  Help me determine where to resume."
- Chosen on resume: "DFlash2 change 2" — "Park the Nemotron ledger as-is, then propose the Qwen taps + qwen38_dflash recipe change."
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."
- Carried from the spike (2026-10-28): "I agree with this being a separate recipe."
- Carried from the spike (2026-10-28): "my usage is almost entirely code generation", "the server is for a single developer", "typically the concurrency is <8"

## Why

The acceptance spike (`qualify/results/dflash2-acceptance/`) showed DFlash2 beating the checkpoint's MTP by 15% on code_edit at width 8, the operator's regime; the decision recorded in `docs/TODO.txt:1` and the 2026-10-28 plan is a separate recipe built in three changes, of which the history-draft port (change 1) is archived and this is change 2.

## What Changes

- New recipe `ignition/recipes/qwen38_dflash/` (alias `qwen3.8-flash-dflash2`): the Qwen3.8-Flash-Next target with the `incoai/Qwen3.8-Flash-Next-DFlash2` draft (`a22a1744`, fetched into the hub cache like LiLiCorr) replacing the MTP module; imports the qwen38 decoder, weights, state, QSA/GDN verification, resume, maintenance, and contract modules rather than copying them.
- Qwen taps: the qwen38 decoder gains an optional feature tap that writes the four-stream mean of completed layers 5, 15, 25, 35, 45 into a `[tokens, 12800]` auxiliary buffer; today's `layer_outputs` trace and every existing call site stay byte-identical.
- Qwen state: `HybridStateLayout` gains optional draft K/V window regions (five layers × 2048 positions × 8 heads × 128, E4M3) and keeps the MTP-only regions out when no MTP draft is prepared; the transfer kernel's lag-one handling stays keyed to the MTP region names, so the new regions restore as plain state (draft lag 0).
- Shared DFlash draft block: Nemotron's `PreparedContext`, `DraftAttention`, `DraftMLP`, and `PreparedDraft` move to `ignition/components/dflash_draft.py`, widened with optional attention sink, BF16-or-NVFP4 projections, and an injected selector; Nemotron imports them with unchanged behavior and the new recipe pairs them with `PreparedCandidateSelector`.
- Shared round: `PreparedParallelDraftRound` gains an optional owner-supplied admission so a proposal-only round can run on a target whose state is not paged; the new recipe's round composes that proposal-only round with Qwen's `PreparedVerificationState`, GDN commit, and the shared verify/commit, recording the copy mask so `--history-drafts` works on day one.
- Prefill writes the draft context for every prompt token's tapped features; verification rounds write it for accepted tokens; `-no-mtp` graphs, resume, and maintenance follow the qwen38 catalog without draft QSA tables.
- Recipe declaration: `Speculation(draft_tokens=7, token_capacity=128)`, TP1 only, `mixed_prefill=False` for this change (the mixed round reads the MTP owner; deferred to change 3 if the served comparison needs it), measured `RankCosts` for rank 1.
- Tests: CPU contract/declaration tests; CUDA parity of the prepared draft block and selector against `ignition/reference/qwen38_dflash2.py` on the pinned draft; CUDA tap equality against the canonical reference at the five layers; Nemotron's existing `test_dflash_cuda.py` covers the lifted code.
- Docs: a Qwen3.8 Flash DFlash2 subsection in `docs/model-preparation.md`; the `docs/TODO.txt:1` decision entry resolved.
- Not in this change: served-throughput qualification against the MTP package (change 3), TP2/TP4 draft sharding, mixed prefill+decode rounds, the Nemotron compiled-forward fault (`openspec/debug/nemotron-serve-device-error.md`).

## Capabilities

### New Capabilities

- `qwen38-dflash2-speculation`: DFlash2 block drafting, draft-context storage, and verification for the Qwen3.8-Flash-Next target as an exportable, servable recipe.
- `dflash-draft-block`: the shared sliding-window DFlash draft block (context projection, noncausal block attention over a per-row ring, dynamic grouped convolutions, MLP) usable by any recipe that stores the draft windows in its state row.

### Modified Capabilities

- `history-drafts`: the shared round's copy substitution now also serves a round whose target admission is owner-supplied (Qwen DFlash2); existing requirements unchanged, one scenario added.

## Impact

- New files: `ignition/recipes/qwen38_dflash/{__init__,recipe,config,draft_weights,speculative_model,bootstrap}.py`, `ignition/components/dflash_draft.py`, tests under `tests/recipes/qwen38_dflash/` and `tests/components/`.
- Edited: `ignition/recipes/qwen38/{decoder,state,speculative_state}.py` (optional parameters only), `ignition/components/parallel_draft.py` (optional parameter), `ignition/recipes/nemotron_h/{dflash,bootstrap}.py` (imports), `ignition/recipes/__init__.py`, docs.
- Hardware: GPU 0 (RTX PRO 6000, 96 GB) exports and serves TP1; GPU 1 (RTX 5060 Ti, 16 GB) runs the component CUDA tests with fabricated or pinned draft weights.
- Package contract unchanged (execution ABI 5, contract 3); the new recipe declares `draft_lag=0`, no `draft_page_tables`, and the four copy inputs.
- Risk: the tap must equal the canonical layer output the draft was trained on; `qualify/layer_comparison.py` already measures prepared-vs-canonical layer residuals and the design uses it as the acceptance check.

## Blast Radius

1. `ignition/recipes/qwen38/decoder.py:378` — `layer_outputs[index].copy_(s.residual)` is the only per-layer tap today; the feature tap is added beside it
2. `ignition/recipes/qwen38/decoder.py:211` — `PreparedDecoder.__init__` gains the optional tap argument; `:238` validates `layer_outputs` geometry and is the model for validating the new buffer
3. `ignition/recipes/qwen38/state.py:33` — `HybridStateLayout.build` adds MTP regions whenever `speculative_tokens > 0` (`:65`); the DFlash layout needs the widened GDN/PLE/QSA histories without `target_residual*`/`draft_*` and with the new window regions
4. `ignition/recipes/qwen38/state.py:489` — exact-restore kernel treats regions named `draft_*`/`target_residual*` as lag-one draft state; the new regions must not match those prefixes
5. `ignition/recipes/qwen38/speculative_state.py:181` — `prepare_commit` writes `target_residual` and `target_residual_position`; the DFlash round needs the `last_hidden`/`previous_accepted` part only
6. `ignition/recipes/qwen38/bootstrap.py:375` — owner construction; `:440` prefill wrapping, `:452` round construction, `:494` resume wrapping, `:600` `draft_lag=1 if has_mtp`: the protocol the new bootstrap reproduces with `draft_lag=0`
7. `ignition/recipes/qwen38/maintenance.py:236` and `ignition/recipes/qwen38/state.py:631` — read `speculation.previous_accepted`, `speculation.draft_tables`, `speculation.cache[48]`; the DFlash owner exposes `previous_accepted` and `draft_tables=None`
8. `ignition/recipes/qwen38/export.py:19` — disposable ranges enumerate MTP owner fields by name; the new recipe needs its own list
9. `ignition/recipes/qwen38/decoder_state.py:271` — `maintain_draft_state` keys on `pool.layout.speculative_tokens`; the DFlash prefill state must not maintain draft QSA tables
10. `ignition/recipes/qwen38/recipe.py:1` — declaration precedent (`Checkpoint`, `RankCosts`, `Speculation`); `ignition/recipes/__init__.py:24` — registry gains `qwen38_dflash`
11. `ignition/components/parallel_draft.py:125` — `PreparedBatch` validator is built unconditionally and `:344` `propose()` runs it; `:301` proposal-only rounds already skip the target — the widening point for owner-supplied admission
12. `ignition/components/packed_speculation.py:84` — the only existing proposal-only consumer; its handoff of `drafts/probabilities/valid/copy_mask` is the pattern the Qwen round follows
13. `ignition/recipes/nemotron_h/dflash.py:42`, `:84`, `:121`, `:143` — `PreparedContext`, `DraftAttention`, `DraftMLP`, `PreparedDraft` lifted; `:201` `SpeculativePreparation` and the LiLiCorr glue stay
14. `ignition/recipes/nemotron_h/bootstrap.py:181`, `:226` — imports of the lifted classes
15. `ignition/recipes/nemotron_h/pool.py:43` — per-row `draft{i}.key/value` E4M3 window regions: the storage precedent for the Qwen layout
16. `ignition/components/candidate_selector.py:8` — `PreparedCandidateSelector` has no caller since 73a5611a; the new round calls it through `ignition/components/parallel_draft.py:399`
17. `ignition/components/grouped_convolution.py:35` and `ignition/components/window_attention.py:20` — reused unchanged
18. `ignition/reference/qwen38_dflash2.py:1` — reference forward and selector walk for CUDA parity tests
19. `tests/recipes/nemotron_h/test_dflash_cuda.py:1` — covers the lifted block; `tests/recipes/test_declarations.py:382` — declaration test with a draft checkpoint to extend
20. `ignition/recipes/nemotron_h/recipe.py:15` — draft fetch via `snapshot_download` into the hub cache: the pattern for the DFlash2 draft
21. `docs/model-preparation.md:342` — Qwen3.8 Flash section gains the DFlash2 export; `docs/TODO.txt:1` — decision entry resolved
22. `ignition/recipes/qwen38/mixed_model.py:35` — `PreparedMixedRound` reads the MTP owner's prefill and round; excluded by `mixed_prefill=False`
23. grepped 'DFlash2DraftModel\|selector_rank\|successor_codebook' across ignition/recipes ignition/components helios flare — no sites; the only integration is the removed Kimi one (`git show 73a5611a^:ignition/recipes/kimi3/dflash.py`)
24. grepped 'layer_outputs=' across ignition/ qualify/ — no production caller sets the decoder tap; the qualify tools consume traces only
