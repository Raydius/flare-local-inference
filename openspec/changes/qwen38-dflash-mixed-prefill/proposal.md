# Prepare mixed prefill rounds for the qwen38_dflash package

Tier: 2
Type: feature

## Stated Requirements

- "Mixed-prefill fix"

Assumed: the fix lets the `qwen38_dflash` package capture and serve mixed rounds, where prefill and decode share a step, the same way the MTP package does.
Assumed: the mixed round keys on whether the residual region is present in the layout, not on recipe identity; MTP behavior stays byte-identical.
Assumed: the recipe declares mixed-prefill support, matching qwen38, and `ignition export --mixed-prefill` selects it, so the later served comparison runs both packages with the same scheduling.
Assumed: served qualification against MTP stays a separate change.

## Why

The DFlash2 package refuses `mixed_prefill` (`ignition/recipes/qwen38_dflash/bootstrap.py:79`), while the MTP package it will be compared against prepares mixed rounds by default (`ignition/recipes/qwen38/recipe.py:106`). A served comparison without mixed rounds would measure the scheduler difference, not the drafter. The TODO entry names one blocker — `prepare_residual` reads `target_residual`, which the DFlash layout omits — but reading the code shows a second: qwen38's `PreparedMixedRound` builds MTP's absorption and speculative round classes directly, so DFlash has no way to supply its own prompt-feature ring writes and tapped verification round.

## What Changes

- `PreparedDecoderState.prepare_residual` keys the residual checkpoint on whether the layout holds a `target_residual` region, not on `speculative_tokens`. MTP and dense layouts are unchanged.
- qwen38's `PreparedMixedRound` takes the prompt-absorption and speculative-round constructors as optional arguments that default to today's MTP classes; the qwen38 call site does not change.
- The packed mixed target forward carries the draft's feature taps when the speculative round asks for them, and the mixed state splits the packed features into prompt and decode buffers the same way it already splits hidden and residual.
- `qwen38_dflash` supplies a prompt-feature context writer and a widened `PreparedDFlashRound` that accepts a decoder factory and residual output, then prepares the mixed catalog with the same shapes qwen38 derives.
- The recipe declares `mixed_prefill=True`, so `--mixed-prefill` is accepted at export, and the config stops rejecting it.

## Capabilities

### New Capabilities

### Modified Capabilities

- `qwen38-dflash2-speculation`: the capacity requirement changes from `mixed_prefill=False` to mixed rounds prepared by default, and a new requirement covers what a DFlash mixed round writes.

## Impact

- Ignition only: the qwen38 mixed round and decoder state, the `qwen38_dflash` recipe, its export ranges, and their tests. No Helios, Protobuf, or package-format change; Helios already schedules `mixed` graphs from signatures (`helios/src/execution/manifest.rs:69`).
- Packages exported by `qwen38_dflash` gain mixed graphs and need re-export to use them. qwen38 packages are unaffected, but the qwen38 identity hashes every qwen38 source file (`ignition/recipes/qwen38/bootstrap.py:210`), so editing `mixed_model.py` and `decoder_state.py` re-keys qwen38 prefix caches; existing packages still load.
- Authorship constraint: AGENTS.md bars Claude Opus 5.5 from changing projection packing, fused arithmetic, stream overlap, or scratch aliasing in optimized model paths. This change composes existing operations and allocates new non-aliased buffers; it must not change `MixedPacking`, `MixedScratch` aliasing, or decoder scratch reuse. Any task that would needs a different author.

## Blast Radius

1. ignition/recipes/qwen38/decoder_state.py:271 — `prepare_residual` reads `target_residual` whenever `speculative_tokens` is set; the DFlash layout has speculative tokens and no such region.
2. ignition/recipes/qwen38/state.py:82 — `target_residual` exists only when `speculative_tokens and mtp`; the region-presence check reads this.
3. ignition/recipes/qwen38/mixed_model.py:35 — `PreparedMixedRound` hard-codes `PreparedPrefillAbsorption` and `PreparedSpeculativeRound`.
4. ignition/recipes/qwen38/mixed_state.py:201 — `PreparedMixedState.finish` splits hidden and residual into roles; packed feature taps need the same split.
5. ignition/recipes/qwen38/decoder.py:395 — taps are written before `state.finish()` at line 405, so the split can run in `finish`.
6. ignition/recipes/qwen38_dflash/speculative_model.py:235 — `PreparedDFlashRound` builds its own `PreparedDecoder`, with no decoder factory or residual output.
7. ignition/recipes/qwen38_dflash/speculative_model.py:219 — `PreparedPrefillContext` is the DFlash prompt ring write; the mixed prompt half needs the same write over gathered prompt features.
8. ignition/recipes/qwen38_dflash/bootstrap.py:79 — config rejects `mixed_prefill`; lines 162, 237, 403 size sample rows, graph count, and the memory report without mixed shapes.
9. ignition/recipes/qwen38_dflash/recipe.py:93 — declares `mixed_prefill=False`.
10. ignition/recipes/qwen38_dflash/export.py:7 — disposable ranges must cover the mixed round's new buffers.
11. ignition/recipes/qwen38/bootstrap.py:458 — the qwen38 mixed catalog loop; DFlash follows it and the qwen38 call stays byte-identical.
12. openspec/specs/qwen38-dflash2-speculation/spec.md:58 — requirement declares `mixed_prefill=False`.
13. grepped '"mixed"\|PreparedMixedRound\|mixed_shapes' across ignition/ — only qwen38 and qwen38_dflash build mixed rounds; deepseek4 and kimi3 only declare the kind.
14. grepped 'prepare_residual' across ignition/ — called from ignition/recipes/qwen38/decoder.py:314 and ignition/recipes/qwen38/mixed_state.py:166; no other recipe uses `PreparedDecoderState`.
15. tests/recipes/test_qwen38_mixed.py, tests/recipes/test_qwen38_mixed_cuda.py, tests/recipes/qwen38_dflash/test_bootstrap.py:42 — existing tests; the last asserts the mixed-prefill rejection this change removes.
