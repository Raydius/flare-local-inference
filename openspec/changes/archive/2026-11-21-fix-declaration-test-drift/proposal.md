# Green the Qwen and Elephant-1 declared-cost tests

Tier: 1 — one test file is edited; the other cited paths are read-only evidence for its expectations
Type: bug

External Review: Waived — Tier 1 (test-only expectations in one file, no behavior fork)

## Stated Requirements

- "I don't understand, what does this achieve?" — the value stated and accepted: restore the declared-cost-versus-prepared-layout guard and stop every future Qwen or Elephant-1 change from re-deriving which failures are pre-existing.
- "tier 1" — the operator chose the test fix over a `docs/TODO.md` note only.
- Assumed: only `tests/recipes/test_declarations.py` changes; the failures from an uninstalled checkpoint become skips, which the run record reports as unrun checks rather than passes.
- Assumed: no production file changes; if the corrected Qwen expectation exposes a genuine declared-cost mismatch, that becomes its own change instead of a fix folded in here.

## Root Cause

missing-edge-case: `tests/recipes/test_declarations.py` was not updated when two later changes landed and never covered an absent checkpoint. It compares the Qwen Flash recipe's speculation against a `Speculation` literal without the `method` field (`tests/recipes/test_declarations.py:53`) that the draft-method work added; it still expects Elephant-1 `max_concurrency == 2` (`tests/recipes/test_declarations.py:523`) although the recipe declares `concurrency=2, max_concurrency=4` (`ignition/recipes/elephant1/recipe.py:48`) with the concurrency-4 measurement recorded in `f774579e`; it calls `draft.DraftConfig.load` (`tests/recipes/test_declarations.py:104`) while `ignition/recipes/qwen38_dense/draft_config.py` exposes module-level `load` and `from_dict` with no such class; and the Kimi K3 cases die inside `resolve` at `ignition/recipes/__init__.py:249` (`checkpoint directory is unavailable: /data/models/Kimi-K3-MXFP4-CSF`) before the skip at `tests/recipes/test_declarations.py:88-89`, so a missing checkpoint reports as a failure instead of an unrun check.

## Blast Radius

1. `tests/recipes/test_declarations.py:53` — the `Speculation(draft_tokens=5, token_capacity=128)` literal lacks `method="mtp"`; with it, the assertion holds and the tie at `:67-70` passes at TP1/2/4 (122,014,720 / 61,246,464 / 31,091,712 bytes, run today against `HybridStateLayout.build`).
2. `tests/recipes/test_declarations.py:523` — expects `max_concurrency == 2`; `ignition/recipes/elephant1/recipe.py:48` declares 4, and the remaining assertions in that test (`:534`, `:537`) are unaffected by the ceiling.
3. `tests/recipes/test_declarations.py:104` — `draft.DraftConfig.load(draft_snapshot, config)`; the module API is `load(snapshot, target)` (`ignition/recipes/qwen38_dense/draft_config.py:23`).
4. `tests/recipes/test_declarations.py:84` and `:448` — the two `resolve(alias)` call sites that raise for the uninstalled Kimi K3 checkpoint; both need an absent-checkpoint skip that re-raises any other `UsageError`.
5. grepped `checkpoint directory is unavailable` across `ignition/` — one raise site, `ignition/recipes/__init__.py:249`; no other test-side helper for it.
6. grepped `DraftConfig` across `ignition/recipes/qwen38_dense` — no class definition; `DSparkConfig` lives at `ignition/recipes/qwen38_dense/dspark_config.py:40`, which the same test already uses for the kimi path at `tests/recipes/test_declarations.py:94`.
7. grepped `max_concurrency` across `ignition/recipes/elephant1` — the only source is the recipe declaration at `ignition/recipes/elephant1/recipe.py:48`, so no production change is implied by the test update.
