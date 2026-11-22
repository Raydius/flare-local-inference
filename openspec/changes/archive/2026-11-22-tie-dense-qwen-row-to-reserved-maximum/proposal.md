# Tie the dense Qwen declared row to the reserved sixteen-position maximum

Tier: 1 — one test file is edited; the recipe, pool and documentation cited below are read-only evidence
Type: bug

External Review: Waived — Tier 1 (one test's comparison in one file, no behavior fork)

## Stated Requirements

- "What is your recommendation, and what is the goal?" answered with: fix the test comparison, leave the recipe alone, send nothing upstream; goal is that `tests/recipes/test_declarations.py` checks the documented reservation rule and reports no failure.
- "yes, proceed" — the operator accepted that recommendation.
- Assumed: only `tests/recipes/test_declarations.py` changes besides this change's records and one `openspec/lessons.md` entry; the Kimi K3 cases keep skipping because that checkpoint is not installed.

## Root Cause

wrong-mental-model: the dense Qwen branch of `test_fp8_cache_views_and_planned_bytes` (`tests/recipes/test_declarations.py:127`) built the continuation row at the widest prepared verification width, eight, and required it to equal the declared `RankCosts.state_row_bytes` of 193,865,984 (`ignition/recipes/qwen38_dense/recipe.py:99`). The declaration is a planning reservation for the widest verification the recipe supports, not the row a default preparation allocates: `docs/model-preparation.md:397` states "Planning reserves 193,865,984 continuation bytes per row, the supported sixteen-position maximum; narrower preparations allocate their exact smaller continuation layout", and the recipe comment at `ignition/recipes/qwen38_dense/recipe.py:96` says the same. The archived change `2026-11-21-fix-declaration-test-drift` recorded the resulting failure as a recipe defect; that reading is replaced by this one.

## Blast Radius

1. `tests/recipes/test_declarations.py:127-134` — the dense Qwen branch: the pool is built at `max_draft_tokens + 1` (sixteen positions with `max_draft_tokens=15`, `ignition/recipes/qwen38_dense/recipe.py:93`), and the row of the widest prepared verification (`max(speculation.widths)`, eight) must be smaller.
2. `tests/recipes/test_declarations.py:151` — `assert pool.layout.row_bytes == costs.state_row_bytes` is unchanged and now compares the sixteen-position row with the declaration.
3. `ignition/recipes/qwen38_dense/pool.py:39` — `continuation_layout(config, mtp, verification_width)` accepts widths 1 to 16; read-only.
4. `ignition/recipes/qwen38_dense/bootstrap.py:238` — preparation allocates at the largest configured verification width; read-only, which is why a seven-draft preparation holds a 186,001,664-byte row.
5. `openspec/changes/archive/2026-11-21-fix-declaration-test-drift/tasks.md:11` — box 3's `discovery:` line calls the mismatch a defect; a `correction:` line there points to this change.
