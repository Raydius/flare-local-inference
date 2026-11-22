# Verification — fix-declaration-test-drift

## Verdict: PASS-WITH-FINDINGS

## Evidence audited
- [1. Correct the Qwen Flash speculation expectation. In `tests/recipes/t] — tests/recipes/test_declarations.py:64 — "    assert recipe.speculation == Speculation(draft_tokens=5, token_capacity=128, method=\"mtp\") and recipe.max_draft_tokens == 7"
- [2. Match Elephant-1's declared concurrency. In `tests/recipes/test_dec] — tests/recipes/test_declarations.py:536 — "    assert (recipe.concurrency, recipe.max_concurrency, recipe.prefill_chunk) == (2, 4, 16384)"
- [3. Call the qwen38_dense draft-config module API. In `tests/recipes/te] — tests/recipes/test_declarations.py:115 — "        config = replace(config, draft=draft.load(draft_snapshot, config))"
- [4. Skip uninstalled checkpoints instead of failing. In `tests/recipes/] — tests/recipes/test_declarations.py:23 — "def resolve_installed(alias):"

## Findings
1. `tests/recipes/test_declarations.py` still reports one failure: `test_fp8_cache_views_and_planned_bytes[True-qwen3.8-27b-1]` asserts `pool.layout.row_bytes == costs.state_row_bytes` at `:146` and gets 186,001,664 against the declared 193,865,984 (`ignition/recipes/qwen38_dense/recipe.py:99`). Neither side was patched here: the mismatch is a dense Qwen recipe decision — lower the declared constant to the 8-wide row preparation allocates, or prepare the 16-wide verification the constant assumes. It is recorded with its arithmetic on the box-3 `discovery:` line as a terminal `declined:` disposition and left visible rather than skipped, so the file's remaining red line names a real defect instead of hiding one.
   Dimension: Tested
