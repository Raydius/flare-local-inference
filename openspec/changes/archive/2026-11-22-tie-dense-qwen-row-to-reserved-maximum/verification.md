# Verification — tie-dense-qwen-row-to-reserved-maximum

## Verdict: PASS

## Evidence audited
- [1. Compare the dense Qwen declaration with the reserved maximum. In th] — tests/recipes/test_declarations.py:130 — "            verification_width=model.recipe.max_draft_tokens + 1 if drafting else 1)"
- [2. Replace the recorded reading of the mismatch. Add a `correction:` l] — openspec/changes/archive/2026-11-21-fix-declaration-test-drift/tasks.md:12 — "  correction: the 193,865,984-byte declaration is not a recipe defect — it is the planning reservation for the supported sixteen-position ve"
