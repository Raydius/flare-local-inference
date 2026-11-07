# Bound the Muse Glimmer DFlash2 parity test's hidden rows by the oracle's measured noise floor

Tier: 1 — one test file, one constant and its docstring; no behavior fork
Type: bug

External Review: Waived — Tier 1 (a test tolerance in one file, root cause confirmed by ledger)

## Stated Requirements

- "a parity failure (relative L2 above 2e-2, or a first proposal that is not the target's greedy token) is recorded as a `discovery:` and fixed in its own change; this change edits no source" (glimmer-dflash2-parity-test-run proposal.md, Assumed line the operator let stand; this is that own change)
- "It's clear, go ahead and do it" (the operator's authorization for the agent's GPU-0 container runs on 2026-11-07; the fix's verification run rides on it)
- "/rad-debug" (the operator chose diagnosis over the tolerance-only proposal; the Root Cause below cites the ledger)

## Root Cause

wrong-mental-model: `RELATIVE_L2 = .02` (tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:44) bounds the block's normalized hidden rows as if they were as stable as the anchor row, but the drafter's five layers with the two-tap block convolution compound BF16 rounding along the block: the oracle disagrees with itself by 0.029-0.036 on rows 4-15 when only its attention kernel changes (openspec/debug/glimmer-dflash2-proposal-rows-hidden-drift.md:26), the same ramp-then-plateau and magnitude as the serving drift 0.026-0.034, while the logits (0.0074, bound held) and the selector walk (15 of 15 tokens equal to the oracle's) show the drafter itself is right. The hidden-row bound sits below the reference's own noise floor; the logits bound is calibrated.

## Blast Radius

1. tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:44 — `RELATIVE_L2 = .02`, the one constant both comparisons read; the hidden comparison gets its own bound
2. tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:19-21 — the docstring's Tolerance paragraph states 2e-2 for both; it names the two bounds and the measured floor
3. tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:292 — the assertion `errors["hidden"] <= RELATIVE_L2 and errors["logits"] <= RELATIVE_L2`
4. grepped 'RELATIVE_L2' across tests/recipes/muse_glimmer_nvfp4/ — test_reference_cuda.py and test_vision_cuda.py define their own constants for the target and the vision tower; untouched (their rows have no block chain)
5. ignition/components/dflash2.py:250 — `window=a.window - 1`, the one-key mask difference, is carried, shared by the Spark and Qwen drafters, and refuted as the cause (ledger H2); no edit
6. grepped 'relative L2 at most 2e-2' across openspec/changes/ — the open operator leg of glimmer-dflash2-parity-test-run (box 1) whose Verify names "relative L2 at most 2e-2"; it is re-read against the fixed test once this lands, by the operator
