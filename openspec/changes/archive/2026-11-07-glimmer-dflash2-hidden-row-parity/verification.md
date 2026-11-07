# Verification — glimmer-dflash2-hidden-row-parity

## Verdict: PASS

Tier 1 inline pass (no auditor, no browser pass). `radflow-lint --tier-check`: clean — the diff is one file, 7 insertions / 4 deletions, inside the ≤4-file / ≤120-line envelope.

- Implemented: PASS — tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:47 `HIDDEN_RELATIVE_L2 = .04`, :295 `assert errors["hidden"] <= HIDDEN_RELATIVE_L2 and errors["logits"] <= RELATIVE_L2`, :19-23 the Tolerance paragraph naming both bounds and the measured floor; `RELATIVE_L2 = .02` at :44 untouched.
- Tested: PASS — the task's Verify ran on GPU 0 in the 8fe522a3 ignition container (log glimmer-dflash2-parity-fixed.log, line 13 `1 passed, 1 warning in 17.36s`), with the printed figures identical to the two failing runs before the edit (anchor hidden 0.016424991190433502, proposals hidden 0.02768363431096077, logits 0.0029/0.0074), so the edit changed the bound and nothing else the test measures.
- Reachable / Reused: n/a — a test constant; no new helper, no duplicated code (the sibling CUDA tests keep their own constants, Blast Radius item 4).
- Optimal: nothing reported — one constant beside the existing one, the comment one line, the docstring reworded in place.
- Faithful: PASS — bullet 1 (a parity failure is fixed in its own change) is this change; bullet 2 (the operator's authorization) covers the GPU-0 verification run the task cites; bullet 3 ("/rad-debug") is honoured by the Root Cause citing openspec/debug/glimmer-dflash2-proposal-rows-hidden-drift.md:26 (`outcome: confirmed — two oracles differing only in attention kernel differ by 0.029-0.036 on rows 4-15`). Reverse trace: all three hunks (docstring, constant, assertion) trace to the Root Cause; no other behavior changed.

## Evidence audited
- [In tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py add `HIDDEN_RE] — tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:19 — "Tolerance (BF16 accumulation order through five layers): relative L2 at most 4e-2 on the"

## Advisory rules judged
- Stage decay: none — no `## Follow-ups` heading, no `discovery:` lines, the one task ticked with apply-time evidence.
- Declined discoveries: none.
