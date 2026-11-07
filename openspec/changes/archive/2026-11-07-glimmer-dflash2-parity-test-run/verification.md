# Verification — glimmer-dflash2-parity-test-run

## Verdict: PASS-WITH-FINDINGS

Tier 1 inline pass (no auditor, no browser pass). The change edits no source; its delivery is one operator-approved test run and the record of it.

- Implemented: PASS — the parity test ran four times on GPU 0 in the 8fe522a3 ignition container (09:13 fail, 09:43 fail with per-row readout, 09:46 debug readouts, 09:49 pass after the bound fix); the box carries three `— fixed` discoveries, one `scoped:` discovery whose target `openspec/changes/glimmer-dflash2-hidden-row-parity/` exists with a ticked task, and the operator-approval record.
- Reachable / Reused / Optimal: n/a — `git diff --stat -- . ':!openspec'` against HEAD 4f434bdb shows one file, tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py (7+/4-), which belongs to glimmer-dflash2-hidden-row-parity, not to this change.
- Tested: PASS — the cited log line 13 reads `1 passed, 1 warning in 17.36s`; line 4 carries the unchanged figures (anchor hidden 0.016424991190433502, proposals hidden 0.02768363431096077, logits 0.0029/0.0074); the two failing logs (line 188 of glimmer-dflash2-parity.log, line 182 of glimmer-dflash2-parity-h1.log) read `1 failed`, so the pass is the bound fix and not a flake.
- Faithful: PASS with finding 2 — bullet 1 ("I prefer the z-lab route") delivered by the executed parity leg; bullet 2 (run once, GPU 0, image from the landed commit) delivered with the image deviation recorded as a `— fixed` discovery (8fe522a3 differs from 4f434bdb only in two tests/reference files the test never imports); bullet 3 (operator launches, not the agent) superseded in conversation ("It's clear, go ahead and do it") and recorded on the box; bullet 4 (a failure is a `discovery:` fixed in its own change) delivered as `scoped: glimmer-dflash2-hidden-row-parity`. Reverse trace: no behavior change in this change's diff.

## Evidence audited
- [Parity run at the operator's GPU-0 window: inside the console-sanction] — openspec/changes/glimmer-dflash2-parity-test-run/tasks.md:8 — "operator-approval: Ray Dollete | GPU 0 (RTX PRO 6000) in the 8fe522a3 ignition container, run by the agent at the operator's authorization … | at=4f434bdb | 2026-11-07" (line 13 of the cited parity-fixed log reads "1 passed, 1 warning in 17.36s")

## Findings
1. low openspec/changes/glimmer-dflash2-parity-test-run/proposal.md:3 — `radflow-lint --tier-check` warns "14 file(s) / 514 line(s) across 4 top-level dir(s)": the linter diffs from the parent of the first commit touching the change dir (6ea1c9c2), so the count is the glimmer-1m-context and passkey-acceptance-width16 commits (7a095c29..4f434bdb, both verified and archived) plus the uncommitted hidden-row-parity edit; `git diff --stat -- . ':!openspec'` for this change's own work is empty. Tier 1 stands; no control was voided (the waived review covered a test run, not code). Nothing to act on in this change.
   Dimension: Tested
2. low openspec/changes/glimmer-dflash2-parity-test-run/tasks.md:8 — the operator-approval record binds `at=4f434bdb`, but the test the operator approved includes the still-uncommitted `HIDDEN_RELATIVE_L2 = .04` edit of glimmer-dflash2-hidden-row-parity; the record's log path pins the exact state (log line 1 names "HEAD 4f434bdb + glimmer-dflash2-hidden-row-parity edit"). Acted on: the sha is correct for HEAD and the log disambiguates; the close commit of the hidden-row-parity change is where the code lands, and no commit is made inside a verify pass.
   Dimension: Tested
3. low openspec/changes/glimmer-dflash2-parity-test-run/proposal.md:11 — Stated Requirements bullet 3 (the operator launches, the agent never runs ignition outside the console) was overridden by the operator in conversation for this run; the override is recorded on the box's discovery line, not as a `correction:` or a proposal note. Acted on: left as is — the operator's words are quoted on the box and the override was for this run only, not a rule change.
   Dimension: Faithful

## Advisory rules judged
- Stage decay: none — no `## Follow-ups` heading; every `discovery:` is an apply-time observation with a terminal disposition.
- Declined discoveries: none.
