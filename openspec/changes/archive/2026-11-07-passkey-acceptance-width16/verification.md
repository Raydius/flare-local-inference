# Verification — passkey-acceptance-width16

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-07
Tier: 1 (inline pass)

## Dimensions
- Implemented: PASS — qualify/speculative_benchmark.py:30 width class `([0-9]+)`, batch class `([1248])` unchanged; tests/reference/test_dflash2_long_context.py:119 `test_draft_counts_reads_width_16_rounds`; qualify/results/glimmer-passkey/summary.json 15 cells and 1m-depth09-max-tokens-1024.json 1 cell carry non-zero, non-null `accepted_per_round` (git diff: only that field changed, 15+1 pairs).
- Reachable: PASS — qualify/long_context_passkey.py:88 `draft_counts(delta)["accepted_per_round"]` → qualify/dflash2_long_context.py:73 `speculative_benchmark.round_metrics(delta)["shapes"]` → :401 `_ROUND.fullmatch`.
- Tested: PASS — `.venv/bin/python -m pytest tests/reference/test_dflash2_long_context.py tests/reference/test_long_context_passkey.py -q` → 18 passed; the new test fails with the class reverted to `[1-8]` (1 failed). Other `round_metrics`/`draft_counts` readers: tests/reference/test_speculative_benchmark.py test_dflash2_served.py test_dflash2_draft_sweep.py → 55 passed.
- Optimal: nothing — one character class, one test, data re-derived through the existing `metric_delta`/`draft_counts` path.
- Reused: PASS — the re-derivation used `speculative_benchmark.metric_delta` and `dflash2_long_context.draft_counts` as the proposal names; no new helper.
- Faithful: PASS — bullet 1 (acceptance "which the probe records per cell") now holds for the Glimmer width-16 shape; bullet 2 (0.0/null readout) is the fixed Root Cause. Reverse: the regex, the test, and the 16 result values trace to the Root Cause; the recorded 2.2 discovery (67/32, not 67/27) changes no code.

## Evidence audited
- [Widen `_ROUND` in `qualify/speculative_benchmark.py:30` from `width="(] — qualify/speculative_benchmark.py:30 — "_ROUND = re.compile(r'^flare_speculative_(rounds|gpu_nanoseconds|emitted_tokens|offered_drafts|accepted_drafts)_total\{batch="([1248])",width="([0-9]+)"\}$')"
- [Re-derive `accepted_per_round` for the 15 cells of `qualify/results/gl] — qualify/results/glimmer-passkey/1m-depth09-max-tokens-1024.json:1 — "      "accepted_per_round": 1.8917910447761195"

## Findings
1. low openspec/changes/passkey-acceptance-width16/proposal.md:3 — `radflow-lint --tier-check` warns "5 file(s) / 44 line(s) across 3 top-level dir(s)". The linter diffs from the parent of the first commit touching the change dir (7a095c29, the glimmer-1m-context verify commit that scaffolded this change), so its count includes that commit's one-line docs/model-preparation.md edit (`git diff --numstat 7a095c29^ -- . ':(exclude)openspec'`). This change's own diff is 4 files / 42 lines / 2 dirs (qualify, tests), inside the Tier 1 envelope; the declaration stands and no review control was voided. No action.
   Dimension: Tested
