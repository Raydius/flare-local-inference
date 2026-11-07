<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Count width-16 speculative rounds in the passkey probe's acceptance readout

Tier: 1 — the diff touches 4 files (the regex, one test file, two result files); sites 2, 3 and 5 are read-only readers cited for the behavior they inherit
Type: bug

External Review: Waived — Tier 1 (one regex character class, one results re-derivation from raw metrics already on disk, two test pins)

## Stated Requirements

- "Assumed: the DFlash2 drafter rides unchanged — its five layers are sliding (window 2048) and its context rings are 2048 positions (`ignition/recipes/muse_glimmer_nvfp4/draft_config.py`), so the longer window touches it only through acceptance, which the probe records per cell." (glimmer-1m-context proposal.md Stated Requirements; scaffolded by /rad-verify of glimmer-1m-context, finding 1)
- "`accepted_per_round` reads 0.0 on the first three cells and null after, while `/metrics` shows width-16 drafting live at 1M (157 of 328 drafts accepted on the depth-0.5 cell)" (glimmer-1m-context tasks.md 2.2 discovery)

## Root Cause

missing-edge-case: `_ROUND` admits only `width="([1-8])"`, so the Muse Glimmer DFlash2 shape (`MAX_WIDTH = 16`, widths 1..16) is dropped from `round_metrics(...)["shapes"]` and `draft_counts` sums only the width-1 rows — 0 accepted over 6 rounds on the 1M cells (0.0), no width-1 rounds below 1M (null) — while the raw `metrics-before/after.txt` under `~/models/.cache/glimmer-1m-passkey-raw/grid/1000000-0.5/` show a width-16 delta of 67 accepted / 130 offered over 27 rounds (qualify/speculative_benchmark.py:30)

## Blast Radius

1. qualify/speculative_benchmark.py:30 — `_ROUND` width class `[1-8]` becomes `[0-9]+` (Qwen widths 1..8 still match byte-identically; `batch` stays `[1248]`).
2. qualify/dflash2_long_context.py:73-78 — `draft_counts` sums the shapes; unchanged, now sees width 16.
3. qualify/long_context_passkey.py:88 — `accepted_per_round` per cell comes from `draft_counts(delta)`; unchanged.
4. qualify/results/glimmer-passkey/summary.json:16 — 15 cells carry `accepted_per_round` 0.0 (3) / null (12); re-derived offline from the raw `metrics-before.txt`/`metrics-after.txt` per cell (`speculative_benchmark.metric_delta` over the parsed files), no GPU run.
5. qualify/dflash2_served.py:184, qualify/speculative_benchmark.py:438,495, qualify/dflash2_draft_sweep.py:57 — other `round_metrics`/`draft_counts` readers; widths above 8 were silently absent and now appear as extra shape rows.
6. grepped '\[1-8\]' across qualify/ tests/ — only site 1; grepped 'round_metrics\|draft_counts' across qualify/ tests/ — the callers in sites 2, 3, 5 and no test pins a width above 8.
