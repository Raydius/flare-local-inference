# Commit the 2026-11-04 DFlash2 draft-count sweep curve under qualify/results

Tier: 2
Type: feature

External Review: design.md — Peer Review Verdict (re-tiered to Tier 2 after rad-verify FAIL condition 3: the Tier 1 self-lowering below `--suggest-tier` had waived this control)

## Stated Requirements

- "commit-dflash2-draft-sweep-curve" (operator's rad-propose invocation after the explore session established: "Tell me more about the sweep follow through" — "1. The curve exists only in a cache directory — commit it", accepted as "Commit-the-curve as its own change, spike run separately as investigation").
- "Assumed: the committed set is summary.json plus the 18 trials' observation-stripped measurements.json, mirroring the committed dflash2-served convention, written to qualify/results/dflash2-draft-sweep/." (explore session, 2026-11-05)
- "Assumed: the strip reuses whatever mechanism produced the committed dflash2-served measurements rather than adding a new qualify module." (explore session, 2026-11-05)
- "Assumed: docs/TODO.md's sweep-adjacent record gets a pointer to the committed curve; no server rerun, no sweep-tool change, no controller change, no served-comparison re-record." (explore session, 2026-11-05)

## Blast Radius

1. grepped '"arms"' in the cache copy /home/rjdollete/.cache/dflash2-served/sweep/summary.json — top-level curve key on line 2; the 48K file is the only copy of the 18-trial curve (adaptive + fixed-0..7 x c1/c8, all transport_passed, no width_overrun) and sits in a volatile cache directory.
2. grepped '"observations":' in the cache copy /home/rjdollete/.cache/dflash2-served/sweep/adaptive/c1-r0/measurements.json — line 15442; the 18 full trial records are ~18M each because `speculative_benchmark.run` writes them with the top-level `observations` key (qualify/speculative_benchmark.py:549, :563).
3. qualify/results/dflash2-served/dflash2-adaptive/c1-r0/measurements.json:17132 — `"observations_omitted"` — the committed shape to mirror: identical keys minus `observations`, plus that note naming the full-output path; committed footprint of that run is 17 files / 11M.
4. qualify/dflash2_draft_sweep.py:67 — `summarize` produced the cache summary.json verbatim; the committed copy is that file unchanged.
5. docs/TODO.md:7 — the open `qwen38_dflash` entry cites `qualify/results/dflash2-served/summary.json`; the committed-curve pointer lands on this entry.
6. grepped 'qualify/results' across repository .py files — no code consumers; the results tree is read by humans and docs only, so no test or reader changes.
7. grepped 'sweep' across docs/ — docs/model-preparation.md:516 documents the sweep command and its `--output DIR`; docs/TODO.md has no sweep entry; only the TODO pointer is added.
