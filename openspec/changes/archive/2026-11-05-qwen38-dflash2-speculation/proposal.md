<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Commit the cache-only DFlash2 requalification and yarn4 probe records the speculation spec cites

Tier: 2
Type: bug

External Review: design.md — Peer Review Verdict

## Stated Requirements

- "qwen38-dflash2-speculation" (operator's rad-propose invocation, 2026-11-05; the name of the main spec `openspec/specs/qwen38-dflash2-speculation/spec.md`)
- "Spec evidence drift" — "Commit the cache-only fold, rebase2 and yarn4 measurements the spec cites and correct its tok/s figures to the committed summary" (operator's answer to the scope question, 2026-11-05)
- Assumed: the pre-fold rebase run (`~/.cache/dflash2-served/rebase/`, the source of the 329.1 / 516.5 tok/s the spec cites) is committed with the other three, since the spec's baseline is that run and not the committed `dflash2-served` comparison.
- Assumed: each served run lands as its own directory under `qualify/results/` — `dflash2-requalify/{rebase,fold,rebase2}/` and `dflash2-long-context/` — as `summary.json` byte-identical plus each trial's `measurements.json` with the raw `observations` list stripped and an `observations_omitted` note, the shape `dflash2-served` and `dflash2-draft-sweep` already use.
- Assumed: the yarn4 probe commits `summary.json` only; its two per-prompt `request.json` (0.5–0.9 M of catalog prompt text), `observation.json` (raw SSE events) and metrics snapshots stay in the cache, named by the note the TODO pointer carries.
- Assumed: the spec's "pre-fold package's 329.1 tokens per second at concurrency 1 and 516.5 at concurrency 8 (`qualify/results/dflash2-served/summary.json`…)" is corrected to cite the committed rebase run, and the fold and yarn4 requirements name their committed `summary.json`; no other requirement text changes.
- Assumed: `docs/TODO.md`'s open `qwen38_dflash` entry gains pointer lines to the committed runs; `docs/TODO.md:229`'s stale `~/benchmarks/gather-byte-ladder/` citation and the `indent=1` serialization of the 16 committed `dflash2-served` records are a different class and stay out.
- Assumed: no server rerun, no change to `qualify/dflash2_served.py`, `qualify/dflash2_long_context.py` or any other code, and the cache copies are never modified.

## Root Cause

silent-failure: the fold's spec sync wrote the requalification baseline as "329.1 tokens per second at concurrency 1 and 516.5 at concurrency 8 (`qualify/results/dflash2-served/summary.json`…)" (`openspec/specs/qwen38-dflash2-speculation/spec.md:133`, committed in `48ae6728`), but that committed file holds the 2026-11-02 served comparison whose `dflash2-adaptive` c1/c8 r0 medians are 271.3 / 512.7 (`qualify/results/dflash2-served/summary.json`, `arms.dflash2-adaptive.1.0` and `.8.0`); the 329.1 / 516.5 pair exists only in `~/.cache/dflash2-served/rebase/summary.json`, and the fold's own requalification (`~/.cache/dflash2-served/fold/summary.json`: 260.9 / 339.9 at c1, 528.2 at c8), the rebase2 fixed-7 probe (`~/.cache/dflash2-served/rebase2/summary.json`: 565.5 / 417.0) and the yarn4 past-262k probe (`~/.cache/dflash2-served/yarn4/summary.json`: 3.69 vs 2.11 accepted per round, ratio 1.75, criterion met) were never committed, so every citation that proves the spec's three measured requirements points at a path that disagrees or at a volatile cache directory, and nothing fails — no code reads `qualify/results` (grepped `qualify/results` across `*.py`: no match, confirmed by `openspec/changes/archive/2026-11-05-commit-dflash2-draft-sweep-curve/verification.md` Reachable).

## Blast Radius

1. openspec/specs/qwen38-dflash2-speculation/spec.md:133 — the Requirement "The folded package serves the measured baselines" cites 329.1 / 516.5 against the committed served summary, which holds 271.3 / 512.7; corrected to the committed rebase run.
2. openspec/specs/qwen38-dflash2-speculation/spec.md:135-137 — Scenario "Requalification" records medians "against the baselines" with no committed record; the delta names `qualify/results/dflash2-requalify/fold/summary.json`.
3. openspec/specs/qwen38-dflash2-speculation/spec.md:155,161-163 — the past-262k probe's `summary.json` is required but its only copy is `~/.cache/dflash2-served/yarn4/summary.json`; the delta names `qualify/results/dflash2-long-context/summary.json`.
4. qualify/results/dflash2-served/summary.json:1805 — `"median": 271.33…` under `arms.dflash2-adaptive.1.0`, and :2897 `"median": 512.70…` under `.8.0` (16 trials, committed `25640c8d`, 2026-11-02): the file the spec cites for a different run's numbers; untouched by this change.
5. grepped '"median"' in /home/rjdollete/.cache/dflash2-served/rebase/summary.json — line 11 `329.077…` (dflash2-adaptive c1-r0) and line 119 `516.505…` (c8-r0): the pre-fold baseline, entry gb202-tp1-202611021657; its two `dflash2-adaptive/c{1,8}-r0/measurements.json` (16.8 M / 16.1 M) carry top-level `observations` at line 15404.
6. grepped '"dflash2-adaptive"' in /home/rjdollete/.cache/dflash2-served/fold/summary.json — line 3; 3 trials (c1-r0 260.9, c1-r1 339.9, c8-r0 528.2 tok/s): the fold requalification, entry gb202-tp1-202611031627; three `measurements.json` (16.7 M / 17.4 M / 15.5 M), `observations` at line 15458 of c1-r0.
7. grepped '"fixed-7"' in /home/rjdollete/.cache/dflash2-served/rebase2/summary.json — line 3; 2 trials (c1-r0 565.5, c8-r0 417.0 tok/s): the rebase-onto-585159c9 serving probe, entry gb202-tp1-202611041408; two `measurements.json` (17.0 M / 16.8 M), `observations` at line 15431 of c1-r0.
8. grepped '"ratio"' in /home/rjdollete/.cache/dflash2-served/yarn4/summary.json — line 188 `1.7533…`; `accepted_per_round` 3.69 at line 35 (`past_native`, 319,652 tokens) and 2.11 at line 155 (`sub_native`, 199,819 tokens), criterion 0.9 met, both retrievals passed; written by `qualify/dflash2_long_context.py:137` with `write_json`.
9. qualify/results/dflash2-served/dflash2-adaptive/c1-r0/measurements.json:17132 — `"observations_omitted"`: the committed trial-record shape (17 keys, `observations` stripped, note last) the new records mirror; `qualify/results/dflash2-draft-sweep/adaptive/c1-r0/measurements.json:17466` is the same shape in `qualify/provenance.py:50` `write_json` format.
10. docs/TODO.md:7-13 — the open `qwen38_dflash` entry already points at `dflash2-served` and `dflash2-draft-sweep`; the new pointer lines land here.
11. grepped `qualify/results` across `*.py`, `*.sh`, `*.toml`, `*.yaml` — no code reads the results tree, so no test or reader changes; `git check-ignore qualify/results/x.json` exit 1 — not ignored.
12. grepped `329.1\|516.5` across `docs/` and `openspec/specs/` — only `openspec/specs/qwen38-dflash2-speculation/spec.md:133`; `docs/TODO.md` cites paths, not figures.
