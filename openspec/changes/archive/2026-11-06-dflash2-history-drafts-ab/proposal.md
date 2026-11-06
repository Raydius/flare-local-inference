# Measure history drafts at fixed-7 on the served DFlash2 package

Tier: 2
Type: feature

## Stated Requirements

- "do we want history drafts?" (explore session, 2026-11-05 — answered: nobody has run on and off side by side; the A/B was proposed and is this change)
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."
- Carried from the DFlash2 changes: "Exports and serves run containerized from a committed ref (build.sh), never host venv"; "main stays the upstream mirror"; "Commit/push only when asked"; "Update vault tracker Part A when state changes"
- Carried from the operator's memory: batch every GPU step into one serving-instance stop; the count of stops is the cost, not minutes

Assumed (open readings, corrected by the operator at one line each):
- Assumed: the A/B is the sweep's `fixed-7` arm with history copies off and on, at concurrency ceilings 1 and 8, on the same image `local/flare-ignition:a945c147`, package `gb202-tp1-202611040437`, catalog and frozen workload the committed 2026-11-04 curve used, so the flag is the only variable; the off arm is rerun in the same session rather than read from the committed rows alone.
- Assumed: no re-export and no driver change — `qualify/dflash2_draft_sweep.py --history-drafts --arms fixed-7` already serves the arm with copies on and the benchmark verifies the runtime gauge against it.
- Assumed: the four trials are committed under `qualify/results/dflash2-history-drafts/` in the committed curve's shape (summary.json per run byte-identical, `observations` stripped from each measurements.json with the `observations_omitted` note), and `docs/TODO.md`'s `qwen38_dflash` entry gets one pointer line.
- Assumed: the change reports the numbers; serving with `--history-drafts` (a console profile edit and an image rebuild from `deploy`, since `e6fc2ef6` lacks `6657f7e5`) is the operator's follow-up, not this change.

## Blast Radius

1. qualify/dflash2_draft_sweep.py:33 — `arms(model_id, history_drafts=…)` sets `history_drafts` on every arm; :153 the `--history-drafts` switch; :108 `select_arms(…, args.arms)` narrows to `fixed-7`; :120 `container_command(…, history_drafts=arm["history_drafts"])` passes the flag to the container; unchanged.
2. qualify/dflash2_served.py:48–69 — `container_command` appends `--history-drafts` when asked; unchanged.
3. qualify/speculative_benchmark.py:524–526 — the benchmark raises when `flare_history_drafts_enabled` differs from the arm, so an on-arm trial that served without copies cannot be recorded; unchanged.
4. qualify/results/dflash2-draft-sweep/summary.json:1595 — committed `fixed-7` rows: c1 median 537.3 tok/s, 2.44 accepted per round, 0.555 acceptance; c8 351.7 tok/s, 3.87 per round, 0.559; the control the new rows are read against.
5. qualify/results/dflash2-draft-sweep/fixed-7/c1-r0/measurements.json:16782 — the committed trial shape (`observations` stripped, `observations_omitted` present) the new files mirror; `qualify/provenance.py:49` `write_json` is its serializer.
6. docs/TODO.md:12–13 — the `qwen38_dflash` entry's sweep pointer; the A/B pointer lands beside it.
7. helios/src/scheduler/speculation.rs:137 — `6657f7e5`'s `observe` routes copied depths to the copy curve; image `a945c147` (2026-11-03) predates it, but at `--no-mtp-adaptive-verification` with `--mtp-draft-tokens 7` the learned curve selects no width, so the trials are unaffected; the draft-source counters (`drafts_by_source` in each row) already split copies from MTP proposals.
8. openspec/specs/history-drafts/spec.md:60 — "Qwen DFlash2 round copies": the `qwen38_dflash` package admits `--history-drafts`; openspec/specs/dflash2-draft-sweep/spec.md:10 — "History copies SHALL be off unless the sweep is asked for them"; both carried, neither requirement changes.
9. openspec/changes/archive/2026-11-04-dflash2-draft-sweep/tasks.md:34 — the sweep's exact command; its host artifacts, consumed, not repo files: `~/models/.cache/flare/packages/models/ckpt/gb202-tp1-202611040437`, `~/.cache/dflash2-served/workload2`, `~/.cache/dflash2-served/yarn4/catalog-dflash2.json`, image `local/flare-ignition:a945c147` (`docker images` lists it); all present 2026-11-06.
10. openspec/changes/archive/2026-11-04-dflash2-draft-sweep/tasks.md:5 — "GPU 0 belongs to the owner's console instance — never `docker stop` it; the owner stops it before the sweep"; today the owner's instance (`local/flare-ignition:e6fc2ef6`, 96 GB) occupies it; GPU 1 has 16 GB and cannot hold the model; the four trials run in one stop of the owner's instance.
11. grepped 'dflash2-history-drafts' across qualify/ docs/ tests/ openspec/specs — no sites; the results directory is new. grepped 'qualify/results' across qualify/ helios/ ignition/ tests/ — no reader; the committed rows are records only.

## Why

The owner serves DFlash2 at fixed-7 with history copies off because the only measurement with copies on (2026-11-02 served comparison, 62–68 % of copies accepted) ran under the adaptive controller and the curve `6657f7e5` later fixed, and the only fixed-7 measurement (2026-11-04 sweep) ran with copies off. Code editing rewrites text already in the context, which is exactly what history copies propose, so the flag may be free throughput; one side-by-side run answers it.

## What Changes

- Run `qualify/dflash2_draft_sweep.py --arms fixed-7` twice on the sweep's image, package, catalog and workload: once without and once with `--history-drafts`, each at concurrency 1 and 8 — four trials, one serving container each, about 4 minutes of GPU time in one stop of the owner's instance.
- Commit the two runs under `qualify/results/dflash2-history-drafts/{off,on}/` in the committed curve's shape, and add one pointer line with the four medians to `docs/TODO.md`'s `qwen38_dflash` entry.
- Report the comparison (c1 and c8 tokens per second, accepted per round, drafts by source) in the vault tracker; the serving decision stays the owner's.
- No code, no export, no driver or benchmark change, no spec requirement change.

## Capabilities

### New Capabilities

- none — the sweep's `--history-drafts` switch is already specified in `dflash2-draft-sweep`; this change records one run of it.

### Modified Capabilities

- none

## Impact

- Repo: new `qualify/results/dflash2-history-drafts/` (4 stripped measurements.json + 2 summary.json, under 4M) and one `docs/TODO.md` line.
- Host: GPU 0 for about 4 minutes while the owner's instance is stopped; raw output under `~/.cache/dflash2-served/history-drafts/`.
- Serving: unchanged; a switch to `--history-drafts` needs a `deploy` image carrying `6657f7e5` and a console profile edit, both outside this change.
