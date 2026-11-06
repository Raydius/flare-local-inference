<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Design — qwen38-dflash2-speculation

## Approach

- `qualify/results/dflash2-requalify/rebase/summary.json`, `fold/summary.json`, `rebase2/summary.json` — byte copies of the three cache runs' summaries (rebase and fold from `qualify/dflash2_served.py:190` `summarize`; rebase2 from `qualify/dflash2_draft_sweep.py:67` `summarize`, whose trial entries carry eight extra keys — `fixed_width`, `offered_drafts`, `accepted_drafts`, `speculative_rounds`, `accepted_per_round`, `acceptance_rate`, `width_overrun`, `arm`): rebase = the pre-fold baseline the spec cites (329.1 / 516.5), fold = the Requalification scenario's record (260.9, 339.9 / 528.2), rebase2 = the fixed-7 probe on the entry now serving (565.5 / 417.0).
- `qualify/results/dflash2-requalify/<run>/<arm>/c<ceiling>-r<rep>/measurements.json` × 7 (rebase 2, fold 3, rebase2 2) — each cache trial record minus the top-level `observations` list (~17 M → ~700 K) plus `"observations_omitted": "raw per-request events and analyses are kept out of the repository; the full output is ~/.cache/dflash2-served/<run>"`, serialized in `qualify/provenance.py:50` `write_json` format; key-for-key the shape of `qualify/results/dflash2-draft-sweep/adaptive/c1-r0/measurements.json` and the served exemplar.
- `qualify/results/dflash2-long-context/summary.json` — byte copy of `~/.cache/dflash2-served/yarn4/summary.json` (`qualify/dflash2_long_context.py:137`, already `write_json`); the per-prompt `request.json`, `observation.json` and metrics snapshots stay in the cache (prompt text and raw SSE events, 1.7 M, no reader).
- `specs/qwen38-dflash2-speculation/spec.md` delta — MODIFIED "The folded package serves the measured baselines": the baseline cites `qualify/results/dflash2-requalify/rebase/summary.json` and the Requalification scenario records against `…/fold/summary.json`; MODIFIED "A scaled package serves and drafts past the native context": the Acceptance-buckets scenario names `qualify/results/dflash2-long-context/summary.json`. Figures stay 329.1 / 516.5 — they were right, the path was wrong. rad-sync carries the delta into the main spec.
- `docs/TODO.md` `qwen38_dflash` entry (`:7-13`) — two pointer lines: the requalification runs and the past-262k probe, each naming its committed `summary.json`; the probe line names the 2026-11-03 second run, since `~/.cache/dflash2-served/yarn4/summary-run1-token-echo.json` beside it is the failed first run (`client_error` SSE bound, `ratio: null`).
- No code: no reader of `qualify/results` exists (grep empty), the two probes are untouched, the cache runs are never modified. Verification is inspection — `cmp` on the four summaries, key-set equality of the 7 records against the served exemplar, no `observations` key, total under 10 M.

Decisions:
- Commit the rebase run too (beyond the operator's three named runs): it is the only source of the 329.1 / 516.5 baseline the spec requires the fold to be compared against; without it the corrected citation would still point at a cache path.
- One `dflash2-requalify/<run>/` tree rather than three top-level `dflash2-served-*` directories: the runs are the same probe on successive entries, and one parent keeps `qualify/results/` to one entry per evidence kind.
- Strip `observations` rather than commit raw (~115 M for 7 records): the served and sweep records set the precedent; the note names the cache path.
- Byte-exact summaries: they are the citation targets; re-serializing would make committed and cache copies differ for no gain.
- Spec figures corrected by path, not by value: the committed `dflash2-served` 271.3 / 512.7 is the 2026-11-02 comparison, a different entry; the fold's baseline is the rebase entry.

Risks:
- [Cache wiped before the data lands] → task 1.1 and 1.2 copy first; the operator's commit is the only later step.
- [Requalification read as a verdict] → every summary row carries `passed: false` from the workload's semantic manual-review cases while `transport_passed` is true, as the committed served record does; the TODO lines name them recorded runs, not passes.
- [rebase2 is a fixed-7 arm, not dflash2-adaptive] → the spec's Requalification scenario names the adaptive trials; rebase2 is recorded as the serving entry's probe and the spec text does not claim it as a requalification.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 4/3, escalated to tier 1)
Certainty: 0.9
Caller Agreement: 0.95
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.95
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.85

### Review Feedback

Consensus feedback recorded verbatim:
- "Commit four new result trees under `qualify/results/` corresponding to the cache runs … copy `summary.json` byte-identical, strip the heavy `observations` arrays from each `measurements.json` and replace them with an `observations_omitted` note pointing to the cache path, serialized using the repository's `write_json` format (~700 KB per file). For YaRN, commit only `summary.json` and leave the bulky raw prompt/SSE files (`request.json`, `observation.json`) in the cache." — ACCEPTED: this is the Approach (tasks 1.1 and 1.2); the peers' four sibling directories became `dflash2-requalify/{rebase,fold,rebase2}/` plus `dflash2-long-context/`, one parent per evidence kind.
- "Update `openspec/specs/qwen38-dflash2-speculation/spec.md` to correct the baseline, requalification, and YaRN paths while preserving the correct figures (e.g., 329.1 / 516.5 for the rebase baseline), and update `docs/TODO.md` pointer lines." — ACCEPTED: the spec delta corrects paths and keeps figures; task 1.3 adds the TODO lines.
- "Code, tests, and the existing `qualify/results/dflash2-served/` tree must remain untouched. Risks include data loss if the cache is deleted before copy, repo bloat from accidentally committing unstripped measurements or raw YaRN logs, and mismatched or stale citations in the spec." — ACCEPTED: the task constraints forbid touching those paths; task 1.1's verify fails on any surviving `observations` key or a total over 10 M; the delta's paths are the committed ones task 1.1 and 1.2 create.
- "An inline Python pass or simple one-off script using existing `write_json` rules is fully sufficient without requiring a persistent committed script." — ACCEPTED: no new module.

## Bomb-squad findings

- Root cause chain: the 329.1 / 516.5 pair originates only in `openspec/changes/archive/2026-11-02-rebase-dflash-onto-upstream-6fa01a49/tasks.md:30` (cache path); the fold's delta `openspec/changes/archive/2026-11-03-fold-dflash-draft-onto-dflash2/specs/qwen38-dflash2-speculation/spec.md:45` attached those figures to the committed comparison's path (the vault's evidence index has both rows: `qualify/results/dflash2-served/summary.json` for the comparison, `~/.cache/dflash2-served/rebase/` for the baseline); `48ae6728` synced it verbatim. radflow-lint skips `~`/absolute paths (`~/.local/bin/radflow-lint:3783-3784`) and reads only tasks.md evidence and the Blast Radius, never spec deltas (`:3856-3858`); no repo rule names where qualification evidence lives (`AGENTS.md` silent; `docs/model-preparation.md:512,516` free-form `--output DIR`).
- This change: all four cache runs clean — no NaN/Infinity, every summary trial has its `measurements.json` (2/3/2), summary medians equal each record's `measurements.code_edit.observed_group_output_tokens_per_second.median`, 7/7 key sets identical to the served exemplar after dropping `observations`. Fixed from the hunt: Approach bullet 1 now attributes `rebase2/summary.json` to `qualify/dflash2_draft_sweep.py:67` (sweep-shaped entries, eight extra keys); the TODO probe line names the second yarn4 run so `summary-run1-token-echo.json` (failed, `ratio: null`) is not taken as the record.
- Out of scope, same class (left for their own changes): `openspec/specs/qwen38-dflash2-speculation/spec.md:122` cites the 0.1053 near-tie measurement with no committed record (`~/.cache/fold-gpu/{repeatability,margin}.log` only); `spec.md:29` cites acceptance figures that match `qualify/results/dflash2-acceptance/*.json` but names no path; `docs/TODO.md:229` `~/benchmarks/gather-byte-ladder/` absent on this host; vault evidence index row 104 still calls the draft-count sweep cache-only. Process recommendations (lint the spec delta's repo paths; an AGENTS.md rule that spec-cited figures live under `qualify/results/<run>/summary.json`) are recorded here, not acted on.
