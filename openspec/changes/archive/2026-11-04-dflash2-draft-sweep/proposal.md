# Sweep the DFlash2 draft count on the served yarn4 package

Tier: 2
Type: feature

## Stated Requirements

- "While we are here, can we also create a way to test the ideal number of drafts?"
- "Do not bother running it on the fold entry" — the sweep runs on the yarn4 entry `gb202-tp1-202611040437` only
- "Next change, after this one closes" — scheduled after `qwen38-dflash-rope-scaling`, now archived
- Carried from the DFlash2 changes: "Exports and serves run containerized from a committed ref (build.sh), never host venv"; "Owner launches console runs" (none needed here: the package already holds every width); "main stays the upstream mirror"; "Never land on deploy. Never rad-close this worktree — offer a commit instead"; "Commit/push only when asked"; "Update vault tracker Part A when state changes"
- "Planning only this session (propose's boundary)"

Assumed (open readings, corrected by the operator at one line each):
- Assumed: arms are fixed draft counts 0 through 7 (`flare run --mtp-draft-tokens N --no-mtp-adaptive-verification`, every round at width N + 1) plus the adaptive controller as shipped, at concurrency 1 and 8, on the frozen 28-case workload.
- Assumed: history drafts are off in every arm, matching the owner's console profile (`run:--history-drafts` empty), so the curve is the learned draft's; a `--history-drafts` switch mirrors the served comparison's setting when wanted.
- Assumed: draft count 0 (`--mtp-draft-tokens 0`, speculation disabled) is the floor arm; the benchmark's emission cross-check learns that a non-speculating arm emits through decode graphs only (peer review).
- Assumed: the sweep reports the curve; the owner reads the ideal count off it. No automatic pick.

## Blast Radius

1. flare/src/cli.rs:889–891 — `--mtp-draft-tokens N` ("maximum drafts per round; 0 disables speculation"); :934–940 `--no-mtp-adaptive-verification` ("use the maximum width where captured, otherwise width one"); together they fix every round at width N + 1; unchanged
2. ignition/recipes/declaration.py:181–187 — `widths` = 1..draft_tokens + 1 and `fixed_width = draft_tokens + 1`: width is drafts plus the pending input; unchanged
3. qualify/dflash2_served.py:48 — `container_command` gains `mtp_draft_tokens=None` and `adaptive_verification=True` keywords (defaults byte-identical, as `history_drafts` and `max_context_tokens` were added); :88 `Server`, :151 `drafts_by_source`, :162 `trial_summary` reused by the sweep; :120 `run` stays the two-arm comparison driver
4. qualify/speculative_benchmark.py:496–498 — `run` binds the arm by id from `speculative_workload.arms()`; the sweep needs its own arm dicts (`history_drafts`, `speculative_fixed_width`, `model_id`), so `run` takes an optional `arm_policy` on its args that replaces the lookup; :518–520 the history-drafts gauge check stays and is what makes the arm's history setting verified against the runtime; :372 `validate_arm` requires the complete adaptive graph set, which every arm of this sweep has (same package); :388 the shape band accepts a fixed-width run as-is; :419 `summarize_group` and its emission cross-check at :447 (`shape_emitted + count == emitted`, or `<=` with mixed launches) hold only for a speculating arm, so it gains a `speculating=True` parameter under which a draft-count-0 arm is checked as `shape_emitted == 0` with no speculative launches
5. qualify/dflash2_long_context.py:120–130 — the pattern the sweep copies for a self-served container (`container_command(..., history_drafts=False, max_context_tokens=…)` under `Server`); unchanged
6. qualify/speculative_workload.py:278–284 `arms`, :299 `execution_plan` — the frozen comparison arms and plan; untouched (the sweep carries its own arm list and runs no plan file)
7. tests/reference/test_dflash2_served.py:60 — asserts `--mtp-draft-tokens` absent from the default command; stays true; a new case covers the widened keywords
8. tests/reference/test_speculative_benchmark.py:194–216 — arm validation cases; a new case covers `arm_policy` replacing the id lookup
9. grepped 'dflash2_draft_sweep' across qualify/ tests/ docs/ — no sites: `qualify/dflash2_draft_sweep.py` and `tests/reference/test_dflash2_draft_sweep.py` are new, on the shape of qualify/dflash2_long_context.py:120 and tests/reference/test_dflash2_long_context.py:72
10. docs/model-preparation.md:458–470 — the DFlash2 section gains the sweep command beside the probe
11. grepped 'workload2|catalog-dflash2' across openspec/changes/archive/2026-11-04-qwen38-dflash-rope-scaling/tasks.md — the frozen workload under `~/.cache/dflash2-served/workload2/` and the yarn4 catalog under `~/.cache/dflash2-served/yarn4/` are the host artifacts the sweep consumes, not repo files
12. grepped 'mtp_draft_tokens|no_mtp_adaptive_verification' across helios/src/server/ — no metrics gauge exposes the configured draft count (only `flare_history_drafts_enabled`, helios/src/server/metrics.rs:657), so a fixed-width arm is verified from its realized round widths; grepped 'arms()' across qualify/ tests/ — qualify/speculative_benchmark.py:498, qualify/speculative_benchmark.py:564 and qualify/dflash2_served.py:182 only; openspec/lessons.md does not exist in this repo

## Why

The owner serves the DFlash2 package with a draft count chosen by hand (Mtp Draft Tokens 4 on the first profile, the recipe's 7 elsewhere) and has no measurement of what each count costs or yields. The adaptive runs record yield per width but never the throughput of holding one width, because a wider round costs more GPU time per round. A sweep over fixed counts on the frozen workload gives the tokens-per-second curve that names the ideal count for served code generation.

## What Changes

- `qualify/dflash2_draft_sweep.py`: for each arm (draft counts 0 through 7 fixed, plus adaptive) and each concurrency (1, 8), serve the package in its own container with the arm's flags, run the frozen workload through `speculative_benchmark.run`, and reduce with `trial_summary`; write `summary.json` keyed by arm and concurrency with code_edit tokens per second, rounds and emitted tokens per width, drafts by source, acceptance per round, and print the curve as a table; a fixed arm whose realized rounds exceed width N + 1 is reported as a finding.
- `container_command` gains `mtp_draft_tokens` and `adaptive_verification` keywords; `speculative_benchmark.run` accepts an `arm_policy` dict in place of the id lookup, and `summarize_group` takes the arm's speculation setting so a non-speculating arm's emissions are checked against decode graphs. All default to today's behavior.
- Runs once, on the yarn4 entry, 18 trials of under a minute each per repetition. No export, no component or runtime change, the frozen workload and comparison arms untouched.

## Capabilities

### New Capabilities

- `dflash2-draft-sweep` — served throughput and acceptance per fixed draft count on one DFlash2 package

### Modified Capabilities

- none

## Impact

- Code: new `qualify/dflash2_draft_sweep.py` and its test; keyword widenings in `qualify/dflash2_served.py` and `qualify/speculative_benchmark.py` with their tests; one docs paragraph.
- GPU: about 15 minutes on GPU 0 while the owner's instance is stopped; the yarn4 package and image `a945c147` as they are.
- Results: `~/.cache/dflash2-served/sweep/` and the vault tracker; the owner's console profile setting is the owner's follow-up, not this change.
