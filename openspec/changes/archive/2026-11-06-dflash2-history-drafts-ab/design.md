<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Design — dflash2-history-drafts-ab

## Approach

- Two invocations of the existing driver, no code: `python -m qualify.dflash2_draft_sweep --arms fixed-7 --concurrencies 1 8` with the sweep's image `local/flare-ignition:a945c147`, package `gb202-tp1-202611040437`, catalog `~/.cache/dflash2-served/yarn4/catalog-dflash2.json` and workload `~/.cache/dflash2-served/workload2`, once to `~/.cache/dflash2-served/history-drafts/off` and once with `--history-drafts` to `.../on`. Four trials, one container each, in one stop of the owner's GPU 0 instance; the off arm is rerun so both arms share the session (same host state, same image) instead of reading the control from a run two days old.
- Each run's `summary.json` (qualify/dflash2_draft_sweep.py:67 `summarize`) carries the comparison columns per trial: code_edit tokens per second (median/p99/min/max), accepted per round, acceptance rate, and `drafts_by_source` — the counter that splits `copy` from `mtp` drafts, which is what makes copy acceptance visible without `6657f7e5`. The benchmark's gauge check (qualify/speculative_benchmark.py:524) refuses a trial whose runtime `flare_history_drafts_enabled` disagrees with the arm, so an "on" row that served off cannot be recorded.
- Commit under `qualify/results/dflash2-history-drafts/{off,on}/`: `summary.json` byte-identical, and `fixed-7/c{1,8}-r0/measurements.json` with the top-level `observations` list removed and `observations_omitted` naming the cache path, serialized by `qualify/provenance.py:49` `write_json` — the shape of the committed curve (change `commit-dflash2-draft-sweep-curve`).
- `docs/TODO.md` `qwen38_dflash` entry: one line after the sweep pointer naming the committed path and the four medians (off c1/c8, on c1/c8).
- Vault tracker Part A: the Later item becomes done with the numbers; the serving decision (`--history-drafts` on the console profile, which also needs an image from `deploy` carrying `6657f7e5`) is recorded as the owner's open call, not taken.
- Specs: `.openspec.yaml` `skip_specs: true` — `dflash2-draft-sweep` already specifies the `--history-drafts` switch and the gauge check; `history-drafts` already specifies that the Qwen DFlash2 round copies. Recording one run of them changes no requirement.

Decisions:
- Same image as the curve (`a945c147`), not the running `e6fc2ef6` nor a fresh `deploy` build: the only committed fixed-7 numbers come from `a945c147` and package `gb202-tp1-202611040437`; the fix `a945c147` lacks (`6657f7e5`) touches only the learned curve, which a `--no-mtp-adaptive-verification` fixed-7 arm never consults. One new build would add a variable and a GPU stop.
- Rerun the off arm (2 extra trials, ~2 minutes) rather than compare against the committed rows alone: the committed rows stay the reference the TODO line cites; the same-session pair is the decision basis.
- Numbers reported, not judged into a serving change: the owner chose fixed-7 on 2026-11-04 after reading a curve; the same shape here.

Risks:
- [GPU 0 not freed] → the driver's `/health` wait times out at 900 s; the task's precondition is `nvidia-smi` showing GPU 0 near 0 MiB and no flare container, checked before the first container starts; the owner's instance is never stopped by the agent.
- [On arm's copies rarely accepted on this workload] → still an answer: the TODO line and tracker record it as "no gain at fixed-7", and the flag stays off.
- [c8 regression with copies on] → reported per ceiling; the owner's 2026-11-04 rule ("maximum concurrency → MTP instead") makes c1 the deciding ceiling, recorded as such in the tracker, never absorbed.
- [Cache wiped before commit] → the strip runs in the same apply pass as the run; the 6 files sit in the work tree until the operator asks for the commit.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/2, escalated to tier 1)
Certainty: 0.85
Caller Agreement: 0.95
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.9
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.8
- local-ai-qllm (qLLM) — cancelled (quorum met before it responded), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Run a single-session A/B comparison using the existing `qualify/dflash2_draft_sweep.py` driver to execute both the OFF and ON arms on a fixed image/package (a945c147 and gb202-tp1-202611040437) at concurrencies 1 and 8. Rerunning the OFF arm in the same session eliminates environmental variance and maintains it as the sole variable against the ON arm. This requires only one owner-instance stop. Since fixed-7 never consults the learned curve, the pre-6657f7e5 bug is inert." — ACCEPTED: this is the Approach's first bullet and first decision.
- "Changes are strictly limited to a new results directory `qualify/results/dflash2-history-drafts/` (with summary.json and observations-stripped measurements.json), one pointer line in `docs/TODO.md`, and the local vault tracker state. Helios runtime code, container entrypoints, and the committed control rows in `qualify/results/dflash2-draft-sweep/` must not change." — ACCEPTED: matches proposal Blast Radius 1–6 and 11; tasks restate "touch no file under qualify/*.py, helios/, ignition/, tests/ or qualify/results/dflash2-draft-sweep/".
- "Prefix-cache pollution, image/package drift, and concurrency divergence (c1 vs c8) must be guarded against via fresh container instantiation per trial, identical frozen workload cases, and separate reporting per concurrency ceiling rather than aggregate metrics." — ACCEPTED: the driver starts one container per trial with an empty prefix cache (qualify/dflash2_draft_sweep.py:120); the task pins the image tag, package path, catalog and workload to the committed curve's; the TODO line and tracker report c1 and c8 separately, never a combined figure.
- "The existing driver should be used as-is because it is already hardened, incorporates gauge checks, and handles container lifecycles without redundant code implementation." — ACCEPTED: no driver change.
