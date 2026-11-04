# Design — dflash2-draft-sweep

## Approach

Scope and the Stated Requirements: proposal.md. No export, no component, runtime, scheduler, or workload change; the sweep is a qualification driver over `flare run` flags the package already honors.

1. **Arms** (`qualify/dflash2_draft_sweep.py`, new): `ARMS = [adaptive] + [fixed(n) for n in 0..7]`, each a benchmark arm dict on the shape of `speculative_workload.arms()` (`id`, `drafter="dflash2"`, `model_id`, `history_drafts`, `speculative_fixed_width`) plus the two `flare run` settings it implies: `fixed-{n}` → `mtp_draft_tokens=n, adaptive_verification=False, speculative_fixed_width=n + 1` (`fixed-0` is the no-speculation floor: width 1, decode graphs only); `adaptive` → `mtp_draft_tokens=None, adaptive_verification=True, speculative_fixed_width=None`. `history_drafts` comes from the sweep's `--history-drafts` switch (default off, the console profile's setting) and is verified by the benchmark's existing gauge check (`speculative_benchmark.py:518`–`:520`).
2. **Container** (`qualify/dflash2_served.py:48`): `container_command` gains `mtp_draft_tokens=None` (emits `--mtp-draft-tokens N` when set) and `adaptive_verification=True` (emits `--no-mtp-adaptive-verification` when false); defaults keep every existing call byte-identical (`tests/reference/test_dflash2_served.py:60` stays true).
3. **Arm injection** (`qualify/speculative_benchmark.py:498`): `arm = getattr(args, "arm_policy", None) or next(a for a in arms() if a["id"] == args.arm)`; `summarize_group(..., speculating=True)` gains the flag `run` passes as `arm.get("speculative_fixed_width") != 1`: when false, `speculative_emissions_match_raw_ids` becomes `shape_emitted == 0` and the rounds-match-launches check holds at zero, so the floor arm's transport passes on decode graphs; the rest of `run` (model-id override, `validate_arm`, the history gauge check, groups, summaries) is unchanged. The CLI's `--arm` choices stay the two comparison arms.
4. **Trials**: for each arm and each concurrency in `(1, 8)`, one fresh container named `dflash2-sweep-{arm}-c{c}` under `Server`, then `speculative_benchmark.run(SimpleNamespace(workload, arm=arm["id"], arm_policy=arm, concurrency, base_url, catalog, output=<out>/<arm>/c{c}-r0, repetition=0, timeout, model_id))` as `dflash2_served.run` does (`:120`–`:136`); each trial is reduced with `trial_summary` and the driver prints one JSON line per trial. Order: adaptive first, then fixed 0 through 7, so the reference arm lands even if a later fixed arm fails; `--repetitions` (default 1) repeats the whole order with `-r{k}` folders. `--arms` narrows the list (`fixed-4 adaptive`), `--concurrencies` the ceilings.
5. **Summary** (`summarize(output)`, sweep-owned, the served `summarize` is keyed to the comparison arms): `summary.json` keyed by arm and concurrency with `code_edit_output_tokens_per_second` (median, p99, min, max as `trial_summary` returns), `rounds_by_shape`, `drafts_by_source`, `accepted_per_round` (accepted drafts over speculative rounds, from `rounds_by_shape`), `acceptance_rate`, `transport_passed`, `requests_failed`, and `width_overrun` — true when a fixed arm's rounds include a width above N + 1; plus a printed table (arm × concurrency → code_edit median tok/s, accepted per round, acceptance rate). The `fixed_width` is recorded on each arm's row so the curve reads without the flag table.
6. **Tests** (`tests/reference/test_dflash2_draft_sweep.py`, CPU): the arm table (ids, widths, flags); `container_command` with the two keywords (default command unchanged; `mtp_draft_tokens=4, adaptive_verification=False` appends `--mtp-draft-tokens 4 --no-mtp-adaptive-verification`); `run` with fake `popen`/`stop`/`benchmark` starts one container per trial in the declared order with the arm's flags and stops it (the served driver's test pattern, `tests/reference/test_dflash2_served.py:106`); `summarize` from fabricated measurements computes accepted-per-round and flags a width overrun; `tests/reference/test_speculative_benchmark.py` gains a case that `arm_policy` replaces the id lookup and still validates against the catalog.
7. **Docs**: `docs/model-preparation.md` DFlash2 section (`:458`–`:470`) adds the sweep command after the probe's.
8. **Run**: on GPU 0 after the owner stops their instance: `.venv/bin/python -m qualify.dflash2_draft_sweep --workload ~/.cache/dflash2-served/workload2 --image local/flare-ignition:a945c147 --package …/ckpt/gb202-tp1-202611040437/model.safetensors --catalog ~/.cache/dflash2-served/yarn4/catalog-dflash2.json --model-id ckpt --publisher-keys … --trust-store … --output ~/.cache/dflash2-served/sweep`; 18 trials, about 15 minutes; results into the vault tracker (Now, Done, evidence index, a history section) and the Next item 7 ticked.

## Decisions

- Fixed width through serving flags, not re-export — the package captures every width 1 through 8 (`validate_arm` requires the complete set), and `--no-mtp-adaptive-verification` pins "the maximum width where captured" (`flare/src/cli.rs:934`); comparative numbers still come from a served package through `flare run`, as AGENTS.md requires.
- History drafts off by default — the owner's profile serves without copies, and copies would add width-independent yield that hides the draft's own curve; the switch exists so the served comparison's setting (copies on) can be reproduced.
- A draft-count-0 floor arm (peer review) — the curve needs the no-speculation throughput to show whether drafting pays at each concurrency; `--mtp-draft-tokens 0` serves through decode graphs, so the benchmark's emission cross-check (`speculative_benchmark.py:447`) takes the arm's speculation setting instead of assuming speculative counters.
- Sweep-owned arm list and summary rather than extending `speculative_workload.arms()` or `dflash2_served.summarize` — those are the frozen comparison's contract (spec `dflash2-served-comparison`: "exactly two arms"); the sweep injects its arm through `arm_policy` and reduces with the shared `trial_summary`.
- Fresh server per trial, as the comparison does — `speculative_benchmark.run` refuses a used server (`:521`–`:523`); at about 15 s per start the cost is small and the prefix cache starts empty for every arm.
- Reported, not picked — the ideal count depends on which concurrency the owner weights; the table shows both.

## Risks / Trade-offs

- [A fixed arm's realized widths fall below N + 1 when fewer proposals or pending tokens are available] → that is the runtime's rule ("otherwise width one"); the summary reports the width histogram and flags only overruns, never underruns.
- [Throughput noise between single trials (the comparison saw about ±8 % between repetitions)] → the code_edit median is over eight requests within a trial; `--repetitions` (default 1, the operator's time budget) repeats the order and the summary keeps every repetition side by side; the curve's shape, not a single pair, names the count.
- [`arm_policy` lets a caller pass an arm whose `history_drafts` disagrees with the server] → the existing gauge check fails the trial before any request, unchanged.
- [GPU 0 busy] → the driver's container fails to start and `Server` raises with the log path; nothing else is touched.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.825
Caller Agreement: 0.4
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.85
- local-ai-qllm (qLLM) — cancelled (quorum met before it responded), agreement n/a
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.8
- deepseek-flash (deepseek-flash) — cancelled (quorum met before it responded), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Sweep fixed draft counts N (about 1 to 8) using `--mtp-draft-tokens N` with `--no-mtp-adaptive-verification`, run serially on the single GPU." — ACCEPTED: Approach item 1 (counts 0 through 7; 8 is beyond the package's captured widths).
- "Include a draft-count-0 (no speculation) floor arm." — ACCEPTED: the design dropped it to keep the benchmark untouched; caller agreement 0.4 sits below the 0.9 floor on this point and on repetitions, and the design is revised instead — `fixed-0` is the floor arm and `summarize_group` takes the arm's speculation setting (Approach items 1 and 3, Decisions).
- "Keep history copies off as the primary curve to match production." — ACCEPTED: Approach item 1, Decisions.
- "Start a fresh container for every trial so prefix-cache and adaptive learned state do not carry across arms." — ACCEPTED: Approach item 4 (the benchmark refuses a used server).
- "Detect width underrun from offered vs. accepted drafts per width and log it." — ACCEPTED: `rounds_by_shape` carries offered, accepted, rounds, and emitted per width in the summary; overrun is flagged, underrun reported as the histogram (Approach item 5, Risks).
- "Use repeated trials with median and p99 aggregation to handle noise." — ACCEPTED in part: median and p99 over the eight code_edit requests are reported per trial; `--repetitions` keeps repetitions side by side; the default stays 1 because the operator set the time budget (about 15 minutes per repetition) and asked for the measurement, not a study — raised to 2 when two arms land within the ±8 % spread.
