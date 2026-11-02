# Design — qwen38-dflash-served-comparison

## Approach

Motivation and scope: see proposal.md. The paired HTTP harness from the acceptance spike (`qualify/speculative_workload.py`, `qualify/speculative_benchmark.py`) already records frozen requests, raw token streams, idle-fenced counters, and per-group throughput against the native service; this change points it at exported packages instead of a debug-serve launch and widens it to the owner's regime.

1. **Arms are drafters, not fixed widths.** `arms()` (`qualify/speculative_workload.py:254`) becomes two adaptive arms, `mtp-adaptive` and `dflash2-adaptive`, each with `history_drafts=True` and the package's model id (`nvidia/Qwen3.8-Flash-Next-FP4-Local` for MTP, `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` for DFlash2). The spike's `no-mtp` and `mtp-fixed-N` arms go: no fixed-width package exists for this comparison, and the decision recorded in the proposal rejects fixed-width exports. `execution_plan` keeps alternating arm order per repetition and schedules concurrency (1, 2, 4, 8).
2. **Eight code_edit cases.** `CATEGORIES` pins `code_edit=8`; four edits are added to the list at `qualify/speculative_workload.py:141` over the same `_code()` source (rename a parameter across validators, add a logging line per validator, add type hints, extract the shared range check), each with `{"kind": "code"}` semantics and budget 1024 like the existing four. `request_groups` and the `--concurrency` choice accept 8; the validator's count message says 28 cases. The workload is regenerated with `generate` (native frontend + snapshot) at run time, as the spike did; `qualify/dflash2_acceptance.py:21` keeps reading the same first 24 case ids.
3. **Catalog facts come from `flare inspect`, not a memory report.** `speculative_benchmark.py` takes `--catalog PATH`, the JSON of `flare inspect PACKAGE --json --graphs --unverified`, in place of `--memory-report`. `validate_arm` checks `model.model_id` against the arm, `deployments[0].max_active == 8`, that `ranks[0].execution_graphs` holds `speculative{b}x{w}` for b in (1, 2, 4, 8) and w in 1..8 with `b*w <= 128`, and at least one `kind == "mixed"` graph with `draft_tokens` set (the mixed rounds the proposal requires). The idle check reads `flare_free_request_slots == max_active`; page and snapshot conservation compare each group's after-metrics to the trial's initial free-plus-cached totals instead of memory-report capacities. `_ROUND` matches batch 1, 2, 4, 8; `speculative_shape_band` bounds `b*w` by the catalog's largest speculative row count.
4. **Driver for the 16 trials.** `qualify/dflash2_served.py` runs the execution plan: for each repetition, concurrency, and arm it starts the package in the serving container (`flare run PACKAGE --gpus 0 --history-drafts --max-concurrency 8 --host 0.0.0.0 --port 8111`, the 3a probe's flags), waits for `/health`, runs `speculative_benchmark.run` with the trial's catalog, stops the container, and writes `qualify/results/dflash2-served/<arm>/c<N>-r<R>/measurements.json`. A fresh server per trial is the workload's cache protocol. Container flags are the vault's (DFlash2 Recipe Progress §8), parameterized by image tag and package path; the script never reads an environment variable to choose behavior.
5. **Summary.** The driver's `summarize` writes `qualify/results/dflash2-served/summary.json`: per arm and concurrency, code_edit `observed_group_output_tokens_per_second`, the width histogram from `flare_speculative_rounds_total{batch,width}` and emitted tokens per round, `flare_speculative_drafts_total` by source, and `flare_sampling_graph_launches_total{kind="mixed"}` so the mixed rounds are proven live. The vault progress note and `docs/TODO.md` are updated from it.
6. **MTP re-export.** Operator leg: `ignition export qwen3.8-flash --gpus 0 --mixed-prefill --max-concurrency 8` on image `local/flare-ignition:f829c06f` with the same kernel folder the DFlash export used; the existing `ckpt` entries (concurrency 16, 1 M context, no mixed graphs) are not comparable. The DFlash arm serves `gb202-tp1-202610312330`.
7. **Graph bytes.** `flare inspect ENTRY --json --memory-map` on the 3a DFlash entry gives the captured graph allocation; it is compared with `_GRAPH_BYTES[1] + _AUXILIARY_BYTES` (`ignition/recipes/qwen38_dflash/recipe.py:39`, `:89`). The constant and `tests/recipes/test_declarations.py:489` change only if the measured bytes exceed the declaration; either way the task records both numbers.

Constraints carried from AGENTS.md: no Helios, Heliopause, contract, package-format, or recipe-execution change; `qualify/` tools stay client-side and never start a model from Python; measurements compare `flare run` packages, never debug-serve; no environment variable selects behavior; exports and serving run in the container from a committed ref; the owner stops and restarts the GPU 0 instance, never an agent.

Validation: CPU tests in `tests/reference/test_speculative_workload.py` and `test_speculative_benchmark.py` are updated for the new arms, concurrency 8, 28 cases, and catalog validation (a fixture built from the real inspect JSON of the 3a entry); the driver's trial planning and summary are unit-tested on fabricated measurements; the served runs are the operator legs.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.8
Caller Agreement: 0.6
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.7
- local-ai-qllm (qLLM) — cancelled, agreement n/a (quorum met first)
- deepseek-flash (deepseek-flash) — cancelled, agreement n/a (quorum met first)

### Review Feedback

Consensus feedback recorded verbatim:
- "Compare the two served packages as shipped, with the adaptive width controller on, not fixed-width arms. Re-export the MTP package (qwen38) with mixed prefill+decode rounds and history drafts on, to match the existing DFlash2 package." — ACCEPTED: Approach steps 1 and 6; history drafts are a serve-time flag, so both arms serve with `--history-drafts`.
- "Read the package's limits from `flare inspect PACKAGE --json --graphs` ... instead of the memory report, and derive the idle slot count, catalog checks, and widths from that output." — ACCEPTED: Approach step 3.
- "Report output tokens/s and the realized (batch,width) round histogram per drafter, and don't pool concurrency levels." — ACCEPTED: the summary keys on arm and concurrency; no pooled mean is written.
- "Run strictly sequentially on the single GPU with a fresh server per trial and cleanup on exit, keep the before/after counter fences and conservation checks, measure shipped controller defaults without tuning, and don't treat the offline 6.49 vs 5.62 result as confirmation." — ACCEPTED: the driver stops its container in a `finally`; no `--mtp-*` tuning flag is passed; the summary reports served numbers only.
- "Screen with one repetition at concurrency 1 and 8 before expanding." — ACCEPTED: the driver takes `--trials` so the operator leg runs the two-trial screen first, then the full plan.
- Caller agreement 0.6 is below the 0.9 floor and certainty 0.8 below 0.9 — RECORDED: the two responding peers agree with each other on every point above and differ from the caller only in listing a driver script and the graph-bytes check, which they did not address; no approach point was rejected.
