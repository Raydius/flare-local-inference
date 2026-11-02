# Compare the served DFlash2 and MTP packages on code_edit under the adaptive controller

Tier: 2
Type: feature

## Stated Requirements

- "change 3B in the roadmap" — the vault progress note §7 and `docs/TODO.md:7`: served throughput versus the MTP package on code_edit at width 8, concurrency 1–8, both packages exported with `--mixed-prefill`; decide from the memory report whether the recipe's declared graph bytes need raising for the mixed catalog.
- "Adaptive controller, as served" — selected over "Fixed width 8 exports" and "Serve-time width cap only": two exports, re-export MTP with `--mixed-prefill`, report tokens per second plus the width histogram each drafter actually ran.

Assumed: both packages serve with `--history-drafts`, as the 3a probe sessions did.
Assumed: serve sessions alternate on GPU 0 with the owner's Qwen instance stopped by the owner.
Assumed: the result is recorded under `qualify/results/` and the vault progress note, and the TODO entry drops; TP sharding of the draft stays out.

## Why

The DFlash2 recipe exists on the strength of an offline acceptance measurement (6.49 vs 5.62 emitted tokens per round at width 8 on code_edit); the owner's workload is served code generation at concurrency under 8, and no measurement yet shows whether the served DFlash2 package beats the served MTP package there. The one harness that records speculation against the native HTTP service assumes the spike's catalog and cannot run against today's packages.

## What Changes

- Re-export the MTP package (`qwen3.8-flash`) with `--mixed-prefill --max-concurrency 8` on the image that exported the DFlash mixed package, so both packages carry mixed rounds; no MTP mixed package could export before `8cbb0334`.
- `qualify/speculative_workload.py` gains an adaptive arm per drafter (DFlash2, MTP; history drafts on), concurrency 8 in its schedule, and four more frozen code_edit cases so a concurrency-8 group holds eight distinct code_edit requests.
- `qualify/speculative_benchmark.py` validates the arm against the package's real catalog (widths 1–8, batches up to 8, token capacity 128) instead of the spike's 8-row band, and reads batch counters up to 8.
- The two arms run on GPU 0 in the serving container, alternated per repetition; results land in `qualify/results/dflash2-served/` with the width histogram from the speculative counters and the mixed launch counter proving the mixed rounds were live.
- The DFlash recipe's declared graph bytes rise only if the 3a export's memory report shows the mixed catalog exceeds today's `_GRAPH_BYTES[1] + _AUXILIARY_BYTES`; otherwise the declaration test records the measured value as the evidence.
- `docs/TODO.md:7` drops the qualification clause; the vault progress note records the numbers.

## Capabilities

### New Capabilities

- `dflash2-served-comparison`: paired served measurement of the DFlash2 and MTP packages through the native HTTP service on the frozen speculation workload, with the adaptive controller's width histogram recorded.

### Modified Capabilities

- none — `qwen38-dflash2-speculation` already declares `mixed_prefill=True` and the graph-byte cost; raising the cost constant changes a number the spec states by reference to the layout, not a requirement.

## Impact

- `qualify/` tools and their CPU tests under `tests/reference/`; possibly one constant in `ignition/recipes/qwen38_dflash/recipe.py` and its declaration test. No Helios, Heliopause, contract, or package-format change.
- GPU 0 for two export-sized sessions plus the benchmark runs; roughly 100 GB of package cache for the MTP mixed entry. Exports and serving run only in the `local/flare-ignition:<sha>` container from a committed ref (vault: DFlash2 Recipe Progress §8).
- Measurement validity: both arms sample with the frozen workload's seed, so token streams differ across drafters; the comparison is on throughput and completed work, as the workload's comparison protocol already states.

## Blast Radius

1. qualify/speculative_workload.py:254 — `arms()` lists only MTP-shaped fixed-width and adaptive arms; the DFlash2 adaptive arm is added.
2. qualify/speculative_workload.py:263 — `request_groups` accepts concurrency 1, 2, 4 only.
3. qualify/speculative_workload.py:287 — `execution_plan` schedules trials over (1, 2, 4).
4. qualify/speculative_workload.py:28 — `CATEGORIES` pins code_edit=4; the validator requires exactly these counts.
5. qualify/speculative_workload.py:148 — the four code_edit case definitions; four more are added from the same source module.
6. qualify/speculative_benchmark.py:371 — `validate_arm` requires widths 1–8 and the spike's `b*w <= 8` shape set.
7. qualify/speculative_benchmark.py:431 — `speculative_shape_band` check asserts `b*w <= 8`.
8. qualify/speculative_benchmark.py:29 — `_ROUND` regex matches batch 1, 2, 4 only.
9. qualify/speculative_benchmark.py:543 — `--concurrency` choices (1, 2, 4).
9a. qualify/speculative_benchmark.py:352 — `is_idle` requires `flare_free_request_slots == 4`, the spike server's concurrency; both packages serve 8.
9b. qualify/speculative_benchmark.py:379 — `validate_arm` requires the memory report's tuning checkpoint to equal qwen38's `REVISION`; the DFlash package targets `local-inference-lab/...CSF-QAD@dc5b8ab2`, and an exported package has no memory report file (`qualify/server.py:431` reads it from a debug-serve launch).
10. tests/reference/test_speculative_workload.py:56, :72, :78 — pin concurrencies (1, 2, 4) and the arm list.
11. tests/reference/test_speculative_benchmark.py:173 — pins the spike catalog in the arm-validation fixture.
12. ignition/recipes/qwen38_dflash/recipe.py:39 and :89 — `_AUXILIARY_BYTES` and the rank-1 graph bytes the memory report decides on.
13. tests/recipes/test_declarations.py:489 — pins the DFlash graph bytes.
14. docs/TODO.md:7 — the change 3 entry.
15. grepped 'speculative_benchmark\|speculative_workload' across the repo — consumers are `qualify/dflash2_acceptance.py:21` (`build_workload`, `token_sha`, `json_bytes`; its traces keep the same case ids) and the two reference tests; no further sites.
16. grepped 'mixed-prefill' across `~/LocalApps/vllm-env/console` — one switch in `flare_params.py`, default disabled (vault: Flare Developer Learnings §19); the owner's console package stays non-mixed and is not touched.
