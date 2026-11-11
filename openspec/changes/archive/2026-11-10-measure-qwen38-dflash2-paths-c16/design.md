# Design — measure-qwen38-dflash2-paths-c16

## Approach

One small Flare patch, then measurement. The plan has three parts.

1. Prerequisites without GPU time.
   - Console: a separate change in `~/LocalApps/vllm-env` (its own radflow store and UI rules) adds the configuration fields `defaults.speculative_method` and `preparation.force_moe_decode_a16`, the two settings `ignition/cli/config.py:24-26` already accepts and the console cannot pass.
   - Draft-folder patch (operator's direction, 2026-11-10): `ignition/recipes/qwen38/recipe.py:62` uses `Path(plan.draft_model)` when set and falls back to the pinned `snapshot_download`; `_check_draft_model` (`ignition/cli/preparation.py:127`) accepts the flag for `qwen38` only when the speculative method is dflash2, checking `config.json` and `model.safetensors`, the files `qwen38_dflash` declares. One CPU test in `tests/cli/` covers folder given, folder absent, and MTP refusal. One commit on `deploy` above the stack, a README Active-patches row, image rebuilt with `build.sh deploy`.
   - A static check of the keyword arguments each arm's recipe passes to Ignition callables, as run before the Rebase3 export.
2. Arms, each a duplicated console profile on the image built from the patch commit, at `--max-concurrency 16`, yarn 4.0 / 262144, fixed-7, the served `/ckpt` and the same draft folder (sha256 of `config.json` and `model.safetensors` recorded once): U (`qwen38`, speculative method dflash2), Q (U plus `--online-mxfp8` and `--draft-head-quantization mxfp8`), A (U plus A16 decode), M (`qwen38_dflash` with `--mixed-prefill`). Order U, Q, A, M: U is the smallest catalog and answers whether 1M context fits at 16; Q changes only the draft; A rewrites about 60 GiB of experts, so free disk is checked and unkept arm packages are removed first; M is the arm most likely to fail on memory. A failed export, a KV cache under 1,048,576 tokens, or a failed 1M passkey probe is that arm's recorded result. If U cannot hold 1M at 16, the remaining arms are re-planned with the operator before more GPU time.
3. Measurements per arm against the console-served instance, one fresh console start per trial: the fixed-7 `code_edit` probe at concurrency 1 and 8, three repetitions each (`~/.cache/dflash2-served/rebase3/probe.py` and `rate.py`, moved into `qualify/`); three 16-client runs with `~/LocalApps/llm-inference-bench/llm_decode_bench.py`; `qualify/long_context_passkey.py --base-url` at 1M; export stage timings, the lowest free GPU memory the export log reports, loaded GiB, KV tokens, graph counts, disk growth. The served entry `gb202-tp1-202611101936` gets the same repetitions as the baseline.

Decision rule, fixed before measuring: mixed prefill is worth porting only if arm M beats the best of U, A, Q on `code_edit` tok/s by more than the measured run-to-run spread; otherwise the recommendation is to retire `qwen38_dflash`. A16 and the MXFP8 draft are recommended only if tok/s or acceptance improves and the 1M passkey probe passes.

Results go to `qualify/results/dflash2-paths-c16/` and the vault tracker; the baseline profile is served again at the end unless the operator picks an arm.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.825
Caller Agreement: 0.6
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.95
- local-ai-qllm (qLLM) — ok, agreement n/a
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.4
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.7

First call (2026-11-10, before the operator directed the draft-folder patch): quorum not achieved, weight 1/2, certainty 0.45, caller agreement 0.55; response withheld, nothing incorporated. The verdict above is the single call on the revised design.

### Review Feedback

Consensus feedback recorded verbatim:
- "Do not port mixed graphs preemptively." — ACCEPTED: already the change's scope; the port is a later change decided by arm M.
- "First land the draft-folder loader patch with a CPU test and verify identical draft digests on both paths." — ACCEPTED: step 1 lands the patch; step 2 records the draft files' sha256 once for all arms.
- "Then export Path U baseline at 16 sequences and 1M context, with stop rules on OOM, insufficient KV cache (below 1,048,576 tokens), or passkey failure." — ACCEPTED: U runs first; a failed passkey probe is added to the recorded-result rules.
- "Next, test option (2) MXFP8 draft projections and head, which only changes draft graphs and is likely the cheapest export. Run option (1) BF16-decode only if throughput still misses the target, since it rewrites about 60 GiB and is structurally incompatible with mixed graphs. Cap exports at about three." — ACCEPTED in part: Q no longer depends on A and runs before it; this revision also answers the below-floor certainty and caller agreement, which came from the arm order. REJECTED in part: A still runs and the count stays four, because the operator asked for A16 decode by name and arm M is the question the study exists to answer.
- "Measure with fresh-server trials at c=1, 8, and 16, repeated three times, and act only on gains beyond run-to-run noise." — ACCEPTED: the 16-client run is repeated three times like the others.
- "Keep Path F as rollback until U passes." — ACCEPTED: every arm is a copied profile; the served entry and its profile are untouched and served again at the end.
- "Risks: late export failure from renamed keyword arguments (add signature checks and a short dry run), OOM during tuning at 16 sequences, quality drift from numeric changes, benchmark noise, and draft-file mismatch." — ACCEPTED: signature check, the export log's lowest free GPU memory, passkey and acceptance counters, three repetitions, draft digests. REJECTED: the short dry run, which would add a console export per arm; the signature check covers the failure class seen.
- "Resolve the capacity contradiction first: 1,150,976 cached tokens cannot hold 16 simultaneous 1M-token prompts." — ACCEPTED: the proposal states the target as 16 decode slots sharing a cache that holds one 1M prompt.
- "Performant: keep prompt and decode graphs separate, capture only served batch sizes, and persist autotune and kernel caches across exports." — ACCEPTED for caches: all arms use one image, so one kernel folder. REJECTED for the catalog: the recipes define it and this change does not edit them.
- "Correct: gate on greedy-output agreement with Path F, counters, passkey, and tuning peak memory at 16." — ACCEPTED: counters, passkey, memory. REJECTED: greedy-output agreement, because A16 and MXFP8 change arithmetic by design and the paths are not expected to match token for token.
- "Efficient: reuse compiled artifacts when only draft components change, and delete superseded packages before the disk rewrite." — ACCEPTED: free disk is checked and unkept arm packages removed before arm A.
