# Design — qwen38-dflash-rope-scaling

## Approach

Scope and the Stated Requirements: proposal.md. No shared component, Helios, scheduler, or workload change; the target-side scaling path is qwen38's, inherited and unchanged.

Assumed (open readings of the request, corrected by the operator at one line each):
- Assumed: the frozen 28-case speculation workload is untouched; the "> 262k long case" is a separate probe script with its own two prompts, because the workload validator caps every case at 262,144 tokens, pins the case count, and the workload is served to the 262k MTP arm as well.
- Assumed: the draft's rotary stays unscaled (absolute positions, `rope_theta` 10000) and the probe is the decision input; a past-native acceptance loss is reported to the owner as a finding and reopens that decision in its own change — this change never scales the draft.
- Assumed: the qualification package is exported through the owner's console (`/ckpt --recipe qwen38_dflash --draft-model … --rope-parameters {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}`, mixed prefill on, `--max-concurrency 8`) from an image built at the committed tip, and served with Max Context Length raised to 1,048,576; the alias form `ignition export qwen3.8-flash-dflash2-yarn4` is the same declaration and is covered by the declaration tests, not a second export.
- Assumed: the probe serves the package itself through `qualify.dflash2_served.container_command` on GPU 0 with history drafts OFF, so offered/accepted counts measure the DFlash2 draft alone (the fold's served counters mix `mtp` and `copy` sources: `~/.cache/dflash2-served/fold/summary.json` `drafts_by_source`).

1. **Declaration** (`ignition/recipes/qwen38_dflash/recipe.py:18`, `:86`): `_CHECKPOINT` stays the base `Checkpoint`; `checkpoints=(_CHECKPOINT, replace(_CHECKPOINT, alias="qwen3.8-flash-dflash2-yarn4", context=1_048_576, rope_scaling=RopeScaling(4., 262144)))` with qwen38's comment (frequency-only interpolation, no attention temperature; `ignition/recipes/qwen38/recipe.py:86`–`:90`). The base alias stays first so bare-repository resolution keeps naming it (`tests/recipes/test_declarations.py:497`). `Recipe.check_rope_scaling` (`ignition/recipes/declaration.py:301`) validates the pair at declaration time.
2. **Preparation path, unchanged**: `prepare` already passes `plan.declared_context` and `plan.checkpoint.rope_scaling` (`recipe.py:67`–`:68`); `PreparationConfig` inherits qwen38's anchor and `context ∈ (native, scaling.context]` checks (`ignition/recipes/qwen38/bootstrap.py:135`–`:143`); the target's `RotaryTables.allocate(..., scaling=config.rope_scaling)` (`qwen38_dflash/bootstrap.py:202`) interpolates frequencies; the identity hashes the scaling (`:93`). The draft's `_rotary` (`ignition/components/dflash2.py:32`–`:38`) and ring slot `positions mod 2048` (`:88`) take absolute positions up to 1,048,575 as-is.
3. **Tests**: `tests/recipes/test_declarations.py` gains `test_dflash_yarn_variant_extends_the_declared_serving_context` on the pattern of `:407`: same recipe object, draft bytes and costs equal the base alias's, `fitted(scaled, 1).declared_context == 1_048_576` with `demand_pages == 4 ×` the base plan's, `plan.to_dict()["rope_scaling"]["factor"] == 4.`, and the adapter passes `config.context == 1_048_576` and `config.rope_scaling is scaled.checkpoint.rope_scaling` into `prepare_model` (monkeypatched as `:511` does); the alias tuple at `tests/cli/test_models.py:25` and the "supported models" strings at `:113` and `tests/cli/test_commands.py:57` gain `qwen3.8-flash-dflash2-yarn4` after `qwen3.8-flash-dflash2`; `rope_scaled(resolve("qwen3.8-flash-dflash2"), {"factor": 4})` is asserted to return a checkpoint with `context == 1_048_576` and `draft_bytes` unchanged (the console path, `ignition/cli/preparation.py:63`–`:88`).
4. **Probe** (`qualify/dflash2_long_context.py`, new; CPU test in `tests/reference/test_dflash2_long_context.py`): two chat requests built from `qualify.speculative_workload._catalog(n)` with the workload's long-services question (quote timeout, retry limit, owner of record 7, the middle record, and the last record; `max_tokens` 512, the workload's `SAMPLING`), one with n sized to about 200k tokens and one to about 320k (about 66.6 tokens per record: `long-services-768` is 51,172 tokens), the past-native request sent first so no prefix is reused; each request is sent through `speculative_benchmark.stream_request` with `/metrics` read before and after (`read_metrics`, `metric_delta`, `round_metrics`, `drafts_by_source` as `dflash2_served.trial_summary` reads them), and the answer is checked by substring for the three records' `timeout_ms`, `retry_limit`, and `owner` values. Output: `summary.json` with per-request prompt tokens, retrieval pass/fail, offered/accepted draft counts, accepted-per-round, output tokens per second, and the server log. The container is `dflash2_served.container_command` widened with `history_drafts=True` (default keeps today's command byte-identical; the probe passes `False`) and `max_context_tokens=None` (the probe passes `1_048_576`). Criterion recorded with the numbers: retrieval of all three records passes on both prompts, and the past-native accepted-per-round is within 10 % of the sub-native one; a larger gap is a finding for the owner, never absorbed.
5. **Docs**: `docs/model-preparation.md:434`–`:456` gains the yarn4 alias, the `--rope-parameters` form, and that the draft's positions are unscaled; `docs/TODO.md` records the follow-up only if the probe yields a finding.
6. **Validation ladder**: CPU suites in the host venv → image from the committed tip (`FLARE_FORK_SRC=$PWD ~/LocalApps/vllm-env/build/flare-ignition/build.sh wt/dflash-s-e2vk`, kernel folder linked) → owner-launched console export with `--rope-parameters` factor 4 → the probe on GPU 0 → vault tracker Part A.

## Decisions

- `replace(_CHECKPOINT, …)` rather than converting `_CHECKPOINT` to qwen38's keyword dict — the DFlash recipe's declaration tests and the directory-resolution test read `_CHECKPOINT` as a `Checkpoint` (`tests/cli/test_models.py:331`); `replace` keeps every pinned byte count shared without restating them.
- Draft rotary unscaled — RoPE is relative and the draft attends only inside a 2048-position ring, so the dot products past 262k depend on offsets below 2048 exactly as below 262k; what changes is the absolute angle's float32 rounding (up to about 0.06 rad at position 1M on the highest-frequency pair, about 0.03 rad at 262k, identical in the reference `ignition/reference/qwen38_dflash2.py:77`–`:81`). The probe measures the effect; scaling the draft would change the component for every DFlash2 recipe and is an upstream decision.
- A probe script rather than a 29th workload case — the frozen workload's identity, 262,144 cap, and the two-arm comparison (the MTP arm serves 262k) would all break; the served-comparison spec freezes the first 24 ids and the 28-case contract.
- History drafts off in the probe — copies would dominate acceptance on a copy-from-context retrieval prompt and hide the draft's own behavior; the fold's served numbers stay the throughput reference, the probe is a draft-quality measurement.
- One export, two prompts — the sub-native prompt on the same scaled package is the control (scaled target, unscaled draft, positions below 262k), so the comparison isolates position range from the package change.

## Risks / Trade-offs

- [The 1M-context export does not fit the 96 GB GPU with mixed prefill and concurrency 8] → the plan refuses before loading with its page and byte report; the retry lowers `--max-concurrency` (the MTP production profile serves 1M on the same GPU), recorded as a discovery.
- [Past-native acceptance drops beyond 10 %] → recorded as a finding with both counts; the draft stays unscaled in this change; the tracker's Next list gains the draft-rotary decision for the owner.
- [Retrieval fails past 262k while acceptance holds] → that is the target's YaRN quality (frequency-only, never measured past 262k on this fork: Learnings §7), reported separately from the draft; the probe's summary keeps the two checks apart.
- [The probe's 320k prefill at about 15k tokens/s takes about 20 s, with 512 output tokens] → the per-request timeout is 600 s as the served probe's; the ready timeout 900 s.
- [Derivative-recipe drift] → nothing in `qwen38_dflash/bootstrap.py` changes; the rebase trap stays recorded in the fold's design.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 4/2, escalated to tier 1)
Certainty: 0.7875
Caller Agreement: 0.8
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.75
- local-ai-qllm (qLLM) — ok, agreement 0.9
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-flash (deepseek-flash) — ok, agreement 0.8

### Review Feedback

Consensus feedback recorded verbatim:
- "Add the -yarn4 checkpoint (context 1,048,576, RopeScaling factor 4 over 262144) to the qwen38_dflash tuple so the existing preparation path scales the target's YaRN rotary, honors --rope-parameters, enforces the context bound, and hashes the scaling into package identity. Keep the draft's rotary unscaled by default and decide any draft change by measurement, recording acceptance below and above 262,144 in separate buckets." — ACCEPTED: Approach items 1–2 and 4 (two prompts, two buckets); certainty 0.79 and caller agreement 0.8 sit below the 0.9 floor because the four peers split on whether to pre-scale the draft (two) or measure first (two); the measurement-first reading is the Stated Requirement ("keep the draft's `Rotary` unscaled … unless a > 262k probe says otherwise") and is kept.
- "Run target-only retrieval first so target-side failures are not blamed on the draft." — REJECTED: verification samples from the target's own distribution, so a draft cannot change which tokens are emitted; a retrieval failure is target-side by construction, and the probe already reports retrieval and acceptance as separate checks (Approach item 4); a draft-free serve would be a second export for no new information.
- "Report learned-draft and history-copy metrics separately, with copy drafts disabled for learned-draft runs. Isolate prefix-cache reuse between measured prompts or arms." — ACCEPTED: history drafts off in the probe's container and the past-native prompt sent first (Approach item 4, Decisions).
- "Guard 1M-context export memory and float32 rotary precision near 1,000,000." — ACCEPTED: Risks (plan refuses before loading; the angle rounding is quantified in Decisions and measured by the probe).
- "Keep the frozen 262k suite and its MTP A/B baseline intact, adding a separate long suite for 1M cases." — ACCEPTED: the frozen workload is untouched; the long measurement is the probe script, two prompts rather than a suite, since one export and two requests answer the question.
