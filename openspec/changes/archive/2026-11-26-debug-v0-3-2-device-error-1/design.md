# Design — debug-v0-3-2-device-error-1

## Approach

### Names used below

- Failing core: `gb202-tp1-202611260714`, exported from `local/flare-ignition:0.3.2-r2` (core `d55fc75d`, recipes `79621a35`, FlashInfer `d73da5c4`) by the console profile copy `2026-11-24-flare-dflash2-mxfp8-a16---v0-3-2-r1-1795583163871-copy-261125_225319` ("2026-11-25 Flare DFlash2 MXFP8 A16 - v0.3.2-r2").
- Working core: `gb202-tp1-202611202125`, exported 2026-11-20 from `local/flare-ignition:7566bead` at 16 sequences, served correctly by the new runtime (ledger R1 evidence).
- Ledger: `openspec/debug/new-stack-package-nan-logits.md`. Every experiment below is one ledger line in its grammar (hypothesis, prediction both ways, Verify, Done-when, evidence, outcome); tasks.md names the line, the ledger holds the result.
- Served check: the first request of the rebase change's task 7.2, "What is the capital of France? Answer in one word.", sent to the instance the run starts. Device error 1 or HTTP 500 is a failure; "Paris" is a pass.

### Leading candidate

FlashInfer f975f6c2e, inside 598c8547..0aa01681, added a vectorized path to the native grouped RMSNorm kernel used for Qwen3.8's hyperconnection norm (`_GroupedRmsNorm` in `flashinfer/experimental/b12x/norm/hyperconnection/_cute.py`). For every hidden size divisible by 8, when its pointers are 16-byte aligned, that path loads the gain as raw words and decodes them as BF16 unless the kernel was built with `fp32_weight=True` (`_cute.py:146`, `_cute.py:199`; before the change, 598c8547 read the gain through its typed pointer and had no such flag). The plan compiler, which prepared graphs use, still builds it as `native._GroupedRmsNorm(streams, hidden, False)` with no `fp32_weight`, at the same line in 598c8547, 0aa01681 and the fork pin d73da5c4 (`_preparation.py:57`); only the convenience function passes the flag (`_cute.py:1041`). Qwen3.8 plans this operation with an FP32 gain, `1.0 + hc_norm.weight.float()`, not zero-centered (`hyperflare_recipes/recipes/qwen38/hyperconnection.py:38`, `:165`), hidden size 2560, four streams. Decoding FP32 gain words as BF16 pairs yields arbitrary values, some infinite or NaN, in the first decoder layers of every target forward, which matches every symptom: deterministic, every request, the first forward, independent of tuning and storage. This is read from code; nothing has run it.

One fact disagrees: the operator states that upstream with default parameters serves this checkpoint on this hardware. If that run used FlashInfer 0aa01681 or later, the same kernel ran in its forward, and the candidate must also explain why default parameters avoid it; the vectorized path is chosen at run time only when the state, gain and output pointers are 16-byte aligned and the state row stride is a multiple of 8 (`_cute.py:142-144`), so a storage layout difference could select the scalar path. Step 2 measures both layouts.

### Order

1. **Read (no GPU), done.** The bomb-squad findings below. Ledger H3 also already ran (`~/models/.cache/nan-debug/h3.out`: stored data verified, 98.7 GiB, and the request still failed); it is recorded as refuted.
2. **Reproduce the kernel defect on GPU 1 (seconds, no GPU 0, no export).** H10 in a CPU-and-GPU-1 container of `local/flare-ignition:0.3.2-r2`: FlashInfer's own `test_ordinary_rmsnorm_uses_affine_weight_without_offset` (`tests/experimental/b12x/norm/test_prepared_hyperconnection.py:488`, FlashInfer checkout at d73da5c4), and a direct comparison of the planned `grouped_rmsnorm` against a torch reference with Qwen's exact query (four streams, hidden 2560, FP32 gain `1 + small values`, not zero-centered, BF16 state), once with 16-byte-aligned pointers and once with an offset that forces the scalar path. GPU 1 is an RTX 5060 Ti, compute capability 12.0, the same SM120 kernels.
   - aligned case wrong and scalar case right → candidate confirmed for the kernel; continue at step 3;
   - both right → candidate refuted; continue at the fallback steps.
3. **Fix — needs a model switch.** The fix changes how a CuTe DSL kernel is constructed: the plan compiler must build `_GroupedRmsNorm` with `fp32_weight` derived from the declared `weight_dtype`. `AGENTS.md` bars this session's model, Claude Opus 5.5, from that work, and the operator asked to be told before any kernel work (2026-11-26). The reply stops here with the step-2 evidence; on another model, the fix is one commit on the fork's FlashInfer branch above d73da5c4, with the FlashInfer test from step 2 failing before it and passing after, a pin and lock update in both forks, and an image rebuilt with `build.sh`.
4. **Confirm and serve (one cold export, about 20 minutes, console).** Ledger C1: the operator's unchanged `v0.3.2-r2` profile copy is re-exported from the fixed image into a fresh kernel folder; the served checks of the rebase change's task 7.2 pass (France prompt, three-turn chat, image request, `/v1/models`); the same profile on the unfixed image is the failing half, already recorded. The ledger closes; task 7.2 resumes in the rebase change.

### Fallback steps, only if step 2 refutes the candidate

5. **Serve-time check on the failing core (about five minutes, console).** Ledger H4: `run:--embedding-table-storage` `ram` on the profile copy (`run:` fields are in the console's `serve` section, outside the export fingerprint, `console/flare_packages.py:370`, so the existing entry is served without exporting).
6. **Separate concurrency from the stack (one export, about 20 minutes, console).** H7: a copy of the failing profile with Max Num Seqs 16 and nothing else changed.
7. **Localize the first non-finite stage (a hand run).** H8: `ignition debug-serve --direct` with the failing profile's preparation flags and configuration, then with `--eager` (`docs/direct-execution.md`); a scratch-only forward hook reports the first non-finite module under `--eager` when both fail. The console cannot run `debug-serve`, so this needs the operator's explicit exception, asked once with the exact `docker run` command.
8. **One of the operator's parameters at a time back to its default (about 20 minutes each, console).** Each run returns one parameter of the failing profile to its default: `export:--rope-parameters` cleared (H5), `preparation.force_moe_decode_a16` false, `export:--online-mxfp8` false, `export:--draft-head-quantization` default, or `export:--autotune-effort` `off` (H6). A run that passes names the parameter whose path holds the defect; the goal stays a core with all of the operator's parameters.
9. **Reduce and fix.** The confirmed cause becomes a component test at the prepared shape that fails without the fix; a cause in kernel code stops the work for a model switch, as in step 3.

### Constraints that bind every step

- Each GPU 0 run starts on its own go from the operator, after the ledger line with both predictions is written. A redirect supersedes the earlier target.
- Every long process is announced before it starts: what, why, expected runtime, log path.
- A finding the ledger does not predict stops the sequence until the operator decides.
- Exports and serves go through the console; images are built locally with `build.sh` from committed refs; container inputs live under `~/models/.cache/`, never `/tmp`.
- Each new image tunes into its own kernel folder. A toggle export on the same image reuses selected kernels, which can hide a tuning fault; step 8's autotune run uses tuning off, not with a warm folder.
- This session runs on Claude Opus 5.5, which this project's `AGENTS.md` forbids from designing or implementing CUDA C++, PTX, CuTe DSL or Triton kernels, or changes to projection packing, fused arithmetic, stream overlap or scratch aliasing. As soon as a step points at kernel code, the work stops and the reply tells the operator so, with the evidence; the operator switches models before any kernel work (the operator, 2026-11-26).
- No environment variable selects behavior; a diagnostic switch never lands in the repository.
- Push nothing; commit only on the `wt/` branches; never on `deploy` or `main`.

### Decision held for the operator

The silent-failure class (an export signs a core whose forward yields non-finite logits) would be closed by a finiteness check of warmed forwards during export. It is added scope with its own design questions (which graphs, which inputs, cost) and is offered once the cause is known; it is not planned here.

### Explicit rejections

- Many blind toggle exports before localizing — rejected: each costs 20 minutes and can mislead when a tuning fault is nondeterministic; one localization pair replaces several.
- Bisecting the 19 FlashInfer commits — rejected: the reading names f975f6c2e, and step 2 tests it directly.
- An export from the unpatched upstream stack — rejected: upstream serves this checkpoint with default parameters on this hardware, and the goal is the operator's parameters (the operator, 2026-11-26).
- Clamping or masking non-finite logits in sampling — rejected: it hides the cause and weakens a numerical contract (`AGENTS.md`, "Reject weakened guarantees").

## Reframe

On 2026-11-26, after H10 was confirmed and upstream FlashInfer's c3ab1d04 was found to remove the defect, the operator chose to move both forks to upstream's `main` branches instead of exporting from a fixed `v0.3.2` image ("Move now, no v0.3.2 export"; "Untagged main, now"). The model that changed: this change assumed the fix would be proven and served on the `v0.3.2` base. The base itself is replaced, and the newer FlashInfer revision contains the fix, so the pin, image and export of steps 3 and 4 move to tasks 9.2 to 9.6 and 7.2 of `rebase-forks-onto-upstream-v0-3-2`. What stays here: the ledger, the confirmed cause, and check C1, whose passing half is that change's task 7.2. The fallback steps apply only if that export fails with the same error.

Lesson: openspec/lessons.md#check-upstream-for-the-fix-before-writing-it — fetch a pinned dependency's upstream and read its log for the defective file before designing a fix

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.8
Caller Agreement: 0.8
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.7
- local-ai-qllm (qLLM) — cancelled, agreement n/a
- deepseek-deepseek-flash (deepseek-flash) — cancelled, agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Run the Python diagnostic mode on the failing settings, eager and compiled-graph, with a finiteness probe at each layer or stage to find the first non-finite tensor and whether it appears in eager or only in graph replay." — ACCEPTED as fallback step 7, behind the direct kernel reproduction of step 2; the per-module probe runs only when eager also fails, as a scratch hook.
- "Test one factor at a time, eager, at concurrency 8 versus 16: forced 16-bit MoE decode, YaRN rope factor 4 versus 1, the multiply-and-sum bmm versus a reference matmul, fresh autotune versus cached or pinned kernel configs, draft model and MXFP8 draft head on or off" — ACCEPTED in part as fallback steps 6 and 8, one change per run; the multiply-and-sum and draft-head toggles run only if localization points at the candidate selector or the draft.
- "bisect kernel-library commits only if the probe implicates a kernel." — ACCEPTED: the reading implicated one FlashInfer commit, which step 2 tests without bisecting.
- "Confirm with multiple prompts and seeds." — ACCEPTED: a pass in any step is confirmed with the three-turn chat and a second prompt before it counts.
- "Spend one approved export on the fix plus a new export-time finiteness gate on prefill and first-decode logits." — ACCEPTED in part: the fix export is step 4; the gate is held for the operator as added scope.
- "Likely fix classes: MXFP8 zero-amax scale guards and inactive-slot masking, zero-initialized or correctly sized gated-delta/KV buffers (check for hardcoded 16), fp32 accumulation for bmm or fp16 overflow in MoE, and pinned or re-tuned batch-8 kernel configs." — ACCEPTED as candidates to check against the localized stage; none is applied without confirmation.
- "Risks: confounded changes, autotune and uninitialized-memory nondeterminism, eager-versus-graph divergence, diagnostics not exercising the same kernels, fp32 masking fp16 overflow, legitimate -inf logits tripping the check, and clamping NaNs to pass the check rather than removing the cause." — ACCEPTED: one change per run, the autotune run without a warm folder, the explicit rejection of clamping, and step 2's scalar-path control.

## Bomb-squad findings

- FlashInfer `flashinfer/experimental/b12x/norm/hyperconnection/_preparation.py:57` (d73da5c4) builds `_GroupedRmsNorm(streams, hidden, False)` without `fp32_weight`, so f975f6c2e's vectorized path decodes Qwen3.8's FP32 hyperconnection gain (`hyperflare_recipes/recipes/qwen38/hyperconnection.py:165`) as BF16 — fixed: 2 (reproduce), 3 (fix, after a model switch), 4 (confirm)
- The same plan path serves Qwen3.8's MTP `final_mixer` (`hyperflare_recipes/recipes/qwen38/mtp.py:274`), so MTP cores are exposed too — fixed: 3 (the plan-compiler fix covers every caller; step 3's FlashInfer test covers the path)
- `hyperflare_recipes/components/rms_norm.py:46` (`PreparedRMSNorm`) accepts FP32 weights and plans the same native kernel when not zero-centered without a residual; no FP32 caller found in a non-exhaustive scan of 25 callers — fixed: 3 (the fix is in the shared plan compiler, so any FP32 caller is covered)
- `hyperflare_recipes/components/expert_overlap.py:266` allocates the shared-expert MXFP8 encoded buffers with `torch.empty` where 7566bead used `torch.zeros`; the reading keeps the uninitialized values in padding rows — declined: low rank; tested in fallback step 9 only if step 2 refutes the leading candidate
- cuBLAS workspace in the capture pool (ledger H9) — declined: no defect found; graph-pool segments are exported as zero-initialized ranges and workspace contents do not cross launches
- Sibling kernels `_CombineNorm`, `_PackedCombineNorm` and `_cute.py:739` keep a default-false `fp32_weight` — declined: every current call site passes the flag
- Shape-keyed global split-K buffers `_TC_SPLITK_BUFFERS` shared across call sites and streams — declined: not on the failing B=1 prefill path; reported to the operator as a FlashInfer review item
