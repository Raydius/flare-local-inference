# Find and remove the cause of device error 1 in Qwen cores exported on the v0.3.2 stack

Tier: 2
Type: bug

## Stated Requirements

- "Begin the debug"
- "What is likely the issue here?" (2026-11-25, on the Qwen core that failed its first request)
- "the upstream commit has already proven to work with this checkpoint with default parameters with this hardware.  I'm not interested in replicating that scenario here.  Assume it works. The goal is to run it with MY parameters" (2026-11-26)
- "for cost efficiency, tell me if you need to work on kernel code and I will switch to a different model to be compliant" (2026-11-26)
- "Is there anything else new upstream that I would be interested in?" (2026-11-26)
- "It sounds like a lot more stable upstream is available, let's move the goal post instead of doing extra work" (2026-11-26); asked whether the fixed v0.3.2 image gets one Qwen export first, the operator chose "Move now, no v0.3.2 export"

Assumed: the debug targets core `gb202-tp1-202611260714`, exported from `local/flare-ignition:0.3.2-r2` (core `d55fc75d`, recipes `79621a35`, FlashInfer `d73da5c4`), which answers its first chat request with `prepared model reported device error 1`.
Assumed: this is its own change because the operator invoked propose for it; task 7.2 of `rebase-forks-onto-upstream-v0-3-2` stays unchecked until this change names and removes the cause.
Assumed: it continues the ledger `openspec/debug/new-stack-package-nan-logits.md` (H1 refuted, H2 inconclusive, H3 to H6 open) and writes no second ledger.
Assumed: every GPU 0 run starts on its own go from the operator and goes through the console; a run the console cannot express is named as a hand run and needs the operator's explicit exception.
Assumed: the fix lands on the `wt/rebase-forks-onto-upstream-v0-3-2` branches of the fork that holds the defect; a defect in upstream or FlashInfer code is fixed in the fork's copy or pin and reported upstream by the operator's choice.
Assumed: upstream `c0c4f798` with recipes `07d7bb3` serves this checkpoint correctly with default parameters on this hardware; no run replicates that. The suspects are the operator's parameters on the fork stack: forced A16 MoE decode, online MXFP8, the MXFP8 draft head, rope scaling factor 4, 8 sequences, fixed 7 draft tokens, and the draft from a local folder.
Assumed: when the cause or its fix needs work on kernel code (CUDA C++, PTX, CuTe DSL, Triton, or the fused-arithmetic mechanisms `AGENTS.md` restricts), the work stops and the operator is told, so the operator can switch models.
Assumed: "removed" means a core exported from the operator's unchanged profile passes the served checks of the rebase change's task 7.2.

## Root Cause

silent-failure: an export completes, signs and verifies a core whose first target forward yields non-finite logits, and the first check of them is the sampling kernel at serve time, which reports code 1 per request (ignition/components/sampling.py:404). The arithmetic source of the non-finite values is not known; the tasks find it. Known so far: the fault is in the exported core, not the runtime (the same runtime serves the 2026-11-20 core); it is not a replay error (two stacks replayed independently fail alike); the two cores' prepared metadata differ only in sequence count (ledger H1).

## Why

The fork has no core on upstream `v0.3.2` that serves the operator's Qwen profile, so `v0.3.2-r2` cannot be released and the rebase change is stopped at its served-export gate.

## What Changes

- Run the open ledger's experiments in order of cost and discriminating power, each recorded with its prediction and outcome, until one cause is confirmed.
- Fix that cause on the `wt/` branches with a test that fails without the fix.
- Re-export the operator's profile and pass the rebase change's served checks.
- Close the ledger and hand the result back to `rebase-forks-onto-upstream-v0-3-2` task 7.2.

## Capabilities

### New Capabilities

None. The change restores behavior the `fork-base-move` delta of the rebase change already requires (a served export answers before a release). `.openspec.yaml` sets `skip_specs: true`; if the fix adds behavior, such as an export-time finiteness check, this change is revised to carry its delta.

### Modified Capabilities

None.

## Impact

- GPU 0 time: several five-minute serve runs of the failing core and up to four 20-minute exports, each on the operator's go.
- One or two extra local preparation images (the unpatched upstream pair; a variant per code toggle).
- The fix may touch the recipes fork, the core fork, or the FlashInfer pin.

## Blast Radius

1. ignition/components/sampling.py:404 — the non-finite check that reports code 1; the only place the fault is observed.
2. helios/src/error.rs:127 — Helios surfaces the code as `prepared model reported device error {code}`.
3. grepped `git diff --stat 07d7bb3 HEAD -- hyperflare_recipes/recipes/qwen38 hyperflare_recipes/components` in the recipes fork — 18 files, 488 insertions, 164 deletions of fork code on the Qwen3.8 Flash preparation path (`dflash2.py`, `gated_delta.py`, `parallel_draft.py`, `rotary.py`, `varlen_attention.py`, `qwen38/attention.py`, `qwen38/recipe.py`, `candidate_selector.py`).
4. grepped `git log --oneline 598c8547..0aa01681` in FlashInfer — 19 commits between the last serving pin and this one; `39aab6039` rewrites the W4A16 MoE top-k kernel and MoE and draft tuning, `063edcc0f` and `994df9c4f` the MXFP8 and BF16 GEMV kernels.
5. grepped `flare_fields` of console profile `2026-11-24-flare-dflash2-mxfp8-a16---v0-3-2-r1-1795583163871` — non-default settings on the failing path: `preparation.force_moe_decode_a16` true, `--online-mxfp8`, `--draft-head-quantization mxfp8`, `--rope-parameters` factor 4, `--max-concurrency 8`, `--no-mtp-adaptive-verification`.
6. grepped `git log ba42481a..c0c4f798 -- ignition/preparation ignition/planning.py` in core — `1bf68649` consolidates `PreparationConfig` and adds `common_preparation_kwargs`, the path the declared context and rope scaling travel.
7. openspec/debug/new-stack-package-nan-logits.md:19 — the open ledger: H1 refuted, H2 inconclusive, H3 (stored data), H4 (embedding tables), H5 (rope scaling), H6 (autotuned kernel) not run.
8. openspec/changes/archive/2026-11-26-rebase-forks-onto-upstream-v0-3-2/tasks.md:201 — task 7.2, stopped on this failure; tasks 7.3, 7.4 and 8.1 wait behind it.
9. grepped `isfinite` across `ignition/preparation` and `ignition/cli` — no finiteness check of a prepared forward's output during export.
10. grepped `docker ps` and `/v1/models` on 2026-11-25 — the failing core is the running console instance on GPU 0 (`max_concurrency` 8, `max_model_len` 1048576).

Lessons consulted: `openspec/lessons.md` entries `serve-from-the-operators-running-profile` and `a-replacement-core-keeps-the-replaced-profiles-capabilities`.
