# Verification — measure-qwen38-dflash2-paths-c16

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-10
Tier: 2

## Dimensions
- Implemented: pass — all nine ticked boxes are backed by files that exist and say what the evidence claims. Source patch read at ignition/cli/preparation.py:129-130, ignition/recipes/qwen38/recipe.py:62-69 and :138, ignition/recipes/declaration.py:222, README.md:47. Every per-arm figure in tasks.md (code_edit medians, 16-client tok/s, KV tokens, loaded GiB, graph counts, export entries, passkey results) matches `qualify/results/dflash2-paths-c16/summary.json`, the README table, and the logs under `~/.cache/dflash2-served/paths-c16/`; a child investigator recomputed all 30 code_edit and 15 16-client figures from the trial records and I re-opened the summary, the four `Allocate cache` log lines, `A/ignition.toml` and `M/passkey/summary.json`. Wording slips in tasks.md that do not change a result: `U/export-command.txt` and `U/ignition.toml` are not on disk (the command is in the console journal, the TOML in vllm-env's state folder); the commit holds 45 `measurements.json`, not 30; `baseline/trials.log` has no entry for c1-r0's start; one arm M concurrency-8 trial launched 12 mixed graphs where a discovery says at most 11; the image label equals `f83d3120`, not `git rev-parse deploy`, as task 2's discovery records. Not re-checked here: pytest counts, the kwcheck output (no saved copy), and the live `/api/server/status`.
- Reachable: pass — `ignition export` → `_prepare` (ignition/cli/serve.py:160) → `derive` → `_check_draft_model` (ignition/cli/preparation.py:108, its only call site), with the speculative method resolved from flag, `[models]` then `[defaults]` (ignition/cli/preparation.py:222) before the check; `plan.draft_model` is set at ignition/cli/preparation.py:123 and read at ignition/recipes/qwen38/recipe.py:62. `SpeculativeMethod.draft_files` has one writer (recipe.py:138) and one reader (preparation.py:130). `validate_package` and `rate` are called at qualify/dflash2_console_probe.py:61 and :63; both scripts are `python -m` entry points.
- Tested: no covering command for any of the nine boxes (no `Verify:` clause), so nothing was run. Read against the diff, tests/cli/test_draft_checkpoints.py:164 fails without the recipe edit (it asserts `draft_snapshot == folder` with no download). The two console scripts have no test (finding 5).
- Optimal: one advisory note, finding 4.
- Reused: imports hold — qualify/dflash2_console_probe.py:21 `from qualify import speculative_benchmark`, :22 `from qualify.dflash2_console_rate import rate`, :23 `from qualify.dflash2_draft_sweep import arms, select_arms, trial_row`; qualify/dflash2_console_rate.py imports only the standard library. "Follows their form" (proposal.md) is a shape claim and the shape matches ignition/recipes/qwen38_dflash/recipe.py:54-60 and ignition/recipes/muse_glimmer_nvfp4/recipe.py:21-28; no artifact or comment claims a reuse an import fails to back. Gap: `validate_package` copies part of `validate_arm` (finding 3).
- Faithful: bullet 1 partially covered — yarn 4, A16 decode, the MXFP8 draft and mixed prefill are measured (tasks 5-8), tensor parallel and history drafts are scoped out by Assumed lines, and direct BF16 feature averaging is addressed nowhere (finding 2); bullets 2 and 5 are answered by the recommendation (task 9, qualify/results/dflash2-paths-c16/README.md:79); bullet 3 by `--max-concurrency 16` in every arm (tasks 5-8); bullet 4 by task 2 (commit f83d3120). Reverse: the only changes to existing behavior are `--draft-model` accepted for `qwen38` with dflash2 (bullet 4) and the default-preserving `validate` parameter of `speculative_benchmark.run` (task 5 discovery, fixed); the MTP refusal, the pinned download and the other recipes are unchanged. The recommendation follows the design's decision rule; one sentence overstated arm A's result (finding 1).
- Environment: doctor — all workflow controls present. suite runs: 0 in 240 min, 0 deliberate re-runs, budget 3. Lint on this change: exit 1, warnings only (seven evidence lines the citation grammar could not parse; one bare name `sha256:8`). Record budget printed `changed_files 0` because the change's commits are already on the branch; `ticked_boxes 9` over `fanout_ceiling 4` opened the fan-out gate and two read-only investigators ran.

## Evidence audited
- [1. Console prerequisite, in `~/LocalApps/vllm-env` as that repository'] — qualify/results/dflash2-paths-c16/README.md:29 — "| A | U plus `force_moe_decode_a16 = true` |" (the generated TOML itself, outside this repo, reads `speculative_method = "dflash2"` at line 2 and `force_moe_decode_a16 = true` at line 5 of `~/.cache/dflash2-served/paths-c16/A/ignition.toml`)
- [2. Draft-folder patch on `deploy`: in `ignition/recipes/qwen38/recipe.] — ignition/cli/preparation.py:130 — "files = model.recipe.draft_files if method is None else method.draft_files"
- [3. Image and preflight, no GPU: `build.sh deploy` builds the image for] — qualify/dflash2_console_rate.py:23 — "def rate(measurements):"
- [4. Baseline, after the operator confirms GPU 0 may be taken for this s] — qualify/results/dflash2-paths-c16/README.md:65 — "The baseline serves 8 sequences, so 8 of its 16 clients decode and 8 wait in the queue for the whole"
- [5. Arm U: copy the Rebase3 profile as *2026-11-10 Flare DFlash2 U c16*] — qualify/results/dflash2-paths-c16/README.md:53 — "| KV tokens | 1,150,976 | 1,138,688 | 1,118,208 | 1,126,400 | 909,312 |"
- [6. Arm Q: as task 5 with profile *2026-11-10 Flare DFlash2 Q c16* and,] — qualify/results/dflash2-paths-c16/README.md:86 — "(436.7 against 395.3; 1231.2 against 1272.7). It costs 20,480 KV tokens."
- [7. Arm A: before exporting, confirm at least 90 GiB free on the `~/mod] — qualify/results/dflash2-paths-c16/README.md:87 — "- **A16 decode: not recommended.** Against arm U it is lower at concurrency 1 (511.2 against 537.1) and at"
- [8. Arm M: as task 5 with profile *2026-11-10 Flare DFlash2 M c16*, kee] — qualify/results/dflash2-paths-c16/README.md:63 — "| 1M passkey | not run | 3 of 3 | 3 of 3 | 3 of 3 | 0 of 3, HTTP 400 |"
- [9. Results and recommendation: commit under `qualify/results/dflash2-p] — qualify/results/dflash2-paths-c16/README.md:79 — "- **Retire `qwen38_dflash`; do not port mixed prefill on this evidence.** Arm M does not beat the best arm"

Tested answer for boxes 1-9: no covering command.

## Findings
1. qualify/results/dflash2-paths-c16/README.md:88 — low, suggestion. The A16 paragraph said "at concurrency 8 and on acceptance it lies inside arm U's range"; arm A's concurrency-8 acceptance rate (median 0.623, range 0.581–0.632) reaches above arm U's range (0.563–0.617). The "not recommended" conclusion stands: no arm A cell improves beyond the spread and two are lower with disjoint ranges. Fixed in this pass: the sentence now gives both ranges.
   Dimension: Faithful
2. openspec/changes/measure-qwen38-dflash2-paths-c16/proposal.md:8 — low, suggestion. Stated requirement 1 names "direct BF16 feature averaging"; no Assumed line, task, arm or README line addressed it, so the bullet is partially covered. Fixed in this pass as far as the record goes: README.md:102 now lists it under "Not measured". Whether it is worth carrying onto upstream's path remains unanswered.
   Dimension: Faithful
3. qualify/dflash2_console_probe.py:26 — suggestion. `validate_package` repeats `validate_arm`'s model-id check (qualify/speculative_benchmark.py:374-375) and its speculative-graph loop (:378-383) instead of widening `validate_arm` with a defaulted option. No artifact claims it is shared. Scoped: `reuse-validate-arm-in-console-probe`, task 1.
   Dimension: Reused
4. qualify/dflash2_console_rate.py:29 — suggestion. `rate` indexes `measurements["observations"]`, which every committed `qualify/results/dflash2-paths-c16/*/c*-r*/measurements.json` omits, so the documented command raises KeyError on the committed records and the figures reproduce only from the measuring host's cache; the docstring at :24 also names acceptance counters the function does not return. Scoped: `reuse-validate-arm-in-console-probe`, task 2.
   Dimension: Optimal
   Discovery: task 9 — "a trial's `measurements.json` is 14 MB with per-request observations (416 MB for 30); the committed files drop `observations` and carry an `observations_omitted` note"
5. qualify/dflash2_console_probe.py:26 — low. `validate_package` and `rate`, which produce every code_edit figure the recommendation rests on, have no test; no file under tests/ imports either script. Scoped: `reuse-validate-arm-in-console-probe`, task 1 (the added assertion) and task 2's done-when.
   Dimension: Tested

## Advisory rules judged
- Stage decay: none — no follow-ups heading, no task added after this file existed, and each discovery is dated by the step that surfaced it.
- Declined discoveries: n/a — every discovery is `fixed`; none is declined.

## Live UI pass
n/a — no UI surface
