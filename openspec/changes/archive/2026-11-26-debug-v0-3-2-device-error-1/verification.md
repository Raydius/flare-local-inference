# Verification — debug-v0-3-2-device-error-1

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-26
Tier: 2
Basis: full
Depth: full

## Dimensions
- Implemented: every checked box is backed by the ledger, the GPU 1 logs, the FlashInfer fix commit and the served-checks record; one record defect (ledger H10 left unchecked although task 2 claimed it checked) was fixed in this audit (Finding 1). No QA goal: the change names no release-QA item.
- Reachable: the fix sits on the only prepared-graph construction site of the kernel, `_compile_hyperconnection` in FlashInfer `_preparation.py:57` (the other constructor, `_cute.py:1041`, already passed `fp32_weight`); the served path reaches it through the image whose FlashInfer 675984cd contains upstream c3ab1d046 (`git merge-base --is-ancestor` confirmed), proven by the served export `gb202-tp1-202611261556`. No code in this repository was added, so nothing is orphaned.
- Tested: no checked box carries a `Verify:` clause, so each records `no covering command`. The tests the change relied on were read, not rerun: the FlashInfer test fails 8 of 32 on d73da5c4 and passes 32 of 32 with the fix (`task3-test-unfixed.log`, `task3-test-fixed.log`), so it discriminates the change; the GPU 1 script shows 126/71680 non-finite outputs without the fix and 0 with it.
- Optimal: no `discovery:` line implicates a quality issue; nothing to report.
- Reused: the fix is upstream's own commit, `git cherry-pick -x` of c3ab1d046 (trailer "(cherry picked from commit c3ab1d046a45431338e8ca9f11f549878e82d514)" on 49efc285), and the shipped base takes it from upstream FlashInfer 0d48af53/675984cd rather than from the fork commit; no artifact claims a shared module, so no import list applies. Reuse claim holds.
- Faithful: every Stated Requirements bullet is covered — "Begin the debug" and "What is likely the issue here?" by tasks 1 to 2 and ledger H10 (confirmed); "run it with MY parameters" by task 5 (the unchanged profile served "Paris", the chat, the image request, and `max_concurrency` 8 / `max_model_len` 1048576); the kernel-code bullet by task 3, done as an upstream cherry-pick rather than written in-session; the upstream question and "move the goal post" by design.md `## Reframe` and the task 4 and 5 `replan:` lines. No externally observable behavior of this repository changed, so the reverse trace is empty.
- Reviewed: no high or medium findings; 2 low; Peer coverage: partial (the code diff returned; the prose records were reviewed by the auditor only).
- Environment: doctor exit 1 — omp trailer and tool-call hooks absent in this checkout and 21 other worktrees (WARN), engine-stamp offline NOTE; changed_files 0; suite runs: 0 in 240 min, 0 deliberate re-runs, budget 3; verify runs: 0 ran, 0 reused, 0 covered.

## Evidence audited
- [1. Record in the ledger: H3 refuted from `~/models/.cache/nan-debug/h3] — openspec/debug/new-stack-package-nan-logits.md:29 — "  outcome: refuted" — no covering command
- [2. Reproduce H10 on GPU 1: in a container of] — openspec/debug/new-stack-package-nan-logits.md:38 — "  outcome: confirmed" — no covering command
- [3. Fix the plan compiler on the fork's FlashInfer, on a branch above d] — openspec/lessons.md:54 — "- Resolution: the fix commit is a cherry-pick of `c3ab1d04`" — no covering command; also read outside the repo: task3-test-fixed.log line 127 "32 passed, 99 warnings in 10.98s"
- [5. On the operator's go, export the unchanged profile "2026-11-25 Flar] — openspec/debug/new-stack-package-nan-logits.md:39 — "- [x] C1 Confirmed theory explains every symptom above and a predicted minimal intervention flips R1." — no covering command

## Findings
1. suggestion (fixed) — openspec/debug/new-stack-package-nan-logits.md:36: ledger H10 carried its evidence and `outcome: confirmed` but its box read `- [ ]`, while task 2's evidence states "ledger H10 ticked confirmed". The ledger grammar checks a line once its Verify ran. Fixed in this audit by checking the H10 box; the line's text is otherwise unchanged.
   Dimension: Implemented

## Code review
Review standard: 001ce8ddb3cf5aa072e28e2cc5bbab28ff39cdc0
Peer part: path=flashinfer/experimental/b12x/norm/hyperconnection/_preparation.py part=1/1 blob=c5b80510c50c0486aeee88c436ebb9d2e4dc9735 status=returned reason=FlashInfer fork commit 49efc285
Peer part: path=flashinfer/experimental/b12x/norm/hyperconnection/_cute.py part=1/1 blob=58cda1d27b1b348bd834f580b97ec5a9bf368625 status=returned reason=FlashInfer fork commit 49efc285
Peer part: path=tests/experimental/b12x/norm/test_prepared_hyperconnection.py part=1/1 blob=54cbf261de932c18f89df679f08185ed171d7f02 status=returned reason=FlashInfer fork commit 49efc285
Peer part: path=openspec/lessons.md part=1/1 blob=26ff1c64d14a0a1b93a09f989dd97fad464be812 status=returned reason=this change's hunk only; the other three entries belong to rebase-forks-onto-upstream-v0-3-2
Peer part: path=openspec/changes/debug-v0-3-2-device-error-1/proposal.md part=1/1 blob=486fffe5f26f6d500e06800570b3e45d97411edb status=missing reason=prose record not sent; auditor review only
Peer part: path=openspec/changes/debug-v0-3-2-device-error-1/design.md part=1/1 blob=b71bed1e3fc52ec98559468131b2d5d32e98792e status=missing reason=prose record not sent; auditor review only
Peer part: path=openspec/debug/new-stack-package-nan-logits.md part=1/1 blob=adacbeb56c48b2f2efdaf7040ddb8911087c5b84 status=missing reason=prose ledger not sent; auditor review only
Peer coverage: partial

Call 1 (FlashInfer diff plus the lessons hunk):
Quorum: achieved (weight 4/3, tier 2)
Certainty: 0.75
Caller Agreement: n/a (not returned)
Arbiter: consensus "no security issues; the fix is correct; the version bump is needed to invalidate cached kernels"
Sources:
- claude-haiku-5-5 — ok, agreement 0.6
- qLLM — ok, agreement 0.8
- gemini-flash-lite-latest — ok, agreement 0.6
- deepseek-flash — ok, agreement 0.2
- claude-sonnet-5-5 — ok, agreement 0.7

Low findings:
- flashinfer/experimental/b12x/norm/hyperconnection/_preparation.py:58 — `fp32_weight` is true only for float32; a float16 gain would still take the BF16 decode. PLAUSIBLE, no float16 caller today; low.
- openspec/debug/new-stack-package-nan-logits.md:40 — C1's served flip also changes the core and recipes base, so it is not a one-variable intervention; the ledger states this and names H10's GPU 1 test as the isolated flip, which follows the operator's "Move now, no v0.3.2 export". Low, documentation.

NOT A DEFECT, checked:
- Another `_GroupedRmsNorm(` site omitting `fp32_weight` (peer item) — the only other constructor passes it: flashinfer/experimental/b12x/norm/hyperconnection/_cute.py:1041.
- The added hidden-2560 case fails on the parent (peer item) — `task3-test-unfixed.log` shows 8 failed of 32, the FP32-gain padding-0 cases including hidden 2560.

## Advisory rules judged
- Stage decay: no — no `## Follow-ups` heading, no task added after verification.md, and the discoveries were written when found.
- Declined discoveries: no `declined:` lines in tasks.md; design.md's bomb-squad declines (the `torch.empty` buffer, cuBLAS workspace, sibling kernels, split-K buffers) are adequate, since the confirmed cause and the served export leave none on the failing path.
- Split-offs: no — the live sibling `audit-v0-3-2-rebase` quotes its own operator requests ("Systematically recheck all aspects of the job …").

## Live UI pass
n/a — no UI surface
