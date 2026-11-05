# Design — upstream-first-prs

## Approach

- Order of the operator legs: owner issue first (AGENTS.md: contributions address an explicit maintainer request; a new operation contract needs the owner's prior approval), then the FlashInfer #67 refresh, then the Flare mixed-export PR. Drafting and auditing of every leg runs now; only the opening waits on the operator, and opening (3) is the operator's call after (1) is filed.
- Owner issue (text in the vault note "Upstream PR Drafts.md", filed by the operator on NVIDIA/flare-local-inference): re-scoped to the fork's deltas over upstream `d68ed8f5` — mixed-prefill DFlash2 rounds (upstream's `bootstrap.py` rejects "DFlash2 requires separate prefill and verification graphs"; ask whether a mixed round embedding a DFlash2 draw is a shape variant of the existing mixed family or a new contract needing approval), YaRN serving for the 1M-token Flash Next checkpoint (`patch/06` declaration + `patch/07 --rope-parameters`), and the served measurements (draft-width sweep, fixed-7 vs adaptive at c1/c8); names the incoming mixed-export defect PR. No duplicate recipe is offered: upstream's Flash Next DFlash2 supersedes the fork's `qwen38_dflash` recipe except for those deltas.
- FlashInfer PR #67 (~/dev/flashinfer, no radflow root; tasks live here as in the rebase change's task 1.2): `git rebase --onto origin/main 42ba2f16 pr/evict-planned-jits` (one commit `925ac3ab`, merge-tree clean on `1f8ed8a3`); `uvx ruff@0.12.8 format --check` on the four files; `tests/experimental/b12x/preparation/test_first_use_compile.py` on the rebased tip and on pristine `origin/main` (20 passed / 3 identical pre-existing failures expected); `git push --force-with-lease fork pr/evict-planned-jits:fix/evict-planned-jits` so PR #67's head ref updates in place (a PR head cannot be renamed; closing and re-opening loses the thread). No DCO line: FlashInfer's main commits carry none. The operator pastes the audited body with the PR template's checklist filled (pre-commit/ruff run, tests listed, experimental section untouched).
- MoE guard PR: dropped with evidence — upstream `3c2729e7` (#71, Mark Yong) guards both the main and swapped-FC1 flatten sites with the same `cute.rank(...) == 2` test and adds live-graph oracle tests; upstream Flare `5a1640c8` pins `1f8ed8a3` which contains it. The "pin lags code" note is also retired (upstream now pins its main tip). The fork keeps `Raydius/flashinfer@49fe5449` until the next rebase re-pins to `1f8ed8a3` + the #67 commit (Later item in the tracker).
- Flare mixed-export PR: in the existing worktree /tmp/review-defects (branch `review/defect-prs` at `9acb56f2`), `git rebase --onto upstream/main 585159c9 --exec 'git commit --amend --no-edit -s'` so both commits gain `Signed-off-by: Ray Dollete <rjdollete@gmail.com>` while keeping their `Design-Signed-Off-By` trailer (the `.githooks/commit-msg` hook re-checks it on amend); re-read `d68ed8f5`'s 28-line `decoder.py` hunk against the `sampler` flag; run `tests/recipes/test_qwen38_mixed.py`, `tests/preparation/test_execution.py`, `tests/cli/test_progress.py` on the rebased tip and confirm the new cases fail on pristine `upstream/main`; update local `pr/mixed-speculative-export` to the rebased tip and push with lease to `origin`. Base branch `main`; the body is the audited draft (the `selection.rs` retention observation stays as an out-of-scope note).
- Vault: "DFlash2 Recipe Progress.md" Part A (Now: this change; Next 2–4 and 8 rewritten for the superseded PR, the moved premise and the re-scoped issue; Later: re-pin at the next rebase, and the fork recipe versus upstream `d68ed8f5`), section 18 Dropped list; "Upstream PR Drafts.md" carries the final issue text, PR #67 body and the Flare PR body.
- Nothing in this repository's tracked tree changes except the change record; no export or GPU run is needed (the PR's GPU evidence was gathered in section 18 on `585159c9` and is cited as such; the rebase onto `0009b89d` is validated on CPU, and the PR text says so).

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/2, escalated to tier 1)
Certainty: 0.78
Caller Agreement: 0.60
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.75
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-deepseek-flash (deepseek-flash) — cancelled (quorum met first), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Order: (1) owner issue to Luke Alonso on NVIDIA/flare-local-inference, scoped to the fork-unique items and requesting approval for mixed prefill+decode DFlash2 rounds; (2) refresh FlashInfer PR #67 by rebasing onto current main 1f8ed8a3 with audited commit 925ac3ab, force-pushing with lease so the PR head updates; (3) open the Flare defect PR, rebased onto upstream main 0009b89d, with the 28-line decoder drift resolved by hand and the model-contract doc reviewed, both commits signed off for DCO, and CPU tests re-run on the rebased tip." — ACCEPTED: this is the approach; the decoder hunk is read by hand in task 2.1 even though the merge is textually clean (the peers' "silent semantic drift from clean merges" risk).
- "Drop the superseded MoE rank-guard (upstream #71 merged; the fork pins a commit containing it)." — ACCEPTED with a correction: the fork pins `49fe5449` (its own guard), upstream Flare pins `1f8ed8a3` (#71); the re-pin is a Later item.
- "Drop the duplicated DFlash2 recipe (upstream d68ed8f5) and keep YaRN and throughput data out of the defect PR." — ACCEPTED: the issue names YaRN and the measurements; the defect PR carries only its two commits.
- "Guard against sign-off failures, stale PR heads, silent semantic drift from clean merges, stale test signals, and scope creep." — ACCEPTED: sign-off via `rebase --exec`, the #67 head updated by lease push to the PR's own ref, tests re-run on both the rebased tip and pristine upstream, no third PR. Certainty 0.78 and caller agreement 0.60 sit below the 0.9 floor: the peers answered blind and proposed the same five steps; the gap is their omission of the operator-leg split and the vault records, which no peer disputed and which the tasks carry.
