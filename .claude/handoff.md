# Handoff — enter the radflow lane for the first upstream PRs

Stage: between changes; no live radflow change. The upstream-PR preparation so far (branches, reviews) ran outside the lifecycle and is an unverified input, not finished work. Paste the prompt below as-is.

Budget exception: three governing owner quotes and the unverified-input branch list.

## Owner's verbatim words
- "The current progress is operating outside of the radflow lifecycle and it's causing inconsistent quality. The handoff should guide the agent into the appropriate lane."
- "Can we make sure all of the code is QA'd and optimal before doing that? It will be reviewed by NVIDIA developers and I don't want to have any careless errors or oversights."
- "I want to go with fixed-7 for now. if I want maximum concurrency performance I'll just use MTP instead."

## Resume prompt
```
Context: Flare fork, worktree ~/dev/flare-local-inference.worktrees/dflash-s-e2vk, branch wt/dflash-s-e2vk at 1b46d4c5 on deploy 89683bff (patch stack on upstream 585159c9; main = 585159c9). Tree clean, in sync with origin; no live radflow change. Serving: profile "2026-11-04 Flare DFlash2 Rebase2", image e6fc2ef6, entry gb202-tp1-202611041408 (fixed-7). GPU 0 is the owner's instance — never docker stop it.

Next action: run /rad-propose for a change "upstream-first-prs" (normal lane, Tier 2 or 3 per lint; peer quorum before tasks). Outcome: the first upstream PR set opened in order — (1) the owner issue to Luke Alonso (Flash-Next DFlash2 on the shared component + YaRN; whether a mixed-prefill DFlash round is a variant of the existing graph family), which gates every Flare PR; (2) FlashInfer: update open PR #67 and open the MoE direct-SFB guard PR on local-inference-lab/flashinfer; (3) Flare: open the mixed-speculative-export PR on NVIDIA/flare-local-inference. The FlashInfer clone (~/dev/flashinfer) has no radflow root: its tasks live in this repo's change (precedent: the rebase change's task 1.2). Opening each issue/PR is an operator leg.

Unverified inputs the change must audit, not trust: Raydius/flashinfer pr/evict-planned-jits 925ac3ab and pr/moe-direct-sfb-flat-fragment 5b830a60 (both on their main 42ba2f16); Raydius/flare-local-inference pr/mixed-speculative-export 9acb56f2 (on 585159c9; worktree /tmp/review-defects). Ad hoc reviews outside any change (vault "DFlash2 Recipe Progress.md" section 18; drafts in "Upstream PR Drafts.md") inform the proposal; the change's own external review and rad-verify are authoritative and may rewrite them. Dropped with evidence: the opaque-write rule PR (its reproducers pass on pristine 585159c9). The branch pins Raydius/flashinfer@49fe5449 (one guard only); the recipe PR re-pins later.

Read first: vault "Control Plane/Flare/DFlash2 Recipe Progress.md" Part A and sections 13, 17, 18; "Upstream PR Drafts.md"; the archived rebase change's tasks 1.2 and 2.2 discoveries (FlashInfer pin).

Constraints: main stays the upstream mirror; never land on deploy; never rad-close this worktree — offer a commit instead; commit or push only when asked; upstream-bound work runs as radflow changes, never as side work; exports and GPU suites run containerized from a committed ref (build.sh) — mixed CUDA tests need the qualification heliopause (~/.cache/rebase2-gpu1/helio-site) and pytest (~/.cache/fold-gpu/site) on PYTHONPATH; CUDA tests on GPU 1; update the vault tracker Part A when state changes.
```
