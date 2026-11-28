# Handoff — rework-pr212-config-driven-yarn

Rework upstream PR #212 (NVIDIA/hyperflare-inference) to yatesdr's review: YaRN selected by the checkpoint config, transformers reference with the attention factor, plus a linked hyperflare-recipes PR; and fix the fork's serving package in the same GPU-0 window.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

## Stopping point

Stage: rad-apply complete; every box is checked. Published 2026-11-27: #212 force-pushed to `3c73cd1d`, recipes PR local-inference-lab/hyperflare-recipes#68 opened, reply posted. Next: `/rad-simplify`, then `/rad-verify`.

The GPU-0 window is closed. The console serves `2026-11-27 Flare v0.3.2-r2 YaRN attention factor` (id `…copy-261127_221132`, package `gb202-tp1-202611280643`).

## State to know

- PR heads, pushed: core `pr/rope-scaling` = `3c73cd1d` (worktree `~/dev/flare-local-inference.worktrees/pr212-core`, backup tag `backup/pr-rope-scaling-15d8b0b0`); recipes `pr/qwen38-config-yarn` = `671b6609` (worktree `~/dev/hyperflare-recipes.worktrees/pr212-recipes`).
- Branch `local/pr212-evidence` = `ec62b550` locks core 3c73cd1d for the test image; never push it.
- Recipes fork `deploy` has unpushed commits `87c08409` (attention factor, probe tasks) and `96e21548` (results under `qualify/results/yarn-config/`).
- Test profiles `2026-11-27 PR212 YaRN config MTP c8` and `2026-11-27 PR212 native MTP c8` remain in the console.
- This change directory is untracked in the core repo; nothing committed for it yet.
