# Handoff — rebase-forks-onto-upstream-v0-3-2

**Objective.** Move the core fork (`~/dev/flare-local-inference`) and the recipe fork (`~/dev/hyperflare-recipes`) from upstream `v0.3.2` on to upstream's untagged `main` tips, export the owner's Qwen profile at eight sequences plus Muse Glimmer and Elephant-1, release both forks as `v0.3.2-r2`, then compare with the earlier attempt.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Apply, 27 of 31 tasks checked. The second move is done (tasks 9.1 to 9.6): both forks sit on upstream `main` commits, gated, with both images rebuilt. Next: task 7.2, whose GPU 0 export waits for the owner's go. Nothing is running; both `wt/` worktrees are clean.

Budget exception: the resume prompt carries the full state inline, as the owner requires of a resume prompt.

## Resume prompt

```
/rad-apply rebase-forks-onto-upstream-v0-3-2

Read openspec/changes/rebase-forks-onto-upstream-v0-3-2/handoff.md first. Continue at task 7.2: read the console profile and the running instance's arguments, show the owner the field differences, and export on GPU 0 only on the owner's go; then 7.3, 7.4 (each on its own go) and 8.1. State on 2026-11-26:

- Bases: core upstream c915b5f0 (27 past v0.3.2), recipes upstream 7d70e230, FlashInfer 0d48af53. The release name stays v0.3.2-r2.
- Core worktree ~/dev/flare-local-inference.worktrees/rebase-forks-onto-upstream-v0-3-2, branch wt/rebase-forks-onto-upstream-v0-3-2 at 5ade4ab0 (154 commits on c915b5f0). Recipes worktree ~/dev/hyperflare-recipes.worktrees/rebase-forks-onto-upstream-v0-3-2, same branch at 5525a84c (83 commits on 7d70e230), locking core 5ade4ab0. Both unpushed; deploy in both forks has not moved. Rollback: backup/pre-rebase-20261126-* tags in both forks.
- FlashInfer pin: Raydius/flashinfer 675984cd (0d48af53 plus the fork's one commit), branch evict-planned-jits-0d48af53, pushed to the fork remote on the owner's go. Local worktree of it: ~/models/.cache/rebase-v0.3.2/trees/flashinfer-pin. The remote branch rmsnorm-fp32-weight-0aa01681 (49efc285) is no longer used; deleting it needs the owner's go.
- Images: local/flare-ignition tag 0.3.2-r2 is image 6a735b098533, built from core 5ade4ab0 and recipes 5525a84c (also tagged 5ade4ab0-5525a84c); FlashInfer's RMSNorm gain test passes inside it on GPU 1. Serving image local/hyperflare-inference tag 0.3.2-r2-5ade4ab0 prints flare 0.3.2-r2. The failing image d55fc75d-79621a35 and the unexported b14a8bce-cb9b93e5 keep their commit-pair tags.
- Gates: core adds no failing test to upstream c915b5f0's (gate/second-core-run3); recipes adds one id that needs core's completion_support.py and passes with it (gate/final-recipes, gate/second-recipes-focus). Verdicts: ~/models/.cache/rebase-v0.3.2/verdicts/9.2.md and 9.5.md.
- Unrun on this base, to watch in the exports: the rope-scaled Qwen tables through upstream's fused query and key normalization (7.2); Muse Glimmer and Elephant-1 preparation after the collective settings were removed from their bootstraps, and Elephant-1's block attention on upstream's reworked dense attention (7.3, 7.4). A passing 7.2 also closes check C1 of openspec/debug/new-stack-package-nan-logits.md.
- Profile check: of the 52 run and export flags in the profile "2026-11-25 Flare DFlash2 MXFP8 A16 - v0.3.2-r2", 48 are in the image's help; the 4 absent are the same 4 absent from the 0.3.2-r1 image. Upstream removed --collective-tuning and --collective-ladder; the profile sets neither. GPU 0 showed no serving process on 2026-11-26.
- Recorded on the owner's decision: upstream's MiMo V2 bootstrap passes buffers= to PreparedParallelDraftRound, which declares no such parameter (recipes docs/TODO.txt). The fork does not change it.
- Open for the owner: whether history-copy depths belong in upstream's GPU confidence calibration samples (core docs/TODO.md); the draft-head refusal message question from an earlier handoff has no record in verdicts.md.
- Tools: scratch ~/models/.cache/rebase-v0.3.2/ (tools/launch.sh, tools/baseline.sh with CORE_BASE and TEMPLATES, tools/compare.sh, tools/check-kwargs.py, tools/tuple3.py; baseline/T, baseline/R2, gate/, replay/second-move/, verdicts/). uv sync of the recipes lock needs GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.$HOME/dev/flare-local-inference/.git.insteadOf GIT_CONFIG_VALUE_0=https://github.com/Raydius/flare-local-inference.git until the core commit is pushed. Change a lock by editing the pinned revisions and confirm with uv lock --check in a container (uv 0.13.9); the host uv rewrites the lock.
- GPU 1 kernel test recipe: openspec/changes/debug-v0-3-2-device-error-1/handoff.md, with FlashInfer's whole tests tree.
- Constraints: push nothing without the owner's go; never commit on deploy or main; GPU 0 console instance is owner-managed; each export starts on its own go; a found defect stops all work; no kernel code on Claude Opus 5.5; never git clean; announce every long process with name, runtime and log path; watch host memory on builds.
```
