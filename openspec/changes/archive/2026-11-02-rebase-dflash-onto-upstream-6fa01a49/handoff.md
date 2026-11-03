# Handoff — rebase-dflash-onto-upstream-6fa01a49

## Objective and location
Rebase the fork's patch stack (`deploy`, `patch/*`, `base/current` = 56f5702a) and `wt/dflash-s-e2vk` onto upstream `6fa01a49`, so the DFlash2 work sits on upstream's new DFlash2 components before the follow-on fold of `components/dflash_draft.py` into `components/dflash2.py`. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk` (branch tip `b91e06d7`, pushed); main checkout `~/dev/flare-local-inference` on `deploy` (`4718693a`). Tier 3, `Concurrency: graph`.

## Stage and stopping point
Planning complete and lint clean (proposal, design with peer-review quorum and bomb-squad findings, tasks; specs skipped). No task started. The change directory is untracked in the worktree and nothing is committed; leave it untracked through the rebase (untracked files survive `git rebase`).

## Facts the next session needs
- `main` is the upstream mirror, reset to `6fa01a49` and force-pushed on 2026-11-02 after a rad-close had landed the DFlash branch on it. Never merge into or rebase `main`; never `/rad-close` this branch (memory `dflash-worktree-stays-open`).
- GPU 0 is held right now by the owner's console DFlash2 profile (`vllm-console-…-copy-261102_002456`, image `b91e06d7`, ~96 GB). Task 3.2 needs the owner to stop it; never `docker stop` it.
- `~/LocalApps/vllm-env` has 15 uncommitted files from another session (the console's `--recipe`/`--draft-model` fields); out of scope, do not touch.
- The highest-risk item is the bomb-squad's: `qwen38_dflash/bootstrap.py:459` reads the deleted `qwen38/assets/chat_template.jinja`; the port is in task 2.1 and fails only after capture if missed.
- Session model is now Opus 5.5. FORK.md records the operator's override of the AGENTS.md Opus kernel/optimization ban for this fork; the conflict resolutions here are carries of existing code, not new kernels.

## Resume prompt
```
Read openspec/changes/rebase-dflash-onto-upstream-6fa01a49/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, then run /rad-apply rebase-dflash-onto-upstream-6fa01a49. Start with tasks 1.1 and 1.2 (independent). main stays the upstream mirror; GPU 0 belongs to the owner until they stop their instance for task 3.2.
```
