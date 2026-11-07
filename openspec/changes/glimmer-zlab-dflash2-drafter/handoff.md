# Handoff — glimmer-zlab-dflash2-drafter

## Objective and location
Swap the `muse_glimmer_nvfp4` recipe's drafter from `meta-models/Muse-Glimmer-30B-assistant` to `z-lab/Muse-Glimmer-30B-DFlash2` (`b54ffdd1`) through the shared DFlash2 path. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb`, branch `wt/glimmer-zlab-dflash2` from deploy `8f1bacfe`. Change dir `openspec/changes/glimmer-zlab-dflash2-drafter/` (untracked; planning complete, Tier 2, review quorum 3/3).

## Stage and stopping point
Planning complete, no code edited, 0/7 tasks ticked. Prior change `rename-muse-glimmer-nvfp4` landed on deploy `8f1bacfe` (recipe name `muse_glimmer_nvfp4`, alias unchanged).

## State the next session needs
- The console profile `2026-11-06-flare-muse-glimmer-1793992412838` still reads `--recipe muse_glimmer`, `--draft-model meta-models/...`, image `37f3613d`; it is stopped (operator stopped it) and GPU 0 is idle. It is updated only in task 3.1, together with the new image; never launch it before.
- Export deferred by the operator until this drafter lands (one export, not two).
- Tests: CPU runs need `PATH=/tmp/protoc-bin:$PATH PROTOC=/tmp/protoc-bin/protoc`; CUDA tests GPU 1 only, host venv, RSS watchdog under 14 GiB; write long outputs under `/home/rjdollete/models/.cache/` (`/tmp` is on the near-full root disk). Deploy baseline of 62 pre-existing CPU failures: `/home/rjdollete/models/.cache/glimmer-rename-t21-full-baseline.txt` (task 2.2 compares against it).
- Local checkpoints: target hub cache `models--original-el8--Muse-Glimmer-30B-NVFP4/snapshots/b47e1e9f…`; drafter `~/models/z-lab/Muse-Glimmer-30B-DFlash2` (81 tensors, 5,544,328,424 B).
- Open technical question settled by task 2.1, not by the operator: whether the DFlash2 head applies the target multiplier/softcap (config, Spark form) or the bare head; fallback rule is in the task's Done-when.
- Earlier close commits need the trailer `Design-Signed-Off-By: Claude Fable 5.1 (claude-fable-5-1); design-review=a38f` (recompute from `docs/design-review.md`) plus the Co-Authored-By line; commit only for task 3.1's WIP build or at close.
- Agent memory: `drafter-preference-dflash2`, `recipe-naming-lineage`, `gpu0-container-tiers`.

## Owner's verbatim words
- "I prefer the z-lab route."
- "we can keep the alias"
- "Since this is not a functional difference, can we bundle this with the rope/yarn change?" — superseded: the rename is landed; this change carries its own GPU window (operator chose "After the rename lands today").

## Resume prompt
```
Read openspec/changes/glimmer-zlab-dflash2-drafter/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb (branch wt/glimmer-zlab-dflash2), then run /rad-apply glimmer-zlab-dflash2-drafter from the ready frontier (tasks 1.1, 1.2, 1.3 are independent). Recheck git status and the console status before task 3.1; GPU 0 only in that task.
```
