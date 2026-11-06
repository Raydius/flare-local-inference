# Handoff — history-drafts A/B at fixed-7

Objective: decide whether DFlash2 serving should add `--history-drafts`. Change `dflash2-history-drafts-ab` (main checkout ~/dev/flare-local-inference, deploy 58ac27be, clean, pushed). Stage: propose in progress — BY ANOTHER SESSION (proposal.md written 2026-11-06 00:52 PST; two `claude --print` processes live). Its proposal already fixes the design: `qualify/dflash2_draft_sweep.py --arms fixed-7` run twice (off, then `--history-drafts`) at c1 and c8 on image a945c147, package gb202-tp1-202611040437, catalog ~/.cache/dflash2-served/yarn4/catalog-dflash2.json, workload ~/.cache/dflash2-served/workload2 (all present); results to qualify/results/dflash2-history-drafts/{off,on}/; serving switch is the owner's follow-up. Control rows: qualify/results/dflash2-draft-sweep/summary.json fixed-7 c1 537.3 / c8 351.7 tok/s; repetition band ≈ ±8 % (vault tracker §16).

Blocker inside the change: the trials need GPU 0, held by the owner's instance (image e6fc2ef6, 96 GB); only the owner stops it (one stop for all four trials). GPU 1 (16 GB) cannot hold the model.

Unratified: my "turn it on if c1 improves and c8 doesn't drop" rule is a suggestion, not a decision.

## Owner's verbatim words
- "I want to go with fixed-7 for now. if I want maximum concurrency performance I'll just use MTP instead."
- "Create a handoff prompt to kick off the AB"

## Resume prompt
```
Context: Flare fork, main checkout ~/dev/flare-local-inference on deploy 58ac27be (clean, pushed; commit hook active — Design-Signed-Off-By per docs/design-review.md). Read .claude/handoff.md first. Change dflash2-history-drafts-ab exists in openspec/changes/ and was mid-/rad-propose by ANOTHER session at 2026-11-06 00:52 PST.

First, before touching that change: run `ls -la --time-style=full-iso openspec/changes/dflash2-history-drafts-ab/`, `pgrep -af 'claude --print'` and `radflow-lint --next openspec/changes/dflash2-history-drafts-ab`. If files changed within the last ~10 minutes or that session is still running, do not write into the change — report its state and ask the operator whether to wait or take it over. Never run `openspec new change` or a second /rad-propose on that name.

If it is idle: run the lifecycle command radflow-lint --next names (currently /rad-propose dflash2-history-drafts-ab; proposal.md is written, design/review/tasks are not). Keep the proposal's design (four trials, same image/package/workload as the committed sweep; results under qualify/results/dflash2-history-drafts/; serving switch is the owner's follow-up). The sweep command with exact paths is in openspec/changes/archive/2026-11-04-dflash2-draft-sweep/tasks.md line 34; add --history-drafts for the on run and a new --output. The GPU 0 stop is an operator leg: ask once, quoting it.

Constraints: never docker stop the owner's instance; exports/serves run containerized from a committed ref (build.sh now defaults to deploy); main stays the upstream mirror; commit/push only when asked; update vault "Control Plane/Flare/DFlash2 Recipe Progress.md" Part A when state changes (its Later item on history drafts points at this A/B).
```
