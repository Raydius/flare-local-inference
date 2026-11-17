# Handoff — elephant-1-trained-mask

**Objective.** Serve Elephant-1 with its trained two-block attention mask, starting with a feasibility stage that changes no serving code. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/elephant-05odxt`, branch `wt/elephant-05odxt`, HEAD `1a361326` (pushed). Stage: planning complete, apply not started; 4 tasks open. The change directory is untracked.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

## Resume prompt

```
/rad-apply elephant-1-trained-mask
Read openspec/changes/elephant-1-trained-mask/handoff.md first, then proposal.md "## Stated Requirements".
Worktree /home/rjdollete/dev/flare-local-inference.worktrees/elephant-05odxt, branch wt/elephant-05odxt, HEAD 1a361326; the change directory is untracked.
State: planning is complete with the peer-review gate overridden by the operator (design.md records it). Nothing is implemented. Core gb202-tp1-202611161942 still serves on GPU 1 at http://127.0.0.1:8120; the Qwen instance is back on GPU 0.
Next: task 1, the mask ablation (trained twice, causal, suffix-only, prefix-only) on the reference side of qualify/elephant1_decision_parity.py. It needs GPU 0 for about 15 minutes: ask the operator to stop Qwen, wait until GPU 0 is free, run, then say it is free. Tasks 2 and 3 need no GPU 0 and can go first. Task 4 is the operator's build decision; no serving code changes before it.
Vendor reference run: docker run --gpus device=0 --memory 24g -v ~/models/baselabs/elephant-1:/ckpt:ro -v ~/models/.cache/elephant1-parity/vendor:/work -v ~/models/.cache/elephant1-parity/out:/out -v "$PWD/qualify":/src/qualify:ro -w /src local/elephant1-vendor:81ce62f python -m qualify.elephant1_decision_parity reference --checkpoint /ckpt --device cuda:0 --output /out/<file>.json
Constraints: you must be Fable 5.2 for kernel, projection-packing or stream-overlap code. Flare exports and checkpoint loads run only through the console. Do not push the FlashInfer branch. Host Python is /home/rjdollete/dev/flare-local-inference/.venv/bin/python (the worktree has no .venv).
```

## State a fresh session cannot derive

- **Previous change.** `elephant-1-text-recipe` is archived and pushed but not landed on `deploy`; `/rad-close` is the operator's to run. Its console package-record procedure is in `openspec/changes/archive/2026-11-16-elephant-1-text-recipe/handoff.md`.
- **Operator decisions.** Maximum concurrency stays 2 ("no leave it at 2 for now"). The served core predates the packer storage edit in `ignition/components/weights.py`; the next export needs a new footprint approval (`docs/TODO.md`).
- **Open, operator's call.** Whether the body packer and `nvfp4_draft_head` share one function (archived `verification.md`, finding 1).
- **Earlier reference outputs.** `~/models/.cache/elephant1-parity/out/` holds `reference.json`, `core.json` and `probes.json` from the two-mask run; the stage reports are in `qualify/results/elephant1-decision-parity/`.
- **Unverified hunt readings.** Bomb-squad findings in design.md on record identity and checkpoint endpoints were not confirmed against the code; the span mechanism and prefix-key contents were.
- **Review.** The tier-2 peer timed out twice on this brief; expect the same on a stage 2 review unless its budget is raised in Quinn-app's configuration.

Budget exception: the resume prompt is self-contained by the owner's standing preference, and the vendor reference command is recorded nowhere else in the repository.
