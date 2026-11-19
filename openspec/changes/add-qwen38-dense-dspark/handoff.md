# Handoff — add-qwen38-dense-dspark

**Objective.** Add DSpark as a second speculative method (`--speculative-method dspark`) on the dense Qwen recipe `ignition/recipes/qwen38_dense/`, with draft `RadixArk/Qwen3.8-27B-DSpark` @ `b9a5dbdf` and checkpoint `RadixArk/DaisyRose-27B-NVFP4` @ `c4d85c5d` as alias `daisyrose-27b`. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv`, branch `wt/daisyrose-ta-ttv`.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Planning complete (Tier 3): proposal, design, spec delta and tasks exist, untracked and uncommitted. No project code edited. Next: apply, starting with tasks 1 and 3 (independent; `Concurrency: graph`).

**State to carry.**
- Peer review never reached quorum (two calls; the Sonnet 5.5 peer timed out both times). The operator chose "Override and continue"; `design.md` records the `Override:` line. The linter warning about it is expected. No peer feedback was incorporated.
- Bomb-squad ran once; its findings and dispositions are in `design.md` `## Bomb-squad findings`. Do not dispatch it again.
- Unresolved until task 5: draft page dtype. FP8 if `paged_decode.supports` accepts noncausal, unwindowed tail pages at this geometry, else BF16. b12x is not installed in this worktree's `.venv`, so this could not be checked statically.
- Unverified reading: block of 7 with `logit_offset = 0` (position i proposes draft i+1), taken from the model card's "seven proposals, block size 7". Tasks 6 and 7 test it.
- Not included: context scaling past 262,144 tokens — "We can make context scaling a followup".
- Task 7 needs the operator: GPU 0's Qwen instance is owner-managed, and the task carries an operator-approval record.
- Local inputs: `~/models/RadixArk/Qwen3.8-27B-DSpark` (complete), `~/models/RadixArk/DaisyRose-27B-NVFP4` (21G, 6 shards on 2026-11-18; confirm the download finished). Fetched configs, cards and the draft's `dflash.py`/`dspark.py` as text: `~/models/.cache/hf-cards-dspark/`.
- The worktree `.venv` lacks `transformers` and `b12x`; an earlier change ran CPU tests through the main checkout's interpreter. Nothing is committed; commit only when asked.

**Resume prompt.**

```
/rad-apply add-qwen38-dense-dspark

Read openspec/changes/add-qwen38-dense-dspark/handoff.md first, in worktree /home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv (branch wt/daisyrose-ta-ttv). Planning is complete and uncommitted; no code has been edited. Apply the change: add the `dspark` speculative method to ignition/recipes/qwen38_dense/ (draft RadixArk/Qwen3.8-27B-DSpark @ b9a5dbdf) and the `daisyrose-27b` checkpoint (RadixArk/DaisyRose-27B-NVFP4 @ c4d85c5d). Start with tasks 1 and 3, which are independent. The peer-review quorum was overridden by the operator and the lint warning about it is expected; bomb-squad already ran. Task 5 decides the draft page dtype (FP8 if the paged-decode kernel supports the geometry, else BF16). Task 7 needs the operator for GPU 0. Context scaling past 262,144 tokens is not included. Do not commit unless asked.
```
