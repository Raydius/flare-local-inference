# Handoff — add-qwen38-dense-dspark

**Objective.** Finish the change by replacing three recipe-local copies of shared DFlash2 code with reuse, then re-export and re-approve the DaisyRose core. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv`, branch `wt/daisyrose-ta-ttv` (= `deploy` + 6 commits, tip `d262792d`, not pushed).

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Verify ran: PASS-WITH-FINDINGS (`verification.md`). The auditor unchecked tasks 3 and 5 for Reused findings 1 to 3. Operator chose "Fix by reusing shared code" (`design.md` `## Reuse decision after verification`). No fix started.

**Next action.** For each finding, widen the shared component with an optional argument defaulting to today's behaviour, and import it:
1. `dspark_weights.py` loader and shape table → `ignition/components/dflash2_weights.py` (`projector` switch).
2. `dspark.py` `rotate` → `_rotary` in `ignition/components/dflash2.py` (gain).
3. `dspark.py` `PreparedContext` → `PreparedContext` in `ignition/components/dflash2.py` (page slots, no window).
Flash Next, Muse Glimmer and Nemotron prepare through these; their behaviour must not change. Then test, commit, rebuild, re-export the 4x profile, re-ask the operator leg (it is stale once code moves past `0c0f5860`).

**State to carry.**
- Serving now, do not stop without the owner's word: profile "2026-11-19 Flare DaisyRose-27B DSpark c16 YaRN4" (id in `~/models/.cache/dspark-task7/profile-id-yarn4`), core `gb202-tp1-202611191014`, image `local/flare-ignition:0c0f5860`, GPU 0 port 8100 as `Qwen3.6`. The owner uses it.
- Console rules (memory `each-export-start-needs-its-own-go`): each start needs the owner's go for that profile; `POST /api/server/stop` names no profile, read `active_model_id` first. Helper: `~/models/.cache/dspark-task7/console.py`.
- Uncommitted: auditor's artifact fixes (proposal, design, spec, tasks, `docs/TODO.md`), `verification.md`, `.radflow-audit`, this file.
- Tests: interpreter `/home/rjdollete/dev/flare-local-inference/.venv/bin/python`; GPU 1 with `CUDA_VISIBLE_DEVICES=1 FLARE_RUN_CUDA_TESTS=1`. Run `tests/components/test_paged_attention_tail_cuda.py`, `test_dflash2*.py`, the DSpark CUDA and reference files before any export. 18 host tests fail identically on `deploy`.
- Rebuild: `build.sh <sha>`, then `cp -a` the kernel folder `~/models/.cache/flare/kernels/images/c607218614fe` to the new image id (export 35 min seeded, 46+ min unseeded).
- Commit hook digest is `design-review=1af4`. FlashInfer patching is permitted by the owner; none was needed.
- Vault notes: `/data/CloudVault/Personal Projects/AI Control Plane/HyperFlare/DaisyRose/`; update after the re-export.
- Open after this: concurrency 1 and 8 trials, `flare inspect` costs, scheduler / reasoning-effort / prefix-disk settings (`docs/TODO.md`).

Budget exception: the owner requires a self-contained resume prompt, and the serving-instance and console rules must travel with it.

**Resume prompt.**

```
/rad-apply add-qwen38-dense-dspark

Read openspec/changes/add-qwen38-dense-dspark/handoff.md first, in worktree /home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv (branch wt/daisyrose-ta-ttv). Verification unchecked tasks 3 and 5 for three reuse findings (verification.md); I chose "Fix by reusing shared code". Widen ignition/components/dflash2_weights.py and dflash2.py with optional arguments that default to today's behaviour, and make dspark_weights.py and dspark.py import them (loader and shape table, _rotary with a gain, PreparedContext with page slots). Other recipes' behaviour must not change. Test on GPU 1, commit, and build the image. The DaisyRose 4x core is serving on GPU 0 and I am using it: do not stop or restart it until I say so. After I give the go, re-export, then ask me to re-approve.
```
