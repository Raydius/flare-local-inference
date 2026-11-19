# Handoff — add-qwen38-dense-dspark

**Objective.** Close the change: every task is checked. The operator approved on the GPU 1 results without a re-export ("I'll treat the GPU 1 test as approval", 2026-11-19) and asked for the change to be merged down. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv`, branch `wt/daisyrose-ta-ttv`, not pushed.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Apply complete, 13 of 13. Next: rad-simplify, rad-verify, rad-sync, rad-archive, then rad-close typed by the operator.

**Next action.** Run the remaining lifecycle steps in order. No export is owed by this change; image `local/flare-ignition:0b0beaf8` is built with a seeded kernel folder (`~/models/.cache/flare/kernels/images/35fa42c7001d`) for when the operator wants a core with image input.

**State to carry.**
- Serving now, do not stop without the owner's word: profile "2026-11-19 Flare DaisyRose-27B DSpark c16 YaRN4" (id in `~/models/.cache/dspark-task7/profile-id-yarn4`), core `gb202-tp1-202611191014`, image `local/flare-ignition:0c0f5860`, GPU 0 port 8100 as `Qwen3.6`, text only. The owner uses it.
- First runs at the export, with no smaller reproduction available: `bootstrap.prepare` with vision, the dense decoder with three-axis positions and the stored offset, Helios loading the image contract. If the export fails, reduce the failure to a GPU 1 test before another GPU 0 run. GPU 1 covers the vision graph, the upload generation and the position computation (`tests/recipes/test_qwen38_dense_vision*.py`, `test_qwen38_dense_positions.py`).
- Console rules (memory `each-export-start-needs-its-own-go`): each start needs the owner's go for that profile; `POST /api/server/stop` names no profile, read `active_model_id` first. Helper: `~/models/.cache/dspark-task7/console.py`.
- Tests: interpreter `/home/rjdollete/dev/flare-local-inference/.venv/bin/python`; GPU 1 with `CUDA_VISIBLE_DEVICES=1 FLARE_RUN_CUDA_TESTS=1`. Loader tests need the image: `docker run --gpus '"device=1"' ... -v "$PWD":/work -e PYTHONPATH=/work:$HOME/.cache/fold-gpu/site --entrypoint /opt/flare/.venv/bin/python local/flare-ignition:<sha> -m pytest ...` (vault note "Two-GPU Development Workflow", section 4).
- Commit hook digest is `design-review=1af4`.
- After approval: record the operator-approval line at the exported commit, update `docs/model-preparation.md` (the served-core paragraph still describes the language-only core) and the vault notes under `/data/CloudVault/Personal Projects/AI Control Plane/HyperFlare/DaisyRose/`, then rad-simplify and rad-verify.
- Open after this: concurrency 1 and 8 trials, `flare inspect` costs, the image checks in `docs/TODO.md`.

Budget exception: the owner requires a self-contained resume prompt, and the serving-instance and console rules must travel with it.

**Resume prompt.**

```
/rad-verify add-qwen38-dense-dspark

Read openspec/changes/add-qwen38-dense-dspark/handoff.md first, in worktree /home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv (branch wt/daisyrose-ta-ttv). All 13 tasks are checked: the reuse fix (9ab62856) and image input for the dense Qwen recipe (0b0beaf8) are committed and tested on GPU 1, and I approved on those results without a re-export. Finish the lifecycle (simplify, verify, sync, archive) so I can close and merge. The text-only DaisyRose 4x core is serving on GPU 0 and I am using it: do not stop or restart it. No export is owed; image local/flare-ignition:0b0beaf8 is built for a later export with image input.
```
