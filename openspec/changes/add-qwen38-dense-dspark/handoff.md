# Handoff — add-qwen38-dense-dspark

**Objective.** Re-export the DaisyRose 4x core with the reuse fix and image input, then get the operator's re-approval. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv`, branch `wt/daisyrose-ta-ttv` (= `deploy` + 9 commits, not pushed).

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Apply. Tasks 1 to 6 and 8 to 13 are checked; task 7 (export, serve, operator approval) is open. The reuse findings of `verification.md` are fixed in 9ab62856. Image input for the dense recipe (operator addition, 2026-11-19) is in 0b0beaf8: tasks 10 to 13.

**Next action.** Wait for the operator's go for the export start. Then: seed the kernel folder of image `local/flare-ignition:0b0beaf8` from `~/models/.cache/flare/kernels/images/c607218614fe`, point the profile at the image, start it through the console, and check the served core: `/v1/models` reports `max_images_per_prompt` 4 and image input, a chat request, an image request, accepted drafts per round, a prompt past 262,144 tokens. Diff the `flare` section of `/v1/models` against the replaced profile's capabilities before asking for approval (`openspec/lessons.md` `a-replacement-core-keeps-the-replaced-profiles-capabilities`).

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
/rad-apply add-qwen38-dense-dspark

Read openspec/changes/add-qwen38-dense-dspark/handoff.md first, in worktree /home/rjdollete/dev/flare-local-inference.worktrees/daisyrose-ta-ttv (branch wt/daisyrose-ta-ttv). The reuse fix (9ab62856) and image input for the dense Qwen recipe (0b0beaf8) are committed and tested on GPU 1; image local/flare-ignition:0b0beaf8 is built or building. Task 7 is open: the DaisyRose 4x core must be re-exported from that image and re-approved. The DaisyRose 4x text-only core is serving on GPU 0 and I am using it: do not stop or restart it until I say go. After my go, seed the kernel folder, re-export through the console, check image input on the served core, then ask me to re-approve.
```
