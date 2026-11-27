# Handoff — debug-v0-3-2-device-error-1

**Objective.** Find and remove why the Qwen core `gb202-tp1-202611260714` fails every request with `prepared model reported device error 1`.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Apply, 3 of 10 tasks checked; no work is open here. The cause is confirmed (ledger `openspec/debug/new-stack-package-nan-logits.md` H10) and the fix is upstream FlashInfer `c3ab1d04`. On 2026-11-26 the owner moved the goal to upstream `main` (design.md `## Reframe`): task 4 is obsolete, and task 5 is done when task 7.2 of `rebase-forks-onto-upstream-v0-3-2` passes its served checks; that run ticks ledger check C1. Tasks 6 to 10 apply only if that export fails with the same error. Continue in the rebase change; its `handoff.md` holds the resume prompt.

**GPU 1 kernel test recipe.** Folder `~/models/.cache/device-error-1/`: `flashinfer-49efc285/` (tests and the two changed files of the fix), `scratch/h10_grouped_rmsnorm.py`, logs `task3-*.log`. Command: `docker run --rm --gpus device=1 --ipc host --shm-size 8g --ulimit memlock=-1:-1 --user $(id -u):$(id -g) -v ~/LocalApps/vllm-env/state/flare/passwd:/etc/passwd:ro -v <dir>:<dir> -v ~/.cache:~/.cache -e HOME=$HOME -e TRITON_CACHE_DIR=~/.cache/device-error-1-triton -e PYTHONPATH=~/.cache/flare-pytest-target --entrypoint bash <image>`, then `/opt/flare/.venv/bin/python -m pytest -p no:cacheprovider -q tests/experimental/b12x/norm/test_prepared_hyperconnection.py::test_ordinary_rmsnorm_uses_affine_weight_without_offset` from a `git archive` of FlashInfer's `tests`, `pytest.ini` and `pyproject.toml` (the image has no pytest; the target folder supplies it). To test source the image does not hold, mount the changed files over the installed ones under `/opt/flare/.venv/lib/python3.12/site-packages/flashinfer/`.

**Open question for the owner, not a hold:** offer an export-time finiteness check once the newer base serves (design.md `### Decision held for the operator`).
