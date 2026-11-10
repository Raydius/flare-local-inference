# Handoff: rebase-deploy-onto-upstream-8c2e782d (2026-11-10)

**Objective.** Rebase the fork's patch stack and `deploy` onto upstream `e774a468` (operator: "Retarget to e774a468"). Repo `/home/rjdollete/dev/flare-local-inference`, branch `deploy` = `77f0e65b`. Nothing is pushed; `backup/pre-rebase-20261109-*` tags hold the old refs. The change directory is untracked; commit it at close.

**Stage.** `/rad-apply`, tasks 1.1–2.2 ticked, 3.1–3.5 open. Operator: "Ok let's proceed with all recommended steps. The current server has been stopped." GPU 0 is free; the owner restarts the console instance, never `docker stop` it.

**Scratch.** `G=~/models/.cache/rebase-e774a468` (scripts, logs, worktrees `trees/tip-deploy`, `trees/pristine`; also `/tmp/upstream-e774a468`). Image `local/flare-ignition:77f0e65b` is built from `deploy`. Docker cannot mount the session's `/tmp`; use paths under `$G`.

## Open state

- **3.1 has a real tip-only failure.** `$G/tip/` is complete (`_done`); `$G/pristine/` was still running when this session ended and may be cut short (no `_done` = rerun `$G/run-gate.sh /tmp/upstream-e774a468 $G/pristine`). `tests/cli/test_mtp_only.py` passes on pristine and fails on the tip: `test_recipe_adapters_preserve_mtp_only_and_validate_direct_configs` for `[False-muse_glimmer_nvfp4]`, `[True-qwen38_dflash]`, `[True-muse_glimmer_nvfp4]`. The fork's two recipes do not yet follow upstream's `--mtp-only` adapter contract. Fix in the recipes, commit on `deploy`, rebuild the image from the new sha, then compare every other failing file's node IDs tip against pristine (`_exit.txt` plus per-file logs). The tip run started before `77f0e65b` restored `compilation.py`; rerun `tests/preparation/test_aot.py` on the tip.
- **3.2 nearly done, unticked.** Image checks passed (`uv lock --check`, flashinfer `ef6d6592`, binary from `flare-cli/`). `$G/rust-baseline.log`: 14 fork tests pass on deploy; pristine `flare-cli` fails the same 2 orbit agent tests as deploy. Deploy also failed `actual_load_core_leases_managed_local_entries_and_releases_on_failure`, which pristine did not: rerun it alone to tell a flake from a regression. The pristine Helios run (deploy: 1338 passed, 8 failed) was in progress; if the log has no Helios result, rerun it: `docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp --memory 14g --gpus device=1 -v $G/trees/tip-deploy:/trees/tip-deploy -v $G/trees/pristine:/trees/pristine -v $G/cargo-target:/target -v $G/cargo-target-pristine:/target-pristine -v $G/cargo-home:/cargo-home -v $G/rust-baseline.sh:/rust-baseline.sh:ro --entrypoint bash local/flare-ignition:77f0e65b /rust-baseline.sh`.
- **3.3 not started.** Run `$G/gpu1-gate.sh` in the image with `--gpus device=1 --memory 14g`, run as the user with `HOME=/tmp`, mounting `$G/trees/tip-deploy:/work`, `~/models/incoai/Qwen3.8-Flash-Next-DFlash2:/ckpt/draft`, `~/models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD:/ckpt/target`, `$G/site:/site`, `$G/site-helio:/site-helio`, `$G/cargo-home:/cargo-home`. After a new commit, update the worktree and the image tag first. Record two things on the task: the opaque-write retirement evidence (9 cases pass with and without the rule) came from the host venv on GPU 1, not a container; the trace-based parity case skips because `$HOME/.cache/dflash2-probe/run2/traces` no longer exists.
- **3.4 is an operator leg.** Ask once, quoting its criterion from `tasks.md`; export fixed-7 at concurrency 8 on the final image as console profile "2026-11-10 Flare DFlash2 Rebase3".
- **3.5** push with lease against the backup shas, then vault notes and the vllm-env image row.

Run one heavy container at a time (14 GiB ceiling). Check `docker ps` for a leftover `local/flare-ignition:77f0e65b` container from this session before starting another.

Budget exception: the two container commands exist nowhere else on disk.

## Resume prompt

```
Read openspec/changes/rebase-deploy-onto-upstream-8c2e782d/handoff.md in /home/rjdollete/dev/flare-local-inference, then run /rad-apply rebase-deploy-onto-upstream-8c2e782d. Recheck git, docker ps and ~/models/.cache/rebase-e774a468 before acting. First fix the tip-only tests/cli/test_mtp_only.py failures in the qwen38_dflash and muse_glimmer_nvfp4 recipes (task 3.1), then finish 3.2, run 3.3 on GPU 1, and ask me once for the 3.4 GPU 0 export leg. GPU 0 is free; I restart the console instance myself.
```
