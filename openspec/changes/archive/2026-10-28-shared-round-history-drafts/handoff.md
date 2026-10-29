# Handoff — shared-round-history-drafts

## Objective and location
Port history-draft substitution from Qwen's MTP round into the shared `PreparedParallelDraftRound` — first of three changes toward a DFlash2 recipe for Qwen3.8-Flash-Next (the acceptance spike is archived, commit `bc3ca1fe`).
Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk`, branch `wt/dflash-s-e2vk`. Change dir `openspec/changes/shared-round-history-drafts/` (Tier 3; review quorum not achieved, `Override: Ray Dollete` in design.md).

## Stage and stopping point
Apply and simplify done; `verification.md` = **PASS-WITH-FINDINGS**. Tasks 1.1–1.5 DONE; 1.6 (served qualification, operator leg) **BLOCKED** (tasks.md discovery lines). Next: `rad-sync` (delta `specs/history-drafts/spec.md` unsynced), then `rad-archive`.

## Uncommitted work
Everything for this change is uncommitted on the branch: 16 modified files, new `tests/components/test_history_drafts.py`, and `openspec/changes/{shared-round-history-drafts,nemotron-export-compiler-publish}/`. Owner's standing instruction (given for the spike): "Go ahead and commit and push the spike, but I want to continue working on this branch/wt." — commit on this branch, no `rad-close`. Commits need the `docs/design-review.md` acknowledgment trailer; the hook is enabled.

## Blocker and follow-up
`ignition export` of `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` fails on this host at "Prepare and tune kernels": `compiler completed without publishing required artifacts` (`flare.gated_rms_norm`). Reproduced without this change. Logs: `/tmp/dflash2-probe/nemotron/export.log`, `export-image.log`. `openspec/changes/nemotron-export-compiler-publish/` is scaffolded and empty; unknown cause → start with `/rad-debug`. Unproven theory: the pool's planning step compiles then evicts artifacts, and a later job in a worker whose in-process Triton cache holds the kernel rewrites nothing (`b12x/_lib/compile_pool.py:405-440`, `preparation/session.py:2062-2075`).
Staged for the served check: `flare/target/release/flare` built from this tree (`PROTOC=/tmp/dflash2-probe/protoc/bin/protoc`), publisher profile `/tmp/dflash2-probe/nemotron/config/`, script `/tmp/dflash2-probe/nemotron/qualify.py`.

## Environment
Owner stopped the Qwen instance (LocalAI console, port 8100) for the export; GPU 0 is free and it is still down — remind them to restart it. Host venv lacks `flare_heliopause`/io_uring; weight-loading GPU work runs in `local/flare-ignition:e8bac4a6` with the serving container's seccomp/memlock/ipc settings. Pre-existing unrelated failures: 3 Kimi-checkpoint cases in `tests/recipes/test_declarations.py`.

## Resume prompt
```
Read openspec/changes/shared-round-history-drafts/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, then run /rad-sync shared-round-history-drafts (verification passed; the history-drafts delta spec is unsynced). After sync and archive, commit this change on wt/dflash-s-e2vk with the design-review acknowledgment and push; do not rad-close. Task 1.6 stays BLOCKED on the Nemotron export defect, which continues in openspec/changes/nemotron-export-compiler-publish/ via /rad-debug.
```
