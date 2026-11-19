# Handoff — rebase-deploy-onto-upstream-1a532755

**Objective.** Rebase the Flare fork patch stack (`base/current`, `patch/00`, `01`, `03`, `06`, `07`, `deploy`) from upstream `6b68b9b4` onto `upstream/main` `1a532755`, then gate and publish.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Propose complete: `radflow-lint` exit 0, `specs` skipped (`skip_specs: true`), 0 of 8 boxes ticked. Peer review quorum achieved (certainty 0.8125, caller agreement 0.7, reconciled in design.md `### Review Feedback`); bomb-squad's six plan corrections recorded in design.md `## Bomb-squad findings`.

**Location.** `/home/rjdollete/dev/flare-local-inference`, branch `deploy`, tree clean apart from this untracked change dir. Recheck mutable git state first.

**Already done — verify, do not redo.**
- `main` == `origin/main` == `upstream/main` == `1a532755` (`scripts/fork/sync.sh`, 2026-11-18). `base/current` still `6b68b9b4`; `deploy` still `50dade75`, unrebased.
- Backup tags `backup/pre-rebase-20261118-{deploy,main,base-current,patch-00,01,03,06,07}` at the design.md step 1 commits.
- Probe replays and logs in `~/models/.cache/rebase-1a532755/` (`probe.sh` reruns it if the tip moved). Scratch branches `probe/1a532755-{theirs,both}` and their worktrees under that cache dir are disposable after apply.

**Next action.** `/skill:rad-apply rebase-deploy-onto-upstream-1a532755` from task 1.1.

**Open / gating.**
- Task 1.2 gates the rest: decide the FlashInfer pin against upstream's new `2cce3e67` by running the fork's three eviction cases in the serving image; it fixes `pyproject.toml:25` and `uv.lock`, which gate the CPU suite. `~/dev/flashinfer` already has `2cce3e67`, `7245b1e0b`, `2f55cfd98`.
- Rebase stops: five conflicted files (`docs/TODO.txt`, `pyproject.toml`, `uv.lock`, `qualify/speculative_benchmark.py`, `tests/reference/test_speculative_benchmark.py`), each with a written rule in design.md step 4; expected range-diff 120 `=` and 5 `!` (`de8ada69`, `6cbf42d4`, `cb071b57`, `6f2c9d5b`, `4128ea63`).
- Checklist digest moved `ed22` → `1af4`: acknowledge `1af4` on commits this change writes only; replayed commits continue with `core.hooksPath=/dev/null`.

**Standing owner holds.** Re-export and served re-measurement of the entries the console serves — the Qwen, Muse Glimmer and Elephant entries — are out of scope. GPU 0 is the owner's console — never `docker stop` it; GPU 1 is the gate card. Heavy runs one file at a time under a 14 GiB watchdog on the real PID. Images and serving from committed refs via `~/LocalApps/vllm-env/build/flare-ignition/build.sh`.

**Unverified.** No GPU execution of the rebased stack; gates 3.1–3.3 have not run. The operator's "review it again, what's missing?" was answered with the artifact and bomb-squad state; no artifact edit followed that last clean lint.
