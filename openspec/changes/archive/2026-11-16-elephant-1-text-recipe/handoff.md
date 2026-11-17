# Handoff — elephant-1-text-recipe

**Objective.** Serve the text-only Elephant-1 decision model through Flare inside 12 GB. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/elephant-05odxt`, branch `wt/elephant-05odxt`, HEAD `a79300e8`, clean. Stage: rad-apply; tasks 1–3 and 5 checked, 4, 6, 7 open.

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

## Resume prompt

```
/rad-apply elephant-1-text-recipe
Read openspec/changes/elephant-1-text-recipe/handoff.md first, then proposal.md "## Stated Requirements".
Worktree /home/rjdollete/dev/flare-local-inference.worktrees/elephant-05odxt, branch wt/elephant-05odxt, HEAD a79300e8.
State: core gb202-tp1-202611161942 is exported and serves on GPU 1 (RTX 5060 Ti) via console profile "2026-11-16 Flare Elephant-1 c2 GPU 1" in the Decision tab (instance llm2, port 8120), image local/flare-ignition:9efef537; operator approved 9,608 MiB. Tasks 1-3 and 5 are checked.
Next: task 6, qualify/elephant1_decision_parity.py against the vendor DecisionEngine (served instance at http://127.0.0.1:8120); then task 4's leftovers (flare inspect on the core; ignition debug-serve --direct --eager forward); then task 7 docs and regression sweep.
Constraints: you must be Fable 5.2 for kernel, projection-packing or stream-overlap code (design-review principle 14). Exports and checkpoint loads run only through the console, on an image built with ~/LocalApps/vllm-env/build/flare-ignition/build.sh from a committed ref. Ask before taking GPU 0 (owner's Qwen instance). Do not push the FlashInfer branch.
```

## State a fresh session cannot derive

- **Console package records.** The console fingerprints a package by image id, `--gpus` value, export fields and Max Num Seqs. Changing any of them on an Elephant-1 profile triggers `ignition export`, which fails on GPU 1 (16 GB). Fix: rewrite `~/LocalApps/vllm-env/state/flare/entries/<profile-id>` to `<fingerprint> /flare-packages/models/ckpt/gb202-tp1-202611161942`; read the fingerprint from the failed container's args (`docker inspect <name> --format '{{json .Args}}'`, the 64-hex before `"$entry"`). API: `POST /api/server/start?instance_id=llm2`; token is `VLLM_CONSOLE_API_TOKEN` in the environ of `vllm-env/bin/python -m console`.
- **GPU 1 run settings** (run-only, no re-export): `gpu_memory_utilization` 0.88 and `run:--kv-cache-limit concurrency`. Only the "GPU 1" copy has them; the original profile "2026-11-16 Flare Elephant-1 c2" was set to `device=1` by the operator without them and fails at "Allocate cache".
- **FlashInfer patch.** Memory `flashinfer-gdn-head-ratio-patch`: local branch `gdn-head-ratio` (2f55cfd98), carried as an uncommitted patch file in `~/LocalApps/vllm-env/build/flare-ignition/`.
- **Host tooling.** The worktree has no `.venv`; CPU and GPU 1 tests run with `/home/rjdollete/dev/flare-local-inference/.venv/bin/python` from the worktree. Helios unit tests: `PROTOC=~/.cache/elephant-protoc/protoc cargo test --locked --manifest-path helios/Cargo.toml --features runtime-core --target-dir ~/.cache/elephant-helios-target --lib <name>`.
- **Unverified.** Full Helios suite after `9efef537`; concurrency 4; `flare inspect`; the direct eager forward (task 4 names it, never run); parity. The 5060 Ti is in no Flare hardware platform list: serving there is a measurement, not a qualification (recorded as declined in tasks.md).
- **Superseded core** `gb202-tp1-202611161928` (20.7 GiB) is still on disk; deletion goes through the profile editor's "Delete package".
- **Baseline test failures.** 91 tests in the GDN/dense-recipe files fail on an unmodified tree on this host (missing checkpoints, `protoc`, stale `flare_heliopause`); compare against a `git archive HEAD` copy before attributing a failure.

Budget exception: the resume prompt is self-contained by the owner's standing preference, and the console package-record procedure is recorded nowhere else.
