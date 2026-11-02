# Handoff — qwen38-dflash-mixed-prefill

## Objective and location
Mixed prefill rounds for the `qwen38_dflash` recipe (change 3a of the DFlash2 plan). Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk`, branch `wt/dflash-s-e2vk`, Tier 2, normal lane. Commits `53729a21`, `62992f39`, `88d3c904`, `8cbb0334`, `f829c06f` on top of `248fdc56`.

## Stage and stopping point
Apply complete: all 9 boxes ticked with evidence; task 8's `operator-approval:` record (Ray Dollete, 2026-10-31, at=f829c06f…) is on disk but **uncommitted** (`tasks.md` modified). rad-simplify ran: nothing to simplify, Verify guard passed. `/rad-verify` was started and the operator interrupted the auditor dispatch before it ran; no `verification.md` exists. `radflow-lint --next` says rad-verify.

## Facts the verifier will hit
- CPU verifies: `tests/recipes -k "mixed or dflash"` 84 passed; `test_declarations.py` has 3 Kimi K3 failures needing `/data/models/Kimi-K3-MXFP4-CSF`, absent here (environmental, identical on baseline).
- GPU 1 CUDA suites were attributed per file against a clean worktree of `8cbb0334`: every failure identical on both sides (block-FP8 weights, checkpoint read mode, dense MLA, dspark projection, FP8 indexer, MSA attention, package-storage export capture refusing the 5060 Ti). `tests/preparation/test_aot.py` 32 passed with the three new reproducers.
- Served check done in Docker: image `local/flare-ignition:f829c06f`, entry `gb202-tp1-202610312330` in `~/models/.cache/flare/packages`, probes coherent, mixed counter advancing; log `~/.cache/dflash2-probe/mixed/chain.log`. Probe containers were stopped; GPU 0 is free.
- Exports, serving, and full-model qualification run only in Docker, never from the host venv or this session's cgroup. Commands: vault `Control Plane/Flare/DFlash2 Recipe Progress.md` §8 and `Two-GPU Development Workflow.md`. Memory: `exports-run-containerized`, `search-vault-before-host-work`.
- `qualify/mixed_prefill_check.py` is the real-model mixed-versus-prefill check (`--no-compile` control); the bisect script was deleted on purpose.

## Owner's verbatim words
- "In practice we do this containerized to easily contain dependencies, why aren't we doing that here?"
- "Approved" (the operator leg, criterion as written in tasks.md task 8).

## After verify
rad-sync (delta spec `specs/qwen38-dflash2-speculation/spec.md`), rad-archive, rad-close. Change 3b (served comparison vs MTP, both packages exported with `--mixed-prefill`; MTP needs a re-export) is tracked in `docs/TODO.md` and the vault progress note §7.

## Resume prompt
```
Read openspec/changes/qwen38-dflash-mixed-prefill/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, then run /rad-verify qwen38-dflash-mixed-prefill. Dispatch the fresh-context auditor synchronously; GPU work only in Docker per the handoff.
```
