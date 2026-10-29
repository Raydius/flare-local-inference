# Handoff — nemotron-export-compiler-publish

## Objective and location
Fix why `ignition export` of `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` fails (`compiler completed without publishing required artifacts`), so the served qualification of archived `shared-round-history-drafts` task 1.6 can run. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk`, branch `wt/dflash-s-e2vk`. Change dir `openspec/changes/nemotron-export-compiler-publish/` (Tier 2 bug, all four planning artifacts lint-clean, uncommitted, with `openspec/debug/nemotron-export-compiler-publish.md`).

## Stage and stopping point
Apply: tasks 1.1-1.4 ticked with evidence (lint clean); 1.5 (operator leg) awaits the owner's go-ahead for GPU 0 and a committed ref for the image build. Fork fix: `Raydius/flashinfer@da45a722` on `fix/evict-planned-jits`, PR https://github.com/local-inference-lab/flashinfer/pull/67. Repin is uncommitted in the worktree (`pyproject.toml`, `uv.lock`, `docs/TODO.txt`, `tests/preparation/test_compile_pool.py`); the venv carries da45a722 via `uv pip install --no-deps` because `uv sync` cannot build `flare-heliopause` on this host. Image recipe: `~/LocalApps/vllm-env/build/flare-ignition/build.sh [ref]` archives a committed ref (`FLARE_FORK_SRC` selects the clone), so 1.5 needs the repin committed first. The C1 export finished (`exit=0`, 3159 s) and published `packages/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/gb202-tp1-202610290914`; GPU 0 is free and the Qwen instance is not running. Root cause and fix shape: proposal.md "Root Cause", design.md "Approach". Owner's standing orders: stay on this branch/worktree; no `rad-close`; the owner stops/restarts the Qwen instance (LocalAI console, port 8100, GPU 0) themselves — never `docker stop` it.

## Essential state
- 2026-10-29: `/tmp/dflash2-probe` was deleted at the owner's request. The repinned package is at `~/models/.cache/dflash2-probe/packages/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/gb202-tp1-202610291532`; its signing keys, `config.toml`, `qualify.py`, `h1_pool_publish.py`, `run-f2.sh`, and the f1/f2 export logs are in `~/models/.cache/dflash2-probe/keep/`. The `/tmp` paths below are historical.
- Fork fix lands on the owner's GitHub fork (gh account `Raydius`, read-only on `local-inference-lab/flashinfer`); sibling clone `~/dev/flashinfer` is checked out at the pinned `f91f257e`, clean.
- Verified intervention (mounted copies, not in-tree): `/tmp/dflash2-probe/nemotron/instrument/compile_plan.py` (3-line `_PLANNED_JITS` patch), `session.py`/`compile_pool.py` (diagnostics only). GPU-free reproducer: `/tmp/dflash2-probe/nemotron/h1_pool_publish.py worker-plans gated` (basis for task 1.3). Launch scripts: `run-r1.sh` (fails), `run-c1.sh` (passes).
- C1 export (container `nemotron-export-c1`, log `/tmp/dflash2-probe/nemotron/export-c1.log`, kernel cache `kernels-c1`) was still running at "Warm compiled forwards" after passing tune (2087 s) and forward compile; if it finishes it writes a package to `/tmp/dflash2-probe/nemotron/packages` that can serve task 1.6 (config profile `/tmp/dflash2-probe/nemotron/config/`, script `qualify.py`). Check `docker ps` / the log before using GPU 0; remind the owner to restart the Qwen instance when GPU 0 is free.
- Peer review certainty 0.83 (config min 0.9) is recorded verbatim in design.md; lint did not flag it.
- Host venv lacks `flare_heliopause`/io_uring; weight-loading GPU work runs in `local/flare-ignition:e8bac4a6` (flags in `run-r1.sh`). Pre-existing unrelated failures: 3 Kimi cases in `tests/recipes/test_declarations.py`.

## Resume prompt
```
Read openspec/changes/nemotron-export-compiler-publish/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, then run /rad-apply nemotron-export-compiler-publish. Stay on wt/dflash-s-e2vk; no rad-close. Before any GPU task check whether the C1 export container is still running and ask the owner before using GPU 0.
```
