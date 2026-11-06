# Refresh deploy's fork docs against upstream 972e28e4

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Review the current state of deploy and update documentation"

## Why

README, FORK.md and `docs/TODO.md` on `deploy` contain four statements that stopped being true on 2026-11-05. README dates every Still-needed verdict against upstream `0009b89d`; upstream is now `972e28e4`. README calls the patch/07 bench fix "pending" in a change that is archived; that fix landed only on `pr/rope-scaling`. FORK.md's build section describes an image kind and a control image that the fork no longer uses. `docs/TODO.md` says GPU execution of the yarn4 package is blocked, but the passkey runs executed it. An agent following these docs would build the wrong image and misread what the fork still carries.

## What Changes

Update the four statements so they match the repo and the host as of 2026-11-05. No verdict changes: upstream `972e28e4` still lacks `--max-context-tokens`, `--generation-defaults`, `--rope-parameters`, the yarn4 alias, `--draft-model` and a copy mask in `AcceptanceProfile::observe`, and it pins FlashInfer `1f8ed8a3`. `git diff --stat 0009b89d upstream/main -- ignition/cli/bench.py ignition/recipes/qwen38 ignition/components/dflash2.py helios/src/scheduler/speculation.rs` is empty.

## Root Cause

wrong-mental-model: the docs were written on the assumptions that upstream stood at `0009b89d`, that the bench fix would reach `deploy` through its change, and that the fork built `local/flare:<sha>` images (README.md:6, README.md:20, FORK.md:115-132, docs/TODO.md:55).

## Blast Radius

1. README.md:6 — "Upstream tip at last check: `0009b89d`".
2. README.md:10 — "'Still needed' is judged against upstream `0009b89d`".
3. README.md:20 — patch/07 row: "bench forwarding fix pending: change `rope-pr-bench-forwarding`"; `git branch --contains 2ca523b2` lists only `pr/rope-scaling`, and `patch/07-rope-parameters` is `f1ce181c`.
4. README.md:21 — DFlash2 recipe row: "Against `0009b89d`, only mixed prefill and the yarn4 alias" (the recipe paths are unchanged in `0009b89d..972e28e4`).
5. FORK.md:115-132 — "Build a local image": `local/flare:<sha>` from the repo `Dockerfile` and the control image `local/flare:e7b8c016`, which `docker images local/flare` no longer lists. Serving images are `local/flare-ignition:<sha>` from `~/LocalApps/vllm-env/build/flare-ignition/build.sh <ref>`; the running container uses `local/flare-ignition:e6fc2ef6`.
6. docs/TODO.md:55-56 — "GPU execution is blocked in this environment …", contradicted by the same entry's 2026-11-05 passkey sentence (docs/TODO.md:56-61).
7. Searched README.md, FORK.md and docs/ for `0009b89d|local/flare:|bench forwarding fix pending`: no other sites outside openspec/.
