# Evict planned Triton kernels from compiler workers for every JIT module

Tier: 2
Type: bug

## Stated Requirements

- "run /rad-debug nemotron-export-compiler-publish on the Nemotron export failure (`compiler completed without publishing required artifacts`, flare.gated_rms_norm)"
- "Reproduce from /tmp/dflash2-probe/nemotron/export.log first"
- "GPU 0 is held by the Qwen instance on port 8100, so ask before stopping it"
- "I'll stop it myself" / "It is stopped now" — the owner stops and restarts the Qwen instance; the agent never does
- "Stay on wt/dflash-s-e2vk; no rad-close"

## Root Cause

silent-failure: a compiler worker that planned a job keeps the planning `DeferredTritonKernel` in the JIT's `device_caches`, because `run_observed` registers a JIT for eviction only when `fn.__module__` is `b12x` or `b12x.*` (`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:572-576`) and `evict_planning_artifacts` sweeps only that registry (`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:129`); every Ignition Triton kernel (`ignition.components.gated_norm._gated_rms_norm` and the other `flare.*` families) is skipped. When the same worker later runs the job, Triton's in-memory cache returns the deferred kernel, nothing is lowered or written, and `_run_job` still reports the planned keys complete because it only compares key sets (`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_pool.py:405-440`); the parent then finds no artifact on disk (`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/preparation/session.py:2062-2075`). Confirmed in `openspec/debug/nemotron-export-compiler-publish.md:23` (H1) and `:24` (C1: the export passes "Prepare and tune kernels" with planned JITs added to the sweep; fails without).

## Blast Radius

1. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:572-576` — `run_observed` populates `_NATIVE_JITS` by module name; the registry the fix widens (grepped '_NATIVE_JITS' in the fork clone `~/dev/flashinfer` at pinned `f91f257e` — same four hits, lines 57/129/143/576, file byte-identical to the installed copy).
2. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:129` — `evict_planning_artifacts` sweeps `_NATIVE_JITS` only; must also sweep JITs that planning handed `_plan_triton` (`.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:607-608`).
3. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_plan.py:143` — `evict_unretained_triton` reuses `_NATIVE_JITS` for parent-side reclaim of unretained kernels; untouched, so Ignition kernels keep their current reclaim behaviour.
4. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_pool.py:454-465` — `_describe_jobs` calls the sweep in the worker after planning; the site where the defect manifests.
5. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_pool.py:100-118` — `compile_in_process` (no compiler workers) calls the same sweep before compiling in-process; same defect, untested path.
6. `.venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/_lib/compile_pool.py:405-440` — `_run_job` accepts a result that lowered nothing; left as is, the artifact check in the parent is the guard.
7. Ignition families planned through `CompileJob` with non-b12x Triton kernels: `ignition/components/gated_norm_preparation.py:82`, `ignition/components/qk_norm_rope.py:146`, `ignition/components/msa_attention.py:404`, `ignition/components/moe_sm103.py:396,541,543`, `ignition/components/sm103_projection.py:497` — grepped `CompileJob.create` across `ignition/`; these five files also import Triton.
8. grepped `@triton.jit` across `.venv/.../b12x` — every hit is a `b12x.*` module (already in `_NATIVE_JITS`); grepped `from flashinfer` across `b12x` — no Triton kernel imported from outside `b12x`, so b12x's own families are unaffected.
9. `pyproject.toml:25` and `uv.lock:401-403` — the FlashInfer pin (`local-inference-lab/flashinfer@f91f257e`) that must move to a commit carrying the fix; `docs/model-preparation.md:135-140` describes the repin procedure.
10. grepped `b12x._lib.compile_pool` across `tests/` — no hits (only `tests/components/test_collective_owner_order_cuda.py` imports `b12x._lib.compile_plan`); `tests/preparation/test_compile_pool.py:1-30` tests Ignition's wrapper only, so no test drives a b12x `CompilePool` job through plan-then-compile in one worker (the shape of `/tmp/dflash2-probe/nemotron/h1_pool_publish.py`).
11. grepped `evict_planning_artifacts|_NATIVE_JITS|plan_compilations` across the fork clone's `tests/` — hits in `tests/experimental/b12x/preparation/test_first_use_compile.py`, `test_parallel_discovery.py`, `test_compile_keys_ignore_device_identity.py`, `tests/experimental/b12x/comm/test_pcie_preparation_compile.py`; none plans a non-b12x JIT.

## Why

`ignition export` of `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` fails deterministically at "Prepare and tune kernels" on this host, which blocks the served qualification of the archived `shared-round-history-drafts` change (task 1.6) and the DFlash2 recipe work behind it.

## What Changes

- In the pinned b12x fork, record every JIT that planning hands a deferred kernel and have `evict_planning_artifacts` sweep those JITs as well as `_NATIVE_JITS`, so a worker (or the in-process path) always lowers and publishes a planned program it later compiles.
- Move this repository's FlashInfer pin to the commit carrying that fix (`pyproject.toml`, `uv.lock`).
- Add a GPU-free Ignition test that submits a `flare.gated_rms_norm` compile job to a single-worker b12x `CompilePool` and asserts the published artifact is available.

## Capabilities

### New Capabilities
- `kernel-preparation`: programs planned by a compiler worker are published to the persistent cache when that worker compiles them, for every Triton kernel module.

### Modified Capabilities
- none

## Impact

- Dependencies: FlashInfer pin bump; the fix is committed on a fork of `local-inference-lab/flashinfer` (the owner has read-only access upstream) and proposed upstream as a PR; the pin returns to `local-inference-lab` once merged.
- Code: `tests/preparation/test_compile_pool.py` (new test); no Ignition source change.
- Workflows: every `ignition export`/`debug-serve` preparation that plans a `flare.*` Triton family through the compiler pool; the Nemotron export package for task 1.6.
