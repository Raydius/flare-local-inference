# Keep warmup physical retention to integrated GPUs so discrete exports fit

Tier: 1 — two files edited (`capture.py` and its unit test); the other cited sites are the unchanged mechanism and its direct callers
Type: bug

External Review: Waived — Tier 1 (one condition at one call site, mechanism untouched)

## Stated Requirements

- "Determine where the rebase regression took place and implement the solution"
- "Our version didn't have this problem at all, so we should have that as a working baseline to compare it to."
- "main stays the upstream mirror"

## Root Cause

architectural-shortcoming: upstream `a794c99d` wraps the entire compiled-warmup catalog in `warmup_storage.retain_transient_backing()` (`ignition/preparation/capture.py:64`), so every transient segment freed by a warmed forward keeps its physical handle until the last forward finishes (`heliopause/src/allocator.rs:69-101`, `RetainedBacking.retain`/`finish`); the workaround targets `CUDA_ERROR_ILLEGAL_INSTRUCTION` seen only on DGX Spark (integrated memory) yet applies to every device, and on the discrete 96 GB GB202 the retained backing of 149 `qwen38_dflash` forwards exceeds the concurrency-8 mixed catalog's headroom at forward 52–70 (`openspec/debug/dflash-export-oom-in-warm-compiled-forwards.md:23`; without the scope the same export completes, entry `gb202-tp1-202611021237`).

## Blast Radius

1. `ignition/preparation/capture.py:64` — the only production caller; takes the retention scope on integrated devices and `nullcontext()` on discrete ones
2. `ignition/preparation/package_storage.py:177` — `retain_transient_backing` mechanism, unchanged; `ignition/preparation/package_storage.py:52` already exposes `integrated`, and `ignition/preparation/package_storage.py:75` and `ignition/preparation/package_storage.py:134` already branch on it
3. `tests/preparation/test_transient_retention.py:40` — capture scope test passes a `SimpleNamespace` warmup storage; gains `integrated` and a discrete case expecting no `begin`/`finish`
4. `tests/preparation/test_package_storage_cuda.py:369` — calls the mechanism directly on GPU 1 (discrete), unaffected
5. grepped 'retain_transient_backing\|_retention_scope' across the repo — no further sites
6. `heliopause/src/allocator.rs:69` — `RetainedBacking.retain`, unchanged; integrated exports still retain and release exactly as upstream validated

## Why

The first `qwen38_dflash` export after the rebase onto upstream `6fa01a49` OOMs in "Warm compiled forwards" (2/2 runs) where image `f829c06f` exported the same catalog; task 3.2 of `rebase-dflash-onto-upstream-6fa01a49` is blocked on it.

## What Changes

- `capture_forwards` enters the physical-retention scope only when `warmup_storage.integrated`; discrete devices release transient backing between warmed forwards as before `a794c99d`.
- The capture scope unit test covers both device kinds.

## Capabilities

- **New Capabilities**: none
- **Modified Capabilities**: none — preparation memory behavior, no spec-level requirement changes (`skip_specs: true`)

## Impact

`ignition/preparation/capture.py`, `tests/preparation/test_transient_retention.py`; package and serving contracts unchanged; Heliopause not rebuilt. Fix lands on `wt/dflash-s-e2vk`; `main` stays the upstream mirror.
