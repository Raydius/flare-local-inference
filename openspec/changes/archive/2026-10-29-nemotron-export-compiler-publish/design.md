# Design — nemotron-export-compiler-publish

## Approach

- Fork fix (branch from `local-inference-lab/flashinfer@f91f257e` in the sibling clone `~/dev/flashinfer`, pushed to the owner's GitHub fork and proposed upstream):
  - `flashinfer/experimental/b12x/_lib/compile_plan.py`: add `_PLANNED_JITS = weakref.WeakSet()` beside `_NATIVE_JITS`; in `compile_or_plan`, where planning swaps `self.compile = _plan_triton` (line 607), add the JIT to `_PLANNED_JITS`; make `evict_planning_artifacts` iterate `{*_NATIVE_JITS, *_PLANNED_JITS}` (line 129). `evict_unretained_triton` (line 143) keeps `_NATIVE_JITS` only, so retention reclaim of application kernels is unchanged. The C1 run proved this shape (`openspec/debug/nemotron-export-compiler-publish.md:24`).
  - `flashinfer/experimental/b12x/_lib/compile_pool.py` `_run_job` (after `job.invoke()`, line 414): raise if `_planning_artifacts(result)` yields any program — a result still carrying an unresolved `Deferred*Kernel` means the worker lowered nothing. Disk-cache hits return real `CompiledKernel`/`CompiledCuTeProgram` objects and pass.
  - `flashinfer/experimental/b12x/attention/mla_compress/_cute.py:279`: replace `@lru_cache(maxsize=None)` on `compile_compress` with `@program_cache` so its planning stub is swept by `evict_planning_artifacts` (bomb-squad finding 1; same class, CuTe dialect).
  - Fork tests in `tests/experimental/b12x/preparation/test_first_use_compile.py`: a non-b12x JIT (`fn.__module__ = "ignition.components.x"`) handed a planning compile is evicted; `_run_job` rejects a result holding an unresolved stub; `compile_compress` is a `program_cache` memo.
- This repository: move the pin in `pyproject.toml:25` to the fork commit and refresh `uv.lock` (`uv lock --upgrade-package flashinfer-python`, then `uv sync --locked --extra prepare --extra test`, per `docs/model-preparation.md:135-140`).
- `tests/preparation/test_compile_pool.py`: GPU-free regression test that builds a one-worker b12x `CompilePool` from a fixed device description under fresh `TRITON_CACHE_DIR`/`B12X_COMPILE_CACHE_DIR` (`tmp_path`), submits the `flare.gated_rms_norm` compile job through `submit` (worker plans, then compiles), and asserts `pool.ready`, `compiled_program_available`, and the `__grp___gated_rms_norm.json` group file. Fails on the old pin (`available [False]`, see the ledger's H1 evidence).
- `docs/TODO.txt`: follow-ups from the hunt that this change does not take: module-gated freeze guard (`compile_plan.py:548-556,587-593`); retiring the manual `_NATIVE_JITS` registrations in `ignition/components/moe_sm103.py:353,383` and `sm103_projection.py:254`; retention reclaim for non-b12x Ignition kernels (`compile_plan.py:141-156`).

## Conventions consulted

- `docs/model-preparation.md:135-140` — FlashInfer is pinned by Git revision in `pyproject.toml`; local development installs the sibling checkout; the pin and `uv.lock` move together.
- `ignition/patches/__init__.py:12-17` — Ignition's bundled-patch mechanism is Torch-Inductor only (hard check on `torch/_inductor/` paths); not extended to FlashInfer.
- AGENTS.md "Validation and performance" — tests must exercise the affected behaviour; the regression test drives a real b12x pool worker, not a mock.

## Explicit Rejections

- Register every JIT in `_NATIVE_JITS` (widen the existing registry) — rejected: `_NATIVE_JITS` also feeds `evict_unretained_triton`, which would start evicting Ignition kernels launched outside a plan and force silent recompiles (bomb-squad finding 5).
- Make `_run_job` fail when a job compiled zero programs — rejected: disk-cache hits legitimately compile nothing; the stub guard discriminates correctly.
- Extend `ignition/patches` to patch FlashInfer in place — rejected: the mechanism is scoped to Torch Inductor by design and a pin bump is the documented path.
- Ignition-side workaround (hand-register `_gated_rms_norm`, `qk_norm_rope`, `msa_attention` JITs in `_NATIVE_JITS` as `moe_sm103.py:353` does) — rejected: copies the workaround the hunt identified as the bug's fingerprint and opts those kernels into retention reclaim.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/2, escalated to tier 1)
Certainty: 0.8333333333333334
Caller Agreement: 0.6
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.7
- local-ai-qllm (qLLM) — ok, agreement 0.9
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.2
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.9

### Review Feedback

Consensus feedback recorded verbatim:
- "Add a separate planning-only WeakSet (e.g. `_PLANNED_JITS`) populated in the hooked `JITFunction.run` whenever planning returns a `DeferredTritonKernel`, regardless of `__module__`. `evict_planning_artifacts` should evict only deferred stubs from `_NATIVE_JITS | _PLANNED_JITS`, leaving real compiled kernels alone. Do not widen `_NATIVE_JITS`, and leave `evict_unretained_triton` unchanged." — ACCEPTED: the registry is populated in the `_do_compile` hook at the planning swap (the only place a stub is created), which is equivalent and narrower.
- "Add a `_run_job` guard that rejects a returned `DeferredTritonKernel` and verifies the on-disk artifact exists, but do not require lowering, since disk-cache hits are valid." — ACCEPTED in part: the stub guard is added (via `_planning_artifacts`, covering CuTe stubs too); the on-disk check stays in the parent (`session.py:2062-2075`) rather than being duplicated in the worker.
- "Regression test: a GPU-free single-worker plan-then-compile test asserting no deferred stub is returned and the on-disk artifact exists, plus a parent-side test that application kernels survive `evict_unretained_triton`." — ACCEPTED: the pool test is added here; the `evict_unretained_triton` survival test is added in the fork beside the eviction test.

## Bomb-squad findings

- latent-defect — `.venv/.../b12x/attention/mla_compress/_cute.py:279-280`: `compile_compress` is `@lru_cache`, invisible to `evict_planning_artifacts`; a worker that planned then compiles the MLA-compress job returns the CuTe stub. Taken into this change (fork edit + `_run_job` guard).
- latent-defect — `compile_pool.py:100-116` `compile_in_process` (compile_workers=0; ~35 Ignition GPU tests and `ignition/preparation/tuning.py:100`): same hole on a cold cache; covered by `_PLANNED_JITS` since the same hook runs in the parent.
- latent-defect — `compile_plan.py:548-556,587-593`: the kernel-resolution freeze guard is module-gated too, so an unprepared Ignition kernel can lower during capture without the guard firing. Not taken: recorded in `docs/TODO.txt`.
- needs-test — `compile_plan.py:141-156`: retention reclaim skips non-registered Ignition kernels (losing warp candidates stay module-loaded); three kernels are hand-registered (`ignition/components/moe_sm103.py:353,383`, `sm103_projection.py:254`). Not taken: recorded in `docs/TODO.txt`.
- confirmed-safe — separate registries (`_PLANNED_JITS` vs `_NATIVE_JITS`); Ignition call sites do not read planned stubs after planning; `_load_triton_only`/`forbid_lowering` unaffected.
- needs-test — `compile_pool.py:413-416`: a worker-side stub guard is safe and cheap (disk-cache hits never yield an unresolved stub). Taken.
