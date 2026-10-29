# kernel-preparation Specification

## Purpose

Kernel preparation plans and compiles every prepared program through b12x's compiler pool; this capability covers what a planned program must become once a worker compiles it.

## Requirements

### Requirement: Planned Triton programs are published by the worker that compiles them
A compiler worker that planned a compile job SHALL lower and publish that job's Triton programs to the persistent kernel cache when it later compiles the job, for every Triton kernel module, so that `compiled_program_available` is true for each program the pool reports complete.

#### Scenario: Ignition Triton kernel planned and compiled by the same worker
- **WHEN** a single-worker b12x `CompilePool` receives the `flare.gated_rms_norm` compile job through `submit` (the worker plans the job, then compiles it)
- **THEN** `pool.ready` is true for the job's programs, `compiled_program_available` is true for each, and the Triton group file for `_gated_rms_norm` exists under `TRITON_CACHE_DIR`

#### Scenario: Nemotron export prepares its kernels
- **WHEN** `ignition export` prepares `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` with compiler workers
- **THEN** "Prepare and tune kernels" completes without `compiler completed without publishing required artifacts`
