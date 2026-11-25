# Raydius fork: patch notes

This fork carries local changes as a patch stack on
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) (GitHub now
redirects it to `NVIDIA/spark-local-inference`). `deploy` (this branch) is the stack plus the
fork's serving work; `main` mirrors upstream unchanged. Model recipes, their reusable
components, and the qualification tools live in
[local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes);
the fork's recipe work is on that repository's `deploy` branch,
[Raydius/hyperflare-recipes](https://github.com/Raydius/hyperflare-recipes), which installs
this branch as its `ignition`.

The stack sits on upstream `c0c4f798` (`v0.3.2`). Each move of that base, and what the move
carried or retired, is recorded in [FORK.md](FORK.md#branch-model).

Each row below is one local change: what it does, and whether upstream still lacks it.

Published builds: the serving image `ghcr.io/raydius/hyperflare-inference`, published by
[`.github/workflows/container.yml`](.github/workflows/container.yml) from each `vX.Y.Z`
tag ([docs/container.md](docs/container.md#publish)), and the prepared Qwen 3.8 core
[`raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M`](https://huggingface.co/raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M),
whose model card names the image tag and the launch options its core is served with.
The workflow stamps the tag version into the image before building and publishes one
`X.Y.Z` tag plus one `sha-<commit>` tag per release, with no `latest`; an image tag
identifies its tagged commit, and `flare --version` alone does not identify the patch
set, so a consumer pins the tag or digest it qualified against.

"Still needed" is judged against the stack base above, and recipe-side rows in the recipe fork's notes against `local-inference-lab/hyperflare-recipes` `07d7bb39`. None of the core patches' features is present in that base (checked by `git grep` at `c0c4f798`): `--generation-defaults` and serving overrides, `RopeScaling` and `--rope-parameters`, `--recipe` / `--draft-model`, block-end markers and text prefill spans, a persisting L2 cache limit bounded by the serving device, and copy-mask attribution in `AcceptanceProfile::observe`; upstream publishes no serving image.

## Active patches

- **`patch/00-fork-tooling`** — added 2026-10-20
  - What it does: [FORK.md](FORK.md) and `scripts/fork/{sync,rebase}.sh`: the patch-stack workflow. Never upstreamed
  - **Still needed:** Yes — fork tooling, never upstreamed
- **Serving image published to GHCR** — added 2026-11-16 · Release (`.github/workflows/container.yml`)
  - What it does: On each `vX.Y.Z` tag on `deploy`, builds the repository's `Dockerfile` with the version stamped by `scripts/set_release_version.py`, pushes `ghcr.io/raydius/hyperflare-inference:X.Y.Z` and `:sha-<commit>`, then pulls the `sha-` tag back and requires `flare --version` to report the tag and `flare run --help` to list `--generation-defaults`, `--max-context-tokens` and `--chat-template`. A manual run on a version tag rebuilds that release; on a branch it pushes only `:sha-<commit>` at version `0.0.0`
  - **Still needed:** Yes — upstream publishes no serving image
- **`patch/01-generation-defaults`** — added 2026-10-21 · Serving
  - What it does: `flare run --generation-defaults <JSON>` merges sampling keys, `max_new_tokens` and `chat_template_kwargs` into the package's server config (unknown key or out-of-range value is a startup error naming it). Packages keep loading
  - **Still needed:** Yes — upstream has no request-default override; its server config refuses a `generation_defaults` key
- **`patch/03-context-and-chat-template`** — added 2026-10-21 · Serving
  - What it does: `flare run --max-context-tokens <N>` lowers the per-request context (`max_sequence_tokens`); above the package's prepared context, or off its page size, is a startup error naming both. `flare run --chat-template <FILE>` replaces the package's `chat_template` asset with the file (hashed at startup); a package with no Jinja frontend is a startup error. The package file is never written
  - **Still needed:** Yes — upstream has neither flag (`--kv-cache-limit context` caps KV capacity, not the per-request context)
- **`patch/06-qwen38-yarn-declaration`** — added 2026-10-21 · Prepared (`ignition/`)
  - What it does: Johannes Bertens's `ec0a64f2` ([`pr-yarn4`](https://github.com/JohannesBertens/flare-local-inference/tree/pr-yarn4)), cherry-picked unchanged: `RopeScaling` declaration (factor, original context, beta_fast/beta_slow), YaRN frequency-only interpolation of the Qwen3.8 rotary tables, the `qwen3.8-flash-yarn4` alias (1,048,576 tokens), and plan/identity/limits derived from the declared context. No Helios or package-format change. Since 2026-11-24 the core half (the declaration, planning and limits) is here and the recipe half (the qwen38 rotary tables and the alias) is on the recipe fork's `deploy`
  - **Still needed:** Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges
- **`patch/07-rope-parameters`** — added 2026-10-22 · Prepared (`ignition/cli`)
  - What it does: `ignition export` / `ignition debug-serve` `--rope-parameters <JSON>` applies `patch/06`'s scaling to any resolved checkpoint, as a Hugging Face / vLLM YaRN `rope_parameters` object (`factor` required; `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `max_position_embeddings` optional). `mscale` / `attention_factor` / unknown keys are refused; recipes that declare no rope scaling refuse the flag. Not forwarded to multi-host peer ranks
  - **Still needed:** Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges (PR #212's `ignition bench` forwarding fix `2ca523b2` is carried on `deploy` as a separate commit above the stack, not inside `patch/07`; it retires with `patch/07`)
- **Elephant-1 trained block mask (`5acb610d`, `6d67fd70`)** — added 2026-11-17 · Serving (`helios/`) and Prepared (`ignition/`)
  - What it does: A prepared contract may declare a block-end marker (`FrontendContract.block_end_marker`, field 11). Helios splits a prompt holding the marker once into two atomic prefill spans, serves a prompt without it causally, and refuses one holding it more than once; spans travel in the request spec, no prefix-cache key exists inside a span (a hit there falls back to an earlier checkpoint), and the end of the first block becomes a checkpoint boundary. A preempted request keeps its generated tokens in the last block. `PreparedVarlenAttention` takes optional key segment lengths. The `elephant1` recipe runs its eight full-attention layers under the checkpoint's trained two-block mask with a 16,384-token prefill chunk. Against the vendor runtime the served core agrees on 0.956 of argmax answers with a 0.021 mean probability delta (0.924 / 0.037 with every layer causal), short of the 0.01 precision bound for the packed core (`qualify/results/elephant1-decision-parity/`). `ignition debug-serve --direct` uploads no block distances and runs every prompt causally
  - **Still needed:** Yes — upstream declares spans only for image layouts on vision cores
- **Persisting L2 cache limit bounded by the serving device (`9efef537`)** — added 2026-11-16 · Serving (`helios/`)
  - What it does: A core records the exporting card's persisting L2 cache size; Helios now lowers it to the serving card's maximum, so a core exported on an RTX PRO 6000 (25,165,824 bytes) loads on an RTX 5060 Ti (20,971,520)
  - **Still needed:** Yes — upstream replays the recorded size and `flare run` fails at `cuCtxSetLimit` on the smaller card; no PR yet
- **FlashInfer `flashinfer-gdn-head-ratio.patch` (Ignition image recipe, `vllm-env/build/flare-ignition/`)** — added 2026-11-16 · Prepared (image)
  - What it does: The b12x Gated DeltaNet prefill and decode operators accept two value heads per key head as well as three. Source: local FlashInfer branch `gdn-head-ratio` (`a2f9c47a0`, on `598c8547`, re-cut 2026-11-19), unpushed; `build.sh` applies it to every image it builds, and a patch that no longer applies fails the build. The `pyproject.toml` pin does not carry it, so `elephant1` preparation fails in an environment installed from the pin alone
  - **Still needed:** Yes — until the pinned FlashInfer revision contains the change; no FlashInfer PR yet
- **Copy-depth acceptance attribution (`6657f7e5`)** — added 2026-11-04 · Serving (`helios/`)
  - What it does: The scheduler routes each accepted depth of a round to the copy curve when its copy bit is set, so `--history-drafts` copies no longer inflate the MTP/DFlash2 acceptance curve the packed-round allocator reads
  - **Still needed:** Yes — upstream `AcceptanceProfile::observe` still takes no copy mask (an upstream defect for MTP too; no PR yet)
- **`ignition export --recipe` / `--draft-model` (`2c8ba959`)** — added 2026-11-02 · Prepared (`ignition/cli`)
  - What it does: `--recipe` names the recipe that prepares a local checkpoint directory; `--draft-model` supplies a local draft folder
  - **Still needed:** Yes — upstream has neither flag. No two recipes share an architecture since `qwen38_dflash` was retired, so `--recipe` selects nothing: it confirms that the named recipe declares the directory's architecture
- **Recipe-side patches** — moved 2026-11-24 · Prepared (`Raydius/hyperflare-recipes`)
  - What it does: The fork's recipes and recipe components (Muse Glimmer, Elephant-1, DaisyRose DSpark and rope scaling, history drafts on the parallel-draft round, `--draft-model` for `qwen38` DFlash2), the FlashInfer pin and the qualification tools live on the `deploy` branch of [Raydius/hyperflare-recipes](https://github.com/Raydius/hyperflare-recipes), each listed in its README's fork notes
  - **Still needed:** Yes — the recipe fork pins this `deploy` as its `ignition` source

## Retired patches

- **Image input for the dense Qwen recipe (`1f486f63`, `751eeefc`)** — added 2026-11-19, retired 2026-11-24 · Prepared (`ignition/recipes/qwen38_dense`, `ignition/recipes/qwen38/vision.py`)
  - What it did: `qwen38_dense` prepares the checkpoint's vision encoder with the Flash Next recipe's encoder (widened for the dense model's width), gives the decoder three-axis rotary positions with a stored per-request position offset, and accepts up to four images of at most 32 MiB per request; `--language-model-only` prepares no encoder. The vision graph, upload and position computation pass on an RTX 5060 Ti; a complete export with image input, Helios loading its contract, and an image request to a served core have not run ([docs/TODO.md](docs/TODO.md))
  - **Why retired:** Upstream's #386 (`0d45b17a`) prepares the dense recipe's vision encoder with three-axis positions; the fork's two commits were dropped in the 2026-11-24 rebase and DaisyRose DSpark and its rope scaling carry on #386
- **FP8 paged prefill kernel call passes `kv_tile_ranges` (`c44217ca`)** — added 2026-11-19, retired 2026-11-24 · Prepared (`ignition/components/paged_tma.py`)
  - What it did: Passes `None` for the last launch parameter of FlashInfer's SM120 FP8 paged prefill kernel, which attends each request's whole history, as FlashInfer's own wrapper does. Without it the first execution, when compiled forwards are warmed, raises `TypeError: Expects 16 parameters`, so every dense Qwen export fails after compiling its forwards. `tests/components/test_paged_attention_tail_cuda.py` covers both prefill cases on SM120
  - **Why retired:** Upstream's `paged_tma.py` passes `None` for `kv_tile_ranges` at `c0c4f798`; the commit emptied in the 2026-11-24 rebase
- **Qwen preparation accepts the locked Transformers release** — added 2026-11-15, retired 2026-11-24 · Prepared (`ignition/recipes/qwen38`)
  - What it did: `qwen38`'s model-tree and MTP-module construction accept the `transformers==5.23.0` distribution that `pyproject.toml` locks, as well as a Git install of revision `1d9654e7`, which was the only accepted source before
  - **Why retired:** Upstream #456 (`173eb256`) requires the locked 5.23.0 release itself; the commit emptied in the 2026-11-24 rebase
- **Mixed speculative export fixes (`ecb6c1dc`, `388d3d21`)** — added 2026-10-31, retired 2026-11-19
  - What it did: Built the packed mixed decoder with verification off and `sampler=False`, because verification mode aliased the sample rows to the hidden rows and rejected the mixed round's scratch, and emitted the proposal depth of mixed rounds with speculation
  - **Why retired:** Upstream closed [PR #211](https://github.com/NVIDIA/spark-local-inference/pull/211) on 2026-11-18 as superseded: `d68ed8f5` aliases the sample rows only when their shape equals the hidden rows' (never in a mixed round, so verification mode builds no sampler and aliases nothing, which is what `sampler=False` did), and `4a24d78a` derives the draft depth of speculative mixed graphs. The depth hunk had already folded into upstream's code; the decoder flag, the mixed-round call and the depth assertions in `tests/preparation/test_execution.py` now match `ba42481a`. On the retiring commit the Qwen mixed, execution, bootstrap, speculative-state, DFlash2 and parallel-draft CPU suites pass, and on GPU 1 `tests/recipes/test_qwen38_mixed_cuda.py` passes 24 of 24. Packages exported with the patch keep loading
- **DFlash2 recipe `qwen38_dflash` (`8046cf12`, fold `df9a23d1`, `05289f11`, `cd47b4d8`)** — added 2026-10-31, retired 2026-11-10
  - What it did: Served Qwen3.8-Flash-Next with the `incoai/Qwen3.8-Flash-Next-DFlash2` block draft through a recipe of its own: aliases `qwen3.8-flash-dflash2` and `qwen3.8-flash-dflash2-yarn4`, a pin of `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2`, mixed prefill rounds embedding a DFlash2 draw, and history drafts on its rounds
  - **Why retired:** Upstream's `qwen38` recipe serves the same draft (`--speculative-method dflash2`). Measured at 16 sequences and a 1,048,576-token context on one RTX PRO 6000 (`qualify/results/dflash2-paths-c16/README.md`), the fork recipe reached 535.1 and 355.6 `code_edit` tok/s at concurrency 1 and 8 against 555.9 and 436.7 for upstream's path with the MXFP8 draft, held 909,312 KV tokens (the server refused a 1M prompt), and exported in 92 minutes against 28. Decode throughput during concurrent long prefill was not measured and is not a workload this fork needs. Lost with it: mixed prefill and history drafts for Qwen 3.8 DFlash2 packages, the QAD pin (that repository id now resolves to `qwen38` with MTP proposals unless `dflash2` is given), the tools `dflash2_tap_check.py` and `mixed_prefill_check.py`, and the CUDA parity check of the shared draft component against the reference on the pinned Flash-Next draft. Packages exported with the recipe keep loading
  - The hooks the fork had added to `ignition/recipes/qwen38/` (`mtp=` and `draft_windows=` in `state.py`, feature taps in `decoder.py`, feature buffers in `mixed_state.py`, `feedback=` in `speculative_state.py`) and the `"dflash"` projector entry in `ignition/components/dflash2_weights.py` are gone (2026-11-21): upstream rejected DFlash2 drafts inside mixed prefill rounds, so mixed rounds are MTP-only and DFlash2 serves with separate prefill and verification graphs. The `admission` and `draft_page_blocks` parameters of `PreparedParallelDraftRound`, their test, and the absorption and speculative-round hooks in `mixed_model.py` were removed with the move to upstream `6b68b9b4`, whose functional draft round cannot run them
- **Opaque-write copy rule (`80946209`)** — added 2026-10-31, retired 2026-11-10
  - What it did: Kept a captured copy as an invocation when it read storage an earlier opaque call wrote or its destination was written again later
  - **Why retired:** On upstream `e774a468` its five reproducers in `tests/preparation/test_aot.py` (opaque writes before ordinary reads, counts and ids written by copies for opaque readers, deferred-then-opaque ids, opaque partial writes feeding a count) and a case whose copy destination is read only after the graph all pass on GPU without the rule; upstream `fa0eadad` keeps the first group's copies in order, and the post-graph case is deferred by upstream and still publishes the opaque write. The reproducers stay as regression tests
- **FlashInfer MoE fragment guard (`49fe5449`, in the `Raydius/flashinfer` pin)** — added 2026-10-29, retired 2026-11-10
  - What it did: Flattened the dynamic MoE kernel's direct SFB fragment only when it was still nested, so `moe.decode` tuning stopped failing with `TypeError: Value after * must be an iterable, not int`
  - **Why retired:** FlashInfer #71 (`e239dfb3a`) guards both flatten sites and is in upstream's pin `85ea6e76`
- **`patch/02-graph-shared-memory`** — added 2026-10-21, retired 2026-10-22
  - What it did: On package load, raised a graph kernel's `MAX_DYNAMIC_SHARED_SIZE_BYTES` to the dynamic shared memory its captured node recorded. Fixed Ignition-prepared Qwen3.8-Flash packages failing `cuGraphAddKernelNode_v2` with `INVALID_VALUE` (43 prefill nodes at 57,600 B vs an exported 49,152)
  - **Why retired:** Upstream `37c87e1d` does the same: it raises a function's shared-memory allowance from its captured nodes
- **`patch/04-minijinja-empty-reverse-slice`** — added 2026-10-21, retired 2026-10-22
  - What it did: Vendored minijinja 2.24.0 with one fix: reversing an empty sequence (`[][::-1]`) rendered nothing instead of panicking. A chat template that splits off system messages and reverses the rest (froggeric's Qwen-Fixed-Chat-Templates) failed every request with a system prompt
  - **Why retired:** Upstream Helios now renders templates with its own Jinja VM, whose `slice` returns an empty list for `[][::-1]`; minijinja is only a test oracle
- **`patch/05-replace-cached`** — added 2026-10-21, retired 2026-10-22
  - What it did: `ignition run --replace-cached` kept only the current launch's package in the package cache, purging other snapshots and unreferenced blobs just before export
  - **Why retired:** Upstream `7d3331c5` removed `ignition run`
- **`ignition run` forwarding in `patch/01` and `patch/03`** — added 2026-10-21, retired 2026-10-22
  - What it did: `ignition run` forwarded `--generation-defaults`, `--max-context-tokens` and `--chat-template` to Flare
  - **Why retired:** Upstream `7d3331c5` removed `ignition run`; the `flare run` flags remain
- **FlashInfer `flashinfer-delta-prefill-source-dependencies.patch` (Ignition image recipe, `vllm-env/build/flare-ignition/`)** — added 2026-10-22, retired 2026-10-23
  - What it did: Declared `__b12x_source_dependencies__` for the delta-prefill compile factory's computed imports (FlashInfer `0765eaf1`), without which every Qwen3.8 preparation failed in "Prepare and tune kernels"
  - **Why retired:** Upstream's FlashInfer pin `9c9e94ef` (Flare `c922eb07`) declares the same four dependencies
- **FlashInfer `flashinfer-qsa-singleton-default.patch` (Ignition image recipe)** — added 2026-10-22, retired 2026-10-23
  - What it did: Kept QSA at its single `warps=2` configuration at the default autotune effort, because Ignition's Qwen3.8 QSA request had no benchmark and multi-candidate preparation failed
  - **Why retired:** Upstream `aca7888f` gives Ignition representative QSA tuning workloads, so the candidates race properly; keeping the patch would suppress that tuning
- **`patch/08-docker-context-templates`** — added 2026-10-22, retired 2026-10-26
  - What it did: Admitted `ignition/recipes/{qwen38,step5,kimi3}/assets/chat_template.jinja` into the Docker build context and copied them beside `helios/` in the build stage, because `helios/src/frontend/templates.rs` `include_str!`s them and upstream's `.dockerignore` excluded all of `ignition/` (since `947a5a6d`)
  - **Why retired:** Upstream `230c5b40` carries the template assets in its Docker context
- **`patch/09-forward-storage-fit`** — added 2026-10-27, retired 2026-10-28
  - What it did: A cached compiled forward whose shared-storage bases needed more bytes than the live allocation raised `BindingMismatch` and recompiled, instead of failing in `as_strided` (`setStorage … out of bounds`)
  - **Why retired:** Upstream `317c1ecc` binds compiled storage as regions and preserves storage mutations without cloning whole buffers, so restoring a cached forward onto a smaller allocation now succeeds: in-image on `local/flare-ignition:ec2f234d` (with the patch still applied, upstream `56f5702a`, FlashInfer `f91f257e` from `7c1a9fde`) the four-allocation cycle restores with correct outputs and recompiles `[1,1,1,1]` — the guard never fires and the patch's regression test (which asserted `[1,2,2,2]`) fails on the absence of recompiles alone
- **Warmup retention scoped to integrated GPUs (`026839f1`)** — added 2026-11-02, retired 2026-11-04
  - What it did: Kept warmup physical retention to integrated GPUs so discrete-GPU DFlash2 exports fit (export OOM after the `6fa01a49` rebase, cause upstream `a794c99d`)
  - **Why retired:** Upstream #154 pools retained mappings across compiled warmup; the 149-forward export fits without it (removed in `e6fc2ef6`)

---

<p align="center">
  <img src="docs/images/hyperflare.png" alt="Flare — ASCII wordmark above a supermassive solar flare" width="912">
</p>

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

Hyperflare is a local inference server for NVIDIA SM120/SM121 GPUs, built for
long conversations, coding agents, and repeated tool use. Its goal is to make
the memory, bandwidth, and compute in workstations and compact systems such
as DGX Spark go further, while keeping generation responsive.

This is a research prototype, and is made available via open-source to enable
technical collaboration and collect early feedback.

The design rests on a few principles:

- **Embrace the standard PyTorch ecosystem at build time; serve from Rust.**
  Ignition uses PyTorch, Transformers, and FlashInfer/b12x to load models,
  tune kernels, compile forwards, and capture CUDA graphs into a runtime core.
  Helios, the Rust runtime, serves that core without Python. Compilation and
  graph capture finish before serving.

- **Build on the model ecosystem.** Reuse canonical model implementations
  wherever their dataflow fits, and specialize the parts that need different
  layouts or execution. Model development keeps the tools and libraries that
  make the Python ecosystem productive.

- **Make memory and state explicit.** Plan buffers and execution capacities
  ahead of time. Treat conversation state—including recurrent state alongside
  KV caches—as a whole, so reuse, branching, and resuming preserve it correctly.

- **Keep the runtime small and shared.** Model recipes own model computation;
  the runtime owns scheduling, cache storage, and request handling. Core
  Ignition in this repository provides the recipe-free preparation library and
  CLI runner; official model recipes and reusable b12x model components live in
  [local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes).
  A focused hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Hyperflare trades that breadth for a focused
serving contract around selected models and local workloads. Hyperflare is not, and
will never be, a replacement for production inference engines like vLLM.

Think of Hyperflare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md) and the
official model recipes in
[local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes).
For native Windows development and serving prebuilt model cores, follow
[Windows serving](docs/windows.md#build-and-test-on-windows).
See the [CLI guide](docs/cli-design.md) for configuration and the
[core design](docs/core-design.md) for the architectural reasoning.

## Licensing and contributions

Hyperflare's project code is licensed under [Apache 2.0](LICENSE). Contributions
require the sign-off described in [CONTRIBUTING.md](CONTRIBUTING.md).
Third-party attributions are listed in [NOTICE](NOTICE); upstream terms remain
applicable to third-party code and assets.

The PyTorch-derived image resizing implementation in `helios/` retains the
[PyTorch BSD-style license](helios/licenses/PyTorch.txt), also distributed with
[Heliopause](heliopause/licenses/PyTorch.txt). The DeepSeek recipe assets in
[local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes)
retain their MIT license. The
`heliopause/LICENSE` and `heliopause/NOTICE` symlinks refer to the root files for
Python package distribution.

Upstream chat templates are fetched with the model checkpoints and annotated
locally using bundled insertion patches. Gemma uses pinned Google sources.
The custom Kimi segmented renderer is included as project code. Downloaded
model assets and exported model cores retain their upstream terms; see
[frontend preparation](docs/model-development.md#sampling-and-frontend-metadata).
