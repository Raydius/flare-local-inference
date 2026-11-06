# Raydius fork: patch notes

This fork carries local changes on top of
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) (GitHub now redirects it to `NVIDIA/spark-local-inference`).
`deploy` (this branch) is the patch stack plus the DFlash2 work merged above it; `main` mirrors upstream unchanged.
Stack base: upstream `585159c9` (2026-11-04). Upstream tip at last check: `972e28e4` (2026-11-05), which includes upstream's own Flash Next DFlash2 (`d68ed8f5`); the next rebase folds the fork recipe onto it.
History: 2026-11-04: Rebased onto `585159c9` (60 commits: ABI 6 package-defined completion capacity, sixteen-token completions, DFlash2 compact prefill + window staging, QAD draft quantization, warmup mapping reuse #154, Selene host tier, FlashInfer pinned to `560fef16`); one conflict (`flare/src/main.rs`) kept both sides; no patch retired. 2026-11-02: Rebased onto `6fa01a49` (61 commits: Muse Spark, Qwen3.8-27B DFlash2 through shared `components/dflash2.py`, vocabulary-parallel sampling #135, collective ladder, FlashInfer floated to `@main`); two conflicts (`planning.py`, `test_models.py`) kept both sides; no patch retired.
Maintenance workflow: [FORK.md](FORK.md).

"Still needed" is judged against upstream `972e28e4`.

## Active patches

| Patch | Added | Layer | What it does | Still needed |
| --- | --- | --- | --- | --- |
| `patch/00-fork-tooling` | 2026-10-20 | — | [FORK.md](FORK.md) and `scripts/fork/{sync,rebase}.sh`: the patch-stack workflow. Never upstreamed | Yes — fork tooling, never upstreamed |
| `patch/01-generation-defaults` | 2026-10-21 | Serving | `flare run --generation-defaults <JSON>` merges sampling keys, `max_new_tokens` and `chat_template_kwargs` into the package's server config (unknown key or out-of-range value is a startup error naming it). Packages keep loading | Yes — upstream has no request-default override; its server config refuses a `generation_defaults` key |
| `patch/03-context-and-chat-template` | 2026-10-21 | Serving | `flare run --max-context-tokens <N>` lowers the per-request context (`max_sequence_tokens`); above the package's prepared context, or off its page size, is a startup error naming both. `flare run --chat-template <FILE>` replaces the package's `chat_template` asset with the file (hashed at startup); a package with no Jinja frontend is a startup error. The package file is never written | Yes — upstream has neither flag (`--kv-cache-limit context` caps KV capacity, not the per-request context) |
| `patch/06-qwen38-yarn-declaration` | 2026-10-21 | Prepared (`ignition/`) | Johannes Bertens's `ec0a64f2` ([`pr-yarn4`](https://github.com/JohannesBertens/flare-local-inference/tree/pr-yarn4)), cherry-picked unchanged: `RopeScaling` declaration (factor, original context, beta_fast/beta_slow), YaRN frequency-only interpolation of the Qwen3.8 rotary tables, the `qwen3.8-flash-yarn4` alias (1,048,576 tokens), and plan/identity/limits derived from the declared context. No Helios or package-format change | Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges |
| `patch/07-rope-parameters` | 2026-10-22 | Prepared (`ignition/cli`) | `ignition export` / `ignition debug-serve` `--rope-parameters <JSON>` applies `patch/06`'s scaling to any resolved checkpoint, as a Hugging Face / vLLM YaRN `rope_parameters` object (`factor` required; `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `max_position_embeddings` optional). `mscale` / `attention_factor` / unknown keys are refused; recipes that declare no rope scaling refuse the flag. Not forwarded to multi-host peer ranks | Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges (PR #212's `ignition bench` forwarding fix `2ca523b2` is carried on `deploy` as a separate commit above the stack, not inside `patch/07`; it retires with `patch/07`) |
| DFlash2 recipe `qwen38_dflash` (`8046cf12`, fold `df9a23d1`, `05289f11`, `cd47b4d8`) | 2026-10-31 | Prepared (`ignition/`) | Serves Qwen3.8-Flash-Next with the `incoai/Qwen3.8-Flash-Next-DFlash2` block draft (pinned `a22a1744`) in place of MTP, on the shared `components/dflash2.py`: aliases `qwen3.8-flash-dflash2` and `qwen3.8-flash-dflash2-yarn4` (1,048,576 tokens, `9f764425`), mixed prefill rounds embedding a DFlash2 draw (`890180c1`), completion records sized from the prepared draft width | On base `585159c9`, yes. Against `972e28e4`, only mixed prefill and the yarn4 alias: upstream `d68ed8f5` serves Flash Next DFlash2 itself (`--speculative-method dflash2`) and rejects mixed prefill. Fold onto it at the next rebase; [issue #210](https://github.com/NVIDIA/spark-local-inference/issues/210) asks the owner about mixed prefill |
| History drafts on the parallel-draft round (`95752d66`) | 2026-10-28 | Prepared (`ignition/components`) | `flare run --history-drafts` copies decoded-history tokens into parallel-draft (DFlash2) rounds through `substitute_copies` and the copy mask, as it already did for MTP | Yes — upstream has history drafts only for MTP rounds |
| Copy-depth acceptance attribution (`6657f7e5`) | 2026-11-04 | Serving (`helios/`) | The scheduler routes each accepted depth of a round to the copy curve when its copy bit is set, so `--history-drafts` copies no longer inflate the MTP/DFlash2 acceptance curve the packed-round allocator reads | Yes — upstream `AcceptanceProfile::observe` still takes no copy mask (an upstream defect for MTP too; no PR yet) |
| Mixed speculative export fixes (`ecb6c1dc`, `a088b0b6`, `388d3d21`) | 2026-10-31 | Prepared (`ignition/`) | Builds the packed mixed decoder without verification aliasing (`sampler=False`), declares no draft page tables on the DFlash2 speculation owner, and emits the proposal depth of mixed rounds with speculation | Yes — open upstream as [PR #211](https://github.com/NVIDIA/spark-local-inference/pull/211) |
| Opaque-write copy rule (`80946209`) | 2026-10-31 | Prepared (`ignition/preparation`) | Keeps a captured copy as an invocation when it reads opaquely written storage or its destination is mutated later | Probably not — its reproducers pass on pristine `585159c9`; retire after a re-export and served comparison (it costs an extra launch per such copy) |
| `ignition export --recipe` / `--draft-model` (`2c8ba959`) | 2026-11-02 | Prepared (`ignition/cli`) | Prepares a local checkpoint directory with a named recipe and a local draft folder | Yes — upstream has neither flag |
| FlashInfer pin `Raydius/flashinfer@49fe5449` (`8c955dc8`) | 2026-10-29 | Prepared (`pyproject.toml`) | FlashInfer main `42ba2f16` plus the fix that evicts planned Triton kernels for every JIT module and a MoE fragment guard | Yes — upstream pins `1f8ed8a3`, which has the MoE guard (#71) but not the eviction fix ([FlashInfer PR #67](https://github.com/local-inference-lab/flashinfer/pull/67), open). Re-pin to `1f8ed8a3` + #67 at the next rebase |
| DFlash2 qualification tools (`bee64b00`, `25640c8d`, `a75e845d`, `9f764425`) | 2026-10-28 | Tooling (`qualify/`) | `dflash2_acceptance.py`, `dflash2_served.py`, `dflash2_draft_sweep.py`, `dflash2_long_context.py` and their results: acceptance, served DFlash2-vs-MTP comparison, draft-count sweep, past-262k probe | Yes — fork measurement tooling, not upstreamed |

## Retired patches

| Patch | Added | Retired | What it did | Why retired |
| --- | --- | --- | --- | --- |
| `patch/02-graph-shared-memory` | 2026-10-21 | 2026-10-22 | On package load, raised a graph kernel's `MAX_DYNAMIC_SHARED_SIZE_BYTES` to the dynamic shared memory its captured node recorded. Fixed Ignition-prepared Qwen3.8-Flash packages failing `cuGraphAddKernelNode_v2` with `INVALID_VALUE` (43 prefill nodes at 57,600 B vs an exported 49,152) | Upstream `37c87e1d` does the same: it raises a function's shared-memory allowance from its captured nodes |
| `patch/04-minijinja-empty-reverse-slice` | 2026-10-21 | 2026-10-22 | Vendored minijinja 2.24.0 with one fix: reversing an empty sequence (`[][::-1]`) rendered nothing instead of panicking. A chat template that splits off system messages and reverses the rest (froggeric's Qwen-Fixed-Chat-Templates) failed every request with a system prompt | Upstream Helios now renders templates with its own Jinja VM, whose `slice` returns an empty list for `[][::-1]`; minijinja is only a test oracle |
| `patch/05-replace-cached` | 2026-10-21 | 2026-10-22 | `ignition run --replace-cached` kept only the current launch's package in the package cache, purging other snapshots and unreferenced blobs just before export | Upstream `7d3331c5` removed `ignition run` |
| `ignition run` forwarding in `patch/01` and `patch/03` | 2026-10-21 | 2026-10-22 | `ignition run` forwarded `--generation-defaults`, `--max-context-tokens` and `--chat-template` to Flare | Upstream `7d3331c5` removed `ignition run`; the `flare run` flags remain |
| FlashInfer `flashinfer-delta-prefill-source-dependencies.patch` (Ignition image recipe, `vllm-env/build/flare-ignition/`) | 2026-10-22 | 2026-10-23 | Declared `__b12x_source_dependencies__` for the delta-prefill compile factory's computed imports (FlashInfer `0765eaf1`), without which every Qwen3.8 preparation failed in "Prepare and tune kernels" | Upstream's FlashInfer pin `9c9e94ef` (Flare `c922eb07`) declares the same four dependencies |
| FlashInfer `flashinfer-qsa-singleton-default.patch` (Ignition image recipe) | 2026-10-22 | 2026-10-23 | Kept QSA at its single `warps=2` configuration at the default autotune effort, because Ignition's Qwen3.8 QSA request had no benchmark and multi-candidate preparation failed | Upstream `aca7888f` gives Ignition representative QSA tuning workloads, so the candidates race properly; keeping the patch would suppress that tuning |
| `patch/08-docker-context-templates` | 2026-10-22 | 2026-10-26 | Admitted `ignition/recipes/{qwen38,step5,kimi3}/assets/chat_template.jinja` into the Docker build context and copied them beside `helios/` in the build stage, because `helios/src/frontend/templates.rs` `include_str!`s them and upstream's `.dockerignore` excluded all of `ignition/` (since `947a5a6d`) | Upstream `230c5b40` carries the template assets in its Docker context |
| `patch/09-forward-storage-fit` | 2026-10-27 | 2026-10-28 | A cached compiled forward whose shared-storage bases needed more bytes than the live allocation raised `BindingMismatch` and recompiled, instead of failing in `as_strided` (`setStorage … out of bounds`) | Upstream `317c1ecc` binds compiled storage as regions and preserves storage mutations without cloning whole buffers, so restoring a cached forward onto a smaller allocation now succeeds: in-image on `local/flare-ignition:ec2f234d` (with the patch still applied, upstream `56f5702a`, FlashInfer `f91f257e` from `7c1a9fde`) the four-allocation cycle restores with correct outputs and recompiles `[1,1,1,1]` — the guard never fires and the patch's regression test (which asserted `[1,2,2,2]`) fails on the absence of recompiles alone |
| Warmup retention scoped to integrated GPUs (`026839f1`) | 2026-11-02 | 2026-11-04 | Kept warmup physical retention to integrated GPUs so discrete-GPU DFlash2 exports fit (export OOM after the `6fa01a49` rebase, cause upstream `a794c99d`) | Upstream #154 pools retained mappings across compiled warmup; the 149-forward export fits without it (removed in `e6fc2ef6`) |

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
  the runtime owns scheduling, cache storage, and request handling. A focused
  hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Hyperflare trades that breadth for a focused
serving contract around selected models and local workloads. Hyperflare is not, and
will never, be a replacement for production inference englines like vLLM.

Think of Hyperflare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md).
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
[Heliopause](heliopause/licenses/PyTorch.txt). The DeepSeek recipe assets retain
their [MIT license](ignition/recipes/deepseek41/assets/LICENSE). The
`heliopause/LICENSE` and `heliopause/NOTICE` symlinks refer to the root files for
Python package distribution.

Upstream chat templates are fetched with the model checkpoints and annotated
locally using bundled insertion patches. Gemma uses pinned Google sources.
The custom Kimi segmented renderer is included as project code. Downloaded
model assets and exported model cores retain their upstream terms; see
[frontend preparation](docs/model-development.md#sampling-and-frontend-metadata).
