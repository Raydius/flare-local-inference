# Raydius fork: patch notes

This fork carries local patches on top of
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference).
`deploy` (this branch) is the full patch stack; `main` mirrors upstream unchanged.
Stack base: upstream `a66c4e41` (2026-10-23).
Maintenance workflow: [FORK.md](FORK.md).

## Active patches

| Patch | Added | Layer | What it does |
| --- | --- | --- | --- |
| `patch/00-fork-tooling` | 2026-10-20 | — | [FORK.md](FORK.md) and `scripts/fork/{sync,rebase}.sh`: the patch-stack workflow. Never upstreamed |
| `patch/01-generation-defaults` | 2026-10-21 | Serving | `flare run --generation-defaults <JSON>` merges sampling keys, `max_new_tokens` and `chat_template_kwargs` into the package's server config (unknown key or out-of-range value is a startup error naming it). Packages keep loading |
| `patch/03-context-and-chat-template` | 2026-10-21 | Serving | `flare run --max-context-tokens <N>` lowers the per-request context (`max_sequence_tokens`); above the package's prepared context, or off its page size, is a startup error naming both. `flare run --chat-template <FILE>` replaces the package's `chat_template` asset with the file (hashed at startup); a package with no Jinja frontend is a startup error. The package file is never written |
| `patch/06-qwen38-yarn-declaration` | 2026-10-21 | Prepared (`ignition/`) | Johannes Bertens's `ec0a64f2` ([`pr-yarn4`](https://github.com/JohannesBertens/flare-local-inference/tree/pr-yarn4)), cherry-picked unchanged: `RopeScaling` declaration (factor, original context, beta_fast/beta_slow), YaRN frequency-only interpolation of the Qwen3.8 rotary tables, the `qwen3.8-flash-yarn4` alias (1,048,576 tokens), and plan/identity/limits derived from the declared context. No Helios or package-format change |
| `patch/07-rope-parameters` | 2026-10-22 | Prepared (`ignition/cli`) | `ignition export` / `ignition serve` `--rope-parameters <JSON>` applies `patch/06`'s scaling to any resolved checkpoint, as a Hugging Face / vLLM YaRN `rope_parameters` object (`factor` required; `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `max_position_embeddings` optional). `mscale` / `attention_factor` / unknown keys are refused; recipes that declare no rope scaling refuse the flag. Not forwarded to multi-host peer ranks |
| `patch/08-docker-context-templates` | 2026-10-22 | Build (`Dockerfile`, `.dockerignore`) | Upstream's `Dockerfile` no longer builds (since `947a5a6d`; still true at `a66c4e41`, whose `2de0a539` fixed only the Debian build context): `helios/src/frontend/templates.rs` `include_str!`s `ignition/recipes/{qwen38,step5}/assets/chat_template.jinja`, which the build context excluded. The context and build stage now carry those two files. Retire when upstream fixes its Dockerfile |

## Retired patches

| Patch | Added | Retired | What it did | Why retired |
| --- | --- | --- | --- | --- |
| `patch/02-graph-shared-memory` | 2026-10-21 | 2026-10-22 | On package load, raised a graph kernel's `MAX_DYNAMIC_SHARED_SIZE_BYTES` to the dynamic shared memory its captured node recorded. Fixed Ignition-prepared Qwen3.8-Flash packages failing `cuGraphAddKernelNode_v2` with `INVALID_VALUE` (43 prefill nodes at 57,600 B vs an exported 49,152) | Upstream `37c87e1d` does the same: it raises a function's shared-memory allowance from its captured nodes |
| `patch/04-minijinja-empty-reverse-slice` | 2026-10-21 | 2026-10-22 | Vendored minijinja 2.24.0 with one fix: reversing an empty sequence (`[][::-1]`) rendered nothing instead of panicking. A chat template that splits off system messages and reverses the rest (froggeric's Qwen-Fixed-Chat-Templates) failed every request with a system prompt | Upstream Helios now renders templates with its own Jinja VM, whose `slice` returns an empty list for `[][::-1]`; minijinja is only a test oracle |
| `patch/05-replace-cached` | 2026-10-21 | 2026-10-22 | `ignition run --replace-cached` kept only the current launch's package in the package cache, purging other snapshots and unreferenced blobs just before export | Upstream `7d3331c5` removed `ignition run` |
| `ignition run` forwarding in `patch/01` and `patch/03` | 2026-10-21 | 2026-10-22 | `ignition run` forwarded `--generation-defaults`, `--max-context-tokens` and `--chat-template` to Flare | Upstream `7d3331c5` removed `ignition run`; the `flare run` flags remain |
| FlashInfer `flashinfer-delta-prefill-source-dependencies.patch` (Ignition image recipe, `vllm-env/build/flare-ignition/`) | 2026-10-22 | 2026-10-23 | Declared `__b12x_source_dependencies__` for the delta-prefill compile factory's computed imports (FlashInfer `0765eaf1`), without which every Qwen3.8 preparation failed in "Prepare and tune kernels" | Upstream's FlashInfer pin `9c9e94ef` (Flare `c922eb07`) declares the same four dependencies |
| FlashInfer `flashinfer-qsa-singleton-default.patch` (Ignition image recipe) | 2026-10-22 | 2026-10-23 | Kept QSA at its single `warps=2` configuration at the default autotune effort, because Ignition's Qwen3.8 QSA request had no benchmark and multi-candidate preparation failed | Upstream `aca7888f` gives Ignition representative QSA tuning workloads, so the candidates race properly; keeping the patch would suppress that tuning |

---

<p align="center">
  <img src="docs/images/flare-logo.png" alt="Flare — ASCII wordmark above a solar flare" width="912">
</p>

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

Flare is a local inference server for NVIDIA SM120/SM121 GPUs, built for
long conversations, coding agents, and repeated tool use. Its goal is to make
the memory, bandwidth, and compute in workstations and compact systems such
as DGX Spark go further, while keeping generation responsive.

This is a research prototype, and is made available via open-source to enable
technical collaboration and collect early feedback.

The design rests on a few principles:

- **Embrace the standard PyTorch ecosystem at build time; serve from Rust.**
  Ignition uses PyTorch, Transformers, and FlashInfer/b12x to load models,
  tune kernels, compile forwards, and capture CUDA graphs into a runtime package.
  Helios, the Rust runtime, serves that package without Python. Compilation and
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
hardware, and deployment patterns. Flare trades that breadth for a focused
serving contract around selected models and local workloads. Flare is not, and
will never, be a replacement for production inference englines like vLLM.

Think of Flare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md).
See the [CLI guide](docs/cli-design.md) for configuration and the
[core design](docs/core-design.md) for the architectural reasoning.
