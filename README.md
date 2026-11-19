# Raydius fork: patch notes

This fork carries local changes on top of
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) (GitHub now redirects it to `NVIDIA/spark-local-inference`).
`deploy` (this branch) is the patch stack plus the DFlash2 and Elephant-1 work merged above it; `main` mirrors upstream unchanged.
Stack base: upstream `1a532755` (2026-11-18). Upstream has carried its own Flash Next DFlash2 (`d68ed8f5`, `--speculative-method dflash2`) since 2026-11-04; the fork's own recipe for it, `qwen38_dflash`, was retired on 2026-11-10 and Qwen 3.8 DFlash2 packages are exported through upstream's path.
History: 2026-11-18: Rebased onto `1a532755` (81 commits: #289 bounds history proposals by logical lane length, checkpoint-work estimation without quoting snapshot storage and quoted-work sizing of ordinary and mixed prefill graphs, `80c97c50` passes mutated dtype-viewed temporaries to storage calls through their owner, 16-bit PNG and GIF input with Pillow-equivalent conversion, the Selene pool series, FlashInfer re-pinned to `2cce3e67`, design checklist digest `1af4`); five commits conflicted and kept both sides on upstream's structure — the FlashInfer pin, the `docs/TODO.txt` rename with upstream's four new entries folded into `docs/TODO.md`, the opaque-write copy pair retired against upstream's `_lower_storage` rewrite with its reproducers kept, and the code_edit benchmark where the fork's file already subsumes upstream's `be4ff01e` — one commit dropped with its `AGENTS.md` link subsumed by the rename's resolution, upstream's #289 made the fork's `packing.rs` lane-length change independently so only the fork's comment remains, and the FlashInfer pin was re-cut on `2cce3e67` as `Raydius/flashinfer@be90bff55`, carrying only stub rejection and the `compile_compress` program cache; no patch retired. 2026-11-15: Rebased onto `6b68b9b4` (75 commits: KV and capacity fixes #253, #286 and #285, native MXFP8 DFlash2 draft bodies (#291), the one-layout DFlash2 embedding binding (#250), Inductor-planned model temporaries with forwards that take their metadata per call, Inkling 2 Small NVFP4, prefix-cache reuse under resource pressure (#298), least-recently-used host prefix reclaim (#302), FlashInfer pinned to `4fd836dc`, design checklist digest `ed22`); 27 commits conflicted and kept both sides on upstream's structure, Muse Glimmer's image-preparation number moved from 10 to 11 beside Inkling's padded grid, Muse Glimmer moved onto the per-call forwards with its own attention module, the retired `qwen38_dflash` recipe's mixed-round hooks were dropped, and no patch retired. 2026-11-10: Rebased onto `e774a468` (48 commits: the terminal binary moved to `flare-cli/` over a shared `flare` library with a Windows desktop, packages renamed cores (`runtime_core`, `core_storage`), Kimi K3.1 and its DFlash2, exports that retain no CUDA graph templates, `--mtp-only`, `--force-moe-decode-a16`, FlashInfer pinned to `85ea6e76`); twelve commits conflicted and kept both sides on upstream's structure, one carry commit follows the renames, the opaque-write copy rule and the FlashInfer MoE fragment guard retired. 2026-11-04: Rebased onto `585159c9` (60 commits: ABI 6 package-defined completion capacity, sixteen-token completions, DFlash2 compact prefill + window staging, QAD draft quantization, warmup mapping reuse #154, Selene host tier, FlashInfer pinned to `560fef16`); one conflict (`flare/src/main.rs`) kept both sides; no patch retired. 2026-11-02: Rebased onto `6fa01a49` (61 commits: Muse Spark, Qwen3.8-27B DFlash2 through shared `components/dflash2.py`, vocabulary-parallel sampling #135, collective ladder, FlashInfer floated to `@main`); two conflicts (`planning.py`, `test_models.py`) kept both sides; no patch retired.
Maintenance workflow: [FORK.md](FORK.md).

Published builds: the serving image `ghcr.io/raydius/hyperflare-inference:<version>`, built from this branch by `.github/workflows/container.yml` on each `vX.Y.Z` tag ([docs/container.md](docs/container.md#publish)), and the prepared Qwen 3.8 core [`raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M`](https://huggingface.co/raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M), which runs on that image.

"Still needed" is judged against upstream `1a532755` (2026-11-18), the stack base. None of the active patches' features is present there: `--generation-defaults` and serving overrides, `RopeScaling` and `--rope-parameters`, the `muse_glimmer_nvfp4` recipe, history-draft copies on parallel-draft rounds, mixed speculative rounds exported without a sampler, `--recipe` / `--draft-model`, the `elephant1` recipe, block-end markers and text prefill spans, and a persisting L2 cache limit bounded by the serving device.

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
  - What it does: Johannes Bertens's `ec0a64f2` ([`pr-yarn4`](https://github.com/JohannesBertens/flare-local-inference/tree/pr-yarn4)), cherry-picked unchanged: `RopeScaling` declaration (factor, original context, beta_fast/beta_slow), YaRN frequency-only interpolation of the Qwen3.8 rotary tables, the `qwen3.8-flash-yarn4` alias (1,048,576 tokens), and plan/identity/limits derived from the declared context. No Helios or package-format change
  - **Still needed:** Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges
- **`patch/07-rope-parameters`** — added 2026-10-22 · Prepared (`ignition/cli`)
  - What it does: `ignition export` / `ignition debug-serve` `--rope-parameters <JSON>` applies `patch/06`'s scaling to any resolved checkpoint, as a Hugging Face / vLLM YaRN `rope_parameters` object (`factor` required; `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `max_position_embeddings` optional). `mscale` / `attention_factor` / unknown keys are refused; recipes that declare no rope scaling refuse the flag. Not forwarded to multi-host peer ranks
  - **Still needed:** Yes — until [PR #212](https://github.com/NVIDIA/spark-local-inference/pull/212) merges (PR #212's `ignition bench` forwarding fix `2ca523b2` is carried on `deploy` as a separate commit above the stack, not inside `patch/07`; it retires with `patch/07`)
- **Muse Glimmer recipe `muse_glimmer_nvfp4` (`9834def8`, `85463f36`, `37f3613d`, `c4f4fa7c`, `8fe522a3`, `d77a2055`, `906ee060`)** — added 2026-11-06 · Prepared (`ignition/recipes`)
  - What it does: Serves `original-el8/Muse-Glimmer-30B-NVFP4` (alias `muse-glimmer-30b-nvfp4`) on one GPU at TP1: W4A4 NVFP4 decoder projections with a BF16 head, up to four images per request, eight concurrent requests by default and sixteen at `--max-concurrency 16` with 128 packed verification rows (served at 16 on one RTX PRO 6000: 57.3 GiB of KV, 9,076,736 tokens), and a declared 1,048,576-token context over the native 131,072 (no rope scaling; the sliding layers stay bounded at 2,048). Drafting uses the `z-lab/Muse-Glimmer-30B-DFlash2` block drafter (pinned `b54ffdd1`) at 15 draft tokens; `ignition export --draft-model` supplies the drafter offline. Passkey retrieval at 1M: 14 of 15 cells within a 256-token cap, the 15th at 1,024
  - On `1a532755`: the recipe's image profile is `ImagePreparationKind` 11 and `ImageSettings` field 11, because upstream's Inkling padded grid holds 10. A Glimmer core exported before 2026-11-15 records 10; a hand-run `flare run` of it on a binary built from this base is not rejected, reads the profile as the padded grid, and gives invalid image results, so such a core stays on the image that exported it. The text and draft forwards take their metadata per call: Glimmer's attention is its own module (`muse_glimmer_nvfp4/attention.py`), `muse_spark/attention.py` is upstream's text, and the draft path uses upstream's shared `DFlash2Preparation`. The recipe's tests in `tests/recipes/muse_glimmer_nvfp4` pass on SM120 on this base: attention parity under graph replay with synthetic weights, and, against the checkpoint and its DFlash2 drafter on a 96 GB GB202, vision parity, the text reference comparison and the width-16 draft round
  - **Still needed:** Yes — until [PR #229](https://github.com/NVIDIA/spark-local-inference/pull/229) merges (it carries the recipe rebased on `972e28e4`, without this fork's compilation lowering, the renumbering, or the per-call forwards)
- **Elephant-1 recipe `elephant1` (`2e8f5d58`, `c2f2d052`, `e65c6c30`, `8797f6db`)** — added 2026-11-16 · Prepared (`ignition/recipes`, `ignition/components`)
  - What it does: Serves `baselabs/elephant-1` (alias `elephant-1-9b`, served name `Elephant-1`), a text-only Qwen3.5-9B decision model, inside 12 GB: preparation packs the BF16 body projections to NVFP4 W4A16 (`pack_body_nvfp4` in `ignition/components/weights.py`) and keeps the embedding, head, norms and Gated DeltaNet scalars in BF16; the 333 vision tensors are skipped and image inputs refused. Context 16,384 tokens, concurrency 2. A decision is read with Completions `max_tokens: 1` and `logprob_token_ids` naming the option letters. The shared Gated DeltaNet buffers and the dense Qwen decoder take their widths and query format from each model's head counts and cache. Preparation needs the FlashInfer GDN head-ratio patch below. Core `gb202-tp1-202611171634` (exported on GB202, see the next entry) serves on an RTX 5060 Ti in 10,428 MiB; that card is in no Flare hardware platform list
  - **Still needed:** Yes — upstream has no Elephant-1 recipe; its Gated DeltaNet buffers assumed 48 value heads
- **Elephant-1 trained block mask (`5acb610d`, `6d67fd70`)** — added 2026-11-17 · Serving (`helios/`) and Prepared (`ignition/`)
  - What it does: A prepared contract may declare a block-end marker (`FrontendContract.block_end_marker`, field 11). Helios splits a prompt holding the marker once into two atomic prefill spans, serves a prompt without it causally, and refuses one holding it more than once; spans travel in the request spec, no prefix-cache key exists inside a span (a hit there falls back to an earlier checkpoint), and the end of the first block becomes a checkpoint boundary. A preempted request keeps its generated tokens in the last block. `PreparedVarlenAttention` takes optional key segment lengths. The `elephant1` recipe runs its eight full-attention layers under the checkpoint's trained two-block mask with a 16,384-token prefill chunk. Against the vendor runtime the served core agrees on 0.956 of argmax answers with a 0.021 mean probability delta (0.924 / 0.037 with every layer causal), short of the 0.01 precision bound for the packed core (`qualify/results/elephant1-decision-parity/`). `ignition debug-serve --direct` uploads no block distances and runs every prompt causally
  - **Still needed:** Yes — upstream declares spans only for image layouts on vision cores
- **Persisting L2 cache limit bounded by the serving device (`9efef537`)** — added 2026-11-16 · Serving (`helios/`)
  - What it does: A core records the exporting card's persisting L2 cache size; Helios now lowers it to the serving card's maximum, so a core exported on an RTX PRO 6000 (25,165,824 bytes) loads on an RTX 5060 Ti (20,971,520)
  - **Still needed:** Yes — upstream replays the recorded size and `flare run` fails at `cuCtxSetLimit` on the smaller card; no PR yet
- **FlashInfer `flashinfer-gdn-head-ratio.patch` (Ignition image recipe, `vllm-env/build/flare-ignition/`)** — added 2026-11-16 · Prepared (image)
  - What it does: The b12x Gated DeltaNet prefill and decode operators accept two value heads per key head as well as three. Source: local FlashInfer branch `gdn-head-ratio` (`2f55cfd98`, on `7245b1e0b`), unpushed; `build.sh` applies it to every image it builds, and a patch that no longer applies fails the build. The `pyproject.toml` pin does not carry it, so `elephant1` preparation fails in an environment installed from the pin alone
  - **Still needed:** Yes — until the pinned FlashInfer revision contains the change; no FlashInfer PR yet
- **History drafts on the parallel-draft round (`95752d66`)** — added 2026-10-28 · Prepared (`ignition/components`)
  - What it does: `flare run --history-drafts` copies decoded-history tokens into parallel-draft (DFlash2) rounds through `substitute_copies` and the copy mask, as it already did for MTP. A/B at fixed-7 on a Qwen3.8 DFlash2 package of the retired `qwen38_dflash` recipe (2026-11-06, `qualify/results/dflash2-history-drafts/`, `2e91f176`): off 497.7 / 346.4 tok/s, on 380.7 / 341.6 tok/s (c1 / c8 `code_edit` medians; 52 % of copies accepted)
  - **Still needed:** Yes — upstream has history drafts only for MTP rounds. Upstream's `qwen38` DFlash2 path declares no copy inputs, so no Qwen 3.8 DFlash2 package takes the flag since `qwen38_dflash` was retired
- **Copy-depth acceptance attribution (`6657f7e5`)** — added 2026-11-04 · Serving (`helios/`)
  - What it does: The scheduler routes each accepted depth of a round to the copy curve when its copy bit is set, so `--history-drafts` copies no longer inflate the MTP/DFlash2 acceptance curve the packed-round allocator reads
  - **Still needed:** Yes — upstream `AcceptanceProfile::observe` still takes no copy mask (an upstream defect for MTP too; no PR yet)
- **Mixed speculative export fixes (`ecb6c1dc`, `388d3d21`)** — added 2026-10-31 · Prepared (`ignition/`)
  - What it does: Builds the packed mixed decoder without verification aliasing (`sampler=False`) and emits the proposal depth of mixed rounds with speculation
  - **Still needed:** Yes — open upstream as [PR #211](https://github.com/NVIDIA/spark-local-inference/pull/211)
- **`ignition export --recipe` / `--draft-model` (`2c8ba959`)** — added 2026-11-02 · Prepared (`ignition/cli`)
  - What it does: `--recipe` names the recipe that prepares a local checkpoint directory; `--draft-model` supplies a local draft folder
  - **Still needed:** Yes — upstream has neither flag. No two recipes share an architecture since `qwen38_dflash` was retired, so `--recipe` selects nothing: it confirms that the named recipe declares the directory's architecture
- **`--draft-model` for `qwen38` DFlash2 proposals** — added 2026-11-10 · Prepared (`ignition/`)
  - What it does: `ignition export --recipe qwen38` with speculative method `dflash2` loads the DFlash2 draft from the `--draft-model` folder (`config.json`, `model.safetensors`) and downloads the pinned `incoai/Qwen3.8-Flash-Next-DFlash2` snapshot only without the flag. With MTP proposals the flag is still refused ("loads no external draft")
  - **Still needed:** Yes — upstream's `qwen38` DFlash2 path reads the draft only from the hub cache, so an offline export or a draft other than the pin needs the folder
- **FlashInfer pin `Raydius/flashinfer@be90bff55`** — added 2026-10-29, re-cut 2026-11-15 and 2026-11-18 · Prepared (`pyproject.toml`)
  - What it does: Upstream's pin `local-inference-lab/flashinfer@2cce3e67` plus one commit (branch `evict-planned-jits-2cce3e67`): a compiler worker rejects a job whose result is an unresolved planning stub, and `compile_compress` memoizes through the program cache so its kernels are evicted with the rest. Upstream's own planned-JIT registry (`_PLANNED_TRITON_JITS`), which since `4fd836dc` also covers application (non-b12x) JITs, replaces the fork's registry, which is no longer carried
  - **Still needed:** Yes — on unmodified `2cce3e67` the eviction case for application JITs passes, and the stub-rejection and `compile_compress` cases fail; both pass with the commit ([FlashInfer PR #67](https://github.com/local-inference-lab/flashinfer/pull/67), open, head `7245b1e0` since 2026-11-15; the `2cce3e67` re-cut is not pushed there)
- **Qwen preparation accepts the locked Transformers release** — added 2026-11-15 · Prepared (`ignition/recipes/qwen38`)
  - What it does: `qwen38`'s model-tree and MTP-module construction accept the `transformers==5.23.0` distribution that `pyproject.toml` locks, as well as a Git install of revision `1d9654e7`, which was the only accepted source before
  - **Still needed:** Yes — at `1a532755` upstream locks the 5.23.0 release while `qwen38/model.py` and `qwen38/mtp_weights.py` still require the Git revision, so `ignition export` of a Qwen3.8 checkpoint from the locked environment stops with "Qwen preparation requires Transformers"
- **Elephant-1 qualification tools** — added 2026-11-16 · Tooling (`qualify/`)
  - What it does: `elephant1_decision_parity.py` renders 160 decision probes by the vendor rules and compares a served core with the vendor runtime (OneJev `81ce62f`) in two stages, the mask and the storage precision; `--masks` scores mask ablations, and `core --reverse` with `report --recached` compares answers from an empty prefix cache with cached ones. `elephant1_block_attention_probe.py` checks the dense noncausal attention operator at Elephant-1's geometry and the prefill storage arithmetic. Results in `qualify/results/elephant1-decision-parity/` and `qualify/results/elephant1-mask-feasibility/`
  - **Still needed:** Yes — fork measurement tooling, not upstreamed
- **DFlash2 qualification tools (`bee64b00`, `25640c8d`, `a75e845d`, `9f764425`, `d3919444`, `d77a2055`, `4f434bdb`)** — added 2026-10-28 · Tooling (`qualify/`)
  - What it does: `dflash2_acceptance.py`, `dflash2_served.py`, `dflash2_draft_sweep.py`, `dflash2_long_context.py` and their results: acceptance, served DFlash2-vs-MTP comparison, draft-count sweep, past-262k probe. `dflash2_served.py` and `dflash2_draft_sweep.py` accept only packages with eight request slots and mixed graphs, which are packages of the retired `qwen38_dflash` recipe; `dflash2_console_probe.py` and `dflash2_console_rate.py` run the same workload against a console-served package of any prepared concurrency (results: `qualify/results/dflash2-paths-c16/`). In those results, A16 decode (`force_moe_decode_a16`) added to upstream's `qwen38` DFlash2 path with the MXFP8 draft lowers `code_edit` from 555.9 to 518.7 tok/s at concurrency 1 and the 16-client aggregate from 1,231.2 to 1,152.0 tok/s, with non-overlapping ranges over three trials each, so the fork serves that path with A16 decode off. Earlier results of the retired recipe: served comparison `qualify/results/dflash2-served/`, draft sweep `dflash2-draft-sweep/`, history-drafts A/B `dflash2-history-drafts/`, requalification runs `dflash2-requalify/`, past-262k probe `dflash2-long-context/`. `long_context_passkey.py`: five-digit passkey retrieval over prompt length × depth, against a launched package or a running server (`--base-url`), reporting width-16 speculative rounds and accepted drafts per round
  - **Still needed:** Yes — fork measurement tooling, not upstreamed

## Retired patches

- **DFlash2 recipe `qwen38_dflash` (`8046cf12`, fold `df9a23d1`, `05289f11`, `cd47b4d8`)** — added 2026-10-31, retired 2026-11-10
  - What it did: Served Qwen3.8-Flash-Next with the `incoai/Qwen3.8-Flash-Next-DFlash2` block draft through a recipe of its own: aliases `qwen3.8-flash-dflash2` and `qwen3.8-flash-dflash2-yarn4`, a pin of `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2`, mixed prefill rounds embedding a DFlash2 draw, and history drafts on its rounds
  - **Why retired:** Upstream's `qwen38` recipe serves the same draft (`--speculative-method dflash2`). Measured at 16 sequences and a 1,048,576-token context on one RTX PRO 6000 (`qualify/results/dflash2-paths-c16/README.md`), the fork recipe reached 535.1 and 355.6 `code_edit` tok/s at concurrency 1 and 8 against 555.9 and 436.7 for upstream's path with the MXFP8 draft, held 909,312 KV tokens (the server refused a 1M prompt), and exported in 92 minutes against 28. Decode throughput during concurrent long prefill was not measured and is not a workload this fork needs. Lost with it: mixed prefill and history drafts for Qwen 3.8 DFlash2 packages, the QAD pin (that repository id now resolves to `qwen38` with MTP proposals unless `dflash2` is given), the tools `dflash2_tap_check.py` and `mixed_prefill_check.py`, and the CUDA parity check of the shared draft component against the reference on the pinned Flash-Next draft. Packages exported with the recipe keep loading
  - Left without a production caller, unchanged: the hooks the fork added to `ignition/recipes/qwen38/` (`mtp=` and `draft_windows=` in `state.py`, feature taps in `decoder.py`, feature buffers in `mixed_state.py`, `feedback=` in `speculative_state.py`); `projector="dflash"` in `ignition/components/dflash2_weights.py`. The `admission` and `draft_page_blocks` parameters of `PreparedParallelDraftRound`, their test, and the absorption and speculative-round hooks in `mixed_model.py` were removed with the move to upstream `6b68b9b4`, whose functional draft round cannot run them
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
  the runtime owns scheduling, cache storage, and request handling. A focused
  hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Hyperflare trades that breadth for a focused
serving contract around selected models and local workloads. Hyperflare is not, and
will never be, a replacement for production inference engines like vLLM.

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
