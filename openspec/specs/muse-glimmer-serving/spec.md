# muse-glimmer-serving Specification

## Purpose

Serves Muse Glimmer 30B NVFP4 on one GPU through its own `muse_glimmer_nvfp4` recipe: text with the Spark prompt grammar, images through a Glimmer image profile, and DFlash block drafting from the published assistant checkpoint.

## Requirements

### Requirement: The recipe pins its target and drafter
`ignition export muse-glimmer-30b-nvfp4 --gpus 0` SHALL prepare `original-el8/Muse-Glimmer-30B-NVFP4` at revision `b47e1e9f77d9a86e18cda5e318c68c632a7fbe59` (`MuseGlimmerForConditionalGeneration`) on one GPU through the `muse_glimmer_nvfp4` recipe (identity contract `muse-glimmer-nvfp4.text.v1`, workload schema `muse_glimmer_nvfp4`), with the drafter `z-lab/Muse-Glimmer-30B-DFlash2` at revision `b54ffdd11fa9cfe2af370012e5763d492c904128` fetched into the hub cache unless `--no-mtp` is given or `--draft-model` names a local folder; more than one GPU is rejected; the declared `stored_bytes` equals the checkpoint index `total_size` and the declared `draft_bytes` equals the drafter's `model.safetensors` size.

#### Scenario: Alias export
- **WHEN** the operator runs `ignition export muse-glimmer-30b-nvfp4 --gpus 0` with the pinned snapshot in the hub cache
- **THEN** preparation loads the text weights, the vision tower and the DFlash2 drafter, and writes a package whose served model name is `Muse-Glimmer-30B`

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** export fails before loading weights with an error naming one GPU

#### Scenario: Old recipe name is gone
- **WHEN** `muse_glimmer` is given as a recipe name
- **THEN** the recipe list carries only `muse_glimmer_nvfp4`, leaving `muse_glimmer` free for the upstream mixed-precision recipe

#### Scenario: Assistant drafter rejected
- **WHEN** `--draft-model` names a folder whose `config.json` is the `MuseGlimmerAssistantModel` form (no `dflash_config`)
- **THEN** preparation fails before loading draft weights with an error naming the unsupported DFlash2 geometry

### Requirement: Text projections load from the checkpoint's ModelOpt NVFP4 metadata
Preparation SHALL read the `quantization_config` `config_groups`/`ignore` form (4-bit, group 16 weights and input activations) through the shared parser, mark every `model.language_model.layers.*.{self_attn.{q,k,v,gate,o}_proj, mlp.{gate,up,down}_proj}` NVFP4 and leave `lm_head`, `embed_tokens`, norms and `model.vision_*` BF16; q/k/v load as one fused NVFP4 group, the attention gate as its own projection, mlp gate/up as one fused group; input scales stored in `model-inputscales.safetensors` resolve through the index.

#### Scenario: Fused groups honor calibration
- **WHEN** layer weights load
- **THEN** q/k/v share one `weight_scale_2`/`input_scale` pair, gate/up share one, and the attention gate keeps its own, with no `ValueError` from the fused-group check

#### Scenario: Gemma 4 unchanged
- **WHEN** Gemma 4 NVFP4 configuration is parsed through the lifted parser
- **THEN** the resulting layer map equals the map the private Gemma parser produced

### Requirement: The decoder matches the canonical forward
The prepared decoder SHALL compute each layer as `input_layernorm → attention → post_attention_layernorm → residual add → pre_feedforward_layernorm → SwiGLU → post_feedforward_layernorm → residual add` with centered (1 + w) RMSNorms, attention with scale-less q/k RMSNorm, query × `qk_scale_factor`, rope 500000 on sliding layers and no positional encoding on full layers, the sigmoid gate applied to the raw attended value (no output norm), FP8 E4M3 sliding rings of 2048 and 128-token paged full-layer histories, the normed embedding, and logits `20·tanh(logits × output_multiplier / 20)`.

#### Scenario: Reference parity
- **WHEN** a traced prompt longer than 2048 tokens runs through the prepared prefill and decode graphs and through `ignition.reference.muse_glimmer`
- **THEN** the softcapped logits agree within 1e-1 relative L2 with top-1 agreement of at least 0.98 and the reference argmax inside the prepared decode top-3, and the residual stream and final hidden states sit no farther from the FP8-history reference than twice the distance the reference itself moves when only its history precision changes (E4M3 to BF16), per layer and on the final hidden, on sliding and full layers across the window boundary

#### Scenario: Spark unchanged
- **WHEN** Muse Spark attention is prepared after the widening
- **THEN** its output norm, single fused projection and session identity are byte-identical to before

### Requirement: The frontend renders the Glimmer template with recipient headers and ATEM tools
The package SHALL carry the checkpoint tokenizer and the annotated Glimmer `chat_template.jinja` (source sha256 `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`), with `RECIPIENT_HEADERS` reasoning (`<|start|>assistant to=self<|message|> … <|eom|>`), `ATEM` tools, `<|eot|>`/`<|end_of_text|>` as end markers, `reasoning_strength`, `knowledge_cutoff` and `current_date` reachable through `chat_template_kwargs`, and `<|video|>` rejected.

#### Scenario: Checkpoint parity
- **WHEN** the frontend parity test renders system/user/assistant turns, reasoning on and off, tool calls with multiple and reordered results, and continuation against the pinned tokenizer and template
- **THEN** prompt bytes, token IDs and message endpoints match the Transformers rendering

#### Scenario: Video rejected
- **WHEN** a request carries a video content part
- **THEN** the request fails with a validation error and no graph runs

### Requirement: Images follow the Glimmer image profile
The frontend SHALL resize each image to the integer patch grid (factor 28) under 4096 merged tokens closest to its aspect ratio, resample with antialiased Lanczos, normalize with mean and std 0.5, emit patches `(temporal 2, channel, 14, 14)` in window order (32×32-patch cells, raster within a cell) with `cu_window_seqlens`, per-image `cu_seqlens`, a raster `reverse_index`, position ids and four-tap bilinear position-table interpolation, and render `<|image_start|>` + `<|patch|>` × (grid ÷ 4) + `<|image_end|>`; the prepared vision graph SHALL run 37 window layers on the cell segments and 13 full layers on the image segments, un-permute, pixel-shuffle dim-major, apply the adapter, projection and RMSNorm, and inject the features at the `<|patch|>` positions.

#### Scenario: Vision parity
- **WHEN** the four card images and a ragged-grid image run through the prepared vision graph and through `MuseGlimmerModel.get_image_features`
- **THEN** the merged features agree feature-for-feature within BF16 tolerance and the token counts equal `smart_resize`'s

#### Scenario: Every image profile declares its width
- **WHEN** any `ImageSettings` variant, including `PatchMerge`, is materialized in Helios
- **THEN** its patch width and interpolation taps are the profile's own, and a MiniMax M3 image request no longer fails "image preprocessing differs from its prompt geometry"

### Requirement: DFlash drafting proposes up to fifteen tokens
With drafting enabled, each speculative round SHALL embed the pending token and fifteen mask tokens (201818, raw embedding without the embedding norm), run the five-layer DFlash2 draft over them against context K/V projected from target layers 1, 13, 25, 37 and 49 (`fc`, `hidden_norm`, per-layer key norm and rope) in 2048-position rings, with two-tap grouped convolutions (kernel 2, group 16) before each attention and MLP, the rank-256 top-16 candidate selector tracing one path through the block, the target's `output_multiplier` and softcap applied to the draft head as `dflash_config` declares them, verify through the target with sliding-cache rollback, and commit accepted tokens into the rings; `--no-mtp` skips the draft download, weights, state and graphs.

#### Scenario: Width-sixteen round
- **WHEN** a width-16 round's fifteen proposals all pass verification
- **THEN** sixteen tokens are emitted and sixteen ring positions are written

#### Scenario: Text-only package
- **WHEN** the operator exports with `--no-mtp`
- **THEN** no drafter download occurs and the package carries no speculative graphs

#### Scenario: Drafter accepts on the pinned checkpoint
- **WHEN** the CUDA acceptance test runs one width-16 round on the pinned target and the z-lab drafter with a greedy prompt
- **THEN** the prepared proposals match the fork's DFlash2 reference within BF16 tolerance and at least one proposal is accepted

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare context 1048576 (the checkpoint's native 131072 is the trained anchor; no rope scaling is declared because the 39 sliding layers never see a rope distance past 2048 and the 13 full layers carry no positional encoding), 128-token pages, concurrency 8 by default and 16 as the `--max-concurrency` ceiling, up to four images of 32 MiB, `language_model_only`, sampling defaults temperature 1.0 / top-p 1.0 / top-k disabled, `max_draft_tokens` 15, a packed speculative catalog of `token_capacity` 128 (full fifteen-draft depth through eight active requests) and `graph_budget` 64, and rank-1 costs whose `state_row_bytes` and `page_bytes` equal the prepared continuation layout and page size with and without drafting and whose `graph_bytes` is the VRAM inventory of the concurrency-16 export; preparation SHALL accept any 128-aligned context from 128 through 1048576 and the exported package SHALL carry `max_sequence_tokens` equal to the prepared context.

#### Scenario: Declaration test
- **WHEN** the declaration test loads the pinned configuration
- **THEN** `continuation_layout(config).row_bytes` and `page_bytes(config)` equal `recipe.costs(1, …)` for the drafting and plain layouts, `RECIPE.context` is 1048576, `RECIPE.concurrency` is 8, `RECIPE.max_concurrency` is 16, and the alias resolves to the `muse_glimmer_nvfp4` recipe

#### Scenario: Context ceiling
- **WHEN** `PreparationConfig(context=1048576 + 128)` is constructed
- **THEN** it is rejected with the error naming the `[128,1048576]` range, while `PreparationConfig(context=1048576)` is accepted

#### Scenario: Served window below the ceiling
- **WHEN** the operator runs the 1048576-token package with `--max-context-tokens 262144`
- **THEN** the served per-request context is 262144 and a request past it is refused with the context error, without re-export

#### Scenario: Default eight-request catalog
- **WHEN** `PreparationConfig(dflash=True, max_active=8, speculative_token_capacity=128)` is constructed
- **THEN** its round shapes hold width-one rounds at batches 1, 2, 4 and 8 and no batch above 8, the packed width-16 rounds `(8, 16, 9)` and `(8, 16, 128)`, no packed row count above 128, at most 64 packed rounds, and 128 sampled rows

#### Scenario: Sixteen-request ceiling catalog
- **WHEN** `PreparationConfig(dflash=True, max_active=16, speculative_token_capacity=128)` is constructed
- **THEN** its round shapes hold width-one rounds at batches 1, 2, 4, 8 and 16, the packed width-16 rounds `(16, 16, 17)`, `(16, 16, 128)` and `(8, 16, 128)`, no packed row count above 128, at most 64 packed rounds, and 256 sampled rows

#### Scenario: Admission at the default and the ceiling
- **WHEN** the operator exports with no `--max-concurrency` flag, or with `--max-concurrency 16`, and serves the package
- **THEN** the plan reports concurrency 8 or 16 respectively, a request beyond that count waits in the queue rather than being refused, `--max-concurrency 17` is refused with the error naming 1 through 16, and the export log's `Allocate cache` line reports the pages that bound each request to `pages ÷ concurrency × 128` guaranteed tokens while a lone request may still reach 1048576
