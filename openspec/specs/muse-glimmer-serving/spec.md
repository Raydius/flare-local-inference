# muse-glimmer-serving Specification

## Purpose

Serves Muse Glimmer 30B NVFP4 on one GPU through its own `muse_glimmer` recipe: text with the Spark prompt grammar, images through a Glimmer image profile, and DFlash block drafting from the published assistant checkpoint.

## Requirements

### Requirement: The recipe pins its target and drafter
`ignition export muse-glimmer-30b-nvfp4 --gpus 0` SHALL prepare `original-el8/Muse-Glimmer-30B-NVFP4` at revision `b47e1e9f77d9a86e18cda5e318c68c632a7fbe59` (`MuseGlimmerForConditionalGeneration`) on one GPU, with the drafter `meta-models/Muse-Glimmer-30B-assistant` at revision `c676cce91658f9568aaf264a416211e2782735b7` fetched into the hub cache unless `--no-mtp` is given; more than one GPU is rejected; the declared `stored_bytes` equals the checkpoint index `total_size`.

#### Scenario: Alias export
- **WHEN** the operator runs `ignition export muse-glimmer-30b-nvfp4 --gpus 0` with the pinned snapshot in the hub cache
- **THEN** preparation loads the text weights, the vision tower and the drafter, and writes a package whose served model name is `Muse-Glimmer-30B`

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** export fails before loading weights with an error naming one GPU

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
With drafting enabled, each speculative round SHALL embed the pending token and fifteen mask tokens (201818, raw embedding without the embedding norm), run the five-layer assistant over them against context K/V projected from target layers 1, 13, 25, 37 and 49 (`encoder.fc`, `output_norm_enc`, per-layer key norm and rope) in 2048-position rings, with no grouped convolution and no candidate selector, verify through the target with sliding-cache rollback, and commit accepted tokens into the rings; `--no-mtp` skips the draft download, weights, state and graphs.

#### Scenario: Width-sixteen round
- **WHEN** a width-16 round's fifteen proposals all pass verification
- **THEN** sixteen tokens are emitted and sixteen ring positions are written

#### Scenario: Text-only package
- **WHEN** the operator exports with `--no-mtp`
- **THEN** no assistant download occurs and the package carries no speculative graphs

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare context 131072, 128-token pages, concurrency 4, up to four images of 32 MiB, `language_model_only`, sampling defaults temperature 1.0 / top-p 1.0 / top-k disabled, `max_draft_tokens` 15, and rank-1 costs whose `state_row_bytes` and `page_bytes` equal the prepared continuation layout and page size with and without drafting.

#### Scenario: Declaration test
- **WHEN** the declaration test loads the pinned configuration
- **THEN** `continuation_layout(config).row_bytes` and `page_bytes(config)` equal `recipe.costs(1, …)` for the drafting and plain layouts, and the alias resolves to the `muse_glimmer` recipe
