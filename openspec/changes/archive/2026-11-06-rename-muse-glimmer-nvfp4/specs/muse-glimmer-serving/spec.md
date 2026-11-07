## MODIFIED Requirements

### Requirement: The recipe pins its target and drafter
`ignition export muse-glimmer-30b-nvfp4 --gpus 0` SHALL prepare `original-el8/Muse-Glimmer-30B-NVFP4` at revision `b47e1e9f77d9a86e18cda5e318c68c632a7fbe59` (`MuseGlimmerForConditionalGeneration`) on one GPU through the `muse_glimmer_nvfp4` recipe (identity contract `muse-glimmer-nvfp4.text.v1`, workload schema `muse_glimmer_nvfp4`), with the drafter `meta-models/Muse-Glimmer-30B-assistant` at revision `c676cce91658f9568aaf264a416211e2782735b7` fetched into the hub cache unless `--no-mtp` is given; more than one GPU is rejected; the declared `stored_bytes` equals the checkpoint index `total_size`.

#### Scenario: Alias export
- **WHEN** the operator runs `ignition export muse-glimmer-30b-nvfp4 --gpus 0` with the pinned snapshot in the hub cache
- **THEN** preparation loads the text weights, the vision tower and the drafter, and writes a package whose served model name is `Muse-Glimmer-30B`

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** export fails before loading weights with an error naming one GPU

#### Scenario: Old recipe name is gone
- **WHEN** `muse_glimmer` is given as a recipe name
- **THEN** the recipe list carries only `muse_glimmer_nvfp4`, leaving `muse_glimmer` free for the upstream mixed-precision recipe

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare context 131072, 128-token pages, concurrency 4, up to four images of 32 MiB, `language_model_only`, sampling defaults temperature 1.0 / top-p 1.0 / top-k disabled, `max_draft_tokens` 15, and rank-1 costs whose `state_row_bytes` and `page_bytes` equal the prepared continuation layout and page size with and without drafting.

#### Scenario: Declaration test
- **WHEN** the declaration test loads the pinned configuration
- **THEN** `continuation_layout(config).row_bytes` and `page_bytes(config)` equal `recipe.costs(1, …)` for the drafting and plain layouts, and the alias resolves to the `muse_glimmer_nvfp4` recipe
