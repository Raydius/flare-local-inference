## MODIFIED Requirements

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
