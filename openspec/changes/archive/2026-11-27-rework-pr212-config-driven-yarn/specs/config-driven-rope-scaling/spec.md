## Purpose

Defines how the upstream contribution extends a Qwen3.8 Flash Next checkpoint's serving context: the checkpoint's own config selects YaRN scaling, the rotary tables follow the transformers reference including its attention factor, and unmeasured configurations are refused.

## ADDED Requirements

### Requirement: The checkpoint config selects YaRN scaling

The Qwen3.8 recipe SHALL apply YaRN scaling only when the resolved checkpoint directory's `config.json` carries `text_config.rope_parameters` with `rope_type` `"yarn"`, and the contribution SHALL offer no alias and no command-line flag that selects scaling.

#### Scenario: YaRN config directory

- **WHEN** a checkpoint directory whose `text_config.rope_parameters` has `rope_type: "yarn"`, `factor: 4.0` and `original_max_position_embeddings: 262144`, with `max_position_embeddings` left at 262144 as the Qwen model card gives it, is resolved
- **THEN** the resolved checkpoint's context is 1,048,576 and the plan report records that context and the rope parameters

#### Scenario: Native checkpoint is unchanged

- **WHEN** the pinned `qwen3.8-flash` checkpoint is resolved
- **THEN** its context is 262,144 and its plan report equals the report on unpatched `upstream/main`

#### Scenario: No second mechanism

- **WHEN** `git grep -n -E 'rope-parameters|yarn4'` runs over both pull-request branches
- **THEN** it finds no alias declaration and no flag definition

### Requirement: Rotary tables follow the transformers reference

For a YaRN config the recipe SHALL take the inverse frequencies and the attention factor from the installed transformers `ROPE_INIT_FUNCTIONS["yarn"]` and SHALL multiply the cos and sin tables by that factor; for a `default` config the tables SHALL be bit-identical to those built on recipes `upstream/main`.

#### Scenario: Factor applied

- **WHEN** tables are built for theta 10,000,000, factor 4 and original context 262,144
- **THEN** the cos table at position 0 equals the reference attention factor 0.1·ln(4) + 1 in every column, and every sampled position equals the reference cos and sin times that factor

#### Scenario: Literal anchor

- **WHEN** the same parameters are given to the reference
- **THEN** the frequencies below column 14 are unscaled, the frequencies above column 22 are divided by 4, and the factor is 1.1386 to four decimal places

#### Scenario: Native tables

- **WHEN** tables are built for the pinned native config
- **THEN** they equal `cos` and `sin` of position times `1 / theta^(2i/64)` bit for bit

### Requirement: Unsupported rope configurations are refused

The recipe SHALL refuse, naming the offending field, a `rope_type` other than `default` or `yarn`, an unknown rope key, an `original_max_position_embeddings` or `max_position_embeddings` other than 262,144, or a ceiling of `factor` times the original that is not a whole number, is not 64-aligned, does not exceed 262,144, or exceeds 1,048,576; explicit `attention_factor`, `mscale` and `mscale_all_dim` SHALL be accepted and passed to the reference.

#### Scenario: Oversized factor

- **WHEN** the config declares `factor: 1000`
- **THEN** resolution fails with a usage error naming `factor` before any preparation starts

#### Scenario: Unimplemented type

- **WHEN** the config declares `rope_type: "longrope"`
- **THEN** resolution fails naming `rope_type`

#### Scenario: Explicit attention factor

- **WHEN** the config adds `attention_factor: 1.0` to a valid YaRN block
- **THEN** the checkpoint resolves and the tables are scaled by 1.0

### Requirement: A scaled context is refused on SM103

The recipe SHALL refuse to plan a checkpoint whose context exceeds 262,144 when any selected GPU is SM103, and the SM103 score-workspace sizing SHALL be unchanged from recipes `upstream/main`.

#### Scenario: SM103 with a YaRN checkpoint

- **WHEN** a YaRN checkpoint directory is planned on an SM103 GPU
- **THEN** planning fails with a usage error saying a scaled context is not measured on SM103

#### Scenario: SM103 with the native checkpoint

- **WHEN** the native checkpoint is planned on an SM103 GPU
- **THEN** the plan equals the plan on unpatched `upstream/main`

### Requirement: Scaled and native preparations have distinct identities

The preparation identity SHALL differ between the native config and a YaRN config, and between two YaRN configs with different ceilings, without any identity entry added by this contribution.

#### Scenario: Identity separation

- **WHEN** identities are computed for the native config, a factor-4 config and a factor-2 config of the same weights
- **THEN** all three differ and the native identity equals the one computed on recipes `upstream/main`

### Requirement: Evidence is measured on the reworked branches

The contribution's GPU evidence SHALL come from packages exported by one image built from the core and recipes pull-request heads, and SHALL include the native and the YaRN package served through the console, the passkey grid, the tracking grid, and the short-context comparison; SM103 at a scaled context SHALL be reported as unmeasured.

#### Scenario: Evidence names its heads

- **WHEN** a figure appears in a pull-request description or reply
- **THEN** it names the core head, the recipes head and the package id it was measured on
