# rope-scaled-export Specification

## Purpose

Defines what an export prepared with rope scaling records about itself, so the rebuild record of a rope-scaled core reproduces the same context.

## Requirements

### Requirement: The rebuild record carries rope scaling
An export whose checkpoint is rope-scaled, by `--rope-parameters` or by an alias that declares scaling, SHALL record the scaling in the core's rebuild record, and the record's reproduction arguments SHALL prepare a core with the same context length. An export given `--draft-model` SHALL record that a local draft folder was supplied.

#### Scenario: Flag-scaled export
- **WHEN** a checkpoint is exported with `--rope-parameters '{"factor": 4.0, "original_max_position_embeddings": 262144}'`
- **THEN** the reproduction arguments in the rebuild record include the same `--rope-parameters` value

#### Scenario: Alias-scaled export
- **WHEN** the alias `qwen3.8-flash-yarn4` is exported
- **THEN** the reproduction arguments resolve to a 1,048,576-token context, not the unscaled checkpoint of the same repository

#### Scenario: Unscaled export is unchanged
- **WHEN** a checkpoint is exported without rope scaling
- **THEN** its reproduction arguments name no rope parameters

### Requirement: Rope-scaled Qwen3.8 Flash Next tables carry the attention factor

A Qwen3.8 Flash Next preparation whose checkpoint is rope-scaled SHALL multiply its rotary cos and sin tables by the YaRN attention factor `0.1 ln(factor) + 1`, so the tables equal those of the transformers rotary embedding for the same YaRN parameters; an unscaled preparation's tables SHALL be unchanged.

#### Scenario: Factor-4 tables

- **WHEN** tables are built with `RopeScaling(4., 262144)`
- **THEN** every gathered cos and sin equals the transformers `Qwen4ExpTextRotaryEmbedding` output for `rope_type: "yarn"`, `factor: 4.0`, `original_max_position_embeddings: 262144`, and position 0's cos is 1.1386 to four decimal places

#### Scenario: Unscaled tables

- **WHEN** tables are built without rope scaling
- **THEN** position 0's cos is exactly 1.0 in every column
