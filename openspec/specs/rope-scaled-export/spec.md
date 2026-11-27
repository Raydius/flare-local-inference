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
