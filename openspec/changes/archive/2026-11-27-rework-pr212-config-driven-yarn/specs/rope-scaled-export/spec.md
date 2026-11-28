## ADDED Requirements

### Requirement: Rope-scaled Qwen3.8 Flash Next tables carry the attention factor

A Qwen3.8 Flash Next preparation whose checkpoint is rope-scaled SHALL multiply its rotary cos and sin tables by the YaRN attention factor `0.1 ln(factor) + 1`, so the tables equal those of the transformers rotary embedding for the same YaRN parameters; an unscaled preparation's tables SHALL be unchanged.

#### Scenario: Factor-4 tables

- **WHEN** tables are built with `RopeScaling(4., 262144)`
- **THEN** every gathered cos and sin equals the transformers `Qwen4ExpTextRotaryEmbedding` output for `rope_type: "yarn"`, `factor: 4.0`, `original_max_position_embeddings: 262144`, and position 0's cos is 1.1386 to four decimal places

#### Scenario: Unscaled tables

- **WHEN** tables are built without rope scaling
- **THEN** position 0's cos is exactly 1.0 in every column
