## ADDED Requirements

### Requirement: Variable-tracking task

`qualify/long_context_passkey.py` SHALL accept `--task tracking` (default `passkey`, whose behaviour is unchanged) and in that task SHALL insert, between catalog lines, a chain of assignment sentences whose first assigns a five-digit value to a variable and whose later ones each assign the previous variable to a new one, spread evenly through the prompt in chain order starting at the cell's depth; the question SHALL ask for the value of the last variable, and a cell SHALL pass when the reply's content contains the first value. The task follows RULER's multi-hop variable tracking.

#### Scenario: Chain construction

- **WHEN** a tracking cell is built with four hops at depth 0.1
- **THEN** the prompt holds one sentence assigning a five-digit value and three sentences each assigning the previous variable, in chain order, and the question names the last variable

#### Scenario: Passkey default unchanged

- **WHEN** the probe runs without `--task`
- **THEN** every request equals the request the probe built before this change

#### Scenario: Tracking pass criterion

- **WHEN** the reply to a tracking cell contains the chain's first value
- **THEN** the cell passes, and a reply naming a variable or a different number fails

### Requirement: Scaled against native comparison below the native context

A comparison of a rope-scaled package with the native package SHALL run both tasks at 8,000, 64,000 and 200,000 tokens on each package with the same seeds, and its record SHALL hold each package's grid plus, per cell, whether the two replies are identical.

#### Scenario: Paired cells

- **WHEN** both summaries for the same task and seed are compared
- **THEN** the comparison lists eighteen cell pairs, each with both `passed` values and a `replies_identical` flag
