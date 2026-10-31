## MODIFIED Requirements

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare `draft_tokens=7`, `token_capacity=128`, rank-1 costs whose `state_row_bytes` equals the prepared layout's row bytes, `mixed_prefill=True`, and TP1 only, rejecting `--gpus` with more than one device.

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** preparation fails before loading with a message naming the single-GPU restriction

#### Scenario: Mixed catalog matches the plan
- **WHEN** the operator exports `qwen3.8-flash-dflash2` with `--mixed-prefill`
- **THEN** the package holds the same mixed shapes the plan reports, for draft-enabled and draft-disabled modes, and the memory report lists them

#### Scenario: Mixed prefill is opt-in
- **WHEN** the operator exports without `--mixed-prefill`
- **THEN** the package holds no mixed graphs and the memory report lists no mixed shapes, as for qwen38

## ADDED Requirements

### Requirement: A mixed round drafts with DFlash2 around one packed target forward
When the plan selects mixed prefill, a mixed graph SHALL run the DFlash2 speculative round for its decode rows and prefill one prompt chunk in the same packed Qwen target forward. The packed forward SHALL tap the draft's target layers, and the round SHALL split the tapped features by role. It SHALL write the prompt chunk's last 2048 positions into the prompt request's rings, and the emitted verification columns into each decode request's rings. These are the same writes a separate prefill graph and speculative graph would make. The qwen38 MTP mixed round SHALL prepare the same graphs and kernels as before.

#### Scenario: Mixed round equals separate graphs
- **WHEN** one prompt chunk and a speculative batch run through a DFlash mixed graph, and the same inputs run through a prefill graph followed by a speculative graph
- **THEN** commits, emitted tokens, and ring contents agree within the verification arithmetic's measured repeatability

#### Scenario: Prompt chunk longer than the window
- **WHEN** a mixed graph's prompt chunk holds more than 2048 tokens
- **THEN** only its last 2048 positions are written to the prompt request's ring, each at slot `position mod 2048`

#### Scenario: Layout without a residual region
- **WHEN** a DFlash prompt role is prepared with draft state maintained
- **THEN** no residual checkpoint is compiled, because the DFlash layout holds no `target_residual` region

#### Scenario: MTP mixed catalog unchanged
- **WHEN** the qwen38 recipe prepares its mixed catalog with MTP
- **THEN** the graph signatures, state regions, and compiled residual checkpoints equal those prepared before this change
