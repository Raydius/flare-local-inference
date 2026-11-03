## ADDED Requirements

### Requirement: The draft runs on the shared DFlash2 component
The Qwen3.8-Flash-Next DFlash2 draft SHALL be prepared by the same shared DFlash2 component that serves Qwen3.8-27B and Muse Spark: its checkpoint loads through the shared loader with the shared tensor-inventory and BF16 checks (`ignition/components/dflash2_weights.py:37`, `:76`), its attention geometry is declared as the shared geometry record (`ignition/components/dflash2_geometry.py:11`), and the block forward, context writer, embedding, head, and candidate selector are the component's (`ignition/components/dflash2.py:44`, `:204`). The recipe SHALL add only what the component does not own: the Flash-Next checkpoint pin and configuration bounds, the hybrid-state ring regions, the feature rows that feed the component's context from Qwen's prefill, mixed-prompt, and verification states, the proposal-only round with the Qwen admission validator (`ignition/recipes/qwen38_dflash/speculative_model.py:69`), history copies (`ignition/components/parallel_draft.py:217`–`:220`), GDN commit, and the mixed prompt context (`ignition/recipes/qwen38_dflash/speculative_model.py:240`). No separate copy of the draft block SHALL exist in `ignition/components/`.

#### Scenario: One shared implementation
- **WHEN** `grep -rn "dflash_draft" ignition/ tests/ qualify/` runs after the change
- **THEN** it finds nothing, `ignition/recipes/nemotron_h/dflash.py` equals upstream's file, and the Flash-Next recipe imports the draft block, scratch, context, geometry, and loader from `ignition/components/dflash2*.py` only

#### Scenario: Flash-Next checkpoint contract
- **WHEN** a draft checkpoint lacks a tensor, carries a misshaped tensor, stores a non-BF16 tensor, or carries an unexpected tensor
- **THEN** the shared loader rejects it before opening the weights, for the Flash-Next geometry as it does for the 27B geometry

#### Scenario: Single GPU through the sharded path
- **WHEN** the recipe prepares on one GPU
- **THEN** the component's sharded embedding, head, and reductions run without a collective runtime and the recipe still refuses `world_size != 1` with the usage error it raises today (`ignition/recipes/qwen38_dflash/recipe.py:45`)

#### Scenario: History copies stay exclusive with vocabulary-parallel sampling
- **WHEN** the recipe's round is constructed with both history-copy buffers and vocabulary-parallel sampling
- **THEN** construction raises `ValueError("history copies and vocabulary-parallel sampling are exclusive")`

### Requirement: Context writes respect liveness and recency
The recipe SHALL present each prefill chunk, mixed prompt chunk, and verification round to the component's context writer so that a token's key/value is written only when the recipe marks it live — a prefill or prompt chunk's last 2048 positions (`ignition/recipes/qwen38_dflash/speculative_model.py:89`–`:101`), a verification round's accepted columns (`:104`–`:111`) — at ring slot `row × 2048 + position mod 2048`, and dead tokens (request id −1 or outside the live set) SHALL resolve to slot −1 so they never address another request's ring.

#### Scenario: Dead tokens
- **WHEN** a batch holds inactive rows whose request ids are −1
- **THEN** no ring slot of any live row is written by those tokens

#### Scenario: Prefill longer than the window
- **WHEN** a prefill chunk longer than 2048 tokens is written
- **THEN** only its last 2048 positions land in the request's rings, each at slot `position mod 2048`, and no two tokens of one chunk write the same slot

### Requirement: The prepared draft reproduces the DFlash2 reference under its storage contract
For the pinned Qwen3.8-Flash-Next-DFlash2 draft, one block through the shared component with key/value round-tripped through an E4M3 ring, followed by the candidate selector, SHALL match `ignition/reference/qwen38_dflash2.py`'s forward, candidates, and greedy walk with the reference's key/value rounded to E4M3 the same way, within the bounds recorded from the measured repeatability of that rounded reference (`tests/recipes/qwen38_dflash/test_draft_cuda.py:16`–`:49`: rings 2e-2, hidden and logits 5e-2 on real features; the fabricated-feature bound recorded separately), with the top-16 shortlist and the walked tokens equal on real features. The bounds SHALL NOT be relaxed to accommodate the component's arithmetic; a bound the component cannot meet is reported with the measured values.

#### Scenario: Parity on real tapped features
- **WHEN** the CUDA test runs on GPU 1 with the pinned draft and a trace of the target's tapped features
- **THEN** the component's hidden states, logits, and rings match the E4M3-rounded reference within the recorded per-output bounds, the top-16 shortlist and walked tokens equal the reference's on every row, and ties resolve to the lowest token id in both

#### Scenario: Fabricated features
- **WHEN** no trace is available
- **THEN** the test runs on random features against the same E4M3-rounded reference and records that bound separately, skipping no assertion silently

### Requirement: The folded package serves the measured baselines
A package exported from the folded recipe SHALL serve the frozen speculation workload through `flare run` with clean transport and no failed requests, and its code_edit output throughput SHALL be compared against the pre-fold package's 329.1 tokens per second at concurrency 1 and 516.5 at concurrency 8 (`qualify/results/dflash2-served/summary.json`, vault DFlash2 Recipe Progress §12), with the drafter's offered and accepted token counts recorded beside them.

#### Scenario: Requalification
- **WHEN** `qualify.dflash2_served` runs trials `0:1:dflash2-adaptive` and `0:8:dflash2-adaptive` against the new entry
- **THEN** both trials report `transport_passed` with zero failed requests, and the medians and acceptance counts are recorded against the baselines; a median more than 5 % below its baseline is reported as a finding, never absorbed
