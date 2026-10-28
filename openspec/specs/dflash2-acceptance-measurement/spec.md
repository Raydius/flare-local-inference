# dflash2-acceptance-measurement Specification

## Purpose

Measures, offline and graph-free, how many tokens per verification round a DFlash2 draft and the checkpoint's MTP head would earn against the Qwen3.8-Flash-Next target, so the decision to build a DFlash2 recipe rests on a number measured on this runtime.

## Requirements

### Requirement: Both drafters are measured from the same target traces
The tool SHALL derive every measurement for the DFlash2 arm and the MTP arm from one stored trace per case containing the prompt tokens, the target's greedy continuation tokens, and per-position target features recorded by a single canonical prefill over prompt plus continuation.

#### Scenario: Shared trace
- **WHEN** `measure` runs on a trace directory
- **THEN** both arms read the same case files, start every round from the same prefix positions for a given width, and accept against the same teacher tokens

#### Scenario: Trace reuse
- **WHEN** `trace` runs and a case's trace file already exists for the same continuation digest
- **THEN** no canonical forward runs for that case

### Requirement: Acceptance length is greedy, teacher-forced, and width-capped
A round at prefix end t SHALL draft k = width − 1 proposals, accept the longest prefix of proposals equal to the teacher tokens t+1.., and advance by accepted + 1; acceptance length SHALL be the mean of accepted + 1 over rounds. Widths 4 and 8 SHALL be reported for every arm and every workload category, with round counts.

#### Scenario: Width cap
- **WHEN** a drafter's seven proposals all match the teacher tokens at width 4
- **THEN** the round records accepted = 3 and advances by 4

#### Scenario: Output
- **WHEN** `measure` completes
- **THEN** it writes JSON with per-category and overall acceptance lengths and round counts for `dflash2` and `mtp` at widths 4 and 8, and the trace cache digest

### Requirement: DFlash2 reference reproduces the published draft semantics
The reference SHALL reject any draft `config.json` that departs from the pinned geometry, SHALL embed the pending token and seven mask tokens (id 248077) through the target embedding, SHALL build context keys and values for the most recent 2048 prefix positions from the mean over the four hyperconnection streams of completed target layers 5, 15, 25, 35 and 45, SHALL attend noncausally within the block, SHALL take candidates as the target `lm_head`'s top 16 per proposal position, and SHALL choose each proposal by the predecessor-conditioned selector score with the previously chosen token as predecessor and the pending token as the first anchor.

#### Scenario: Geometry rejection
- **WHEN** the draft config declares `block_size` other than 8 or `target_layer_ids` other than `[5, 15, 25, 35, 45]`
- **THEN** loading fails with an error naming the field

#### Scenario: Window limit
- **WHEN** the prefix is longer than 2048 tokens
- **THEN** block queries attend only the most recent 2048 context positions plus the block

#### Scenario: Shuffled-feature control
- **WHEN** `measure` runs with target features permuted across positions
- **THEN** the DFlash2 width-4 acceptance length falls below the unpermuted run's, which stays at or above 2.0 on chat and code cases

### Requirement: Draft components agree with the shared Triton components
The reference convolution and selector step SHALL match `PreparedGroupedConvolution` and `PreparedCandidateSelector` on the same inputs within BF16 rounding.

#### Scenario: Component parity on GPU
- **WHEN** the CUDA test runs with `FLARE_RUN_CUDA_TESTS=1`
- **THEN** reference and component outputs agree for the convolution and the chosen proposal ids agree for the selector
