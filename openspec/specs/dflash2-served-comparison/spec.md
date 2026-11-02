# dflash2-served-comparison Specification

## Purpose

Measures the served DFlash2 and MTP packages of Qwen3.8-Flash-Next against each other through the native HTTP service on the frozen speculation workload, with the adaptive width controller as shipped, so the choice of drafter for served code generation rests on tokens per second and the widths each drafter actually ran.

## Requirements

### Requirement: Arms are served packages under the adaptive controller
The workload SHALL define exactly two arms, `mtp-adaptive` and `dflash2-adaptive`, each naming its package's model id, with history drafts enabled and no fixed verification width. The execution plan SHALL schedule two repetitions with alternating arm order over concurrency ceilings 1, 2, 4, and 8, each trial on a fresh server with an empty prefix cache.

#### Scenario: Arm validation against the package
- **WHEN** the benchmark is given the JSON of `flare inspect PACKAGE --json --graphs` and an arm
- **THEN** it accepts only when the package's model id equals the arm's, `max_active` is 8, speculative graphs exist for batches 1, 2, 4, and 8 at widths 1 through 8 with batch × width ≤ 128, and at least one `mixed` graph carries `draft_tokens`

#### Scenario: Fixed-width package rejected
- **WHEN** the package was exported with a fixed verification width or without mixed graphs
- **THEN** validation fails before any request is sent, naming the missing catalog property

### Requirement: Concurrency 8 is a measured regime
`code_edit` SHALL hold eight frozen cases over the same source module so a concurrency-8 group sends eight distinct code-edit requests; request grouping, the `--concurrency` choice, the idle check, the round-counter parser, and the shape band SHALL accept batch 8 and the catalog's token capacity.

#### Scenario: Concurrency-8 group
- **WHEN** a trial runs at concurrency ceiling 8
- **THEN** the code_edit stage sends one group of eight requests, the idle fence waits for `flare_free_request_slots` to equal the package's `max_active`, and every speculative round counter with batch 8 is attributed to the group

#### Scenario: Earlier case ids preserved
- **WHEN** the workload is regenerated
- **THEN** the first 24 case ids, categories, and request bytes equal the spike workload's, so acceptance traces keyed by case id stay valid

### Requirement: The summary reports served throughput and realized widths per arm and concurrency
The driver SHALL run every trial of the execution plan sequentially on one GPU, starting and stopping the serving container per trial, and SHALL write per-trial measurements and one summary keyed by arm and concurrency ceiling, never pooled across ceilings.

#### Scenario: Summary contents
- **WHEN** all trials complete
- **THEN** the summary holds, per arm and concurrency, code_edit output tokens per second per group, rounds and emitted tokens per (batch, width), draft outcomes by source, and the mixed-graph launch count, which is positive for every trial at concurrency 2 or more

#### Scenario: Container cleanup
- **WHEN** a trial fails or the driver is interrupted
- **THEN** the serving container it started is stopped before the driver exits, and the partial measurements written so far are retained
