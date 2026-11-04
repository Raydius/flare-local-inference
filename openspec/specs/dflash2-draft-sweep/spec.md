# dflash2-draft-sweep Specification

## Purpose

Measures one served DFlash2 package across fixed draft counts and the adaptive controller on the frozen speculation workload, so the draft count an operator serves with rests on measured tokens per second and acceptance at each count rather than a guess.

## Requirements

### Requirement: Arms are fixed draft counts plus the adaptive controller on one package
The sweep SHALL serve one package once per arm and concurrency ceiling, on a fresh server with an empty prefix cache, with arms `adaptive` (the controller as shipped) and `fixed-N` for N from 0 through 7 (`--mtp-draft-tokens N` with adaptive verification off, so every round runs at width N + 1 where proposals and pending inputs allow; `fixed-0` serves without speculation), at concurrency ceilings 1 and 8 by default. History copies SHALL be off unless the sweep is asked for them, and the runtime's history-drafts gauge SHALL be checked against the arm before any request is sent. The frozen workload, its two comparison arms, and its execution plan SHALL stay unchanged.

#### Scenario: Arm order and server lifecycle
- **WHEN** the sweep runs with its default arms
- **THEN** it starts one container per arm and ceiling in the order adaptive, fixed-0 … fixed-7, each with that arm's serving flags and without `--history-drafts`, runs the frozen workload against it, and stops the container before the next starts

#### Scenario: Narrowed arms
- **WHEN** the sweep is given `--arms fixed-4 adaptive --concurrencies 8`
- **THEN** only those two trials run, in the declared order

#### Scenario: Default commands unchanged
- **WHEN** the served comparison or the long-context probe builds its container command after the change
- **THEN** the command is byte-identical to the one built before, with no draft-count or adaptive-verification flag

### Requirement: A non-speculating arm passes the benchmark's transport checks
The benchmark SHALL accept an arm that serves without speculation: its groups emit through decode graphs, so the emission cross-check SHALL require no speculative emissions and no speculative launches instead of speculative counters that account for the raw token ids. A speculating arm's checks SHALL stay as they are.

#### Scenario: Floor arm at concurrency 1
- **WHEN** `fixed-0` runs at concurrency ceiling 1 and every request completes
- **THEN** every group reports `transport_passed` with zero speculative rounds and launches, and the trial's code_edit tokens per second is recorded as the no-speculation floor

### Requirement: The summary reports the curve per arm and ceiling
The sweep SHALL write one `summary.json` keyed by arm and concurrency ceiling (and repetition) holding, per trial: code_edit output tokens per second (median, p99, minimum, maximum over its eight requests), rounds, emitted tokens, offered and accepted drafts per (batch, width), drafts by source, accepted drafts per speculative round, acceptance rate, transport result, failed-request count, the arm's fixed width, and a width-overrun flag that is true when a fixed arm ran a round wider than N + 1. It SHALL print a table of arm × ceiling → code_edit median tokens per second, accepted per round, and acceptance rate. The sweep reports the curve; it SHALL NOT choose a count.

#### Scenario: Width overrun
- **WHEN** a `fixed-3` trial's counters include a round at width 5
- **THEN** that trial's `width_overrun` is true and the sweep's exit status is nonzero, with the table still printed

#### Scenario: Underrun is reported, not flagged
- **WHEN** a `fixed-6` trial's rounds include widths below 7 because proposals ran out
- **THEN** the per-width histogram shows them and `width_overrun` stays false
