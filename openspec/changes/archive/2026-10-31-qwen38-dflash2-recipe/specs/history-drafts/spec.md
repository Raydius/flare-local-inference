## ADDED Requirements

### Requirement: A proposal-only round may take its admission from the owner
`PreparedParallelDraftRound` SHALL accept an owner-supplied admission callable in place of the paged batch validator; with it the round builds no paged validator or validation collective, sets its active rows from the callable, and proposes, substitutes copies, and records the copy mask exactly as with the validator. Without it the round SHALL behave as today.

#### Scenario: Qwen DFlash2 round copies
- **WHEN** the qwen38_dflash round's proposal-only shared round runs with Qwen's verification-state admission and a lane carries copies for the round's consumed count
- **THEN** the copies are substituted under the point-mass law and the Qwen commit records the bounded mask

#### Scenario: Default path unchanged
- **WHEN** a recipe constructs the round without an admission
- **THEN** the paged validator runs and the round's outputs are identical to today's
