# history-drafts Specification

## Purpose

Lets any speculative round built on the shared parallel-draft components substitute continuations copied from the serving process's decoded history for its own proposals, so `flare run --history-drafts` works for every recipe that declares the copy inputs, not only Qwen's MTP round.

## Requirements

### Requirement: The shared round substitutes copied continuations as point-mass proposals
A `PreparedParallelDraftRound` SHALL own the `copy_token_ids`, `copy_token_count`, `copy_expected_consumed`, and `copy_generation` graph buffers and, after every drafter path has produced its proposals, SHALL replace proposal position i with the copied token when the row is active, the copy generation equals the request's generation, the expected consumed count equals the round's consumed count, i is below the copy count, the token is within the vocabulary, and i is below the request's token limit minus one. A copied position's retained proposal distribution SHALL be the exact float32 point mass on the copied token, and the position SHALL be valid.

#### Scenario: Copy accepted under the point-mass law
- **WHEN** a lane carries three copied tokens for the round's consumed count and generation
- **THEN** positions 0..2 propose those tokens with q = 1 on each, and verification accepts each exactly when the target's filtered distribution does

#### Scenario: Stale copy ignored
- **WHEN** a lane's copy generation or expected consumed count differs from the request's current values
- **THEN** no position is substituted and the learned proposals stand unchanged

### Requirement: Learned proposals after a copied run stay offered
After substitution the round SHALL keep every learned proposal above the last copied position offered with its retained proposal distribution, so verification remains exact under the rejection sampler and no position is marked invalid by the substitution.

#### Scenario: Learned tail after a copy
- **WHEN** two tokens are copied into a seven-draft round
- **THEN** positions 0 and 1 propose the copies with q = 1, positions 2..6 propose the drafter's tokens with the drafter's q, and all seven positions are valid

### Requirement: The commit records which offered drafts were copies, bounded by the offer
The commit SHALL record a per-row copy mask whose bit i is set exactly when position i was substituted, masked to the final offered draft count derived from the request's token limit, so the recorded mask is always below `1 << offered_drafts`; this bound SHALL apply to every round that records a mask, including Qwen's MTP round.

#### Scenario: Mask reaches the completion record
- **WHEN** positions 0 and 1 are copies
- **THEN** the completion record's copy mask field is 0b11 and the runtime attributes those depths' acceptance to copies

#### Scenario: Long copy on a short lane
- **WHEN** a lane two tokens from its `max_tokens` budget receives a seven-token history continuation
- **THEN** only position 0 is substituted, the recorded mask is 0b1, and the serving process reports no completion validation error

#### Scenario: Split rounds carry the mask
- **WHEN** a proposal-only source round substitutes copies and a verification-only target round commits
- **THEN** the target commit records the source round's mask

### Requirement: Recipes opt in by declaring the copy inputs on their speculative graphs
A recipe whose speculative graph signature lists the four copy fields SHALL have them bound in its model contract, and the runtime SHALL admit `--history-drafts` for its package; a recipe that does not list them SHALL behave as today, with the buffers at inert initial values and no substitution.

#### Scenario: MiMo and Nemotron packages admit the flag
- **WHEN** a MiMo V2.6 Pro or Nemotron Bolt package exported by this change is loaded with `--history-drafts`
- **THEN** loading succeeds and `flare_history_drafts_enabled` reports 1

#### Scenario: Undeclared recipe unchanged
- **WHEN** a recipe on the shared round does not declare the copy fields
- **THEN** its contract carries no copy inputs, `--history-drafts` is still rejected at load, and its rounds propose exactly as before
