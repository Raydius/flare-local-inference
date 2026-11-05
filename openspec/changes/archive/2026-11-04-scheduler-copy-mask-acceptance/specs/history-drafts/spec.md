## MODIFIED Requirements

### Requirement: The commit records which offered drafts were copies, bounded by the offer
The commit SHALL record a per-row copy mask whose bit i is set exactly when position i was substituted, masked to the final offered draft count derived from the request's token limit, so the recorded mask is always below `1 << offered_drafts`; this bound SHALL apply to every round that records a mask, including Qwen's MTP round. The runtime SHALL attribute each offered depth's outcome by that bit, both in the served draft-source counters and in the request's learned acceptance curves: a copied depth updates the copy curve and every other depth updates the package's proposal-source curve, so the curve the packed-round allocator reads for a package's proposal source never includes copied depths.

#### Scenario: Mask reaches the completion record
- **WHEN** positions 0 and 1 are copies
- **THEN** the completion record's copy mask field is 0b11 and the runtime attributes those depths' acceptance to copies

#### Scenario: Copied depths leave the learned MTP curve alone
- **WHEN** an MTP package's round offers five drafts with copies at positions 0 and 1, accepts three, and rejects position 3
- **THEN** the request's copy curve observes depths 0 and 1 as accepted, its MTP curve observes depth 2 as accepted and depths 3 and 4 as rejected, and its MTP curve's depths 0 and 1 are unchanged

#### Scenario: Long copy on a short lane
- **WHEN** a lane two tokens from its `max_tokens` budget receives a seven-token history continuation
- **THEN** only position 0 is substituted, the recorded mask is 0b1, and the serving process reports no completion validation error

#### Scenario: Split rounds carry the mask
- **WHEN** a proposal-only source round substitutes copies and a verification-only target round commits
- **THEN** the target commit records the source round's mask
