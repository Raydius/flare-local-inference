# Attribute copied draft depths to the copy acceptance curve in the scheduler

Tier: 2
Type: bug

## Stated Requirements

- "Scheduler copy-mask acceptance fix"

Assumed: the fix is the `docs/TODO.md:38` item — Helios's acceptance learner routes every depth of a round into the package-wide proposal source's curve, ignoring the completion record's copy mask.
Assumed: depths whose copy-mask bit is set are observed into the `Copy` curve and every other depth into the package's `proposal_source` curve, exactly as `metrics.rs:425` already splits the served counters.
Assumed: the Python mirror scheduler (`ignition/direct/scheduler.py:221`) has no proposal-source axis and stays unchanged.
Assumed: the forecast path (`rounds.rs:206`, draft confidence) is a separate TODO item and stays unchanged.

## Root Cause

wrong-mental-model: `AcceptanceProfile::observe` takes one `ProposalSource` for the whole round and writes every observed depth into `self.prefixes[source]` (`helios/src/scheduler/speculation.rs:136-153`); the commit passes the scheduler-wide `proposal_source` and drops `vector.copy_offered_mask` on the floor (`helios/src/scheduler.rs:2522-2530`). That model held when a package was either MTP or copy. `--history-drafts` substitutes copies per depth inside an MTP round (`ignition/components/parallel_draft.py:59-77`) and records which depths were copies in the mask, so copied depths (point-mass q = 1, near-certain acceptance) now inflate the MTP curve that `rounds.rs:137,160,318` allocates packed-round prefixes from.

## Blast Radius

1. helios/src/scheduler/speculation.rs:136 — `AcceptanceProfile::observe` picks one curve per round; gains a per-depth source from the copy mask
2. helios/src/scheduler.rs:2524 — the only `acceptance.observe` call; passes `vector.copy_offered_mask`
3. helios/src/server/metrics.rs:425 — existing per-depth copy-bit split of `draft_sources`; the precedent the learner follows, unchanged
4. helios/src/scheduler/speculation.rs:176 — unit tests calling `observe` with the current signature
5. grepped 'acceptance.observe\|\.observe(ProposalSource' across helios/src — no further callers
6. helios/src/scheduler/rounds.rs:87,137,160,178,318 — readers of `acceptance.curve(self.proposal_source, …)`; unchanged, they read the now-uninflated MTP curve
7. helios/src/scheduler/tests.rs:1906,1962 — `ProposalSource::Copy` packages record mask 0 or any bits; routing by bit keeps their Copy-curve attribution
8. openspec/specs/history-drafts/spec.md:28-32 — "the runtime attributes those depths' acceptance to copies" names the metrics today; the delta extends it to the learned curve
9. docs/TODO.md:38 — the item this change closes
10. grepped 'copy_offered_mask' across ignition/ and tests/direct — the Python mirror never reads the mask field; unchanged
