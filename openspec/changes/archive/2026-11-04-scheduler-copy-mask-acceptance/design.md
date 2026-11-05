# Design — scheduler-copy-mask-acceptance

## Approach

- `helios/src/scheduler/speculation.rs:136` `AcceptanceProfile::observe` gains a `copy_mask: i64` parameter; inside the depth loop the curve is chosen per depth: `ProposalSource::Copy` when `depth < COPY_DRAFT_CAPACITY && copy_mask & (1 << depth) != 0`, else the caller's `source` — the same predicate `server/metrics.rs:425` uses for `draft_sources`.
- The prefix semantics stay as today: depths `0..observed` are observed with the binary or survival value; only the destination curve changes per depth. `rounds` still increments once per round.
- `helios/src/scheduler.rs:2524` passes `vector.copy_offered_mask` (already validated below `1 << offered_drafts` by `transport.rs:74-75`, so no new validation).
- Copy-source packages (`ProposalSource::Copy`) record masks whose bits already name copies; routing by bit lands them in the Copy curve as before, so `tests.rs:1906,1962` keep their behavior.
- Unit tests in `speculation.rs`: an MTP round whose mask marks depths 0..1 as copies leaves the MTP curve's depths 0..1 untouched and moves the Copy curve's; existing `observe` calls pass mask 0.
- Readers (`rounds.rs:87,137,160,178,318`) are unchanged: they already read the package's `proposal_source` curve, which now excludes copied depths.
- `docs/TODO.md:38` item removed; `openspec/specs/history-drafts/spec.md` delta names the learned curve beside the metrics.
- Not changed: the Python mirror `ignition/direct/scheduler.py:221` (no source axis) and the draft-confidence forecast path `rounds.rs:206` (separate TODO item).

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.965
Caller Agreement: 0.95
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — cancelled (quorum met first), agreement n/a
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.95
- local-ai-qllm (qLLM) — ok, agreement 0.98

### Review Feedback

Consensus feedback recorded verbatim:
- "Update `AcceptanceProfile::observe` to accept the `copy_offered_mask: i64` alongside existing arguments. For each depth `d` in `0..observed`, resolve its source using the exact same predicate as `metrics.rs:425` … and write the survival value to the corresponding curve." — ACCEPTED: this is the approach above.
- "Prefix-survival semantics must be honored so that a single rejection watermark correctly invalidates subsequent depths across curves." — ACCEPTED: the `0..observed` range and the per-depth value are unchanged; only the destination curve varies, so the rejection watermark still zeroes every longer prefix in whichever curve it lands; the unit test asserts a rejection at an MTP depth above a copied depth writes 0 to the MTP curve.
- "Expected blast radius includes … the Python mirror." — REJECTED: `ignition/direct/scheduler.py:221` has no proposal-source axis and reads no mask field; adding one is a new feature on the mirror, not this bug's radius (recorded in proposal Assumed lines).
- "The optimal implementation iterates `0..observed` in a single pass, using bitwise mask checks to dispatch updates … branch-free and without heap allocations." — ACCEPTED: one pass, one `usize` index computed from the bit, no allocation.

## Bomb-squad findings

- helios/src/scheduler/rounds.rs:206-225 `observe_packed_forecasts` — drafter confidence recorded per depth with no mask check while copies replace those positions after the fact; same class, already tracked at docs/TODO.md:34-36 — out of scope, follow-up stays on the TODO.
- helios/src/scheduler/rounds.rs:178 `speculative_information` with learning.rs:48-52 — after per-depth routing, MTP depths copies habitually answer stop refreshing, their weight decays and the recovery probe ranks them least-informed; affects probe spend only — out of scope, noted for the follow-up above.
- helios/src/scheduler/service.rs:236, policy.rs:524 `expected_tokens(self.proposal_source, …)` — same package-wide read as rounds.rs; read the de-inflated curve after the fix — not a defect.
- Copy curve in an MTP package is written, never read; `rounds()` is profile-wide — not a defect.
- Checked clean: survival validation scheduler.rs:2395-2401 vs observe :2524 (same `observed`, width-1 returns early); tests.rs:1906,1962 and phase_tests.rs:1569 Copy packages route to the Copy curve either way; transport.rs:74-75 bound matches the `depth < COPY_DRAFT_CAPACITY` guard.
