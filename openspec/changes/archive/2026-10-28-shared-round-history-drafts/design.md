# Design — shared-round-history-drafts

## Context

See proposal.md — Why. Helios already looks up decoded history and packs the four copy inputs for any speculative lane (`helios/src/model/packing.rs:657`), validates the copy mask against the offered-draft count (`helios/src/execution/transport.rs:65`), and attributes per-depth acceptance to copies (`helios/src/server/metrics.rs:405`); it admits `--history-drafts` only when the contract binds the inputs (`helios/src/model/catalog.rs:88`). Ignition emits those bindings for every `speculative`/`mixed` graph whose signature lists the fields (`ignition/preparation/contract.py:42`, `:72`). Qwen's round owns the buffers and the substitution (`ignition/recipes/qwen38/speculative_model.py:52`, `:270`); the shared `PreparedParallelDraftRound` owns neither and builds its commit without `copy_offered_mask` (`ignition/components/parallel_draft.py:286`). `offered_drafts` is derived only from `token_limits` (`ignition/components/speculation.py:347`), and `valid=False` on any used position rejects the whole row (`:319`).

## Goals / Non-Goals

- Goal: one substitution implementation in the shared round that is correct for every drafter path it serves — sequential Markov/DSpark loops, fused Markov, greedy argmax, TP block broadcast — and for split source/target rounds.
- Goal: MiMo V2.6 Pro and Nemotron Bolt packages admit `--history-drafts`.
- Goal: the copy mask can never exceed `1 << offered_drafts`, on Qwen included.
- Non-goal: moving Qwen onto the shared round; changing the Helios ring, lookup, or metrics; declaring the fields for recipes this change does not qualify.

## Approach

1. `ignition/components/parallel_draft.py` — the round's owner allocates `copy_ids [capacity, 7]`, `copy_count`, `copy_expected`, `copy_generation` once (as `qwen38/speculative_model.py:52` does) and publishes them through a `buffers()` mapping the recipes pass as graph inputs; each round owns a `copy_mask [batch]` beside `drafts`, so proposal-only rounds (no commit) still produce it. `propose()` gains one pass after `proposal_block.finish` (`:379`) and before `candidates[:, 1:]` (`:383`): gate = active ∧ `copy_generation[slot] == requests.generation[slot]` ∧ `copy_expected[slot] == before` ∧ `i < copy_count[slot]` ∧ `0 <= copy_ids[slot, i] < vocab` ∧ `i < limits − 1` with `limits = where(active, query_lengths, 0)`, the same term `valid` uses at `:365`; for gated positions `drafts[:, i] = copy`, `probabilities[:, i] = where(copy, one_hot, probabilities[:, i])` in float32, `valid[:, i] |= copy`, `copy_mask |= 1 << i`. Learned proposals after the copied run stay offered with their retained q. No collective runs in the pass. Width-1 rounds (`drafts` of shape `(b, 0)`) make it a no-op.
2. `ignition/components/speculation.py:522` — the commit records `copy_offered_mask & ((1 << offered_drafts) − 1)`, computed from the same `token_limits` it derives `offered_drafts` from, so the transport invariant holds by construction for every round, including Qwen's and packed targets whose limits shrink after allocation. `PreparedParallelDraftRound` passes `copy_offered_mask=self.copy_mask` at `:286`.
3. `ignition/recipes/qwen38/speculative_model.py:270` — the copy gate gains the `index < token_limits − 1` term (the latent defect; one term, no behavior fork).
4. `ignition/components/packed_speculation.py:112` — the source→target handoff list gains `copy_mask`; the target commit bounds it against the post-allocation limits through item 2.
5. `helios/src/model/packing.rs:370` — the history lookup maximum becomes `lane.tokens − 1` instead of `token_capacity − 1`, so the host never offers more copies than the lane can verify; the device-side bound in item 2 remains the invariant.
6. `ignition/recipes/mimo_v2/bootstrap.py:301` and `ignition/recipes/nemotron_h/bootstrap.py:238` — the speculative graph field tuples gain the four names and the recipe `buffers` dicts gain the round owner's buffers.
7. Tests — CPU: a stub-drafter shared-round test ported from the Qwen pattern (`tests/recipes/test_qwen38_speculative_state.py:121`) covering epoch/expected-consumed gating, point-mass rows, mask bits, and a seven-token hit on a lane with `query_lengths < width` asserting the recorded mask `< 1 << offered`; the same `token_limits` case added to the Qwen test (it sets `token_limits = None` at `:146` today); CUDA: the packed split round beside `tests/components/test_packed_speculation_cuda.py:540` with the mask handed to the target and re-bounded after allocation; Helios: a contract declaring the inputs passes `catalog::validate` with `history_drafts: true` beside `copied_proposals_require_concrete_graph_inputs` (`helios/src/model/tests.rs:613`), and a packing test that a lane shorter than the width receives at most `lane.tokens − 1` copies.
8. Docs — `docs/model-preparation.md:344` moves to the shared speculation description naming Qwen, MiMo, and Nemotron; `docs/mtp-design.md:418` states that learned proposals after a copied run remain offered and exact under rejection sampling and that the commit bounds the mask; `docs/TODO.txt` gains the untested recipes and the two pre-existing observations below.

## Decisions

- Substitute once after all drafter paths, not inside each loop — one rule covers sequential, fused, greedy, and broadcast paths and keeps ranks in agreement under TP without a new collective, because the copy inputs are replicated graph inputs, `active` is agreed by the validation all-reduce, and the pass runs after the proposal broadcast that overwrites `drafts`/`valid`/`probabilities`. Alternative rejected: in-loop substitution for sequential drafters (conditions later drafts on the copy as Qwen's recursion does) — a second code path for a small acceptance gain.
- Keep learned proposals after the copied run offered — `valid` means "numerically unusable distribution" and any used-position invalidity rejects the whole row (`speculation.py:319`), so truncation through `valid` would error the request; verification stays exact because q is always the retained distribution of the actual draw (`docs/mtp-design.md:418`). Alternative rejected: shortening `query_lengths` before verify to drop the tail — it changes the verification width the scheduler planned and duplicates the packed allocator's job.
- Bound the mask in the commit, from `token_limits` — a single device-side invariant that covers the shared round, packed targets after allocation, and Qwen; cheaper than every caller re-deriving it. The host-side bound is kept as well because it is one line and stops the host offering copies the lane cannot verify.
- Opt-in by field declaration — matches how the contract already works (`contract.py:72`); no flag, no per-model switch, no contract-version bump. Undeclared recipes stay inert because the buffers hold their initial values (count 0).
- Qwen keeps its own substitution — its round carries MTP-specific absorb, GDN rollback, and `draft_context_lag=1`; folding it in is a separate change. The gate term is the only Qwen edit.

## Risks / Trade-offs

- [Draft cursor: Qwen records lag 1, the shared round lag 0] → the lookup context is `live.tokens()` including the pending token (`helios/src/scheduler.rs:364`, `:2450`), independent of the cursor; MiMo and Nemotron declare `draft_lag=0` matching the round. No change.
- [Packed split rounds re-size prefixes after `propose()` (`packed_speculation.py:84`)] → the mask is bounded at commit against the post-allocation limits, not at propose time.
- [Fused-Markov path keeps rank-local probabilities] → only the root's q reaches the committed sample (`markov_sampling.py:548`); the point mass is written on every rank and is harmless off-root.
- [Confidence head sizes packed prefixes from learned-token confidence at copied positions (`split_draft.py:31`)] → allocation bias only, not correctness; recorded in `docs/TODO.txt`.
- [Scheduler acceptance learning counts copies as learned acceptance (`scheduler.rs:2297`)] → pre-existing on Qwen; recorded in `docs/TODO.txt`.
- [Served qualification needs a MiMo or Nemotron checkpoint and a free GPU, neither present on this host] → an operator leg; the change does not claim served support until it runs.

## External Review

### Peer Review Verdict

Quorum: not achieved (weight 2/3, escalated to tier 2)
Certainty: 0.6
Caller Agreement: 0.55
Arbiter: ok
Override: Ray Dollete | Override | 2026-10-28
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.1
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.5
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.9
- anthropic-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.35

### Review Feedback

Consensus feedback recorded verbatim:
- none — the server withheld the response because the quorum was not achieved; nothing from it was incorporated. The operator recorded the override above, which releases tasks.md; the design revisions below come from the bomb-squad hunt, not from the withheld review.

## Bomb-squad findings

- `ignition/recipes/qwen38/speculative_model.py:270` — latent defect: the copy gate lacks `index < token_limits − 1`; a history hit longer than the lane's remaining budget sets mask bits ≥ `offered_drafts`, `helios/src/execution/transport.rs:65` rejects, and the error leaves the worker loop via `helios/src/scheduler.rs:2243` and `runtime.rs:894`. Fixed by Approach items 2 and 3; tested by item 7.
- `helios/src/model/packing.rs:368` — the lookup maximum is the graph width, not the lane's token count, so the host can offer more copies than a short lane verifies. Fixed by Approach item 5.
- `ignition/components/speculation.py:319`, `:347` — `valid=False` on a used position rejects the whole row and `offered_drafts` comes only from `token_limits`; truncate-after-copy through `valid` would error requests. Design changed: learned proposals after the copy stay offered.
- `ignition/components/packed_speculation.py:84`–`:110` — `query_lengths` is rewritten after `propose()`; a propose-time mask bound would be stale at commit. Fixed by bounding at commit.
- `ignition/components/parallel_draft.py:313`, `:269` — target-only rounds never call `propose()` and proposal-only rounds have no commit; the mask lives on the round and travels through the packed handoff. Carried as a constraint.
- `ignition/components/split_draft.py:31` — confidence recorded from learned tokens before substitution biases packed prefix allocation for copies; `helios/src/scheduler.rs:2297` — acceptance learning ignores the copy mask. Both pre-existing; recorded in `docs/TODO.txt`.
- TP consistency, draft cursor, float32 one-hot q, `-no-mtp`/width-1 graphs, and `ProposalSource::Copy` — hunt completed, no latent defects of these classes found.
