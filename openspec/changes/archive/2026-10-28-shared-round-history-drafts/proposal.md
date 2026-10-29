# Port history drafts to the shared parallel-draft round

Tier: 3
Type: feature

## Stated Requirements

- "What is the optimal way to begin porting the history drafts?  refer to docs/ for additional suggestions and requirements."
- "why would we implement Dflash2 without history drafts?"
- "I agree with this being a separate recipe."
- "my usage is almost entirely code generation."
- "the server is for a single developer.  Typically the concurrency is <8"

## Why

`flare run --history-drafts` copies continuations from the decoded-history ring into speculative rounds, but only Qwen3.8's MTP round applies them: the point-mass substitution lives in `ignition/recipes/qwen38/speculative_model.py`, so every recipe on the shared `PreparedParallelDraftRound` — including the planned DFlash2 recipe — is rejected at load. The native side is already shared; this port moves the Python side to match, before the DFlash2 recipe is built on it.

## What Changes

- `PreparedParallelDraftRound` owns the four copy input buffers (`copy_token_ids`, `copy_token_count`, `copy_expected_consumed`, `copy_generation`), substitutes point-mass proposals for the copied prefix after every drafter path has produced its proposals, keeps the learned proposals after the copied run offered with their retained q, and hands the `copy_offered_mask` to its commit.
- The shared commit bounds the recorded copy mask by the offered-draft count it derives from the request's token limit, and Qwen's copy gate gains the missing token-limit term — closing a latent serving-loop failure (a long history hit on a lane near its `max_tokens` budget trips the runtime's mask validation).
- Helios's history lookup offers at most the lane's verifiable token count instead of the graph width.
- MiMo V2.6 Pro and Nemotron Bolt declare the copy fields on their speculative graphs so the contract binds them and `flare run --history-drafts` admits their packages.
- Packed split rounds (DeepSeek V4.1, MiniMax) carry the source round's copy mask into the target commit so the metric stays honest there too.
- Qwen3.8's MTP round is untouched; its behavior is carried.
- Recipes that do not declare the fields keep today's behavior: the buffers stay at their inert initial values and no substitution occurs.
- Docs: the history-drafting paragraph moves from the Qwen section to the shared speculation description and names the recipes that support it.

## Capabilities

### New Capabilities

- `history-drafts`: copied continuations from shared decoded history as point-mass proposals inside any speculative round built on the shared parallel-draft components.

### Modified Capabilities

- none — `openspec/specs/` holds only `dflash2-acceptance-measurement`, which does not change.

## Impact

- `ignition/components/parallel_draft.py`, `ignition/components/packed_speculation.py`, `ignition/recipes/mimo_v2/bootstrap.py`, `ignition/recipes/nemotron_h/bootstrap.py`, tests under `tests/components/` and `helios/src/model/tests.rs`, and four docs. No Protobuf, ABI, contract-version, or Helios runtime change: the bindings and the admission gate already exist.
- Validation needs one GPU for the shared-round CUDA tests and, for the served check, an export of MiMo or Nemotron; those packages are not present on this host today.

## Blast Radius

1. `ignition/components/parallel_draft.py:311` — `propose()`; substitution goes after `proposal_block.finish` at `:379` and before `candidates[:, 1:]` at `:383`, so every drafter path (sequential, fused Markov, greedy argmax, broadcast) is covered by one pass.
2. `ignition/components/parallel_draft.py:286` — `PreparedSpeculativeCommit(...)` built without `copy_offered_mask`; `draft_context_lag=0` at `:301` stays.
3. `ignition/components/speculation.py:405` — commit accepts `copy_offered_mask`; `:522` writes it to record field 11; `:524` writes the draft cursor.
4. `ignition/recipes/qwen38/speculative_model.py:52` — the copy buffers and `:270`–`:280` the substitution being generalized; `:73` publishes them as graph buffers. Qwen keeps this code.
5. `ignition/preparation/contract.py:42` — copy bindings emitted for `speculative`/`mixed` kinds; `:72` binds only fields the graph signature lists, which is why undeclared recipes are inert.
6. `ignition/recipes/nemotron_h/bootstrap.py:238` and `ignition/recipes/mimo_v2/bootstrap.py:301` — speculative graph field tuples that gain the four copy names.
7. `ignition/components/packed_speculation.py:112` — source→target handoff of `candidates`/`drafts`; the copy mask joins that list.
8. `helios/src/model/catalog.rs:88` — rejects `history_drafts` unless the contract carries the copy inputs; `helios/src/model/packing.rs:657` fills them for any `Speculative` lane; `helios/src/execution/transport.rs:65` validates the mask against `offered_drafts`; `helios/src/server/metrics.rs:405` attributes per-depth acceptance to copies. All unchanged.
9. `helios/src/model/tests.rs:613` — `copied_proposals_require_concrete_graph_inputs`; a positive case for a contract that declares the inputs is added beside it.
10. `tests/recipes/test_qwen38_speculative_state.py:121` — the copy-epoch/expected-consumed/mask test pattern to port; `tests/components/test_packed_speculation_cuda.py:540` — the shared-round harness to extend.
11. `docs/mtp-design.md:418`, `docs/cli-design.md:183`, `docs/model-preparation.md:344`, `docs/api.md:264`, `docs/scheduling-runtime-design.md:26` — the point-mass law, the flag contract, the Qwen-only paragraph, the metric, and the copy-depth accounting.
12. grepped 'PreparedParallelDraftRound(' across `ignition/recipes/` — also DeepSeek V4 (`ignition/recipes/deepseek4/bootstrap.py:384`), DeepSeek V4.1 (`ignition/recipes/deepseek41/bootstrap.py:251`), Kimi K3 (`ignition/recipes/kimi3/bootstrap.py:357`), MiniMax (`ignition/recipes/minimax_m3/bootstrap.py:358`), Gemma 4 (`ignition/recipes/gemma4/bootstrap.py:425`), Step 5 (`ignition/recipes/step5/bootstrap.py:262`); they share the substitution code but declare no copy fields in this change, so their packages keep rejecting `--history-drafts` until each is qualified.
13. grepped 'copy_token_ids' across `ignition/recipes/` — only `ignition/recipes/qwen38/speculative_model.py:73` publishes the buffers today; no other site.
14. `ignition/components/speculation.py:319` and `:347` — a used position with `valid=False` rejects the whole row, and `offered_drafts` derives only from `token_limits`; the reason truncation cannot ride on `valid` and the place the mask bound lands (`:522`).
15. `helios/src/model/packing.rs:368` — history lookup maximum is `token_capacity − 1`, the graph width, not the lane's tokens.
16. `ignition/components/split_draft.py:31` and `helios/src/scheduler.rs:2297` — confidence recording and acceptance learning that ignore copies; pre-existing, recorded in `docs/TODO.txt`, untouched.
17. grepped 'lessons' across `openspec/` — no `lessons.md` exists; nothing to consult.
