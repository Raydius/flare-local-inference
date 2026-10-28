# Design — qwen38-dflash2-acceptance-spike

## Context

See proposal.md — Why. The target's canonical reference (`ignition/reference/qwen38_decoder.py:348`) already publishes every completed layer's 10240-wide four-stream residual; `qualify/mtp.py:186` already runs the prepared MTP module teacher-forced from stored residuals with cache-only prefix absorption; `qualify/speculative_workload.py` freezes ~20 prompts and `qualify/speculative_benchmark.py` streams token ids from a running server. The draft is a single BF16 safetensors (81 tensors) with no embedding of its own; the target's `embed_tokens` and `lm_head` are dense BF16.

## Goals / Non-Goals

- Goal: one number per (drafter, width, workload category) — mean greedy teacher-forced acceptance length — produced by a committed, re-runnable tool and backed by tests that detect a wrong port.
- Goal: both drafters measured from the same traces, same anchors, same acceptance rule.
- Non-goal: served throughput, captured graphs, stochastic (sampled) acceptance, TP>1, image prompts, history drafts.

## Approach

1. `qualify/dflash2_acceptance.py` — three subcommands sharing one JSON/`.pt` trace format:
   - `generate`: sends the frozen workload prompts (text-only cases) to a running Flare server at `temperature 0`, `max_tokens` ≤ 256, `return_token_ids`, through `qualify.speculative_benchmark`'s streaming parser; writes prompt and continuation token ids per case.
   - `trace`: one `CanonicalTextReference` prefill per case over prompt+continuation with `layer_output` capturing layers 5, 15, 25, 35, 45 (view `(4, 2560)`, mean over streams → 12800 per token) and layer 47 (10240, the pre-mixer residual MTP consumes); saves per-case tensors to a cache directory, skipped when the file exists. Causality makes prefill features at position i identical to decode-time features.
   - `measure`: simulates rounds from the prompt end: draft k proposals (k = width − 1 for widths 4 and 8), accept the longest prefix equal to the teacher tokens, advance by accepted + 1; acceptance length = mean(accepted + 1) per round. Runs the DFlash2 arm and the MTP arm on the same traces and writes JSON with per-category and overall means, round counts, and the trace cache digest.
2. `ignition/reference/qwen38_dflash2.py` — `DFlash2Reference`: strict `config.json` validation against the pinned geometry (architecture `DFlash2DraftModel`, block 8, 5 sliding layers, 32/8 heads × 128, window 2048, hidden 2560, vocab 248320, taps `[5,15,25,35,45]`, mask token 248077, selector rank 256, top-k 16, conv kernel 2, group 16, rope θ 10000); `safe_open` loader; `context(features, positions)` → `hidden_norm(fc(features))` then per-layer `k_proj`/`k_norm`/rope and `v_proj`; `block(pending, position, context)` → target embedding of `[pending, mask×7]` at positions t..t+7, five layers of `input_norm → attention_conv.prepare → GQA attention over context window (≤2048 most recent) + block, noncausal within the block → attention_conv.finish → residual; post_norm → mlp_conv.prepare → SiLU MLP → mlp_conv.finish → residual`, final `norm`; `candidates(hidden, lm_head)` → top-16 ids and logits per block position 1..7; `walk(...)` → sequential per-position argmax of `unary + ⟨A[pred] ⊙ P·h, B[c]⟩` with `pred` the previously chosen token and the pending token as anchor. Ties break to the lowest id, matching `torch.argmax`.
3. MTP arm in `measure`: reuse `qualify.mtp.flare`'s setup (prepared MTP, cache-only absorption of the prefix rows from layer-47 residuals), then seven recursive rows feeding `draft.scratch.residual` as the next row's hidden as `ignition/recipes/qwen38/speculative_model.py:233` does; greedy argmax per row.
4. `tests/reference/test_qwen38_dflash2.py` — CPU on fabricated weights: geometry rejection, output shapes, mask-token substitution, noncausal-within-block and window-limit attention (compare against a dense masked reference), walk equals brute-force best path on a 3-position lattice; CUDA (`FLARE_RUN_CUDA_TESTS=1`): reference convolution vs `PreparedGroupedConvolution`, walk vs `PreparedCandidateSelector`, and `measure` on one cached trace with a **shuffled-feature control** (taps permuted across positions) whose acceptance must collapse toward 1.0 while the real run stays above 2.0 at width 4.
5. `docs/TODO.txt` — one entry: the DFlash2 recipe decision waits on the JSON this tool writes.

## Decisions

- One canonical prefill per case rather than per-token canonical decode — a 48-layer eager decode per token is hours per prompt; prefill gives identical features by causality. Alternative rejected: prepared decoder with `layer_outputs` — needs state-pool setup and touches `ignition/recipes/qwen38/`.
- Greedy continuations from a served package rather than the canonical loop — minutes, and the teacher tokens are the target's real greedy output; both arms share them, so a rare canonical/NVFP4 argmax disagreement costs both equally. Alternative rejected: canonical decode loop — same cost objection as above.
- Greedy teacher-forced acceptance rather than stochastic — deterministic, graph-free, identical for both arms; served acceptance under sampling is a follow-up measurement once a recipe exists.
- Sequential walk rather than a batched lattice — equals the published greedy `_follow_maps` walk and matches `ignition/components/candidate_selector.py:19` per position, which the recipe would reuse.
- Trace cache keyed by case id and continuation digest — rerunning `measure` after a reference change costs no canonical forward.

## Risks / Trade-offs

- [Tap semantics wrong (0/1-based, pre/post-mixer, stream mean)] → fix by the model card's SGLang capture patch: completed, zero-based layer outputs, mean over `hc_count` streams; assert the hook tensor is 10240 wide; the shuffled-feature control catches a dead feature path.
- [BF16 numerics and argmax ties diverge from the trained runtime] → compute attention and selector scores in FP32 from BF16 operands; ties break to lowest id in both arms.
- [Canonical prefill memory at ~800 tokens] → the reference streams checkpoint tensors under its byte budget; cap `max_tokens` at 256 and skip the two `long_context` cases.
- [Eager runtime] → trace once and cache; `measure` is draft-only work (≈1 draft forward per round).
- [A served package is a prerequisite for `generate`] → the operator's deploy branch package; `generate` is optional when a tokens file is supplied.

## Open Questions

- Whether acceptance at width 8 benefits from the draft's full 7 proposals or saturates near width 4 — answered by the measurement itself.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.8
Caller Agreement: 0.6
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — cancelled, agreement n/a
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Compare both drafters offline on the same stored canonical target traces: the teacher tokens and the five tapped layer outputs (5, 15, 25, 35, 45, averaged over the four streams). Use the same anchors, pending tokens, and greedy acceptance rule (accepted length = 1 + longest matching prefix of drafts against target greedy argmax), at width 4 (3 proposals) and width 8 (7 proposals). MTP runs its recursive head from stored residuals. DFlash2 runs one noncausal block pass from the taps, with mask tokens from the target embedding. Run eager with no CUDA graphs and no server." — ACCEPTED: this is the Approach; `generate` is the one server touch and is optional when a tokens file is supplied.
- "Detect a wrong port with synthetic or shuffled-feature controls (acceptance should collapse) and unit tests of the grouped convolution and candidate selector against the existing Triton components." — ACCEPTED: added the shuffled-feature control and the two component-parity CUDA tests to Approach item 4; this closes the gap behind the 0.6 caller agreement and the 0.8 certainty, which rated a draft lacking the control.
- "Guard against masking and window errors, BF16 numerics and argmax ties, 0- versus 1-based layer indexing and tap semantics, memory from trace size and 2048-token K/V windows, and eager runtime by capping length and caching traces." — ACCEPTED: each is a Risk line above with its mitigation; FP32 scores and lowest-id ties added.
