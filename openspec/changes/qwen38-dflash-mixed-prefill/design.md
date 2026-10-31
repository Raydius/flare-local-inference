# Design — qwen38-dflash-mixed-prefill

## Approach

Motivation and scope: see proposal.md. The mixed round already packs one prompt chunk and a speculative batch into one Qwen target forward (`ignition/recipes/qwen38/mixed_model.py:35`); this change lets a second drafter plug into it without changing the MTP path.

1. **Region-keyed residual checkpoint.** `PreparedDecoderState.prepare_residual` (`ignition/recipes/qwen38/decoder_state.py:270`) compiles the `target_residual` checkpoint only when `maintain_draft_state` is set and the pool layout holds a `target_residual` region. Today it keys on `speculative_tokens`; every layout with `speculative_tokens and mtp` has the region (`ignition/recipes/qwen38/state.py:82`), so qwen38 and qwen38_dense compile the same kernels.
2. **Drafter-supplied halves in `PreparedMixedRound`.** Two keyword arguments, `absorption=PreparedPrefillAbsorption` and `speculative_round=PreparedSpeculativeRound`, replace the hard-coded constructors at their single call sites. Their call shapes stay as today: `absorption(prompt_owner, prompt_decoder, name, residual_output=...)` with `begin()`/`finish()`, and `speculative_round(decode_owner, batch, width, name, decoder_factory=..., residual_output=...)` with `prepare()`/`__call__`. The qwen38 bootstrap call stays byte-identical.
3. **Feature taps through the packed forward.** `decoder_factory` already forwards `**kwargs` to `PreparedDecoder`. When the round passes `feature_taps` and a decode-sized `auxiliary`, the factory allocates a packed-size and a prompt-size feature buffer from the drafter's graph cursor, hands the packed buffer to the decoder, and gives `PreparedMixedState` both role buffers. `PreparedMixedState.finish` gathers packed features into the prompt and decode buffers with the existing `StatePartition.gather`, as it does for hidden and residual. Taps are written before `state.finish()` (`ignition/recipes/qwen38/decoder.py:395`, `:405`), so the gathered rows are complete. Without taps nothing is allocated or gathered, so MTP is unchanged.
4. **DFlash halves.** In `ignition/recipes/qwen38_dflash/speculative_model.py`:
   - `PreparedDFlashRound` gains `decoder_factory=PreparedDecoder` and `residual_output=None`, and builds its target through the factory. It registers its decoder in `owner.decoders` only when it built a plain `PreparedDecoder`; the mixed round's packed decoder is registered by the bootstrap loop, as qwen38 does.
   - `PreparedMixedPromptContext` is the absorption: `begin()` does nothing, and `finish()` writes the gathered prompt features into the prompt request's rings with `prefill_feature_rows` over the prompt state's QSA metadata. That is the write `PreparedPrefillContext` does after an ordinary prefill.
   - The owner exposes `sampling_workspace` so the mixed round's prompt sampler shares the round's sampling workspace, as MTP does.
5. **DFlash bootstrap.** `qwen38_dflash/bootstrap.py` drops the `mixed_prefill` rejection and follows the qwen38 catalog loop at `ignition/recipes/qwen38/bootstrap.py:458`. It allocates `MixedScratch`, adds one sample row when mixed shapes exist, counts mixed graphs, passes `prefill_capacity` to the signature, and reports `mixed_shapes`. The recipe declares `mixed_prefill=True`. The no-draft mixed graphs use the target-only owner namespace, as qwen38 does without MTP.
6. **Export.** `qwen38_dflash/export.py` already clears `sampling_buffers` chunks, which hold the new feature buffers. It adds the mixed round's per-graph sampling scratch the same way it lists `PreparedDFlashRound` today.

Constraints carried from AGENTS.md: no Helios, contract, or package-format change; no recipe-identity branches in shared code; no new captured-graph operation family, since `mixed` already exists; new buffers are allocated, never aliased onto decoder scratch; `MixedPacking` and `MixedScratch` stay unchanged.

Validation: CPU unit tests extend `tests/recipes/test_qwen38_mixed.py` and `tests/recipes/qwen38_dflash/`; the CUDA mixed test `tests/recipes/test_qwen38_mixed_cuda.py` gains a DFlash-enabled case comparing ring writes and commits against a split prefill-then-round run; an export of `qwen3.8-flash-dflash2` with mixed prefill runs through `flare run` on GPU 0.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/2, escalated to tier 1)
Certainty: 0.78
Caller Agreement: 0.9
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.7
- local-ai-qllm (qLLM) — ok, agreement n/a
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-flash (deepseek-flash) — ok, agreement 0.95

### Review Feedback

Consensus feedback recorded verbatim:
- "Have PreparedMixedRound compose drafter-supplied prompt-absorption and speculative-round halves plus a target decoder factory, so DFlash can use the packed decoder. The MTP adapter returns the existing classes in the same order, keeping MTP capture byte-identical." — ACCEPTED: this is Approach steps 2 and 4; the defaults are today's classes.
- "Taps are written before state.finish(), which gathers hidden, residual, GDN/PLE metadata and taps by role." — ACCEPTED: Approach step 3.
- "DFlash prompt bookkeeping writes only the prompt's last 2048 tapped positions into its rings using absolute-position mapping, and ring writes happen only on commit." — ACCEPTED: the prompt write reuses `prefill_feature_rows` (absolute positions, slot = position mod 2048); decode writes stay limited to emitted columns via `verification_feature_rows`.
- "Preallocate all tap and ring scratch at capture, keep it disjoint from decoder and existing mixed scratch, and use no new graph op kinds or steady-state allocation." — ACCEPTED: buffers come from the drafter's own cursor; a test asserts their storage ranges are disjoint from decoder and `MixedScratch` storage.
- "Guard against MTP drift with golden region-table and graph-signature diffs, scratch-overlap assertions at capture, ring rollover and preemption handling, fixed padded capture buckets, and memory-report and disposable-buffer updates." — ACCEPTED: tasks add a qwen38 signature/region identity test, the overlap test, a prompt chunk longer than the window in the CUDA mixed test, reuse the shared `mixed_shapes` buckets, and cover the memory report and export ranges. Preemption needs no new handling because rings are plain state rows restored with draft lag 0.
- Certainty 0.78 is below the 0.9 floor — ACCEPTED: the four peers agree on the approach and differ only in emphasis on guards, which are all adopted above; caller agreement 0.9 meets its floor.
