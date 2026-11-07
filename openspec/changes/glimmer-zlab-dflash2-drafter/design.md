# Design — glimmer-zlab-dflash2-drafter

## Approach

- `draft_config.py`: pin `z-lab/Muse-Glimmer-30B-DFlash2` at `b54ffdd11fa9cfe2af370012e5763d492c904128`; `from_dict` asserts the DFlash2 form in the shape of `ignition/recipes/qwen38_dense/draft_config.py` (model_type qwen3, `DFlash2DraftModel`, 6656/19968, 32/8 × 128, 5 sliding layers, window 2048, rope 500000, `num_target_layers` 52, `dflash_config` block 16, conv 2/16, selector 256/16, taps [1, 13, 25, 37, 49], mask 201818) and that `dflash_config` `output_multiplier`/`final_logit_softcapping` equal the target's; `target_layers`, `block_size` and `noise_token` read `dflash_config`.
- `weights.py`: drop `projector="dflash"` so `DraftWeights.load` takes its `dflash2` default and reads `fc`, `hidden_norm`, the convolutions and the selector; the context taps, rings and `DraftScratch` sizing already read block size and geometry from the config.
- `dflash.py`: `PreparedDraft` takes `output_multiplier` and `final_logit_softcapping` from `dflash_config` as `ignition/recipes/muse_spark/dflash.py` does; the bare-head override and its docstring go. The round driver sees a non-`None` selector and traces the block as it does for Spark.
- `recipe.py` / `bootstrap.py`: `draft_bytes` 5,544,328,424 (the z-lab `model.safetensors`), docstring and comments; `MAX_WIDTH` 16 and `draft_files` stay.
- Tests: `test_config.py` carries the z-lab `config.json` and DFlash2 faults; `test_draft_weights.py` asserts the 81-tensor set through the default projector and keeps the projector-switch guard; `test_dflash_cuda.py` replaces the Transformers assistant oracle with the fork's own DFlash2 reference (context taps through `fc`, five layers with two-tap convolutions, selector, head transform) in the shape of `tests/recipes/muse_spark/test_dflash_checkpoint_cuda.py`, reads `FLARE_MUSE_GLIMMER_DFLASH2`, and keeps the acceptance assertion.
- Package identity already hashes the draft config text and the draft shard headers (`bootstrap.py` identity dict), so the new drafter forces a fresh export and cannot reuse the assistant package.
- Stays: the projector-less `dflash` path in `ignition/components/dflash2*.py` (no in-repo user after this change; its removal is a separate decision); the alias, recipe name, served name, speculation declaration (15 drafts, capacity 64, budget 64) and the continuation layout (five draft rings of 2048 × 8 × 128, unchanged geometry).
- GPU window (last task): WIP commit, `local/flare-ignition:<sha8>` built from it, console profile `2026-11-06-flare-muse-glimmer-1793992412838` set to `--recipe muse_glimmer_nvfp4`, `--draft-model z-lab/Muse-Glimmer-30B-DFlash2` and the new image, launched once; served chat with accepted drafts above zero is the operator leg; the vault notes follow.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.95
Caller Agreement: 0.9
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.95
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.95
- local-ai-qllm (qLLM) — cancelled (quorum met first), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "do not keep both drafters selectable, as the community DFlash2 supersedes the first-party assistant and leaves dead code. The draft head must apply the target's output multiplier (0.196) and softcap (20) as declared in the config" — ACCEPTED: one pinned drafter; the head transform comes from `dflash_config`; the GPU-1 acceptance test (task 2.1) is the arbiter before any export.
- "console profile draft-model fields, and caching/package identity mechanisms" — ACCEPTED: the identity hash already covers the draft config and shard headers (Approach); the profile's draft-model and recipe fields and the image tag change in task 3.1.
- "stale package identity/cache reuse" and "mask-token embedding mismatches (token 201818)" — ACCEPTED: identity re-keys by construction; the mask embedding stays the raw table row through `PreparedShardedEmbedding`, and the acceptance test fails if fine-tuning changed its semantics.
- "purging unreferenced assistant paths appropriately" — REJECTED: the projector-less variant in the shared components stays (proposal Assumed 3); its removal is a separate decision and the projector-switch guard test keeps it validated.
- "recomputing static tensor memory allocations in Rust" — REJECTED: Helios reads every size from the package; the draft ring geometry is unchanged, so `continuation_layout` and the declared rank costs stay, and `draft_bytes` is an ignition declaration checked by the declaration test.
