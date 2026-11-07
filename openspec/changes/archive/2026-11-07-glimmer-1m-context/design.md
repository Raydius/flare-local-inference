<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Design — glimmer-1m-context

## Approach

- Declare, don't scale: `ignition/recipes/muse_glimmer_nvfp4/recipe.py:66` `context=1048576` and `bootstrap.py:49` `CONTEXT = 1048576`; a comment on each records the native 131,072 anchor and why no `RopeScaling` applies (39 sliding layers bounded at rope distance 2048, 13 full layers without positional encoding).
- The raise lands on `Recipe.context`, not on a second checkpoint alias: `ignition/recipes/declaration.py:303` refuses `Checkpoint.context` without `rope_scaling`, and widening that check for one NoPE recipe would touch every recipe's declaration path for no second caller.
- Memory is page-bound, not declaration-bound: `ignition/planning.py:325-332` serves `min(declared, pages × 128)`; at 6,656 B/token the 1M window costs 6.5 GiB per request against ~57 GiB left after weights, drafter, graphs and four state rows on the 96 GB RTX PRO 6000, so `demand_pages` (32,768) fits and no precision or TP change is needed.
- Nothing else is position-indexed by the declared size: rotary is computed from `positions.float()` (`ignition/recipes/muse_spark/attention.py:172`), page-table columns clamp to the table width (`:180`), `flare/src/serving_overrides.rs:11` only lowers.
- Pinning tests move with the number: `tests/recipes/muse_glimmer_nvfp4/test_preparation.py:15,34` and `test_config.py:99`.
- Probe widening (`qualify/long_context_passkey.py`): `prompt(..., template_kwargs, max_tokens)` plus `--template-kwargs JSON` (default `{"enable_thinking": false}`), `--max-tokens N` (default 32) and `--tokens-per-record FLOAT` (default `TOKENS_PER_RECORD`); every existing call keeps today's request bytes (`tests/reference/test_long_context_passkey.py` asserts the defaults and the new overrides). The Glimmer arm sends `{"reasoning_strength": "low"}` and 256 tokens because the Glimmer template reasons in the `to=self` channel before content, and the pass criterion stays "content contains the passkey".
- Catalog sizing for Glimmer: `TOKENS_PER_RECORD` (66.63) is Qwen-calibrated; the pinned Glimmer `tokenizer.json` renders the same catalog at 56.54 tokens per record (measured offline on `_catalog(records_for(1_000_000))`: 15,008 records → 848,527 tokens), so `records_for` gains a `tokens_per_record` parameter and the probe a `--tokens-per-record FLOAT` flag (default the Qwen constant); the Glimmer arm passes 56.54 and runs lengths 100k / 200k / 320k / 520k / 1M (the 1M cell lands ≈1,000,000 + prompt overhead, 4.6% under 1,048,576); the summary's `prompt_tokens` per cell is the recorded truth.
- GPU legs are one operator window (memory rule: one stop of the serving instance): console-built image at the landed commit → console export `muse-glimmer-30b-nvfp4 --gpus 0` → probe run with `--image local/flare-ignition:<sha8>`, no `--max-context-tokens`, `--lengths 100000,200000,320000,520000,1000000 --tokens-per-record 56.54 --template-kwargs '{"reasoning_strength": "low"}' --max-tokens 256 --timeout 1800 --ready-timeout 1800 --model-id Muse-Glimmer-30B` → results committed under `qualify/results/glimmer-passkey/summary.json`.
- `docs/model-preparation.md:1107` and the `muse-glimmer-serving` spec's declared capacities move to 1,048,576 with the native anchor named; the `long-context-passkey-retrieval` spec's request-construction requirement gains the two parameters and drops "Qwen3.8" from its purpose.

## Architecture / Requirements

### Reference architecture

Archetype: declared-ceiling context extension of a sliding-window + NoPE hybrid decoder, with retrieval probing as the acceptance gate.
Anchor: the vault's "Muse Glimmer 256k" vLLM profile (2026-10-11) — same model, same reasoning: no YaRN, raise the declared `max_position_embeddings`, measure needles — and the repo's own `long-context-passkey-retrieval` spec, which fixed the 200k/320k/520k/1M × 0.1/0.5/0.9 grid for the Qwen yarn4 package.
Invariants: served context never exceeds pages held (`ignition/planning.py:325-332`, pinned by the plan report); the package's `max_sequence_tokens` equals the declared context (`tests/recipes/muse_glimmer_nvfp4/test_config.py:99`); the probe's default request bytes are unchanged (`tests/reference/test_long_context_passkey.py`); a passkey cell passes only when content carries the key.
Rejected approach: a second checkpoint alias `muse-glimmer-30b-nvfp4-1m` with `Checkpoint.context` — refused by `check_rope_scaling` without rope scaling; admitting it means a shared-declaration change with one caller.
Rejected approach: YaRN / `--rope-parameters` factor 8 — `ignition/cli/preparation.py:63-64` refuses it for a recipe with no scaling declaration, and correctly: the only rope layers are 2048-window sliding layers that never extrapolate.
Deviations: none.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.9
Caller Agreement: 0.85
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.9
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.9
- local-ai-qllm (qLLM) — cancelled (quorum met first), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Extend served context to 1M by raising the recipe's declared context and bootstrap constant to 1,048,576 … Do not use rope scaling or YaRN … Do not use runtime flags for extension, as they only lower context." — ACCEPTED: the approach as drafted.
- "re-calibrate catalog sizing against Glimmer's actual tokenizer rather than Qwen's" — ACCEPTED: the probe gains `--tokens-per-record FLOAT` (default the Qwen constant); the Glimmer value was measured offline against the pinned `tokenizer.json` and the arm passes it, so the 1M cell is a real 1M cell instead of a 960k guess.
- "Adjust `max_tokens` upward, configure the reasoning channel appropriately" — ACCEPTED: `--max-tokens 256` and `--template-kwargs '{"reasoning_strength": "low"}'` on the Glimmer arm.
- "treating the native 131k as a baseline" — ACCEPTED: the Glimmer grid adds a 100k length below the native 131,072 as the in-context baseline row.
- "1M prefill on 13 full-attention layers will cause very high TTFT and memory spikes" — ACCEPTED: the probe's `--timeout` and `--ready-timeout` are raised for the Glimmer arm (1800 s); memory is page-bound and already fits.

## Architecture Review

### Architecture Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.8
Caller Agreement: n/a
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.9
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.7
- local-ai-qllm (qLLM) — cancelled (quorum met first), agreement n/a

### Architecture Review Feedback

Consensus feedback recorded verbatim:
- "It is the standard form for this hybrid architecture, and the deviations (such as template/token additions for the Glimmer-specific reasoning channel …) are justified by the model's design." — ACCEPTED: no change.
- "the 960k sizing" as a deviation — ACCEPTED: removed; `--tokens-per-record` makes the 1M length literal (see Peer Review Feedback).
Violated requirement: "the existing passkey probe run against the re-exported Glimmer package at 200k / 320k / 520k / 1M" — the draft's 960k substitute violated it; reconciled by sizing the catalog with the Glimmer tokens-per-record so the grid is 100k / 200k / 320k / 520k / 1M as stated.
