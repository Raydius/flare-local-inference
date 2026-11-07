<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Serve Muse Glimmer at a 1,048,576-token context and measure passkey retrieval past 131k

Tier: 2
Type: feature

External Review: design.md — Peer Review Verdict

## Stated Requirements

- "Based on the current state of Muse Glimmer implementation, what is the optimal way to scale the context window to something much higher (500k-1M tok)?" (operator, rad-explore, 2026-11-06)
- Assumed: the declared context of the `muse_glimmer_nvfp4` recipe rises from 131,072 to 1,048,576 with no rope scaling — Glimmer's 39 sliding layers never see a rope distance past 2048 and its 13 full layers carry no positional encoding, so there is nothing to interpolate (checkpoint `config.json`: `layer_types` 39 sliding / 13 full, `sliding_window` 2048, `rope_parameters.rope_type` default, `rope_scaling` absent).
- Assumed: 1,048,576 is the ceiling; operators serve smaller windows with the existing `--max-context-tokens`, which only lowers (flare/src/serving_overrides.rs:11).
- Assumed: quality past 131,072 is unknown until measured, and the measurement is the existing passkey probe (`qualify/long_context_passkey.py`) run against the re-exported Glimmer package at 100k (native baseline) / 200k / 320k / 520k / 1M × depths 0.1 / 0.5 / 0.9, recorded under `qualify/results/glimmer-passkey/`; a failed cell is a recorded result, not a blocker on landing the declaration.
- Assumed: the export and the probe run on GPU 0 inside the console-launched `local/flare-ignition:<sha8>` image built from a committed ref, in one operator-opened downtime window; the agent launches neither.
- Assumed: the DFlash2 drafter rides unchanged — its five layers are sliding (window 2048) and its context rings are 2048 positions (`ignition/recipes/muse_glimmer_nvfp4/draft_config.py`), so the longer window touches it only through acceptance, which the probe records per cell.

## Blast Radius

1. ignition/recipes/muse_glimmer_nvfp4/recipe.py:66 — `context=131072` is the plan's `declared_context` (ignition/planning.py:322) and the package's `max_sequence_tokens` (ignition/recipes/muse_glimmer_nvfp4/bootstrap.py:325); becomes 1048576.
2. ignition/recipes/muse_glimmer_nvfp4/bootstrap.py:49,91 — `CONTEXT = 131072` caps `PreparationConfig.context`; rises with the recipe.
3. ignition/recipes/declaration.py:301-309 — `check_rope_scaling` refuses a checkpoint `context` without `rope_scaling`, so the raise lands on `Recipe.context`, not on a second alias; read-only.
4. ignition/planning.py:281-282,325-332 — `demand_pages = concurrency × declared_context ÷ 128` becomes 32,768 pages (26.0 GiB at 851,968 B/page); `context = min(declared, pages × 128)` and `guaranteed_tokens` already clamp to the pages the GPU holds; read-only, the plan report changes.
5. ignition/cli/serve.py:70 — `max_body_bytes = declared_context × 32` grows from 4 MiB to 32 MiB plus images; read-only.
6. ignition/recipes/muse_spark/attention.py:172,180-184 — rotary from `positions.float() × inverse` (no table) and the full-layer page-table column `clamp(max=shape[1]-1)`; the only position-indexed paths, both unbounded by 131072; read-only.
7. tests/recipes/muse_glimmer_nvfp4/test_preparation.py:15,34 — pins `context == 131072` and rejects `131072 + 128`; moves to 1048576 / 1048576 + 128.
8. tests/recipes/muse_glimmer_nvfp4/test_config.py:99 — `RECIPE.context == 131072`; moves to 1048576.
9. docs/model-preparation.md:1107 — "a 131072-token context" in the Glimmer section; reworded with the native 131,072 anchor and why no rope scaling applies.
10. openspec/specs/muse-glimmer-serving/spec.md:88 — "Declared capacities and costs match the prepared layout" says context 131072; delta modifies it.
11. qualify/long_context_passkey.py:48-55 — the request hard-codes `chat_template_kwargs={"enable_thinking": False}` and `max_tokens=32`; Glimmer's template reasons by default (`reasoning_strength` defaults to `high` in the pinned checkpoint template, mirrored by tests/frontend/test_muse_glimmer.py:158) and would spend 32 tokens in the `to=self` channel, so the probe gains `--template-kwargs JSON` (default today's `{"enable_thinking": false}`), `--max-tokens N` (default 32) and `--tokens-per-record FLOAT` (default the Qwen constant), with the Glimmer arm sending `{"reasoning_strength": "low"}`, 256 and 56.54.
12. qualify/dflash2_long_context.py:29 — `TOKENS_PER_RECORD = 51172/768` is calibrated on the Qwen tokenizer; the pinned Glimmer `tokenizer.json` renders the catalog at 56.54 tokens per record (measured: 15,008 records → 848,527 tokens), so `records_for` gains a `tokens_per_record` parameter.
13. tests/reference/test_long_context_passkey.py:12-22 — asserts the request's kwargs and `max_tokens`; gains the two parameters.
14. openspec/specs/long-context-passkey-retrieval/spec.md:5,11,16 — purpose names Qwen3.8 and the request fields pin `enable_thinking`/32; delta modifies the request-construction requirement.
15. grepped '131072\|131_072' across ignition/recipes/muse_glimmer_nvfp4 tests/recipes/muse_glimmer_nvfp4 helios/src/frontend/images/muse_glimmer.rs — only sites 1, 2, 7, 8; grepped 'max_position_embeddings' across ignition/recipes/muse_glimmer_nvfp4 — none (the recipe never reads it); grepped 'enable_thinking' across qualify/ — qualify/long_context_passkey.py:51 and qualify/dflash2_long_context.py (its own workload, untouched).
