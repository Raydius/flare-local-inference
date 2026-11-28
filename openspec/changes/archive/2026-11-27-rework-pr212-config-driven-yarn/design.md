# Design — rework-pr212-config-driven-yarn

## Context

See proposal.md — Why. Three facts on the current upstream code shape the design:

- The Qwen recipe's `RotaryTables.allocate` already reads `text_config.rope_parameters` and builds the FP32 cos and sin tables every rotation kernel gathers from.
- The preparation identity already hashes the bytes of `config.json`.
- Core already calls `recipe.inspect_replacement` when a user passes an unregistered checkpoint directory, and the hook returns an adjusted `Checkpoint`.

## Goals / Non-Goals

- Goal: one mechanism, selected by the checkpoint's own config, that matches the transformers YaRN reference including the attention factor.
- Goal: no change to any kernel, gather, fused arithmetic, or scratch sizing.
- Non-goal: SM103 serving at a scaled context (refused, not implemented).
- Non-goal: moving the fork to the config-driven mechanism; its alias and flag stay.
- Non-goal: a CLI flag or configuration-file key for rope parameters.

## Approach

Two pull requests carry the code; a third piece, the evidence tooling, lands on the recipes fork's `deploy` branch.

| Piece | Repository and branch | Files |
|---|---|---|
| Core | this repository, `pr/rope-scaling`, rebuilt on `upstream/main` | `ignition/declaration.py`, `ignition/planning.py`, their tests, `docs/TODO.txt` |
| Recipe | `~/dev/hyperflare-recipes`, new `pr/qwen38-config-yarn` on recipes `upstream/main` | `qwen38/attention.py`, `checkpoint.py`, `checkpoint_schema.py`, `bootstrap.py`, `recipe.py`, their tests, `docs/model-preparation.md` |
| Evidence | recipes fork `deploy` | `qualify/long_context_passkey.py`, its test, `qualify/results/yarn-config/` |

## Decisions

### D1 — Scaling is selected by the checkpoint directory's config

A user serves a directory whose `config.json` carries `text_config.rope_parameters` with `rope_type: "yarn"` and a matching `max_position_embeddings`; weights may be symlinks.
This is the maintainer's stated requirement and how vLLM and SGLang take the checkpoint's own `rope_parameters`.

- Alternative, a `--rope-parameters` flag: rejected by the maintainer as a second mechanism.
- Alternative, an alias for the same weights: rejected by the maintainer and by upstream's alias rule.

### D2 — Frequencies and factor come from the installed transformers reference

For `rope_type: "yarn"`, `RotaryTables.allocate` builds a canonical `PretrainedConfig` and calls `ROPE_INIT_FUNCTIONS["yarn"]`, as `kimi31/dflash.py` line 58 already does.
The reference honours `partial_rotary_factor`, so it returns the 32 frequencies of Qwen's 64-wide rotary slice.
The PR's own `interpolated_frequencies` helper is dropped.

- Alternative, a local closed form: rejected; the maintainer's review shows a copied formula cannot catch a missing term.

### D3 — The attention factor is folded into the tables once

The cos and sin tables are multiplied by the returned factor when they are built, which is what `modeling_qwen4_exp.py` lines 122-123 do.
Native (`rope_type: "default"`) tables are computed by the unchanged code path and stay byte-identical.
No consumer of the tables changes.

### D4 — The context ceiling lives on the resolved checkpoint

Core's `Checkpoint` gains an optional `context` (absent means the recipe's context) and an optional `rope_parameters` mapping.
`Plan` reads `checkpoint.context or recipe.context` wherever it reads `recipe.context` today, and its report records both values.
The Qwen recipe's `inspect_replacement` hook reads the directory's config and returns the checkpoint with both fields set.
Core still computes `min(ceiling, tokens the pages back)` and never imports a recipe.

- Alternative, raising `Recipe.context` to 1,048,576: rejected; the native checkpoint would then plan for a context it cannot serve.

### D5 — One validator admits two config shapes and caps the ceiling

`_validate_config` accepts the pinned native config, or a YaRN config that meets every line below; anything else is refused with the field named.

- `original_max_position_embeddings` equals 262,144, and `max_position_embeddings` stays 262,144, as the Qwen model card's block leaves it.
- The ceiling is `factor` times the original: a whole number, 64-aligned, above 262,144.
- The ceiling is at most 1,048,576, the largest this change measures.
- Keys are limited to the geometry keys plus `factor`, `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `attention_factor`, `mscale`, `mscale_all_dim`, and `truncate`.
- Any `rope_type` other than `default` or `yarn` is refused.

Explicit `attention_factor`, `mscale`, and `mscale_all_dim` are passed to the reference, not refused.
The bootstrap's `context <= 262144` check becomes a check against the checkpoint's ceiling.

### D6 — A scaled context on SM103 is a usage error

`Qwen38Parameters.validate` refuses a checkpoint whose context exceeds the recipe's native context when any selected GPU is SM103.
The SM103 score-workspace lines stay exactly as they are on recipes `upstream/main`.
This honours the operator's deferral and answers the maintainer's principle-14 finding by removing the hunk.

### D7 — Nothing is added to the preparation identity

`config.json` is already hashed, so a native and a scaled preparation, and two scaled ceilings, never share an identity.
A test pins this.

### D8 — Tests are anchored to the reference

- Scaled tables equal `ROPE_INIT_FUNCTIONS["yarn"]` output times positions, at sampled positions up to the ceiling.
- One case pins literals: ramp bounds 14 and 22 and factor 1.1386 for theta 1e7, factor 4, original 262,144.
- Native tables are bit-identical to recipes `upstream/main`.
- Validator accept and reject cases, the SM103 refusal, and identity separation.

### D9 — Evidence comes from one GPU-0 window on the reworked branches

1. Build one image from the core PR head and the recipes PR head.
2. Export and serve the native package and the YaRN package through the console.
3. Run every probe against the console-served instance with `--base-url`.

The recipes PR cannot lock an unmerged core commit, so the image is built from a local-only commit that locks the core PR head; that commit is never pushed.

### D10 — The long-context task beyond retrieval is RULER variable tracking

RULER (NVIDIA, Hsieh et al. 2024, https://github.com/NVIDIA/RULER) defines multi-hop variable tracking: chained assignments scattered through the haystack, answered only by following the chain.
`qualify/long_context_passkey.py` gains an optional `--task` argument (`passkey` by default, `tracking` new), so the grid, ordering, summary, and `--base-url` handling are reused unchanged.

- Short-context comparison: both tasks at 8,000, 64,000 and 200,000 tokens on the native and the YaRN package, reporting pass rates and whether the greedy replies are identical.

### D11 — Upstream texts say what was measured

The #212 description follows the four-part order of `AGENTS.md`, names the head each figure was measured on, and carries no process statements or generated-with line.
`docs/TODO.txt` names SM103 at a scaled context as unmeasured, with its removal criterion.

### D12 — The fork gains the attention factor in place

The fork's `RotaryTables.allocate` multiplies its scaled cos and sin tables by `0.1 ln(factor) + 1`, the reference default, and a test compares them with the transformers rotary embedding.
The flag, alias, and `RopeScaling` declaration are unchanged, and the preparation identity changes by itself because it hashes the recipe's sources.
The operator's serving profile is duplicated onto the new fork image and exported last, only if the scaled test package's results hold up.

- Alternative, moving the fork to the config-driven mechanism now: rejected; it would change the operator's profiles and the `rope-scaled-export` capability before upstream accepts the mechanism.

## Risks / Trade-offs

- [The Qwen model card may not document the YaRN config block the maintainer quoted] → the first task reads the card; the PR cites it only if it says so.
- [A fused kernel may assume unit-magnitude tables] → the GPU-1 CUDA test compares scaled attention against the transformers model on the host venv before any GPU-0 window.
- [Static YaRN may degrade short prompts] → the short-context comparison reports it as measured; the result is not tuned.
- [The recipes PR's tests need the core field before #212 merges] → the recipes PR states the dependency and its evidence names the core head it ran on.
- [The old 27-cell passkey grid no longer describes the code] → it is rerun; the old numbers are cited only as the `2ca523b2` record.
- [The fork keeps a flag upstream rejected] → recorded in the proposal as a separate decision.

## Migration Plan

1. Recipes branch and core branch are built and tested on CPU and GPU 1.
2. One GPU-0 window produces the evidence.
3. The core branch is force-pushed with a lease to #212, the recipes PR is opened, and one reply on #212 links them.

Rollback is a force-push of `15d8b0b0` back to `pr/rope-scaling`; a backup tag is cut before the push.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.85
Caller Agreement: 0.7
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement n/a
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.8
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.3

### Review Feedback

Consensus feedback recorded verbatim:
- "Select YaRN only from the checkpoint's own config.json (text_config.rope_parameters, rope_type "yarn") through one recipe-side parser and validator. Remove the model alias and the CLI JSON flag." — ACCEPTED: D1 and D5 already take this shape.
- "Refuse unimplemented rope types and malformed or unbounded parameters." — ACCEPTED: D5 lists the admitted keys and refuses the rest.
- "Move the context ceiling out of the fixed Recipe.context constant into a per-checkpoint value derived by the recipe (via inspect_replacement), and keep core's min(ceiling, KV-backed tokens) without importing recipes." — ACCEPTED: D4.
- "Derive inv_freq and attention factor from the installed transformers ROPE_INIT_FUNCTIONS["yarn"], and fold the attention factor (0.1*ln(4)+1 ≈ 1.1386) into the FP32 cos/sin tables once, with no kernel changes." — ACCEPTED: D2 and D3.
- "Cap factor and capacity, since table memory is linear in capacity." — ACCEPTED: D5 gained the 1,048,576 cap; it replaces the draft's plan-time page check, which accounts for the 0.7 caller agreement.
- "Gate family A off at 1M until measured." — ACCEPTED: D6.
- "Anchor tests to the installed transformers reference, not copied formulas." — ACCEPTED: D8.
- "Blast radius: config-byte identity separates caches and artifacts, and unscaled behaviour must stay unchanged." — ACCEPTED: D7 and the bit-identical native test in D8.
- "Evidence beyond passkey: RULER non-retrieval tasks (multi-hop tracing, aggregation, QA), plus a short-context paired comparison of unscaled versus YaRN packages on the same weights." — ACCEPTED in part: D10 takes multi-hop tracing and the paired comparison; aggregation and QA are REJECTED because the maintainer asked for one task beyond retrieval. Certainty 0.85 is below the 0.9 floor on this point only.
