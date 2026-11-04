# Declare the YaRN x4 checkpoint for the DFlash2 recipe and probe drafting past the native context

Tier: 2 — six files are edited (one recipe declaration, one new probe, three test files, one doc); the other ten cited sites are unchanged consumers read to prove the declaration suffices, no shared component or runtime changes
Type: feature

## Stated Requirements

- "Rope scaling for DFlash2: declare `qwen3.8-flash-dflash2-yarn4` (same weights, `context=1_048_576`, `rope_scaling=RopeScaling(4., 262144)`, as `qwen38/recipe.py:86-90`); keep the draft's `Rotary` unscaled (RoPE is relative; ring window 5 × 2048) unless a > 262k probe says otherwise; add a > 262k long case to the workload. Locally any time after 1; upstream only after 5." (vault DFlash2 Recipe Progress, Next item 6) — conflict on the workload clause: the frozen speculation workload caps every case at 262,144 tokens (`qualify/speculative_workload.py:330`), fixes 28 cases, and is served to the 262k MTP package as the other arm, so the past-262k case is a separate probe, not a 29th frozen case (design.md)
- "The DFlash2 recipe declares no rope-scaled checkpoint, so it serves at most 262,144 tokens; the MTP production profile serves 1,048,576 through YaRN factor 4" (vault DFlash2 Recipe Progress §7 log, 2026-11-02: the console's `--rope-parameters` launch failed `qwen3.8-flash-dflash2 declares no rope scaling`)
- "Rope scaling = later item, NOT the fold." (fold proposal) — this is that later item
- Carried from the DFlash2 changes: "Exports and serves run containerized from a committed ref (build.sh), never host venv"; "Owner launches console runs"; "main stays the upstream mirror"; "Never land on deploy. Never rad-close this worktree — offer a commit instead"; "Commit/push only when asked"; "Update vault tracker Part A when state changes"
- "Planning only this session (propose's boundary)"

## Blast Radius

1. ignition/recipes/qwen38_dflash/recipe.py:18 — `_CHECKPOINT` is a `Checkpoint` instance, not qwen38's keyword dict; :86 `checkpoints=(_CHECKPOINT,)` gains the yarn4 variant (`context=1_048_576`, `rope_scaling=RopeScaling(4., 262144)`)
2. ignition/recipes/qwen38_dflash/recipe.py:67–68 — `prepare` already threads `plan.declared_context` and `plan.checkpoint.rope_scaling` into `PreparationConfig`; unchanged
3. ignition/recipes/qwen38_dflash/bootstrap.py:63 — `PreparationConfig(_QwenPreparationConfig)` inherits qwen38's context-bound and anchor checks (`ignition/recipes/qwen38/bootstrap.py:135`–`:143`); :202 builds the target's `RotaryTables` with `scaling=config.rope_scaling`; :93/:455 hash the scaling into the identity; unchanged
4. ignition/components/dflash2.py:32–38 — the draft's rotary: unscaled `theta` inverse frequencies at absolute `positions` in float32; `PreparedContext` writes ring slot `positions mod window` (:88); positions up to 1,048,575 flow through unchanged (decision kept; probe decides)
5. ignition/cli/preparation.py:63–65 — `rope_scaled` accepts `--rope-parameters` once any recipe checkpoint declares `rope_scaling`; the console's directory export (`/ckpt --recipe qwen38_dflash --draft-model … --rope-parameters`) then resolves; unchanged
6. tests/recipes/test_declarations.py:407 — `test_qwen_yarn_variant_extends_the_declared_serving_context` is the pattern; :462–:499 DFlash declaration tests gain the yarn4 case (recipe identity, 4× demand pages, `PreparationConfig` receives the scaling)
7. tests/cli/test_models.py:25 and :113, tests/cli/test_commands.py:57 — the alias tuple and the "supported models" message gain `qwen3.8-flash-dflash2-yarn4` after `qwen3.8-flash-dflash2`
8. tests/cli/test_models.py:315 — `test_rope_parameters_refuse_recipes_without_rope_scaling` uses `glm-5.3-flash`; unaffected
9. tests/recipes/qwen38_dflash/test_bootstrap.py:72–84 — identity test already separates a scaled identity; unchanged
10. docs/model-preparation.md:434–456 — the DFlash2 section gains the yarn4 alias and the `--rope-parameters` form; `docs/TODO.md:7` DFlash entries unchanged
11. qualify/speculative_workload.py:66 `_catalog`, :330 the 262,144 cap, :31 `CATEGORIES` (28 fixed cases) — the frozen workload stays frozen; the probe reuses `_catalog`
12. qualify/speculative_benchmark.py:266 `stream_request`, :331 `read_metrics`, :393 `metric_delta`, :401 `round_metrics` — reused by the new `qualify/dflash2_long_context.py` probe (past-262k and sub-262k prompts, retrieval check, offered/accepted draft counters)
13. qualify/dflash2_served.py:48 `container_command` — reused to serve the yarn4 package for the probe
14. grepped 'rope_scaling|RopeScaling' across ignition/recipes/qwen38_dflash/ — ignition/recipes/qwen38_dflash/recipe.py:68 and ignition/recipes/qwen38_dflash/bootstrap.py:89, :93, :202, :455 only; grepped '262144|262_144|max_position_embeddings' across ignition/recipes/qwen38_dflash/ — no draft-side bound; grepped 'passkey|needle' across qualify/ tests/server/ — no existing long-context retrieval probe; openspec/lessons.md does not exist in this repo

## Why

The owner's console profile serves the production MTP package at 1,048,576 tokens through YaRN ×4, and its first DFlash2 launch with the same `--rope-parameters` was refused because `qwen38_dflash` declares no rope-scaled checkpoint, so a DFlash2 package stops at 262,144 tokens. The target side already takes a declared scaling exactly as qwen38 does; what is missing is the declaration and a measurement of whether the unscaled draft still earns acceptance past the native context.

## What Changes

- `qwen38_dflash` declares `qwen3.8-flash-dflash2-yarn4`: the same pinned target and draft with `context=1_048_576` and `rope_scaling=RopeScaling(4., 262144)`, so `ignition export qwen3.8-flash-dflash2-yarn4` and `--rope-parameters` on the base alias both prepare a 1M-context package; the draft's rotary stays unscaled.
- A qualification probe, `qualify/dflash2_long_context.py`, sends one retrieval prompt below the native context and one past it to the served yarn4 package, checks the quoted records, and records offered and accepted draft counts for each so the past-262k acceptance is compared against the sub-262k one and the fold's served figures.
- Declaration tests, the alias lists, and the preparation guide's DFlash2 section follow.
- No workload, scheduler, component, or Helios change; the frozen 28-case workload is untouched.

## Capabilities

### New Capabilities

- none

### Modified Capabilities

- `qwen38-dflash2-speculation` — the recipe declares the yarn4 checkpoint; a yarn4 package serves retrieval and drafts past 262,144 tokens under a recorded acceptance criterion

## Impact

- Code: `ignition/recipes/qwen38_dflash/recipe.py`; new `qualify/dflash2_long_context.py`; tests under `tests/recipes/` and `tests/cli/`; `docs/model-preparation.md`.
- Packages: a new export (`--rope-parameters` factor 4 or the yarn4 alias) on GPU 0 through the owner's console from an image built at the committed tip; existing 262k packages and their identities are unchanged.
- Upstream: the yarn4 checkpoint joins the recipe PR only after the rope PR (patch/06 + patch/07) merges (tracker §13).
