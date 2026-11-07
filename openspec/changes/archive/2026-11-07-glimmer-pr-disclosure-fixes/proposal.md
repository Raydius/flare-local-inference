<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Restore Gemma 4's NVFP4 parsing and drop the uncalled DFlash2 projector path on PR #229

Tier: 2
Type: bug

## Stated Requirements

- "glimmer-pr-disclosure-fixes"
- "Track progress in your native task list: seed it with the propose phase steps (proposal, design, external review, spec deltas, tasks) now and keep step statuses updated as you work."

Assumed: Gemma 4 gets its upstream behavior back — the shared `modelopt_nvfp4_layers` parser checks `weights` only, as upstream's private Gemma 4 parser did, and the Muse Glimmer recipe checks `input_activations` (4-bit, group 16) itself before calling it.
Assumed: the `"dflash"` projector selector and its convolution-less / selector-less branches are removed from the shared DFlash2 components, with the one test that exercises only them.
Assumed: the fixes land on `pr/muse-glimmer` only, folded into commit 1 so the PR stays two commits; `deploy` keeps its copy until the PR merges and the fork rebases onto it.
Assumed: commit 1's message drops "Gemma 4 calls it" plus the stricter-check wording and states that Gemma 4 accepts and maps the same configurations (the error text now names the ModelOpt config rather than Gemma 4); both commits are re-acknowledged (`design-review=4d3a`), force-pushed with lease, and #229's description is re-synced from `pr-body.md`.

## Root Cause

wrong-abstraction: lifting Gemma 4's private ModelOpt NVFP4 parser into the shared `ignition/preparation/quantization.py` folded one caller's requirement — Muse Glimmer's W4A4 `input_activations` check — into the shared helper (`for key in ("weights", "input_activations")` at quantization.py:33 on the branch), so Gemma 4, whose upstream parser read only `weights` (upstream `ignition/recipes/gemma4/config.py:534-536`), now refuses a weights-only configuration; the `muse-glimmer-serving` spec's own "Gemma 4 unchanged" scenario (openspec/specs/muse-glimmer-serving/spec.md:36-38) states the map must equal the private parser's. The `projector` selector in `ignition/components/dflash2_weights.py:36,49,102` likewise generalizes the shared DFlash2 loader for a drafter (the first-party `encoder.fc` layout from 9834def8) that the z-lab DFlash2 swap left with no caller.

## Blast Radius

1. ignition/preparation/quantization.py:18,33 (branch) — `modelopt_nvfp4_layers` loops over `("weights", "input_activations")`; becomes `weights` only, the docstring follows.
2. ignition/recipes/muse_glimmer_nvfp4/config.py:62 — the one caller that needs W4A4; gains its own `config_groups.group_0.input_activations` 4-bit group-16 check before calling the parser.
3. ignition/recipes/gemma4/config.py:524 — calls the shared parser; unchanged code, behavior returns to upstream's for weights-only configurations.
4. tests/recipes/test_gemma4.py:186,238 (branch) — the two `input_activations` fixture lines the PR added are removed, so the fixtures equal upstream's again and exercise the weights-only path.
5. tests/preparation/test_quantization.py:105-127 (branch) — `test_weight_only_group_is_rejected` and the `input_activations` row of `test_malformed_blocks_are_rejected` assert the shared parser refuses weights-only; they become an acceptance case (weights-only parses) and move the refusal to the Glimmer config tests.
6. tests/recipes/muse_glimmer_nvfp4/test_config.py:32-34 — the Glimmer fixture carries `input_activations`; gains a case that a weights-only Glimmer config raises `ValueError`.
7. ignition/components/dflash2_weights.py:36,49-55,102-113 (branch) — `_PROJECTION_NAMES`, the `projector=` keyword on `DraftWeights.load` and `validate_metadata`, and the `convolutions` / `selector=None` branches; restored to upstream's form.
8. ignition/components/dflash2.py:283,305 and the `convolution is None` / `selector is None` branches (branch; `git diff upstream/main -- ignition/components/dflash2.py` hunks 22-28, 40-61, 74-110) — removed; the `block_size` field without default that the block-16 drafter needs stays.
9. tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py:146-152 — `test_projector_switch_keeps_the_dflash2_tensor_set_apart`, the only caller of `projector="dflash"`; removed.
10. grepped `projector=` and `"dflash"` across `ignition/` and `tests/` on the branch — only items 7-9 (other `projector` hits are unrelated vision-projector names in minimax_m3, kimi3, nemotron_h, step5 and the `projector_type` draft-config keys).
11. grepped `modelopt_nvfp4_layers` across `ignition/` and `tests/` on the branch — items 1-3 and 5 only.
12. grepped history with `git log -S'"encoder.fc"'` and `-S'input_activations'` over 585159c9..deploy (ignition/components/dflash2_weights.py, ignition/preparation/quantization.py) — both introduced by 9834def8 (the first Glimmer commit), so neither behavior predates Glimmer.
13. openspec/specs/muse-glimmer-serving/spec.md:28-38 — the requirement says the parser reads "weights and input activations"; the delta moves the input-activation check to the recipe and keeps the Gemma 4 scenario.
14. commit 230e3718 message on `pr/muse-glimmer` and GitHub NVIDIA/spark-local-inference#229 description — both describe the parser; commit 1 is rewritten, the description re-synced from openspec/changes/archive/2026-11-07-upstream-muse-glimmer-pr/pr-body.md:9.

## Why

PR #229 changes Gemma 4's accepted configurations without that being Gemma 4's change to make, and the shared DFlash2 loader carries an option nothing calls; both cost upstream review time and the first contradicts the recipe's own spec.

## What Changes

- The shared ModelOpt NVFP4 parser checks `weights` only; the Muse Glimmer recipe checks its W4A4 `input_activations` itself. Gemma 4 parses exactly what upstream's private parser parsed.
- The shared DFlash2 weight loader, validator and prepared draft lose the uncalled `"dflash"` projector path.
- Commit 1 is rewritten with the fixes and a corrected message; `pr/muse-glimmer` is force-pushed with lease; #229's description is updated.

## Capabilities

### New Capabilities

- none

### Modified Capabilities

- `muse-glimmer-serving` — the quantization requirement: input activations are the recipe's check, and Gemma 4 keeps its weights-only parsing.
