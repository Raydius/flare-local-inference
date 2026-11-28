# Rework PR #212 into config-driven YaRN scaling with the reference attention factor

Tier: 2
Type: feature

## Stated Requirements

- "Help me review the current status of PR #212"
- "Rework to the maintainer's review"
- "SM103 has to be deferred out of this scope"
- "go with the recommended route" — the option offered as "Test packages plus your corrected serving package (Recommended): one stop, about 2.5 hours": port the attention-factor fix to the fork and re-export the operator's serving profile in the same GPU-0 window.

Not included: the SM103 score-workspace change and any SM103 measurement — "SM103 has to be deferred out of this scope"

## Why

Upstream PR #212 (`NVIDIA/hyperflare-inference`) cannot merge: maintainer `yatesdr` requested changes on 2026-11-19, and the branch predates the 2026-11-22 recipe split.
The review asks for the YaRN reference definition including its attention factor, one general config-driven mechanism instead of an alias plus a flag, and evidence beyond retrieval.

## What Changes

- The Qwen3.8 recipe reads YaRN `rope_parameters` from the checkpoint's own `text_config` and computes frequencies and the attention factor with the installed transformers reference.
- The serving-context ceiling comes from the checkpoint config, carried on the resolved checkpoint, instead of from an alias.
- **BREAKING** (PR branches only): the `qwen3.8-flash-yarn4` alias and the `--rope-parameters` flag leave the upstream contribution.
- A scaled context on an SM103 GPU is refused, so the unmeasured SM103 path is never taken.
- The work splits into a new recipes pull request (`local-inference-lab/hyperflare-recipes`) and a rewritten #212 holding only the core change.
- New evidence on the reworked branches: export and serving, the passkey grid, a long-context task beyond retrieval, and a short-context comparison against the unscaled package.
- The #212 description is rewritten in the four-part order with no process statements.
- The fork's `deploy` branches keep their current alias and flag, and the fork's Qwen3.8 Flash Next rotary tables gain the YaRN attention factor.
- The operator's serving profile is re-exported on the corrected fork image, after the test packages are measured.

## Capabilities

### New Capabilities

- `config-driven-rope-scaling`: what a checkpoint config's YaRN block does to rotary tables, the context ceiling, the preparation identity, and which configurations are refused.

### Modified Capabilities

- `long-context-passkey-retrieval`: gains a variable-tracking task and a scaled-against-native comparison at lengths below the native context.
- `rope-scaled-export`: a rope-scaled Qwen3.8 Flash Next export applies the YaRN attention factor to its rotary tables.

## Impact

- Core PR branch `pr/rope-scaling` (this repository): rebuilt on `upstream/main`, core files only.
- Recipes PR branch (new, in `~/dev/hyperflare-recipes`): `hyperflare_recipes/recipes/qwen38/` and its tests.
- Recipes fork `deploy`: `qualify/long_context_passkey.py` and new result files.
- GPU 0: one serving-instance stop for two exports and the probe runs.
- Outward-facing: a force-push to #212, a new recipes pull request, and a reply on #212.

## Blast Radius

1. grepped `git merge-tree --write-tree --name-only upstream/main origin/pr/rope-scaling` — 16 conflicting files; 8 of the PR's 18 files sit in paths `upstream/main` (`74618652`) deleted.
2. grepped `git show upstream/main:ignition/declaration.py` — `Recipe.context` at line 182 is the only context ceiling; `Checkpoint` has no context field.
3. grepped `git show upstream/main:ignition/planning.py` — `Plan.context` at line 350 and the page bound at 355 read `recipe.context`; the report at 382-383 records it.
4. grepped `git show upstream/main:ignition/registry.py` — `_resolve_replacement` at line 189 calls `recipe.inspect_replacement` at 198, the existing hook that returns an adjusted `Checkpoint`.
5. grepped recipes `upstream/main` (`8df08c21`) `hyperflare_recipes/recipes/qwen38/attention.py` — `RotaryTables.allocate` at line 43 reads `config.rope_parameters` at 46 and builds the cos and sin tables at 51-60.
6. grepped the same file — the SM103 score workspace at lines 261-265 scales with `context`; this change leaves it untouched.
7. grepped recipes `upstream/main` `qwen38/bootstrap.py` — line 189 rejects a context above 262,144; line 274 hashes `config.json` into the preparation identity; line 488 passes `config.context` to the rotary tables.
8. grepped recipes `upstream/main` `qwen38/checkpoint_schema.py` — `max_position_embeddings: 262144` at line 49 is enforced by `_validate_config` at line 113 of the recipe's checkpoint module.
9. grepped recipes `upstream/main` `qwen38/recipe.py` — `validate` at line 88 already refuses by GPU capability; `context=262144` at line 176.
10. grepped recipes `upstream/main` `kimi31/dflash.py` — line 58 calls `ROPE_INIT_FUNCTIONS["yarn"]`, the precedent for using the transformers reference.
11. grepped `.venv/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py` — lines 122-123 multiply cos and sin by `attention_scaling`.
12. ignition/cli/preparation.py:53 — the fork's `rope_scaled` flag path, which refuses `attention_factor` at line 80; carried unchanged on `deploy`.
13. ignition/declaration.py:134 — the fork's `Checkpoint.context` and `rope_scaling` fields; carried unchanged on `deploy`.
14. grepped recipes fork `deploy` `qualify/long_context_passkey.py` — the probe this change widens; `qualify/results/yarn4-passkey/` holds the grid the attention factor invalidates.
15. openspec/specs/long-context-passkey-retrieval/spec.md:1 — the capability that gains the tracking task.
16. openspec/specs/rope-scaled-export/spec.md:9 — names the flag and alias on the fork; unchanged because `deploy` is unchanged.
17. openspec/specs/upstream-contribution-branch/spec.md:9 — the rules both PR branches follow.
18. grepped recipes fork `deploy` `hyperflare_recipes/recipes/qwen38/attention.py` — `RotaryTables.allocate` builds scaled tables from `interpolated_frequencies` with no attention factor; line 520 of the dense recipe's decoder does the same.
19. grepped `GET /api/profiles` — the operator's profile `2026-11-26 Flare v0.3.2-r2` exports with `--rope-parameters` factor 4.0 on the `0.3.2-r2` image.
20. grepped `gh pr list --repo local-inference-lab/hyperflare-recipes --author Raydius --state all` — no recipes pull request exists yet.

Lessons consulted: `openspec/lessons.md` entries `check-upstream-for-the-fix-before-writing-it`, `judge-a-patch-against-what-upstream-added-not-what-it-lacks`, and `serve-from-the-operators-running-profile`.
