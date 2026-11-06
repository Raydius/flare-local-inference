# Honor --draft-model in the Muse Glimmer recipe so the console can prepare it offline

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Go ahead and build a profile in the console for this" (operator, 2026-11-06)
- "muse_glimmer recipe honors --draft-model (plan.draft_model) and declares draft_files, so the console's offline Ignition launch with a mounted /flare-draft prepares the drafter instead of trying to download it" (hotfix description, 2026-11-06)

## Why

The console launches every Ignition export with `HF_HUB_OFFLINE=1` and supplies an external draft by mounting its folder at `/flare-draft` and passing `--draft-model /flare-draft` (`~/LocalApps/vllm-env/docs/operations/flare_runtime.md` "Preparing with another recipe and an external draft"). The Muse Glimmer recipe ignores `plan.draft_model` and always calls `snapshot_download` for its pinned assistant, and it declares no `draft_files`, so `ignition export --draft-model` is refused with "loads no external draft" and a launch without it dies with "cannot download". No console profile can prepare the drafting package until the recipe takes the flag.

## What Changes

`ignition/recipes/muse_glimmer/recipe.py` declares `draft_files=("config.json", "model.safetensors")` and uses `plan.draft_model` as the draft snapshot when given, downloading the pinned assistant only when it is not, as `qwen38_dflash` does.

## Root Cause

missing-edge-case: the recipe's `prepare` was written against the alias export on this host, where the hub cache resolves the pinned assistant, and never read `plan.draft_model` (ignition/recipes/muse_glimmer/recipe.py:19-24); the `Recipe` declaration omits `draft_files`, which `_check_draft_model` requires before it accepts the flag (ignition/cli/preparation.py:127-129).

## Blast Radius

1. ignition/recipes/muse_glimmer/recipe.py:19-27 — `prepare` builds `drafting` from `snapshot_download` only; gains the `plan.draft_model` branch.
2. ignition/recipes/muse_glimmer/recipe.py:51-72 — the `Recipe(...)` declaration; gains `draft_files=("config.json", "model.safetensors")`.
3. ignition/cli/preparation.py:125-134 — `_check_draft_model` reads `model.recipe.draft_files` and checks each file exists in the folder; unchanged, it is what the declaration satisfies.
4. ignition/recipes/qwen38_dflash/recipe.py:54-59,107 — the precedent: `draft = plan.draft_model` with the download as fallback and `draft_files` declared; unchanged.
5. tests/recipes/muse_glimmer/test_config.py and tests/cli/test_draft_checkpoints.py — the declaration tests that pin the recipe; `grep -rn "draft_files" tests/` shows which rows assert it (tests/cli/test_draft_checkpoints.py), so the Glimmer row gains the expectation.
6. grepped `draft_model` across ignition/recipes/ — only qwen38_dflash and the recipe-neutral CLI read it; no other recipe is affected.
