# Keep Muse Glimmer's malformed-quantization errors a ValueError

Tier: 1 — two files, under 20 lines, no behavior fork (malformed input keeps its pre-move ValueError)
External Review: Waived — Tier 1
Type: bug

## Stated Requirements

- "act on each finding; never file it" — rad-verify finding 1 of glimmer-pr-disclosure-fixes: the new Glimmer input-activation check raises `AttributeError` instead of `ValueError` for a non-mapping `quantization_config` or `config_groups`.

## Root Cause

missing-edge-case: `ignition/recipes/muse_glimmer_nvfp4/config.py:63` (on `pr/muse-glimmer`, /tmp/glimmer-pr) chains `config.get("quantization_config", {}).get("config_groups", {}).get("group_0", {})`; the `{}` defaults apply only to absent keys, so a present `None` or list value raises `AttributeError` before `modelopt_nvfp4_layers` (quantization.py:25-31), whose `isinstance(..., Mapping)` guards raised `ValueError` for the same inputs before glimmer-pr-disclosure-fixes moved the check.

## Blast Radius

1. ignition/recipes/muse_glimmer_nvfp4/config.py:63-66 (branch) — guard each level with `isinstance(..., Mapping)` as quantization.py does.
2. tests/recipes/muse_glimmer_nvfp4/test_config.py:119-128 (branch) — add `quantization_config` None / list and `config_groups` None cases expecting `ValueError`.

## Why

Recipe config loaders signal a refused checkpoint with `ValueError`; an `AttributeError` reads as a code bug and escapes callers that catch `ValueError`.

## What Changes

- The Glimmer input-activation check type-guards each level and raises its `ValueError` for any malformed shape.
- The fix folds into commit 1 of `pr/muse-glimmer` (re-acknowledged, force-pushed with lease) so NVIDIA/spark-local-inference#229 stays two commits.
