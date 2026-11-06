# Forward --rope-parameters through ignition bench on deploy

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Update these gaps"
- Operator's choice: "Fix commit on deploy (Recommended)": one new commit on top of `deploy`, no history rewrite, no force-push.

## Why

On `deploy`, `ignition bench --rope-parameters <JSON>` parses the flag and then drops it, so a rope-scaling request goes silently unapplied instead of being honoured or refused. The fix already exists as `2ca523b2` on `pr/rope-scaling` (PR #212) but never reached the fork.

## What Changes

`ignition bench` passes `args.rope_parameters` to `resolve_model` like every other preparation command. The dense recipe then refuses the flag because it declares no rope scaling. The commit goes on top of `deploy` and `patch/07` is left untouched, by the operator's choice; the commit retires with patch/07 once #212 merges.

## Root Cause

silent-failure: `ignition/cli/bench.py:134` calls `resolve_model(configuration, args.model or configuration.model or "qwen3.8-27b")` without the `rope_parameters` argument that `ignition/cli/serve.py:133 (and :180, :269, :429)` pass, so the parsed flag never reaches validation.

## Blast Radius

1. ignition/cli/bench.py:134 — the one call site that drops the flag.
2. tests/cli/test_models.py:319 — gains the regression case from `2ca523b2`.
3. grepped `resolve_model(` across ignition/cli/: ignition/cli/bench.py:134 is the only call that omits `rope_parameters`; the four calls in ignition/cli/serve.py pass it.
