# Correct FORK.md's build.sh default after vllm-env 931be1f

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Update these gaps"

## Why

FORK.md tells the reader to always pass `deploy` because build.sh defaults to `patch/07-rope-parameters`. vllm-env `931be1f` changed that default to `deploy`, so the sentence now describes behavior that no longer exists.

## What Changes

The sentence says that build.sh defaults to `deploy`.

## Root Cause

wrong-mental-model: the sentence was written in `2ef90089` for the old default (FORK.md:126).

## Blast Radius

1. FORK.md:126 — the default-ref sentence.
2. grepped `patch/07-rope-parameters` in README.md and docs/: no other statement of the build default.
