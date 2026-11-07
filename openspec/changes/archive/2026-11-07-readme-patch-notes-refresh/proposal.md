# Bring deploy's README patch notes up to date with the Muse Glimmer work and upstream af0907d9

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Is the README.md in `deploy` up to date with all patch notes?" — answered no; this hotfix acts on the gaps that answer named.

## Why

README.md was last updated at `795d753f` (2026-11-06). The 19 commits since then landed the Muse Glimmer 30B NVFP4 recipe (z-lab DFlash2 drafter, 1,048,576-token context, `--draft-model` honored), the passkey probe's `--base-url` and width-16 readout, and the history-drafts A/B result. None of these appear in the patch notes, and the header still names upstream `972e28e4` while upstream is at `af0907d9` (2026-11-07).

## What Changes

README.md gains a Muse Glimmer Active-patches row (still needed until NVIDIA/spark-local-inference#229 merges), the passkey probe in the qualification-tools row, and the history-drafts A/B result in its row; the header and every row's "Still needed" verdict are re-judged against `af0907d9`, whose only effect on the table is the FlashInfer pin moving from `1f8ed8a3` to `2e4f4c2c` (FlashInfer PR #67 is still open and absent from the new pin).

## Root Cause

order-of-operations: the Glimmer changes closed and landed on `deploy` without a README step, so the patch-notes table was never touched after `795d753f` (README.md:12).

## Blast Radius

1. README.md:6 — header names the upstream tip `972e28e4`; becomes `af0907d9`.
2. README.md:10 — "Still needed is judged against `972e28e4`"; becomes `af0907d9`.
3. README.md:12-27 — Active patches table: new Muse Glimmer row; history-drafts row (README.md:20), FlashInfer row (README.md:25) and qualification-tools row (README.md:26) updated. Checked at `af0907d9`: no `generation_defaults` override (helios/src/server/config.rs:47 still refuses it), no `--max-context-tokens`/`--chat-template` in flare, no `RopeScaling`/`--rope-parameters`, no `--recipe`/`--draft-model` export flags, `AcceptanceProfile::observe` still has no copy mask (helios/src/scheduler/speculation.rs), PRs #211/#212 and issue #210 still open; upstream's compilation.py change refactors fresh-value tracking, not the opaque-write rule.
