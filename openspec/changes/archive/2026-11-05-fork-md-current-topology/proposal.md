# Correct FORK.md's deploy, hook, upstream and upstreaming statements

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "Fork md"

## Why

FORK.md is the fork's maintenance doctrine, and four of its statements are now false. An agent that follows it would describe `deploy` wrongly. It would also expect commits to go through without the design acknowledgment the active hook demands.

## What Changes

Four factual sentences in FORK.md are rewritten. The doc rules themselves are unchanged:
- `deploy` is described as the patch stack plus the merged DFlash2 work.
- The hook is described as enabled.
- The `upstream` remote gets a note on its GitHub redirect.
- The scope paragraph records that proven patches now go upstream as PRs.

## Root Cause

wrong-mental-model: FORK.md was written when `deploy` was the five patches plus one README commit, the hook was off, and nothing went upstream (FORK.md:9-13, FORK.md:30, FORK.md:33-36). Since then, `wt/dflash-s-e2vk` was merged onto `deploy`. The local `.git/config` sets `core.hooksPath=.githooks`, and 35 of 48 fork commits since `585159c9` carry `Design-Signed-Off-By`, the first being `ccb4dd35` (2026-10-21). PRs #211 and #212 are open upstream.

## Blast Radius

1. FORK.md:9-14 — Scope paragraph: "Upstreaming to NVIDIA is out of scope" and "The hook stays disabled."
2. FORK.md:21 — `upstream` remote row: the URL now redirects to `NVIDIA/spark-local-inference`.
3. FORK.md:30 — branch-model diagram: the `deploy` comment.
4. FORK.md:33-36 — the `deploy` bullet: "one commit above the topmost patch".
5. grepped 'hook stays disabled|one commit above' across README.md and docs/ — no further sites. README.md's patch notes were corrected in `4ddd6e52`.
