# Show every patch's status and PR link on github.com by rewriting the README patch tables as lists

Tier: 2
Type: bug
Mode: hotfix
External Review: None — hotfix

## Stated Requirements

- "I see, the `Still needed` column for some reason does not show on github.com and horizontal scrolling is not intuitive."

## Why

On https://github.com/Raydius/flare-local-inference (default branch `deploy`) the Active patches table renders 1,101 px wide in an 838 px README box at a 1280 px window; the "Still needed" column starts at x=915, past the visible edge, so every PR link (#212, #211, #229, issue #210, FlashInfer #67) and every retirement reason is hidden behind a horizontal scroll readers do not discover.

## What Changes

Both patch tables become per-patch bullet lists (the changelog convention of Keep a Changelog, https://keepachangelog.com/en/1.1.0/): the patch name and date as the lead, then "What it does" and "Still needed" (or "Why retired") as labelled lines that wrap to any width. No patch text changes.

## Root Cause

wrong-abstraction: a five-column Markdown table carries paragraph-length cells, and GitHub's README CSS renders an overflowing table as a horizontal scroll box instead of wrapping it (README.md:14, README.md:33).

## Blast Radius

1. README.md:14-29 — Active patches table (14 rows) becomes a list.
2. README.md:33-42 — Retired patches table (9 rows) becomes a list.
