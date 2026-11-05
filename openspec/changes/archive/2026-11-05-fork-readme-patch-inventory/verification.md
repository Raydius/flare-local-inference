# Verification — fork-readme-patch-inventory

## Verdict: PASS

## Evidence audited
- [Rewrite README.md:1-31 (the "Raydius fork: patch notes" section only; ] — README.md:14 — "| Patch | Added | Layer | What it does | Still needed |"

Tier check: `radflow-lint --tier-check` clean; `git diff --stat deploy` = README.md only (22+/11-), hunks at lines 3-42, above the wordmark `---` (line 44).
Faithful: the stated requirement ("determine what fork patches are stil necessary and update the README") maps to the Still-needed column (13 Active rows) and the warmup-retention Retired row; no change outside README.md. Spot-checked claims against git: upstream `ignition/cli/__init__.py:164` `--speculative-method` with `dflash2`; upstream `ignition/recipes/qwen38/speculative_model.py:233` passes `copy_offered_mask` (history drafts on MTP only); deploy `ignition/recipes/qwen38_dflash/config.py:10` draft revision `a22a1744`.
