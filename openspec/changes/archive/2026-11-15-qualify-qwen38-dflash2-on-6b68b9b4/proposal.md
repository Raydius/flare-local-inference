# Measure arm Q's re-export on upstream 6b68b9b4 against arm Q

Tier: 2
Type: feature

## Stated Requirements

- "Can we separate the Qwen exploration into a follow-up instead of grouping it here?"
- For the arm Q copy export already running when that was asked: "Let it finish" (over "Stop it now")
- "Propose it now" (for measuring the exported copy `gb202-tp1-202611160316` against arm Q's 555.9 / 436.7 / 1231.2 tok/s and passkey 3/3)
- "When we started this whole change I was running A16, and was running it for DAYS.  That is clearly the profile I'm most interested in."
- "Why would you rebase a profile that I wasn't even running?"
- "I **specifically asked in the beginning why you were using "arm Q"** and did not get an answer"
- "The A16 profile has the most up to date settings that I care about"
- "Here is the updated plan: just finish the change as written.  I will export A16 and benchmark it myself."

Assumed: the operator's A16 profile (*2026-11-10 Flare DFlash2 MXFP8 A16*) is exported and measured on the rebased image by the operator, outside this change.
Assumed: the measurement is the one `rebase-deploy-onto-upstream-29192ef7` planned and moved here (its design.md Revisions, 2026-11-15 18:20): `code_edit` at concurrency 1 and 8, three trials each; 16 clients, three trials; the 1M passkey at depths 0.1, 0.5 and 0.9, run exactly as `qualify/results/dflash2-paths-c16/README.md` ran arm Q.
Assumed: the subject is the existing entry `gb202-tp1-202611160316` of console profile *2026-11-14 Flare DFlash2 Q Rebase4* (image `local/flare-ignition:3b4a1a2e`), exported 2026-11-15; it is not re-exported. Its export log is the one `~/models/.cache/rebase-6b68b9b4/console/Q4/export.log` keeps.
Assumed: the reference is arm Q as recorded (`qualify/results/dflash2-paths-c16/summary.json` key `Q`), not a fresh re-measurement of arm Q.
Assumed: a median outside ±8 % of arm Q's, or a passkey below 3/3, is reported with its range and an entry in `docs/TODO.md`; this change investigates nothing further and changes no code.
Assumed: the result is recorded as a new section and a `Q4` key in `qualify/results/dflash2-paths-c16/` (README, `summary.json`, trial files without per-request observations), in that folder's existing format.
Assumed: GPU 0 is freed by the owner through the console before the trials; afterwards GPU 0 goes to the operator's choice.

## Why

`deploy` moved onto upstream `6b68b9b4` and is published, but the Qwen3.8 DFlash2 configuration the owner serves (arm Q) has not been served from the rebased code. Carry commit `3b4a1a2e` lets `qwen38` preparation run on the locked `transformers==5.23.0` instead of revision `1d9654e7`; its only GPU evidence is that the export completed. Upstream's KV-capacity changes also raised the cache from 1,118,208 to 1,449,984 tokens. Until the rebased core is measured, it is unknown whether it serves at arm Q's speed and acceptance and still passes the 1M passkey, so arm Q cannot be moved onto it.

## What Changes

- Serve entry `gb202-tp1-202611160316` (arm Q4) through the console and run the nine trials and the 1M passkey with arm Q's commands.
- Add a section for the rebased arm (`Q4`) to `qualify/results/dflash2-paths-c16/README.md`, with arm Q beside it, and its figures to `summary.json`.
- On a gap beyond ±8 % or a failed passkey cell, add one `docs/TODO.md` entry naming it.

## Capabilities

### New Capabilities

None — measurement and its record only; `skip_specs: true`.

### Modified Capabilities

None.

## Impact

- No source change. Repository files: the results folder and, conditionally, `docs/TODO.md`.
- GPU 0 is held for about twenty minutes of trials and the passkey; the operator's serving is stopped meanwhile.

## Blast Radius

1. qualify/results/dflash2-paths-c16/README.md:93 — new section beside the arm QA section, same table form
2. qualify/results/dflash2-paths-c16/summary.json:1 — new key `Q4` in the existing per-arm schema (keys baseline, U, Q, A, M, QA)
3. docs/TODO.md:35 — the Qwen entries; one entry added only on a gap or failed cell
4. ignition/preparation/qwen38.py:247 — `supported_transformers`, exercised (not changed) by the export under measurement; called from ignition/recipes/qwen38/model.py:27 and ignition/recipes/qwen38/mtp_weights.py:86
5. qualify/dflash2_console_rate.py:23 — `rate()`, used read-only to compute the `code_edit` medians, as for arm Q
6. grepped 'paths-c16\|dflash2_console_probe' across qualify/ docs/ — only qualify/dflash2_console_probe.py (the probe itself) and the results README; no other document records arm Q's figures
