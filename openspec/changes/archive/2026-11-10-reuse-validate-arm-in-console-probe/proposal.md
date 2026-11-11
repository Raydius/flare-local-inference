# Check console-probe packages with validate_arm and let the rate tool read committed records

Tier: 1 — four files, under 120 lines, no behavior fork: the second directory is one added test assertion
Type: bug
Retired: archived incomplete — tasks-open

External Review: Waived — Tier 1 (two qualify scripts and one optional parameter; no serving or recipe code)

## Stated Requirements

- "Import the module that already does the job." — global working rules, Use existing modules
- "widen it with an optional prop that defaults to today's behavior, so every existing call site stays byte-identical" — global working rules, Use existing modules

Assumed: scaffolded by the verification audit of `measure-qwen38-dflash2-paths-c16` (findings 2 and 3 of its verification.md), not requested in a session; no measured figure changes.

## Root Cause

wrong-abstraction: `validate_package` repeats `validate_arm`'s model-id and speculative-graph checks because `validate_arm` cannot skip its 8-slot and mixed-graph checks, and `rate` indexes `observations`, a key the committed trial records omit (qualify/dflash2_console_probe.py:26)

## Blast Radius

1. qualify/dflash2_console_probe.py:26 — `validate_package` copies qualify/speculative_benchmark.py:374-383; line 61 passes it to `run`.
2. qualify/speculative_benchmark.py:372 — `validate_arm` hard-codes `max_active != 8` and the mixed-graph check; an optional keyword defaulting to today's checks lets the probe import it.
3. qualify/dflash2_console_rate.py:29 — `measurements["observations"]` raises KeyError on every `qualify/results/dflash2-paths-c16/*/c*-r*/measurements.json`, which carry `observations_omitted`; line 24's docstring names acceptance counters the function does not return.
4. tests/reference/test_speculative_benchmark.py:195 — the six existing `validate_arm` calls pass two positional arguments and stay unchanged.
5. grepped 'validate_arm\|validate_package\|console_rate' across qualify/ tests/ — no further sites.
