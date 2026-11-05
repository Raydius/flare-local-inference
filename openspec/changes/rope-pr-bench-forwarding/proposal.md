# Forward --rope-parameters on `ignition bench` so the flag is refused, not dropped

Tier: 1 — one line in one file on the PR branch plus one test case; the suggest-tier count reads the Root Cause's cited sites, which are the same two files
Type: bug

External Review: Waived — Tier 1 (one line on the PR branch `pr/rope-scaling`; no behavior fork)

## Stated Requirements

- Scoped by `/rad-verify` of change `upstream-rope-pr` (verification.md finding 1): "`--rope-parameters` is registered on `ignition bench` but `bench.run` does not forward it to `resolve_model`, so the value is parsed and silently dropped while bench refuses its other unsupported preparation flags explicitly."

## Root Cause

missing-edge-case: patch/07 added `--rope-parameters` to `_preparation_arguments`, which `bench` shares (line 51 and line 153 of ignition/cli/__init__.py in /tmp/rope-pr), and threaded the value through every `resolve_model` call in serve.py (line 131 of ignition/cli/serve.py in /tmp/rope-pr; ignition/cli/serve.py:133 in this fork) but not the one in bench.py (line 134 of ignition/cli/bench.py in /tmp/rope-pr; ignition/cli/bench.py:134 in this fork), so bench accepts the flag and ignores it.

## Blast Radius

1. ignition/cli/bench.py:134 — `resolve_model(configuration, args.model or ...)` drops `args.rope_parameters` (same line on `pr/rope-scaling` in /tmp/rope-pr); forwarding it makes the dense recipe's "declares no rope scaling" refusal (ignition/cli/preparation.py:64 on the PR branch) reach bench.
2. grepped 'rope_parameters' across /tmp/rope-pr/ignition/cli — `__init__.py` lines 144-153 (parser), `preparation.py` lines 40-44 (`resolve_model`), `serve.py` lines 131, 177, 266, 428 (forwarded); bench.py has no hit — no further sites.
3. grepped 'def test_rope_parameters_' in /tmp/rope-pr/tests/cli/test_models.py — 13 cases, all export/debug-serve; a bench case is the test that fails without the fix.
