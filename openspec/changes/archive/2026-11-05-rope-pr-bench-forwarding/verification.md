# Verification — rope-pr-bench-forwarding

## Verdict: PASS

## Evidence audited
- [In `/tmp/rope-pr` (branch `pr/rope-scaling`, open as NVIDIA/spark-loca] — ignition/cli/bench.py:134 — "    model = resolve_model(configuration, args.model or configuration.model or "qwen3.8-27b", getattr(args, "rope_parameters", None))"

Read on the PR tip `2ca523b2` in /tmp/rope-pr (clean tree). Tier check clean; `git diff d7388f75 2ca523b2 --stat` = ignition/cli/bench.py (2 +-) and tests/cli/test_models.py (+9), within Tier 1.
Reachable: ignition/cli/__init__.py:49 registers `command("bench", bench.run, ...)`; the new test drives it through `ignition.cli.main` (tests/cli/test_models.py:322).
Tested: `PYTHONPATH=/tmp/rope-pr .venv/bin/pytest tests/cli/test_models.py -q -k rope_parameters` (dflash-s-e2vk venv) — 15 passed; the fail-before run is recorded on the task.
Reused/Optimal: same `getattr(args, "rope_parameters", None)` forwarding as ignition/cli/serve.py:131/177/266; no new helper.
Faithful: the stated requirement (bench parses and silently drops the flag) is delivered — bench now forwards it and the dense recipe's refusal reaches the user; no other behavior changes. PR #212 head = 2ca523b2.
