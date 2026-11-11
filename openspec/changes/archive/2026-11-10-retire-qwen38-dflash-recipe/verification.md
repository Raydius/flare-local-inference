# Verification — retire-qwen38-dflash-recipe

## Verdict: PASS

Date: 2026-11-10
Tier: 3

## Dimensions
- Implemented: pass — all four ticked boxes confirmed against the tree and the host. Commits `6594e779` (24 files, 17 insertions, 3,011 deletions) and `69e7d70f` (4 files) exist; `ignition/recipes/qwen38_dflash/`, `tests/recipes/qwen38_dflash/`, `qualify/dflash2_tap_check.py`, `qualify/mixed_prefill_check.py` and `tests/reference/test_dflash2_tap_check.py` are absent; `ignition/recipes/__init__.py:39` registers no `qwen38_dflash`; `git grep -n -w qwen38_dflash -- ignition tests qualify ':!qualify/results'` and the alias grep print nothing; `git grep` over README.md and docs returns README.md:57 for the alias and README.md lines 6, 33, 34, 43, 51, 56 for the name, each a retirement statement, as the task 3 evidence says. No file under `ignition/components/`, `ignition/recipes/qwen38/`, `ignition/reference/`, `helios/`, `flare-cli/`, `qualify/results/` or `openspec/changes/archive/` differs between `13463865` and HEAD. Host state: packages `…101936`, `…110050`, `…110138` present and `…110224`, `…110410` gone; 134 GiB free on `~/models`; images `f83d3120` and `7ebef730` present; serving log line 22 reads 1118208 KV tokens; port 8100 `/health` 200. The lint's six stale-citation warnings name Blast Radius paths this change deleted or shortened (warn level, expected for a removal).
- Reachable: pass — the change adds no code path. `ignition.recipes.recipes()` imports and returns 14 recipes with no `qwen38_dflash`; `aliases()` lists no `dflash2` alias; no surviving module under `ignition/`, `qualify/` or `tests/` imports the deleted package or the two deleted tools.
- Tested: no covering command on any of the four boxes (none carries a `Verify:` clause). Read, not rerun: the recorded runs `/tmp/c16-ret/pre.txt` and `post.txt` both end `77 failed … 1 error`, the failing sets differ by one ID (`tests/cli/test_commands.py::test_failures_use_the_documented_exit_codes[arguments0-…]`, whose parameter id carries the edited message), and every collected ID that disappeared is under a deleted file, one of the tests named in task 2, or a case parametrized over the retired recipe or its aliases. The rebuilt layout case at `tests/recipes/test_qwen38_mixed.py:128` passes the same arguments the deleted `state_layout(True)` passed (`speculative_tokens=7, world_size=1, mtp=False, draft_windows=(5, 2048, 8, 128)`). The one coverage loss (CUDA parity of the shared draft component on the pinned Flash-Next draft) is disclosed in README.md:58.
- Optimal: nothing to report — documentation statements checked against code hold (`ignition/planning.py:555` refuses `--mixed-prefill` for `dflash2`; `ignition/recipes/qwen38/recipe.py:119` declares five draft tokens, at most seven; `ignition/recipes/qwen38/bootstrap.py:247` and `helios/src/cache/hash.rs:37` are the lines docs/TODO.md cites).
- Reused: pass — no component, helper or tool added; the one new test line calls the existing `HybridStateLayout.build` (`from ignition.recipes.qwen38.state import HybridStateLayout`, tests/recipes/test_qwen38_mixed.py:125). No artifact or comment claims a reuse the imports do not show.
- Faithful: pass — "it does not. let's retire it." is delivered by tasks 2 and 3 (`6594e779`, `69e7d70f`); "Arm Q (Recommended)" by task 1 (arm Q serving since 20:39:42 per the console log, Rebase3 profile, package and image kept). Every behavior change (the two aliases and `--recipe qwen38_dflash` gone, history drafts and the QAD pin lost for Qwen 3.8 DFlash2, arm A and arm M packages removed) traces to the retirement bullet and its recorded Assumed lines.
- Environment: doctor: all workflow controls present. suite runs: 0 in 240 min, 0 deliberate re-runs, budget 3. The console's `/api/server/status` answers "Missing bearer token" to this session, so the active profile was read from the serving log and `/health` instead; `pyflakes` is not installed in the test venv.

## Evidence audited
- [1. Serve arm Q in place of Rebase3. Before stopping anything: `GET /ap] — qualify/results/dflash2-paths-c16/README.md:53 — "| KV tokens | 1,150,976 | 1,138,688 | 1,118,208 | 1,126,400 | 909,312 |"
- [2. Remove the recipe and its tests in one commit. Record the failing n] — ignition/recipes/__init__.py:39 — "return (qwen38, qwen38_dense, glm53, deepseek41, deepseek4, mimo_v2, kimi3, kimi31, step5,"
- [3. Documentation in one commit. `README.md`: line 6 no longer says the] — README.md:57 — "aliases `qwen3.8-flash-dflash2` and `qwen3.8-flash-dflash2-yarn4`"
- [4. Housekeeping. Remove the arm A (`gb202-tp1-202611110224`) and arm M] — openspec/changes/retire-qwen38-dflash-recipe/tasks.md:18 — "`df` on `~/models` — 64 GiB free before, 132 GiB after arm A, 134 GiB after arm M"

Boxes 1 and 4 changed host state, not repository files, so their repository lines above are the record and the audit read the host directly: the console serving log `vllm-llm-20261110T203942.log` line 22 ("15.9 GiB · 1118208 KV tokens", arm Q's figure in the table quoted for box 1), `ls` of the package folder, `df` on `~/models` (134 GiB free), the `local/flare-ignition` image tags, port 8100 `/health`, and the vault note `Qwen3.8 Flash Next/DFlash2 Recipe Progress.md` line 30 ("deleted through the console's Delete package on their profiles (2026-11-10 23:04); 134 GiB free on `~/models`") with the Retirement paragraph at `Flare Fork Patch Notes.md` line 22.

Tested answers: box 1 no covering command; box 2 no covering command; box 3 no covering command; box 4 no covering command.

## Findings
none

## Advisory rules judged
- Stage decay: none — no follow-ups heading, no task added after the last tick, and each `discovery:` line describes something met while doing its task.
- Declined discoveries: n/a — every `discovery:` line ends `fixed`; none is declined.

## Live UI pass
n/a — no UI surface
