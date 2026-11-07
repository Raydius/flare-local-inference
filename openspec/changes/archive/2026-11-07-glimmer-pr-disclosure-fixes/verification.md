# Verification — glimmer-pr-disclosure-fixes

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-07
Tier: 2

## Dimensions
- Implemented: pass — `git diff 31f4e6be a4e0df5e` in /tmp/glimmer-pr holds every claimed edit: quantization.py:32-34 checks `weights` only; muse_glimmer_nvfp4/config.py:62-66 adds the input-activation check; dflash2_weights.py equals upstream/main (empty diff) and dflash2.py differs from upstream only in the `block_size: int` hunk; test_gemma4.py diff against upstream is 0 lines; test_draft_weights.py loses the projector test.
- Reachable: pass — Glimmer check runs on every load: recipes/__init__.py:36 → muse_glimmer_nvfp4/recipe.py:41 → bootstrap.py:144 → weights.py:101 `ModelConfig.load` → config.py:34 `from_dict` → config.py:62-66. Gemma 4 reaches the narrowed parser at gemma4/config.py:524. No caller passes `projector=` (grep over ignition/, tests/, scripts/ and other python dirs: no hits); every DFlash2 draft config declares conv geometry (muse_glimmer_nvfp4/draft_config.py:39, qwen38/draft_config.py:13, qwen38_dense/draft_config.py:14); no consumer unpacks `scratch_tensors()` as a 1-tuple. (Child investigator evidence; call-site lines confirmed against the diff.)
- Tested: pass with one finding — task 1-3 Verify commands re-run here (with `FLARE_TEST_TEMPLATE_DIR=/tmp/flare-test-templates`, per task 2's discovery): `27 passed, 1 skipped` exit 0; `82 passed, 6 skipped` exit 0 and Gemma 4 diff `0`; grep empty and `54 passed, 39 skipped` exit 0. Task 4 Verify exit 0 but prints 22 ids, all ENOMEM ids that `enomem-upstream-main2.txt` shows failing identically on pristine upstream (`cmp` equal); `comm -23` against baseline ∪ enomem with consistent `LC_ALL=C sort -u` → 0 lines; cargo logs 30/41/124 passed. Task 5 Verify: two commits, both trailers, grep count 0, status empty. Task 6 Verify: remote = HEAD = #229 head `a4e0df5e…`, body test `false`. The malformed-shape path the change introduced has no test (finding 1).
- Optimal: advisory — no Optimal finding; the error-type regression is recorded under Faithful.
- Reused: pass — no reuse claim in the artifacts or code comments; the shared parser is imported, not copied: `ignition/recipes/muse_glimmer_nvfp4/config.py:12: from ignition.preparation.quantization import modelopt_nvfp4_layers`, `ignition/recipes/gemma4/config.py:10: from ignition.preparation.quantization import matches_pattern, modelopt_nvfp4_layers, quantized_layers`. No other helper validates 4-bit group-16 `input_activations` (only kimi3/config.py:162 reads the key, requiring it absent), so the inline Glimmer check duplicates nothing; keeping it out of the shared parser is the peer-review-accepted design.
- Faithful: covered with one partial — Stated bullet "glimmer-pr-disclosure-fixes" (with its Assumed lines) is delivered by tasks 1-6 and the diff above; the native-task-list bullet is a session-process instruction (task list seeded and complete). Reverse trace: Gemma 4 weights-only acceptance, the Glimmer-owned activation error, and the projector removal all trace to the Root Cause; the malformed-config `ValueError`→`AttributeError` change does not (finding 1).
- Environment: clean (`radflow-lint --doctor`: all workflow controls present; tier-check clean).

## Evidence audited
Code paths below are repo-relative and were opened in the `pr/muse-glimmer` worktree /tmp/glimmer-pr (HEAD a4e0df5e), where every code edit landed; the `deploy` checkout keeps its pre-PR copy by design.
- [1. Failing-first: add `test_weight_only_group_parses` to `tests/prepar] — tests/preparation/test_quantization.py:108 — "weights_only = modelopt_nvfp4_layers(_glimmer_raw(input_activations=False), _glimmer_candidates())"
- [1. Failing-first (Glimmer cases)] — tests/recipes/muse_glimmer_nvfp4/test_config.py:120 — "def test_rejects_a_checkpoint_without_nvfp4_input_activations(activations):"
- [2. Parser fix: `ignition/preparation/quantization.py` `modelopt_nvfp4_] — ignition/recipes/muse_glimmer_nvfp4/config.py:66 — "raise ValueError(\"Muse Glimmer requires 4-bit group_size=16 NVFP4 input activations\")"
- [3. DFlash2 projector removal: in `ignition/components/dflash2_weights.] — ignition/components/dflash2_weights.py:45 — "def load(cls, snapshot, config, device, shard, *, progress=None, online_mxfp8=False):"
- [4. Suites: `/home/rjdollete/models/.cache/glimmer-pr/run-suites.sh /tm] — openspec/changes/archive/2026-11-07-upstream-muse-glimmer-pr/pr-body.md:14 — "1379 passed, 34 failed, 184 skipped on the branch" (opened also: /home/rjdollete/models/.cache/glimmer-pr/branch-pytest.log line 2016, "34 failed, 1379 passed, 184 skipped, 16 warnings in 173.68s (0:02:53)")
- [5. History: commit the edits as `git commit -s --fixup 230e3718`, then] — docs/design-review.md:3 — "Every commit requires review of its staged diff against the principles below." (history evidence itself is the Verify output: `9348d099` with `Design-Signed-Off-By: Claude Opus 5.5 (claude-opus-5-5); design-review=4d3a`, `a4e0df5e` parent `9348d099`, grep count 0)
- [6. Push and description: `git -C /tmp/glimmer-pr push --force-with-lea] — openspec/changes/archive/2026-11-07-upstream-muse-glimmer-pr/pr-body.md:9 — "1. *Add the Muse Glimmer 30B NVFP4 single-GPU recipe with images and DFlash2 drafting.* Preparation runs on exactly one GPU."

## Findings
1. suggestion (low) — ignition/recipes/muse_glimmer_nvfp4/config.py:63 — `group = config.get("quantization_config", {}).get("config_groups", {}).get("group_0", {})` runs before `modelopt_nvfp4_layers`, so a present-but-non-mapping `quantization_config` (`None`, `[]`) or `config_groups` (`None`) now raises `AttributeError: 'NoneType' object has no attribute 'get'` where the pre-change code (31f4e6be, parser's `isinstance(..., Mapping)` guards at quantization.py:25-31) raised `ValueError("expected modelopt NVFP4 quantization_config")` / `ValueError("… requires config_groups.group_0")`. Reproduced against both trees. Not traced to a Stated Requirement; partial (side effect of a traced move), so not FAIL condition 6. Acted on: scaffolded `openspec/changes/glimmer-activation-check-mapping-guard/` (fix folds into commit 1 of `pr/muse-glimmer`, which needs a re-acknowledged force-push, so it is its own change rather than an auditor edit to a published PR).
   Dimension: Faithful

## Advisory rules judged
- Stage decay: no — no `## Follow-ups` heading, no tasks added after verification; every discovery line was written at the task that surfaced it (template-dir env, run-suites.sh feature flag, ENOMEM host condition, fixup hook rejection, commit 2 trailer).
- Declined discoveries: n/a — no `declined:` lines; all six discoveries carry `— fixed`. The task 5 discovery keeping commit 2's original Fable 5.1 acknowledgment (content unchanged, digest unchanged `4d3a`) is adequate.

## Live UI pass
n/a — no UI surface
