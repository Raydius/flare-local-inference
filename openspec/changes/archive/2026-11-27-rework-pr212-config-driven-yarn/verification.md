# Verification — rework-pr212-config-driven-yarn

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-27
Tier: 2
Basis: full
Depth: full

## Dimensions
- Implemented: all 22 checked boxes hold; each cited file was opened in its checkout (core worktree `pr212-core` at 3c73cd1d, recipes worktree `pr212-recipes` at 671b6609, recipes fork `deploy` at 96e21548) and quoted below. `gh pr view` confirms #212 at head 3c73cd1d and recipes #68 at head 671b6609, both open and mergeable. No `QA goal:` line and no release-QA item, so no goal attack ran.
- Reachable: every new path is reached. `Checkpoint.context` is set by the Qwen hook registered at recipes `recipe.py:177`, called from core `ignition/registry.py:198`, and read through `Plan.declared_context` (`ignition/planning.py:348`) by `planning.py`, `preparation/bootstrap.py:132`, `cli/serve.py:67`, `cli/help.py:46` and `parameters.py:1276`; no reader of `recipe.context` remains in `ignition/`. `serving_context` is called from `checkpoint.py:162`, `bootstrap.py:490` and the hook; the YaRN table branch is reached from `bootstrap.py:494`; the SM103 refusal runs under `planning.plan` (`planning.py:448`), which passes the checkpoint and GPUs. The fork's factor is reached by the existing `--rope-parameters` path, and `--task` and `--compare` are parsed in the probe's `main`.
- Tested: no task carries a `Verify:` clause, so every box records `no covering command` and nothing was run here. Read against the diffs, the new tests fail without their change (the core test is absent on the base; the fork test compares with the reference embedding). Two low gaps are recorded as findings 1 and 2.
- Optimal: no admissible finding. One repetition is listed as a low under Code review: `cli/serve.py:67` spells `checkpoint.context or recipe.context` while `plan.declared_context` is in scope.
- Reused: every reuse claim is backed by an import. Recipes PR: `from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS` (`qwen38/attention.py`, the same call `kimi31/dflash.py:58` makes), `from hyperflare_recipes.recipes.qwen38.checkpoint import inspect_qwen38_checkpoint, serving_context` (`bootstrap.py`), `from .checkpoint import CheckpointError, serving_context` and `from .checkpoint_schema import ARCHITECTURE, MODEL_ID, REVISION, TEXT_GEOMETRY` (`recipe.py`). Probe: `from qualify.speculative_workload import SAMPLING, _catalog, _message, value_sha` and `from qualify.dflash2_long_context import TOKENS_PER_RECORD, draft_counts, measure, records_for`. No rotary validator existed before `serving_context` (`grep rope` over `checkpoint_schema.py` on the base finds none).
- Faithful: all four Stated Requirements are covered and every behaviour change traces to one. Review of #212 is the proposal's Why and Blast Radius; the rework is sections 2 to 6; the SM103 deferral is task 3.4 (refusal, score workspace untouched); the recommended route is tasks 7.1 to 7.3 and 5.3. Tasks 5.3 and 7.3 ran on GPU 0 and left a re-exported profile serving: that is the fourth Stated Requirement itself, inside the one window the operator chose, so it is the deliverable and no pre-close production check.
- Reviewed: the change code review ran over the core commit, the recipes PR commit, the fork commit, and the change's proposal, design, specs and drafts. No high or medium finding; 16 lows are listed under Code review. Peer coverage: complete.
- Environment: doctor reports omp trailer and tool-call hooks absent in this checkout and its worktrees, tool-call announcements off, and the `product-board` skill link missing; the peer-review server is present. changed_files 0 (the change's code lives on `pr/rope-scaling` and in the recipes repository; this checkout holds only the untracked change directory). suite runs: 0 in 240 min, 0 deliberate re-runs, budget 3. verify runs: 0 ran, 0 reused, 0 covered. The three audit tracks were read by the auditor directly, with no child dispatched. The installed host transformers is 5.21.0.dev0, where the commit messages name 5.23.0.

## Evidence audited
- [1.1 Read the original Qwen3.8 Flash Next model card and record, in a `] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:16 — "notes: https://huggingface.co/Qwen/Qwen3.8-Flash-Next documents the block under" (no covering command)
- [2.1 Tag the current PR head as `backup/pr-rope-scaling-15d8b0b0`, then] — .git/refs/tags/backup/pr-rope-scaling-15d8b0b0:1 — "15d8b0b0d38e9ef0575b6e7915154525deac5a76" (no covering command)
- [2.2 In `ignition/declaration.py` add to `Checkpoint` an optional `cont] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:25 — "(Checkpoint.context, rope_parameters)" (no covering command)
- [2.3 Add a `docs/TODO.txt` entry of at most two lines naming a scaled c] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:28 — "evidence: recorded in task 3.7" (no covering command)
- [2.4 Commit the core change as one commit with both trailers and a four] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:31 — "core commit 3c73cd1d on 74618652 (Signed-off-by, Design-Signed-Off-By design-review=15f3)" (no covering command)
- [3.1 Create `pr/qwen38-config-yarn` in a worktree on the fetched recipe] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:38 — "with the core worktree on PYTHONPATH — 15 recipes" (no covering command)
- [3.2 In `hyperflare_recipes/recipes/qwen38/attention.py`, make `RotaryT] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:40 — "(RotaryTables.allocate yarn branch)" (no covering command)
- [3.3 In the recipe's checkpoint module and `checkpoint_schema.py`, make] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:42 — "6 accepted and 12 refused configurations, hook test passed" (no covering command)
- [3.4 In `Qwen38Parameters.validate`, refuse a checkpoint whose context ] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:45 — "test_scaled_context_is_refused_on_sm103 passed" (no covering command)
- [3.5 Add an identity test: the preparation identities of the native con] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:47 — "test_scaled_and_native_configurations_have_distinct_identities passed" (no covering command)
- [3.6 On GPU 1 with the host venv (`CUDA_VISIBLE_DEVICES=1 FLARE_RUN_CUD] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:49 — "60 passed (30 native, 30 YaRN over a 1,048,576-row table) on the RTX 5060 Ti" (no covering command)
- [3.7 Update the recipes repository's `docs/model-preparation.md` with t] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:51 — "recipes commit 671b6609 on 8df08c21" (no covering command)
- [4.1 Widen `qualify/long_context_passkey.py` with an optional `--task` ] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:56 — "(chain, tracking_question, prompt task argument, --task)" (no covering command)
- [4.2 Add the paired comparison record: given a native and a scaled summ] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:58 — "test_compare_pairs_native_and_scaled_cells — 18 pairs, passed" (no covering command)
- [5.1 Prepare a YaRN checkpoint directory under `~/models` as a symlink ] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:63 — "whose only difference is the card's YaRN block" (no covering command)
- [5.2 Build one image from the core PR head and the recipes PR head, usi] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:66 — "built the image tagged `3c73cd1d-ec62b550` under" (no covering command)
- [5.3 With the operator's go-ahead for one GPU-0 window, export and serv] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:71 — "on package gb202-tp1-202611280514" (no covering command)
- [7.1 In the recipes fork's `deploy` checkout, make `RotaryTables.alloca] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:79 — "(RotaryTables.allocate, factor applied when scaling is set)" (no covering command)
- [7.2 Commit the fork fix and the probe changes of section 4 on the reci] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:83 — "recipes fork deploy commit 87c08409 (design-review=8fdd, unpushed)" (no covering command)
- [7.3 After the test packages of task 5.3 are measured, and only if the ] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:86 — "max_model_len 1048576" (no covering command)
- [6.1 Draft, in this change directory, the new #212 description, the rec] — openspec/changes/rework-pr212-config-driven-yarn/drafts/pr212-reply.md:14 — "6. **Description.** Rewritten in the four-part order, without process statements." (no covering command)
- [6.2 After the operator approves the drafts, force-push `pr/rope-scalin] — openspec/changes/rework-pr212-config-driven-yarn/tasks.md:95 — "https://github.com/local-inference-lab/hyperflare-recipes/pull/68 (head 671b6609, MERGEABLE)" (no covering command)

No box was skipped.

### Code read outside this checkout

The linter accepts only paths of this checkout, so the boxes above cite the task record. The lines below are what the auditor read in the checkouts that hold the code.

| Box | Checkout | File | Line | Line read |
|---|---|---|---|---|
| 2.2 | core worktree `pr212-core` | `ignition/declaration.py` | 104 | context: int \| None = None |
| 2.3 | recipes worktree `pr212-recipes` | `docs/TODO.txt` | 363 | [ ] Qualify a YaRN-scaled Qwen3.8 Flash Next context on SM103. Planning rejects it |
| 2.4 | core worktree `pr212-core` | `tests/cli/test_planning.py` | 361 | def test_checkpoint_context_replaces_the_recipe_ceiling(): |
| 3.1 | recipes fork `deploy` checkout | `.git/worktrees/pr212-recipes/HEAD` | 1 | ref: refs/heads/pr/qwen38-config-yarn |
| 3.2 | recipes worktree `pr212-recipes` | `hyperflare_recipes/recipes/qwen38/attention.py` | 75 | cos.mul_(scale) |
| 3.3 | recipes worktree `pr212-recipes` | `hyperflare_recipes/recipes/qwen38/checkpoint.py` | 113 | def serving_context(text: dict) -> int: |
| 3.4 | recipes worktree `pr212-recipes` | `hyperflare_recipes/recipes/qwen38/recipe.py` | 92 | and any(gpu.capability == (10, 3) for gpu in context.gpus)): |
| 3.5 | recipes worktree `pr212-recipes` | `tests/recipes/test_qwen38_rope_scaling.py` | 140 | def test_scaled_and_native_configurations_have_distinct_identities(): |
| 3.6 | recipes worktree `pr212-recipes` | `tests/recipes/test_qwen38_norm_rope_and_terminal_prefill_cuda.py` | 41 | # YaRN tables carry the attention factor and cover positions beyond the trained 262,144. |
| 3.7 | recipes worktree `pr212-recipes` | `docs/model-preparation.md` | 521 | `text_config`, leave `max_position_embeddings` at 262144, and export the |
| 4.1 | recipes fork `deploy` checkout | `qualify/long_context_passkey.py` | 60 | def chain(seed, length, depth, records, passkey, hops=HOPS): |
| 4.2 | recipes fork `deploy` checkout | `qualify/long_context_passkey.py` | 137 | def compare(pairs): |
| 5.1 | YaRN checkpoint folder under the models directory | `config.json` | 125 | factor |
| 5.2 | recipes fork `deploy` checkout | `qualify/results/yarn-config/yarn-passkey.json` | 5 | image tag 3c73cd1d-ec62b550 (recipes 671b6609 plus a local commit locking core 3c73cd1d) |
| 5.3 | recipes fork `deploy` checkout | `qualify/results/yarn-config/yarn-tracking.json` | 6 | ckpt/gb202-tp1-202611280514 |
| 7.1 | recipes fork `deploy` checkout | `hyperflare_recipes/recipes/qwen38/attention.py` | 74 | factor = 0.1 * math.log(scaling.factor) + 1.0 |
| 7.2 | recipes fork `deploy` checkout | `qualify/results/yarn-config/fork-serving-passkey.json` | 5 | image tag 86029f80-87c08409 |
| 7.3 | recipes fork `deploy` checkout | `qualify/results/yarn-config/fork-serving-passkey.json` | 7 | 2026-11-27 Flare v0.3.2-r2 YaRN attention factor |

## Findings
1. low, suggestion — /home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/tests/recipes/test_qwen38_rope_scaling.py:147 — the closing assertion `identity(NATIVE) == identity(dict(NATIVE))` compares one input with a copy of itself, so it passes whatever `_identity` does. Task 3.5's "native identity equals the one on recipes `upstream/main`" rests on `_identity` being absent from the diff, not on this test. Recorded only; the audit edits no test.
   Dimension: Tested
2. low, suggestion — /home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/cli/serve.py:67 — the request body limit and the help line (`ignition/cli/help.py:46`) now read the checkpoint ceiling and no test covers either; the core test checks the plan, the report and the prefill-chunk bound. The body limit was exercised only by the live 1,000k-token requests. Recorded only; the audit edits no test.
   Dimension: Tested

## Code review
Review standard: 001ce8ddb3cf5aa072e28e2cc5bbab28ff39cdc0
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/proposal.md part=1/2 blob=d0342a54dc61473719b4b131aba90f149c181709 status=returned reason=call-3
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/proposal.md part=2/2 blob=d0342a54dc61473719b4b131aba90f149c181709 status=returned reason=call-4
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/design.md part=1/2 blob=66121d01e9c960ee8bfb457352d4cc4fe34b4ca8 status=returned reason=call-3
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/design.md part=2/2 blob=66121d01e9c960ee8bfb457352d4cc4fe34b4ca8 status=returned reason=call-4
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/specs/config-driven-rope-scaling/spec.md part=1/1 blob=2b66f910e7b1bead19c60d1523ba0536601c8c88 status=returned reason=call-3
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/specs/long-context-passkey-retrieval/spec.md part=1/1 blob=11cb1b5f5363f02419491b360869169c56bae22e status=returned reason=call-3
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/specs/rope-scaled-export/spec.md part=1/1 blob=18280888f9658f01680bb833aeab4b1c900d45e7 status=returned reason=call-3
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/drafts/pr212-description.md part=1/1 blob=0213a7f630178821736de775db7d560d695e3b9a status=returned reason=call-4
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/drafts/pr212-reply.md part=1/1 blob=673b19c6f71c527c7215c762ddf6015816cc3cce status=returned reason=call-4
Peer part: path=openspec/changes/rework-pr212-config-driven-yarn/drafts/recipes-pr-description.md part=1/1 blob=022c5622d897b7ecde5f11a7bbbd101099da5c65 status=returned reason=call-4
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/cli/help.py part=1/1 blob=ea4e4eeaf73df5c71f9d887120965219c13653a5 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/cli/serve.py part=1/1 blob=ed652359130b1d2d03baefa7c4ed272512f82d35 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/declaration.py part=1/1 blob=eed71a80c4944e82daf7b0cc6e1be02cf8d3829c status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/parameters.py part=1/1 blob=f4551077d72705d8d950f99c9f04e20facf7fb56 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/planning.py part=1/1 blob=467cb849cdee69c5900f3b4b3f7b5cf97c189dfb status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/ignition/preparation/bootstrap.py part=1/1 blob=01df6aa0609455b4bb502447e329344d6d247da1 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/flare-local-inference.worktrees/pr212-core/tests/cli/test_planning.py part=1/1 blob=d872d126a6273c04675a0f17c53e95c78d2689d0 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/docs/TODO.txt part=1/1 blob=538acfceaa7ac8b7bb688a1cf51780bc37e33916 status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/docs/model-preparation.md part=1/1 blob=4970bddd389b9382871a22b37d4e9af205f6733d status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/hyperflare_recipes/recipes/qwen38/attention.py part=1/1 blob=673d3e44f28da6752f6c5466a1f8d790e246aacf status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/hyperflare_recipes/recipes/qwen38/bootstrap.py part=1/1 blob=1d3165d465e8cfe1ced77988130621ff42417b19 status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/hyperflare_recipes/recipes/qwen38/checkpoint.py part=1/1 blob=fdcb9052159473fb8551b45391edf25134ce6d5f status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/hyperflare_recipes/recipes/qwen38/checkpoint_schema.py part=1/1 blob=31d66949ae9b963bc022f56a1e1b6d196ba712fb status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/hyperflare_recipes/recipes/qwen38/recipe.py part=1/1 blob=0ddf71af60398cfffe95994638cd45df955896e9 status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/tests/recipes/test_qwen38_norm_rope_and_terminal_prefill_cuda.py part=1/1 blob=e8f3382e85a63f2e627f195116bae58a6d3b69db status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes.worktrees/pr212-recipes/tests/recipes/test_qwen38_rope_scaling.py part=1/1 blob=b841c590b0f3b9e1c0ad7a1f39bbdcc0b7f3d0e9 status=returned reason=call-2
Peer part: path=/home/rjdollete/dev/hyperflare-recipes/hyperflare_recipes/recipes/qwen38/attention.py part=1/1 blob=000f015b488da1799ecd9cf3aa9893c6955d6a76 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/hyperflare-recipes/qualify/long_context_passkey.py part=1/1 blob=69b7f14c48d02bd81e222e77ca7e54e67c2e8621 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/hyperflare-recipes/tests/recipes/test_qwen38_attention.py part=1/1 blob=424994a717237e7a0b9de18770cca1609d4d9c38 status=returned reason=call-1
Peer part: path=/home/rjdollete/dev/hyperflare-recipes/tests/reference/test_long_context_passkey.py part=1/1 blob=abbf9a3db9cf6fa5da31fa39dee6e4111f034546 status=returned reason=call-1
Peer coverage: complete

Scope notes:

- The formal diff of this checkout is the untracked change directory; the code commits live in other checkouts and were reviewed from `git show` of 3c73cd1d, 671b6609 and 87c08409.
- The documents went to the peers without their title and `Tier:`/`Type:` lines, the design's transcribed External Review section, and the spec files' Purpose and `## ADDED Requirements` header lines.
- The six recorded result files of fork commit 96e21548 (`qualify/results/yarn-config/*.json`) were read by the auditor and not sent to the peers.
- No secret candidate was found on any line, so nothing was masked.

Call 1 (core commit 3c73cd1d and fork commit 87c08409):
Quorum: achieved (tier 2, weight 4/3)
Certainty: 0.825
Caller Agreement: 0.5
Arbiter: ok
Sources: claude-haiku-5-5 ok 0.8; qLLM ok 0.3; gemini-flash-lite-latest ok 0.4; deepseek-flash ok 0.3; claude-sonnet-5-5 ok 0.85

Call 2 (recipes commit 671b6609, with excerpts of two drafts):
Quorum: achieved (tier 2, weight 4/3)
Certainty: 0.825
Caller Agreement: not returned
Arbiter: ok
Sources: claude-haiku-5-5 ok 0.8; qLLM ok 0.25; gemini-flash-lite-latest ok 0.1; deepseek-flash ok 0.05; claude-sonnet-5-5 ok 0.85

Call 3 (proposal What Changes, design Decisions, the three spec deltas):
Quorum: achieved (tier 2, weight 4/3)
Certainty: 0.825
Caller Agreement: 0.3
Arbiter: ok
Sources: claude-haiku-5-5 ok 0.8; qLLM ok 0.25; gemini-flash-lite-latest ok 0.15; deepseek-flash ok 0.25; claude-sonnet-5-5 ok 0.85

Call 4 (remaining proposal and design sections, the three drafts in full):
Quorum: achieved (tier 2, weight 4/3)
Certainty: 0.8
Caller Agreement: 0.4
Arbiter: ok
Sources: claude-haiku-5-5 ok 0.7; qLLM ok 0.2; gemini-flash-lite-latest ok 0.4; deepseek-flash ok 0.3; claude-sonnet-5-5 ok 0.9

Low findings (no action asked):

1. CONFIRMED, low — recipes `qwen38/checkpoint.py:137-138`: a `factor` of NaN or Infinity makes `int(context)` raise ValueError or OverflowError instead of the CheckpointError naming `factor`; OverflowError is outside the core caller's except list (`ignition/registry.py:201`) and surfaces as a traceback. Needs a malformed config; the configuration is still refused.
2. CONFIRMED, low — recipes `qwen38/recipe.py:155`: a `config.json` whose root is not an object raises AttributeError, which the core caller does not convert to a usage error.
3. CONFIRMED, low — recipes `qwen38/checkpoint.py:130`: the optional YaRN keys are admitted by name with no type or range check, so a wrong value fails late in the reference, and `attention_factor: 0` builds all-zero tables.
4. CONFIRMED, low — recipes `qwen38/checkpoint.py:122`: a rotary block with no `rope_type` is refused, while `RotaryTables.allocate` (`attention.py:51`) treats a missing key as `default`. The pinned config carries `rope_type: "default"`, so today's checkpoint is unaffected.
5. CONFIRMED, low — recipes `qwen38/bootstrap.py:190`: a native configuration passes the first context bound up to 1,048,576 and is refused later at the rotary site (`bootstrap.py:491`).
6. CONFIRMED, low, wording — `drafts/recipes-pr-description.md:9` and the 671b6609 commit body say the default factor applies "unless the configuration gives `attention_factor` or `mscale`"; the reference overrides it only for `attention_factor`, or for `mscale` and `mscale_all_dim` together.
7. CONFIRMED, low — fork `qualify/long_context_passkey.py:141`: `compare` reads `row["task"]`, so a summary written before this change (no `task` key) raises KeyError.
8. CONFIRMED, low — fork `qualify/long_context_passkey.py:163`: the per-cell folder is `{length}-{depth}` with no task, so two tasks sent to one `--output` overwrite each other's captures.
9. PLAUSIBLE, low — fork `qualify/long_context_passkey.py:83`: `prompt(task="tracking")` with the default `length=None` seeds the variable names with `None` silently.
10. CONFIRMED, low — core `ignition/cli/serve.py:67` repeats `checkpoint.context or recipe.context` with `plan.declared_context` in scope.
11. CONFIRMED, low, wording — `design.md:33` (D1) says the config carries "a matching `max_position_embeddings`", which D5 and the spec replaced with `max_position_embeddings` left at 262,144.
12. CONFIRMED, low, wording — `design.md:25` lists `docs/TODO.txt` and only two source files under the core branch; the entry landed in the recipes repository and the core commit touches six source files (task 2.2 and 2.3 discoveries record both).
13. CONFIRMED, low, wording — `proposal.md:30` states the serving-profile re-export without the condition `design.md:117` (D12) and task 7.3 carry.
14. CONFIRMED, low, wording — the long-context spec's "eighteen cell pairs" scenario speaks of "both summaries for the same task and seed", where eighteen is the count over both tasks.
15. PLAUSIBLE, low — the "No second mechanism" scenario's `git grep` would also match a doc or test mention, so it shows absence only while nothing mentions the words; it finds nothing on both PR branches today.
16. PLAUSIBLE, low — `proposal.md` Blast Radius item 19 names the operator's profile and image tag; these are the operator's own repository's records, and the published texts name only GPU models and package ids.

NOT A DEFECT items checked:

- "The fork applies `0.1 ln(factor) + 1` with no clamp for a factor at or below 1" — `ignition/declaration.py:86` refuses a `RopeScaling` whose factor is not above 1.
- "The fork ignores an explicit `attention_factor` or `mscale`" — `ignition/cli/preparation.py:79` refuses those keys on the fork's flag.
- "Other readers of `recipe.context` remain" — `git grep 'recipe\.context' -- ignition` on 3c73cd1d finds only the four `checkpoint.context or recipe.context` sites.
- "The plan report records both values, so the native report differs from `upstream/main`" — `ignition/planning.py:389` adds `rope_parameters` only when the checkpoint sets one, and `checkpoint_context` existed before.
- "The reply says the alias and flag are removed while the design keeps them" — the reply speaks of the PR branches, where `git grep -E 'rope-parameters|yarn4'` finds nothing; the design keeps them on the fork's `deploy` only.
- "Serving on SM120 is not backed by the tests" — backed by the recorded grids, `qualify/results/yarn-config/yarn-passkey.json:6` (served context 1,048,576, 18 of 18 completed).
- "Thirty-six short-reply cells miscounts" — 18 short cells on each of two packages, per `comparison.json` (18 pairs).

## Advisory rules judged
- Stage decay: no. Section 7 was added during apply on the operator's own choice (the fourth Stated Requirement); tasks.md has no follow-up heading and no task postdates this file.
- Declined discoveries: four, each adequate. The `tests/preparation` interpreter crash reproduces on the pristine base and is disclosed in the commit. The fork docstring and the four-line TODO entry are wording whose fix would move a head every measurement names. The dense Qwen3.8 27B recipe's missing factor is a different model whose reference behaviour is unverified and which no window measured; it stays an open question for the operator and changed no carried behaviour.
- Split-offs: no. `openspec/changes/` holds only this change and `archive/`.

## Live UI pass
n/a — no UI surface
