<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Raise Muse Glimmer's admitted concurrency from 4 to 16 and serve it

Tier: 2 — the diff is four files (recipe declaration, catalog capacity and inventory, two pinning tests, one doc) plus a results folder; the other thirteen sites are read-only planning and runtime consumers cited to show the ceiling is already shared policy
Type: feature

External Review: design.md — Peer Review Verdict

## Stated Requirements

- "glimmer-concurrency-ceiling -- the target is 16" (operator, rad-propose, 2026-11-07)
- "Follow-up change after the PR opens" (operator, 2026-11-07, choosing where the concurrency raise on the 96 GB GB202 lands — recorded in the archived glimmer-activation-check-mapping-guard handoff)
- "I will most likely run it at C8.  What is optimal?" (operator, rad-apply, 2026-11-07; chose "128 rows" when asked for the packed target-row capacity)
- "Also consider that this recipe should be optimized for a single RTX Pro 6000 (96 GB)" (operator, rad-apply, 2026-11-07)
- "why not concurrency 8 with max_concurrency 16?" (operator, rad-apply, 2026-11-07; chose "concurrency 8, ceiling 16" when asked)
- Decided: `Recipe.max_concurrency` becomes 16 (the ceiling `--max-concurrency` may reach) and `Recipe.concurrency` becomes 8 (what an unflagged export prepares); the console passes `--max-concurrency <max_num_seqs>` on every export (`~/LocalApps/vllm-env/console/launchers/docker.py:908`), so the console profile's `max_num_seqs` 16 selects the ceiling package.
- Assumed: the raise lands on deploy as this change and is carried to PR #229 (`/tmp/glimmer-pr`, `pr/muse-glimmer`) as a third commit after it is served and measured, the way the 1M-context commit was; nothing here amends the PR's existing commits.
- Decided: the DFlash2 round catalog declares `token_capacity=128` (full 15-draft depth through 8 active requests, 7 drafts at 16) and keeps `graph_budget=64`; measured on the served instance, drafts at depth 8–15 carry 18% of the accepted tokens, which 64 rows would drop at C8, and on the 96 GB GPU the doubled speculative scratch is small against ~55 GiB of KV pages.
- Assumed: the served check is one console export of the committed ref at the new default plus sixteen simultaneous streaming requests against that instance with the `/metrics` round and acceptance counters read before and after; a slower-than-batch-4 round is a recorded result, not a blocker on landing the declaration.

## Blast Radius

1. ignition/recipes/muse_glimmer_nvfp4/recipe.py:68 — `concurrency=4, max_concurrency=4`: `derive` refuses `--max-concurrency` above `max_concurrency` (ignition/planning.py:540-541) and the default is what an unflagged export prepares; they become 8 and 16.
2. ignition/recipes/muse_glimmer_nvfp4/recipe.py:56-57 — `_GRAPH_BYTES = 4_617_488_302` is the VRAM inventory of the concurrency-4 export `gb202-tp1-202611061529` (graph scratch 1,901 MiB + capture pool 1,126 MiB + …); decode/resume/restore graphs at batches 8 and 16 and five more packed rounds grow it, so it is re-measured from the concurrency-16 export's printed inventory (ignition/preparation/runtime_package.py:238-240).
3. ignition/recipes/muse_glimmer_nvfp4/bootstrap.py:68 — `PreparationConfig.max_active: int = 4` is a construction default; `ignition/recipes/muse_glimmer_nvfp4/recipe.py:35` passes `plan.concurrency`, so every export path already carries the plan's value; read-only.
4. ignition/components/capacity.py:98-131,150-162 — `parallel_speculative_shapes(16, 1..16, 128, 64)` yields 64 packed rounds + 5 width-one at the ceiling (10/13/16/16/9 across batches 1/2/4/8/16, packed rows 17–128 at batch 16, 9–128 at batch 8) and 64 + 4 at the default 8 (13/16/18/17 across 1/2/4/8); the 10 endpoint shapes sit under the 64 budget, so `derive`'s `packed_speculative_shapes` check (ignition/planning.py:449-456) passes; read-only.
5. ignition/planning.py:291,336,470 — `demand_pages` becomes 16 × 8192 = 131,072 pages (104 GiB) and is clamped to the pages the GPU holds; served `context = min(declared, pages × 128)` stays 1,048,576 while `guaranteed_tokens` falls to `pages ÷ 16 × 128`; snapshot target 8 × 16 = 128; read-only, the plan report changes.
6. ignition/recipes/declaration.py:303-307 — `live_rows(16, speculation)` = 1 + 16 × (1 + 0) = 17 permanently backed rows × 62,914,560 B = 1.0 GiB (0.25 GiB today); read-only.
7. helios/src/execution/manifest.rs:306-307,435-438 — `request_capacity` pads 16 to 16 and `max_active` is bounded 1..64; read-only.
8. ignition/cli/serve.py:68 — `queue_capacity = 4 × concurrency`, `frontend_workers = concurrency` follow the plan (16 workers); ignition/cli/help.py:33 prints `1..16`; read-only.
9. tests/recipes/muse_glimmer_nvfp4/test_config.py:99 — pins `RECIPE.concurrency == 4`; moves to 8.
10. tests/recipes/muse_glimmer_nvfp4/test_preparation.py:45-62 — the DFlash catalog test constructs `PreparationConfig` at the default `max_active` 4 and asserts batches 1/2/4 and `(4, 16, 64)`/`(4, 16, 5)`; gains a default-8 case (`(8, 16, 128)`, `(8, 16, 9)`, no batch above 8, 128 sampled rows) and a ceiling-16 case (`(16, 16, 128)`, `(16, 16, 17)`, 256 sampled rows), each within the 64-round budget.
11. tests/recipes/muse_glimmer_nvfp4/test_preparation.py:15,24-30 — pins `config.max_active == 4` for the construction default and rejects `image_concurrency=8` at that default; unchanged because site 3 is unchanged.
12. docs/model-preparation.md:1110,1134 — "four concurrent requests" and "64 speculative rounds across batches one, two, and four with up to 64 packed target rows"; reworded to eight by default and sixteen at `--max-concurrency 16`, the batch list for both, 128 rows.
13. openspec/specs/muse-glimmer-serving/spec.md:88 — "Declared capacities and costs match the prepared layout" says concurrency 4; delta modifies it.
14. qualify/speculative_workload.py:31,288 — the frozen speculation workload holds at most 8 `code_edit` cases per stage and `request_groups` admits concurrency 1/2/4/8 only, so `qualify/speculative_benchmark.py:571` cannot load 16 requests at once without changing the frozen workload; the served check at 16 uses sixteen direct streams and the `/metrics` counters instead, and both files stay untouched.
15. grepped 'concurrency=4\|max_concurrency=4\|four concurrent\|batches one, two, and four' across ignition/recipes/muse_glimmer_nvfp4 tests/recipes/muse_glimmer_nvfp4 docs/model-preparation.md openspec/specs/muse-glimmer-serving — sites 1, 9, 12, 13 only (docs/model-preparation.md:1029,1166,1183 are the Muse Spark and Gemma 4 sections); grepped 'max_active' across ignition/recipes/muse_glimmer_nvfp4 — ignition/recipes/muse_glimmer_nvfp4/bootstrap.py lines 68, 91, 103, 110, 117, 121, 154, 219, 350 and 371, every one derived from the passed value, no second constant.
