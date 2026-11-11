# Measure upstream's qwen38 DFlash2 path against the fork's qwen38_dflash at 16 sequences and 1M context

Tier: 2
Type: feature

## Stated Requirements

- "I want to find the optimal configuration that uses the upstream `qwen38` with `--speculative-method dflash2` but from my recipe also gains mixed prefill, available history drafts, built-in yarn4, available A16 decode, optional MXFP8 draft projections, optional tensor parallel, and with direct BF16 feature averaging."
- "The goal is to have the \"ultimate\" recipe that is usable for all `qwen38` purposes (including dflash2 and other future drafts)"
- "In addition to 1M context, I also want to target max num seqs = 16."
- "We will need to add the patch that allows the draft folder"
- "Essentially I want to merge the wins from my recipe with the wins from upstream to end up with the optimal configuration.  The question is whether my work has anything positive to contribute to the upstream version, and if so, we can work from there."

Assumed: the only source change is the draft-folder patch: `--draft-model` is accepted with `--recipe qwen38` when the speculative method is dflash2 and is rejected, as today, for MTP. Porting mixed prefill and history drafts onto upstream's `qwen38` DFlash2 path is a later change that these numbers justify or cancel.
Assumed: four arms, each a console export of the served checkpoint (`/ckpt` = `huginnfork/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-Abliterated@v3`) with yarn 4.0 / 262144, fixed-7, `--max-concurrency 16`: (U) `--recipe qwen38` with speculative method dflash2; (Q) U plus `--online-mxfp8` and `--draft-head-quantization mxfp8`; (A) U plus A16 decode; (M) the fork's `--recipe qwen38_dflash` with `--mixed-prefill`.
Assumed: "max num seqs = 16" means 16 prepared decode slots sharing one KV cache that must still hold a single 1,048,576-token prompt; sixteen simultaneous 1M prompts cannot fit on this GPU and are not a target.
Assumed: an arm that fails to export, or whose KV cache holds fewer than 1,048,576 tokens at 16 sequences, is a recorded result and is not repaired here.
Assumed: tensor parallel is not measured (one 96 GB GPU); history drafts stay off in every arm, as served today.
Assumed: the baseline is the served entry `gb202-tp1-202611101936` (fork recipe, mixed prefill, 8 sequences), and it is served again at the end unless the operator picks an arm.

## Why

The fork serves DFlash2 through its own recipe, and upstream now ships DFlash2 with options the fork lacks (A16 decode, MXFP8 draft). Upstream's path rejects mixed prefill, the fork's main addition, so the two cannot be combined without a port; no measurement yet says whether mixed prefill is worth porting, or whether 1M context still fits at 16 sequences on 96 GB.

## What Changes

- Console (`~/LocalApps/vllm-env`, outside this repo): two Ignition configuration fields, `defaults.speculative_method` and `preparation.force_moe_decode_a16`, so a profile can export upstream's DFlash2 path and A16 decode. The console exposes neither today.
- Fork patch: upstream's `qwen38` DFlash2 path loads the draft from the `--draft-model` folder when one is given, and downloads the pinned snapshot only without it. Today the flag is refused for `qwen38` ("loads no external draft"). A CPU test covers both cases; the image is rebuilt from the new `deploy` tip.
- Four console profiles copied from *2026-11-10 Flare DFlash2 Rebase3*, one per arm, exported on the image built from this change's patch commit.
- Per arm: plan and memory figures, the fixed-7 `code_edit` probe at concurrency 1 and 8, a 16-client throughput run, and the 1M passkey check.
- Results committed under `qualify/results/dflash2-paths-c16/` and written to the vault tracker with a recommendation: port mixed prefill, or retire `qwen38_dflash`.

## Capabilities

### New Capabilities

None. No requirement of this repository changes; `.openspec.yaml` sets `skip_specs: true`.

### Modified Capabilities

None.

## Impact

- Flare source: the draft-folder patch (`ignition/recipes/qwen38/recipe.py`, the `--draft-model` check in `ignition/cli/preparation.py`, one test, README's patch table). One results directory and its probe scripts are added under `qualify/`.
- vllm-env console gains two profile fields and needs one restart of `localai.service`.
- GPU 0 is taken from the served instance for three to four exports (roughly 40 to 90 minutes each) and their probes.
- Model disk: A16 repacks the routed experts (about 60 GiB); 170 GiB is free.

## Blast Radius

1. ignition/planning.py:555 — upstream rejects `--mixed-prefill` with speculative method dflash2; arms U, A, Q run without mixed graphs.
2. ignition/recipes/qwen38/bootstrap.py:132 — the same rejection in `PreparationConfig`; line 150 rejects A16 decode with mixed prefill.
3. ignition/recipes/qwen38/recipe.py:62 — the DFlash2 draft comes from `snapshot_download(..., cache_dir=plan.hub_cache)`; `plan.draft_model` is not read on this path. The patch reads it first.
9. ignition/cli/preparation.py:131 — `_check_draft_model` refuses `--draft-model` for a recipe with no `draft_files`; `qwen38` declares none (ignition/recipes/declaration.py:281 default). The patch lets the DFlash2 method name its draft files.
10. ignition/recipes/qwen38_dflash/recipe.py:54 and ignition/recipes/muse_glimmer_nvfp4/recipe.py:21 — the two recipes that already honor `plan.draft_model`; the patch follows their form.
4. ignition/cli/config.py:24 — `defaults.speculative_method` and (line 26) `preparation.force_moe_decode_a16` are configuration-file settings, which the console's generated TOML can carry.
5. ignition/recipes/qwen38_dflash/recipe.py:95 — the fork recipe declares `concurrency=32, max_concurrency=64` and `mixed_prefill=True`; arm M exports it at 16 for the first time.
6. qualify/speculative_benchmark.py:571 — the frozen workload's concurrency choices stop at 8; the 16-client measurement needs a second probe.
7. grepped 'speculative-method\|force-moe-decode-a16\|speculative_method\|force_moe_decode_a16' in ~/LocalApps/vllm-env/console/flare_params.py (outside this repo) — no field for either; lines 345-353 hold the neighbouring `defaults.*` configuration fields.
8. grepped 'CONCURRENCIES\|choices=(1, 2, 4, 8)' across qualify/ — `qualify/dflash2_draft_sweep.py:30` and `qualify/speculative_benchmark.py:571`; no probe in this repo drives 16 clients.
