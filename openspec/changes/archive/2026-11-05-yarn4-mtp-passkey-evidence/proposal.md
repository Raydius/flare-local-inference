# Export and serve the yarn4 MTP package and measure passkey retrieval past 262,144 tokens for PR #212

Tier: 2
Type: feature

## Stated Requirements

- "A" — the operator's pick from the explore summary: "Close #212's two gaps: export and serve a yarn4 **MTP** package, then run a needle/passkey probe past 262k on both packages"
- "Marked `triage:design-review`: extending the declared Qwen context from 262,144 to 1,048,576 tokens changes the supported serving configuration and needs an explicit decision on supported models, the CLI surface, and long-context quality acceptance. The PR reports formula/identity tests and a fork's DFlash2 probe, but the declared MTP package has not been exported or served and needle/passkey quality beyond the native context is unmeasured. Those remain acceptance gaps before presenting the extended ceiling as supported." (reviewer `original-el8` on NVIDIA/spark-local-inference#212, 2026-11-05)
- "Qualify qwen3.8-flash-yarn4 long-context serving: prepare and export the rope-scaled package, measure the extended score-workspace chunking and graph bytes, and validate needle/passkey retrieval, image requests, and speculation past the native 262,144-token context against native-quality baselines." (`docs/TODO.md:45`, carried into the PR's `docs/TODO.txt`) — conflict: image requests and score-workspace chunking are out of this change; the reviewer named only the export/serve and needle/passkey gaps
- Carried from the DFlash2 changes (vault DFlash2 Recipe Progress): "Exports and serves run containerized from a committed ref (build.sh), never host venv"; "Owner launches console runs"; "GPU 0 belongs to the owner's console instance when it is running; never `docker stop` it"; "Commit/push only when asked"; "Update vault tracker Part A when state changes"
- "Planning only this session (propose's boundary)"

## Blast Radius

1. qualify/dflash2_long_context.py:30–57 — `records_for`, `record_ids`, `prompt` build a service-catalog prompt sized to a token target; :123–143 `measure` and `run` serve one container and send prompts; the new passkey probe imports `records_for` and `measure` and adds a passkey sentence to the catalog instead of a second copy of the sizing
2. qualify/dflash2_long_context.py:50 — `SAMPLING` (temperature 1.0, top_k 20) is the workload's; a quality probe wants near-greedy sampling; grepped 'temperature == 0' across helios/src — helios/src/model.rs:228 refuses `temperature=0` ("greedy decoding incorrectly collapses the model's probability distribution"), so the probe sends `top_k=1` with the workload's other fields
3. qualify/dflash2_served.py:48 `container_command` — serves any package through `flare run` on `--gpus device=0` with `--max-context-tokens`; reused as-is for the MTP yarn4, DFlash2 yarn4, and native MTP packages (GPU 1 is a 16 GiB RTX 5060 Ti, so GPU 0 is the only host for a 1M-context Qwen3.8 package)
4. qualify/speculative_benchmark.py:266 `stream_request`, :331 `read_metrics`, :393 `metric_delta` — reused through `dflash2_long_context.measure`; `analysis["content"]` (:254) is the text the passkey check reads
5. qualify/speculative_workload.py:66 `_catalog` — the haystack; the probe splices the passkey sentence between catalog lines at a depth fraction; the frozen 28-case workload is untouched
6. tests/reference/test_dflash2_long_context.py:12–81 — the pattern for the new `tests/reference/test_long_context_passkey.py` (prompt shape, check, summary, and a `run` with fakes; no container)
7. ignition/cli/preparation.py:53 `rope_scaled` — `--rope-parameters {"factor":4.0,"original_max_position_embeddings":262144}` builds the same `RopeScaling(4., 262144)` at context 1,048,576 as the declared `qwen3.8-flash-yarn4` (ignition/recipes/qwen38/recipe.py:86–90); ignition/recipes/qwen38/bootstrap.py:218–229 `_identity` hashes only factor, anchor, and betas, so the console's flag export and an alias export share one preparation identity
8. grepped 'REF=' across ~/LocalApps/vllm-env/build/flare-ignition/build.sh — line 13 `REF="${1:-patch/07-rope-parameters}"` builds the `local/flare-ignition` image tagged `<sha8>` from any committed fork ref (git archive of the sha, FlashInfer patches from the script dir); the PR head `2ca523b2` (`pr/rope-scaling`, upstream `0009b89d` + two commits) is such a ref, so the served package comes from the PR's code, not the fork's patch stack; the production MTP5 package `gb202-tp1-202610281731` (fork image `9d4f6e2a`, 2026-10-28) exists but is not the evidence — operator's decision 2026-11-05: one GPU-0 window, everything in it
9. grepped 'rope-parameters' across ~/LocalApps/vllm-env/console/ — flare_params.py line 258 declares the console's `export:--rope-parameters` profile field; the owner's production MTP profile already uses factor 4 on the fork image (vault Flare Developer Learnings §7), so the console path is proven
10. docs/TODO.md:45 — the yarn4 qualification item records the measurement once it exists; the same text sits in the PR's `docs/TODO.txt`
11. grepped 'results/' across qualify/dflash2_served.py qualify/dflash2_acceptance.py — `qualify/results/dflash2-served/summary.json` is the committed-evidence convention (qualify/dflash2_served.py:190 `summarize`); the probe writes `qualify/results/yarn4-passkey/summary.json` beside it
12. grepped 'needle/passkey' across docs/TODO.md — docs/TODO.md:47 is the sentence the reviewer quoted; the evidence is posted as one comment on NVIDIA/spark-local-inference#212 answering the two named gaps; the PR's code is not amended
13. grepped 'needle|passkey' across qualify/ tests/ upstream/main — no existing retrieval-quality probe in the fork or upstream; grepped 'dflash2_long_context' across tests/ — `tests/reference/test_dflash2_long_context.py` only

## Why

PR #212 is blocked on two acceptance gaps the reviewer named: the yarn4 MTP package has never been exported or served from the PR's code, and nothing has measured whether the model still retrieves content placed beyond 262,144 tokens under frequency-only YaRN. Both are GPU measurements this host can make now; nothing else on the DFlash queue has a reviewer waiting.

## What Changes

- An image is built from the PR head (`2ca523b2`) and the owner's console exports the MTP package with `--rope-parameters` factor 4 (context 1,048,576); `flare inspect` and `/v1/models` record the served ceiling.
- A new probe, `qualify/long_context_passkey.py`, serves one package at a time through the existing harness and sends passkey prompts — a random digit string spliced into the service catalog at depth 10 %, 50 %, and 90 % of prompts near 200k (below native), 320k, 520k, and 1,000k tokens — under `top_k=1`, checking that the answer contains the passkey; `summary.json` holds a pass/fail grid per package.
- The grid runs on the MTP yarn4 package, the DFlash2 yarn4 package (`gb202-tp1-202611040437`), and the native 262k MTP package at the 200k length as the native-quality baseline.
- `docs/TODO.md:45` records the result; a comment on #212 reports the export, the served ceiling, and the grid.
- No recipe, component, runtime, or PR-code change; the frozen workload is untouched.

## Capabilities

### New Capabilities

- `long-context-passkey-retrieval` — a rope-scaled package retrieves a passkey placed beyond the native context, measured per package against a native baseline

### Modified Capabilities

- none

## Impact

- Code: new `qualify/long_context_passkey.py` and `tests/reference/test_long_context_passkey.py`; `docs/TODO.md`; `qualify/results/yarn4-passkey/`.
- GPU 0, one downtime window: one image build (CPU, before the window), one MTP export (~25 min), three serving sessions (~30 prompts of up to 1M tokens, ~1 h); the owner stops and restarts the console instance around them.
- Upstream: one comment on #212; the branch `pr/rope-scaling` stays at `2ca523b2`.
