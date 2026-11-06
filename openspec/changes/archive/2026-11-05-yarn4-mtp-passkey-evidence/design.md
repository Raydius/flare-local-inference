# Design — yarn4-mtp-passkey-evidence

## Approach

Scope and the Stated Requirements: proposal.md. No recipe, component, runtime, workload, or PR-code change; this change produces a package, a probe, and a measurement.

Assumed (open readings of the request, corrected by the operator at one line each):
- Operator's decision 2026-11-05 (after two deferrals, reversed: "If the GPU needs to be used at all, I'd rather just do all of them"): one GPU-0 downtime window holds everything — the served MTP package comes from an image built at the PR head `2ca523b2` (`pr/rope-scaling` = upstream `0009b89d` + the two PR commits), not the fork image, so the reviewer sees the PR's own code serve; the production package `gb202-tp1-202610281731` (fork image `9d4f6e2a`) is not used.
- Assumed: the export goes through the owner's console with `export:--rope-parameters {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}` and Max Context Length 1,048,576, the route the production MTP profile already uses; the alias `qwen3.8-flash-yarn4` resolves to the same `RopeScaling(4., 262144)` at 1,048,576 (`ignition/cli/preparation.py:53` `rope_scaled`, `ignition/recipes/qwen38/bootstrap.py:218` `_identity`), so one export covers both PR commits and the comment says so.
- Assumed: "needle/passkey" is passkey retrieval as in Mohtashami & Jaggi 2023 (Landmark Attention, arXiv:2305.16300): one sentence "The pass key is NNNNN. Remember it." buried in filler at a depth, then "What is the pass key?"; the filler is the workload's service catalog (`qualify/speculative_workload._catalog`) rather than the paper's repeated grass sentence, so the haystack is the same text the earlier probes used. NIAH (gkamradt) is the same shape with a prose needle; RULER (NVIDIA, arXiv:2404.06654) is the heavier multi-needle suite and is not run.
- Assumed: the grid is 4 lengths × 3 depths — targets 200k (below native), 320k, 520k, 1,000k prompt tokens; depths 0.1, 0.5, 0.9 — on each yarn4 package, and the 200k row on the native 262k MTP package (`gb202-tp1-202611020610`) as the native-quality baseline; one request per cell, longest first so no prefix is reused.
- Assumed: a cell passes when the five-digit passkey appears in the reply's content; the package passes a length when all three depths pass; the comment reports the raw grid and names any failing cell — this change sets no accept/reject threshold for the PR, which is the reviewer's call.
- Assumed: the DFlash2 yarn4 package is served fixed-7 as it was exported (`gb202-tp1-202611040437` was exported adaptive; it is served as-is with history drafts off) and the MTP packages with their defaults; speculation stays on in every cell because the question is served quality, not draft acceptance.

1. **Image** (CPU, before the window): `FLARE_FORK_SRC=$PWD ~/LocalApps/vllm-env/build/flare-ignition/build.sh pr/rope-scaling` → the `local/flare-ignition` image tagged `2ca523b2`; kernel-folder link as the rebase2 image (vault tracker §12). The build applies the script dir's FlashInfer patches on top of the tree's pin (`1f8ed8a3`); a failed build or export at "Prepare and tune kernels" is recorded as a discovery and the fallback is the fork image `e6fc2ef6` (same rope code), named as such in the comment. The DFlash2 and native packages are served by the images that exported them (`a945c147`; the native entry's record under `~/LocalApps/vllm-env/state/flare/entries/`).
2. **Export** (owner launches, opens the window): duplicate the production MTP profile through the console API, set `docker_image` to the `2ca523b2` tag, keep `export:--rope-parameters` factor 4 and `max_model_len` 1,048,576; record catalog/tune/compile/warm/export timings, the cache entry id, `flare inspect` (`max_sequence_tokens` 1,048,576, graph count) and served `/v1/models` `max_model_len` into `~/.cache/yarn4-passkey/console-export.log` and `inspect.json`.
3. **Probe** (`qualify/long_context_passkey.py`, new): `prompt(records, depth, passkey, *, model)` builds the catalog with `records_for(tokens)` from `dflash2_long_context`, splits the catalog lines, inserts `The pass key is {passkey}. Remember it. {passkey} is the pass key.` after line `round(depth × records)`, and asks "What is the pass key? The pass key is" with `max_tokens` 32, the workload's `SAMPLING` overridden by `top_k=1` (temperature 0 is refused: `helios/src/model.rs:228`), seeded, `return_token_ids=False`; `passkey` is a five-digit string from `random.Random(seed)`, one per cell. `measure` is `dflash2_long_context.measure` (metrics before/after kept so the summary also carries accepted-per-round per cell). `run` serves one package through `dflash2_served.container_command(history_drafts=False, max_context_tokens=args.max_context_tokens)` on GPU 0 and iterates lengths descending × depths; `summary.json` holds per cell prompt tokens, passkey, reply, `passed`, request seconds, accepted-per-round, and a `grid` of pass/fail by length × depth; exit 0 when every cell passed. CLI: `--image --package --publisher-keys --trust-store --output --model-id --max-context-tokens --lengths 200000,320000,520000,1000000 --depths 0.1,0.5,0.9 --timeout 900 --ready-timeout 900`.
4. **Tests** (`tests/reference/test_long_context_passkey.py`, on the pattern of `tests/reference/test_dflash2_long_context.py`): the needle sits at the requested line and the catalog is otherwise `_catalog(records)`; the request carries `top_k=1`, `max_tokens` 32, no token echo; `check` passes on a reply containing the digits and fails otherwise; `summary` builds the grid and `all_passed`; `run` with fakes serves once, sends the longest length first, and writes `summary.json`.
5. **Runs** (operator stops the console instance first; GPU 0 is the only host): native MTP 262k package at 200k only (`--lengths 200000 --max-context-tokens 262144`), then the MTP yarn4 package, then the DFlash2 yarn4 package, outputs under `~/.cache/yarn4-passkey/<package>/`; the three `summary.json` files are copied to `qualify/results/yarn4-passkey/{native-mtp,mtp-yarn4,dflash2-yarn4}.json`.
6. **Record**: `docs/TODO.md:45` gains one line with the grid result and the results path (the item stays open for image requests and score-workspace chunking); vault tracker Part A (Now / Done) and section 11; one comment on #212 with the export facts, the served ceiling, the grid per package, and the identity note from the second Assumed line.

## Decisions

- Passkey, not the three-record question — the earlier probe's question (quote three records' fields) is retrieval too, but the reviewer asked for needle/passkey by name and the passkey form has a published definition and a single unambiguous pass criterion per cell.
- `top_k=1` rather than the workload's `top_k=20` — the measurement is whether the model can retrieve, not how it samples; Flare refuses `temperature=0`, and `top_k=1` is the supported way to make the reply deterministic.
- Catalog filler rather than the paper's repeated sentence — the catalog is the text the fold, yarn4, and sweep probes already served, so prefill behavior at these lengths is known; a repeated sentence would also compress to very few distinct tokens and understate the haystack.
- One request per cell, no repeats — 27 cells at up to 1M tokens is about an hour of GPU 0; a failing cell is re-run once by hand and recorded as such before it is reported.
- The native 262k MTP package at 200k as the baseline — "native-quality baselines" (`docs/TODO.md:45`) means the unscaled package; its 200k row says whether a sub-native failure on the scaled package is the scaling or the model.
- No PR threshold — the reviewer asked for the measurement; what counts as supported is the maintainers' decision and the comment presents the grid without claiming acceptance.
- A new probe module importing `dflash2_long_context` rather than widening it — the existing probe's criterion (acceptance ratio) and exit code are the DFlash2 spec's; a second criterion in the same `run` would fork its behavior.

## Risks / Trade-offs

- [The PR-head image fails to build or export because the fork's FlashInfer patches do not apply on upstream's pin] → fallback to `e6fc2ef6` for the export, stated in the comment; the probe code is the same either way.
- [A package refuses to serve under a newer image's package reader] → every run uses the package's own exporting image (step 1).
- [A 1,000k prompt exceeds the package's KV budget at concurrency 1] → the DFlash2 yarn4 package reported 1,142,784 KV tokens; the MTP yarn4 package is checked on `flare inspect` before the run and the longest length is lowered to fit, recorded as a discovery.
- [The 1M prefill takes longer than the request timeout] → prefill ran about 15k tokens/s on this GPU (about 70 s at 1M); timeout 900 s per request.
- [Passkey fails past 262k on both packages] → that is the frequency-only YaRN finding Learnings §7 names as unmeasured; reported as the grid, with the native baseline row showing the model retrieves below native; the PR's fate is the reviewer's.
- [Passkey digits appear in the catalog by chance] → the catalog carries five-digit runs only as `revision=r1000+i` once `i` reaches 9,000 (none at the 200k length, 6,008 distinct values at the 1,000k length), so `passkey_for` redraws while the candidate occurs in that cell's catalog; the check requires the exact five-digit passkey, and the test asserts the passkey is absent from `_catalog(records)`.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.925
Caller Agreement: 0.9
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — cancelled (quorum met before it responded), agreement n/a
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.95

### Review Feedback

Consensus feedback recorded verbatim:
- "Build the container image at the PR head to satisfy the reviewer's concern regarding the FlashInfer pin, and serve the MTP scaled package from it." — ACCEPTED: Approach step 1 as drafted (the operator deferred it twice on 2026-11-05, then chose the full plan once a GPU-0 window was needed anyway); the fork image stays the recorded fallback only.
- "A unique needle (e.g., GUID or random string) embedded in a synthetic record-line filler at controlled lengths (straddling native 262k up to 1,048k) and depths. Use sampling with `top_k=1` (avoiding `temperature=0` rejections), a max tokens limit, and exact-substring matching for the pass criterion." — ACCEPTED: the seeded five-digit passkey is unique per cell and the test asserts it is absent from the filler; a GUID would cost the passkey form its published definition, so the digits stay.
- "Baseline: Run identical probes against the existing unscaled native (262k) MTP package and the DFlash2 scaled variant." — ACCEPTED for the DFlash2 package (full grid); REJECTED for the native package past 200k — a 262k package cannot take the 320k+ lengths, so its baseline is the 200k row at all three depths.
- "GPU resource contention: Ensure production serving on the 96 GB RTX PRO 6000 is stopped before launching the 1M-context container" — ACCEPTED: every GPU task is an operator step that opens with stopping the console instance; the agent never stops it.
- "Divergence from FlashInfer pin differences between the PR head and fork." / "KV-budget and wall-clock limitations." — ACCEPTED: Risks rows 1–3 as drafted.
- "Performant: ... optimize by caching/reusing KV contexts where feasible." — REJECTED: cells run longest-first on purpose so each prefill is cold; a prefix hit would make a shorter cell's retrieval depend on the longer one's cache, which is not what the reviewer asked about.
- "Correct: Validate preparation hashes, enforce exact string matching for passkey recovery, and track draft-acceptance/Prometheus metrics." — ACCEPTED: step 2 records `flare inspect` (identity and ceiling) in `inspect.json`; exact match and per-cell accepted-per-round are in step 3.
