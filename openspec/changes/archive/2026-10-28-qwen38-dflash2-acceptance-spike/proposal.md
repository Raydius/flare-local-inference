# Measure DFlash2 against MTP acceptance on the Qwen3.8-Flash-Next target

Tier: 2
Type: feature

## Stated Requirements

- "What would be necessary to support DFlash2 as a speculative mechanism?"
- "Yes, the draft checkpoint exists here: https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2"
- "I already downlaolded it in ~/models/"
- "I agree with this being a separate recipe."
- "Acceptance spike" — selected over "Full recipe": "PyTorch reference of DFlash2 vs MTP acceptance length on Flare's target at widths 4 and 8; no graphs, no recipe. Decides go/no-go cheaply."

## Why

A DFlash2 recipe for Qwen3.8-Flash-Next is Tier 3 work (feature taps in the overlapped decoder, a new state-row layout, a new speculative round). The only published evidence that it beats the checkpoint's MTP is the model card's SGLang numbers at three drafts; Flare's MTP already verifies up to eight tokens with history drafts. A cheap, graph-free acceptance measurement on Flare's own target decides whether the recipe is worth building.

## What Changes

- New PyTorch reference of the `DFlash2DraftModel` draft (`fc`, `hidden_norm`, five noncausal sliding-window layers with two-tap grouped convolutions, final norm, target `lm_head` candidates, predecessor-conditioned selector, greedy lattice walk) loaded from the pinned `incoai/Qwen3.8-Flash-Next-DFlash2` safetensors.
- New `qualify` tool that records greedy target continuations with the completed-layer features DFlash2 was trained on (layers 5, 15, 25, 35, 45, mean of the four hyperconnection streams) through the existing canonical reference, then measures teacher-forced acceptance length for DFlash2 and for the checkpoint's MTP from the same traces, reporting widths 4 and 8.
- CPU tests on fabricated weights for the draft forward's shapes, block masking, and greedy walk; a CUDA test gated by `FLARE_RUN_CUDA_TESTS` comparing the reference convolution to the Triton component and running the tool end to end on one short trace.
- A `docs/TODO.txt` entry pointing from the recipe decision to the recorded measurement.
- No recipe, no captured graph, no package or contract change, no edit to `ignition/recipes/qwen38/`.

## Capabilities

### New Capabilities

- `dflash2-acceptance-measurement`: teacher-forced acceptance-length measurement of a DFlash2 draft against the Qwen3.8-Flash-Next target, with the MTP baseline measured the same way.

### Modified Capabilities

- none — the main `openspec/specs/` store is empty; no existing requirement changes.

## Impact

- New files only: `ignition/reference/qwen38_dflash2.py`, `qualify/dflash2_acceptance.py`, `tests/reference/test_qwen38_dflash2.py`; one line in `docs/TODO.txt`.
- Runs on one SM120/SM121 GPU with the NVFP4 target at `~/models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` and the BF16 draft at `~/models/incoai/Qwen3.8-Flash-Next-DFlash2` (revision `a22a1744`, 1,206,131,200 bytes). Canonical per-token decode is slow; prompt and token counts are tool arguments.
- Greedy teacher-forced acceptance is a proxy for the stochastic acceptance Flare serves; both drafters are measured identically, so the comparison is relative. The model card's numbers use sampling and are not directly comparable.

## Blast Radius

1. `ignition/reference/qwen38_decoder.py:348` — `_layer_hook` publishes each completed decoder layer's four-stream residual; the tool consumes layers 5, 15, 25, 35, 45 from it.
2. `tests/reference/test_qwen38_decoder.py:196` — existing producer of `layer_outputs`/`teacher_tokens` traces through `CanonicalTextReference`; the tool follows this shape.
3. `qualify/mtp.py:186` — teacher-forced `PreparedMTP` execution from stored residuals; the MTP arm reuses its setup.
4. `ignition/recipes/qwen38/speculative_model.py:233` — recursive MTP proposals feed `draft.scratch.residual` back as the next row's hidden; the MTP arm reproduces this recursion for seven drafts.
5. `ignition/components/grouped_convolution.py:1` — Triton grouped convolution the CUDA test compares the reference against.
6. `ignition/components/candidate_selector.py:19` — predecessor-conditioned correction over a top-k shortlist; the reference's greedy walk must agree with it per position.
7. `tests/recipes/nemotron_h/test_dflash_cuda.py:31` — `convolve` reference for two-tap block convolution; reused form.
8. grepped '_score_edges\|_follow_maps\|selector_walk' across `ignition/ qualify/ tests/` — no sites; the published DFlash2 selector semantics live only in the external SGLang clone (`~/dev/sglang`, `python/sglang/srt/models/dflash.py`), which the reference reproduces.
9. grepped 'DFlash2DraftModel\|selector_rank\|successor_codebook' across `ignition/ tests/ flare/ helios/` — no sites; Kimi's DFlash2 integration was removed in 73a5611a.
10. grepped 'layer_outputs=' across `ignition/ qualify/` — no caller sets the prepared decoder's tap; the spike uses the canonical reference instead and leaves `ignition/recipes/qwen38/decoder.py:378` untouched.
