# dflash-draft-block Specification

## Purpose

A shared sliding-window DFlash draft block — context projection into per-row key/value rings, noncausal block attention over the ring, dynamic grouped convolutions, and MLP — that any recipe storing draft windows in its state row composes with its own selector and verifier.

## Requirements

### Requirement: The block is a component with recipe-neutral inputs
`ignition/components/dflash_draft.py` SHALL provide the context writer, draft attention, draft MLP, block forward, scratch, and rotary used by Nemotron's LiLiCorr draft today, taking rings as tensors, per-token positions/rows/live masks for the context, an optional attention sink, and projections chosen by weight representation (BF16 tensor or quantized weight), with the selector left to the calling recipe.

#### Scenario: Nemotron unchanged
- **WHEN** `tests/recipes/nemotron_h/test_dflash_cuda.py` runs against the LiLiCorr checkpoint after the lift
- **THEN** every case passes with the bounds it has today and `nemotron_h/dflash.py` only adapts paged metadata and appends the lattice selector

#### Scenario: BF16 draft
- **WHEN** a recipe passes BF16 projection weights
- **THEN** the block binds unquantized projections without activation quantization

### Requirement: Context writes respect liveness and recency
The context writer SHALL write a token's key/value only when the caller marks it live, at ring slot `row × window + position mod window`, and SHALL leave dead tokens at slot −1 so a dead token's row index never selects another request's ring.

#### Scenario: Dead tokens
- **WHEN** a batch holds inactive rows whose request ids are −1
- **THEN** no ring slot of any live row is written by those tokens

### Requirement: The prepared block reproduces the DFlash2 reference under its storage contract
For the pinned Qwen3.8-Flash-Next-DFlash2 draft, one block through the component with key/value round-tripped through an E4M3 ring, followed by the candidate selector, SHALL match `ignition/reference/qwen38_dflash2.py`'s forward, candidates, and greedy walk with the reference's key/value rounded to E4M3 the same way (the storage contract the acceptance measurement validated: E4M3 and BF16 key/value give identical code_edit acceptance on real tapped features), within bounds recorded from the measured repeatability of that rounded reference (its own movement under one-ulp input noise and an FP32-projection variant), with the top-16 shortlist and the walked tokens — the outputs that decide acceptance — equal on real features; the comparison runs on real tapped features of the target when a trace is available and on fabricated features otherwise.

#### Scenario: Parity on real tapped features
- **WHEN** the CUDA test runs on GPU 1 with the pinned draft and a trace of the target's tapped features
- **THEN** the block's hidden states, logits, and rings match the E4M3-rounded reference within the recorded per-output bounds, the top-16 shortlist and walked tokens equal the reference's on every row, and ties resolve to the lowest token id in both

#### Scenario: Fabricated features
- **WHEN** no trace is available
- **THEN** the test runs on random features against the same E4M3-rounded reference and records that bound separately, skipping no assertion silently
