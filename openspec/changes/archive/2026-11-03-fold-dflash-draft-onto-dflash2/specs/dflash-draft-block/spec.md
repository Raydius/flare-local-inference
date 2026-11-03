## REMOVED Requirements

### Requirement: The block is a component with recipe-neutral inputs
**Reason**: `ignition/components/dflash_draft.py` is deleted; upstream's `ignition/components/dflash2.py` is the one shared DFlash2 block and Nemotron returns to its own self-contained module (`git show upstream/main:ignition/recipes/nemotron_h/dflash.py`).
**Migration**: Qwen3.8-Flash-Next composes the shared component through the recipe adapter specified in `qwen38-dflash2-speculation`; Nemotron's LiLiCorr draft keeps its upstream implementation and its CUDA test bounds.

### Requirement: Context writes respect liveness and recency
**Reason**: The slot rule (`row × window + position mod window`, dead tokens at −1) now lives in the shared component's context writer (`ignition/components/dflash2.py:85`–`:88`); the recipe-level guarantee moves to `qwen38-dflash2-speculation`.
**Migration**: See `qwen38-dflash2-speculation` — "Context writes respect liveness and recency".

### Requirement: The prepared block reproduces the DFlash2 reference under its storage contract
**Reason**: The parity contract is a property of the Flash-Next recipe on the shared component, not of a separate module.
**Migration**: See `qwen38-dflash2-speculation` — "The prepared draft reproduces the DFlash2 reference under its storage contract"; the recorded bounds and the shortlist/walk equality are carried unchanged.
