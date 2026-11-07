# Run the Muse Glimmer DFlash2 parity test in the flare-ignition container

Tier: 1 — no source edit; one test run of an existing file at an operator-opened GPU-0 window
Type: feature

External Review: Waived — Tier 1 (a test run, no code edited)

## Stated Requirements

- "I prefer the z-lab route." (the owner's words carried from glimmer-zlab-dflash2-drafter; the z-lab drafter's parity against the fork's own DFlash2 oracle is the one leg of that change never executed)
- Assumed: the rewritten `tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py` runs once, inside the console-sanctioned `local/flare-ignition:<sha8>` image built from the landed commit, on GPU 0, in a window the operator opens; never on the host venv (the direct loader has no io_uring there) and never on the 16 GiB GPU 1 (the 30B target does not fit).
- Assumed: the operator's rules recorded 2026-11-06 hold — no ignition runs off the old recipe name, no ignition runs outside the console — so the run is scheduled and launched by the operator, not the agent.
- Assumed: a parity failure (relative L2 above 2e-2, or a first proposal that is not the target's greedy token) is recorded as a `discovery:` and fixed in its own change; this change edits no source.

## Blast Radius

1. tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:84 — the `reference` oracle (and `selector_walk` at :126) rewritten by glimmer-zlab-dflash2-drafter task 2.1 and never executed; its one run is the delivery.
2. ignition/recipes/muse_glimmer_nvfp4/dflash.py:27 — `PreparedDraft` reads `output_multiplier`/`final_logit_softcapping` from `dflash_config`; the parity test is the arbiter of that head transform. Read-only here.
3. grepped 'FLARE_MUSE_GLIMMER_DFLASH2' across tests/ — only tests/recipes/muse_glimmer_nvfp4/test_dflash_cuda.py:42 reads it; no further sites.
