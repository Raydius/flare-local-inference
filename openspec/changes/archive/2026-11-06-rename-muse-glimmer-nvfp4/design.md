# Design — rename-muse-glimmer-nvfp4

## Approach

- `git mv ignition/recipes/muse_glimmer ignition/recipes/muse_glimmer_nvfp4` and `git mv tests/recipes/muse_glimmer tests/recipes/muse_glimmer_nvfp4`: the move keeps history and lets upstream PR #192's `ignition/recipes/muse_glimmer/*` files add cleanly at the next rebase.
- In the moved recipe.py set `Recipe(name="muse_glimmer_nvfp4")`; the alias `muse-glimmer-30b-nvfp4`, the served model name and every other declaration field stay byte-identical.
- In the moved bootstrap.py set `contract="muse-glimmer-nvfp4.text.v1"` and `workload_schema="muse_glimmer_nvfp4"`: the package identity re-keys once, so the old cache entry is orphaned (disk only) and one re-export follows landing.
- Follow the directory in the five import sites (`ignition/recipes/__init__.py`, `ignition/reference/muse_glimmer.py`, the moved tests) and the string-keyed sites (`tests/cli/test_draft_checkpoints.py`, `tests/preparation/test_frontend_metadata.py`, `tests/recipes/test_vision_scratch.py`, the monkeypatched module name in the moved test_preparation.py).
- Docs: one asset path in `docs/model-preparation.md`; spec: the delta's two MODIFIED requirements, synced by rad-sync, with the Purpose line corrected by hand at sync since it names the recipe.
- Stays: the Helios `muse_glimmer` image profile and proto enum, the `image_settings` key in `ignition/preparation/frontend.py`, the reference module name, `tests/frontend/test_muse_glimmer.py`, the Spark attention test identities (model-level, precision-independent).
- Bundled landing: this change ends at its CPU-only archive on branch `wt/glimmer-nvfp4-rename`; the next fork rebase onto upstream is proposed on the same branch and its GPU-0 window builds `local/flare-ignition:<new sha8>`, sets the console profile's `export:--recipe` to `muse_glimmer_nvfp4` and its `docker_image` to the new tag, relaunches once (the export re-keys) and updates the five vault notes; one close lands both.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.95
Caller Agreement: 0.95
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.95
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.95
- local-ai-qllm (qLLM) — cancelled (quorum met first), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Perform a directory move using `git mv` … and string edits to avoid add/add conflicts during rebase. Rename the `Recipe(name=...)`, alias, identity contract string …, and `workload_schema` to re-key cached prepared packages and prevent identity collision with upstream." — ACCEPTED for the directory, name, contract and workload schema; the alias rename was overturned after the review by the operator ("we can keep the alias"), so the alias collision with upstream PR #192 is accepted and recorded in the proposal's Stated Requirements.
- "ensure coverage of packaging metadata, CI matrices, test parametrizations, alias tuples, frontend metadata maps, vision scratch paths, documentation, and the main spec delta. Downstream console profiles, Docker image tags, and vault notes must also be updated." — ACCEPTED: Blast Radius 15 records that pyproject.toml carries no package-data line for this recipe and grepped CI finds none; the remaining classes are Blast Radius 7-17.
- "silent wrong-model resolution if upstream lands and an un-updated console profile matches the new upstream recipe" — ACCEPTED: the console profile's recipe field is updated in the bundle change's GPU window before any rebased image is launched, and because the alias is kept, the recipe field is the only console input that would otherwise match upstream's recipe after a rebase.
- "orphaned caches cleaned up to save disk space" — REJECTED: the old cache entry under `~/.cache/flare-ignition-glimmer-ctx` is removed by the operator's disk hygiene, not by this change; it is outside the repo and the root disk cleanup is already the operator's call.
