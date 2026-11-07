# Rename the Muse Glimmer recipe to muse_glimmer_nvfp4 so the recipe name states its NVFP4 design

Tier: 2 — mechanical rename of a recipe name, two identity strings and the import paths that follow a directory move; no behavior fork, every site is a string literal or import path
Type: feature

## Stated Requirements

- "Let's make sure the naming convention of our recipe makes it clear that it is designed for NVFP4 and not mixed precision."
- "what if the primary name is simply muse_glimmer_nvfp4 ?"
- "we can keep the alias"
- "Since this is not a functional difference, can we bundle this with the rope/yarn change?"
- "or dflash2.  take your pick."
- Assumed: the recipe name becomes `muse_glimmer_nvfp4`; the alias `muse-glimmer-30b-nvfp4` and the served model name `Muse-Glimmer-30B` stay.
- Assumed: the bundle partner is the next fork rebase onto upstream (the rope/yarn work's Step B: rebase and re-check the export under format-4), proposed later on this same branch; this change applies, verifies and archives CPU-only, is not closed on its own, and the console profile update, the one re-export and the vault edits ride in that change's single GPU-0 window.
- Assumed: the package directory moves to `ignition/recipes/muse_glimmer_nvfp4/` (and the test package to `tests/recipes/muse_glimmer_nvfp4/`) so the upstream mixed-precision recipe's files add cleanly at the next rebase instead of colliding file by file.
- Assumed: the identity contract becomes `muse-glimmer-nvfp4.text.v1` and the workload schema `muse_glimmer_nvfp4`, so prepared packages re-key; the console profile update and the one re-export belong to the bundle change's GPU window, not to this change.
- Assumed: the Helios image profile, the proto enum, the `image_settings` key `muse_glimmer` in `ignition/preparation/frontend.py`, the reference module `ignition/reference/muse_glimmer.py` and the main spec capability `muse-glimmer-serving` stay (model-level, precision-independent); the delta MODIFIES the two requirements that quote the recipe name.

## Blast Radius

1. ignition/recipes/muse_glimmer/recipe.py:55 — `Recipe(name="muse_glimmer"` becomes `muse_glimmer_nvfp4`; the alias on ignition/recipes/muse_glimmer/recipe.py:59 stays.
2. ignition/recipes/muse_glimmer/bootstrap.py:174 — identity `contract="muse-glimmer.text.v1"` becomes `muse-glimmer-nvfp4.text.v1` (re-keys prepared packages).
3. ignition/recipes/muse_glimmer/bootstrap.py:179 — `workload_schema="muse_glimmer"` becomes `muse_glimmer_nvfp4`.
4. ignition/recipes/muse_glimmer/__init__.py:4 — the package (15 files) moves to `ignition/recipes/muse_glimmer_nvfp4/` with `git mv`; grepped '^(from|import) ignition' across ignition/recipes/muse_glimmer/ — every absolute import targets `ignition.components.*`, `ignition.preparation.*` or `ignition.recipes.muse_spark.*`, none its own package, so file contents are unchanged by the move.
5. ignition/recipes/__init__.py:40 — `from .muse_glimmer.recipe import RECIPE as muse_glimmer` follows the directory.
6. ignition/reference/muse_glimmer.py:32 — `from ignition.recipes.muse_glimmer.config import PREFIX, PROJECTIONS` follows the directory; the module itself stays.
7. tests/recipes/muse_glimmer/test_config.py:14 — the test package (7 files) moves to `tests/recipes/muse_glimmer_nvfp4/`; its `ignition.recipes.muse_glimmer.*` imports (tests/recipes/muse_glimmer/test_preparation.py:9, tests/recipes/muse_glimmer/test_vision_cuda.py:27, tests/recipes/muse_glimmer/test_contract.py:15) and `tests.recipes.muse_glimmer.test_reference_cuda` imports (tests/recipes/muse_glimmer/test_dflash_cuda.py:33, tests/recipes/muse_glimmer/test_vision_cuda.py:30) follow.
8. tests/recipes/muse_glimmer/test_preparation.py:69 — monkeypatched module name `ignition.recipes.muse_glimmer.bootstrap`.
9. tests/cli/test_draft_checkpoints.py:18,73,83,107,131,132 — recipe name `muse_glimmer` in parametrizations and the capacity map.
10. tests/preparation/test_frontend_metadata.py:25,194,197 — recipe name in the sampling tuple, the profile map and the preprocessor map (the profile value `muse_glimmer` stays).
11. tests/recipes/test_vision_scratch.py:32 — module path `recipes.muse_glimmer.vision`.
12. docs/model-preparation.md:1143 — asset path `ignition/recipes/muse_glimmer/assets/chat_template.patch.json`; the alias on docs/model-preparation.md:1098,1148 stays.
13. openspec/specs/muse-glimmer-serving/spec.md:5,80 — Purpose and the "Declared capacities and costs match the prepared layout" scenario name the `muse_glimmer` recipe; the alias on openspec/specs/muse-glimmer-serving/spec.md:10,13 stays.
14. grepped 'muse_glimmer|muse-glimmer' across ignition/ helios/ tests/ docs/ pyproject.toml — every other site is the model-level image profile, proto enum, `ignition/preparation/frontend.py:117-130` profile key, `tests/frontend/test_muse_glimmer.py`, `tests/recipes/test_muse_spark_attention.py:189-190` test identities, `tests/cli/test_models.py:25` alias tuple, `tests/recipes/muse_glimmer/test_config.py:82` alias assertion and the transformers import at `ignition/recipes/muse_glimmer/vision.py:21`; all stay.
15. grepped 'muse_glimmer' in pyproject.toml — no package-data line exists today, so none moves.
16. grepped 'recipe muse_glimmer' in /tmp/glimmer-profile-put.json — the console profile `2026-11-06-flare-muse-glimmer-1793992412838` carries `"export:--recipe": "muse_glimmer"` and `docker_image` tag `37f3613d`; both are updated after landing (outside the repo).
17. grepped 'muse_glimmer' across the vault folder `Flare/` — five notes (`README.md`, `Flare Developer Learnings.md`, `Muse Glimmer/README.md`, `Muse Glimmer/Muse Glimmer Recipe Progress.md`, the DFlash troubleshooting note) name the old recipe name and are updated after landing (outside the repo).
