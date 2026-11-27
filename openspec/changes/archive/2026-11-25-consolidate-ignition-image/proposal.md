# Publish the consolidated preparation image as the fork's canonical ghcr image

Tier: 2
Type: feature
Retired: operator kept the light serving image published, because others serve the packages ("Ok then let's keep the light serving image where it is, build the ignition image for local only"); nothing was applied

## Stated Requirements

- "Ok let's simplify this and consolidate the ignition image and make that the canonical one holding the tag in ghcr"
- "Move 0.3.2-r1" — selected over cutting `0.3.2-r2`: the existing `0.3.2-r1` and `sha-de89d0c8…` tags move onto the consolidated image
- "This machine" — selected over GitHub Actions: the image is built and pushed from this workstation

Assumed: the image keeps the name `ghcr.io/raydius/hyperflare-inference`.
Assumed: the 163 MB serving image stops being published; `.github/workflows/container.yml` is removed, and `release.yml` (binary archives) is unchanged.
Assumed: the Dockerfile, build script and FlashInfer patch move from the console repo (`~/LocalApps/vllm-env/build/flare-ignition/`) into the fork under `scripts/fork/image/`, and the console copy is deleted.
Assumed: the image's `ENTRYPOINT` becomes `flare`, matching the published tag's `docker run IMAGE run CORE` contract; `ignition` is reached with `--entrypoint /opt/flare/.venv/bin/ignition`, as the console already does.
Assumed: the republished `0.3.2-r1` is built from tag `v0.3.2-r1` (`de89d0c8`) and recipes `d8cd2098`, and reports `flare 0.3.2-r1`.

## Why

Two images carry the same Flare build: the 163 MB serving image that CI publishes, and the 24.6 GB preparation image built only on this workstation. The workstation exports and serves from the preparation image, so the published tag names an image nobody here runs, and the local image reports `flare 0.0.0` because only CI stamps the release version.

## What Changes

- New `scripts/fork/image/` in the fork: the preparation image's Dockerfile (with the launcher guard from the console change `guard-ignition-launcher`), its FlashInfer patch, and `build.sh`.
- `build.sh <core-ref> <recipes-ref> [--push]`: keeps today's recipe-lock check and `local/flare-ignition:<core8>-<recipes8>` tag. On a release tag it stamps the version with `scripts/set_release_version.py`, also tags `ghcr.io/raydius/hyperflare-inference:<version>` and `:sha-<commit>`, and with `--push` pushes both, pulls the `sha-` tag back, and runs the existing version and fork-flag checks.
- **BREAKING** for `ghcr.io/raydius/hyperflare-inference:0.3.2-r1` pullers: the tag moves from the 163 MB serving image (`sha256:3704cd54…`, still pullable by digest) to the 24.6 GB image with `ignition`, PyTorch, FlashInfer and the recipes.
- `.github/workflows/container.yml` is deleted; the root `Dockerfile` (upstream's serving image) stays, unpublished.
- Docs: `docs/container.md` Publish section, README published-builds line and patch row, `FORK.md` "Build a local image", and `THIRD_PARTY_NOTICES.md` for the bundled preparation stack and NVIDIA container license.
- Console repo: `build/flare-ignition/` is deleted and `docs/operations/flare_runtime.md` points at the fork's script.

## Capabilities

**New Capabilities**

- None.

**Modified Capabilities**

- `release-versioning` — the container image is published by the fork's image script from this workstation instead of a tag-triggered workflow, and the published image is the preparation image.

## Impact

Fork: `scripts/fork/image/{Dockerfile,build.sh,flashinfer-gdn-head-ratio.patch}` (new), `.github/workflows/container.yml` (deleted), `docs/container.md`, `README.md`, `FORK.md`, `THIRD_PARTY_NOTICES.md`, `tests/test_fork_image_build.py` (new).
Console repo: `build/flare-ignition/` (deleted), `docs/operations/flare_runtime.md`.
Registry: `0.3.2-r1` and `sha-de89d0c8fdfc06d48b2594c46cbc2b96e4a3430a` re-pointed; `0.2.5`, which the published model card pins, is untouched.
Recipes fork: `deploy` (`d8cd2098`, unpushed) is pushed so the image's recipe label names a public commit.

## Blast Radius

1. `.github/workflows/container.yml:5-6` — triggers on every `v*` / `v*-r*` tag and pushes the serving image; left in place it would overwrite the moved tag on the next release.
2. `.github/workflows/container.yml:65-86` — the version and fork-flag checks; carried into `build.sh` unchanged.
3. grepped 'locks ignition at' in the console repo's `build/flare-ignition/build.sh` — hit at its line 26, the recipe-lock check; carried.
4. grepped '^ENTRYPOINT' in the console repo's `build/flare-ignition/Dockerfile` — hit at its line 67, `ENTRYPOINT ["/opt/flare/.venv/bin/ignition"]`, after the guarded launcher at line 61; the entry point changes to `flare`.
5. grepped '_ENTRYPOINTS' in the console repo's `console/flare_params.py` — hit at its line 529, which passes `/usr/local/bin/flare` and `/opt/flare/.venv/bin/ignition` explicitly, so the console is unaffected by the entry point.
6. grepped 'FLARE_IMAGE_REPOS =' in the console repo's `console/flare_packages.py` — hit at its line 65, which matches `local/flare-ignition`; that tag is kept, so cache commands keep finding the image.
7. `scripts/set_release_version.py:16-17` — accepts `X.Y.Z` and `-r<N>`; reused, not edited.
8. `docs/container.md:36-62` — fork-added Publish section describes the workflow; rewritten for the script.
9. `README.md:18` and `README.md:38` — published-builds pointer and patch-inventory row name the workflow and the serving image.
10. `FORK.md:218-234` — "Build a local image" points at the console's `build.sh`.
11. `THIRD_PARTY_NOTICES.md:5-10` and `:58-67` — state that preparation dependencies and container images are not distributed; a public preparation image distributes them, and `/NGC-DL-CONTAINER-LICENSE` §1(c) and §2 set the derived-container terms.
12. grepped 'build/flare-ignition/build.sh' in the console repo's `docs/operations/flare_runtime.md` — hit at its line 320, the image build instructions.
13. `openspec/specs/release-versioning/spec.md:12` and `:58` — requirements "Accepted release tag forms" and "Published version must equal the tag" name the container workflow.
14. HF card `raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M` README.md:48 — pins `:0.2.5`, not `0.3.2-r1`; unaffected.
15. grepped 'hyperflare-inference' and '3704cd54' across the fork outside `openspec/changes/archive/` — only the sites above.
