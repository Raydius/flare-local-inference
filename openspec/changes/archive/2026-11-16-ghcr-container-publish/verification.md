# Verification — ghcr-container-publish

## Verdict: PASS

Tier 1 inline pass. `radflow-lint --tier-check` is clean: the change still touches one new workflow and one doc section, plus its own record. Every stated requirement is delivered: a tag-triggered workflow (`push` on `v[0-9]+.[0-9]+.[0-9]+`) publishes to GHCR, and the image is built from this fork's Dockerfile and tree, so it carries the fork's runtime changes. The pull-back check requires the three fork flags and passed in CI on v0.2.5 (run 43935254426). The behavior changes made after apply each trace to a recorded line: the operator's rename to `hyperflare-inference` (`correction:`), the ref-type tag handling and the Buildx setup step (`discovery: … — fixed`). Task 4, added before close at the operator's request, puts the published image and core in `README.md`: a pointer under the maintenance-workflow line and an Active patches entry beside `patch/00-fork-tooling`; the stated requirement quoting that request is in the proposal.

## Evidence audited
- [1. Create `.github/workflows/container.yml`: trigger `push: tags: ['v[] — .github/workflows/container.yml:1 — "name: Container"
- [2. Add a `## Publish` section to `docs/container.md` between `## Build] — docs/container.md:36 — "## Publish"
- [3. First real publish outside the development clone: tag the fork stac] — .github/workflows/container.yml:1 — "name: Container"
- [4. Add to `README.md` a published-builds pointer under the maintenance] — README.md:10 — "Published builds: the serving image `ghcr.io/raydius/hyperflare-inference:<version>`, built from this branch by `.github/workflows/container.yml` on each `vX.Y.Z` tag ([docs/container.md](docs/container.md#publish)), and the prepared Qwen 3.8 core [`raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M`](https://huggingface.co/raydollete/Qwen3.8-Flash-Next-HyperFlare-A16-Abliterated-1M), which runs on that image."
