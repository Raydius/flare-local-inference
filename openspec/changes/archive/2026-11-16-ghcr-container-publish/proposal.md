# Publish the serving container to GHCR on version tags

Tier: 1 — two files (one new workflow, one doc section), no runtime code or behavior fork
Type: feature

External Review: Waived — Tier 1 (one new workflow plus one documentation section; no runtime code)

## Stated Requirements

- "Let’s create a tag triggered ghcr flow"
- "I would do ghcr"
- "I only provided JohnB’s upload as a format example. I would need to package the whole runtime including the relevant patchwork"

## Blast Radius

1. `.github/workflows/container.yml` (new) — the tag-triggered workflow this change adds; shape copied from `.github/workflows/release.yml:3-6` (trigger) and `.github/workflows/release.yml:23-25` (default-branch gate).
2. `docs/container.md:17-28` — the local-only `docker build -t flare:local .` section; gains the GHCR publish instructions and the digest-pinning note.
3. grepped 'ghcr|buildx|docker/build-push|packages: write' across `.github/workflows/`, `docs/container.md`, `docs/release.md` — no image-publishing workflow exists; the GHCR push path is new.
4. grepped 'set_release_version' across `.github/workflows/` — `.github/workflows/release.yml:64` stamps before `cargo build --locked`; the new workflow stamps the same script so `flare --version` reports the tag instead of the tree's `0.0.0`.
