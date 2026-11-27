# release-versioning Specification

## Purpose

Defines how the fork names its releases: the git tag forms that publish, the version every artifact reports, and how the core fork and the recipe fork are paired at one tag.

## Requirements

### Requirement: Accepted release tag forms
A tag push SHALL publish a release only when the tag is `vMAJOR.MINOR.PATCH` or `vMAJOR.MINOR.PATCH-rN`, where each number is a decimal integer without leading zeros. `vMAJOR.MINOR.PATCH` SHALL equal the newest upstream release tag that is an ancestor of the upstream commit the fork is based on, and `N` counts fork revisions under that tag from 1. When the base is past that tag, `FORK.md` SHALL record the base commit of each fork beside the release name. The release and container workflows SHALL refuse any other tag name before resolving a version.

#### Scenario: Revision tag publishes
- **WHEN** tag `v0.3.2-r2` is pushed on a commit of the default branch
- **THEN** the container workflow and the release workflow run for that commit

#### Scenario: The base is past the newest upstream tag
- **WHEN** the forks sit on an upstream commit 27 commits past `v0.3.2` and upstream has published no newer tag
- **THEN** the release is named `v0.3.2-r2` and `FORK.md` names the core and recipes base commits

#### Scenario: Malformed tag is refused
- **WHEN** a tag named `v0.3.2-r02`, `v0.3.2-rc1` or `v0.3.2-r` is pushed
- **THEN** no version-tagged image and no release is published

### Requirement: One version string per release
The release version SHALL be the tag name without its leading `v`. The serving image tag, the Cargo version stamped into the release crates, and the string `flare --version` prints SHALL all equal it. The workflow SHALL fail when the published image reports another version.

#### Scenario: Artifacts agree
- **WHEN** `v0.3.2-r2` is published
- **THEN** the image is `ghcr.io/raydius/hyperflare-inference:0.3.2-r2` and `flare --version` inside it prints `0.3.2-r2`

#### Scenario: Version stamping refuses other strings
- **WHEN** `scripts/set_release_version.py` is given `0.3.2-rc1` or `0.3.2-r02`
- **THEN** it exits non-zero and writes no manifest

### Requirement: The two forks are paired at one tag
A fork release SHALL carry the same tag name in the core fork and the recipe fork. The recipe fork's tagged commit SHALL lock `ignition` at the core fork's tagged commit. The local preparation image for the release SHALL be built from those two commits, SHALL be tagged `local/flare-ignition:<version>`, and SHALL record both commits in its labels.

#### Scenario: Aligned tags
- **WHEN** `v0.3.2-r2` exists in both forks
- **THEN** the `ignition` entry of the recipe fork's `uv.lock` at that tag names the commit `v0.3.2-r2` points to in the core fork

#### Scenario: Mismatched lock is refused
- **WHEN** the preparation image is built from a recipe commit whose lock names another core commit
- **THEN** the build stops before any image layer is built and names both commits
