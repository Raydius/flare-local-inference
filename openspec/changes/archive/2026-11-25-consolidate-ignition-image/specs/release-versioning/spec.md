## MODIFIED Requirements

### Requirement: Accepted release tag forms
A tag push SHALL start the release workflow, and the image script SHALL publish an image,
only when the tag matches `vMAJOR.MINOR.PATCH` or `vMAJOR.MINOR.PATCH-r<DIGITS>`, where each
version component is a non-negative decimal integer without leading zeros and `<DIGITS>` is
one or more decimal digits with no leading zero. No other ref SHALL publish a release: a ref
outside both forms starts no release run, the leading-zero form the trigger glob admits
anyway (`v0.3.2-r01`) is refused by the release workflow's guard step before a version is
resolved, and the image script refuses to push any core ref that is not an accepted tag.

#### Scenario: Bare release tag publishes
- **WHEN** a tag `v0.3.4` on the default branch is pushed and the image script is run with
  `--push` on that tag
- **THEN** the release workflow runs for that commit and the script publishes the image

#### Scenario: Revision tag publishes
- **WHEN** a tag `v0.3.2-r1` on the default branch is pushed and the image script is run with
  `--push` on that tag
- **THEN** the release workflow runs for that commit and the script publishes the image

#### Scenario: Unaccepted ref is ignored
- **WHEN** a tag in any other form (`v0.3.2-R1`, `v0.3.2-rc1`, `v0.3.2-r`, `v0.3.2-r1-2`,
  `v0.3`) or a branch is given to the image script with `--push`
- **THEN** the script exits non-zero naming the ref, before building, and pushes nothing;
  and a pushed tag outside both forms starts no release run

### Requirement: One version string per release
The release version SHALL be the tag name with its leading `v` removed, and every published
artifact of that release SHALL carry that exact string: the container image tag, the Cargo
package version stamped into the release crates, the string `flare --version` reports, and
the GitHub release name, which uses the tag including its leading `v`. The commit tag
`sha-<full-commit-sha>` SHALL be published beside the version tag on the same image.

#### Scenario: Revision release artifacts agree
- **WHEN** `v0.3.2-r1` is published
- **THEN** the image is tagged `0.3.2-r1` and `sha-<commit>`, the stamped Cargo version is
  `0.3.2-r1`, and the GitHub release is named `v0.3.2-r1`

#### Scenario: Bare release is unchanged
- **WHEN** `v0.3.4` is published
- **THEN** the image is tagged `0.3.4` and `sha-<commit>` and the stamped Cargo version is
  `0.3.4`

#### Scenario: Manual dispatch stays unstamped
- **WHEN** the release workflow is started manually on a branch, or the image script builds
  a branch without `--push`
- **THEN** the build keeps version `0.0.0`; the workflow creates no GitHub release and the
  script tags only `local/flare-ignition:<core8>-<recipes8>`

### Requirement: Published version must equal the tag
A published artifact whose reported version differs from the tag it was built from SHALL
fail the publish. After pushing, the image script SHALL pull the published `sha-<commit>`
tag rather than reuse the local build, require `flare --version` to print
`flare <version>`, and require `flare run --help` to list `--generation-defaults`,
`--max-context-tokens`, and `--chat-template` (carried from
`.github/workflows/container.yml:65-86`). The binary archive check SHALL require the same
`--version` string from an unpacked archive in a clean container.

#### Scenario: Revision release reports its revision
- **WHEN** the image published from `v0.3.2-r1` is pulled and run with `--version`
- **THEN** it prints `flare 0.3.2-r1`

#### Scenario: Disagreement fails the run
- **WHEN** the pulled image prints a version other than the release's version string
- **THEN** the script exits non-zero naming both strings

## ADDED Requirements

### Requirement: Published image carries the preparation tools
The published image SHALL be the preparation image built by `scripts/fork/image/build.sh`:
it contains `/usr/local/bin/flare`, the `ignition` command at
`/opt/flare/.venv/bin/ignition` with the recipe fork's recipes registered, and the
`/NGC-DL-CONTAINER-LICENSE` of its NVIDIA CUDA base image. Its entry point SHALL be
`flare`. It SHALL be built from the tagged core commit and a recipe-fork commit whose
`uv.lock` pins `ignition` at that exact core commit, and labeled with both commits. The
script SHALL publish only a core commit that is an ancestor of the remote default branch.

#### Scenario: Image serves and prepares
- **WHEN** the published image is run with `run --help`, and with
  `--entrypoint /opt/flare/.venv/bin/ignition` and `export --help`
- **THEN** both print their usage

#### Scenario: Mismatched recipe lock is refused
- **WHEN** the recipe ref's `uv.lock` pins `ignition` at a commit other than the core ref
- **THEN** the script exits non-zero naming both commits, before building

#### Scenario: Unmerged commit is refused
- **WHEN** `--push` is given for a tag whose commit is not an ancestor of the remote
  default branch
- **THEN** the script exits non-zero before building and pushes nothing
