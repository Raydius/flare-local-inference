## Purpose

Defines how this fork names and publishes its own releases: which git tag forms start a
release, how the single version string is derived from the tag, and what every published
artifact — container image, binary archive, GitHub release, and the `flare --version`
output — must report so a consumer can identify the exact patch set it obtained.

## ADDED Requirements

### Requirement: Accepted release tag forms
A tag push SHALL start the release and container workflows only when the tag matches
`vMAJOR.MINOR.PATCH` or `vMAJOR.MINOR.PATCH-r<DIGITS>`, where each version component is a
non-negative decimal integer without leading zeros and `<DIGITS>` is one or more decimal
digits with no leading zero. No other ref SHALL start either workflow.

#### Scenario: Bare release tag publishes
- **WHEN** a tag `v0.3.4` is pushed to the repository's default branch
- **THEN** both the release workflow and the container workflow run for that commit

#### Scenario: Revision tag publishes
- **WHEN** a tag `v0.3.2-r1` is pushed to the repository's default branch
- **THEN** both the release workflow and the container workflow run for that commit

#### Scenario: Unaccepted ref is ignored
- **WHEN** a tag in any other form is pushed (`v0.3.2-R1`, `v0.3.2-rc1`, `v0.3.2-r`,
  `v0.3.2-r01`, `v0.3.2-r1-2`, or `v0.3`)
- **THEN** neither workflow runs and the push reports no release

### Requirement: One version string per release
The release version SHALL be the tag name with its leading `v` removed, and every published
artifact of that run SHALL carry that exact string: the container image tag, the Cargo
package version stamped into the release crates, the string `flare --version` reports, and
the GitHub release name, which uses the tag including its leading `v`. The commit tags
`sha-<full-commit-sha>` SHALL be published beside the version tag and carry the same
version string.

#### Scenario: Revision release artifacts agree
- **WHEN** `v0.3.2-r1` is published
- **THEN** the image is tagged `0.3.2-r1` and `sha-<commit>`, the stamped Cargo version is
  `0.3.2-r1`, and the GitHub release is named `v0.3.2-r1`

#### Scenario: Bare release is unchanged
- **WHEN** `v0.3.4` is published
- **THEN** the image is tagged `0.3.4` and `sha-<commit>` and the stamped Cargo version is
  `0.3.4`, as before this capability existed

#### Scenario: Manual dispatch stays unstamped
- **WHEN** either workflow is started manually on a branch
- **THEN** it builds with version `0.0.0` and publishes only the `sha-<commit>` image tag,
  and no version tag or GitHub release is created

### Requirement: Published version must equal the tag
A published artifact whose reported version differs from the tag it was built from SHALL
fail the run. The container workflow SHALL pull the published `sha-<commit>` tag rather than
reuse the local build, require `flare --version` to print `flare <version>`, and require
`flare run --help` to list `--generation-defaults`, `--max-context-tokens`, and
`--chat-template`. The binary archive check SHALL require the same `--version` string from
an unpacked archive in a clean container.

#### Scenario: Revision release reports its revision
- **WHEN** the image published from `v0.3.2-r1` is pulled and run with `--version`
- **THEN** it prints `flare 0.3.2-r1`

#### Scenario: Disagreement fails the run
- **WHEN** the pulled image prints a version other than the run's version string
- **THEN** the check step fails and the run is reported as failed

### Requirement: Version stamping accepts both forms
The version-stamping step SHALL accept exactly the two accepted version strings, write the
given string to each release crate's manifest and update its lockfile entry, and reject any
other string with a usage error that names the accepted forms. Checking an unstamped tree
SHALL succeed without writing files.

#### Scenario: Revision string is stamped
- **WHEN** the stamping step is given `0.3.2-r1`
- **THEN** each release crate reports version `0.3.2-r1` in its manifest and lockfile

#### Scenario: Malformed string is refused
- **WHEN** the stamping step is given `0.3.2-r01`, `0.3.2-1`, `0.3.2+R1`, or `release`
- **THEN** it exits with a usage error naming the accepted forms and writes no file

#### Scenario: Unstamped tree checks clean
- **WHEN** the stamping step is run in check mode with `0.0.0` on a tree that has not been
  stamped
- **THEN** it succeeds and modifies no manifest or lockfile

### Requirement: Fork releases occupy their own tag namespace
A fork release tag SHALL NOT reuse the name of a tag in the upstream repository. Where the
release's base version equals an upstream tag, the fork SHALL publish under the revision
form `vMAJOR.MINOR.PATCH-r<DIGITS>` so that the two namespaces remain distinguishable.

#### Scenario: Base version already used upstream
- **WHEN** the fork releases work whose upstream base is the commit that upstream tags
  `v0.3.2`
- **THEN** the fork publishes `v0.3.2-r1` and no fork tag named `v0.3.2` is created

#### Scenario: Mirror fetch does not overwrite a fork tag
- **WHEN** the upstream mirror fetch reports that a fetched tag would clobber a fork tag of
  the same name
- **THEN** the fetch keeps the fork's tag and reports the collision as a warning instead of
  failing the sync (carried behavior, `scripts/fork/sync.sh:30-34`)
