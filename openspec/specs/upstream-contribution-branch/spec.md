# upstream-contribution-branch Specification

## Purpose

A fork patch set prepared as a pull-request branch for the upstream repository: cut on the upstream tip, carrying only the patch's own changes, validated on the branch and on the pristine base, with evidence disclosed by the artifact it came from.

## Requirements

### Requirement: The branch carries only the patch

A PR branch SHALL be cut on the recorded `upstream/main` tip and SHALL contain only the hunks of the commits that implement the patch; code, fields, aliases, tests, results, and planning records that exist only on the fork SHALL be absent, and every name the branch imports or every dataclass field it passes SHALL exist on the upstream tip.

#### Scenario: A fork-only field is dropped
- **WHEN** a recipe on the fork passes a `Recipe` field or reads a `Plan` attribute that `git grep` does not find at the upstream tip
- **THEN** the branch copy omits it and `from ignition.recipes import recipes; recipes()` succeeds on the branch

#### Scenario: A mixed-history file takes only the patch's hunks
- **WHEN** `git log 585159c9..deploy -- <file>` names a commit outside the patch
- **THEN** the branch takes that file's hunks from the patch's commits only

### Requirement: Commits satisfy upstream's contribution rules

Every commit on the branch SHALL carry `Signed-off-by: Ray Dollete <ray@raydollete.com>` and one `Design-Signed-Off-By: <reviewer>; design-review=<digest>` trailer whose digest is the first four hex characters of the SHA-256 of the staged upstream `docs/design-review.md`, written after reading that document and upstream's prose skill and after reviewing the commit against them; a rebase onto an upstream tip whose `docs/design-review.md` differs SHALL re-review and re-acknowledge every commit rather than only replace the digest. The commit message body and the PR description SHALL answer, in this order, why the change is needed and why this approach, what exactly changed (before and after), the claimed benefit or result (expected distinguished from measured), and the evidence for each claim with untested checks and limitations stated; every figure SHALL name the base commit or package it was measured on, and the body SHALL carry no process assurances (statements that work was reviewed or is ready).

#### Scenario: Digest matches the staged document
- **WHEN** the commit is created in the branch worktree
- **THEN** `python3 -c` over `git show :docs/design-review.md` prints the digest the trailer carries and the `.githooks/commit-msg` hook accepts the commit

#### Scenario: Rebase across a checklist change
- **WHEN** a PR branch is rebased onto an upstream tip whose `docs/design-review.md` digest differs from the one its trailers carry
- **THEN** every rebased commit carries the new digest, its message body follows the four-part order, and figures measured on the earlier base are labeled with that base

### Requirement: Validation runs on the branch and the pristine base

The patch's CPU test suites SHALL run on the branch and on a pristine checkout of the same upstream tip; the branch SHALL show no failure the base lacks, the patch's own tests SHALL be absent or failing on the base, and tests skipped on gates (environment variables, hardware, fixtures) SHALL be listed as unrun.

#### Scenario: Baseline comparison
- **WHEN** the same pytest and cargo selections run in both worktrees
- **THEN** `comm` of the sorted FAILED/ERROR ids shows nothing only on the branch

### Requirement: Evidence is disclosed by source

The PR body SHALL attribute every GPU result to the package or test run it came from (package id, image, commit) and SHALL state which checks were not run on the branch.

#### Scenario: Fork-package evidence
- **WHEN** served acceptance or retrieval numbers come from a package built from fork commits
- **THEN** the body names that package and says the rebased branch has not been exported or served
