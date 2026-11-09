## MODIFIED Requirements

### Requirement: Commits satisfy upstream's contribution rules

Every commit on the branch SHALL carry `Signed-off-by: Ray Dollete <ray@raydollete.com>` and one `Design-Signed-Off-By: <reviewer>; design-review=<digest>` trailer whose digest is the first four hex characters of the SHA-256 of the staged upstream `docs/design-review.md`, written after reading that document and upstream's prose skill and after reviewing the commit against them; a rebase onto an upstream tip whose `docs/design-review.md` differs SHALL re-review and re-acknowledge every commit rather than only replace the digest. The commit message body and the PR description SHALL answer, in this order, why the change is needed and why this approach, what exactly changed (before and after), the claimed benefit or result (expected distinguished from measured), and the evidence for each claim with untested checks and limitations stated; every figure SHALL name the base commit or package it was measured on, and the body SHALL carry no process assurances (statements that work was reviewed or is ready).

#### Scenario: Digest matches the staged document
- **WHEN** the commit is created in the branch worktree
- **THEN** `python3 -c` over `git show :docs/design-review.md` prints the digest the trailer carries and the `.githooks/commit-msg` hook accepts the commit

#### Scenario: Rebase across a checklist change
- **WHEN** a PR branch is rebased onto an upstream tip whose `docs/design-review.md` digest differs from the one its trailers carry
- **THEN** every rebased commit carries the new digest, its message body follows the four-part order, and figures measured on the earlier base are labeled with that base
