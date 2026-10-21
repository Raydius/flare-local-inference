#!/usr/bin/env bash
# Rebase the whole patch stack onto a new upstream base.
#
# Usage:
#   scripts/fork/rebase.sh <new-base>
#
# <new-base> may be any git ref:
#   v0.9.3            a release tag
#   upstream/main     bleeding upstream
#   <sha>             an arbitrary commit
#   pr/45723          an unmerged upstream PR (fetched from refs/pull/45723/head)
#
# Mechanics: `git rebase --onto <resolved> base/current <tip> --update-refs`
# moves every branch in the stack forward in one shot (rerere replays known
# conflict resolutions). On success, base/current is retagged to the new base.
#
# See FORK.md for the full workflow.
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
BASE_REF="${BASE_REF:-base/current}"
STACK_TIP="${STACK_TIP:-deploy}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }

[[ $# -eq 1 ]] || die "Usage: $0 <new-base>   (tag | branch | sha | pr/<N>)"
new_base_arg="$1"

# Block only on tracked changes; untracked files are safe across a rebase.
{ git diff --quiet && git diff --cached --quiet; } || die "Tracked changes present; commit or stash first."
git rev-parse --verify --quiet "${BASE_REF}" >/dev/null \
  || die "${BASE_REF} does not exist. Create it first: git tag ${BASE_REF} <ref>"
git rev-parse --verify --quiet "refs/heads/${STACK_TIP}" >/dev/null \
  || die "Stack tip branch '${STACK_TIP}' does not exist."

# Guard against the --update-refs coincident-tip footgun: git updates every
# branch strictly *inside* the rebased range, but silently skips one that
# points at the SAME commit as the tip being rebased — leaving it on stale,
# pre-rebase commits. Refuse rather than corrupt the stack.
tip_sha="$(git rev-parse "refs/heads/${STACK_TIP}")"
coincident="$(git for-each-ref --format='%(refname:short)' \
  --points-at "${tip_sha}" refs/heads/ \
  | grep -vx "${STACK_TIP}" || true)"
if [[ -n "${coincident}" ]]; then
  warn "These branch(es) point at the SAME commit as the tip '${STACK_TIP}':"
  warn "  ${coincident//$'\n'/ }"
  warn "--update-refs would NOT advance them, silently leaving them on stale commits."
  die "Make '${STACK_TIP}' sit strictly above the topmost patch (a deploy-only commit), or set STACK_TIP to the topmost patch branch itself."
fi

# Resolve pr/<N> to a real commit by fetching the PR head.
if [[ "${new_base_arg}" =~ ^pr/([0-9]+)$ ]]; then
  pr_num="${BASH_REMATCH[1]}"
  say "Fetching ${UPSTREAM_REMOTE} PR #${pr_num} head"
  git fetch "${UPSTREAM_REMOTE}" "refs/pull/${pr_num}/head:refs/remotes/${UPSTREAM_REMOTE}/pr/${pr_num}"
  new_base="${UPSTREAM_REMOTE}/pr/${pr_num}"
else
  new_base="${new_base_arg}"
fi

git rev-parse --verify --quiet "${new_base}^{commit}" >/dev/null \
  || die "Cannot resolve new base '${new_base_arg}' to a commit."

resolved_sha="$(git rev-parse "${new_base}^{commit}")"
old_sha="$(git rev-parse "${BASE_REF}^{commit}")"

if [[ "${resolved_sha}" == "${old_sha}" ]]; then
  say "Stack already sits on ${new_base_arg} (${resolved_sha:0:12}); nothing to do."
  exit 0
fi

say "Rebasing stack '${STACK_TIP}' from ${old_sha:0:12} onto ${new_base_arg} (${resolved_sha:0:12})"
say "All branches between ${BASE_REF} and ${STACK_TIP} will advance (--update-refs)."

if git rebase --onto "${resolved_sha}" "${BASE_REF}" "${STACK_TIP}" --update-refs; then
  git tag -f "${BASE_REF}" "${resolved_sha}" >/dev/null
  say "Success. ${BASE_REF} now points at ${new_base_arg} (${resolved_sha:0:12})."
  say "Next: build + verify, then push:"
  say "  git push origin --force-with-lease 'patch/*' ${STACK_TIP}"
  say "  git push origin -f ${BASE_REF}"
else
  warn "Rebase stopped with conflicts. Resolve them, then:"
  warn "  git rebase --continue      (repeat until done)"
  warn "After the rebase finishes, pin the base manually:"
  warn "  git tag -f ${BASE_REF} ${resolved_sha}"
  warn "Or abort entirely:  git rebase --abort"
  exit 1
fi
