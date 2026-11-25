#!/usr/bin/env bash
# Refresh the upstream mirror and report how far the patch stack has drifted.
#
# - Fetches upstream (with tags and PR-independent refs).
# - Fast-forwards local `main` to `upstream/main` (fails loudly if `main` ever
#   diverged, which would mean a local patch leaked onto the mirror).
# - Pushes the refreshed mirror branch to the fork remote (NO_PUSH=1 skips;
#   FORK_REMOTE overrides `origin`). Never force-pushes.
# - Reports how far `base/current` is behind `upstream/main` and behind the newest
#   upstream release tag, and names the newest fork release. Upstream releases
#   (`vMAJOR.MINOR.PATCH`) and fork releases (`vMAJOR.MINOR.PATCH-rN`) are two tag
#   namespaces, so each is selected by its own form and reported apart.
#
# See FORK.md for the full workflow.
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
MIRROR_BRANCH="${MIRROR_BRANCH:-main}"
BASE_REF="${BASE_REF:-base/current}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

# Upstream and fork release tags share the `v` prefix and differ only by the `-rN`
# suffix, so membership in each set is decided by these anchored forms alone.
UPSTREAM_RELEASE_TAG_RE='^v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$'
FORK_RELEASE_TAG_RE='^v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)-r(0|[1-9][0-9]*)$'

# newest_tag <regex> — read tag names, one per line on stdin, and print the one whose
# captured numeric components are greatest, compared component by component so `r10`
# outranks `r2` and `v0.3.10` outranks `v0.3.9`. Prints nothing when no line matches.
# The comparison is arithmetic, not collation, so no lexical sort order is usable here.
newest_tag() {
  local re="$1" tag best="" n cmp
  local -a parts best_parts=()
  while IFS= read -r tag; do
    [[ "${tag}" =~ ${re} ]] || continue
    parts=("${BASH_REMATCH[@]:1}")
    cmp=0
    for n in "${!parts[@]}"; do
      if ((10#${parts[n]} > 10#${best_parts[n]:-0})); then
        cmp=1
        break
      elif ((10#${parts[n]} < 10#${best_parts[n]:-0})); then
        cmp=-1
        break
      fi
    done
    if ((cmp > 0)); then
      best="${tag}"
      best_parts=("${parts[@]}")
    fi
  done
  printf '%s\n' "${best}"
}

# Block only on tracked changes; untracked files (scratch, build output) are safe.
if ! git diff --quiet || ! git diff --cached --quiet; then
  warn "Tracked changes present; commit or stash before syncing."
  exit 1
fi

say "Fetching ${UPSTREAM_REMOTE} (branches + tags, pruning stale refs)"
git fetch --prune --no-tags "${UPSTREAM_REMOTE}"
# A fork release tag may share a name with an upstream tag; git refuses to
# clobber it, which must not stop the mirror. Keep the fork's tag and say so.
if ! tag_out="$(git fetch --tags "${UPSTREAM_REMOTE}" 2>&1)"; then
  printf '%s\n' "${tag_out}" | grep 'would clobber' >&2 || { printf '%s\n' "${tag_out}" >&2; exit 1; }
  warn "Kept the fork's tag(s) above; ${UPSTREAM_REMOTE} uses the same name(s) for other commits."
fi

# Fast-forward the mirror branch without leaving the current checkout.
current_branch="$(git symbolic-ref --quiet --short HEAD || echo '')"
say "Fast-forwarding ${MIRROR_BRANCH} -> ${UPSTREAM_REMOTE}/${MIRROR_BRANCH}"
if [[ "${current_branch}" == "${MIRROR_BRANCH}" ]]; then
  if ! git merge --ff-only "${UPSTREAM_REMOTE}/${MIRROR_BRANCH}"; then
    warn "${MIRROR_BRANCH} could not fast-forward — it has diverged from upstream."
    warn "A local patch may have leaked onto the mirror. Investigate before continuing."
    exit 1
  fi
else
  # Update the ref in place; refuses (non-zero) if it is not a fast-forward.
  if ! git fetch . "${UPSTREAM_REMOTE}/${MIRROR_BRANCH}:${MIRROR_BRANCH}"; then
    warn "${MIRROR_BRANCH} could not fast-forward — it has diverged from upstream."
    warn "A local patch may have leaked onto the mirror. Investigate before continuing."
    exit 1
  fi
fi

# Keep the fork's mirror branch current too (skip with NO_PUSH=1).
# Deliberately NOT a force push: origin/main should only ever fast-forward;
# a refusal here means the remote mirror diverged and deserves a look.
if [[ "${NO_PUSH:-0}" != "1" ]] && git remote get-url "${FORK_REMOTE:-origin}" >/dev/null 2>&1; then
  say "Pushing ${MIRROR_BRANCH} -> ${FORK_REMOTE:-origin}"
  if ! git push "${FORK_REMOTE:-origin}" "${MIRROR_BRANCH}"; then
    warn "Push of ${MIRROR_BRANCH} to ${FORK_REMOTE:-origin} was refused — the remote mirror has diverged."
    warn "Investigate before force-pushing. (Local sync itself succeeded.)"
  fi
fi

# Drift report (best-effort; base/current may not exist yet on first setup).
if git rev-parse --verify --quiet "${BASE_REF}" >/dev/null; then
  behind_main="$(git rev-list --count "${BASE_REF}..${UPSTREAM_REMOTE}/${MIRROR_BRANCH}")"
  say "base/current is ${behind_main} commit(s) behind ${UPSTREAM_REMOTE}/${MIRROR_BRANCH}"

  # Upstream's newest release is read from the remote, not from local tags: a fork
  # tag that blocked the tag fetch keeps its name locally, so the local list cannot
  # stand in for what upstream actually publishes. `ls-remote` lists annotated tags
  # twice, once peeled, and the peeled line is normalised to the tag name.
  upstream_tags="$(git ls-remote --tags "${UPSTREAM_REMOTE}" 2>/dev/null \
    | sed -e 's|.*refs/tags/||' -e 's|\^{}$||' || true)"
  latest_upstream_tag="$(newest_tag "${UPSTREAM_RELEASE_TAG_RE}" <<< "${upstream_tags}")"
  if [[ -n "${latest_upstream_tag}" ]]; then
    behind_tag="$(git rev-list --count "${BASE_REF}..${latest_upstream_tag}" 2>/dev/null || echo '?')"
    say "base/current is ${behind_tag} commit(s) behind newest upstream release tag ${latest_upstream_tag}"
  else
    say "no upstream release tags on ${UPSTREAM_REMOTE}; upstream release drift unmeasured"
  fi

  latest_fork_tag="$(newest_tag "${FORK_RELEASE_TAG_RE}" <<< "$(git tag --list 'v*')")"
  if [[ -n "${latest_fork_tag}" ]]; then
    say "newest fork release ${latest_fork_tag}"
  else
    say "no fork release tags (vMAJOR.MINOR.PATCH-rN)"
  fi
else
  warn "${BASE_REF} does not exist yet — create it with: git tag ${BASE_REF} <ref>"
fi

say "Sync complete."
