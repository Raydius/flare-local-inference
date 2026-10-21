#!/usr/bin/env bash
# Refresh the upstream mirror and report how far the patch stack has drifted.
#
# - Fetches upstream (with tags and PR-independent refs).
# - Fast-forwards local `main` to `upstream/main` (fails loudly if `main` ever
#   diverged, which would mean a local patch leaked onto the mirror).
# - Pushes the refreshed mirror branch to the fork remote (NO_PUSH=1 skips;
#   FORK_REMOTE overrides `origin`). Never force-pushes.
# - Reports how far `base/current` is behind both `upstream/main` and the
#   newest release tag, so drift is visible regardless of base style.
#
# See FORK.md for the full workflow.
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
MIRROR_BRANCH="${MIRROR_BRANCH:-main}"
BASE_REF="${BASE_REF:-base/current}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

# Block only on tracked changes; untracked files (scratch, build output) are safe.
if ! git diff --quiet || ! git diff --cached --quiet; then
  warn "Tracked changes present; commit or stash before syncing."
  exit 1
fi

say "Fetching ${UPSTREAM_REMOTE} (branches + tags, pruning stale refs)"
git fetch --prune --tags "${UPSTREAM_REMOTE}"

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

  latest_tag="$(git tag --list 'v*' --sort=-v:refname | head -n1 || true)"
  if [[ -n "${latest_tag}" ]]; then
    behind_tag="$(git rev-list --count "${BASE_REF}..${latest_tag}" 2>/dev/null || echo '?')"
    say "base/current is ${behind_tag} commit(s) behind newest release tag ${latest_tag}"
  fi
else
  warn "${BASE_REF} does not exist yet — create it with: git tag ${BASE_REF} <ref>"
fi

say "Sync complete."
