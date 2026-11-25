# Publish fork releases under a revision tag namespace

Tier: 2
Type: feature

## Stated Requirements

- "so let's tag this as v0.3.2R"
- "actually let's add a dash, so v0.3.2-r"
- "B" — selecting "B — fork revision tags, then `v0.3.2-r1`": widen both release workflows' tag filters and the version script so a fork revision tag publishes, then tag the current `deploy` tip `v0.3.2-r1`
- "I don't want stale images to be referenced in the readme"

## Why

A fork release has to borrow upstream's tag namespace. Both tag-triggered workflows
accept only `vX.Y.Z` and `scripts/set_release_version.py` refuses anything else, so the
fork's first release on upstream's `v0.3.2` base (`c0c4f798`) would have to be a bare
`v0.3.3` — a name upstream will itself use, which is exactly the collision
`scripts/fork/sync.sh:30-34` already tolerates for `v0.2.3`. Tagging `v0.3.2-r` today
publishes nothing: no workflow matches that ref, and the version script rejects the
string, so the release fails silently as a successful tag push.

## What Changes

- Both tag-triggered workflows also accept `vMAJOR.MINOR.PATCH-r<N>`
  (`.github/workflows/container.yml:9`, `.github/workflows/release.yml:5`).
- `scripts/set_release_version.py` accepts the optional `-r<N>` revision: it validates
  the numeric core with the existing `STABLE_VERSION` pattern and the revision suffix
  with its own pattern, then stamps the single full string into the three release
  crates' manifests and lockfiles (`scripts/set_release_version.py:11-15`).
- One version string end to end: the tag `v0.3.2-r1` yields image tags `0.3.2-r1` and
  `sha-<commit>`, and `flare --version` prints `flare 0.3.2-r1` because clap reports the
  Cargo package version (`flare-cli/src/cli.rs:30`). The workflows' existing string
  equality gates compare against that same value, so a disagreement between the tag and
  the binary stays a build failure.
- `docs/container.md`, `docs/release.md`, and README's serving-image row state the accepted
  tag forms and why the fork needs its own namespace.
- Unchanged: the bare `vX.Y.Z` path, the manual-dispatch `0.0.0` path, the tagged-commit
  default-branch gate, the Dockerfile, and every serving code path.

## Capabilities

**New Capabilities**

- `release-versioning` — the fork's release identity: which tag forms start a release,
  how the version string is derived from the tag, and what `flare --version` must report.

**Modified Capabilities**

- None. The ten capabilities under `openspec/specs/` describe model preparation and
  serving behavior; none covers release publishing.

## Impact

`.github/workflows/container.yml`, `.github/workflows/release.yml`,
`scripts/set_release_version.py`, `scripts/fork/sync.sh`, `docs/container.md`,
`docs/release.md`, `README.md`, plus `tests/test_set_release_version.py` (new) and
`tests/test_fork_sync.py`.

Untouched: `Dockerfile`, `packaging/` (its `PACKAGE_VERSION` is an argument, not derived
from a tag — `packaging/debian/package.sh:12`), and the committed Cargo manifests, which
the workflows stamp at build time only, as `docs/release.md:43` already requires.

Principal risk: a malformed revision tag producing an image whose reported version
disagrees with its tag. Contained by the version-equality and fork-flag gates that
already exist (`.github/workflows/container.yml:65-86`,
`.github/workflows/release.yml:88-91`) plus script-level rejection of any tag form
outside the two accepted shapes.

## Blast Radius

1. `.github/workflows/release.yml:5` — tag filter `['v[0-9]+.[0-9]+.[0-9]+']`: a `-r` tag never starts a release run.
2. `.github/workflows/container.yml:9` — same filter: a `-r` tag never builds the serving image.
3. `scripts/set_release_version.py:16` — `STABLE_VERSION` matches only a bare `X.Y.Z`.
4. `scripts/set_release_version.py:106` — `STABLE_VERSION.fullmatch(args.version)` sends `parser.error` for `0.3.2-r1`.
5. `scripts/set_release_version.py:11-15` — `CRATES` receives the stamped string in `flare-cli/Cargo.toml`, `flare-winui/bridge/Cargo.toml`, `flare-winui/launcher/Cargo.toml` and their lockfiles.
6. `.github/workflows/container.yml:32` — `version="${GITHUB_REF_NAME#v}"` derives the image tag and the stamp argument from the tag name.
7. `.github/workflows/container.yml:68` — `EXPECTED: flare ${{ steps.image.outputs.version }}` is compared to the pulled image's `--version`.
8. `.github/workflows/release.yml:30` — same derivation for the binary archives; `.github/workflows/release.yml:89` compares it inside the Ubuntu 24.04 container.
9. `.github/workflows/release.yml:34` — `--check 0.0.0` on the manual-dispatch path must keep passing.
10. `.github/workflows/release.yml:210` — `gh release create "$GITHUB_REF_NAME"` names the GitHub release after the full tag, `v` included.
11. `flare-cli/src/cli.rs:30` — clap's `version` reports the Cargo package version, so the stamped string is what `--version` prints.
12. `docs/container.md:38-39` and `docs/release.md:44` — both document the accepted tag form and the resulting image or archive version.
13. `README.md:36` — the serving-image patch row restates the trigger and the tag form.
14. `scripts/fork/sync.sh:71` — the `git tag --list 'v*' --sort=-v:refname` newest-release computation cannot separate fork `-rN` releases from upstream's bare tags once both exist; `scripts/fork/sync.sh:30-34` is the clobber tolerance that shows the two namespaces already collide.
15. grepped `tags:` across `.github/workflows` — only `release.yml` and `container.yml` trigger on tags; `helios-lifecycle.yml` and `native-serving.yml` do not.
16. grepped `PACKAGE_VERSION` across `packaging/` — supplied as an argument and validated by `dpkg --validate-version` (`packaging/debian/package.sh:12`), never derived from a git tag; no site.
17. grepped `container|release|tag|version` across `openspec/lessons.md` — only `openspec/lessons.md:26` (`serve-from-the-operators-running-profile`) matches, on serving configuration; no release-automation entry exists to consult.
18. `tests/test_fork_sync.py:26-45` — the existing scenario builds upstream and fork tags of the same name in `tmp_path`; the two-namespace report extends it.
19. grepped `set_release_version` across `tests/` — no hits; the release script has no test today, so `tests/test_set_release_version.py` is new.
