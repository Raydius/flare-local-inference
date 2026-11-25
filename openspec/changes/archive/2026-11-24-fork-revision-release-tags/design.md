# Design — fork-revision-release-tags

## Approach

Two namespaces, one version string. The git tag carries the fork namespace
(`v0.3.2-r1`); everything downstream uses that tag minus its leading `v`, so no step ever
holds a second spelling of the release.

- `scripts/set_release_version.py` — widen the accepted argument to `X.Y.Z` or `X.Y.Z-rN`:
  split an optional `-r<digits>` suffix, validate the numeric core with the existing
  `STABLE_VERSION` (line 16) and the suffix with a new
  `REVISION_SUFFIX = re.compile(r"-r(0|[1-9]\d*)$")`, and reject anything else with a
  `parser.error` that names both accepted forms. The value stamped stays the single full
  string, so `check`, `set_version`, and `replace_manifest_version` keep their current
  shape. Verified on this machine: Cargo accepts the string as a pre-release version — a
  scratch crate at `version = "0.3.2-r1"` parses in `cargo metadata`, records
  `version = "0.3.2-r1"` in `Cargo.lock`, and compiles (`Compiling vmeta v0.3.2-r1`), and
  `cargo update --package <pkg> --precise 0.3.2-r1` succeeds. Precedence is irrelevant
  because nothing depends on these crates by version: `flare-cli` reaches `flare` and
  `flare-helios` by path (`flare-cli/Cargo.toml:23,44`).
- `.github/workflows/container.yml` — `tags:` (line 9) gains the revision form:
  `['v[0-9]+.[0-9]+.[0-9]+', 'v[0-9]+.[0-9]+.[0-9]+-r[0-9]+']`. Nothing else in the version
  path changes: line 32 already derives `version` from `${GITHUB_REF_NAME#v}`, and the stamp
  argument (line 46), the image tags, and `EXPECTED: flare <version>` (line 68) all consume
  that one output.
- `.github/workflows/release.yml` — the same widened filter at line 5; the derivation at
  line 30 already yields `0.3.2-r1` for the archives, the Ubuntu container comparison at
  line 89, and `gh release create "$GITHUB_REF_NAME"` at line 210. The `--check 0.0.0`
  dispatch path (line 34) is untouched.
- Both workflows gain one boundary step that re-validates `$GITHUB_REF_NAME` against the two
  accepted forms and fails naming the ref. The trigger pattern is the only link in the chain
  whose semantics nothing else asserts, and widening a pattern today widens what may start a
  release with no other gate.
- `scripts/fork/sync.sh` — line 71's newest-release computation
  (`git tag --list 'v*' --sort=-v:refname`) cannot separate the two namespaces now that fork
  releases carry `-rN`. Report base drift against the newest upstream tag fetched from
  `upstream`, and the newest fork release separately, both selected by explicit form and
  ordered numerically by component so `r10` follows `r2`.
- `docs/container.md` (lines 38-39), `docs/release.md` (line 44), and `README.md` (the
  serving-image row at line 36) state both accepted tag forms, the resulting image tag and
  `--version` string, why the fork keeps a namespace of its own, and how a consumer finds the
  newest fork release (the repository's releases page and the registry tag list — the
  workflow publishes no `latest` tag, and image tags are not ordered by semver).
- `tests/test_set_release_version.py`, new, follows `tests/test_fork_sync.py:1-45`, which
  drives a real repository script inside `tmp_path`: copy the script into a scratch tree
  holding the three release-crate manifests and lockfiles, run it, and assert that `0.3.2-r1`
  and `0.3.4` stamp all three manifests and lockfiles with the full string; that
  `0.3.2-r01`, `0.3.2-1`, `0.3.2+R1`, and `release` exit non-zero with a usage error naming
  the accepted forms and modify nothing; and that `--check 0.0.0` succeeds without writing.
  Run with `CARGO_NET_OFFLINE=true`; the scratch crates declare no dependencies, so stamping
  needs no network.
- `tests/test_fork_sync.py` — extend with the two-namespace case: upstream tags `v0.3.2`, the
  fork releases `v0.3.2-r1` and `v0.3.2-r10`, and assert the reported newest fork release is
  `r10` and base drift is measured against upstream's tag.

## Architecture / Requirements

### Reference architecture

Archetype: tag-triggered release pipeline with a single source of truth for the version.
Anchor: Semantic Versioning 2.0.0 — pre-release identifiers are dot-separated alphanumerics,
numeric identifiers carry no leading zeros — and the pre-release-identifier convention
GoReleaser and semantic-release use for build qualifiers after the patch number. Both the
container image tag and the package version derive from the one tag string rather than from
separate configuration.
Invariants: one tag yields one version string and every artifact carries it; the pipeline
fails rather than publishes when an artifact's reported version disagrees with the tag
(`.github/workflows/container.yml:65-86`, `.github/workflows/release.yml:88-91`); the fork's
and upstream's tag namespaces stay disjoint and distinguishable by form alone.
Rejected approach: Cargo build metadata (`0.3.2+r1`) — rejected because it makes the reported
version differ from the image tag, so the existing equality gates would need a mapping
between two spellings, and build metadata is ignored in precedence, which invites `0.3.2+r1`
and `0.3.2` to be treated as the same release.
Deviations: the suffix is fork-only (`-r<N>`), not a conventional `-rc.N`, so pre-release
precedence sorts a fork revision below the bare upstream version. Harmless here: these crates
are never published to a registry and no manifest depends on them by version
(`flare-cli/Cargo.toml:23,44`).

## External Review

### Peer Review Verdict

Quorum: achieved (weight 4/3, escalated to tier 2)
Certainty: 0.8
Caller Agreement: 0.9
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5-ff58b3742fba2169ade71fc8bb6925b4a7ea909550dd87dcaedf68281b669524 (claude-haiku-5-5) — ok, agreement 0.3
- local-ai-qllm-74262720f1174d592ba4e03cd0f0dd095c75557b9ae5a5dfcf78d49db991f33f (qLLM) — ok, agreement 0.2
- google-gemini-rpa-models-gemini-flash-lite-latest-b76944eca2a969f9bd0f5d9c243b98a344a520198dce7233453af029c984f6ee (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-deepseek-flash-96a4e20b1795d40312254a598ede0ade3fb04eb67e0d115a8d5de6a7cc2406d4 (deepseek-flash) — ok, agreement 0.6
- anthropic-claude-sonnet-5-5-386ce35ed356619b8b943067556706c001bea15588fff9c54208146c3e453139 (claude-sonnet-5-5) — ok, agreement 0.9

### Review Feedback

Consensus feedback recorded verbatim:
- "Encode the release as tag `vX.Y.Z-rN` (a SemVer pre-release identifier on the upstream base), and stamp the identical string `X.Y.Z-rN` into the Cargo manifests and lockfiles, the `--version` output, and the Docker/archive names so the existing equality gates still hold." — ACCEPTED: the approach keeps exactly one string across all four places.
- "Widen both workflow tag filters (GitHub globs cannot express the regex, so use an explicit pattern), and change the release script's validator to fullmatch the numeric core plus `-rN` suffix." — ACCEPTED: two explicit patterns per workflow, and the script validates with `fullmatch` on both parts.
- "In the mirror script, select fork and upstream tags by explicit regex and order by (base, N) numerically rather than by `git tag --sort=-v:refname`." — ACCEPTED: new task on `scripts/fork/sync.sh:71`, with the two-namespace case added to `tests/test_fork_sync.py`.
- "the script or stamping drops the `-rN` suffix so binaries report the bare base version" — ACCEPTED: the existing `--version` equality gates catch it, and the new test asserts the full string lands in every manifest and lockfile.
- "Docker metadata patterns skip `latest` or floating tags for pre-releases; a `contains('-')` heuristic misclassifies releases" — REJECTED: no metadata action is used (image tags are listed explicitly at `container.yml:58`) and no step tests for a dash, so neither failure mode exists here.
- "git or SemVer ordering ranks `X.Y.Z-rN` below upstream `X.Y.Z` or misorders `r10` before `r2`. Equality checks must compare the full string, not a prefix or substring, and the regex must be anchored." — ACCEPTED: ordering is handled by the sync.sh task, comparisons stay full-string at `container.yml:74` and `release.yml:89`, and every new pattern is anchored.

## Architecture Review

### Architecture Review Verdict

Quorum: achieved (weight 5/3, escalated to tier 2)
Certainty: 0.7666666666666666
Caller Agreement: n/a
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5-ff58b3742fba2169ade71fc8bb6925b4a7ea909550dd87dcaedf68281b669524 (claude-haiku-5-5) — ok, agreement 0.65
- local-ai-qllm-74262720f1174d592ba4e03cd0f0dd095c75557b9ae5a5dfcf78d49db991f33f (qLLM) — ok, agreement 0.85
- google-gemini-rpa-models-gemini-flash-lite-latest-b76944eca2a969f9bd0f5d9c243b98a344a520198dce7233453af029c984f6ee (models/gemini-flash-lite-latest) — ok, agreement 0.15
- deepseek-deepseek-flash-96a4e20b1795d40312254a598ede0ade3fb04eb67e0d115a8d5de6a7cc2406d4 (deepseek-flash) — ok, agreement 0.75
- anthropic-claude-sonnet-5-5-386ce35ed356619b8b943067556706c001bea15588fff9c54208146c3e453139 (claude-sonnet-5-5) — ok, agreement 0.7

### Architecture Review Feedback

Consensus feedback recorded verbatim:
- "The tag-triggered, single-source-of-truth pipeline with --version equality gates is the standard form." — ACCEPTED: recorded as confirmation of the reference architecture.
- "The design does not address the stated requirement \"I don't want stale images to be referenced in the readme\"; nothing updates or validates README image tags on a new tag push, so this is a gap that counts as a violation." — REJECTED: the requirement is already met by commit `f49312c7` on `deploy`, which removed every specific image reference from `README.md`; a step that writes tag names into the README would recreate the defect the operator named, and the tag-and-digest pin belongs to the core's model card, as `docs/container.md#publish` requires.
- "consumers need a way to find the newest fork revision" — ACCEPTED: `docs/container.md` states where the published set is listed (the repository's releases page and the registry tag list) and that no `latest` tag exists.
- "The -r<N> deviation sorts below bare 0.3.2 under SemVer, which is acceptable for crates that are never registry-published, but it is less harmless for the published GHCR image and binary archives" — ACCEPTED as documentation: `docs/container.md` says image tags are not ordered by semver, so the newest release is read from the releases page rather than derived from a version comparison. No ordering is added to the registry.
Violated requirement: none — "I don't want stale images to be referenced in the readme" is satisfied by the README wording already merged in `f49312c7`; this change adds no image reference, and the accepted feedback adds only prose that names no version.
