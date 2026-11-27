## Purpose

Defines how the core fork and the recipe fork move to a new upstream base: what is proven when work divides between the two repositories, what each fork patch must show on the new base, and what must serve before a release.

## ADDED Requirements

### Requirement: Division between the forks conserves every changed path
When a base move divides fork work between the core repository and the recipes repository, the move SHALL first replay the fork onto an upstream commit at which both repositories' trees coexist, and SHALL then prove that every path the fork changes at that commit has its content in exactly one of the two results. The core result SHALL change no path upstream extracted, and the recipes result SHALL change only extracted paths. A record SHALL map each fork commit to its core commit, its recipes commit, or the statement that it emptied.

#### Scenario: A fork-added recipe file
- **WHEN** a fork commit adds a file under a directory upstream extracted to the recipes repository
- **THEN** the file is present in the recipes result with the same content, absent from the core result, and the commit map names the recipes commit

#### Scenario: A commit touching both sides
- **WHEN** a fork commit changes a core path and an extracted path
- **THEN** the core result carries the core change, the recipes result carries the extracted change, and the map names both commits

#### Scenario: A stray file fails the proof
- **WHEN** the core result holds a file under an extracted root
- **THEN** the conservation check fails and names the path

### Requirement: Every fork patch carries a verdict on the new base
Each active patch listed in the fork's `README.md` SHALL carry one verdict on the new base: kept, adapted, or retired. A kept or adapted verdict SHALL rest on a failing probe of unpatched upstream, on a review of every upstream mechanism added in the moved range that enumerates what the patch extends, and on the patch's passing tests. A retired verdict SHALL name the upstream commit that supersedes the patch.

#### Scenario: A flag meets a new enumeration without a conflict
- **WHEN** upstream adds, inside the moved range, a record that enumerates preparation flags and a fork flag replays past it with no conflict
- **THEN** the patch's verdict is adapted, the flag is added to the record, and a test fails if a preparation flag is neither recorded nor listed as not affecting the prepared model

#### Scenario: Upstream now provides the behavior
- **WHEN** the patch's own probe passes on unpatched upstream
- **THEN** the verdict is retired and the patch notes name the upstream commit

#### Scenario: A fork flag disappears
- **WHEN** a flag present in `flare run --help` or `ignition export --help` before the move is absent after it
- **THEN** the move is incomplete unless that flag's patch carries a retired verdict

### Requirement: A served export precedes the release
A base move SHALL NOT be released until a package exported by the preparation image built from the release commits serves on the operator's GPU: a correct answer to a fixed factual prompt, a multi-turn chat, an image request when the replaced package accepted images, and a `/v1/models` capability comparison against the replaced package. A failing export or served check SHALL stop the move until its cause is found.

#### Scenario: The export serves
- **WHEN** the operator's profile is exported at its own sequence count on the new stack and the served checks pass
- **THEN** the release steps may run on those commits

#### Scenario: The package fails requests
- **WHEN** a request to the served package returns a server error
- **THEN** no tag is pushed, all other work stops, and diagnosis begins with an export from the unpatched upstream stack
