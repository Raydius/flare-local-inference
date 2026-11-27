# Binary releases

The GitHub Actions workflow `.github/workflows/release.yml` publishes a GitHub
Release when a version tag is pushed. A release holds one archive per platform:

| Asset | Runner | Executable |
|---|---|---|
| `flare-linux-amd64.tgz` | `ubuntu-24.04` | `flare/flare`, the CLI |
| `flare-linux-arm64.tgz` | `ubuntu-24.04-arm` | `flare/flare`, the CLI with the RoCE transport |
| `flare-windows-amd64.zip` | `windows-2025` | `flare/Flare.exe`, the [desktop client](../flare-winui/README.md) |
| `flare-windows-arm64.zip` | `windows-11-arm` | `flare/Flare.exe`, the desktop client |

Archive names carry no version, so
`releases/latest/download/flare-linux-arm64.tgz` always names the most recent
release. Each archive also contains `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, `licenses/`, and `third_party/`, as
[third-party notices](../THIRD_PARTY_NOTICES.md) describes.

A release contains no Windows CLI, Debian package, Selene manager, or
container image. Build those from source: [Windows serving](windows.md),
[Debian packages](debian.md), [the Selene manager](../selene-manager/README.md),
and [container](container.md). Ignition, Python, and model cores are separate
preparation inputs and are never part of a release.

## Host requirements

Every executable loads the NVIDIA driver at run time; no archive contains CUDA
libraries, and no build links the CUDA toolkit.

- The Linux executables are built on Ubuntu 24.04 and require glibc 2.39 or
  later.
- The arm64 Linux executable links `libibverbs` for multi-host RoCE serving.
  Install `libibverbs1` and `ibverbs-providers`.
- The Windows desktop requires Windows 11. It bundles its Windows App SDK and
  Visual C++ runtime files. `Flare.exe` is unsigned.

Release executables embed no publisher verification profiles; publisher
acceptance follows [signing](signing.md) as it does for a source build.

## Publish a release

Development checkouts use `0.0.0` in the three release manifests and their
lockfiles. Do not commit a release-version bump. Tag the intended commit with a
[release tag](#release-tags) and push the tag. The build jobs use
`scripts/set_release_version.py` to apply the tag's version to
`flare-cli/Cargo.toml`, `flare-winui/bridge/Cargo.toml`,
`flare-winui/launcher/Cargo.toml`, and their lockfiles in each runner's
checkout.

The workflow publishes to the repository that received the tag. Running it
through `workflow_dispatch` performs every build and check and uploads the four
archives as workflow artifacts without creating a release.

Enable the repository's immutable-releases setting so that published assets
cannot be replaced and `gh release verify-asset` can check a downloaded file.
The workflow attaches every asset to a draft and publishes the draft last, the
order that setting requires.

## Release tags

A tag publishes a release only when its name has one of two forms. Each number
is a decimal integer without leading zeros.

| Tag | Meaning | Version |
|---|---|---|
| `vMAJOR.MINOR.PATCH` | The upstream release the fork is based on | `MAJOR.MINOR.PATCH` |
| `vMAJOR.MINOR.PATCH-rN` | Fork revision `N` of that upstream release, counted from 1 | `MAJOR.MINOR.PATCH-rN` |

The release version is the tag name without its `v`. `flare --version`, the
Cargo version in the three release manifests, and the
[serving image](container.md#publish) tag all equal it: tag `v0.3.2-r2` gives
version `0.3.2-r2`. Cargo orders `0.3.2-r2` as a pre-release of `0.3.2`; no
release crate is published to a registry, so that ordering selects nothing.

A revision tag names the newest upstream release tag that is an ancestor of the
fork's base; it does not state that the base is that tag's commit. The base of
`v0.3.2-r2` is upstream `c915b5f0`, 27 commits past `v0.3.2`. The "Stack base"
line of [README.md](../README.md) and the record of bases in
[FORK.md](../FORK.md) give the base commit of each revision.
The Python packages keep their own versions.

The workflows' tag filters are glob patterns and also match numbers with
leading zeros, so each workflow checks the exact form before it resolves a
version and fails on any other name, such as `v0.3.2-r02`, `v0.3.2-rc1`, or `v0.3.2-r`.
`scripts/set_release_version.py` refuses the same strings and then writes no
manifest.

## Builds and checks

All builds use Rust 1.94.1 and the locked Cargo manifests. Every job must pass
before anything is published; no target can be skipped.

- **Version.** A tag push requires the tagged commit to be an ancestor of the
  default branch, the tag name to be a [release tag](#release-tags), and all
  three release manifests and lockfiles to contain `0.0.0`. Each build job applies the tag's version before compiling. A
  dispatched run builds `0.0.0` without publishing a release.
- **Linux.** The runner builds `flare` with
  `cargo build --locked --release --manifest-path flare-cli/Cargo.toml` and
  packages it with `packaging/archive.py`. The extracted archive then runs in
  an `ubuntu:24.04` container that holds only that image's libraries, plus the
  two RoCE packages on arm64: `flare --version` must print `flare VERSION`, and
  `flare run --help`, `flare agent --help`, and `flare orbit --help` must
  succeed. The job also requires the license and notice files in the archive.
- **Windows.** The runner installs tools with
  `scripts/setup_windows.ps1 -RustToolchain 1.94.1 -BuildToolsOnly`, builds
  with `scripts/build_winui.ps1 -Platform PLATFORM -SingleFile`, and packages
  with `packaging/archive.py --winui-packages`. `Flare.exe` is then started
  from a directory that contains no other file. The launcher must exit with
  code zero, and `flare-winui.exe` must open a window from the extracted
  runtime and still be running 15 seconds later.
- **Publication.** `gh release create --draft --generate-notes` attaches the
  four archives, then `gh release edit --draft=false` publishes the release.
  If publication fails after the draft exists, delete the draft before
  rerunning the job.

## Qualification

The runners have no NVIDIA GPU. A release establishes that each executable
builds from the locked manifests, starts on its operating system and
architecture, and ships with its notices. It does not establish model serving.
Serving qualification is stated per platform in [Windows serving](windows.md),
[the desktop client](../flare-winui/README.md), and
[model preparation and serving](model-preparation.md), and a release does not
extend it. The workflow runs no crate test suites; `Native serving` runs them
on x86_64 Linux and x86_64 Windows only.

Qualified on x86_64 Linux outside GitHub Actions: packaging the CLI with
`packaging/archive.py` and the workflow's clean-container check, including its
rejection of a version that differs from the expected one. Stamping
`0.3.2-r2` with `scripts/set_release_version.py` and reading the three stamped
manifests with `cargo metadata --locked` are qualified on x86_64 Linux, as are
the tag checks of both workflows run as shell scripts. The desktop archive
layout is qualified with a placeholder in place of `Flare.exe` and the pinned
NuGet packages downloaded from nuget.org.

Not qualified: any run of the workflow on GitHub-hosted runners, including one
started by a `vMAJOR.MINOR.PATCH-rN` tag, the arm64
Linux build and check, the Windows build and startup check on either
architecture, and publication.
