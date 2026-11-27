# Design — consolidate-ignition-image

## Approach

- **Home:** the preparation image's Dockerfile, FlashInfer patch and build script move to `scripts/fork/image/` in the fork. That path is fork-only, so upstream rebases never touch it, and the image is built from the same repository that defines the release.
- **Dockerfile:** carried from the console repo with the guarded launcher and its build-time check. Two edits: `FROM` pins the CUDA base by the digest the last build resolved (`sha256:44a9504c…`), and `uv` is copied from a pinned version instead of `:latest`. `ENTRYPOINT ["/usr/local/bin/flare"]` with `CMD ["--help"]`.
- **`build.sh <core-ref> <recipes-ref> [--push]`:** resolves both refs and keeps the recipe-lock check. When the core ref is an accepted release tag, version is the tag minus `v`, and `scripts/set_release_version.py <version>` runs inside the `git archive` context before `docker build`; otherwise the build stays `0.0.0`.
- **Tags:** every build tags `local/flare-ignition:<core8>-<recipes8>`, which the console's `FLARE_IMAGE_REPOS` already matches. A release build also tags `ghcr.io/raydius/hyperflare-inference:<version>` and `:sha-<commit>`.
- **Push gate:** `--push` refuses, before building, a core ref that is not an accepted tag or whose commit is not an ancestor of `origin/deploy`.
- **Checks:** before pushing, the local image must report `flare <version>`, list the three fork flags in `run --help`, answer `ignition export --help`, and contain `/NGC-DL-CONTAINER-LICENSE`. After pushing, the script pulls `sha-<commit>` and repeats the version and flag checks, as `container.yml:65-86` did.
- **Removed:** `.github/workflows/container.yml`. The root `Dockerfile` and `release.yml` stay as they are.
- **Republish `0.3.2-r1`:** run `build.sh v0.3.2-r1 <recipes deploy> --push` once the recipes fork's `deploy` is pushed. Record the replaced digest (`sha256:3704cd54…`) and the new digest in the task's evidence.
- **Console repo:** delete `build/flare-ignition/`, and point `docs/operations/flare_runtime.md` at the fork's script.
- **Notices:** `THIRD_PARTY_NOTICES.md` states that the published image bundles the preparation environment under each package's own license. It also states that the image is a derived container under NVIDIA Deep Learning Container License §1(c), whose §2 terms carry to pullers, and that `/NGC-DL-CONTAINER-LICENSE` stays in the image.
- **Test:** `tests/test_fork_image_build.py` drives `build.sh` with a stub `docker` on `PATH` and temporary git repos. It covers the refusal paths (bad tag, unmerged commit, lock mismatch), the stamping argument, and the tag set.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/3, escalated to tier 1)
Certainty: 0.8666666666666667
Caller Agreement: 0.75
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.85
- google-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.5
- deepseek-flash (deepseek-flash) — ok, agreement 0.85

### Review Feedback

Consensus feedback recorded verbatim:
- "Move the preparation Dockerfile and build.sh into the Flare repo so the image is built from the tagged source, with the console repo only consuming it." — ACCEPTED: already the approach (`scripts/fork/image/`).
- "Build from a clean git archive of the tag, stamp the version with set_release_version.py, pin the CUDA base by digest, and fail closed unless the recipes uv.lock pins the core commit (the recipes commit must be pushed first)." — ACCEPTED: added the CUDA base digest pin and a pinned `uv` copy; the recipes push is its own task.
- "Verify the built image (flare --version, the three fork flags in run --help, the ignition launcher, the license file) before pushing :<version> and :sha-<commit>, then pull back by digest." — ACCEPTED: pre-push checks added; the pull-back check stays.
- "Retagging 0.3.2-r1 changes its digest and the entrypoint, which can break pinned consumers; announce it or publish -r2 alongside, and record old and new digests." — ACCEPTED in part: old and new digests are recorded in the task evidence. REJECTED in part: the operator chose to move `0.3.2-r1` rather than publish `-r2`.
- "Workstation builds lack CI provenance and are less correct than a CI build from the tag; a self-hosted CI build would be better if the constraint loosens." — REJECTED: the operator chose this machine; the push gate requires an accepted tag on `origin/deploy`, and the image labels name both commits.
- "Blast radius also includes GHCR permissions and visibility, container.yml, docs, THIRD_PARTY_NOTICES, the 0.2.5 model-card pin, SBOM/provenance, and console docs." — ACCEPTED in part: all except SBOM/provenance are in the Blast Radius; the package is already public, and `gh` holds `write:packages`. REJECTED in part: SBOM/provenance attestation answers no stated requirement.
- "Licensing under NVIDIA DLC §1(c)/§2 needs notices rewritten and a legal read." — ACCEPTED in part: the notices are rewritten against §1(c) and §2. REJECTED in part: a legal read is the operator's call and is named in the reply, not a task.
