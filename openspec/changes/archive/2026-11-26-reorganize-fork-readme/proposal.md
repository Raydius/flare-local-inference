# Reorganize the opening of the fork's README into short labeled sections

Tier: 1 — two documentation files change; the code paths cited are read, not edited
Type: feature
External Review: Waived — Tier 1 (two documentation files; no code, behavior or procedure changes)

## Stated Requirements

- "The fork’s readme needs better organization.  The wall of text at the beginning is not readable.  Help me reorganize this"
- "That is a good start.  Also create a separate section to discuss 5060ti compatibility" (2026-11-26, after reviewing the reorganized opening)
- "Is the upstream git path still accurate?" (2026-11-26; it was not: GitHub resolves both former names to `NVIDIA/hyperflare-inference`)

Assumed: "the beginning" is everything above `## Active patches` (README.md lines 1 to 12); the patch entries themselves keep their text.
Assumed: the dated History paragraph is removed from the README and replaced by a link, because FORK.md's branch-model section records every base move since 2026-10-24; the items only the README named (for 2026-11-25 and 2026-11-26) move into those FORK.md entries.

## Blast Radius

1. README.md:5 — the opening's base, two-repository and DFlash2 sentences, one line of about 900 characters.
2. README.md:7 — the History paragraph, one line of about 9,700 characters covering seven base moves.
3. README.md:10 — the published-builds and release-name sentences.
4. README.md:12 — the verdict rule, a list of active patch features that the patch list repeats, and the links to the follow-up lists.
5. FORK.md:405 — links `README.md#raydius-fork-patch-notes`; the H1 stays, so the anchor keeps working.
6. grepped `Stack base|History:` in scripts/fork and tests — no tool parses the README opening.
7. grepped `moved to upstream` in FORK.md — ten dated entries, 2026-10-24 to 2026-11-26, covering every History entry of the README.
8. README.md:3 and FORK.md:4 — name the upstream repository `NVIDIA/flare-local-inference`; `gh api repos/NVIDIA/flare-local-inference` and `repos/NVIDIA/spark-local-inference` both return `NVIDIA/hyperflare-inference`.
9. FORK.md:34 — the `upstream` remote row gives the old URL and says GitHub redirects it to `NVIDIA/spark-local-inference`.
10. grepped `spark-local-inference/pull/` in README.md — four PR links (#211, #212, #229); each resolves under `NVIDIA/hyperflare-inference`.
11. helios/src/core/naming.rs:21 — the GPU platform lists; the RTX 5060 Ti's PCI ID `0x2d04` is in none of them, which export (heliopause/src/export/capture.rs:2161) and hardware detection (flare/src/hardware.rs:95) both read.
