# Re-audit the v0.3.2 rebase and its follow-ups from source, and fix what it finds

Tier: 3
Type: bug
Retired: superseded by rebase-forks-onto-upstream-v0-3-2, which redid the move and attributed each defect this audit listed in its comparison.md; nothing was applied here

## Stated Requirements

- "It sounds like the rebase to 0.3.2 went terribly, honestly."
- "Systematically recheck all aspects of the job, including the followup which should apply the patches as necessary."
- "I don't trust any of the work that was done."

Assumed: "the job" is the archived change `rebase-deploy-onto-upstream-2fe05e15` (core moved from upstream `ba42481a` to `c0c4f798` = `v0.3.2`, and the fork split into core `Raydius/flare-local-inference` `deploy` and recipes `Raydius/hyperflare-recipes` `deploy`), plus everything built on it afterwards: `fork-revision-release-tags` (tag `v0.3.2-r1`, image `ghcr.io/raydius/hyperflare-inference:0.3.2-r1`), the recipe fork's pin commits `01fa2d2`, `c442120` and `d8cd209`, and the vllm-env two-ref image recipe that built `local/flare-ignition:0.3.2-r1-d8cd2098`.
Assumed: "the followup which should apply the patches as necessary" means this change fixes what the audit finds, on core `deploy`, recipe `deploy` and the vllm-env image recipe, one task per defect; no second change is opened.
Assumed: "I don't trust any of the work" means no earlier evidence counts: the rebase's `verification.md`, its gate summaries under `~/models/.cache/rebase-2fe05e15/`, and its range-diff are inputs to re-derive, never proof.
Assumed: the audit baseline is the pre-rebase fork (`backup/pre-rebase-20261124-deploy` and the five `backup/pre-rebase-20261124-patch-*` tags on `ba42481a`) and the last image whose export served correctly, `local/flare-ignition:7566bead`.
Assumed: the acceptance test is the operator's own use: a fresh export of profile "2026-11-24 Flare DFlash2 MXFP8 A16 - v0.3.2-r1" on the corrected stack answers the R1 prompt and coherent chat, served from the console; Muse Glimmer and Elephant-1 each get one fresh export that serves.
Assumed: every export or GPU 0 run starts only on its own go from the operator, who stops and restarts the GPU 0 console instance.
Assumed: the open debug ledger `openspec/debug/new-stack-package-nan-logits.md` continues inside this change; its confirmed cause becomes a task here.
Assumed: the upstream PR branches (`pr/muse-glimmer`, `pr/rope-scaling`) were outside the job and stay outside this audit.

## Root Cause

wrong-mental-model: the rebase took clean conflict resolution plus baseline-relative CPU and component test gates as proof that each fork patch still works, and declared fresh exports out of scope; a fork patch that meets an upstream mechanism without a textual conflict is never re-wired, and nothing runs the patched stack end to end. The fork's `--rope-parameters` (`patch/07`) predates upstream's rebuild record, which `da037284` added inside the rebase range, so `reproduction_arguments` never emits it (`ignition/preparation/rebuild_manifest.py:231`); and the first fresh export on the new stack produced a package whose every request fails with "prepared model reported device error 1" (`openspec/debug/new-stack-package-nan-logits.md:6`), a failure no gate in the rebase could see (`openspec/changes/archive/2026-11-24-rebase-deploy-onto-upstream-2fe05e15/proposal.md:60`).

## Why

The v0.3.2 stack is what the operator now exports and what the published `0.3.2-r1` serving image carries, and its first real export is unusable. Two defects already found came from integration points the rebase never checked, so the remaining fork patches, pins and image steps need the same check before the stack is trusted.

## What Changes

- Account for every pre-rebase fork commit: each of the 153 commits on `backup/pre-rebase-20261124-deploy` maps to a commit on core `deploy`, on recipe `deploy`, or to a recorded drop or supersession, re-derived from the trees rather than the `Split-From:` trailers alone.
- Check each active fork patch (README "Active patches") against every mechanism upstream added in `ba42481a..c0c4f798` that touches the same concern, and re-wire the ones it missed; `--rope-parameters` into the rebuild record is the first.
- Re-check the dependency pins: the FlashInfer pin `d73da5c4` against the 1.4 decision it claims to implement, the GDN head-ratio patch, and the core pin `de89d0c8` in the recipe lock.
- Re-check the follow-ups: the `v0.3.2-r1` tag, the published serving image, and the local preparation image's provenance.
- Find and fix the cause of the device-error-1 package through the existing ledger.
- Re-export the operator's profile, Muse Glimmer and Elephant-1 on the corrected stack and serve each.

## Capabilities

### New Capabilities

- `rope-scaled-export`: an export prepared with `--rope-parameters` records those parameters in its rebuild record, so the recorded reproduction arguments rebuild the same context.

### Modified Capabilities

None.

## Impact

- Core fork `deploy` (`Raydius/flare-local-inference`), recipe fork `deploy` (`Raydius/hyperflare-recipes`), and the vllm-env image recipe `~/LocalApps/vllm-env/build/flare-ignition/`.
- A corrected stack gets a new revision tag (`v0.3.2-r2`) and serving image only if a fix touches serving code.
- GPU 0 time for diagnosis and three exports, each started on the operator's go.

## Blast Radius

1. ignition/preparation/rebuild_manifest.py:231 — `reproduction_arguments` emits every preparation flag except the fork's `--rope-parameters`; `from_plan` (line 321) records no rope field either, so a rope-scaled package's rebuild record describes a 262,144-token build.
2. grepped `git log ba42481a..c0c4f798 -- ignition/preparation/rebuild_manifest.py` — `da037284` and `1bf68649`; `git show ba42481a:ignition/preparation/rebuild_manifest.py | grep -c reproduction` = 0, so the rebuild record arrived inside the rebase range.
3. ignition/cli/preparation.py:54 — `rope_scaled`, the fork's `patch/07` entry point the rebuild record never sees.
4. ignition/cli/__init__.py:54 — the fork's `--draft-model` is declared on `export` only; `debug-serve` cannot load the console's draft folder, which blocks the ledger's H7 diagnostic.
5. ignition/preparation/capture.py:25 — upstream's warmup-stream and cuBLAS-workspace capture rewrite, a candidate path for the device-error-1 package (ledger H9).
6. openspec/debug/new-stack-package-nan-logits.md:6 — 1314 of 1314 requests failed; R1 deterministic; H1 and H3 refuted, H2 inconclusive.
7. openspec/changes/archive/2026-11-24-rebase-deploy-onto-upstream-2fe05e15/proposal.md:22 — the proof the rebase relied on: the baseline-relative gate set; line 60 puts re-exports and served measurement out of scope.
8. grepped `git rev-list --count ba42481a..backup/pre-rebase-20261124-deploy` = 153 fork commits before; `c0c4f798..deploy` = 148 with 139 `Split-From:` trailers; recipe `main..deploy` = 74 with 67 trailers; `~/models/.cache/rebase-2fe05e15/dropped.txt` names 2 drops and `core-emptied.txt` 15 emptied commits.
9. README.md:33 — the active patch list: `patch/00`, `01`, `03`, `06`, `07`, the GHCR image, the Elephant-1 block mask, the persisting L2 limit, the GDN head-ratio patch, copy-depth attribution, `--recipe`/`--draft-model`, and the recipe-side patches.
10. grepped the recipe fork `pyproject.toml` at `d8cd209` — line 31 pins FlashInfer `Raydius/flashinfer@d73da5c4` (branch `evict-planned-jits-0aa01681`, one commit past the port onto `0aa01681`), line 38 pins core `de89d0c8`.
11. grepped `flashinfer-*.patch` in the vllm-env image recipe's `build.sh` (outside this repo) — line 41 copies every FlashInfer patch into the image build; the GDN head-ratio patch is applied there.
12. grepped `git tag --points-at c0c4f798` — `base/current`, `v0.3.2`; `v0.3.2-r1` tags `de89d0c8`, the commit the published serving image records.
