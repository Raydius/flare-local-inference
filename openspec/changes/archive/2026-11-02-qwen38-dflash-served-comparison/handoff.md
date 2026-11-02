# Handoff — qwen38-dflash-served-comparison

## Objective and location
Change 3b: served DFlash2 vs MTP on code_edit under the adaptive controller, both packages mixed-prefill. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk`, branch `wt/dflash-s-e2vk` at `c977814b`; nothing committed yet (`git status` lists the dirty files). Tier 2, `Concurrency: graph`.

## Stage and stopping point
Apply: 1.1, 1.2, 1.3, 2.1, 3.1 ticked with evidence (3.1 carries its operator-approval). 3.2 in progress: the four-trial screen ran via `qualify/dflash2_served.py run … --output ~/.cache/dflash2-served/results`; all four finished, containers stopped. Trials: transport fine, 0 failed requests, mixed launches positive (24 at c1, 61 at mtp c8), code_edit median tok/s mtp c1 266 / dflash2 c1 306 / mtp c8 499 — but every trial reports `passed: false` solely on `pages_conserved`.

## Blocker to fix first (1.2 code, carried behavior)
`summarize_group` targets `free + cached kv pages == initial free + cached`, but Helios now also parks pages in `flare_snapshot_backing_pages` (group-000: free 18560→17978, cached 0→6, snapshots 64→60 free / 4 cached). Fix the conservation check to include snapshot backing pages (likely `free + cached + snapshot_backing == flare_total_kv_pages`; verify against `~/.cache/dflash2-served/results/mtp-adaptive/c1-r0/group-000-*/group.json`), add a test, note a `discovery:` on 1.2, then rerun the screen into a fresh output dir, then the full 16-trial plan (same command without `--trials`).

## Facts the next session needs
- Inputs: workload `~/.cache/dflash2-served/workload` (generated, 28 cases, validated), plan `…/workload/execution-plan.json`, catalogs `~/.cache/dflash2-served/catalog-{mtp,dflash2}.json`, packages `~/models/.cache/flare/packages/models/nvidia_2FQwen3.8-Flash-Next-FP4-Local/gb202-tp1-202611020610/model.safetensors` (MTP, exported with `--mtp-draft-tokens 7`) and `…/local-inference-lab_2FQwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD/gb202-tp1-202610312330/model.safetensors`; signer files `--publisher-keys ~/.cache/flare-ignition-dflash-ctx/development-publisher.json --trust-store ~/.cache/flare-ignition-dflash-ctx/trust.toml`; image `local/flare-ignition:f829c06f`. `run --help` lists the flags.
- GPU 0 only in Docker; owner's Qwen instance is stopped (GPU 0 was free). Never `docker stop` anything you did not start by name.
- `measurements.json` is ~18 MB per trial (raw events); 3.3's committed copy must strip `observations` or commit only summary + compact per-trial data — decide then.
- 3.2 is an operator leg: ask once with its criterion verbatim after the full plan passes.
- `tests/frontend/qwen38.json` names the DFlash target; `generate` now checks asset fingerprints only (discovery on 1.1).

## Resume prompt
```
Read openspec/changes/qwen38-dflash-served-comparison/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, then run /rad-apply qwen38-dflash-served-comparison. Fix the pages_conserved check first (snapshot backing pages), then rerun the 3.2 screen and the full plan; GPU work only in Docker.
```
