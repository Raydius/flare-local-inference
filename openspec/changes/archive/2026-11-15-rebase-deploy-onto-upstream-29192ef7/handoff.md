# Handoff — rebase-deploy-onto-upstream-29192ef7

Superseded: this change is archived, and tasks.md in this directory records its final state; the stage, task states and resume prompt below describe an earlier point in the change.

Objective: rebase `base/current` → `patch/00…07` → `deploy` onto upstream `6b68b9b4`, porting Muse Glimmer so it still exports. Repo `~/dev/flare-local-inference`, branch `deploy` `3b4a1a2e` (local, not pushed).

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action. Later decisions: tasks.md 2.3 `correction:` and design.md Revisions (2026-11-15, Transformers 5.23.0).

Budget exception: task 3.4 stopped mid-way and its results are in no artifact yet, so its state list is kept in full.

Stage: apply. Tasks through 3.3 are checked with evidence. 3.4 is in progress and its results are NOT yet in tasks.md; 3.5 is not started.

Task 3.4 state (scratch `O=~/models/.cache/rebase-6b68b9b4/console`; console helpers `~/.cache/dflash2-served/paths-c16/lib.sh`):
- Glimmer: profile *2026-11-14 Flare Muse Glimmer c16 Rebase4* (`$O/G4-id.txt`, image `21e90706`) exported entry `gb202-tp1-202611160143` (`$O/G4/export.log`, 1 h 49 min unseeded) and was serving when this was written. Text, image and sixteen streams passed (`$O/G4/checks.json`). `flare inspect --frontend --unverified` gives `muse_glimmer` for it, `padded_grid` for the owner's `…080045`, `qwen` for arm Q (`$O/inspect-frontend-*.json`); the `--graphs` inspections exit 0 (`$O/inspect-*.json`).
- Glimmer 1M passkey was running in the background at depth 0.1 (`$O/G4/passkey.log`, output `$O/G4/passkey/`). If no `rc=` line or summary exists and no `qualify.long_context_passkey` process runs, rerun `$O/glimmer-passkey.sh` with the Glimmer profile serving. Compare with 14/15 (1M: 2 of 3 at 256 reply tokens).
- Qwen: profile *2026-11-14 Flare DFlash2 Q Rebase4* (`$O/Q4-id.txt`) points at `local/flare-ignition:3b4a1a2e`. Not exported: the attempt on `21e90706` failed on the Transformers check that `3b4a1a2e` fixes. Next: stop the console, `$O/export.sh Q4` (background; an unseeded export takes over an hour), then `$O/q-arm.sh` (nine trials and the passkey; written, never run). Compare medians with 555.9 / 436.7 / 1231.2 tok/s and passkey 3/3; report gaps beyond ±8 %.
- Then ask the operator once for the 3.4 leg (criterion in tasks.md), including whether the image answer in `checks.json` matches `color-bars.png`, and which profile to serve afterwards. The owner's originals (arm Q on `f83d3120`, Glimmer c16 on `5ca5a9e7`) are untouched.

Other state:
- Two unpushed carry commits from this session: `21e90706`, `3b4a1a2e`. The image for Qwen equals `deploy`; the Glimmer export's image is one commit behind (design.md Revisions).
- Vault: DFlash2 Recipe Progress "Now (2026-11-15, 17:15)" is current to the Glimmer export start; update it on each state change. Glimmer note and Fork Patch Notes wait for 3.5.
- For the 3.5 report: upstream defects `mimo_v2/bootstrap.py` `buffers=`, unported `muse_spark`, and the Transformers lock against the `qwen38` revision check; the unused `Pooled*` hunks (tasks.md 2.3).

## Resume prompt

```
/rad-apply rebase-deploy-onto-upstream-29192ef7

Read openspec/changes/rebase-deploy-onto-upstream-29192ef7/handoff.md first.

Context: repo ~/dev/flare-local-inference, branch deploy (3b4a1a2e, not pushed), rebased onto upstream 6b68b9b4. Tasks through 3.3 are done. Task 3.4 is mid-way: the Glimmer export on the rebased image serves and passed its text, image and sixteen-stream checks; its 1M passkey may still be running or need a rerun; the Qwen (arm Q copy) export has not run yet.

Continue task 3.4 from the handoff's state list: finish the Glimmer passkey, export and measure the Qwen copy profile with the saved scripts, record the evidence in tasks.md, then ask me for the operator check. Then 3.5.

Standing constraints:
- Exports and serving go through the console from a committed ref; never docker stop the owner's instance; the owner's original profiles and packages are never edited.
- Heavy commands run in their own 14 GiB scope or container; kill processes only by PID.
- New commits run with the hook on, acknowledged against design-review digest ed22.
- Update the vault tracker on every state change.
```
