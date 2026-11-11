# Handoff — measure-qwen38-dflash2-paths-c16

Repo `~/dev/flare-local-inference`, branch `deploy` (tip `d9dda606`, pushed, clean except this untracked change directory). Stage: planning complete, apply not started (0/9 tasks).

## Resume prompt

```
/rad-apply measure-qwen38-dflash2-paths-c16

Context you need beyond the change's own files:
- Goal: decide whether the fork's qwen38_dflash recipe (mixed prefill) adds anything over upstream's
  `--recipe qwen38` with speculative method dflash2, at max num seqs 16 and 1M context, and whether
  A16 decode and the MXFP8 draft help. Measurement plus one small patch; no other recipe code changes.
- Task 1 (console fields) is done on the console side: vllm-env change flare-dflash2-a16-config-fields
  is archived (commit b8253db), fields `config:defaults.speculative_method` and
  `config:preparation.force_moe_decode_a16` exist, localai.service restarted 2026-11-10 15:19. NOT yet
  checked: that a launched profile's generated /flare-state/ignition/<id>.toml carries both settings;
  check it on arm U's launch, then tick task 1 with that evidence.
- Start with tasks 2 and 3 (draft-folder patch, image rebuild, no-GPU checks); they need no GPU.
- HOLD before task 4: I asked the owner whether GPU 0 is free for the study or whether the Rebase3
  profile should serve meanwhile; no answer yet. Ask once before any GPU 0 work. The console server is
  currently stopped (port 8100 down since the console restart), so GPU 0 is idle.
- Baseline package: entry gb202-tp1-202611101936, console profile "2026-11-10 Flare DFlash2 Rebase3"
  (id in ~/.cache/dflash2-served/rebase3/rebase3-id.txt), image local/flare-ignition:7ebef730.
- Tools from the last session: ~/.cache/dflash2-served/rebase3/console.sh (console REST helper),
  probe.py and rate.py there (fixed-7 probe of the console-served instance; task 3 moves them into
  qualify/). The benchmark's transport_passed is always false on a console-served instance (console
  chat template changes prompt ids); rate.py computes tok/s without that gate.
- Lessons that bit last time: read the vault note "Flare Developer Learnings" section 21 before any
  export (keyword-argument signature check; never alias a console entry record to a copied profile;
  a failed export's kernel folder can be moved to the rebuilt image's id).
- Commits on deploy need the design-review trailer (checklist digest was cfcf; recompute from the
  staged docs/design-review.md) and the repo prose skill; attribute commits to the model that wrote them.
- Session memory ceiling 14 GiB; exports and serving only through the console.
```

## State not in the artifacts

- Peer review: first call failed quorum; the single call on the revised design passed (2/2, certainty 0.825, caller agreement 0.6); reconciliation is in design.md.
- Leftovers the owner has not decided: unused images `local/flare-ignition:77f0e65b` and `a1051e26`; vllm-env commits local, not pushed.
