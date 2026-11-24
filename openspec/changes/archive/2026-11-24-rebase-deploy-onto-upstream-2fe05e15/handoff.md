# Handoff — rebase-deploy-onto-upstream-2fe05e15

Objective: pull upstream into `main` and rebase `deploy`, splitting the fork into core (`~/dev/flare-local-inference`) and recipes (`~/dev/hyperflare-recipes`, GitHub `Raydius/hyperflare-recipes`) as the owner chose ("Split in this change").

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

Stage: `/rad-apply`, 17 of 20 tasks done (through 4.4, 2026-11-24). Base retargeted to `c0c4f798` (upstream `v0.3.2`; design.md `## Revisions`). Scratch/evidence: `~/models/.cache/rebase-2fe05e15/`.

## State to recheck before acting
- Core split stack: local `split/unified` = `c3cf996a` (140 commits on `c0c4f798`; 4.3 folded a Helios test fix into `555efe97`), `split/patch-00/01/03/06/07`; also pushed as `origin/split/deploy` = `c3cf996a` (staging; delete in 5.3). Real `patch/*` and `deploy` (`e52149ba`) are untouched; leases for 5.2: `deploy` vs origin `3f8f25e4`, patch refs vs their `backup/pre-rebase-20261124-*` shas.
- Recipe fork: local `deploy` = `123d97d` (71 commits on recipe `main` `07d7bb39`), not pushed. Its lock pins ignition `c3cf996a` and FlashInfer `Raydius/flashinfer@d73da5c4` (via `[tool.uv] override-dependencies`).
- vllm-env: `build.sh`, `Dockerfile` (incl. the 4.3 entry-script fix), `console/flare_params.py` edited, uncommitted (commit in 5.3). `build.log` and `model_configs.json` dirty — `model_configs.json` is not this change's; leave it.
- Image for gates: `local/flare-ignition:c3cf996a-123d97de` (the superseded `0efa58ce-1e7fbe5b` and `0efa58ce-b118ed9d` can be removed in 5.3).

## Then
5.1 `/rad-simplify` — its scope also covers this session's additions: the `555efe97` test line, recipe commit `123d97d`, and the vllm-env Dockerfile entry-script layer. 5.2 publish core, 5.3 push recipe fork + vllm-env commit + vault/README image row + cleanup (`split/*` branches, `split/R` and `*-combined` tags, scratch worktrees `work`, `wtR`, `wtCore`, `wtTip`, recipe `wtRecMain`, flashinfer `check-0aa01681`).

Resume prompt:
```
/rad-apply rebase-deploy-onto-upstream-2fe05e15 — read openspec/changes/rebase-deploy-onto-upstream-2fe05e15/handoff.md first, recheck the refs it lists, then run 5.1 (/rad-simplify over the commits task 5.1 names plus the 4.3 additions in the handoff), then 5.2 and 5.3.
```
