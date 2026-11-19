# Rebase the fork stack and deploy onto upstream 1a532755

Tier: 3
Type: feature

## Stated Requirements

- "pull upstream main and help me rebase deploy"
- "what's your recommendation?" — the operator accepted the recommended base (upstream's moving tip, unpinned) and the probe-before-planning order with "yes"
- "ok let's do the base tip then" — `main` moves to `upstream/main` first, before the stack
- "yes" — the backup tags and the throwaway probe run before the plan is written
- Not included: re-export and re-measurement of the served cores — the operator accepted "What I'd keep out: re-export and re-measurement … the same split you asked for on 2026-11-15" with "yes"; `flare inspect` of existing entries stays in scope

Assumed: the cores kept out of re-export are whichever entries the console serves at the time — the Qwen, Muse Glimmer and Elephant entries as of 2026-11-18, the same split as on 2026-11-15; the console's configuration is the source of truth for which profiles those are, and this set is the author's reading of the accepted exclusion, not a list the operator gave. `flare inspect` and serving an existing core for a functional check stay in scope.
Assumed: the change rebases `base/current`, `patch/00`, `patch/01`, `patch/03`, `patch/06`, `patch/07` and `deploy` onto `upstream/main` `1a532755` with `scripts/fork/rebase.sh`'s command (`git rebase --onto <base> base/current deploy --update-refs`), keeping both sides' behaviour at each stop, as on 2026-11-15.
Assumed: `main` was already fast-forwarded to `1a532755` and pushed to `origin` on 2026-11-18 by `scripts/fork/sync.sh`, so the change starts at `base/current`; the pre-flight re-fetch still runs and retargets the plan if upstream moved again.
Assumed: backup tags `backup/pre-rebase-20261118-*` already exist at the eight refs; the rebase is force-pushed ref-by-ref with `--force-with-lease` against them at the end, `base/current` last.
Assumed: replayed commits keep their historical `Design-Signed-Off-By` trailers and continue with `git -c core.hooksPath=/dev/null rebase --continue`; every commit this change writes is acknowledged against the new checklist digest `1af4`.
Assumed: the fork's FlashInfer pin is re-cut onto upstream's new pin `2cce3e67` carrying the fork's two commits and the image recipe's GDN head-ratio patch, and retires if `2cce3e67` already evicts planned Triton JITs for every JIT module.
Assumed: `skip_specs: true` — a rebase carries every requirement of the nine existing capabilities; only what upstream brings and the four hand-resolved commits change.
Assumed: the proof is the 2026-11-15 gate set minus the console work: baseline-relative host CPU suites against a pristine `1a532755` worktree, a containerized image build, the per-crate Rust suites in that image, and the GPU 1 CUDA files; GPU 0 console re-exports are out.
Assumed: the open PR branches (`pr/muse-glimmer`, `pr/rope-scaling`, `pr/mixed-speculative-export`), the detached worktrees, and the FlashInfer PR #67 head are untouched; upstream's new Discord-contact rule is not applied to those PR descriptions here.

## Why

`deploy` sits on upstream `6b68b9b4`, 81 commits (47 non-merge) behind `upstream/main` `1a532755`. The served image picks up none of it until the stack moves, and this batch bears directly on fork work:

- `ca1b5c31` (#289) bounds history proposals by logical lane length in `helios/src/model/packing.rs`, the file the fork's `--history-drafts` copies extend;
- `10bc91ad`/`d0ce8a2c` estimate checkpoint work without quoting snapshot storage and `7f1ac303`/`714eda0a` size ordinary and mixed prefill graphs to quoted checkpoint work — the same sizing the fork's mixed speculative export and DFlash2 rows depend on;
- `80c97c50` passes mutated dtype-viewed temporaries to storage calls through their owner, replacing the mechanism the fork's opaque-write copy rule patched around;
- `6f541e87`/`80ccb44a`/`ab606762` add 16-bit PNG and GIF input with Pillow-equivalent conversion in the same `helios/src/frontend/images` files that carry the fork's Muse Glimmer profile;
- `90286fdf` re-pins FlashInfer to `2cce3e67` for prepared PCIe collective improvements, below the fork's own pin.

The probe replay already shows the move is small and mechanical, so the value of doing it now is that the fork's four hand-resolved commits stay a maintenance cost rather than a stack drift: 121 of the 125 replayed commits come across byte-identical.

## What Changes

- `base/current` moves `6b68b9b4` → `1a532755`; `patch/00`, `patch/01`, `patch/03`, `patch/06`, `patch/07` and `deploy` replay all 125 non-merge commits (`ddaf7c29`, the one merge commit, flattens as it did on 2026-11-15). `main` is already at `1a532755`.
- Five files conflict and are resolved keeping both sides on upstream's structure: `docs/TODO.txt` (the fork's rename to `docs/TODO.md` plus upstream's new entries), `pyproject.toml` and `uv.lock` (the FlashInfer pin), `qualify/speculative_benchmark.py` and `tests/reference/test_speculative_benchmark.py` (upstream `be4ff01e` qualification metrics beside the fork's code_edit comparison).
- Four commits' patches change under `range-diff` and need hand work: `de8ada69` (history drafts, applies cleanly but against #289's one-line change in `packing.rs`), `6cbf42d4` (pin), `cb071b57` and `6f2c9d5b` (the opaque-write copy pair, which is net zero on `deploy` and must leave `ignition/preparation/compilation.py` equal to upstream's).
- FlashInfer: `Raydius/flashinfer` is re-cut from `2cce3e67` carrying `7245b1e0`'s stub rejection and MLA-compress memo sweep and `2f55cfd98`'s GDN head-ratio change, or the fork pin retires if upstream's `2cce3e67` already passes the three eviction cases; `pyproject.toml` and `uv.lock` follow, and the image recipe's `gdn-head-ratio.patch` is re-checked because `build.sh` fails the build when a patch no longer applies.
- The design checklist digest moves `ed22` → `1af4` (upstream `de75b2ab` added item 17, Discord contact), so every commit this change writes acknowledges `1af4`.
- README patch notes and FORK.md record the base move, and every "Still needed" verdict is re-judged against `1a532755`. Two rows need real attention because upstream already holds what the row claims it lacks: `copy_offered_mask` and `ProposalSource::Copy` are at the base already (`helios/src/execution/transport.rs:101`), and upstream refuses rather than ignores a `generation_defaults` key (`helios/src/server/config.rs:47`).
- No behaviour change to any recipe, Helios policy, contract, or core format beyond what upstream brings. `helios/proto/runtime_core.proto` is untouched upstream since the base, so Muse Glimmer keeps image-preparation number 11 and no core renumbering is needed.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. A rebase carries every requirement (`skip_specs: true`); the `specs` artifact is skipped, not empty.

## Impact

- Branches: `base/current`, `patch/00`, `patch/01`, `patch/03`, `patch/06`, `patch/07`, `deploy`; `main` already moved. Backup tags `backup/pre-rebase-20261118-*` exist at the pre-move commits. `Raydius/flashinfer` gains at most one branch.
- Cores: existing cores keep loading — upstream changed no `helios/proto`, no execution ABI, and no `helios/manifest.md` since the base (checked with `git diff --name-only 6b68b9b4 upstream/main -- helios/proto helios/manifest.md`, which lists only `docs/design-review.md` from the checked set).
- Build: the console image recipe (`~/LocalApps/vllm-env/build/flare-ignition/build.sh`) is unchanged apart from the FlashInfer patch applicability check; `uv.lock` changes only in its `flashinfer-python` source (`uv.lock:403`).
- Upstream PRs #211, #212, #229 and FlashInfer #67: untouched; if the pin retires, the report recommends closing #67, which is the operator's call.
- Out of scope: console/GPU 0 re-exports and served re-measurement; the PR branches and detached worktrees; applying upstream's new Discord-contact rule to existing PR descriptions.

## Blast Radius

1. scripts/fork/sync.sh:42 — the in-place fast-forward of `main`; already run 2026-11-18, so `main`, `origin/main` and `upstream/main` all resolve to `1a532755`.
2. scripts/fork/rebase.sh:77 — `git rebase --onto <base> base/current deploy --update-refs`, the command that moves the stack and retags `base/current`.
3. grepped `backup/pre-rebase-20261118` — the eight backup tags exist at `50dade75` (deploy), `1a532755` (main), `6b68b9b4` (base/current), `4e0ee71e`, `d9838865`, `36b4c667`, `f2e1c724`, `0129d7e0`.
4. grepped `'^STOP'` in `/home/rjdollete/models/.cache/rebase-1a532755/probe.stops.{theirs,both}` — 3 and 5 stopping commits; grepped `'Could not apply'` in `rebase.theirs.log` — 15 under the first, mis-skipping variant; grepped `'^[0-9]+[:.] *!'` in `range-diff-theirs.txt` — 4 changed pairs of 125; the replays are in `trees/probe-{theirs,both}`.
5. grepped `'^CONFLICT'` in the output of `git merge-tree --write-tree 1a532755 deploy` (tree `1eb7d6c5`, exit 1) — five conflicted paths: `docs/TODO.txt`, `pyproject.toml`, `uv.lock`, `qualify/speculative_benchmark.py`, `tests/reference/test_speculative_benchmark.py`; the same set all three methods found.
6. pyproject.toml:25 — the fork's `Raydius/flashinfer@7245b1e0…` pin, stop `6cbf42d4`; upstream `90286fdf` moves the same line to `local-inference-lab/flashinfer@2cce3e67…`.
7. uv.lock:403 — the locked `Raydius/flashinfer` source for the same pin; regenerated by `uv lock` with only `flashinfer-python` allowed to change.
8. grepped `'2cce3e67411fe9eac97f87c3418f485aa406575a'` with `git cat-file -t` in `/home/rjdollete/dev/flashinfer` — `commit`, so the re-cut needs no new fetch; the clone holds `7245b1e0b` (planning-stub rejection, MLA-compress memo sweep) on `4fd836dcd` plus `2f55cfd98` (`gdn-head-ratio`).
9. docs/TODO.txt — stop `b85c54d8` (the fork's rename to `docs/TODO.md` against upstream's new entries); upstream's added entries fold into `docs/TODO.md`, and the `TODO.md` links in AGENTS.md and docs/core-format-design.md:5 follow the fork's rename commit.
10. ignition/preparation/compilation.py:518 and :548 — stops `cb071b57` and `6f2c9d5b`, whose pair is net zero on `deploy` (`git diff 6b68b9b4 deploy -- ignition/preparation/compilation.py` prints 0 lines); upstream `80c97c50` rewrites `_lower_storage` at those lines, so the final file equals upstream's and the pair's reproducers survive in tests/preparation/test_aot.py:1.
11. qualify/speculative_benchmark.py:23 and :229 (`analyze_events`, `stream_request`) and tests/reference/test_speculative_benchmark.py:1 — stop `4128ea63`; upstream `be4ff01e` edits the same qualification-metric helpers in both files.
12. helios/src/model/packing.rs:399 — the fork's history-draft input packing; upstream `ca1b5c31` (#289) changes one line in the same `impl Inputs`; auto-merged in the probe, so the rule is read-both-sides, not a mechanical union.
13. helios/src/scheduler/speculation.rs — the fork adds 31 lines for copy-depth attribution (`0549c4d8`); `git diff 6b68b9b4 upstream/main -- helios/src/scheduler/speculation.rs` is empty (`10bc91ad` touches `policy.rs`, `checkpoints.rs`, `cache_memory.rs`, `tests.rs`), and upstream consumes `copy_offered_mask` only for transport validation (`helios/src/execution/transport.rs:100`, `:144`) with no `acceptance.observe` call site added in the range, so the fork's hunks are load-bearing and carried; the only production caller is `helios/src/scheduler.rs:2561`.
14. helios/src/scheduler/policy_tests.rs:1275 and :1971, helios/src/scheduler/tests.rs:42, helios/src/scheduler/planning_tests.rs:115 — the fork's copy-attribution tests (`0549c4d8`) against upstream `10bc91ad`; auto-merged, reviewed for semantics.
15. helios/src/server/request.rs:227, helios/src/server/runtime.rs:246 and :1406, helios/src/server/runtime/phase_tests.rs:3610 — the Elephant-1 block-end marker spans and preempted continuation (`5acb610d`, `6d67fd70`) against upstream's GIF input (`ab606762`) and #289's phase tests (`ca1b5c31`); auto-merged, the highest-risk semantic overlap because it landed 2026-11-17 and never ran against these upstream commits.
16. helios/src/frontend/images/conditioned.rs:192 and :584 — the Muse Glimmer and PaddedGrid profile arms and the fork's exhaustive feature-width match (`:558` records the sized-PatchMerge fault) against upstream `6f541e87`/`80ccb44a`/`ab606762`; auto-merged, with the standing rule that no `_ => 588` wildcard returns.
17. tests/cli/test_planning.py:327 and tests/preparation/test_checkpoint_source.py:19 and :32 — fork Glimmer planning and checkpoint-source cases against upstream `949966ba`/`67bccf07` peer-export work; auto-merged.
18. docs/design-review.md:150 — upstream `de75b2ab` adds checklist item 17 (Discord contact), moving the digest the hook checks at .githooks/commit-msg:62 from `ed22` (`6b68b9b4`) to `1af4` (`upstream/main`).
19. grepped each active patch's feature on `upstream/main` — `rope-parameters` 0 files, `muse_glimmer_nvfp4` 0, `substitute_copies` 0, `block_end_marker` 0, `elephant1` 0, `max_context_tokens` 0, so `patch/06`, `patch/07`, the Glimmer and Elephant-1 recipes and the serving overrides stay; `generation_defaults` appears only as a rejected key at helios/src/server/config.rs:47 and heliopause/src/export/binding.rs:758, so `patch/01` stays.
20. grepped 'TODO.txt' in AGENTS.md and docs — the fork's rename leaves one upstream-added link per file to repoint, the same class the 2026-11-15 carry commit `b7376ed4` fixed.
21. README.md:6 and README.md:12 — the stack-base line and the "judged against `6b68b9b4`" sentence, both rewritten for `1a532755`; FORK.md:114 gains the 2026-11-18 base-move entry.
22. grepped 'cargo build' in `~/LocalApps/vllm-env/build/flare-ignition/Dockerfile` — it builds `flare-cli/Cargo.toml`; upstream changed `flare-cli/src/orbit/agent.rs` (`9de65fbd`, transferable Orbit primaries) and `flare-cli/Cargo.lock` (`ab606762`) in this range, which the fork never touched, so no console image recipe change is needed but the in-scope `flare inspect` now runs upstream-changed Orbit code and gains an Orbit group check in design step 9.
