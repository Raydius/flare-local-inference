# Design — rebase-deploy-onto-upstream-29192ef7

## Approach

Scope and Assumed lines: proposal.md. `main` is the mirror and only fast-forwards. Everything else moves by `rebase.sh`'s own command. A throwaway probe mapped the conflict sites (`/tmp/probe-6b68b9b4.stops`, 15 stops; rerere off, so `.git/rr-cache` holds only resolutions from earlier real rebases). The 15 are a lower bound: the probe took the replayed side, so about ten later commits that touch the same files (proposal Blast Radius 40) will stop again under real resolutions. Each takes the rule of the stop that first touched its file.

0. **Pre-flight.** Run `git fetch upstream`. If `upstream/main` has moved past `6b68b9b4`, retarget to the new tip, rerun the probe, and record a Revisions entry before step 1 (operator rule: the base is upstream's tip at apply time).
1. **Backups.** Tag `backup/pre-rebase-20261114-<ref>` for each of these refs:

   | Ref | Commit |
   |---|---|
   | `deploy` | `885f2237` |
   | `base/current` | `e774a468` |
   | `main` | `e774a468` |
   | `patch/00` | `93a7163c` |
   | `patch/01` | `a70349b5` |
   | `patch/03` | `612598e8` |
   | `patch/06` | `bc37825c` |
   | `patch/07` | `1d1c12ef` |

2. **Mirror.** Run `scripts/fork/sync.sh` on `deploy` (clean tree). It fast-forwards `main` to `upstream/main` and pushes `main` to `origin`, with no force. Done when `main`, `origin/main` and `upstream/main` resolve to the same commit.
3. **FlashInfer pin decided before the stack moves.** Stop `b6b559de` needs the sha.
   1. In a scratch worktree of `~/dev/flashinfer` at `4fd836dc`, add `ef6d6592`'s three new test cases to `tests/experimental/b12x/preparation/test_first_use_compile.py`, with no fix code. Case 1 (an application JIT planned in a spawned worker is evicted) is rewritten to assert on upstream's `_PLANNED_TRITON_JITS`, because `_PLANNED_JITS` does not exist on `4fd836dc`. Case 2 (`_run_job` rejects a stub result) and case 3 (`compile_compress` is a `program_cache`) are kept as written.
   2. Run that file in a container from the current `flare-ignition` image with the worktree first on `PYTHONPATH`. Expected: case 1 passes, and cases 2 and 3 fail, because `4fd836dc` has no stub check and still uses `@lru_cache` at `mla_compress/_cute.py:279`.
   3. Run `git switch -c evict-planned-jits-4fd836dc 4fd836dc` and port the hunks of every failing case on top of upstream's registry, with no fork `_PLANNED_JITS`. Rerun the file: all cases pass. Run `git push fork evict-planned-jits-4fd836dc` and pin that sha; the README row is re-pointed.
   4. Only if all three cases pass in sub-step 2 does `pyproject.toml` take upstream's line verbatim (`local-inference-lab/flashinfer@4fd836dc…`) and the README row move to Retired.
   5. Protoc for step 4: the `protoc-35.1-linux-x86_64.zip` release from `github.com/protocolbuffers/protobuf`, unpacked under `~/models/.cache/protoc-35.1/`. `protoc --version` prints `libprotoc 35.1`, matching `6b68b9b4`'s gencode header `Protobuf Python Version: 7.35.1`. Neither the host nor `.venv` has a working protoc today.
4. **Stack.** In the main checkout on `deploy`, run `git rebase --onto 6b68b9b4 base/current deploy --update-refs --exec '<py_compile every .py the commit changed>'`, then `git tag -f base/current 6b68b9b4` when it completes. A conflicted commit is continued with `git -c core.hooksPath=/dev/null rebase --continue`, so replayed commits keep their historical `Design-Signed-Off-By` trailers. Resolutions by stop:
   - `a70349b5` (patch/01) `flare-cli/src/cli.rs`: union of upstream's new arguments and `--generation-defaults`.
   - `bc37825c` (patch/06):
     - `declaration.py` keeps both `RopeScaling` and upstream #291's `draft_body_quantizations`.
     - `qwen38/bootstrap.py` and `attention.py` take upstream's Inductor-planned temporaries and #250's one-layout embedding binding, then re-apply the fork's `rope_scaling` identity key (written only for scaled declarations) and its scaled rotary call.
     - `rotary.py` and `planning.py` are a union.
     - The tests keep both sides' functions.
     - `docs/TODO.txt` takes upstream's text plus the fork's entry.
   - `e6320977` (history drafts): take upstream's functional draft round in `parallel_draft.py`, `packed_speculation.py` and `speculation.py`, and keep Inkling's `parallel_draft.py` hunks. Upstream's round holds no per-round tensors after prepare (`_release_examples`) and builds them as locals in `propose()`. History drafts therefore follow upstream's own copy path in `6b68b9b4:ignition/recipes/qwen38/speculative_model.py:310-360`:
     - copies are substituted on the locals in `propose()`;
     - the copy mask travels in `DraftProposalState`;
     - `verify()` passes `copy_offered_mask=` to `commit()`.
     Only the four graph inputs (`copy_token_*`, `copy_generation`) stay persistent storage. The fork's two copy-depth guards (`index < token_limits - 1` and the `(1 << offered) - 1` bound) move into the functional `run()` copy gate and `speculation.py`'s functional commit. `nemotron_h/bootstrap.py` follows the same rule, and docs use upstream's wording. If a hunk cannot be carried without designing new scratch placement, stop and record it as a discovery for the operator.
   - `b6b559de` and every later pin conflict: `pyproject.toml` takes the step-3 sha; `uv lock` regenerates the lock, and only `flashinfer-python` may change.
   - `e8ec2b13` (`qwen38_dflash`, removed later by `6594e779`): `recipes/__init__.py` lists every upstream recipe, `inkling2` included; `qwen38/decoder.py` feature taps and `nemotron_h/dflash.py` are a union. The later removal must apply without conflict.
   - Leftovers of the retired recipe that cannot run on upstream's functional round are dropped in a carry commit on `deploy`: `PreparedParallelDraftRound`'s `admission`/`draft_page_blocks`, whose validator is None while upstream's `_release_examples` calls `self.validator.functional()`, together with `tests/components/test_parallel_draft_admission.py`; and `qwen38/mixed_model.py`'s `absorption`/`speculative_round`/`feature_taps` hunks, with its docstring corrected. The `sampler=False` hunk (the mixed speculative export row) stays.
   - `20ea85a8` (TODO rename): upstream's new `docs/TODO.txt` entries fold into `docs/TODO.md`; `docs/core-format-design.md`, `docs/orbit.md` and `docs/orbit-proposals.md` take upstream's text with the `TODO.md` link.
   - `ae2c7318`, `2d5899c1`, `0ad9d25e`, `ae94472c`, `42c3f512`: union on upstream's temporaries and #164's frozen-prompt instructions.
   - `817d7f86` (warmup retention, retired 2026-11-04): resolve so the later revert applies and the final `ignition/preparation/capture.py` is byte-identical to `6b68b9b4`'s (`git diff e774a468 deploy -- ignition/preparation/capture.py` is empty today).
   - "Union" in these rules means purely additive hunks (arguments, list entries, tests); wherever upstream's temporaries changed the surrounding code, the fork's hunk is ported onto upstream's structure.
   - `a163b372` (Muse Glimmer):
     - `runtime_core.proto` keeps upstream's `PADDED_GRID_IMAGE_PREPARATION = 10` and `padded_grid = 10`, and adds `MUSE_GLIMMER_IMAGE_PREPARATION = 11`, `MuseGlimmerImageSettings muse_glimmer = 11` and the message.
     - `contract.rs` becomes `PaddedGrid = 10, MuseGlimmer = 11`.
     - The `helios/src/frontend/images/*` and `runtime_core/{describe,protobuf}.rs` profile arms are a union.
     - `conditioned.rs` stays exhaustive: upstream's `_ => 588` wildcard does not return, because it would silently restore the old PatchMerge width. Add `Profile::PaddedGrid => padded_grid::FEATURES`, add PaddedGrid to the interpolation match, and extend the coverage index arrays.
     - `runtime_core_pb2.py` is regenerated with a protoc whose gencode header matches `6b68b9b4`'s (`Protobuf Python Version: 7.35.1`).
     - `dflash2.py` and `dflash2_weights.py` take upstream's split. The projector-less hunks apply to the `Pooled*` classes only (`PooledDraftScratch`, `PooledPreparedDraft`, `PooledPreparedContext`), never to the Inductor `DraftScratch`.
     - The tests are a union.
   - `f33522d6` docs and `a1051e26` `test_mtp_only.py`: union.

4b. **Glimmer port**, as carry commits on `deploy` after the rebase, before the record commit. Upstream moved `PreparedGeneration`, `PreparedParallelDraftRound`, `PreparedMaintenance` and `PreparedPromptScoring` to a functional protocol and split the DFlash2 classes. Ten recipes follow it at `6b68b9b4`; `muse_spark`, which Glimmer borrows from, does not.
   - Text path: `muse_glimmer_nvfp4/decoder.py` takes the `6b68b9b4:ignition/recipes/gemma4/decoder.py` shape: `release_outputs()`, and `__call__(metadata)` returning `(hidden, logits, auxiliary, _)`. Prompt scoring uses upstream's functional `PreparedPromptScoring`, and `bootstrap.py` stops passing `buffers=` (lines 269, 282, 300).
   - Draft path: `muse_glimmer_nvfp4/dflash.py` imports the `Pooled*` aliases the way `6b68b9b4:ignition/recipes/muse_spark/dflash.py:12-13` does, and stops passing `buffers=`.
   - Spark modules Glimmer imports: `muse_spark/attention.py` (`PreparedAttention`, `AttentionScratch`), and `muse_spark/decoder.py` (`PreparedResume`, `SoftcappedProjection`) are ported to the same protocol. `muse_spark/dflash.py`'s `DraftForwards` is already on Pooled names upstream. `muse_spark/bootstrap.py` stays as upstream ships it; making Spark export is out of scope.
   - Tests: `tests/recipes/muse_glimmer_nvfp4` stays green without adding `buffers` back to any Inductor class. A new CPU case builds the Glimmer bootstrap against the real component signatures (no stubbed bootstrap) on the meta device, so a removed keyword fails on CPU.
   - Like every rebase resolution here, the port follows upstream's existing pattern and designs no new kernel, fusion, stream or scratch mechanism. FORK.md records the operator's overrule of the Opus 5.5 restriction for this fork.

   After the rebase, run these checks:
   - `git diff 6b68b9b4 deploy -- <file>` for every resolved file shows only fork hunks.
   - `git range-diff e774a468..backup/pre-rebase-20261114-deploy 6b68b9b4..deploy` is read for dropped or altered commits.
   - Each `patch/NN` is an ancestor of `deploy` and a descendant of `6b68b9b4`.
   - `git grep -n 'MuseGlimmer = 10\|MUSE_GLIMMER_IMAGE_PREPARATION = 10\|muse_glimmer = 10' deploy` prints nothing.
   - `git ls-tree -r deploy -- ignition/recipes/qwen38_dflash` prints nothing.
   - Every fork-only Python module imports, and a static check of fork-only calls against the rebased signatures (keyword names on calls into `ignition/components` and `ignition/preparation`) reports no unknown keyword. `py_compile` checks syntax only, so it cannot catch a removed keyword such as `buffers=` or `package_data=`.
5. **Record commit on `deploy`**, hook on, acknowledged against digest `ed22`:
   - README: stack base `6b68b9b4`, the History line, and every "Still needed" judged against `6b68b9b4` (replacing the "judged against `cbf21a12`" sentence).
   - README: the FlashInfer row is retired or re-pointed. The Glimmer row notes image number 11 and the port. It also states that a Glimmer core exported before this change is not rejected by a hand-run `flare run` on the new binary and gives invalid image results.
   - FORK.md: the 2026-11-14 base-move entry.
6. **Host CPU gate**, baseline-relative:
   - `uv sync --locked --extra prepare --extra test`.
   - One pytest per file, with a 14 GiB watchdog on the real PID, on the tip and on pristine `6b68b9b4`. The files are `tests/cli`, `tests/recipes/test_qwen38_bootstrap.py`, `tests/recipes/test_qwen38_mixed.py`, `tests/recipes/test_qwen38_speculative_state.py`, `tests/recipes/muse_glimmer_nvfp4`, `tests/recipes/test_mimo_v2.py`, `tests/components/test_parallel_draft*.py`, `tests/components/test_dflash2.py`, `tests/components/test_rotary_interpolation.py`, `tests/recipes/test_declarations.py`, `tests/preparation`, and `tests/reference/test_speculative_workload.py`.
   - Pass condition: no failure on the tip that is not also on pristine, except fork-recipe parameters of upstream tests that fail for every recipe on this host.
   - Glimmer number check: `ImageSettings.DESCRIPTOR.fields_by_name['muse_glimmer'].number == 11` and `ImagePreparationKind.Value('MUSE_GLIMMER_IMAGE_PREPARATION') == 11`.
7. **Image and Rust.** Run `~/LocalApps/vllm-env/build/flare-ignition/build.sh deploy` → `local/flare-ignition:<sha8>`. Then check:
   - The image's revision label equals `deploy`.
   - In-image `uv lock --check` passes.
   - `cargo check --manifest-path flare-cli/Cargo.toml` passes at the `patch/01`, `03`, `06` and `07` tips.
   - `cargo test` passes for `flare-cli` and for Helios (`--features runtime-core,server`), in a container from that image. This covers the Glimmer and padded-grid frontend tests.
8. **GPU 1** (`--gpus device=1`, `FLARE_RUN_CUDA_TESTS=1`, container from the step-7 image, 14 GiB cap). Run `tests/components/test_packed_speculation_cuda.py` (the history-drafts stop), the Glimmer CUDA tests under `tests/recipes/muse_glimmer_nvfp4` (fabricated-weight cases, which fit on 16 GB), `tests/recipes/test_qwen38_mixed_cuda.py`, `tests/components/test_dflash2.py` and `tests/preparation/test_opaque_operand_materialization_cuda.py`, with pristine `6b68b9b4` as the baseline for any failure.
9. **GPU 0 through the owner's console**, after the owner stops arm Q:
   1. `flare inspect` on the new image reads arm Q's entry `gb202-tp1-202611110138` and the Glimmer entry `gb202-tp1-202611080045`. The Glimmer entry is expected to describe its image profile as padded grid, which shows the recorded limitation.
   1b. Duplicate *2026-11-07 Flare Muse Glimmer c16* as *2026-11-14 Flare Muse Glimmer c16 Rebase4*, differing only in its image; the owner's original profile and package are never edited. Export it (Max Num Seqs 16, 1,048,576-token context), serve it, and check four things: one text request, one image request whose answer describes the image, the 1M passkey at depths 0.1, 0.5 and 0.9 (14/15 recorded at 256 reply tokens), and sixteen simultaneous streams completing. `flare inspect` of the new entry reports the Glimmer image profile.
   Sub-steps 2–4 moved to a follow-up change (Revisions, 2026-11-15 18:20).
   2. Duplicate *2026-11-10 Flare DFlash2 Q c16* as *2026-11-14 Flare DFlash2 Q Rebase4*, differing only in its image, and export it.
   3. Serve it and measure as `qualify/results/dflash2-paths-c16/README.md` does: `code_edit` at c1 and c8 (three trials, `qualify.dflash2_console_probe` and `qualify.dflash2_console_rate`), 16 clients (`llm_decode_bench.py`, three trials), and the 1M passkey at depths 0.1, 0.5 and 0.9.
   4. Compare against arm Q (555.9 / 436.7 / 1231.2, passkey 3/3), reporting any difference beyond ±8 %.
   5. Restore arm Q's profile, or the owner's choice.
10. **Publish and record.**
    - Push `git push --force-with-lease=<ref>:<backup sha> origin <ref>` one ref at a time for `patch/*` and `deploy`, then `git push -f origin base/current`.
    - Vault: the DFlash2 Recipe Progress Now block, Flare Fork Patch Notes, and the Muse Glimmer note (number 11, the port, the new entry, and that the old c16 package stays on `5ca5a9e7`).
    - Report upstream's `mimo_v2/bootstrap.py` `buffers=` defect and the unported `muse_spark` to the operator as upstream issues; filing them is the operator's call.
    - `~/LocalApps/vllm-env/build/README.md`: the image row.

Constraints carried (AGENTS.md, FORK.md, operator memory):
- no behavior change to Helios policy, contracts, recipes or the core format in any resolution, beyond the Glimmer number;
- exports and serving run in the container from a committed ref;
- no environment-variable switches;
- GPU 0 belongs to the owner's console instance;
- `main` is never merged into;
- no audit reports or receipts;
- resolutions carry both sides' code and design no new kernel, fusion, stream or scratch mechanism. FORK.md records the operator's overrule of the Opus 5.5 restriction for this fork; this change still designs nothing new in that area.

Not in scope:
- refreshing `pr/muse-glimmer` (renumbering and port), which is PR #229's own change;
- making upstream's Muse Spark recipe export, and MiMo history drafts (upstream defect);
- declaring `--draft-body-quantization` on `qwen38`;
- refreshing or closing any upstream PR.

## Revisions

- 2026-11-15 18:20, during apply of step 9: the operator asked "Can we separate the Qwen exploration into a follow-up instead of grouping it here?". Step 9's sub-steps 2–4 (export arm Q's copy, nine served trials, the 1M passkey, the comparison with 555.9 / 436.7 / 1231.2 and 3/3) leave this change for a follow-up change. The copy export *2026-11-14 Flare DFlash2 Q Rebase4* on `local/flare-ignition:3b4a1a2e`, running since 18:15, finishes at the operator's choice ("Let it finish" over "Stop it now"); its entry is an input to the follow-up, unverified by this change. This change keeps sub-step 1 (`flare inspect` of arm Q's existing entry, done) and the Glimmer checks. Carry commit `3b4a1a2e` stays on `deploy` and is published by step 10 with no GPU proof in this change beyond the copy export passing the model-tree construction where the first attempt stopped; the follow-up owns that proof. Arm Q's package on `f83d3120` is untouched and serves as before.
- 2026-11-15, during apply of step 9: at `6b68b9b4` upstream locks `transformers==5.23.0` while the `qwen38` recipe accepts only a Git install of revision `1d9654e7`, so the first console export of arm Q on the rebased image stopped at model-tree construction. The operator chose "Accept 5.23.0 (Recommended)" over "Pin the old revision" and "Leave Qwen on f83d3120" ("The goal is to get everything up to the latest, so I expect to have to re-ignite Qwen"). Carry commit `3b4a1a2e` widens the recipe's check; arm Q exports on `local/flare-ignition:3b4a1a2e`. The Glimmer export had already started on `local/flare-ignition:21e90706`, which differs only in that check, its tests and a README row, and is kept. After the operator freed GPU 0, the Glimmer checkpoint tests of step 8 ran there in a container before any export. The console exports are unseeded: no kernel folder is copied from an earlier image, so kernels are tuned and compiled on the new base.
- 2026-11-15, during apply of step 4b: three points of the Glimmer port changed. (1) The operator required that Glimmer be first-class without affecting Spark development, so Glimmer owns `muse_glimmer_nvfp4/attention.py` and `muse_spark/attention.py` returns to upstream's text; step 4b's in-place port of the Spark modules is superseded. (2) The `Pooled*` DFlash2 classes have no functional interface, so Glimmer's draft path uses `DFlash2Preparation` and the Inductor classes. (3) b12x operators need CUDA, so the CPU test is a dataflow test plus a keyword-binding test, not a bootstrap built on the meta device.
- 2026-11-15, at the operator's request ("pull the latest upstream and make any necessary adjustments to the spec"): upstream moved `29192ef7` → `6b68b9b4` (nine commits: #302 least-recently-used host prefix reclaim, #309 snapshot planning, GA102 platform recognition, two Windows fixes, peer-access diagnostics removal, version bumps to `v0.2.2`). Local `main` was fast-forwarded; `origin/main` is unchanged. The trial rebase stops on the same 15 commits and files, and no `ignition/`, proto, pin or design-checklist file changed, so the resolution rules, code citations (now written at `6b68b9b4`), the Glimmer port, the FlashInfer test, protoc 35.1 and digest `ed22` carry over unchanged. Base retargeted throughout.
- 2026-11-14, after the peer review: the bomb-squad check found that the Glimmer recipe cannot export on `29192ef7`. The operator chose "Port Glimmer in this rebase" over the recommended "Rebase without Glimmer exports", which added step 4b, the Glimmer GPU 1 tests and step 9's sub-step 1b. The peer verdict above covers the base move as first drafted; per the review rule it is not re-asked.
- 2026-11-14, at propose: upstream moved `d5c2bd14` → `29192ef7` between explore and propose (four commits, including Inkling 2 Small `8628b7c5`, which claims image number 10). The conflict prediction grew from 36 to 48 files. The operator chose "Glimmer moves to 11".

## External Review

### Peer Review Verdict

Quorum: achieved (weight 5/3, escalated to tier 2)
Certainty: 0.8833
Caller Agreement: 0.55
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.3
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.6
- deepseek-flash (deepseek-flash) — ok, agreement 0.85
- anthropic-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.9

### Review Feedback

Consensus feedback recorded verbatim:
- "Enable rerere, seeded from the probe's conflict map, but review its auto-resolutions rather than trusting them." — ACCEPTED in half: rerere stays enabled for this run and records only reviewed resolutions. REJECTED in half: no seeding from the probe, because the probe took the replayed side blindly and seeding would replay those blind resolutions.
- "Port the fork's intent onto upstream's Inductor planned-temporary-storage model … rather than taking a textual union." — ACCEPTED: this was already step 4's rule for the temporaries files. "Union" in step 4 now applies only to purely additive hunks (arguments, list entries, tests).
- "move Muse Glimmer to 11 in the .proto …, in the Rust contract_enum!, and in regenerated runtime_core_pb2.py (generated with the pinned protoc, never hand-edited). Keep Inkling/PADDED_GRID at 10." — ACCEPTED: already step 4.
- "Reject or route the old field-10 Glimmer core so it is never decoded as padded_grid; keep it on its old image and re-export Glimmer on the new image." — ACCEPTED in part: the core stays on its old image. REJECTED in part: no loader guard is added. The console already ties a cache entry to the image that exported it and cannot serve it on another (found 2026-11-10), so only a manual `flare run` or `flare inspect` can decode it on the new binary, and step 9 records what `inspect` shows. The operator chose no Glimmer re-export in this change.
- "do not cherry-pick the fork eviction commit; re-express its behaviors … on upstream's planned-JIT registry" — ACCEPTED: step 3 never cherry-picks onto `4fd836dc` as-is. It carries a hunk only if the fork's own test case for it fails on upstream's pin, and the carried hunks sit on upstream's registry.
- "Drop historical net-zero commits (reverted capture.py, qwen38_dflash add/delete) only after proving final-tree equivalence." — REJECTED: dropping them rewrites the recorded history, as no earlier base move has done. ACCEPTED as a check: step 4 now also requires `git ls-tree -r deploy -- ignition/recipes/qwen38_dflash` to be empty and `capture.py` to be byte-identical to upstream's.
- "Recompute the design-review digest and regenerate acknowledgment trailers on rewritten commits; do not bypass the hook." — REJECTED for replayed commits: a trailer certifies that a review was actually done, and rewriting 93 trailers to `ed22` without re-reviewing each would be a false acknowledgment (as on 2026-11-10). ACCEPTED for every commit this change writes, with the hook on and digest `ed22` (step 5).
- "Publish with --force-with-lease, moving base/current last." — ACCEPTED: step 10 pushes `base/current` after `deploy`.
- Certainty 0.8833 and caller agreement 0.55 are below the 0.9 floors — RECORDED: the peers agree with the draft's shape (backups, fast-forward-only mirror, FlashInfer decided first, port onto upstream's temporaries, Glimmer at 11, range-diff and fork-only diff checks, CPU then GPU 1 then one GPU 0 export). The gap is the four points rejected above (probe-seeded rerere, dropping net-zero commits, rewriting trailers, a loader guard), each kept for a stated reason. No number is re-asked.

## Bomb-squad findings

- ignition/recipes/muse_glimmer_nvfp4/bootstrap.py:269 — `buffers=` to `PreparedGeneration`/`PreparedParallelDraftRound`/`PreparedMaintenance` (also :282, :300) and the decoder's old `__call__(self)` / `self.scoring()` protocol; the recipe cannot export on `29192ef7`, the CPU test stubs the bootstrap, and upstream's own `muse_spark` is broken the same way — ACCEPTED: the operator chose "Port Glimmer in this rebase"; step 4b ports it with an unstubbed CPU bootstrap test and a GPU 0 export; Blast Radius 25 corrected.
- ignition/components/parallel_draft.py (deploy `propose` `substitute_copies(self.drafts, …)`, commit `copy_offered_mask=self.copy_mask`) — upstream's functional round releases `drafts` etc. after prepare and its functional commit raises without a per-call `copy_offered_mask=`; dropping the mask silently credits copied depths to the drafter — ACCEPTED: step 4's history-drafts rule now follows upstream's copy path (locals in `propose()`, mask in `DraftProposalState`, `copy_offered_mask=` in `verify()`).
- ignition/recipes/muse_glimmer_nvfp4/dflash.py:10 — the old DFlash2 class names are now Inductor variants; the pool lives in `Pooled*` — ACCEPTED: projector-less hunks go to the `Pooled*` classes only; Glimmer imports Pooled aliases like `muse_spark`.
- FlashInfer step 3 — case 1 reads `_PLANNED_JITS` (absent on `4fd836dc`), and cases 2 and 3 must fail on `4fd836dc`, so the planned test could not retire the row — ACCEPTED: case 1 is rewritten to upstream's registry, and cases 2 and 3 are carried on `evict-planned-jits-4fd836dc`; a re-pointed row is expected.
- /tmp/probe-29192ef7.stops — 15 stops is a lower bound; about ten more commits will re-stop under real resolutions — ACCEPTED: Approach preamble; each re-stop takes the rule of the stop that first touched its file; rerere records only reviewed resolutions.
- ignition/components/parallel_draft.py:111 — `admission`/`draft_page_blocks` live only in a test since `6594e779` and cannot run on upstream's round — ACCEPTED: dropped with its test in a carry commit (proposal Assumed line).
- old Glimmer cores — field 10 decodes as `padded_grid` without error and passes the identifier check — ACCEPTED: README, FORK.md and the vault note state it; a loader guard is still not added (Review Feedback); the port's new export replaces them on the rebased image.
- `--exec py_compile` — syntax only, blind to removed keywords in non-conflicting files — ACCEPTED: step 4's post-rebase import of every fork-only module plus a static keyword check of fork-only calls.
- helios/src/frontend/images/conditioned.rs:191 — a union restoring upstream's `_ => 588` wildcard silently reverts the PatchMerge width fix — ACCEPTED: the match stays exhaustive with a PaddedGrid arm.
- ignition/recipes/qwen38/speculative_model.py:285 — the fork's copy guards target a loop that no longer exists — ACCEPTED: both guards move into the functional paths (step 4).
- ignition/recipes/mimo_v2/bootstrap.py:290 (upstream) — still passes the removed `buffers=`, so MiMo history drafts cannot run — ACCEPTED as a report: out of scope, raised with the operator as an upstream issue; MiMo history drafts are not counted as exercised.
- ignition/recipes/qwen38/mixed_model.py:43 — retired-recipe factory parameters and a docstring naming removed methods — ACCEPTED: dropped in the carry commit, keeping `sampler=False`.
- `runtime_core_pb2.py` regeneration — no protoc on the host or in `.venv` — ACCEPTED: step 3 sub-step 5 names the protoc 35.1 release.
- hunt completed — clean: fork-only imports resolve; Glimmer's number appears only symbolically in Python and once in Rust (`contract.rs:68`); `ProcessedImages` fields build; no other enum collisions; the fork CLI files and `main.rs` merge; the qualify tools and #164; `test_checkpoint_source.py`; the optional `graph_resources`; `inductor_tuning.py` and `helios/src/model/packing.rs`; Glimmer's DFlash2 checkpoint loader.
