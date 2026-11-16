# Fast-forward main and rebase the patch stack and deploy onto upstream 6b68b9b4

Tier: 3
Type: feature

## Stated Requirements

- "Review the upstream from Nvidia.  Rebase main and determine if we should merge it into deploy."
- "Go ahead and begin the rebase"
- "Begin with updating main"
- "pull the latest upstream and make any necessary adjustments to the spec"
- Muse Glimmer's image-preparation number, which upstream's Inkling recipe now also claims: "Glimmer moves to 11 (Recommended)" (over "Glimmer keeps 10")
- Muse Glimmer, which cannot export on `6b68b9b4`: "Port Glimmer in this rebase" (over "Rebase without Glimmer exports")
- "Can we separate the Qwen exploration into a follow-up instead of grouping it here?"; for the arm Q copy export already running when this was asked: "Let it finish" (over "Stop it now")

Assumed: "updating main" is `scripts/fork/sync.sh`. It fast-forwards the mirror `main` to `upstream/main` and pushes it to `origin`. `main` is never merged into or landed on.
Assumed: the base is `upstream/main` as it stands when apply runs (`6b68b9b4` as of 2026-11-15, 13 commits past the `d5c2bd14` reviewed in explore, including Inkling 2 Small NVFP4 for TP4, shared computed inputs across prepared operator calls, #298 prefix-cache reuse under resource pressure, #302 least-recently-used host prefix reclaim, and #309 cheaper snapshot planning). Local `main` was fast-forwarded to `6b68b9b4` during the 2026-11-15 revision; the push to `origin` stays in task 1.1. If upstream moves again before the stack moves, the plan is retargeted and recorded under design.md Revisions.
Assumed: `base/current`, `patch/00`…`patch/07` and `deploy` move with `rebase.sh`'s command (`git rebase --onto <base> base/current deploy --update-refs`), as they did on 2026-11-10. Each conflict keeps both sides' behavior on upstream's new structure, and the code as each side runs it today is the truth.
Assumed: the FlashInfer pin row retires if the fork's three eviction test cases pass on upstream's new pin `4fd836dc`. Upstream FlashInfer `810c0849f` added the same planned-JIT registry. If any case fails, the hunks it still needs are carried on a new `Raydius/flashinfer` branch cut from `4fd836dc`.
Assumed: Muse Glimmer moves to `ImagePreparationKind` value 11 and `ImageSettings` field 11, everywhere the number is written: the proto, Helios's contract enum, and the regenerated `runtime_core_pb2.py`. The owner's Glimmer c16 package (`gb202-tp1-202611080045`) keeps serving on its current image; a new Glimmer export on the rebased image records 11.
Assumed: the Glimmer port moves the recipe's text and draft paths onto upstream's functional generation and draft-round protocol, the shape `6b68b9b4`'s `gemma4` decoder uses. The `muse_spark` modules Glimmer imports (`attention.py`, `decoder.py`'s `PreparedResume` and `SoftcappedProjection`, `dflash.py`'s `DraftForwards`) are ported with it, because upstream left them on the removed interfaces. `muse_spark`'s own bootstrap is left as upstream ships it.
Assumed: leftovers of the retired `qwen38_dflash` recipe that cannot run on upstream's functional draft round are dropped, not carried: `PreparedParallelDraftRound`'s `admission`/`draft_page_blocks` with `tests/components/test_parallel_draft_admission.py`, and `qwen38/mixed_model.py`'s `absorption`/`speculative_round`/`feature_taps` hunks. The `sampler=False` mixed-export hunk stays.
Assumed: replayed commits keep their historical `Design-Signed-Off-By` trailers. Only commits this change writes are acknowledged, against upstream's new checklist digest `ed22`.
Assumed: the proof is five checks:
- baseline-relative CPU suites;
- a containerized image build;
- the served arm Q core (`gb202-tp1-202611110138`) and the Glimmer c16 core read by `flare inspect` on the new image (the console cannot serve an existing entry on a different image; found on 2026-11-10);
- (moved to a follow-up change on 2026-11-15; design.md Revisions) one re-export of arm Q at the operator's configuration, served at c1, c8, 16 clients, and the 1M passkey;
- one Glimmer export at its c16 configuration (Max Num Seqs 16, 1,048,576-token context), served with a text request, an image request and the 1M passkey;
- GPU 0, which the owner frees from the console.
Assumed: these are out of scope:
- the open PR branches (`pr/muse-glimmer`, which needs the same renumbering and port, `pr/rope-scaling` and `pr/mixed-speculative-export`) and the detached worktrees;
- making upstream's own Muse Spark recipe export (its bootstrap), and MiMo history drafts, which upstream's `mimo_v2/bootstrap.py` breaks with a removed `buffers=` keyword.
- declaring upstream's `--draft-body-quantization` on `qwen38`.
Assumed: `patch/*`, `deploy` and `base/current` are force-pushed to `origin` with a lease against backup tags, at the end of apply.

## Why

`deploy` sits on upstream `e774a468`, 75 commits behind `upstream/main` `6b68b9b4`. Several of those commits bear directly on this fork's work:
- three KV and capacity fixes (#253, #286, #285) bear on the 1M-context measurements;
- upstream added native MXFP8 DFlash2 bodies (#291);
- upstream changed the DFlash2 embedding layout (#250);
- Inductor now plans model temporaries;
- #298 preserves prefix-cache reuse under resource pressure, and #302 reclaims host prefix rows in least-recently-used order;
- the FlashInfer pin moved twice.

The served image picks up none of this until the stack moves. A throwaway probe rebase stops on 15 of `deploy`'s 93 commits.

## What Changes

- `main` fast-forwards to `6b68b9b4` (done locally on 2026-11-15) and is pushed. `base/current` moves from `e774a468` to `6b68b9b4`. `patch/00`…`patch/07` and `deploy` replay all 93 non-merge commits.
- Fifteen conflicting commits are resolved, keeping both sides on upstream's structure. Most of the conflicts are with upstream's Inductor-planned temporaries (`4a24d78a`, `2b38bf9d`, `16bb710f`). The rest fall in four areas: the qwen38 YaRN declaration (patch/06), history drafts on the shared parallel-draft round, the Muse Glimmer image profile beside Inkling's padded-grid profile, and the FlashInfer pin lines.
- Muse Glimmer is ported onto upstream's functional generation, draft-round and pooled DFlash2 classes so it exports on `6b68b9b4`, together with the `muse_spark` modules it imports.
- History drafts follow upstream's own copy path: copies are substituted on the round's local tensors, the copy mask travels in `DraftProposalState`, and `commit()` receives `copy_offered_mask=`. The fork's two copy-depth guards move into the functional paths.
- Muse Glimmer renumbers from 10 to 11 in `helios/proto/runtime_core.proto`, `helios/src/model/contract.rs`, and the regenerated `ignition/preparation/runtime_core_pb2.py`. Upstream's `PADDED_GRID_IMAGE_PREPARATION = 10` and `padded_grid = 10` stay as upstream wrote them.
- FlashInfer is pinned either to upstream's `4fd836dc`, which retires the pin row, or to a `Raydius/flashinfer` branch on `4fd836dc` carrying only the hunks still needed. `uv.lock` is regenerated.
- README patch notes and FORK.md record the base move, each "Still needed" verdict judged against `6b68b9b4`, any retirements, and the Glimmer renumbering.
- No behavior change to any recipe, Helios policy, contract, or core format beyond what upstream brings and the Glimmer number.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. A rebase carries every requirement (`skip_specs: true`).

## Impact

- Branches: `main`, `base/current`, `patch/00`…`patch/07`, `deploy`, with backup tags created first. `Raydius/flashinfer` gains at most one branch.
- Cores: existing Qwen cores keep loading. Upstream's proto additions are additive and the execution ABI is unchanged. Muse Glimmer cores exported before this change record image number 10, and on the rebased binary that number means Inkling's padded grid. A hand-run `flare run` of one is not rejected and gives invalid image results, so they stay on the image that exported them; the new Glimmer export replaces them on the rebased image.
- Build: the console image recipe is unchanged (it already builds `flare-cli/`). The served image is rebuilt; arm Q's re-export and measurement belong to a follow-up change.
- Upstream PRs #211, #212, #229 and FlashInfer #67 are untouched. If the pin row retires, the report recommends closing #67; closing it is the operator's call.

## Blast Radius

1. scripts/fork/sync.sh:42 — the in-place fast-forward of `main` to `upstream/main`; refuses a non-fast-forward.
2. scripts/fork/rebase.sh:77 — the `git rebase --onto … --update-refs` that moves the stack.
3. grepped '^[0-9a-f]' in the throwaway probe log `/tmp/probe-6b68b9b4.stops` (`e774a468..deploy` replayed onto `6b68b9b4`, rerere off, hooks off, replayed side taken) — 15 stopping commits; sites 4–24 are their files.
4. flare-cli/src/cli.rs:482 — patch/01 `--generation-defaults` beside upstream's new CLI arguments (probe stop `a70349b5`).
5. ignition/recipes/declaration.py:70 — patch/06 `RopeScaling` beside upstream #291 `draft_body_quantizations` and #239 (stop `bc37825c`).
6. ignition/recipes/qwen38/bootstrap.py:97 — patch/06 `rope_scaling` identity against upstream #250's one-layout DFlash2 embedding binding and the Inductor temporaries (stop `bc37825c`).
7. ignition/recipes/qwen38/attention.py:27 — patch/06 scaled rotary beside Inductor-planned temporaries (stop `bc37825c`); with it ignition/components/rotary.py:110, ignition/planning.py:50, tests/cli/test_commands.py, tests/cli/test_models.py, tests/recipes/test_qwen38_bootstrap.py.
8. docs/TODO.md:1 — renamed from `docs/TODO.txt` by the fork; upstream's new `docs/TODO.txt` entries fold in at the rename commit (stops `bc37825c`, `20ea85a8`).
9. ignition/components/parallel_draft.py:59 — history drafts' `substitute_copies` against Inkling `8628b7c5` and Inductor temporaries (stop `e6320977`); with it ignition/components/packed_speculation.py:82 and ignition/components/speculation.py:199.
10. ignition/recipes/nemotron_h/bootstrap.py:201 — history drafts' `allocate_copy_buffers` (stop `e6320977`).
11. ignition/recipes/qwen38/speculative_model.py:58 — history-draft copy inputs (stop `e6320977`); tests/components/test_packed_speculation_cuda.py, tests/recipes/test_mimo_v2.py, tests/recipes/test_qwen38_speculative_state.py with it; docs/cli-design.md, docs/mtp-design.md wording.
12. pyproject.toml:25 — FlashInfer pin `Raydius/flashinfer@ef6d6592`; upstream pins `local-inference-lab/flashinfer@4fd836dc` (stop `b6b559de`).
13. uv.lock:403 — the locked FlashInfer source for the same pin (stop `b6b559de`).
14. grepped '_PLANNED_TRITON_JITS' and 'lru_cache' in `~/dev/flashinfer` at `4fd836dc` (outside this repo) — upstream `810c0849f` added the planned-JIT registry in `_lib/compile_plan.py` line 58, where the fork's `_PLANNED_JITS` replay conflicts; the fork's `_run_job` stub rejection (`_lib/compile_pool.py`) and `compile_compress`'s `lru_cache` → `program_cache` (`mla_compress/_cute.py` line 279) are not in `4fd836dc`.
15. ignition/recipes/__init__.py:37 — recipe list beside upstream's new `inkling2` registration (stop `e8ec2b13`); ignition/recipes/qwen38/decoder.py:204 feature taps in the same stop.
16. docs/core-format-design.md:5, docs/orbit.md:804, docs/orbit-proposals.md:6 — the `TODO.md` link rewrite against upstream edits (stop `20ea85a8`).
17. ignition/recipes/qwen38/mixed_state.py:17 — mixed prefill state beside Inductor temporaries (stop `ae2c7318`); tests/recipes/test_qwen38_mixed.py.
18. ignition/preparation/execution.py:77 — mixed-round proposal depth (patch row "Mixed speculative export") against Inductor-planned temporaries (stop `2d5899c1`).
19. qualify/speculative_workload.py:280 — the served DFlash2/MTP code_edit comparison against upstream #164 frozen-prompt instructions (stop `0ad9d25e`); tests/reference/test_speculative_workload.py.
20. ignition/cli/serve.py:131 — `--recipe` / `--draft-model` (stop `ae94472c`); ignition/cli/bench.py:145 `--rope-parameters` forwarding (stop `42c3f512`).
21. ignition/preparation/capture.py:70 — the warmup-retention commit `817d7f86`, retired 2026-11-04 (README.md:90); its stop is historical and the later revert must leave upstream's retention line (line 88 at `6b68b9b4`) unchanged.
22. helios/proto/runtime_core.proto:368 and :500 — Muse Glimmer `MUSE_GLIMMER_IMAGE_PREPARATION = 10` / `muse_glimmer = 10` against upstream `8628b7c5`'s `PADDED_GRID_IMAGE_PREPARATION = 10` / `padded_grid = 10` (stop `a163b372`); Glimmer moves to 11.
23. helios/src/model/contract.rs:68 — `contract_enum!(ImagePreparationKind { … MuseGlimmer = 10 })`, the Rust mirror of the same number; with it helios/src/frontend/images/mod.rs:198, helios/src/frontend/images/conditioned.rs:191, helios/src/frontend/images/packing.rs:97, helios/src/runtime_core/describe.rs:287, helios/src/runtime_core/protobuf.rs:161 (Glimmer and PaddedGrid profile arms side by side), and ignition/preparation/runtime_core_pb2.py:27 (regenerated).
24. ignition/components/dflash2.py:38 and ignition/components/dflash2_weights.py:49 — Glimmer's projector-less draft layers against `16bb710f`/`cb9f7f55` temporaries (stop `a163b372`); tests/cli/test_planning.py, tests/preparation/test_frontend_metadata.py; docs/model-preparation.md (stop `f33522d6`); tests/cli/test_mtp_only.py:111 (stop `a1051e26`).
25. ignition/recipes/muse_glimmer_nvfp4/recipe.py:12 — `prepare(…, core_storage=, ranks=)` is unchanged at `6b68b9b4`, but the recipe below it is not exportable (entries 31–33); the 2026-11-10 `package_data=` miss came from a rename the conflict files did not show, and this one is the same class.
26. .githooks/commit-msg:62 — the acknowledgment digest; `docs/design-review.md` changed upstream, digest `cfcf` → `ed22`.
27. README.md:6-10 — stack base, history line and the "judged against `cbf21a12`" sentence; README.md:47-49 the FlashInfer pin row.
28. FORK.md:103 — the base/current history entry.
29. grepped upstream `6b68b9b4` for each active patch's feature (`generation_defaults`, `serving_overrides`, `RopeScaling`, `--rope-parameters`, `muse_glimmer_nvfp4`, `substitute_copies`, the `AcceptanceProfile::observe` copy mask, `sampler=False` mixed rounds, `--recipe`/`--draft-model`) — none present; those patches stay.
30. grepped 'cargo build' in the console recipe `~/LocalApps/vllm-env/build/flare-ignition/Dockerfile` (outside this repo, line 46) — already builds `flare-cli/Cargo.toml`; upstream changed only `packaging/` files the recipe does not use; no change.
31. ignition/recipes/muse_glimmer_nvfp4/bootstrap.py:269 — `buffers=` to `PreparedGeneration` (also :282 `PreparedParallelDraftRound`, :300 `PreparedMaintenance`); none accepts it at `6b68b9b4` (ignition/components/paged_generation.py:13 and ignition/components/parallel_draft.py:53 there), so export raises TypeError; the CPU test stubs the bootstrap (tests/recipes/muse_glimmer_nvfp4/test_preparation.py:85).
32. ignition/recipes/muse_glimmer_nvfp4/decoder.py:108 — `__call__(self)`; upstream's `PreparedGeneration` calls `decoder.release_outputs()` and `decoder(metadata)` (ignition/components/paged_generation.py:66 and :82 at `6b68b9b4`); ignition/recipes/muse_glimmer_nvfp4/decoder.py:119 calls `self.scoring()`, which upstream removed from `PreparedPromptScoring`.
33. ignition/recipes/muse_glimmer_nvfp4/dflash.py:10 — imports `DraftScratch`, `PreparedContext`, `PreparedDraft`, which upstream turned into Inductor variants; the buffer pool now lives in `PooledDraftScratch`/`PooledPreparedDraft`/`PooledPreparedContext`, and `muse_spark/dflash.py` moved to them; ignition/recipes/muse_glimmer_nvfp4/dflash.py:35 passes `buffers`.
34. ignition/recipes/muse_spark/attention.py:62, ignition/recipes/muse_spark/decoder.py:195 and :208, ignition/recipes/muse_spark/dflash.py:28 — the Spark classes Glimmer imports; upstream left them on the removed interfaces (ignition/recipes/muse_spark/bootstrap.py:194 still passes `buffers=` at `6b68b9b4`).
35. ignition/components/parallel_draft.py:111 — `admission`/`draft_page_blocks`, live only in tests/components/test_parallel_draft_admission.py:1 since `6594e779`; upstream's `_release_examples` calls `self.validator.functional()` unconditionally, so the path cannot run.
36. ignition/components/speculation.py:390 — `copy_offered_mask` at commit construction; upstream's functional commit raises unless each call passes it (grepped `copy_offered_mask` in `6b68b9b4`'s `speculation.py`, lines 640-642), and dropping it silently credits copied depths to the drafter's acceptance curve.
37. ignition/recipes/qwen38/mixed_model.py:43 — `absorption`/`speculative_round`/`feature_taps` hunks of the retired recipe, and a docstring naming `begin()`/`finish()`, which upstream removed.
38. helios/src/frontend/images/conditioned.rs:191 — the fork's exhaustive feature-width match (the PatchMerge width fix); upstream's `_ => 588` wildcard must not return in the union.
39. ignition/recipes/mimo_v2/bootstrap.py:290 — at `6b68b9b4` upstream still passes the removed `buffers=` to `PreparedParallelDraftRound`; an upstream defect, so MiMo history drafts are not exercised.
40. grepped the probe for later commits touching already-stopped files — 21 replayed cleanly only because the probe took the replayed side; likely re-stops: `e8ec2b13` (`parallel_draft.py`), `855544bf` (`dflash2.py`, `dflash2_weights.py`, `packing.rs`), `565b5936` (`qwen38/decoder.py`), `5a58ba70` (`capture.py`), `1d1c12ef`/`ae94472c`/`f83d3120` (`declaration.py`), `4badb2a0`, `d7adc6fc`, `612598e8`.
41. grepped `git diff --name-only 29192ef7 6b68b9b4` against the fork's changed files (`e774a468..deploy`) — the nine commits past `29192ef7` overlap only flare-cli/src/cli.rs:482, docs/cli-design.md and docs/core-format-design.md:5, all already conflict stops; they change no `ignition/` file, `helios/proto`, `pyproject.toml`, `uv.lock` or `docs/design-review.md`; the trial rebase onto `6b68b9b4` (`/tmp/probe-6b68b9b4.stops`) stops on the same 15 commits and files.
