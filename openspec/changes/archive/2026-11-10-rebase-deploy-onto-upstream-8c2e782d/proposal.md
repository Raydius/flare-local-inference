# Rebase the patch stack and deploy onto upstream e774a468

Tier: 3
Type: feature

## Stated Requirements

- "Help me rebase with the upstream changes"
- Target base chosen: "Latest 8c2e782d (Recommended)" (over `d2b5c5f3`, the commit before the `flare/` → `flare-cli/` split)
- "are all of the patches still relevant?"
- On the fork's `qwen38_dflash` recipe versus upstream's own Flash Next DFlash2 (`d68ed8f5`): "ok carry it as-is."
- "Ok let’s proceed with all recommended steps.  The current server has been stopped."
- Upstream moved to `e774a468` (17 commits past `8c2e782d`) before apply; base chosen: "Retarget to e774a468" (over "Stay on 8c2e782d (Recommended)")

Assumed: `main` stays the upstream mirror and only fast-forwards to `e774a468`; `base/current`, `patch/00`…`patch/07` and `deploy` move through `scripts/fork/sync.sh` and `scripts/fork/rebase.sh e774a468`, as on 2026-11-04.
Assumed: every conflict keeps both sides' behavior on upstream's new structure (`flare-cli/`, "core" naming, `runtime_core`); the code as each side runs it today is the truth.
Assumed: a patch retires only where upstream already does its job: the FlashInfer MoE fragment guard (`49fe5449`, upstream FlashInfer #71) retires; the opaque-write copy rule (`80946209`) retires only if its own reproducer tests pass on the rebased tip without it, and is ported otherwise.
Assumed: the FlashInfer pin becomes upstream's `85ea6e76` plus the eviction fix (`17bd2c8a`, FlashInfer PR #67) on a new `Raydius/flashinfer` branch.
Assumed: replayed commits keep their historical `Design-Signed-Off-By` trailers (the fork is not upstream-bound, FORK.md); only commits this change writes are acknowledged against upstream's new checklist (`design-review=cfcf`).
Assumed: proof is CPU suites (baseline-relative), a containerized image build, the current DFlash2 package served unchanged on the new binary, and one re-export of the DFlash2 yarn4 package at the operator's configuration (fixed-7, concurrency 8) served at c1 and c8; GPU 0 is freed by the owner from the console.
Assumed: the open PR branches (`pr/muse-glimmer`, `pr/rope-scaling`, `pr/mixed-speculative-export`) and the detached worktrees are out of scope.
Assumed: force-pushing `patch/*`, `deploy` and `base/current` to `origin` happens with a lease against the backup tags at close.

## Why

Upstream is 48 commits past the stack's base (`585159c9` → `e774a468`): it split the terminal frontend into `flare-cli/` over a shared Flare library, renamed packages to cores (`runtime_package` → `runtime_core`), added Kimi K3.1, moved its FlashInfer pin twice, and changed the design checklist. The fork's served image cannot pick up any of it until the stack moves, and a throwaway probe rebase stops on 12 of `deploy`'s 79 commits.

## What Changes

- `main` fast-forwards to `e774a468`; `base/current` moves from `585159c9` to `e774a468`; `patch/00`…`patch/07` and `deploy` replay (79 non-merge commits; the two merge commits flatten).
- Twelve conflicting commits resolve keeping both sides: `flare-cli/src/main.rs` (patch/03), qwen38 `_identity`/declaration (patch/06, taking PR #212's already-resolved `config=` shape), "package"→"core" docs, TODO.txt→TODO.md, `pyproject.toml`/`uv.lock`, the `qwen38_dflash` recipe's shared-file hunks, `ignition/cli/{preparation,bench}.py`, Muse Glimmer's protobuf additions onto `runtime_core.proto` (regenerated `runtime_core_pb2.py`), `test_checkpoint_source.py`, README.
- FlashInfer: `Raydius/flashinfer` gains a branch with `17bd2c8a` on `85ea6e76`; `49fe5449` (MoE guard) is dropped as superseded by FlashInfer #71.
- The opaque-write copy rule (`80946209`) is dropped if its reproducers pass on the rebased tip, otherwise ported onto upstream's restructured `_lower_storage`.
- The console image recipe (`~/LocalApps/vllm-env/build/flare-ignition/Dockerfile`) builds the `flare` binary from `flare-cli/`.
- README patch notes and FORK.md record the move, the per-patch verdicts against `e774a468`, and the retirements.
- No behavior change to any recipe, Helios policy, contract, or package format beyond what upstream brings.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None — a rebase carries every requirement (`skip_specs: true`).

## Impact

- Branches: `main`, `base/current`, `patch/00`…`patch/07`, `deploy` (backup tags first); `Raydius/flashinfer` gains one branch.
- Build: the console image recipe path; the served image and the DFlash2 yarn4 package are rebuilt and re-exported.
- Upstream PRs #211, #212, #229 and FlashInfer #67 are untouched.

## Blast Radius

1. scripts/fork/rebase.sh:77 — the `git rebase --onto … --update-refs` that moves the stack.
2. grepped 'manifest-path flare/Cargo.toml' in the console recipe (~/LocalApps/vllm-env/build/flare-ignition/Dockerfile, line 46, outside this repo) — builds the binary from `flare/`; upstream `fa158284` moved it to `flare-cli/Cargo.toml`.
3. pyproject.toml:25 — FlashInfer pin `Raydius/flashinfer@49fe5449`; upstream pins `local-inference-lab/flashinfer@85ea6e76`.
4. uv.lock:403 — the locked FlashInfer source for the same pin (probe stop 4).
5. flare-cli/src/main.rs:34 — patch/03's `serving_overrides` module (upstream moved `flare/src/main.rs` here, beside `use flare::windows`) (probe stop 1).
6. ignition/recipes/qwen38/bootstrap.py:218 — patch/06's `_identity(…, rope_scaling)` against upstream's `config=`/`draft_config=` identity (probe stop 2); PR #212 already resolved this on `91cfb369`.
7. ignition/recipes/qwen38/recipe.py:10 — patch/06's `RopeScaling` import against upstream's `SpeculativeMethod` / `WEIGHT_BYTES` imports.
8. tests/recipes/test_qwen38_bootstrap.py:333 — patch/06's identity test beside upstream's MoE-decode-A16 test.
9. docs/cli-design.md:187 — history-drafts wording (`95752d66`) against upstream's package→core rename (probe stop 3); same in docs/model-preparation.md and docs/mtp-design.md.
10. docs/TODO.md:217 — the FlashInfer follow-ups `e6ccf746` added to `docs/TODO.txt`, which upstream still edits.
11. ignition/recipes/__init__.py:29 — `qwen38_dflash` registration (`8046cf12`) beside upstream's recipe list (probe stop 5).
12. ignition/preparation/inductor_tuning.py:155 — `8046cf12`'s `InductorSelections` hunk.
13. ignition/preparation/rebuild_manifest.py:182 — `8046cf12`'s `declared_views` hunk; tests/preparation/test_rebuild_manifest.py with it.
14. ignition/recipes/qwen38/decoder.py:256 — `8046cf12`'s `PreparedDecoder` hunks (also :396).
15. ignition/recipes/qwen38/state.py:34 — `8046cf12`'s `HybridStateLayout` / `HybridStatePool` hunks (also :77, :126).
16. docs/core-format-design.md:5 — the TODO.md link (`47ccba29`), in the file upstream renamed from `docs/package-format-design.md` (probe stop 6).
17. ignition/preparation/compilation.py:490 — the opaque-write rule (`80946209`) against upstream `fa0eadad`'s deferred/opaque pass in `_lower_storage` (probe stop 7).
18. ignition/cli/preparation.py:100 — `--recipe` / `--draft-model` (`2c8ba959`) (probe stop 8).
19. ignition/cli/bench.py:134 — bench `--rope-parameters` forwarding (`795d753f`) (probe stop 9).
20. helios/proto/runtime_core.proto:368 — Muse Glimmer's `MUSE_GLIMMER_IMAGE_PREPARATION = 10` / `muse_glimmer = 10`; upstream renamed the file to `runtime_core.proto` and `runtime_package_pb2.py` to `runtime_core_pb2.py` (probe stop 10, modify/delete); grepped upstream's enum and oneof — value 10 is free.
21. tests/preparation/test_checkpoint_source.py:19 — import hunk of `a6b06bdb` (probe stop 11).
22. README.md:6 — patch notes stack base and history; README.md:44 the opaque-write row; README.md:50 the FlashInfer row; the README body conflicts with upstream's Hyperflare rename and `docs/images/hyperflare.png` (probe stop 12).
23. FORK.md:103 — the base/current history and the move procedure.
24. .githooks/commit-msg:59 — the acknowledgment check; upstream `fc2c914c` changed `docs/design-review.md`, so a conflicted commit's `--continue` meets a stale `a38f` trailer.
25. grepped upstream `e774a468` for each patch's feature (`generation_defaults`, `max-context-tokens`, `--chat-template`, `RopeScaling`, `--recipe`, `--draft-model`, `substitute_copies`, `AcceptanceProfile::observe` copy mask, mixed-prefill DFlash2) — none present; those patches stay.
26. ignition/components/dflash2.py:280 — Muse Glimmer's projector-less convolutions (`9834def8`) against upstream `98741a3c`'s `attention_factory` / `convolution_runtime` in `PreparedDraft` (probe stop 10 at `e774a468`).
27. ignition/recipes/nemotron_h/bootstrap.py:191 — history drafts' copy-buffer allocation (`95752d66`) against upstream's `serving_modes(drafting, config.mtp_only)` (probe stop 3 at `e774a468`).
