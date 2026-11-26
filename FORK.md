# Fork maintenance — local Flare builds

This fork carries local changes as a **rebaseable patch stack** on top of
`NVIDIA/flare-local-inference`, to produce local `flare` images for the vLLM console
(`~/LocalApps/vllm-env`). The doctrine is the same as `~/dev/vllm`'s
(`git -C ~/dev/vllm show patch/00-fork-tooling:FORK.md`); this file is the Flare
specifics and the live patch index.

**Scope (operator decision, 2026-10-20):** useful local builds first. Upstreaming to
NVIDIA is out of scope until a patch proves worth it, so the upstream contribution gates
in `AGENTS.md` (maintainer-request scope, the `.githooks/commit-msg` design-review hook,
the Claude Opus 5.5 kernel/optimization ban) do not apply here — the operator overruled
them. The hook is nonetheless enabled in this clone (`core.hooksPath=.githooks` in
`.git/config`), so every commit needs one `Design-Signed-Off-By` acknowledgment as
`docs/design-review.md` describes. Patches that prove worth it now go upstream as PRs from
`pr/*` branches cut on `upstream/main` (#211 and #212, opened 2026-11-04/05; #211 closed
2026-11-18 as superseded by upstream, its patch retired 2026-11-19).
`AGENTS.md`'s architecture and package-compatibility rules still describe how Flare works;
break them knowingly, not by accident.

## Remotes

| Remote     | URL                                                  | Role                      |
| ---------- | ---------------------------------------------------- | ------------------------- |
| `origin`   | `git@github.com:Raydius/flare-local-inference.git`   | our fork (push here)      |
| `upstream` | `https://github.com/NVIDIA/flare-local-inference.git` | NVIDIA (fetch-only; push URL is `DISABLE`); GitHub redirects it to `NVIDIA/spark-local-inference` |

## Branch model

```text
base/current (tag → an upstream commit)
  └─ patch/00-fork-tooling      # this file + scripts/fork/ (never upstreamed)
       └─ patch/01-<topic>       # one logical change each
            └─ patch/NN-<topic>  # topmost patch = what you build
                 └─ deploy           # README patch notes + DFlash2 work; GitHub default branch
```

- **`deploy`** — the patch stack, the commits that keep README's patch notes current, and
  the DFlash2 work merged above them (2026-11-04; it has no `patch/NN` branches and is
  inventoried in README's Active patches). It is the GitHub default branch, so the notes show on the repo page, and
  `rebase.sh`'s default `STACK_TIP`. A new patch goes below it: `git rebase --onto patch/NN-<new>
  patch/NN-<old> deploy`, then add the README row and amend.

- **`main`** — pristine mirror of `upstream/main`; only ever fast-forwarded
  (`scripts/fork/sync.sh`, `NO_PUSH=1` to skip pushing it to `origin`).
- **`base/current`** — the upstream commit the stack sits on. Started at **`e7b8c016`**,
  the revision inside the pulled image (`ghcr.io/local-inference-lab/flare@sha256:d872de72…8429`,
  flare 0.1.0). **2026-10-22: moved to `upstream/main` `947a5a6d`** (127 commits). Retired on that
  rebase, because upstream now does the same thing: `patch/02-graph-shared-memory` (upstream
  `37c87e1d` raises a function's shared-memory allowance from its captured nodes),
  `patch/04-minijinja-empty-reverse-slice` (Helios renders templates with its own Jinja VM;
  minijinja is only a test oracle, and the VM's `slice` returns an empty list for `[][::-1]`) and
  `patch/05-replace-cached` (upstream `7d3331c5` removed `ignition run`). `patch/01` and `patch/03`
  lost their `ignition run` forwarding for the same reason. Pre-rebase refs:
  `backup/pre-rebase-20261022-*` tags. **2026-10-23: moved to `upstream/main` `a66c4e41`** (33 commits;
  Qwen Flash export defaults 32 requests / 128 speculative target rows, representative QSA tuning,
  prefix-cache retention repair). No patch retired; `patch/08` stays because upstream `2de0a539`
  fixed only the Debian build context. Conflicts (upstream multi-host GPU selection and
  `roce_chain`, the Qwen default concurrency, the alias list) kept both sides. Pre-rebase refs:
  `backup/pre-rebase-20261023-*` tags. **2026-10-24: moved to upstream `bce032c0`** (42 commits;
  scheduler/graph overlap, Qwen3.8 27B MTP, `flare run --history-drafts`, cuTile preparation, and
  `601b6b11`'s cache-only `ignition export` — no `--output`, packages land in Flare's model cache).
  No patch retired. Conflicts (upstream scheduler terminal coefficients beside `patch/01`'s
  `--generation-defaults`, the API-key type, the recipe help's new image/MLA lines) kept both sides.
  `patch/08` now also carries `ignition/recipes/kimi3/assets/chat_template.jinja`, which
  `templates.rs` embeds since this base. Pre-rebase refs: `backup/pre-rebase-20261024-*` tags. **2026-10-26:
  moved to upstream `0ce6de86`** (19 commits; `<think>` text no longer a reasoning delimiter (#94),
  prose `<tool_call>` / `<function=` no longer abort a response (#67), resident embedding-table scales
  by default (`c370a85c`), `ignition serve` renamed `ignition debug-serve`). Retired
  `patch/08-docker-context-templates`: upstream `230c5b40` admits and copies the qwen38 / step5 /
  kimi3 template assets itself. Conflicts (`patch/06`'s `docs/TODO.txt` entry beside upstream's new
  one; this file's old inventory table) kept both sides; `patch/07`'s `--rope-parameters` stays in the
  preparation arguments `export` and `debug-serve` share. Pinned to `0ce6de86`, not the moving
  `upstream/main`. Pre-rebase refs: `backup/pre-rebase-20261026-*` tags.
  **2026-10-28: moved to upstream `56f5702a`** (35 commits; fused BF16 Markov sampling with
  block-broadcast speculation (#107), storage mutations preserved without cloning (`317c1ecc`),
  captured startup tuning graphs with runtime collective tuning (`f617022d`, execution ABI 5 /
  model contract 3 — prepared packages must be re-exported), bundled Inductor patches before
  preparation (#117), FlashInfer pin `f91f257e` (`7c1a9fde`), row-scaled FP8 Markov-head choices
  (#121)). All 18 stack commits replayed with zero conflicts; `--update-refs` skipped
  `patch/09-forward-storage-fit` (coincident with the stack top), re-pointed to its rebased
  commit `ec2f234d` by hand. Pinned to `56f5702a`, not the moving `upstream/main` (already at
  `acbf2858` when this move ran). Pre-rebase refs: `backup/pre-rebase-20261028-*` tags.
  **2026-10-28: retired `patch/09-forward-storage-fit`** (branch deleted; kept in
  `backup/pre-rebase-20261028-patch-09-forward-storage-fit`): upstream `317c1ecc` binds compiled
  storage as regions without cloning whole buffers, so a cached forward restores onto a smaller
  allocation with correct outputs and the guard never fires — in-image on
  `local/flare-ignition:ec2f234d` the patch's own regression test fails only because the expected
  recompiles (`[1,2,2,2]`) no longer occur (`[1,1,1,1]`). The stack tip is
  `patch/07-rope-parameters` (`e8bac4a6`); five patches remain.
  **2026-11-02: moved to upstream `6fa01a49`** (61 commits; Muse Spark, Qwen3.8-27B DFlash2
  through shared `components/dflash2.py`, vocabulary-parallel sampling (#135), the collective
  ladder, and FlashInfer floated to `@main`). All 14 stack commits replayed; two conflicts
  (`ignition/planning.py`, `tests/cli/test_models.py`) kept both sides. No patch retired. Pinned
  to `6fa01a49`, not the moving `upstream/main`. Pre-rebase refs:
  `backup/pre-rebase-20261102-*` tags.
  **2026-11-04: moved to upstream `585159c9`** (60 commits; execution ABI 6 with package-defined
  completion capacity and sixteen-token completions, DFlash2 compact prefill and window staging,
  QAD draft quantization, warmup mapping reuse across compiled warmup (#154), the Selene host
  tier, FlashInfer pinned to `560fef16`). All 14 stack commits replayed; one conflict
  (`flare/src/main.rs`: upstream's `thinking_token_budget` beside `patch/01`'s defaults apply)
  kept both sides. No patch retired. Pinned to `585159c9`, not the moving `upstream/main`.
  Pre-rebase refs: `backup/pre-rebase-20261104-*` tags.
  **2026-11-10: moved to upstream `e774a468`** (48 commits; the terminal binary moved from
  `flare/` to `flare-cli/` over a shared `flare` library, packages renamed cores
  (`runtime_core`, `core_storage`), Kimi K3.1 and its DFlash2, exports without retained CUDA
  graph templates, `--mtp-only`, `--force-moe-decode-a16`, FlashInfer pinned to `85ea6e76`,
  design checklist digest `cfcf`). Twelve commits conflicted and kept both sides on upstream's
  structure; `patch/01` and `patch/03` moved their modules into `flare-cli/src/`; one carry
  commit follows the renames; the opaque-write copy rule retired (its reproducers pass on GPU without
  it); the FlashInfer MoE fragment guard retired (upstream FlashInfer #71). Conflicted commits were
  continued with the hook off, so replayed commits keep their original acknowledgments.
  Retargeted from `8c2e782d` when upstream moved before the rebase ran. Pre-rebase refs:
  `backup/pre-rebase-20261109-*` tags.
  **2026-11-15: moved to upstream `6b68b9b4`** (75 commits; KV and capacity fixes #253, #286 and
  #285, native MXFP8 DFlash2 draft bodies (#291), the one-layout DFlash2 embedding binding (#250),
  Inductor-planned temporaries with forwards that take their metadata per call, Inkling 2 Small
  NVFP4, #298 and #302 prefix-cache changes, FlashInfer pinned to `4fd836dc`, design checklist
  digest `ed22`). 27 of 93 replayed commits conflicted and kept both sides on upstream's structure.
  History drafts follow upstream's copy path: copies are substituted on the round's local tensors
  and the copy mask travels in `DraftProposalState` to the commit. Muse Glimmer's image number
  moved from 10 to 11. Three carry commits follow the replay: the retired `qwen38_dflash` recipe's
  mixed-round hooks are dropped, `AGENTS.md`'s second TODO link is repointed, and Muse Glimmer
  moves onto the per-call forwards with its own attention module while `muse_spark/attention.py`
  returns to upstream's text. The FlashInfer pin is re-cut on `4fd836dc` as
  `Raydius/flashinfer@7245b1e0`, carrying only stub rejection and the `compile_compress` program
  cache. No patch retired. Retargeted from `29192ef7` when upstream moved before the rebase ran.
  Not yet run on this base: Rust builds, GPU tests and exports. Pre-rebase refs:
  `backup/pre-rebase-20261114-*` tags.
  **2026-11-18: moved to upstream `1a532755`** (81 commits; #289 bounds history proposals by
  logical lane length, checkpoint-work estimation without quoting snapshot storage and
  quoted-work sizing of ordinary and mixed prefill graphs, `80c97c50` passes mutated
  dtype-viewed temporaries to storage calls through their owner, 16-bit PNG and GIF input
  with Pillow-equivalent conversion, the Selene pool series, FlashInfer re-pinned to
  `2cce3e67`, design checklist digest `1af4`). Five of 130 replayed commits conflicted and
  kept both sides on upstream's structure: the FlashInfer pin, the `docs/TODO.txt` rename
  with upstream's four new entries folded into `docs/TODO.md`, the opaque-write copy pair
  (upstream's `80c97c50` rewrite of `_lower_storage` supersedes the rule the pair patched
  around; its reproducers stay in `tests/preparation/test_aot.py`), and the code_edit
  benchmark, where the fork's file already subsumes upstream's `be4ff01e`. One commit
  dropped, its `AGENTS.md` TODO link subsumed by the rename's resolution. Upstream's #289
  made the fork's `packing.rs` lane-length change independently, so only the fork's comment
  remains. The FlashInfer pin is re-cut on `2cce3e67` as `Raydius/flashinfer@be90bff55`,
  carrying only stub rejection and the `compile_compress` program cache; the
  `gdn-head-ratio` image patch applies unchanged at that base. No patch retired. Conflicted
  commits were continued with the hook off, so replayed commits keep their original
  acknowledgments; the record commit acknowledges `1af4`. Host CPU, image, Rust and GPU 1
  gates run after this commit and before the push. Pre-rebase refs:
  `backup/pre-rebase-20261118-*` tags.
  **2026-11-19: moved to upstream `ba42481a`** (15 commits; #396 keeps request-written token
  spellings literal in Jinja prompts, #401 ends the Selene rank restore channels only after
  every window is assembled, #400 counts host-cache hits and reused tokens when a restore
  succeeds, #399 warns when a prefix-cache pool has no local host tier to cascade from, #392
  counts pool put-grant pages from the accepted record, #368 prints checkpoint source rejection
  reasons on rank zero, #367 ends the Compile model forwards phase as failed when compilation
  fails, #403 re-pins FlashInfer to `598c8547` with the GLM decode-pooling negative-slot and
  multi-peer PCIe collective fixes, design checklist digest `1af4`). Three of 133 replayed
  commits conflicted and kept both sides on upstream's structure: the FlashInfer pin, the
  `docs/TODO.txt` rename with upstream's six new entries folded into `docs/TODO.md`, and Muse
  Glimmer's `bracketed_prompt` helper ported onto upstream's token-id-based rewrite of
  `expand_prompt`, which takes `prompt_token_ids` and drops `ImagePrompt.text`; Glimmer's
  expansion path and the profile dispatch pass token ids. No commit dropped. The FlashInfer
  pin is re-cut on `598c8547` as `Raydius/flashinfer@cf1ac676a`, carrying only stub rejection
  and the `compile_compress` program cache: two of the fork's three eviction cases fail on
  unmodified `598c8547` and pass on the re-cut. The `gdn-head-ratio` image patch is re-cut on
  `598c8547` as `a2f9c47a0` and applies unchanged. No patch retired. Conflicted commits were
  continued with the hook off, so replayed commits keep their original acknowledgments; the
  record commit acknowledges `1af4`. Host CPU, image and Rust gates run after this commit and
  before the push. Pre-rebase refs: `backup/pre-rebase-20261119-*` tags.
- **Moving to a newer upstream:** follow [Moving to a newer upstream](#moving-to-a-newer-upstream).

## Moving to a newer upstream

A base move replays the patch stack onto a newer upstream commit. Follow every step in order. A
replay that finishes without a conflict is not evidence that the result is correct: git merges
silently into code upstream restructured, and only the checks below find it.

**Before the replay**

1. Move to each upstream release tag when it is published, and name the target by commit, never
   by the moving `upstream/main`. A short range conflicts less than a long one.
2. Work on a `wt/<change>` branch in its own worktree. `deploy` moves only after the gates below
   pass. Tag `deploy`, every `patch/NN` branch and `base/current` as `backup/pre-rebase-<date>-*`.
3. Record upstream baselines with no fork code: at the target, and at each intermediate commit of
   step 4, run `.venv/bin/python -m pytest tests -q` in a fresh environment and the per-crate
   `cargo test` commands of `AGENTS.md`. Keep the full logs and a sorted list of failing test ids.
   Run them in a container from a `local/flare-ignition` image without a GPU; the host shell
   cannot lock 64 MiB and has no `protoc`, OpenSSL or liburing headers. The container needs
   `--ulimit memlock=-1:-1`, `--shm-size 8g` (the Selene manager tests fill the default 64 MiB),
   a `zstd` binary (the legacy-core tests call it), `UV_PROJECT_ENVIRONMENT` set to the tree's own
   `.venv` (the image points uv at `/opt/flare/.venv`), a memory limit of 14 GiB, and
   `FLARE_TEST_TEMPLATE_DIR` filled by `tests/frontend/prepare_templates.py`, one recipe per call
   so a gated repository fails alone. Synchronize with `uv sync --extra prepare --extra test
   --reinstall-package flare-heliopause`: uv reuses a path dependency's earlier build, so a kept
   environment otherwise tests the previous base's extension. For the recipes fork, record the baseline of its upstream base installed against
   the core target, with `uv sync --extra test` and its CPU `pytest tests -q`.
4. Read `git log --stat <base>..<target>` for commits that rename or move directories. Replay onto
   each such commit first, then onto the target. Git follows a fork edit through a rename only at
   the commit that performs it.
   - When upstream moves directories into the recipes repository, replay onto the core commit
     that renames them while both trees are still in core, and gate that result (step 9).
   - Divide the result by path: replay the core side onto the upstream commit that deletes the
     moved trees, and the recipes side onto the recipes commit whose trees equal core's at the
     rename commit (compare `git rev-parse <commit>:<directory>` in both repositories).
   - Prove the division before going on: for every path in `git diff --name-only <rename commit>
     <replayed tip>`, `git rev-parse <replayed tip>:<path>` equals the blob in exactly one result;
     `git diff --name-only <delete commit> <core result>` names no moved path, and the recipes
     result changes only moved paths. Keep a map from each replayed commit to its core commit,
     its recipes commit, or both.

**The replay**

5. In each fork's clone, run `scripts/fork/rebase.sh <commit>` from a copy outside the worktree,
   with `STACK_TIP` set to the `wt/` branch (the recipes fork has no copy of the script; core's
   acts on the repository of the working directory): the script is part of the stack, and a shell reads a script while it runs.
   For the same reason, never edit a script that a running container or shell is executing. When
   the `wt/` branch still sits on `deploy`'s commit the script refuses the tip; run its
   `git rebase --onto <commit> base/current <tip> --update-refs` and `git tag -f base/current
   <commit>` directly, because `deploy` is meant to stay.
6. Set `merge.renameLimit` above the number of renames in the range and `merge.conflictStyle` to
   `diff3`, so every conflict shows the merge base. Keep `rerere` on, and first move aside the
   `.git/rr-cache` entries recorded by any move that was not verified: `rerere` replays them unseen.
7. Resolve each conflict to both sides on upstream's structure, and log one line per conflicted
   commit. Continue a conflicted commit with the commit-msg hook off, so it keeps its original
   acknowledgment.
   - Apply a fork change where upstream moved the thing it changes. Git pairs a file with the
     nearest similar file, which can be the wrong one, and leaves fork-added files at paths
     upstream renamed away.
   - Regenerate `ignition/preparation/runtime_core_pb2.py` from the merged
     `helios/proto/runtime_core.proto` with the `protoc` release its header names. Never merge it
     by hand. Confirm the fork's field and enum numbers are still free in upstream's proto.
   - Leave the FlashInfer source and `uv.lock` at upstream's through the replay. The pin is
     selected afterwards, by running the fork's FlashInfer cases on upstream's revision.
   - When upstream moves documentation to the recipes repository in a commit after the split,
     take upstream's text in core, count the fork lines set aside in the conflict log, and port
     them to the recipes fork's document in one commit above its replayed stack.
   - In the recipes fork's `pyproject.toml`, name `ignition` by the core fork's URL and the exact
     core commit, then `uv lock`. Until that commit is pushed, resolve it from the local clone
     with a git `url.<local clone>.insteadOf` setting for that command only, so both files keep
     the GitHub URL.
   - Drop a commit that upstream supersedes, and name the upstream commit in the log. Compare the
     two implementations file by file before dropping, and list what the fork's had that
     upstream's lacks.

**After the replay**

8. Check what git merged without a conflict:
   - every `ignition` and `hyperflare_recipes` import in the tree resolves to a module that exists;
   - each fork-changed file that upstream did not touch equals its pre-move content, apart from
     moved module paths;
   - `git range-diff <old base>..<old tip> <new base>..<new tip>`, for each replay that crosses
     no rename, marks with `!` only commits that the conflict log names or whose difference is
     confined to context lines;
   - for each record upstream added in the range that enumerates what a patch extends
     (preparation flags, the rebuild record, recipe registration, weight-container fields, tuning
     identities, declared costs), the patch's item is in it.
9. Run the commands of step 3 on the result. It must add no failing test id to the baseline; fix
   and rerun until it adds none.
10. Give every active patch in `README.md` a verdict on the new base: kept, adapted, or retired
    with the superseding upstream commit. A kept or adapted verdict needs a probe that fails on
    unpatched upstream. Run `NO_PUSH=1 scripts/fork/sync.sh` in each clone as the probe for the
    fork tooling.
11. Set the recipes lock to the final core commit and build the preparation image from the two
    commits with `build.sh <core> <recipes> <version>` in the image recipe, which refuses a lock
    that names another core commit. Export the operator's profile with that image and serve the
    core: a correct answer to a fixed prompt, a multi-turn chat, and an image request
    when the replaced core accepted images. Move `deploy` and tag the release only after it serves.

## Before writing a patch: which layer?

| Layer | Paths | What a change costs |
|---|---|---|
| Serving | `flare/` (CLI), `helios/` (Rust runtime: scheduling, caches, HTTP) | Rebuild the image; the existing 107 GB format-4 packages keep loading |
| Prepared | `ignition/` (recipes, components, kernels, capture), package format | Re-prepare + export from the original checkpoint (PyTorch / FlashInfer env), signed with your own key → a new trust-store approval; older packages must still load |

## Build a local image

Serving images are `local/flare-ignition:<sha8>`, built by the console's recipe from a
committed ref, never the working tree:

```bash
~/LocalApps/vllm-env/build/flare-ignition/build.sh deploy
```

The script archives the ref with its Dockerfile, tags the image with the commit's first eight
hex digits, and labels it `org.opencontainers.image.revision`. With no ref it builds
`deploy` (since vllm-env `931be1f`; it used to build `patch/07-rope-parameters`, which lacks the
DFlash2 work).

Then in the console: duplicate a Flare profile into a dated one, set its Docker image to
`local/flare-ignition:<sha>`, record the image in `~/LocalApps/vllm-env/build/README.md`, and
log the outcome in `docs/models/qwen3-flash-next.md`'s Tuning-findings table.

## Tests

```bash
cargo build --release --manifest-path flare-cli/Cargo.toml
# memlock must be raised as CI does (sudo prlimit --pid $$ --memlock=67108864:67108864)
cargo test --locked --release --manifest-path helios/Cargo.toml \
  --target-dir flare-cli/target --features runtime-core,server
```

## Patch inventory

The inventory lives in [README.md](README.md#raydius-fork-patch-notes) on `deploy`, so it shows on
the fork's GitHub page. Keep both tables current there:

- **Active patches** — add a row when a patch lands.
- **Retired patches** — when upstream absorbs a patch or makes it obsolete, delete its branch and
  move its row to that table with the retirement date and why. Never drop a row.
