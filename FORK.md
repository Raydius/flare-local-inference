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
`pr/*` branches cut on `upstream/main` (#211 and #212, opened 2026-11-04/05).
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
  commit follows the renames; the opaque-write rule runs inside upstream's copy-deferral pass;
  the FlashInfer MoE fragment guard retired (upstream FlashInfer #71). Conflicted commits were
  continued with the hook off, so replayed commits keep their original acknowledgments.
  Retargeted from `8c2e782d` when upstream moved before the rebase ran. Pre-rebase refs:
  `backup/pre-rebase-20261109-*` tags.
- **Moving to a newer upstream:** `scripts/fork/sync.sh`, then
  `scripts/fork/rebase.sh upstream/main` (or a sha; the tip defaults to `deploy`). The script
  refuses a tip that shares a commit with another branch (`--update-refs` would skip it).
  `rerere` is enabled so conflict resolutions replay.

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
