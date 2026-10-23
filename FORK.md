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
them. The hook stays disabled. `AGENTS.md`'s architecture and package-compatibility rules
still describe how Flare works; break them knowingly, not by accident.

## Remotes

| Remote     | URL                                                  | Role                      |
| ---------- | ---------------------------------------------------- | ------------------------- |
| `origin`   | `git@github.com:Raydius/flare-local-inference.git`   | our fork (push here)      |
| `upstream` | `https://github.com/NVIDIA/flare-local-inference.git` | NVIDIA (fetch-only; push URL is `DISABLE`) |

## Branch model

```text
base/current (tag → an upstream commit)
  └─ patch/00-fork-tooling      # this file + scripts/fork/ (never upstreamed)
       └─ patch/01-<topic>       # one logical change each
            └─ patch/NN-<topic>  # topmost patch = what you build
                 └─ deploy           # README patch notes; GitHub default branch
```

- **`deploy`** — one commit above the topmost patch: README's patch notes and this file's
  inventory pointer. It is the GitHub default branch, so the notes show on the repo page, and
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
  `backup/pre-rebase-20261023-*` tags.
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

The repo `Dockerfile` builds only the `flare` binary (Rust) into `debian:trixie-slim`.
Build from the committed tip, not the working tree:

```bash
tip=patch/NN-<top>; sha=$(git rev-parse --short=8 "$tip")
git archive "$tip" | docker build -t "local/flare:$sha" \
  --label org.opencontainers.image.revision="$(git rev-parse "$tip")" -
```

Then in the console: duplicate a Flare profile into a dated one, set its Docker image to
`local/flare:<sha>`, record the image in `~/LocalApps/vllm-env/build/README.md`, and log the
outcome in `docs/models/qwen3-flash-next.md`'s Tuning-findings table.

**Control image:** `local/flare:e7b8c016` is `base/current` built unpatched — the
comparison point for every patched build. Its `/usr/local/bin/flare` is byte-identical to the pulled
image's (checked 2026-10-20), so the local build path reproduces upstream exactly.

## Tests

```bash
cargo build --release --manifest-path flare/Cargo.toml
# memlock must be raised as CI does (sudo prlimit --pid $$ --memlock=67108864:67108864)
cargo test --locked --release --manifest-path helios/Cargo.toml \
  --target-dir flare/target --features runtime-package,server
```

## Patch inventory

The inventory lives in [README.md](README.md#raydius-fork-patch-notes) on `deploy`, so it shows on
the fork's GitHub page. Keep both tables current there:

- **Active patches** — add a row when a patch lands.
- **Retired patches** — when upstream absorbs a patch or makes it obsolete, delete its branch and
  move its row to that table with the retirement date and why. Never drop a row.
