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
```

- **`main`** — pristine mirror of `upstream/main`; only ever fast-forwarded
  (`scripts/fork/sync.sh`, `NO_PUSH=1` to skip pushing it to `origin`).
- **`base/current`** — the upstream commit the stack sits on. Starts at **`e7b8c016`**,
  the revision inside the pulled image the console pins
  (`ghcr.io/local-inference-lab/flare@sha256:d872de72…8429`, flare 0.1.0), so a patched
  build differs from the known-good image only by the patches.
- **Moving to a newer upstream:** `scripts/fork/sync.sh`, then
  `STACK_TIP=patch/NN-<top> scripts/fork/rebase.sh upstream/main` (or a sha). The script
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
comparison point for every patched build.

## Tests

```bash
cargo build --release --manifest-path flare/Cargo.toml
# memlock must be raised as CI does (sudo prlimit --pid $$ --memlock=67108864:67108864)
cargo test --locked --release --manifest-path helios/Cargo.toml \
  --target-dir flare/target --features runtime-package,server
```

## Patch inventory

| Branch | Layer | Purpose |
| --- | --- | --- |
| `patch/00-fork-tooling` | — | This file + `scripts/fork/{sync,rebase}.sh` (copied from `~/dev/vllm`) |
