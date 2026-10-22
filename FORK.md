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
- **`base/current`** — the upstream commit the stack sits on. Started at **`e7b8c016`**,
  the revision inside the pulled image (`ghcr.io/local-inference-lab/flare@sha256:d872de72…8429`,
  flare 0.1.0). **2026-10-22: moved to `upstream/main` `947a5a6d`** (127 commits). Retired on that
  rebase, because upstream now does the same thing: `patch/02-graph-shared-memory` (upstream
  `37c87e1d` raises a function's shared-memory allowance from its captured nodes),
  `patch/04-minijinja-empty-reverse-slice` (Helios renders templates with its own Jinja VM;
  minijinja is only a test oracle, and the VM's `slice` returns an empty list for `[][::-1]`) and
  `patch/05-replace-cached` (upstream `7d3331c5` removed `ignition run`). `patch/01` and `patch/03`
  lost their `ignition run` forwarding for the same reason. Pre-rebase refs:
  `backup/pre-rebase-20261022-*` tags.
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

| Branch | Layer | Purpose |
| --- | --- | --- |
| `patch/00-fork-tooling` | — | This file + `scripts/fork/{sync,rebase}.sh` (copied from `~/dev/vllm`) |
| `patch/01-generation-defaults` | Serving | `flare run --generation-defaults <JSON>` merges sampling keys, `max_new_tokens` and `chat_template_kwargs` into the package's server config (unknown key or out-of-range value is a startup error naming it). Packages keep loading |
| `patch/03-context-and-chat-template` | Serving | `flare run --max-context-tokens <N>` lowers the per-request context (`max_sequence_tokens`) in the server config; above the package's prepared context, or off its page size, is a startup error naming both. `flare run --chat-template <FILE>` replaces the package's `chat_template` asset with the file (hashed at startup); a package with no Jinja frontend is a startup error. The package file is never written |
| `patch/06-qwen38-yarn-declaration` | Prepared (`ignition/`) | Johannes Bertens's `ec0a64f2` (`github.com/JohannesBertens/flare-local-inference` branch `pr-yarn4`), cherry-picked unchanged: `RopeScaling` declaration (factor, original context, beta_fast/beta_slow), YaRN frequency-only interpolation of the Qwen3.8 rotary tables, the `qwen3.8-flash-yarn4` alias (1,048,576 tokens), and plan/identity/limits derived from the declared context. No attention temperature: sparse attention has no logit scale. No Helios or package-format change |
| `patch/07-rope-parameters` | Prepared (`ignition/cli`) | `ignition export` / `ignition serve` `--rope-parameters <JSON>` applies `patch/06`'s scaling to any resolved checkpoint (an alias or a checkpoint directory such as the console's `/ckpt`), as a Hugging Face / vLLM YaRN `rope_parameters` object: `factor` (required), `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `max_position_embeddings` (serving context, default factor × original). Same validation as a declared variant; `mscale` / `attention_factor` / unknown keys are refused; recipes that declare no rope scaling refuse the flag. Not forwarded to multi-host peer ranks |
| `patch/08-docker-context-templates` | Build (`Dockerfile`, `.dockerignore`) | Upstream `947a5a6d`'s repo `Dockerfile` no longer builds: `helios/src/frontend/templates.rs` `include_str!`s `ignition/recipes/{qwen38,step5}/assets/chat_template.jinja`, which the build context excludes. The context and the build stage now carry those two files. Retire when upstream fixes its Dockerfile |

