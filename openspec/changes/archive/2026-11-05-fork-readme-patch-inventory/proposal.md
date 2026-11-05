# List every fork change on deploy in the README patch notes, with what upstream still lacks

Tier: 1 — one documentation section in README.md; the other cited files are read-only evidence for the necessity findings
Type: bug

External Review: Waived — Tier 1 (one documentation section in one file; no code or behavior change)

## Stated Requirements

- "determine what fork patches are stil necessary and update the README in the deploy branch to accurately reflect the patches."

## Root Cause

wrong-mental-model: the README's patch notes (README.md:1-31) assume `deploy` is the five `patch/NN` branches plus one notes commit, as FORK.md:33 describes. The DFlash2 branch was later merged onto `deploy` (commits `bee64b00`..`43017c28`, 32 commits). It brought a recipe, shared-component changes, a Helios scheduler fix, a compiler rule, an export flag, and a FlashInfer re-pin, and none of them got a row. The Stack base line also predates upstream `0009b89d`, which carries its own Flash Next DFlash2 (`d68ed8f5`).

## Necessity findings (against upstream/main `0009b89d`, read-only)

- patch/00 tooling: FORK.md:1-10 states it is never upstreamed.
- patch/01 (`--generation-defaults`): upstream `helios/src/server/config.rs:47` refuses a `generation_defaults` key, and upstream's `flare/src/config.rs:40` `Server` has no sampling defaults. Still needed.
- patch/03 (`--max-context-tokens` / `--chat-template`): grepped `max[-_]context[-_]tokens|--chat-template` across upstream `flare/ helios/`, no hits. Still needed.
- patch/06 and patch/07: grepped `RopeScaling|yarn4` across upstream `ignition/`, no hits. Still needed until PR #212 merges.
- DFlash2 recipe `qwen38_dflash`: on the current base `585159c9` it is the only Flash Next DFlash2. Upstream `d68ed8f5` (in `0009b89d`) duplicates it except for mixed prefill (upstream `ignition/recipes/qwen38/bootstrap.py:206` rejects it) and the yarn4 alias.
- History drafts on the parallel-draft round (`95752d66`): grepped `substitute_copies` in upstream `ignition/`, no hits. Fork-only.
- Scheduler copy-depth attribution (`6657f7e5`): upstream `helios/src/scheduler/speculation.rs:136` `observe` still takes no copy mask (fork `:137` does). Still needed.
- Mixed speculative export fixes (`ecb6c1dc`, `a088b0b6`, `388d3d21`): open upstream as PR #211.
- Opaque-write compiler rule (`80946209`): redundant on `585159c9` per the 2026-11-04 probes. Retirement waits on a re-export.
- `--recipe` / `--draft-model` checkpoint-directory export (`2c8ba959`, deploy `ignition/cli/__init__.py:48-51`): grepped `draft[-_]model` across upstream `ignition/cli`, no hits. Fork-only.
- FlashInfer pin `Raydius/flashinfer@49fe5449` (deploy `pyproject.toml:25`): upstream pins `1f8ed8a3`, which has the MoE guard (#71) but not the eviction fix (FlashInfer PR #67, open). Still needed.
- Warmup retention (`026839f1`): already retired in `e6fc2ef6` (upstream #154). Goes to the Retired table.

## Blast Radius

1. README.md:1-31 — patch-notes header, Active patches and Retired patches tables; the only section this change rewrites.
2. FORK.md:138-140 — points at the README tables; unchanged.
3. grepped 'lessons' under openspec/ — openspec/lessons.md does not exist; no recorded scenario to consult.
