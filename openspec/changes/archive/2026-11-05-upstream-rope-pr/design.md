# Design — upstream-rope-pr

## Approach

- Worktree `/tmp/rope-pr` on branch `pr/rope-scaling` at `upstream/main` `0009b89d` (same pattern as `/tmp/review-defects` for PR #211); `git cherry-pick -s ccb4dd35 f1ce181c` — sign-off added by `-s`, the `Design-Signed-Off-By` trailers ride along.
- Conflict `FORK.md` (deleted upstream): `git rm`; the patch/07 message's FORK.md paragraph is dropped at the commit edit.
- Conflict `ignition/recipes/qwen38/recipe.py` import line: union of both sides (`RopeScaling` + `SpeculativeMethod`).
- Conflict `ignition/recipes/qwen38/bootstrap.py` `_identity`: keep upstream's signature `(manifest, world_size=1, *, config=None, draft_config=None)`; inside the existing `if config is not None:` block add the key only when scaling is declared — `if config.rope_scaling is not None: value["rope_scaling"] = {factor, original_max_position_embeddings, beta_fast, beta_slow}` — so the dict shape for `config=None` and for a native config is unchanged; the caller at the server-config site stays upstream's (`config=config, draft_config=draft_config`). (`_identity` already folds the sha256 of every recipe and component source file, `ignition/recipes/qwen38/bootstrap.py:219-223`, so every recipe commit re-keys the prefix-cache seed by upstream's own design; the identity is a cache seed, not a loading gate, and existing packages keep serving.)
- Conflict `tests/recipes/test_qwen38_bootstrap.py`: keep both appended tests; `test_identity_separates_rope_scaling_preparations` moves to the upstream signature — `_identity(manifest, config=PreparationConfig())` vs `config=PreparationConfig(context=1_048_576, rope_scaling=RopeScaling(4., 262144))` vs factor 8, three distinct values, and `PreparationConfig()` equal to `PreparationConfig(rope_scaling=None)`.
- `docs/TODO.txt` hunk (auto-merged): the "GPU execution is blocked in this environment" sentence is replaced by the fork's evidence line (DFlash2 yarn4 package served at 1,048,576 tokens with a past-262k probe; the MTP `qwen3.8-flash-yarn4` package itself unserved).
- CPU suites on the branch: `tests/components/test_rotary_interpolation.py tests/recipes/test_declarations.py tests/recipes/test_qwen38_attention.py tests/recipes/test_qwen38_bootstrap.py tests/cli/test_models.py tests/cli/test_commands.py tests/preparation` from this worktree's venv with `/tmp/rope-pr` first on `PYTHONPATH`; then the branch's new test files against pristine `/tmp/upstream-main2` (`0009b89d`) expecting the new cases to fail (an ImportError on `RopeScaling` counts as the failure and is recorded as such).
- Push `pr/rope-scaling` to `origin` (Raydius) with lease; no push to `main`, `deploy`, or `upstream`.
- PR text as section 3 of the vault note "Upstream PR Drafts.md": title "Qwen3.8 Flash Next: declared YaRN rope scaling and --rope-parameters for 1M-token serving"; body = the two commit messages' substance (declaration, shared interpolation helper, identity folding, planning ceilings; the flag and its refusals), validation paragraph with the CPU numbers on `0009b89d`, the fork's served DFlash2 yarn4 evidence labeled as such, the unserved MTP yarn4 package named as the material limitation, and the authorship line (patch/06 designed with Claude Opus 5.5; no kernel or fusion code in the diff — AGENTS.md's restriction does not cover it). Its Order line adds: open only after the owner answers #210 (b).
- Vault records: tracker Now/Next 5/Later, patch notes Current stack line.
- Operator leg, gated: open the PR from `Raydius/flare-local-inference:pr/rope-scaling` against `main` once #210 (b) is answered yes; a "no" strikes the item and keeps the branch as the fork's own.
- Record commits on `wt/upstream-rope-pr` (new, from `9804c31b`); `wt/dflash-s-e2vk` no longer exists (landed on `deploy`).

### Conventions consulted

- `openspec/changes/archive/2026-11-04-upstream-first-prs/{proposal,tasks}.md` — PR prep in a `/tmp` worktree on `upstream/main`, DCO via `-s`, CPU suites on branch and pristine base, operator opens; followed verbatim.
- CONTRIBUTING.md:9 — sign-off on every commit.
- AGENTS.md "Model package compatibility" — identity must not re-key existing packages; the `config=None`/native path stays byte-identical.

### Explicit Rejections

- Rebase the whole `patch/06..patch/07` range onto `upstream/main` with `git rebase --onto` — rejected because the stack's patches 00–03 are not part of the PR; two cherry-picks keep the PR to the two commits.
- Keep the fork's `_identity(manifest, world_size, rope_scaling)` signature — rejected because upstream `d68ed8f5` changed every caller to keyword `config=`; the fork shape would re-touch the call sites and diverge from the fold the next rebase does anyway.
- Open the PR now — rejected because AGENTS.md admits contributions that answer an explicit maintainer request, a demonstrated defect, or a measured problem; the owner's answer to #210 (b) is that request.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.875
Caller Agreement: 0.7
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.85
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.3
- deepseek-deepseek-flash (deepseek-flash) — cancelled (quorum met first), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Carry the two patches as cherry-picks (not a full-range rebase, which drags in FORK.md and intermediate state) onto a scratch worktree at upstream 0009b89d, with DCO sign-off on each commit." — ACCEPTED: the Approach's first bullet.
- "Resolve the `_identity` conflict by adopting upstream's keyword `config=` signature and adding the `rope_scaling` key only when scaling is non-None, so native packages produce byte-identical identities and nothing re-keys." — ACCEPTED: the draft had written `"rope_scaling": None` into every `config`-bearing identity, which would have changed the native dict; the key is now added only when a scaling is declared (Approach bullet 4, test bullet 5).
- "Verify with a golden-digest comparison of existing package identities against upstream main." — REJECTED: `_identity` folds the sha256 of every recipe and component source file (`ignition/recipes/qwen38/bootstrap.py:219-223`), so any commit touching the recipe re-keys the digest by upstream's design; the digest seeds the prefix cache (`helios/src/cache/hash.rs:17`, a 32-byte value) and gates no package load. The test asserts the dict-shape property instead (native config equals `rope_scaling=None`, scaled differs).
- "also check the placement of the rope field relative to the `if config is not None:` block, the Rust manifest loader (e.g., deny_unknown_fields) and its matching cache-key computation" — REJECTED for the loader: Helios receives the identity as an opaque 32-byte digest (`helios/src/cache/hash.rs:17`; `helios/src/runtime_package/describe.rs:98` reports it), parsing no keys; the placement point is the ACCEPTED item above.
- "golden fixtures and CLI help snapshots, planning/KV-budget tests affected by the 1,048,576 ceiling, and argparse collisions in the subparser" — ACCEPTED: the CPU suite list names `tests/cli/test_commands.py`, `tests/cli/test_models.py`, and `tests/preparation` (planning ceilings and chunk bounds) on the branch; the patches already passed 2,025 tests on the fork base.
- "no maintainer request or demonstrated defect has been answered, so the PR should be prepared but not opened until the issue is resolved" — ACCEPTED: the operator leg is gated on #210 (b).
- "validation is CPU-only with no GPU long-context quality evidence; AI-assisted design must be disclosed even though no kernel code is touched; the mscale/attention_factor refusal needs justification, and the --rope-parameters flag is the easiest part to reject; the TODO entry is an open task, not a support claim. The PR body should state what is and is not validated, the identity-stability proof, the mscale decision, AI involvement, and that the alias is a naming choice the maintainer may decline." — ACCEPTED: the PR text bullet lists each statement; the two commits stay separate so the declaration can merge without the flag; the body says the fork's served evidence is the DFlash2 yarn4 package and the MTP yarn4 package is unserved, and names the alias as a naming choice.
