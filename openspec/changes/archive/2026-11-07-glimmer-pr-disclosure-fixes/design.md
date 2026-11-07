<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Design — glimmer-pr-disclosure-fixes

## Approach

- Worktree: `/tmp/glimmer-pr` on `pr/muse-glimmer` (head `31f4e6be`, base `972e28e4`). Every code edit lands there; this checkout's `deploy` changes only under `openspec/`.
- Parser: `ignition/preparation/quantization.py` `modelopt_nvfp4_layers` checks `config_groups.group_0.weights` only (4-bit, group 16), as upstream's private Gemma 4 parser did; its docstring drops "and input activations". No flag on the shared function — the one caller that needs W4A4 owns that rule.
- Glimmer: `ignition/recipes/muse_glimmer_nvfp4/config.py` checks `quantization_config.config_groups.group_0.input_activations` is 4-bit group-16 before calling the parser, raising `ValueError("Muse Glimmer requires 4-bit group_size=16 NVFP4 input activations")`.
- Gemma 4: `ignition/recipes/gemma4/config.py` is untouched; `tests/recipes/test_gemma4.py` drops the two `input_activations` fixture lines the PR added, so its fixtures equal upstream's and run the weights-only path.
- Parser tests: `tests/preparation/test_quantization.py` — `test_weight_only_group_is_rejected` becomes `test_weight_only_group_parses` (the Glimmer fixture without `input_activations` returns the same 416-entry map; fails before the fix), and the `input_activations` row leaves `test_malformed_blocks_are_rejected`. Glimmer tests: `tests/recipes/muse_glimmer_nvfp4/test_config.py` gains a parametrized case — input activations absent, 8-bit, group 32 — each raising the new `ValueError`.
- DFlash2: `ignition/components/dflash2_weights.py` loses `_PROJECTION_NAMES`, the `projector=` keyword on `DraftWeights.load` and `validate_metadata`, the `convolutions` flag, the `selector=None` path, and the `dict | None` annotations; `ignition/components/dflash2.py` loses the `convolution is None` / `selector is None` branches. Both return to upstream's text except the `block_size` field without default that the block-16 drafter needs; the target is `git diff upstream/main -- <file>` showing only that hunk plus the Glimmer-needed lines that `git log -p 9834def8 85463f36 a6b06bdb -- <file>` attributes to Glimmer's block-16 drafter. `tests/recipes/muse_glimmer_nvfp4/test_draft_weights.py` drops `test_projector_switch_keeps_the_dflash2_tensor_set_apart`.
- History: the edits go into a `fixup!` of commit 1, `git -c sequence.editor=: rebase --autosquash upstream/main`, so the PR stays two commits; commit 1's message replaces "the ModelOpt NVFP4 config parser moves from the Gemma 4 recipe to `ignition/preparation/quantization.py` and Gemma 4 calls it (results unchanged, …)" with "…and Gemma 4 calls it; Gemma 4 accepts and maps the same configurations, and the parser's errors now name the ModelOpt config rather than Gemma 4; Muse Glimmer checks its 4-bit input activations itself", drops the `projector` sentence, and refreshes the Validation counts. Both commits re-read `docs/design-review.md` and the prose skill, recompute the digest, keep `-s`.
- Validation: the same pytest selection and cargo modules as `upstream-muse-glimmer-pr` task 2.1 (runner `/home/rjdollete/models/.cache/glimmer-pr/run-suites.sh` with `-ra`, interpreter `/home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb/.venv/bin/python`, cargo with `--features server,runtime-package`), compared against the recorded `baseline-failed.txt` for `972e28e4`: no failure only on the branch. The new parser case is run on the pre-fix tree first and must fail.
- Delivery: `git push --force-with-lease=pr/muse-glimmer:31f4e6be origin pr/muse-glimmer` (PR #229 has no review comments, so no reviewed state is lost), then `gh pr edit 229 --body-file` from the archived `pr-body.md` with the Gemma 4 sentence restored to "Gemma 4 parses the same configurations" and the projector sentence removed.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 3/2, escalated to tier 1)
Certainty: 0.965
Caller Agreement: 0.96
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — ok, agreement 0.98
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.95
- local-ai-qllm (qLLM) — cancelled (quorum met before it responded), agreement n/a

### Review Feedback

Consensus feedback recorded verbatim:
- "Keep the shared quantization parser validating `weights` only (preserving upstream Gemma 4 behavior), and place the Glimmer W4A4 `input_activations` requirement (4-bit, group_size 16) in Glimmer's own recipe config logic rather than the shared module." — ACCEPTED: the Parser and Glimmer bullets.
- "Revert the Gemma 4 test fixtures to weights-only. For DFlash2, remove the dead `projector` keyword, the unused branches, and the single test calling that path, restoring the component to handle dflash2 exclusively." — ACCEPTED: the Gemma 4 and DFlash2 bullets.
- "Deliver the fix by amending commit 1, recomputing the review-acknowledgment digest on both commits, and force-pushing (with lease), as the PR has no reviews yet." — ACCEPTED: the History and Delivery bullets; the lease names the expected remote sha `31f4e6be`.
- "Risks include accidentally leaving Glimmer unvalidated for W4A4 if the recipe-side check is missed. Tests that fail before the fix include `tests/preparation/test_quantization.py` or `tests/recipes/test_gemma4.py` using legacy weights-only configs." — ACCEPTED: the Glimmer config case pins the recipe-side check; the parser weights-only case is run on the pre-fix tree and must fail.
- "Avoids premature generalization (Rule of Three) by ensuring shared code only implements what is universally required, keeping caller-specific validation local to the recipe, and stripping speculative dead code." — ACCEPTED: no flag on the shared parser.
