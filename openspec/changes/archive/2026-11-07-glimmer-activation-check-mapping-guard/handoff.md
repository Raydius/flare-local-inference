# Handoff — glimmer-activation-check-mapping-guard

## Where
- Planning repo: /home/rjdollete/dev/flare-local-inference (branch `deploy`).
- Code: worktree /tmp/glimmer-pr, branch `pr/muse-glimmer` = 9348d099 (recipe) + a4e0df5e (1M context), base upstream 972e28e4. origin and NVIDIA/spark-local-inference#229 head = a4e0df5e. #229 had 0 reviews and 0 comments at last check; re-check before any force-push.

## Stage
Tier 1 change, planned (proposal.md, tasks.md), 0/2 tasks ticked, no code edited. Fix: `ignition/recipes/muse_glimmer_nvfp4/config.py:63` chains `.get(..., {})`, so a present non-mapping `quantization_config` or `config_groups` raises AttributeError instead of ValueError. Guard each level with `isinstance(..., Mapping)` as quantization.py does, with failing-first tests in test_config.py.

## Mechanics (cost time last session)
- pytest interpreter: /home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb/.venv/bin/python (the main checkout's .venv has no pytest), with `PYTHONPATH=/tmp/glimmer-pr` and `FLARE_TEST_TEMPLATE_DIR=/tmp/flare-test-templates` (without it, 9 Gemma 4 template rows fail on both trees).
- The `.githooks/commit-msg` hook rejects a bare `git commit --fixup` (no design acknowledgment). Fold into commit 1 with `GIT_SEQUENCE_EDITOR="sed -i '1s/^pick/edit/'" git rebase -i upstream/main`, apply the change, run `git commit --amend -s -F <msg>`, then `git rebase --continue`. Start from commit 1's current message (`git log -1 --format=%B 9348d099`); it carries `Design-Signed-Off-By: Claude Opus 5.5 (claude-opus-5-5); design-review=4d3a` plus Co-Authored-By Fable 5.1 and Opus 5.5. Recompute the digest from `git show :docs/design-review.md`.
- Push: `git push --force-with-lease=pr/muse-glimmer:a4e0df5e3e1ff9d8a536ac0290e878e24a15a852 origin pr/muse-glimmer`.
- If the PR text changes: edit openspec/changes/archive/2026-11-07-upstream-muse-glimmer-pr/pr-body.md, run `tail -n +6` on it into /tmp/glimmer-pr-body.md, then `gh pr edit 229 -R NVIDIA/spark-local-inference --body-file /tmp/glimmer-pr-body.md`. Only needed if commit-1 Validation counts change.
- Full suite runner: /home/rjdollete/models/.cache/glimmer-pr/run-suites.sh <worktree> <prefix> (cargo `--features server,runtime-package`). Baseline failures are in baseline-failed.txt and enomem-upstream-main2.txt. The host was memory-short last run: 22 native export and package tests failed with ENOMEM on both trees.

## Other live state
- `deploy` has uncommitted openspec work: two archived changes, specs/upstream-contribution-branch, a muse-glimmer-serving spec edit, and both live change dirs. Nothing is committed; landing it is the operator's `/rad-close`.
- glimmer-concurrency-ceiling: empty scaffold. The operator chose "Follow-up change after the PR opens" for raising concurrency/max_concurrency=4 on the 96 GB GB202. It still needs `/rad-propose`.

## Resume prompt
```
Read openspec/changes/glimmer-activation-check-mapping-guard/handoff.md, then run /rad-apply glimmer-activation-check-mapping-guard (code in /tmp/glimmer-pr on pr/muse-glimmer; fold into commit 1 via an edit-stop amend, push with lease pinned to a4e0df5e).
```
