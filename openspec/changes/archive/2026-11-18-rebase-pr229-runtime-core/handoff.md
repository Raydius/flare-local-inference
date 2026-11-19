# Handoff — rebase-pr229-runtime-core

**Objective.** Rebase PR #229 (NVIDIA/hyperflare-inference, branch `pr/muse-glimmer`, head `5264a595`) onto the latest `upstream/main`, port Muse Glimmer onto main's runtime-core layout and per-call-metadata forwards, rerun the affected checks, and publish. Repo `~/dev/flare-local-inference`, branch `deploy` (clean except this untracked, uncommitted change directory).

**Requirements.** The owner's words that govern this change are `proposal.md` `## Stated Requirements` — read them before any other action.

**Stage.** Planning complete (Tier 3; proposal, design with peer review and bomb-squad findings, spec delta, tasks; lint clean, `openspec validate` passes). Apply has not started: no worktree exists under `~/models/.cache/pr229-rc/`, and nothing has been pushed or posted.

**State the next session needs.**
- The planning session ran on Claude Opus 5.5, so tasks.md routes edits to restricted Glimmer files and all message writing to a Fable 5.2 sub-agent. The operator then switched the session to Fable 5.2. A Fable 5.2 apply session may do that work itself; the `Design-Signed-Off-By` and `Co-Authored-By` names stay Fable 5.2 either way. On any other model, dispatch as tasks.md says.
- `upstream/main` was `abd56461` at planning. Task 1.1 fetches and retargets if it moved.
- Bomb-squad dispositions in `design.md` `## Bomb-squad findings` are folded into tasks 1.2, 1.3, 1.6 and 1.7. Two are declined (vision-scratch pool-reuse coverage; the 4e-2 hidden-row bound).
- Owner holds: task 1.5 asks once whether GPU 0 is free (never stop its container); task 1.8 pushes and posts only after the owner approves the drafted commit messages, PR description and comment.

## Resume prompt

```
/rad-apply rebase-pr229-runtime-core

Context: repo ~/dev/flare-local-inference, branch deploy. The change rebase-pr229-runtime-core (openspec/changes/rebase-pr229-runtime-core/, untracked, planning complete, lint clean) rebases PR #229 (NVIDIA/hyperflare-inference, branch pr/muse-glimmer, head 5264a595, three commits on d2b5c5f3) onto the latest upstream/main (abd56461 at planning; retarget if it moved). Main renamed runtime package to runtime core, took image number 10 for padded-grid (Glimmer moves to 11), and moved forwards to per-call metadata, so commit 1 absorbs deploy's Fable 5.2 Glimmer port with fork-only code stripped; Muse Spark's attention file stays upstream's text.

Read handoff.md in the change directory first, then proposal.md's Stated Requirements. Apply has not started: begin at task 1.1 (worktrees under ~/models/.cache/pr229-rc/, never /tmp). Work only in that worktree; deploy changes only under the change directory. If this session runs on Fable 5.2, do the restricted-file edits and message writing directly instead of dispatching a sub-agent; trailers name Fable 5.2 either way.

Constraints: 14 GiB memory cap per test scope; stop only your own processes by PID; GPU 0 is the owner's Qwen instance — ask once in task 1.5 whether it is free, never stop it; no export or serving; push with lease and post to the PR only in task 1.8, after I approve the drafted commit messages, PR description and comment.
```
