# Tasks

- [x] FORK.md only. Make four edits:
  - Scope paragraph (FORK.md:9-14): keep the operator's 2026-10-20 decision and the overruled-gates sentence word for word. Replace "The hook stays disabled." with a statement that the hook is enabled in this clone (`core.hooksPath=.githooks` in `.git/config`), so every commit needs one `Design-Signed-Off-By` acknowledgment per `docs/design-review.md`. Add one sentence: patches that prove worth it now go upstream as PRs from `pr/*` branches cut on `upstream/main` (#211, #212, 2026-11-04/05).
  - Upstream remote row (FORK.md:21): note that GitHub now redirects the URL to `NVIDIA/spark-local-inference`.
  - Diagram comment (FORK.md:30): name the DFlash2 work.
  - `deploy` bullet (FORK.md:33-36): say that it carries the patch stack, the README notes commits, and the DFlash2 work merged above them (no `patch/NN` branches; inventoried in README's Active patches).

  Leave all other text unchanged. Verify: git diff --stat deploy -- FORK.md. Done-when: only FORK.md changes outside openspec/, and `grep -c 'hook stays disabled\|one commit above the topmost patch' FORK.md` prints 0.
  evidence: git diff --stat deploy -- . ':!openspec' — FORK.md only (1 file); grep -c 'hook stays disabled\|one commit above the topmost patch' FORK.md — 0; FORK.md:13-16 hook enabled + upstream PRs, FORK.md:25 redirect note, FORK.md:34 diagram, FORK.md:37-39 deploy bullet
