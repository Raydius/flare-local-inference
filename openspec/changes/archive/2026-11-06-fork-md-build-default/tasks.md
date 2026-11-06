# Tasks

- [x] FORK.md:126: replace the sentence that build.sh defaults to `patch/07-rope-parameters` with one saying it defaults to `deploy` (vllm-env `931be1f`). Verify: `grep -c "default ref is .patch/07" FORK.md`. Done-when: it prints 0, and FORK.md is the only file that changes outside openspec/.
  evidence: grep -c "default ref is .patch/07" FORK.md — 0; FORK.md:125-127 now say build.sh builds deploy with no ref (vllm-env 931be1f); git diff --stat — FORK.md only
