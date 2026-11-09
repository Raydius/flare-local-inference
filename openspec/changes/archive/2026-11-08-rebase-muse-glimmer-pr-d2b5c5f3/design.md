# Design — rebase-muse-glimmer-pr-d2b5c5f3

## Approach

See proposal.md — Why. All code work happens in the existing worktree `/tmp/glimmer-pr` (branch `pr/muse-glimmer`, head `8d3c0e81`). The `deploy` checkout changes only under this change directory.

1. **Rebase and regenerate.** Save the pre-rebase trailers, then `git rebase upstream/main` (`d2b5c5f3`). At the stop on `ee24d5fe`, discard both sides of `ignition/preparation/runtime_package_pb2.py` and regenerate it from the auto-merged proto with `/tmp/protoc-bin/protoc --proto_path=helios/proto --python_out=ignition/preparation helios/proto/runtime_package.proto` (`helios/manifest.md:379`). Commits 2 and 3 replay cleanly. The regenerated file must load with field 30, enum value 10 and oneof tag 10, and must equal a fresh regeneration from HEAD's proto.
2. **Re-run the checks the resolution affects, branch against base.** The resolution changes the Protobuf schema module and the recipe's tree under a new base, so the re-run covers:
   - pytest, `-m 'not cuda'`: `tests/preparation`, `tests/cli/test_planning.py`, `tests/cli/test_models.py`, `tests/recipes/muse_glimmer_nvfp4` and `tests/frontend/test_muse_glimmer.py`;
   - Helios `cargo test --features server,runtime-package --lib`, filtered to `runtime_package`, `frontend::images` and `model::tests`.

   Each selection runs once on the rebased head and once on a pristine `d2b5c5f3` worktree (`/tmp/upstream-main2`, moved to `d2b5c5f3`). Each run is one path group at a time inside `systemd-run --user --scope -p MemoryMax=14G`. Pass condition: `comm` over the sorted FAILED/ERROR ids shows nothing only on the branch. CUDA-, snapshot- and drafter-gated cases are listed as unrun.
3. **Rewrite and re-acknowledge three commits.** Each message body is rewritten in the order from `docs/design-review.md` principle 16:
   - why, and why this approach;
   - what changed, before → after;
   - the claimed result, expected or measured;
   - the evidence and its limits.

   Validation figures are split by where they were measured: the targeted re-run on `d2b5c5f3`, and the wider suite counts measured at `972e28e4`, labeled with that base. GPU figures name their fork package and image, and state that this branch was not exported or served. Process assurances are removed. Each message is reviewed against `git show :docs/design-review.md` (digest `3ad6`) and the staged prose skill. The rewrite and the `3ad6` review run on a Claude Fable 5.1 sub-agent (Agent tool, `model: fable`), the model that did the work. Only the `Design-Signed-Off-By` trailer changes, to `Claude Fable 5.1 (claude-fable-5-1); design-review=3ad6`; `Signed-off-by` and the existing `Co-Authored-By` lines stay byte-identical. The three commits are rebuilt with `git commit-tree` from each rebased tree, keeping `GIT_AUTHOR_NAME/EMAIL/DATE`, after `.githooks/commit-msg <msgfile>` accepts each message.
4. **Publish.** Rewrite the PR #229 body in the same order: three commits, eight default and sixteen ceiling, no review assurance, fork-evidence attribution kept, and an overlaps section naming #212 and #192. Read it on its own, without the diff. Then:
   - force-push with lease to `origin pr/muse-glimmer`;
   - `gh pr edit 229 --body-file`;
   - post one comment with the resolution and the re-run results;
   - confirm `headRefOid` equals HEAD and the PR is `MERGEABLE`.

**Decisions**
- **Regenerate the pb2 file, don't hand-merge it.** The file is a deterministic function of the proto, and this protoc reproduces both committed versions byte for byte. A hand merge of serialized offsets cannot be checked by reading. Alternative rejected: picking one side and re-adding the other's descriptor bytes.
- **Re-run the targeted set, scope the old figures.** Re-running the full suite risks the earlier out-of-memory failures, and it is not what the conflict touches. The four-part rule allows old figures if they are labeled with where they were measured. Alternative rejected: re-running the full ten-path set (see External Review).
- **Keep the work attributed to Fable 5.1.** The rebase is messaging, and the work was done by Fable 5.1. The authorship trailers stay unchanged, and the renewed acknowledgment names Fable 5.1. That name is true because a Fable 5.1 sub-agent performs the `3ad6` review. Alternatives rejected: naming Claude Opus 5.5 in the acknowledgment or a `Co-Authored-By` (operator correction: "It is inaccurate to call this authored by Claude Opus 5.5 when the actual work was done by Fable 5.1 -- this is just messaging."); keeping each commit's old reviewer name without re-review (it would assert a `3ad6` review that never ran).
- **Argue coexistence with #192 by quantization layout, and keep #229's alias.**
  - *Same architecture.* Both checkpoints' `config.json` architecture fields are identical.
  - *Different quantization layout.* `original-el8` at `b47e1e9f` applies NVFP4 W4A4 group-16 to every decoder Linear and keeps a BF16 `lm_head`. `nvidia` at `47818374` mixes 180 FP8 targets with 213 NVFP4 W4A16 targets, including `lm_head`.
  - *Mutually exclusive recipes.* Each recipe's checkpoint check rejects the other checkpoint (proposal Blast Radius 9), so neither recipe can serve both, and they are separate recipes rather than competing definitions.
  - *Wording.* The comment says "same architecture, incompatible quantization layout", which a reader can verify from the two `config.json` files. It does not say "different topology".
  - *Alias.* #229 keeps `muse-glimmer-30b-nvfp4`, because its checkpoint is uniformly NVFP4: its model card lists W4A4 on all 416 decoder projections. #192's checkpoint is, by NVIDIA's own card, an AutoQuantize mixed-precision checkpoint (W4A16 NVFP4, FP8 and BF16 fallback; 6.6 effective bits per parameter), and #192's title already says "mixed-precision". The comment asks #192 to take an alias that names its mixed precision and leaves the exact name to its author (operator choice: "#192 renames").
  - *Alternative rejected:* letting the maintainer pick both names.
- **Build commits with `git commit-tree` and run the hook by hand.** This keeps the trailer block exactly as written while still proving the hook accepts it. Alternative rejected: `git rebase -i` rewords, which this harness cannot drive.

**Risks / Trade-offs**
- [A different protoc changes the header or descriptor encoding] → use only `/tmp/protoc-bin/protoc` (libprotoc 35.1), and check the version and the byte-identical reproduction first.
- [The native export tests in `tests/preparation` fail with ENOMEM again] → the base runs under the same cap, so a shared ENOMEM failure is common to both and is reported as such.
- [The force-push overwrites a newer remote head] → use `--force-with-lease` against `8d3c0e81`.
- [The rewritten messages overstate what was checked] → the four-part review step runs before each acknowledgment, and every figure carries its base or package.

## Spec delta

`upstream-contribution-branch` — "Commits satisfy upstream's contribution rules" is modified. The message content list becomes upstream's four-part order, and the same order applies to the PR body.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.825
Caller Agreement: 0.85
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.95
- google-gemini-rpa-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.7
- local-ai-qllm (qLLM) — cancelled (quorum met first)
- deepseek-deepseek-flash (deepseek-flash) — cancelled (quorum met first)

### Review Feedback

Consensus feedback recorded verbatim:
- "Rebase the PR's 3 commits onto upstream d2b5c5f3 rather than merging." — ACCEPTED: Approach step 1 rebases; no merge commit.
- "Do not hand-merge runtime_package_pb2.py or edit its serialized bytes. Regenerate it with the local libprotoc 35.1 binary from the cleanly merged helios/proto/runtime_package.proto, then confirm field 30, enum value 10, and oneof tag 10 are present." — ACCEPTED: already the drafted step 1.
- "Update the design-review trailer to design-review=3ad6 only after actually re-reviewing the staged checklist, not by swapping the hex." — ACCEPTED: step 3 reviews each message against the staged `3ad6` checklist and prose skill before the trailer is written.
- "Rewrite all 3 commit messages and the PR body in the required order … Remove process assurances, correct the commit count from two to three, correct concurrency to eight default and sixteen ceiling, and label old-base or fork-built GPU evidence." — ACCEPTED: steps 3 and 4; old-base labeling was added to step 3 in response.
- "Run only targeted, memory-capped tests because of the earlier OOM and the 16 GiB limit, and rely on CI for heavy GPU validation." — ACCEPTED: step 2 narrowed from the full ten-path suite to the resolution-affected set, with old-base figures scoped in the messages. This revision addresses the 0.85 caller agreement, which fell below the 0.9 floor only on this point. The "rely on CI" part needs no task: GPU acceptance was already out of scope and the PR states it untested. The 0.825 certainty, also under the floor, comes from the Gemini source's 0.7 agreement on the same test-scope point; with step 2 revised, the two peers disagree on no step.
- "Leave the PR #192 alias decision out of scope and flag it to maintainers." — ACCEPTED: out of scope. The maintainer's own `triage:design-review` comment on PR #229 already raises it, so no separate flag is sent. The rewritten body's overlap section names #192 beside #212, because the current body omits it.
