# Verification — scheduler-copy-mask-acceptance

## Verdict: PASS

Date: 2026-11-04
Tier: 2

## Dimensions
- Implemented: PASS — all 4 ticked boxes confirmed in the working tree. `helios/src/scheduler/speculation.rs:137-160` carries the new `copy_mask: i64` parameter and the per-depth curve selection (`:154-158`); `helios/src/scheduler.rs:2531` passes `vector.copy_offered_mask` at the only production `acceptance.observe` call; `docs/TODO.md` no longer contains "ignores the copy mask" (grep exit 1) and the packed-prefix item at `:34-36` is retained; commit `6657f7e5` carries `Design-Signed-Off-By … design-review=a38f`, and `sha256(docs/design-review.md)[:4]` is `a38f`, which `.githooks/commit-msg:62` checks. The test-only `model/tests.rs` repairs named in the second discovery are present at `:719,739,768,782`.
- Reachable: PASS — `observe` is reached from the commit path `RequestScheduler` at `helios/src/scheduler.rs:2524-2532` under `GraphKind::Speculative`; `copy_mask` flows from `CompletionRecord.vector.copy_offered_mask` (`helios/src/execution/transport.rs:31`, bounded at `:74-75`). Readers of the de-inflated curve are unchanged and still reached: `rounds.rs:87,137,160,318`, `policy.rs:524`, `service.rs:236`. The 65 `.observe(` call sites across helios/src are the one production caller plus test callers, all updated to the new arity (suite compiles).
- Tested: PASS — Box 1 `cargo test --manifest-path helios/Cargo.toml --features server --lib scheduler::speculation` → exit 0, 6 passed (incl. `copied_depths_update_the_copy_curve_and_leave_the_mtp_curve_alone`). Box 2 `cargo test … --lib scheduler::` → exit 0, 280 passed, 0 failed, 2 ignored. Read against the diff, the new test fails without the change (depth 0..1 means on the MTP curve would move toward 1, breaking `assert_eq!(after_mtp[..2], mtp[..2])`), so it discriminates. Box 3: no `Verify:` clause — no covering command; its Done-when grep reproduced (no output, exit 1). Box 4: no `Verify:` clause — no covering command; digest check reproduced by hand. Both runs with `PROTOC=/tmp/protoc-vendored/protoc-bin-vendored-linux-x86_64-3.2.0/bin/protoc` (no system protoc on this host).
- Optimal: no findings. One pass over `0..observed`, one index from the bit, no allocation, as the accepted review feedback asked; the predicate is the two-term boolean `metrics.rs:426-427` also uses, duplicated inline rather than extracted (a two-line expression; no `discovery:` line, nothing to report).
- Reused: PASS — no new component, hook, or helper was added; the change widens an existing function with one parameter. Import list: `helios/src/scheduler/speculation.rs:12` defines `COPY_DRAFT_CAPACITY`, consumed in-file at `:154`; `helios/src/server/metrics.rs:426` reads `crate::scheduler::speculation::COPY_DRAFT_CAPACITY` for the served-counter split that the learner now matches. The artifacts claim "the same predicate", not shared code; no false reuse claim.
- Faithful: PASS — the single Stated Requirements bullet ("Scheduler copy-mask acceptance fix", scoped by the four Assumed lines to `docs/TODO.md:38`) is delivered by tasks 1–2 and the `speculation.rs:154-159` / `scheduler.rs:2531` hunks; task 3 closes the TODO item. Reverse trace: the only behavior change (per-depth curve routing) traces to the Root Cause; the `model/tests.rs` edits are test-only compile repairs recorded as a `discovery: … — fixed`; the 16 test-caller updates are recorded as a `discovery: … — fixed`; the Python mirror (`ignition/direct/scheduler.py:221`, no source axis) and `rounds.rs:206` forecast path are untouched as the Assumed lines state. The spec delta extends the existing requirement to the learned curve without altering the recorded-mask bound.
- Environment: clean — `radflow-lint --doctor`: all workflow controls present; tier-check clean; no system `protoc` (vendored binary used for the Verify commands).

## Evidence audited
- [In `helios/src/scheduler/speculation.rs` add a `copy_mask: i64` parame] — helios/src/scheduler/speculation.rs:154 — "let curve = if depth < COPY_DRAFT_CAPACITY && copy_mask & (1i64 << depth) != 0 {"
- [In `helios/src/scheduler.rs` (the `request.acceptance.observe(` call n] — helios/src/scheduler.rs:2531 — "vector.copy_offered_mask,"
- [Remove the closed item from `docs/TODO.md:38-40` ("Scheduler acceptanc] — docs/TODO.md:34 — "- [ ] Packed-prefix confidence is recorded from the learned token at positions a copy later"
- [Before committing follow `docs/design-review.md` (staged-diff review a] — .githooks/commit-msg:62 — "if match[2] != hashlib.sha256(document).hexdigest()[:4]:"

## Findings
none

## Advisory rules judged
- Stage decay: none — no `## Follow-ups` heading; tasks.md was last written at the commit time (19:34) with both `discovery:` lines describing apply-time observations (test-caller arity and a `--lib` compile break hit on the first Verify run); no verification.md existed before this pass.
- Declined discoveries: n/a — both discoveries are `— fixed`; no `declined:` lines.

## Live UI pass
n/a — no UI surface
