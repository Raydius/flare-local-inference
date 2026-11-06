<!-- Written by /rad-verify. Archive is blocked while absent or the Verdict is FAIL. -->
# Verification — qwen38-dflash2-speculation

## Verdict: PASS-WITH-FINDINGS

Date: 2026-11-05
Tier: tier 2

Record budget: changed_files 12, ticked_boxes 3, fanout_ceiling 4 (fan-out opened on changed_files), walkthrough_budget 90. Lint: `--tier-check` clean, exit 0.

## Dimensions
- Implemented: PASS — all three ticked boxes' cited lines exist as claimed (quotes below); `find` lists 10 files under `qualify/results/dflash2-requalify` (3 `summary.json` + 7 `measurements.json`) and 1 under `qualify/results/dflash2-long-context`; every record ends on `observations_omitted` (rebase c1-r0:17392, fold c1-r0:17482, rebase2 c1-r0:17266) with the top-level key order identical to the served exemplar `qualify/results/dflash2-served/dflash2-adaptive/c1-r0/measurements.json` (17 keys).
- Reachable: PASS (n/a for code) — no code was changed; `grep -rn "qualify/results"` across `*.py *.sh *.toml *.yaml *.yml tests/` hits nothing, so the records have no reader to reach (matches proposal.md Blast Radius 11). Every repo path the delta, the main spec, and `docs/TODO.md:14-18` cite resolves (`test -e`, `<run>` expanded to rebase/fold/rebase2); `git check-ignore` exit 1, nothing gitignored; `git diff --stat openspec/specs` empty, main spec untouched until rad-sync.
- Tested: PASS — each ticked box's `Verify:` command ran once as written from the repo root: 1.1 → three `cmp` silent, `OK 7`, `du -sb` 6,121,848 < 10,000,000, exit 0; 1.2 → `cmp` silent, `OK`, 1 file, exit 0; 1.3 → grep printed `2`, exit 0. Round-trip `json.dumps(..., indent=2, allow_nan=False) + '\n'` reproduces all 11 new files byte-for-byte (`qualify/provenance.py:53` format); 0 `NaN|Infinity`, trailing newline present, no empty file.
- Optimal: advisory — one citation nit fixed in place (design.md:6 cited `qualify/dflash2_served.py:198` for `summarize`; the `def` is at :190, :198 is its `write_json` call) — no `discovery:` line exists, so fixed rather than reported. The extra `<run>/` level under `dflash2-requalify/` departs from the one-run-per-directory exemplars but is a recorded decision (design.md Decisions, "One `dflash2-requalify/<run>/` tree").
- Reused: PASS — no code added, so no import list applies; the "mirrors"/"key-for-key" claims (design.md:7, proposal.md Blast Radius 9) are backed by evidence: key order of all 7 records equals the served exemplar's 17-key order; `observations_omitted` wording equals the exemplars' (`qualify/results/dflash2-served/dflash2-adaptive/c1-r0/measurements.json:17132`, `qualify/results/dflash2-draft-sweep/adaptive/c1-r0/measurements.json:17466`) with the run name substituted; serialization equals the repository writer `qualify/provenance.py:53` and the draft-sweep exemplar (indent=2). No duplicated module, no "deliberately not reused" comment.
- Faithful: PASS after fix — proposal.md:11-18 forward: name → delta modifies `qwen38-dflash2-speculation`; "commit fold, rebase2, yarn4 … correct tok/s figures" → tasks 1.1/1.2 (trees exist) + delta:4 (path corrected, figures 329.1/516.5 kept); rebase committed → `qualify/results/dflash2-requalify/rebase/summary.json`; per-run dirs/stripped records → task 1.1; yarn4 summary only → task 1.2 (1 file); spec corrected by path, fold at delta:8, yarn4 at delta:19; TODO pointer lines → `git diff docs/TODO.md` one hunk, 5 insertions inside `qwen38_dflash` (`docs/TODO.md:229` content moved to :234 unchanged; `dflash2-served` untouched); no code → `git status` lists only `docs/TODO.md`, the change dir, and the two result dirs. Reverse: every working-tree change traces (above). The delta initially carried text beyond "no other requirement text changes" (finding 1, fixed — delta trimmed; now spec text differs from the main spec only at the three cited paths plus the rebase entry id inside the citation parenthesis).
- Environment: clean — `radflow-lint --doctor`: "all workflow controls present".

## Evidence audited
- [1.1 Create `qualify/results/dflash2-requalify/{rebase,fold,rebase2}/` ] — qualify/results/dflash2-requalify/rebase/summary.json:11 — "\"median\": 329.07749241001005,"
- [1.2 Copy `/home/rjdollete/.cache/dflash2-served/yarn4/summary.json` by] — qualify/results/dflash2-long-context/summary.json:188 — "\"ratio\": 1.7533304684142674,"
- [1.3 In `docs/TODO.md`, extend the open `qwen38_dflash` entry (the bull] — docs/TODO.md:14 — "Requalification runs (rebase 2026-11-02 adaptive, fold 2026-11-03 adaptive, rebase2 2026-11-04"

Skipped boxes: none (3 ticked, all verified).

## Findings
1. suggestion — openspec/changes/qwen38-dflash2-speculation/specs/qwen38-dflash2-speculation/spec.md:4,8,10-12,15 (pre-fix numbering) — the delta added requirement text beyond the path corrections proposal.md:16 scopes it to ("no other requirement text changes"): a trailing sentence at :4 ("The committed `qualify/results/dflash2-served/summary.json` … is not the baseline"), a record-shape parenthetical at :8, a new `#### Scenario: Serving entry probe` at :10-12 with no counterpart in `openspec/specs/qwen38-dflash2-speculation/spec.md:132-138`, and a new SHALL sentence at :15. Spec-only text, no observable behavior change, so partial coverage rather than FAIL condition 6. Fixed: the four additions were removed; the delta now differs from the main spec only at the cited paths (delta:4, :8, :19). The rebase2-is-not-a-requalification reading stays recorded in design.md:23 and `docs/TODO.md:14-16`.
   Dimension: Faithful
2. low — openspec/changes/qwen38-dflash2-speculation/design.md:6 — cited `qualify/dflash2_served.py:198` as `summarize`; `def summarize(output):` is at `qualify/dflash2_served.py:190` (:198 is the `write_json` call inside it). Fixed in place (citation now :190). No `discovery:` line to reference, so fixed rather than reported as Optimal.
   Dimension: Implemented

## Advisory rules judged
- Stage decay: none — tasks.md holds the three original tasks, no `## Follow-ups` heading, no task added after verification.md existed, no `discovery:` lines at all; the two fixes above are artifact repairs inside the proposal's stated scope, not scope expansion.
- Declined discoveries: n/a — tasks.md carries no `discovery:` or `declined:` lines.

## Live UI pass
n/a — no UI surface
