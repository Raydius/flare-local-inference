<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again.
     Written by /rad-verify. Archive is blocked while absent or the Verdict is FAIL. -->
# Verification — commit-dflash2-draft-sweep-curve

## Verdict: PASS

Date: 2026-11-05
Tier: tier 2 (`radflow-lint --print-record-budget` row, pasted; `changed_files 20`, `ticked_boxes 2`, `fanout_ceiling 4`, `walkthrough_budget 90`)

Re-run after the prior FAIL (condition 3): the proposal now declares `Tier: 2`, matching `--suggest-tier` (`suggested: Tier 2 — blast radius cites 5 file(s) across 2 top-level dir(s)` / `declared: Tier 2`), and `design.md` carries the filled `## External Review` quorum block plus `## Bomb-squad findings`. `--tier-check` still warns (exit 1, not 2) that the 20-file / 2-line diff exceeds the Tier 2 envelope (≤15 files) — the envelope is exceeded only by 18 generated data files — but no control was voided: the peer-review quorum (Tier 2 control) ran and the bomb-squad hunt (Tier 3 control) was dispatched as a precaution, so FAIL condition 3 does not hold. Fan-out gate fired (`changed_files 20` > `fanout_ceiling 4`): the Reused + Reachable + convention-fit track went to one `rad-investigator`; Implemented, Tested and Faithful ran here, and every child citation below was re-opened and confirmed.

## Dimensions
- Implemented: PASS. 19 untracked files under `qualify/results/dflash2-draft-sweep/` (`find` → `summary.json` + 18 `<arm>/c<ceiling>-r0/measurements.json` for `adaptive`, `fixed-0..7` × `c1-r0|c8-r0`); `cmp` against `/home/rjdollete/.cache/dflash2-served/sweep/summary.json` silent; summary top-level keys `['arms', 'trials']` with 9 arms, as `qualify/dflash2_draft_sweep.py:67-70` `summarize` emits. Python pass over all 18 trial records: each is byte-equal to `json.dumps(d, indent=2, allow_nan=False)+'\n'` (`qualify/provenance.py:50-54` `write_json` format), has no top-level `observations`, equals its cache counterpart on every surviving key in the same key order, and ends with the `observations_omitted` note (`qualify/results/dflash2-draft-sweep/adaptive/c1-r0/measurements.json:17466`). Footprint 14,592,149 bytes (`du -b` total; `du -ch` rounds to 15M) < 25M. `docs/TODO.md:12-13` holds the pointer inside the open `qwen38_dflash` entry (`docs/TODO.md:7`). Cache run untouched (`find … -newermt 2026-11-05` empty). `git check-ignore` exit 1 — not ignored.
- Reachable: PASS. No code consumer of the results tree: `grep -rn "qualify/results" --include="*.py" .` → no match (exit 1), nor any hit in `*.sh *.toml *.yaml` or a Makefile (child track, re-run here). The committed curve is reached from prose only, as intended: `docs/TODO.md:13` (new), alongside the existing `docs/TODO.md:11` served pointer and `openspec/specs/qwen38-dflash2-speculation/spec.md:133`. Producer chain intact and unchanged: `qualify/dflash2_draft_sweep.py:27` `from qualify.provenance import write_json`; `qualify/speculative_benchmark.py:549,:563` write the full records the strip starts from. Nothing orphaned.
- Tested: PASS. Box 1.1 `Verify:` ran as written → `cmp` silent, `OK 18`, exit 0. Box 1.2 `Verify:` `grep -c "dflash2-draft-sweep/summary.json" docs/TODO.md` → `1`, exit 0. No `.py` changed (`git diff --stat` → `docs/TODO.md | 2 ++`), so no repo suite bears on the change; nothing untested shipped.
- Optimal: PASS (advisory). The 18 new records follow the repo's one shared serializer contract (`qualify/provenance.py:53`, `indent=2`), whereas all 16 committed `dflash2-served` records are `indent=1` (`qualify/results/dflash2-served/dflash2-adaptive/c1-r0/measurements.json:2` ` "arm": {` vs sweep `:2` `  "arm": {`); the new tree is the conforming one and design.md:46 already records the served set's drift as out of scope. No `discovery:` line exists, so no Optimal finding is admissible; none was warranted.
- Reused: PASS. Import list: the only code touched by the change is none — zero new `.py`, zero new module (`git status --porcelain` → ` M docs/TODO.md` + the data tree). The one serializer the change leans on is the repo's shared `write_json` (`qualify/provenance.py:50`), already imported by the producer (`qualify/dflash2_draft_sweep.py:27 from qualify.provenance import write_json`), and the committed bytes reproduce its output exactly. The "mirrors the committed `dflash2-served` shape key-for-key" claim holds structurally: identical 17-key set and order, `observations_omitted` last in both (served `:17132`, sweep `:17466`), note text identical apart from the run path (`~/.cache/dflash2-served/full-raw` vs `~/.cache/dflash2-served/sweep`). No reusable strip mechanism ever existed to import: `observations_omitted` appears in no `.py`, no code `pop`s/`del`s `observations`, and `git log -S observations_omitted` points only at `25640c8d`, whose strip is recorded as an in-session pass (`openspec/changes/archive/2026-11-02-qwen38-dflash-served-comparison/tasks.md:47`). design.md:39 states the strip was an inline pass — no artifact claims an import it lacks, so no false reuse claim.
- Faithful: PASS. Bullet 1 (commit the curve) → task 1.1, the 19 files. Bullet 2 (summary.json + 18 stripped measurements mirroring the served convention at `qualify/results/dflash2-draft-sweep/`) → task 1.1; shape, layout and note mirrored key-for-key (Reused line), serialization follows `write_json` rather than the served files' `indent=1` — a documented choice (design.md Approach, design.md:46), not a divergence from the requirement's convention of summary + stripped records + note. Bullet 3 (reuse the served strip mechanism, no new qualify module) → covered: no qualify module was added; the served "mechanism" was itself an ad hoc pass (archive tasks.md:47), so the change did the same with the shared serializer format and says so (design.md:39). Bullet 4 (TODO pointer; no server rerun, sweep-tool, controller or served-comparison change) → task 1.2, `docs/TODO.md:12-13`; `git diff` shows no other file. Reverse: the only change to existing content is the two-line TODO hunk `@@ -9,6 +9,8 @@`, traced to bullet 4; no unasked behavior change.
- Environment: `[radflow-lint] doctor: all workflow controls present` (exit 0). `--tier-check` exit 1 warn (20 files > Tier 2 envelope) noted above; no control voided.

## Evidence audited
<!-- Run 'radflow-lint --print-evidence <change-dir>' and PASTE its lines — it emits
     one per ticked box with the box text and the cited path already filled, so nothing
     here is re-typed. The machine never does the audit: its quote slot ships EMPTY and
     the quote is your own work — open the cited file and quote a line you read there.
     radflow-lint then fails this file (exit 2, a violation naming each box) when a line
     under this heading quotes nothing, or still carries an empty ("") or placeholder
     (<...>, TODO, TBD, n/a, none) quote — the leg that stops a tool dump from passing as
     an audit. If >20 boxes: all production-code boxes plus every 2nd remaining box;
     list the skipped. -->
- [1.1 Create `qualify/results/dflash2-draft-sweep/` from the verified 20] — qualify/results/dflash2-draft-sweep/adaptive/c1-r0/measurements.json:17466 — "observations_omitted": "raw per-request events and analyses are kept out of the repository; the full output is ~/.cache/dflash2-served/sweep"
- [1.1 Create `qualify/results/dflash2-draft-sweep/` from the verified 20] — qualify/results/dflash2-draft-sweep/summary.json:11 — "median": 405.46289812762694,
- [1.2 In `docs/TODO.md`, extend the open `qwen38_dflash` entry (the bull] — docs/TODO.md:13 — "`qualify/results/dflash2-draft-sweep/summary.json`."

Box 1.1 verify command reproduced here (exit 0, `OK 18`); box 1.2 verify command reproduced here (`1`, exit 0). No box skipped (2 ticked boxes).

## Findings
<!-- Each finding names its dimension. Optimal also needs a Discovery: back-reference to a
     tasks.md discovery: line — with none, fix it rather than report it. Environment
     observations go on the Environment: line, never here. No '## Follow-ups' in tasks.md. -->
none

## Advisory rules judged
- Stage decay: not observed. `tasks.md` holds the two original boxes only (mtime 18:39, before either verification pass); no `## Follow-ups` heading, no `discovery:`/`correction:` lines; the design.md "Bomb-squad findings" out-of-scope list names other changes' debt without adding tasks here.
- Declined discoveries: n/a — no `discovery: … — declined:` lines in `tasks.md`.
- Armed graphs: n/a — no column-0 `Concurrency: graph` in `tasks.md`.

## Live UI pass
<!-- Filled UX section: the owner's task walkthrough — one column-0 walkthrough: line per on-screen
     Stated Requirement and width (narrow = below the repo's smallest breakpoint, else 390 px; wide = 1440 px),
     artifact under .playwright-cli/. A 'not met' line is a Dimension: Faithful finding AND FAILs
     verification. Cannot run: "skipped — unavailable: <reason>" (filled UX: plus a low Dimension: Tested
     finding); no UI: "n/a — no UI surface". Detector armed: also "detector: clean" or
     "detector skipped — unavailable: <reason>". -->
n/a — no UI surface
