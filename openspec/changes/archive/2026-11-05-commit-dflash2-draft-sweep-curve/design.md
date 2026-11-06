<!-- 1. Write this file. 2. Run `radflow-lint <change-dir>`; its message is the only rule check. 3. Exit 0 or 1: the file is done — stop and go to the next step. Exit 2: change the one line the message names, then run it again. -->
# Design — commit-dflash2-draft-sweep-curve

## Approach

- `qualify/results/dflash2-draft-sweep/summary.json` — byte copy of the cache run's summary (`qualify/dflash2_draft_sweep.py:67` `summarize` output); it is the whole 18-trial curve and the file docs cite.
- `qualify/results/dflash2-draft-sweep/<arm>/c<ceiling>-r0/measurements.json` × 18 — the cache trial record minus the top-level `observations` list (~18M → ~700K each) plus an `observations_omitted` note naming the cache path, serialized with `qualify/provenance.py:49` `write_json`'s format so the bytes match what the generator would emit; this mirrors the committed `qualify/results/dflash2-served/` shape key-for-key.
- `docs/TODO.md` `qwen38_dflash` entry — one pointer line to the committed summary, so the curve is reachable from the open-work record.
- No code: no reader exists for `qualify/results` (grep empty), the sweep tool and benchmark are untouched, the cache run is never modified. Verification is inspection — `cmp` on the summary, key-set equality against the served exemplar, size < 25M.
- Specs: `.openspec.yaml` `skip_specs: true` — the archived `dflash2-draft-sweep` capability already specifies the sweep's output contract; committing one run of it changes no observable behavior.

Decisions:
- Strip observations rather than commit raw (~330M): the committed served record set the precedent and the raw events remain in the cache path the note names.
- Byte-exact summary rather than a reformat: the summary is the citation target; any re-serialization would make the committed and cache copies differ for no gain.

Risks:
- [Cache dir wiped before this lands] → the 19 files already exist in the work tree; the operator's commit is the only remaining step.
- [Curve read as a verdict] → the summary rows carry `passed: false` from semantic manual-review cases, same as the committed served cells; the TODO pointer names it a recorded curve, not a pass.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.95
Caller Agreement: 0.95
Arbiter: ok
Sources:
- anthropic-rpa-claude-sonnet-5-5 (claude-sonnet-5-5) — cancelled (quorum met before it responded), agreement n/a
- google-gemini-models-gemini-flash-lite-latest (models/gemini-flash-lite-latest) — ok, agreement 0.95
- local-ai-qllm (qLLM) — ok, agreement 0.95

### Review Feedback

Consensus feedback recorded verbatim:
- "copying `summary.json` verbatim and processing the 18 per-trial `measurements.json` files to remove the heavy `observations` list while adding an `observations_omitted` pointer key referencing the cache source. Place these 19 files under `qualify/results/dflash2-draft-sweep/`, and add a single pointer line in `docs/TODO.md`." — ACCEPTED: this is the Approach as delivered (tasks 1.1 and 1.2).
- "The expected blast radius is strictly confined to these 19 new JSON files (~14M total) and one documentation line, with zero runtime, code, or logic impact since no code reads `qualify/results`." — ACCEPTED: matches proposal Blast Radius items 1–6.
- "Risks to guard against include data loss from cache cleanup prior to committing, accidentally committing unstripped raw files (bloating repo history), and schema non-compliance or path staleness." — ACCEPTED: the files already sit in the work tree (cache-loss risk closed); task 1.1's verify checks key-set equality against the served exemplar and total size < 25M, which an unstripped file (~18M each) would fail.
- "The optimal implementation involves a programmatic script to strip keys and validate JSON correctness (e.g., using `jq` or stream processing), and nothing needs to re-run as the data already exists." — ACCEPTED in part: the strip was an inline Python pass using the repo's `write_json` format; a committed script was not added because it has one caller and the task's verify command already validates the result.

## Bomb-squad findings

Dispatched as a precaution: `--suggest-tier` prints Tier 2 (blast radius) while `--tier-check` warns the 20-file diff exceeds the Tier 2 envelope; running the Tier 3 control leaves no control voided either way.
- Root cause: no repo rule names where qualification evidence lives (`AGENTS.md:317-322` only says link from docs/TODO.md; `docs/model-preparation.md:512,516` document `--output DIR` with no destination); the sweep change scoped its results to cache + vault (`openspec/changes/archive/2026-11-04-dflash2-draft-sweep/proposal.md:59`), so the gap is process, not a tool default — every qualify tool requires `--output`.
- This change's 19 files: no latent defect — all byte-identical under `write_json` format, no NaN/Infinity, key order preserved, none unstripped, no `.gitignore` rule reaches `qualify/results` (`git check-ignore` exit 1).
- Out of scope, same class (left for their own changes): `openspec/specs/qwen38-dflash2-speculation/spec.md:133` cites 329.1 / 516.5 tok/s against `qualify/results/dflash2-served/summary.json`, which holds 271.3 / 512.7 — the numbers are from the uncommitted `~/.cache/dflash2-served/rebase/summary.json`; the yarn4 long-context probe (`spec.md:155`) and the fold / rebase2 requalifications are cache-only; `docs/TODO.md:229` cites `~/benchmarks/gather-byte-ladder/` which no longer exists; 16/17 committed `dflash2-served` measurements are `indent=1`, not `write_json` output.
