# Design — qualify-qwen38-dflash2-on-6b68b9b4

## Approach

- Subject: entry `gb202-tp1-202611160316` of console profile *2026-11-14 Flare DFlash2 Q Rebase4* (profile id in `~/models/.cache/rebase-6b68b9b4/console/Q4-id.txt`, image `local/flare-ignition:3b4a1a2e`). No re-export; the export log is `~/models/.cache/rebase-6b68b9b4/console/Q4/export.log`.
- Reference: arm Q as recorded in `qualify/results/dflash2-paths-c16/summary.json` key `Q` (image `f83d3120`, measured 2026-11-10). Arm Q is not re-measured; old rows and files stay unchanged.
- Protocol, identical to arm Q's (`~/.cache/dflash2-served/paths-c16/arm.sh` and `trials.sh`): nine trials, each on a fresh console start, `c1:0 c1:1 c1:2 c8:0 c8:1 c8:2 bench16:0 bench16:1 bench16:2`; `code_edit` through `qualify.dflash2_console_probe` on the frozen workload `~/.cache/dflash2-served/workload2` at ceilings 1 and 8; 16 clients through `llm_decode_bench.py --concurrency 16 --contexts 0 --duration 30 --skip-prefill`; then `qualify.long_context_passkey --lengths 1000000` (default depths 0.1, 0.5, 0.9, default model id, as `arm.sh` ran it) on a fresh start. The saved driver `~/models/.cache/rebase-6b68b9b4/console/q-arm.sh` runs exactly this from the `deploy` tree `~/models/.cache/rebase-6b68b9b4/trees/tip-deploy`; it gains only the `passkey-metrics-final.txt` capture that `arm.sh` has.
- Figures: a collector modelled on `~/.cache/dflash2-served/paths-c16/collect.py`, restricted to `Q4`: medians and min/max of `code_edit_median_tok_s` (`qualify.dflash2_console_rate.rate`), acceptance rate and accepted drafts per round (`qualify.dflash2_draft_sweep.trial_row`, policy `fixed-7`), 16-client aggregate, loaded GiB and KV tokens from the export log, passkey grid. Trial files are written without per-request observations, as for the other arms.
- Recorded identity (peer feedback): image id, the Transformers distribution inside the image (`pip show transformers` → 5.23.0), KV capacity 1,449,984 tokens, graph count.
- Comparison: each median against arm Q's, as a percentage; a gap beyond ±8 % or a passkey cell below pass is flagged, with whether the two ranges overlap. The README states the result is the combined rebased stack (upstream `6b68b9b4` plus the fork's carry commits), attributes no gap to a single commit, and lists what stays unmeasured: numerical agreement of the `qwen38` forwards on Transformers 5.23.0 beyond acceptance counters and the passkey.
- Record: a new README section "Arm Q on upstream 6b68b9b4 (arm Q4, measured <date>)" after the arm QA section, in its table form (Q beside Q4); `summary.json` key `Q4`; `Q4/c<N>-r<R>/measurements.json` and `Q4/bench16-r<R>/measurements.json`; the Files section names `Q4`. One commit on `deploy`, hook on, digest `ed22`, following `docs/design-review.md` and the repo-prose skill.
- On a flagged gap or failed cell: one `docs/TODO.md` entry naming the metric, both figures and the results section. No investigation in this change.
- GPU 0: the owner stops arm Q from the console before the trials; `q-arm.sh` stops the server at the end; the owner's arm Q profile is started again unless the owner chooses otherwise.

## Revisions

- 2026-11-15 21:40, during apply of task 1: the operator corrected the subject. "When we started this whole change I was running A16, and was running it for DAYS.  That is clearly the profile I'm most interested in." / "Why would you rebase a profile that I wasn't even running?" / "I **specifically asked in the beginning why you were using "arm Q"** and did not get an answer". The plans of this change and of `rebase-deploy-onto-upstream-29192ef7` took arm Q (A16 off) as the operator's configuration from the paths-c16 study's recommendation; GPU 0's served profile was never checked, and the console marks *2026-11-10 Flare DFlash2 MXFP8 A16* as default. The primary subject becomes that profile: a copy *2026-11-15 Flare DFlash2 MXFP8 A16 Rebase4* (only `docker_image` differs; the profiles differ from arm Q's only in `config:preparation.force_moe_decode_a16`) is exported on `3b4a1a2e` and measured as arm QA4 with the same protocol, against arm QA (`summary.json` key `QA`: 518.7 / 388.8 / 1152.0 tok/s, passkey 3/3). Arm Q4, measured before the correction, stays as secondary evidence. Serving returns to the A16 profile, not arm Q.
- 2026-11-15 22:00, after the A16 copy's export had started: the operator withdrew the A16 arm — "Here is the updated plan: just finish the change as written.  I will export A16 and benchmark it myself." The A16 export was stopped through the console before it completed; the profile copy *2026-11-15 Flare DFlash2 MXFP8 A16 Rebase4* (id `2026-11-10-flare-dflash2-mxfp8-a16-1794383054083-copy-261115_213526`, image `3b4a1a2e`) stays in the console for the operator. This change measures arm Q4 against arm Q as first planned; GPU 0 goes to the operator's choice at the end.

## External Review

### Peer Review Verdict

Quorum: achieved (weight 2/2, escalated to tier 1)
Certainty: 0.875
Caller Agreement: 0.8
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.95
- google-gemini-flash-lite (models/gemini-flash-lite-latest) — ok, agreement 0.8
- local-ai-qllm (qLLM) — cancelled, quorum met first
- deepseek-flash (deepseek-flash) — cancelled, quorum met first

### Review Feedback

Consensus feedback recorded verbatim:
- "Log the image digest, the loaded Transformers version (5.23.0 vs. the pinned revision), and the reported KV capacity (1,449,984 tokens)." — ACCEPTED: the Approach's recorded-identity line; task 2 records them.
- "Add the new results beside the old arm in the README table and summary.json, leaving old rows unchanged." — ACCEPTED: already the record plan; stated explicitly under Reference.
- "numerical drift from the 269-line model-source difference is untested, so matching throughput does not show equivalence; the rebase bundles several changes, so gaps cannot be attributed to one of them" — ACCEPTED: the README states the combined-stack scope, attributes no gap to one commit, and lists numerical agreement as unmeasured.
- "three trials are a small sample; thermal or contention noise; passkey 3/3 is a sanity check, not a quality measure." — ACCEPTED: ranges are compared beside medians and overlap is reported; the limits list keeps the passkey's scope.
- Certainty 0.875 and caller agreement 0.8 are below the 0.9 floors — RECORDED: both peers agree with the protocol; the difference is the added identity logging and attribution caveat, both adopted above.
