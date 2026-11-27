# Verification — reorganize-fork-readme

## Verdict: PASS-WITH-FINDINGS

## Evidence audited
- [1. Rewrite README.md above `## Active patches`: keep the H1 `# Raydius] — README.md:1 — "# Raydius fork: patch notes"
- [2. Add a `## RTX 5060 Ti compatibility` section above `## Active patch] — README.md:27 — "## RTX 5060 Ti compatibility"
- [3. Name the upstream repository `NVIDIA/hyperflare-inference`: the REA] — README.md:3 — "This fork carries local patches on top of [NVIDIA/hyperflare-inference](https://github.com/NVIDIA/hyperflare-inference), the upstream repository formerly named `NVIDIA/flare-local-inference` and `NVIDIA/spark-local-inference`."

## Findings
1. medium (fixed) — FORK.md:213 — Dropping the README's History paragraph lost facts: its 2026-11-26 entry named `d0adb958`, `0a4a6eb1`, #431, #507, #532, the KV-compression switches, the digest `eb2f` and the recipes range (#18, #21, `a2746f49`, `be1a9a11`), and its 2026-11-25 entry named #405, none of which FORK.md's entries for those dates held, so task 1's condition that every fact stays or is in FORK.md's record did not hold. The items are now in FORK.md's two entries; a comparison of every backticked or #-numbered identifier of each dropped entry with FORK.md's entry of the same date reports none missing (`docs/TODO.md` is covered there as "the recipe entries of its follow-up list").
   Dimension: Implemented

## Code review
Review standard: 001ce8ddb3cf5aa072e28e2cc5bbab28ff39cdc0
Peer part: path=FORK.md part=1/1 blob=1143e18042307ca847dc6997fd2360885d5e5eda status=returned reason=quorum achieved, tier 2
Peer part: path=README.md part=1/1 blob=26b2149c171abdf2ab90bacc7693e9915232b7ae status=returned reason=quorum achieved, tier 2
Peer coverage: complete
Quorum: achieved (weight 4 of 3 required, tier 2)
Certainty: 0.875
Caller Agreement: not returned
Arbiter: consensus returned
Sources:
- claude-haiku-5-5 — ok, agreement 0.85
- qLLM — ok, agreement 0.1
- models/gemini-flash-lite-latest — ok, agreement 0.4
- deepseek-flash — ok, agreement 0.3
- claude-sonnet-5-5 — ok, agreement 0.9
Both files were edited after the peer call to fix the findings below; the blobs above are the ones the peers received.
Own review: security none (public names, commit ids and URLs only; the secret scan's matches are the word "token" in prose); composition: anchors `FORK.md#branch-model`, `docs/release.md#release-tags` and `docs/container.md#publish` exist, and the README's links to `NVIDIA/hyperflare-inference`, PRs #211, #212 and #229, the recipes fork, its follow-up list and the Hugging Face core return HTTP 200.
Consensus findings, confirmed against the files:
- CONFIRMED, medium (fixed): the base-move record lacked items of the dropped History entries — Finding 1.
- CONFIRMED, low (fixed): README.md:33 said serving "Works" while README.md:41 says no core has been loaded on the card on `c915b5f0`; the row now reads "Worked before the move to `c0c4f798` … not yet run on `c915b5f0`", and the summary at README.md:29 says "has served".
- CONFIRMED, low (fixed): README.md:21 said each retired patch names the upstream commit that replaced it; some retired entries name a recipe, a FlashInfer PR or a behavior instead, so it now reads "says what replaced it, usually an upstream commit".
- NOT A DEFECT: "one fork commit" for FlashInfer `675984cd` — `git log --oneline 0d48af53..675984cd` in the FlashInfer clone lists one commit.
- NOT A DEFECT: the RTX 5060 Ti section exceeds the opening's reorganization — the operator asked for it ("Also create a separate section to discuss 5060ti compatibility", proposal.md Stated Requirements).
Own low findings: the `flare run` error example's GPU index depends on the host and is written as "GPU <index>" (README.md:35, fixed before the peer call); the export refusal rests on `heliopause/src/export/capture.rs:2164` and the platform lists, not on an export attempted on `c915b5f0`.
