# Qwen 3.8 DFlash2 at 16 sequences and 1M context: upstream's `qwen38` path against the fork's `qwen38_dflash`

Measured 2026-11-10 on one RTX PRO 6000 Blackwell (96 GB, GPU 0), tensor parallel 1.

## Question

Flare has two ways to serve Qwen3.8-Flash-Next with the `incoai/Qwen3.8-Flash-Next-DFlash2` draft:
upstream's `--recipe qwen38` with speculative method `dflash2`, and this fork's `--recipe qwen38_dflash`,
whose addition is mixed prefill (graphs that prefill one request while decoding others). Upstream's path
rejects mixed prefill and offers two options the fork recipe lacks: A16 decode
(`preparation.force_moe_decode_a16`) and an MXFP8 draft (`--online-mxfp8 --draft-head-quantization mxfp8`).

The study asks, at 16 prepared sequences with a 1,048,576-token declared context: does the fork recipe
measure better than upstream's path, and do A16 decode and the MXFP8 draft help.

## Arms

Every arm is a LocalAI console export of the same checkpoint
(`huginnfork/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-Abliterated@v3`) on image `local/flare-ignition:f83d3120`
(commit `f83d3120`), with `--max-concurrency 16`, `--rope-parameters` YaRN factor 4.0 over 262,144, seven
draft tokens at fixed width (`--no-mtp-adaptive-verification`), and the same draft folder
(`config.json` sha256 `24095759…cc3b1b9`, `model.safetensors` sha256 `d6723f2a…4d3186f`).

| Arm | Recipe and options |
|---|---|
| baseline | `qwen38_dflash --mixed-prefill`, 8 sequences, image `7ebef730`; the package served before the study (`gb202-tp1-202611101936`) |
| U | `qwen38`, speculative method `dflash2`, `--mtp-draft-tokens 7` |
| Q | U plus `--online-mxfp8 --draft-head-quantization mxfp8` |
| A | U plus `force_moe_decode_a16 = true` |
| M | `qwen38_dflash --mixed-prefill` |

## Measurements

Each trial ran against a fresh console start of the arm's package.

- `code_edit`: the frozen speculation workload through `qualify.dflash2_console_probe` at concurrency
  ceilings 1 and 8, three trials each. The figure is the median over code_edit groups of output tokens per
  second, from `qualify.dflash2_console_rate`. The console's chat template changes prompt token ids, so the
  benchmark's own gated rate is withheld; compare these figures only with each other.
- 16 clients: `llm_decode_bench.py --host 127.0.0.1 --port 8100 --model Qwen3.6 --concurrency 16 --contexts 0
  --duration 30 --skip-prefill --display-mode plain --no-hw-monitor --no-resume --no-calibration-cache`,
  three trials; the figure is aggregate output tokens per second.
- 1M passkey: `qualify.long_context_passkey --base-url http://127.0.0.1:8100 --lengths 1000000`, depths
  0.1, 0.5 and 0.9 (1,009,836 prompt tokens).

Cells give the median of three trials and, in brackets, the minimum and maximum. No request failed in any
`code_edit` trial (28 requests each) and no 16-client trial reported a failure.

| | baseline (8 seq) | U | Q | A | M |
|---|---|---|---|---|---|
| Export | not exported here | ok, 28 min | ok, 29 min | ok, 31 min | ok, 92 min |
| Loaded GiB | 78.72 | 77.82 | 78.03 | 77.93 | 80.58 |
| KV tokens | 1,150,976 | 1,138,688 | 1,118,208 | 1,126,400 | 909,312 |
| Graphs (mixed) | 173 (81) | 109 (0) | 109 (0) | 109 (0) | 217 (108) |
| Lowest free GPU MiB during capture | not recorded | 12,779 | 12,593 | 12,713 | 10,421 |
| `code_edit` tok/s, concurrency 1 | 534.0 (516.5–544.0) | 537.1 (521.6–539.2) | 555.9 (548.7–572.2) | 511.2 (511.2–511.4) | 535.1 (530.5–541.9) |
| `code_edit` tok/s, concurrency 8 | 356.7 (316.1–383.5) | 395.3 (379.9–438.4) | 436.7 (425.8–455.6) | 413.3 (399.0–416.2) | 355.6 (352.2–369.1) |
| 16 clients, aggregate tok/s | 864.1 (862.6–868.4), 8 streams | 1272.7 (1247.3–1274.2) | 1231.2 (1217.0–1259.9) | 1148.0 (1144.0–1158.8) | 1262.5 (1238.8–1278.2) |
| Acceptance rate, concurrency 1 | 0.606 (0.574–0.607) | 0.588 (0.584–0.661) | 0.627 (0.601–0.629) | 0.616 (0.616–0.619) | 0.595 (0.576–0.609) |
| Acceptance rate, concurrency 8 | 0.608 (0.582–0.626) | 0.579 (0.563–0.617) | 0.630 (0.591–0.639) | 0.623 (0.581–0.632) | 0.602 (0.572–0.602) |
| Accepted drafts per round, concurrency 1 | 2.37 (2.33–2.45) | 2.30 (2.29–2.38) | 2.38 (2.38–2.44) | 2.33 (2.32–2.33) | 2.23 (2.15–2.29) |
| Accepted drafts per round, concurrency 8 | 3.72 (3.43–3.93) | 3.83 (3.53–4.37) | 3.96 (3.78–4.20) | 4.02 (3.77–4.06) | 3.53 (3.34–3.67) |
| 1M passkey | not run | 3 of 3 | 3 of 3 | 3 of 3 | 0 of 3, HTTP 400 |

The baseline serves 8 sequences, so 8 of its 16 clients decode and 8 wait in the queue for the whole
window; its 16-client figure is an 8-stream aggregate.

Arm M's KV cache holds 909,312 tokens, under the 1,048,576 the declared context needs, and the server
refused each 1,009,836-token passkey prompt with HTTP 400 in under a second. Its 108 mixed graphs raise the
loaded size by 2.76 GiB over arm U and leave that much less for the cache.

## Decision rule and recommendation

The rule was fixed before measuring: mixed prefill is worth porting onto upstream's path only if arm M
beats the best of U, Q and A on `code_edit` tokens per second by more than the run-to-run spread; A16 decode
and the MXFP8 draft are recommended only if tokens per second or acceptance improves and the 1M passkey
passes.

- **Retire `qwen38_dflash`; do not port mixed prefill on this evidence.** Arm M does not beat the best arm
  (Q) at either concurrency: 535.1 against 555.9 at concurrency 1 and 355.6 against 436.7 at concurrency 8,
  with ranges that do not overlap Q's. Against arm U it is equal within spread at concurrency 1 and lower at
  concurrency 8 (range 352.2–369.1 against 379.9–438.4). At 16 sequences it also cannot hold a 1M prompt,
  and its export takes 92 minutes against 28.
- **MXFP8 draft: recommended.** Against arm U, `code_edit` at concurrency 1 rises from 537.1 to 555.9 with
  ranges that do not overlap, and the passkey passes. At concurrency 8 and at 16 clients the ranges overlap
  (436.7 against 395.3; 1231.2 against 1272.7). It costs 20,480 KV tokens.
- **A16 decode: not recommended.** Against arm U it is lower at concurrency 1 (511.2 against 537.1) and at
  16 clients (1148.0 against 1272.7), both with ranges that do not overlap. At concurrency 8 its tokens per
  second lie inside arm U's range and its acceptance rate overlaps it (median 0.623 against 0.579, ranges
  0.581–0.632 and 0.563–0.617); no cell improves beyond the spread. The passkey passes.

## Limits of this evidence

- The workload starts few mixed rounds. In arm M the concurrency-8 trials launched 11, 11 and 12 mixed graphs (9 in
  the code_edit group, 129 to 170 tokens emitted) and each concurrency-1 trial 3; the 16-client run uses empty contexts.
  The rule therefore shows that mixed prefill does not pay for its memory and export cost in these
  workloads. Decode throughput while long prompts prefill concurrently, the case mixed graphs exist for,
  was not measured.
- The export log has no line naming A16 decode. Arm A's generated configuration carries
  `force_moe_decode_a16 = true`, its package added 71 GiB of stored data against 7 GiB for arm Q, and it
  holds 7,190 kernels against arm U's 6,969.
- Not measured: tensor parallel, history drafts on, more than one 1M prompt at once, the MXFP8 draft and
  A16 decode combined, direct BF16 feature averaging (no arm varies it), output quality beyond the passkey
  probe and acceptance counters.
- Three trials per cell; the concurrency-8 ranges are wide (arm U spans 58 tok/s).

## Files

`summary.json` holds every figure above per arm, with export stage timings. `<arm>/c<N>-r<R>/measurements.json`
is the benchmark's trial record without its per-request observations (the full output is
`~/.cache/dflash2-served/paths-c16/` on the measuring host); `<arm>/bench16-r<R>/measurements.json` is the
16-client cell without its per-request samples.
