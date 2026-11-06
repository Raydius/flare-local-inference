## MODIFIED Requirements

### Requirement: The folded package serves the measured baselines
A package exported from the folded recipe SHALL serve the frozen speculation workload through `flare run` with clean transport and no failed requests, and its code_edit output throughput SHALL be compared against the pre-fold package's 329.1 tokens per second at concurrency 1 and 516.5 at concurrency 8 (`qualify/results/dflash2-requalify/rebase/summary.json`, the rebase entry gb202-tp1-202611021657; vault DFlash2 Recipe Progress §12), with the drafter's offered and accepted token counts recorded beside them.

#### Scenario: Requalification
- **WHEN** `qualify.dflash2_served` runs trials `0:1:dflash2-adaptive` and `0:8:dflash2-adaptive` against the new entry
- **THEN** both trials report `transport_passed` with zero failed requests, and the medians and acceptance counts are recorded against the baselines in `qualify/results/dflash2-requalify/fold/summary.json`; a median more than 5 % below its baseline is reported as a finding, never absorbed

### Requirement: A scaled package serves and drafts past the native context
A package exported from `qwen3.8-flash-dflash2-yarn4` SHALL serve a retrieval prompt longer than 262,144 tokens through `flare run` and quote records that sit beyond the native context, and the draft's acceptance past the native context SHALL be measured against its acceptance below it on the same package: `qualify/dflash2_long_context.py` sends one prompt of about 320,000 tokens and one of about 200,000 (past-native first, so no prefix is reused), with history drafts off so the counters measure the DFlash2 draft alone, and records for each the retrieval result, offered and accepted draft counts, accepted tokens per round, and output tokens per second. Both prompts SHALL pass retrieval, and a past-native accepted-per-round more than 10 % below the sub-native one SHALL be reported as a finding with both numbers, never absorbed and never answered by scaling the draft inside this change.

#### Scenario: Retrieval past the native context
- **WHEN** the past-native prompt asks for the timeout, retry limit, and owner of the seventh, middle, and last service records
- **THEN** the answer quotes all three records' values, including the last record's, which lies beyond position 262,144

#### Scenario: Acceptance buckets
- **WHEN** the probe finishes both prompts
- **THEN** `qualify/results/dflash2-long-context/summary.json` holds the two prompts' token counts, retrieval results, draft counters, and accepted-per-round side by side, and the ratio past-native over sub-native is printed with the 10 % criterion

#### Scenario: Frozen workload untouched
- **WHEN** the frozen speculation workload is validated after the change
- **THEN** it still holds the 28 ordered cases with every case within 262,144 tokens, and the served comparison's arms and baselines are unchanged
