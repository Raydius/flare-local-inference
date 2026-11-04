## ADDED Requirements

### Requirement: The recipe declares a YaRN x4 checkpoint
The `qwen38_dflash` recipe SHALL declare `qwen3.8-flash-dflash2-yarn4`: the same pinned target revision, draft revision, and declared byte counts as `qwen3.8-flash-dflash2`, with a serving context of 1,048,576 tokens through frequency-only rope interpolation of factor 4 anchored at the 262,144-token native context, exactly as the qwen38 recipe declares `qwen3.8-flash-yarn4`. `qwen3.8-flash-dflash2` SHALL stay the first declaration, so bare repository and directory resolution keep naming it. The scaling SHALL apply to the target's rotary tables and the package identity as it does for qwen38; the draft's rotary and 2048-position ring SHALL take positions beyond 262,144 unchanged.

#### Scenario: Alias export at the extended context
- **WHEN** the operator exports `qwen3.8-flash-dflash2-yarn4`, or `qwen3.8-flash-dflash2` with `--rope-parameters {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}`
- **THEN** the plan declares a 1,048,576-token context with four times the native plan's demanded pages, preparation receives that context and scaling, and the package's identity differs from the native package's

#### Scenario: Directory export through the console
- **WHEN** a checkpoint directory is exported with `--recipe qwen38_dflash --draft-model DIR --rope-parameters {"factor": 4}`
- **THEN** resolution succeeds with the same declared draft bytes as the base alias, instead of failing `qwen3.8-flash-dflash2 declares no rope scaling`

#### Scenario: Supported models list
- **WHEN** an unknown model name is given to `ignition export`
- **THEN** the "supported models" message lists `qwen3.8-flash-dflash2-yarn4` directly after `qwen3.8-flash-dflash2`

### Requirement: A scaled package serves and drafts past the native context
A package exported from `qwen3.8-flash-dflash2-yarn4` SHALL serve a retrieval prompt longer than 262,144 tokens through `flare run` and quote records that sit beyond the native context, and the draft's acceptance past the native context SHALL be measured against its acceptance below it on the same package: `qualify/dflash2_long_context.py` sends one prompt of about 320,000 tokens and one of about 200,000 (past-native first, so no prefix is reused), with history drafts off so the counters measure the DFlash2 draft alone, and records for each the retrieval result, offered and accepted draft counts, accepted tokens per round, and output tokens per second. Both prompts SHALL pass retrieval, and a past-native accepted-per-round more than 10 % below the sub-native one SHALL be reported as a finding with both numbers, never absorbed and never answered by scaling the draft inside this change.

#### Scenario: Retrieval past the native context
- **WHEN** the past-native prompt asks for the timeout, retry limit, and owner of the seventh, middle, and last service records
- **THEN** the answer quotes all three records' values, including the last record's, which lies beyond position 262,144

#### Scenario: Acceptance buckets
- **WHEN** the probe finishes both prompts
- **THEN** `summary.json` holds the two prompts' token counts, retrieval results, draft counters, and accepted-per-round side by side, and the ratio past-native over sub-native is printed with the 10 % criterion

#### Scenario: Frozen workload untouched
- **WHEN** the frozen speculation workload is validated after the change
- **THEN** it still holds the 28 ordered cases with every case within 262,144 tokens, and the served comparison's arms and baselines are unchanged
