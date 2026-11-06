## Purpose

Measure whether a served Qwen3.8-Flash-Next package retrieves a passkey placed at a chosen depth of a prompt, below and beyond the native 262,144-token context, so a rope-scaled package's long-context quality is recorded per package against the native package.

## ADDED Requirements

### Requirement: Passkey prompt construction

`qualify/long_context_passkey.py` SHALL build a chat request whose user message is the frozen workload's service catalog sized by `dflash2_long_context.records_for` to a target prompt-token count, with one passkey sentence `The pass key is <passkey>. Remember it. <passkey> is the pass key.` inserted between catalog lines at `round(depth × records)`, followed by the question `What is the pass key? The pass key is`; the request SHALL carry the workload's sampling fields with `top_k=1`, `max_tokens` 32, a seed, and no token-id or timestamp echo, and the passkey SHALL be a five-digit string drawn per cell from a seeded generator that does not occur in the catalog text.

#### Scenario: Needle at depth

- **WHEN** `prompt(records=4800, depth=0.5, passkey="48213", model="ckpt")` is built
- **THEN** the catalog lines before and after the inserted sentence are exactly `_catalog(4800)` split at line 2400, the last message asks for the pass key, `top_k` is 1, `max_tokens` is 32, and `return_token_ids` is false

#### Scenario: Passkey absent from filler

- **WHEN** a passkey is drawn for any cell
- **THEN** the five-digit string does not occur in `_catalog(records)` for that cell's record count

### Requirement: Pass criterion and grid

A cell SHALL pass when the reply's content contains the cell's passkey; `summary.json` SHALL record per cell the prompt-token count, passkey, reply, `passed`, request seconds, and accepted drafts per round from the metrics delta, plus a `grid` of pass/fail keyed by target length then depth and an `all_passed` flag; the probe's exit status SHALL be 0 only when every cell passed.

#### Scenario: Reply carries the passkey

- **WHEN** the reply content is `48213.` for passkey `48213`
- **THEN** the cell passes

#### Scenario: Reply misses the passkey

- **WHEN** the reply content is `I do not know` or carries a different five-digit string
- **THEN** the cell fails, `all_passed` is false, and the probe exits 1

### Requirement: Serving order and packages

The probe SHALL serve one package per run through `dflash2_served.container_command` with history drafts off and the requested `--max-context-tokens`, and SHALL send cells longest length first, depths in ascending order within a length, so no cell's prompt is a prefix of an earlier one; a run against the native 262,144-token package SHALL take only lengths below the native context.

#### Scenario: Longest first

- **WHEN** the lengths are 200000, 320000, 520000, 1000000 and the depths 0.1, 0.5, 0.9
- **THEN** the first request is the 1,000,000-token cell at depth 0.1 and the last is the 200,000-token cell at depth 0.9, and the serving container is started once

#### Scenario: Native baseline

- **WHEN** the native MTP package is run with `--lengths 200000 --max-context-tokens 262144`
- **THEN** three cells run and the summary's grid has the single length key `200000`
