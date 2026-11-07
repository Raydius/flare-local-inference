## MODIFIED Requirements

### Requirement: Passkey prompt construction

`qualify/long_context_passkey.py` SHALL build a chat request whose user message is the frozen workload's service catalog sized by `dflash2_long_context.records_for` to a target prompt-token count at a tokens-per-record rate (`--tokens-per-record`, default the Qwen-calibrated `TOKENS_PER_RECORD`), with one passkey sentence `The pass key is <passkey>. Remember it. <passkey> is the pass key.` inserted between catalog lines at `round(depth × records)`, followed by the question `What is the pass key? The pass key is`; the request SHALL carry the workload's sampling fields with `top_k=1`, `max_tokens` from `--max-tokens` (default 32), `chat_template_kwargs` from `--template-kwargs` (default `{"enable_thinking": false}`) merged over `add_generation_prompt: true`, a seed, and no token-id or timestamp echo, and the passkey SHALL be a five-digit string drawn per cell from a seeded generator that does not occur in the catalog text.

#### Scenario: Needle at depth

- **WHEN** `prompt(records=4800, depth=0.5, passkey="48213", model="ckpt")` is built with the defaults
- **THEN** the catalog lines before and after the inserted sentence are exactly `_catalog(4800)` split at line 2400, the last message asks for the pass key, `top_k` is 1, `max_tokens` is 32, `chat_template_kwargs` is `{"enable_thinking": false, "add_generation_prompt": true}`, and `return_token_ids` is false

#### Scenario: Reasoning template arm

- **WHEN** the probe runs with `--template-kwargs '{"reasoning_strength": "low"}' --max-tokens 256`
- **THEN** every cell's request carries `chat_template_kwargs` `{"reasoning_strength": "low", "add_generation_prompt": true}` and `max_tokens` 256, and the pass criterion is unchanged: the reply's content contains the passkey

#### Scenario: Tokenizer-calibrated sizing

- **WHEN** the probe runs with `--tokens-per-record 56.54 --lengths 1000000`
- **THEN** the cell's catalog has `round(1000000 / 56.54)` records

#### Scenario: Passkey absent from filler

- **WHEN** a passkey is drawn for any cell
- **THEN** the five-digit string does not occur in `_catalog(records)` for that cell's record count

### Requirement: Serving order and packages

The probe SHALL serve one package per run through `dflash2_served.container_command` with history drafts off and the requested `--max-context-tokens`, or with `--base-url` measure an instance served elsewhere (the console's) that it neither starts nor stops, refusing an unhealthy URL before any cell; it SHALL send cells longest length first, depths in ascending order within a length, so no cell's prompt is a prefix of an earlier one; a run against a native-context package SHALL take only lengths below that package's native context, and a run against a package whose declared context exceeds its checkpoint's native context (the rope-scaled Qwen yarn4 packages, the 1048576-token Muse Glimmer package) SHALL include at least one length below the native context as the baseline row.

#### Scenario: Longest first

- **WHEN** the lengths are 200000, 320000, 520000, 1000000 and the depths 0.1, 0.5, 0.9
- **THEN** the first request is the 1,000,000-token cell at depth 0.1 and the last is the 200,000-token cell at depth 0.9, and the serving container is started once

#### Scenario: Native baseline

- **WHEN** the native MTP package is run with `--lengths 200000 --max-context-tokens 262144`
- **THEN** three cells run and the summary's grid has the single length key `200000`

#### Scenario: Muse Glimmer arm

- **WHEN** the console serves the 1048576-token Muse Glimmer package and the probe runs with `--base-url http://127.0.0.1:8100 --lengths 100000,200000,320000,520000,1000000 --tokens-per-record 56.54 --template-kwargs '{"reasoning_strength": "low"}' --max-tokens 256`
- **THEN** no container is started or stopped, fifteen cells run longest first, each records `prompt_tokens`, `passed`, `request_seconds` and `accepted_per_round`, the 100000 row is the below-native baseline, and `summary.json` lands under `qualify/results/glimmer-passkey/`
