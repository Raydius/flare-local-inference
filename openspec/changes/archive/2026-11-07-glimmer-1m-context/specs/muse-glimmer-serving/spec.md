## MODIFIED Requirements

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare context 1048576 (the checkpoint's native 131072 is the trained anchor; no rope scaling is declared because the 39 sliding layers never see a rope distance past 2048 and the 13 full layers carry no positional encoding), 128-token pages, concurrency 4, up to four images of 32 MiB, `language_model_only`, sampling defaults temperature 1.0 / top-p 1.0 / top-k disabled, `max_draft_tokens` 15, and rank-1 costs whose `state_row_bytes` and `page_bytes` equal the prepared continuation layout and page size with and without drafting; preparation SHALL accept any 128-aligned context from 128 through 1048576 and the exported package SHALL carry `max_sequence_tokens` equal to the prepared context.

#### Scenario: Declaration test
- **WHEN** the declaration test loads the pinned configuration
- **THEN** `continuation_layout(config).row_bytes` and `page_bytes(config)` equal `recipe.costs(1, …)` for the drafting and plain layouts, `RECIPE.context` is 1048576, and the alias resolves to the `muse_glimmer_nvfp4` recipe

#### Scenario: Context ceiling
- **WHEN** `PreparationConfig(context=1048576 + 128)` is constructed
- **THEN** it is rejected with the error naming the `[128,1048576]` range, while `PreparationConfig(context=1048576)` is accepted

#### Scenario: Served window below the ceiling
- **WHEN** the operator runs the 1048576-token package with `--max-context-tokens 262144`
- **THEN** the served per-request context is 262144 and a request past it is refused with the context error, without re-export
