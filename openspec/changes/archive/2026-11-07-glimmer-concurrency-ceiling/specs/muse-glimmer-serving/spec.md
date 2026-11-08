## MODIFIED Requirements

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare context 1048576 (the checkpoint's native 131072 is the trained anchor; no rope scaling is declared because the 39 sliding layers never see a rope distance past 2048 and the 13 full layers carry no positional encoding), 128-token pages, concurrency 8 by default and 16 as the `--max-concurrency` ceiling, up to four images of 32 MiB, `language_model_only`, sampling defaults temperature 1.0 / top-p 1.0 / top-k disabled, `max_draft_tokens` 15, a packed speculative catalog of `token_capacity` 128 (full fifteen-draft depth through eight active requests) and `graph_budget` 64, and rank-1 costs whose `state_row_bytes` and `page_bytes` equal the prepared continuation layout and page size with and without drafting and whose `graph_bytes` is the VRAM inventory of the concurrency-16 export; preparation SHALL accept any 128-aligned context from 128 through 1048576 and the exported package SHALL carry `max_sequence_tokens` equal to the prepared context.

#### Scenario: Declaration test
- **WHEN** the declaration test loads the pinned configuration
- **THEN** `continuation_layout(config).row_bytes` and `page_bytes(config)` equal `recipe.costs(1, …)` for the drafting and plain layouts, `RECIPE.context` is 1048576, `RECIPE.concurrency` is 8, `RECIPE.max_concurrency` is 16, and the alias resolves to the `muse_glimmer_nvfp4` recipe

#### Scenario: Context ceiling
- **WHEN** `PreparationConfig(context=1048576 + 128)` is constructed
- **THEN** it is rejected with the error naming the `[128,1048576]` range, while `PreparationConfig(context=1048576)` is accepted

#### Scenario: Served window below the ceiling
- **WHEN** the operator runs the 1048576-token package with `--max-context-tokens 262144`
- **THEN** the served per-request context is 262144 and a request past it is refused with the context error, without re-export

#### Scenario: Default eight-request catalog
- **WHEN** `PreparationConfig(dflash=True, max_active=8, speculative_token_capacity=128)` is constructed
- **THEN** its round shapes hold width-one rounds at batches 1, 2, 4 and 8 and no batch above 8, the packed width-16 rounds `(8, 16, 9)` and `(8, 16, 128)`, no packed row count above 128, at most 64 packed rounds, and 128 sampled rows

#### Scenario: Sixteen-request ceiling catalog
- **WHEN** `PreparationConfig(dflash=True, max_active=16, speculative_token_capacity=128)` is constructed
- **THEN** its round shapes hold width-one rounds at batches 1, 2, 4, 8 and 16, the packed width-16 rounds `(16, 16, 17)`, `(16, 16, 128)` and `(8, 16, 128)`, no packed row count above 128, at most 64 packed rounds, and 256 sampled rows

#### Scenario: Admission at the default and the ceiling
- **WHEN** the operator exports with no `--max-concurrency` flag, or with `--max-concurrency 16`, and serves the package
- **THEN** the plan reports concurrency 8 or 16 respectively, a request beyond that count waits in the queue rather than being refused, `--max-concurrency 17` is refused with the error naming 1 through 16, and the export log's `Allocate cache` line reports the pages that bound each request to `pages ÷ concurrency × 128` guaranteed tokens while a lone request may still reach 1048576
