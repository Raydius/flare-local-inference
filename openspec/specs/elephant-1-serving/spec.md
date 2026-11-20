# elephant-1-serving Specification

## Purpose

Serve the text-only Elephant-1 decision model as a prepared core inside a 12.5GB device budget, returning calibrated option probabilities through standard generation logprobs while stating the measured deviation from the vendor runtime's trained attention mask and quantization precision.

## Requirements

### Requirement: Decision probabilities through generation logprobs

A prepared `elephant-1-9b` core SHALL serve text-only Completions requests and SHALL return, for a client-rendered prompt ending at the answer line, per-token logprobs at the generated position from which the option probabilities are recovered. The core SHALL accept `max_tokens: 1` with `logprobs` and a `logprob_token_ids` list naming the question's slot-letter tokens, and the response MUST carry the full-vocabulary logprob of every named letter for a question with at most 128 options, whatever the letters' rank. The twenty most likely tokens alone are not a sufficient readout: the model is trained on the letters' relative logits only, and non-letter tokens outrank low-probability letters. The core SHALL NOT require a decision-specific HTTP route, request field, or response field beyond the ordinary Completions response.

#### Scenario: One question's options are recoverable from one request

- **WHEN** a client sends a rendered prompt whose answer line admits options A, B and C with `max_tokens: 1`, `logprobs: 20` and `logprob_token_ids` naming the tokens for A, B and C
- **THEN** the response's logprobs for the generated position include the tokens for A, B and C, and normalizing their probabilities yields the option distribution

#### Scenario: A twenty-option question stays recoverable

- **WHEN** a client sends a rendered prompt with exactly 20 options and names the 20 letter tokens in `logprob_token_ids`
- **THEN** the response contains every option letter's logprob, although the twenty most likely tokens hold only some of the letters

### Requirement: Declared 12.5GB device envelope

The prepared core SHALL declare its context and concurrency, and its measured device memory at those declarations MUST be at most 12.5GB on the target device, including weights, KV pages, recurrent state, captured graphs, and workspace. The declared capacities MUST be the measured ones, never an extrapolation.

#### Scenario: Inspect reports a footprint inside the budget

- **WHEN** the core is exported for the declared context of 16,384 tokens and the declared concurrency, `flare inspect` is run on it, and it is served on GPU 1
- **THEN** the inspected fixed device allocations and the serving process's measured device memory are each at most 12.5GB at the recipe's declared context and concurrency

#### Scenario: An oversized declaration is refused

- **WHEN** a declaration is submitted whose measured footprint exceeds 12.5GB on the target device
- **THEN** the change records the failed budget and the measured value instead of shipping the declaration

### Requirement: Trained-limit and modality refusals

Requests longer than the trained 16,384-token limit SHALL be refused rather than truncated. The vendor runtime also refuses instead of truncating, at its engine's own limit of 32,768 tokens per question; the core applies the trained length the model card states. Image and video inputs SHALL be refused, because the package carries no vision path.

#### Scenario: Over-long prompt is refused

- **WHEN** a request's prompt exceeds 16,384 tokens
- **THEN** the request fails with a refusal naming the limit, and no truncated answer is produced

#### Scenario: Image input is refused

- **WHEN** a request carries an image part
- **THEN** the request fails before execution and no probability is returned

### Requirement: Trained attention mask in the full-attention layers

The core SHALL declare the token sequence that ends the rendered state block as its block-end marker. A prompt holding the marker exactly once has two blocks, the shared prefix through the marker and the question suffix after it; a prompt without it is served causally; a prompt holding it more than once MUST be refused. For a prompt with two blocks the core SHALL run its eight full-attention layers with the trained two-block mask: a token attends to every token of its own block and of every earlier block, where the blocks are the shared prefix and the question suffix. The Gated DeltaNet layers SHALL stay causal. State computed for the prefix block MUST NOT depend on any suffix, and MUST NOT be reused by a request whose prefix block differs in tokens or in extent. Prompt logprobs MUST be null for tokens inside a block. The resulting probabilities MUST be qualified against the vendor reference runtime on a fixed probe set. The qualification SHALL report, per option, argmax agreement, mean and maximum absolute probability delta, and rank agreement for score questions, and MUST separate the mask deviation from the quantization deviation by comparing the BF16 checkpoint under the served mask against the vendor runtime before comparing the packed core against that BF16 forward. Acceptance requires argmax agreement of at least 99% and mean absolute probability delta of at most 0.02 on the probe set; a result below either bound MUST be reported as failed acceptance.

#### Scenario: Mask deviation measured alone

- **WHEN** the probe runs the BF16 checkpoint under the served mask and the vendor runtime over the same fixed states and questions
- **THEN** the report contains the mask-stage agreement and probability deltas with no quantization involved

#### Scenario: A below-threshold result fails acceptance

- **WHEN** the mask-stage report shows argmax agreement below 99% or mean absolute probability delta above 0.02
- **THEN** the qualification is recorded as failed and the core is not described as equivalent to the reference

#### Scenario: One state answers several questions from one prefix

- **WHEN** two requests share a state and differ in their question
- **THEN** each answer equals the answer the same request gives with an empty prefix cache

#### Scenario: An ambiguous block boundary is refused

- **WHEN** a prompt holds the block-end marker more than once
- **THEN** the request is refused and no answer is returned

### Requirement: Declared precision

The prepared core SHALL declare NVFP4 W4A16 storage for the body projections and BF16 storage for the embedding, normalization weights, GDN convolution and scalars, and the precision SHALL be visible in the prepared declaration. The packed core's deviation from the BF16 forward under the served mask SHALL be measured on the same probe set, and a mean absolute probability delta above 0.01 MUST be reported as failed acceptance.

#### Scenario: Precision declaration matches the prepared core

- **WHEN** the core's declared storage is inspected
- **THEN** body projections report NVFP4 W4A16 and the embedding, norms, convolutions and GDN scalars report BF16

#### Scenario: Quantization delta measured

- **WHEN** the probe runs the packed core and the BF16 forward under the served mask over the fixed probe set
- **THEN** the report contains the precision-stage deltas, including the contribution of the 24 recurrent GDN layers
