## Purpose

Serve the text-only Elephant-1 decision model as a prepared core inside a 12GB device budget, returning calibrated option probabilities through standard generation logprobs while stating the measured deviation from the vendor runtime's trained attention mask and quantization precision.

## ADDED Requirements

### Requirement: Decision probabilities through generation logprobs

A prepared `elephant-1-9b` core SHALL serve text-only Completions requests and SHALL return, for a client-rendered prompt ending at the answer line, per-token logprobs at the generated position from which the option probabilities are recovered. The core SHALL accept `max_tokens: 1` with `logprobs` up to 20, and the returned top-20 list MUST cover the slot-letter tokens of every question with at most 20 options. The core SHALL NOT require a decision-specific HTTP route, request field, or response field beyond the ordinary Completions response.

#### Scenario: One question's options are recoverable from one request

- **WHEN** a client sends a rendered prompt whose answer line admits options A, B and C with `max_tokens: 1` and `logprobs: 20`
- **THEN** the response's top-20 logprobs for the generated position include the tokens for A, B and C, and normalizing their probabilities yields the option distribution

#### Scenario: A twenty-option question stays recoverable

- **WHEN** a client sends a rendered prompt with exactly 20 options
- **THEN** the top-20 logprobs still contain every option letter, because no other token outranks a letter the model was trained to emit

### Requirement: Declared 12GB device envelope

The prepared core SHALL declare its context and concurrency, and its measured device memory at those declarations MUST be at most 12GB on the target 12GB-class device, including weights, KV pages, recurrent state, captured graphs, and workspace. The declared capacities MUST be the measured ones, never an extrapolation.

#### Scenario: Inspect reports a footprint inside the budget

- **WHEN** the core is exported for the declared context of 16,384 tokens and the declared concurrency, and `flare inspect` is run on GPU 1
- **THEN** the reported device memory is at most 12GB and matches the recipe's declaration

#### Scenario: An oversized declaration is refused

- **WHEN** a declaration is submitted whose measured footprint exceeds 12GB on the target device
- **THEN** the change records the failed budget and the measured value instead of shipping the declaration

### Requirement: Trained-limit and modality refusals

Requests longer than the trained 16,384-token limit SHALL be refused rather than truncated, matching the vendor runtime's rule. Image and video inputs SHALL be refused, because the package carries no vision path.

#### Scenario: Over-long prompt is refused

- **WHEN** a request's prompt exceeds 16,384 tokens
- **THEN** the request fails with a refusal naming the limit, and no truncated answer is produced

#### Scenario: Image input is refused

- **WHEN** a request carries an image or an image token in its prompt
- **THEN** the request fails before execution and no probability is returned

### Requirement: Causal-attention qualification is measured, not assumed

The core SHALL export ordinary causal attention in every layer, and the resulting probabilities MUST be qualified against the vendor reference runtime on a fixed probe set. The qualification SHALL report, per option, argmax agreement, mean and maximum absolute probability delta, and rank agreement for score questions, and MUST separate the mask deviation from the quantization deviation by comparing the recipe's BF16 canonical forward against the vendor runtime before comparing the packed core against that forward. Acceptance requires argmax agreement of at least 99% and mean absolute probability delta of at most 0.02 on the probe set; a result below either bound MUST be reported as failed acceptance.

#### Scenario: Mask deviation measured alone

- **WHEN** the probe runs the recipe's BF16 canonical forward and the vendor runtime over the same fixed states and questions
- **THEN** the report contains the mask-stage agreement and probability deltas with no quantization involved

#### Scenario: A below-threshold result fails acceptance

- **WHEN** the mask-stage report shows argmax agreement below 99% or mean absolute probability delta above 0.02
- **THEN** the qualification is recorded as failed and the core is not described as equivalent to the reference

### Requirement: Declared precision

The prepared core SHALL declare NVFP4 W4A16 storage for the body projections and BF16 storage for the embedding, normalization weights, GDN convolution and scalars, and the precision SHALL be visible in the prepared declaration. The packed core's deviation from the BF16 canonical forward SHALL be measured on the same probe set, and a mean absolute probability delta above 0.01 MUST be reported as failed acceptance.

#### Scenario: Precision declaration matches the prepared core

- **WHEN** the core's declared storage is inspected
- **THEN** body projections report NVFP4 W4A16 and the embedding, norms, convolutions and GDN scalars report BF16

#### Scenario: Quantization delta measured

- **WHEN** the probe runs the packed core and the BF16 canonical forward over the fixed probe set
- **THEN** the report contains the precision-stage deltas, including the contribution of the 24 recurrent GDN layers
