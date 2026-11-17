## REMOVED Requirements

### Requirement: Causal-attention qualification is measured, not assumed

**Reason**: the core runs the trained mask, so causal attention in every layer is no longer required; the qualification moves into the requirement below.
**Migration**: none; the probe and its acceptance bounds are unchanged.

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Declared precision

The prepared core SHALL declare NVFP4 W4A16 storage for the body projections and BF16 storage for the embedding, normalization weights, GDN convolution and scalars, and the precision SHALL be visible in the prepared declaration. The packed core's deviation from the BF16 forward under the served mask SHALL be measured on the same probe set, and a mean absolute probability delta above 0.01 MUST be reported as failed acceptance.

#### Scenario: Precision declaration matches the prepared core

- **WHEN** the core's declared storage is inspected
- **THEN** body projections report NVFP4 W4A16 and the embedding, norms, convolutions and GDN scalars report BF16

#### Scenario: Quantization delta measured

- **WHEN** the probe runs the packed core and the BF16 forward under the served mask over the fixed probe set
- **THEN** the report contains the precision-stage deltas, including the contribution of the 24 recurrent GDN layers
