# qwen38-dense-dspark Specification

## Purpose

Prepare and serve the dense Qwen recipe's checkpoints with the DSpark draft `RadixArk/Qwen3.8-27B-DSpark` as a selectable proposal method, with draft history covering the whole request, and state the conditions under which the method is qualified.

## Requirements

### Requirement: DSpark is a selectable proposal method of the dense Qwen recipe

The dense Qwen recipe SHALL prepare DSpark proposals when the plan's speculative method is `dspark`, selected by `--speculative-method dspark` or by `speculative_method = "dspark"` for the model in the configuration file. The recipe's default method SHALL remain `dflash2`. Every checkpoint of the recipe SHALL accept the method. A recipe that declares no `dspark` method SHALL refuse the name with a usage error.

#### Scenario: The flag selects DSpark

- **WHEN** `ignition export qwen3.8-27b --speculative-method dspark` is planned
- **THEN** the plan's method is `dspark`, its report names DSpark as the proposal source, and the rebuild metadata records `dspark`

#### Scenario: The configuration file selects DSpark for one model

- **WHEN** the configuration file sets `speculative_method = "dspark"` under `[models."daisyrose-27b"]` and no flag is given
- **THEN** a plan for `daisyrose-27b` uses `dspark` and a plan for `qwen3.8-27b` uses `dflash2`

#### Scenario: Omitting the method keeps DFlash2

- **WHEN** `ignition export qwen3.8-27b` is planned with no method flag and no configured method
- **THEN** the plan's method is `dflash2` and its capacities, costs and catalog equal those planned before this capability existed

#### Scenario: A recipe without the method refuses it

- **WHEN** `--speculative-method dspark` is given for a recipe whose declaration lists no `dspark` method
- **THEN** planning stops with a usage error stating that the model does not prepare this proposal method

### Requirement: The draft is pinned and validated

DSpark preparation SHALL load the draft from `RadixArk/Qwen3.8-27B-DSpark` at revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`, or from a `--draft-model` folder holding `config.json` and `model.safetensors`. Preparation MUST refuse a draft whose configuration differs from the supported geometry: five noncausal full-attention layers at hidden size 5120 with 32 query and 8 key/value heads of 128, target layers 5, 19, 33, 47 and 61, mask token 248070, block size 7, a vanilla Markov head of rank 256, YaRN rotary embedding of factor 32 over 8192 positions, and no sliding window. Preparation MUST refuse a draft whose hidden size, vocabulary or target layer count differs from the target checkpoint's.

#### Scenario: The pinned draft loads

- **WHEN** DSpark preparation runs with no `--draft-model` folder
- **THEN** the draft files come from the pinned revision and every tensor of the draft maps to a declared weight or to the validated, unexecuted confidence head

#### Scenario: A draft folder replaces the download

- **WHEN** `--draft-model FOLDER` names a folder with the two draft files
- **THEN** preparation reads the draft from the folder and downloads nothing for the draft

#### Scenario: A foreign geometry is refused

- **WHEN** the draft configuration declares a sliding window, a different Markov rank, or different target layers
- **THEN** preparation stops with an error naming the unsupported field

### Requirement: DSpark prepares at most seven draft tokens

A DSpark plan SHALL accept zero through seven draft tokens and SHALL refuse more with a usage error, while a DFlash2 plan of the same recipe keeps its limit of fifteen. The recipe's preparation MUST reject a DSpark configuration whose verification width exceeds eight, whatever built it.

#### Scenario: Eight drafts are refused under DSpark

- **WHEN** `--speculative-method dspark --mtp-draft-tokens 8` is planned
- **THEN** planning stops with a usage error stating the limit of seven

#### Scenario: Fifteen drafts remain available under DFlash2

- **WHEN** `--mtp-draft-tokens 15` is planned with the default method
- **THEN** the plan is accepted

### Requirement: Draft history covers every accepted position

Under DSpark the draft SHALL attend every accepted position of the request. Draft keys and values SHALL be stored in page regions addressed by the request's target page table and declared to the runtime as paged key/value history, so prefix reuse, snapshots, eviction and offload preserve draft history together with target history. Draft history MUST depend only on the accepted tokens, not on how they were delivered: one prefill, chunked prefill, resume from a reused prefix, and rounds that reject drafts SHALL leave the same draft keys and values for the same accepted sequence.

#### Scenario: Chunked prefill equals one prefill

- **WHEN** the same prompt longer than one prefill chunk is prepared once in a single prefill and once in chunks
- **THEN** every draft layer's stored keys and values for the prompt's positions are equal in both runs

#### Scenario: A rejected round leaves no proposal in history

- **WHEN** a round accepts fewer drafts than it proposed
- **THEN** the draft pages hold keys and values only for the accepted positions, and the next round's proposals equal those of a request that reached the same accepted sequence without the rejection

#### Scenario: A reused prefix carries its draft history

- **WHEN** a request continues from a cached prefix
- **THEN** its first speculative round attends the prefix's draft history without recomputing it

### Requirement: Draft logits follow the published model

The prepared draft's proposal logits SHALL follow the published `DSparkDraftModel`: context from the five target layers through the draft's projection and normalization, YaRN rotary embedding with the attention factor `0.1 * ln(32) + 1` applied to the rotation, a block of seven positions beginning with the pending token in which each position proposes the next token, and the Markov bias of the previously sampled token added to the draft logits. The confidence head SHALL NOT be executed; prefix selection stays on the host, and both adaptive and fixed-width verification SHALL prepare.

#### Scenario: Logits match the reference on fixed inputs

- **WHEN** the prepared draft and an eager PyTorch reference written from the published `dflash.py` and `dspark.py`, loading the pinned draft weights, receive the same target features, pending token and previously sampled tokens
- **THEN** the draft logits before and after the Markov bias agree within the bound the test derives from BF16 arithmetic and the stored history precision, for every block position

#### Scenario: Fixed-width verification prepares

- **WHEN** `--speculative-method dspark --mtp-draft-tokens 7 --no-mtp-adaptive-verification` is exported
- **THEN** the core's default policy is fixed width eight and its speculative graphs carry no confidence output

### Requirement: Options that do not apply to the draft are refused

A DSpark plan SHALL refuse `--draft-head-quantization`, `--draft-body-quantization` and speculation-study benchmark preparation with a usage error naming DSpark. `--online-mxfp8` SHALL leave the DSpark draft's weights in their checkpoint precision.

#### Scenario: Draft head conversion is refused

- **WHEN** `--speculative-method dspark --draft-head-quantization mxfp8` is planned
- **THEN** planning stops with a usage error stating that the DSpark draft uses the target's vocabulary head

### Requirement: DaisyRose is a checkpoint of the dense Qwen recipe

The recipe SHALL declare the alias `daisyrose-27b` for `RadixArk/DaisyRose-27B-NVFP4` at revision `c4d85c5dfad123426555c22f4e24bb15381f0bdb` with the stored and device byte counts of `nvidia/Qwen3.8-27B-NVFP4`, whose tensor names and total size it shares, and SHALL prepare it with either proposal method or with speculation disabled.

#### Scenario: The alias resolves

- **WHEN** `ignition export daisyrose-27b` is planned
- **THEN** the plan names the pinned repository and revision and the dense Qwen recipe

### Requirement: A dense Qwen checkpoint may extend its context by declared rope scaling

The recipe SHALL declare `daisyrose-27b-yarn4`, the `daisyrose-27b` checkpoint with a 1,048,576-token context through frequency-only rope interpolation by four over the trained 262,144 tokens. Preparation SHALL accept a context above 262,144 tokens only with a declared scaling anchored at 262,144, and SHALL interpolate only the target's rotary frequencies; draft rotary frequencies SHALL be unchanged. A preparation without scaling SHALL keep its frequencies and its preparation identity.

#### Scenario: The scaled alias resolves

- **WHEN** `ignition export daisyrose-27b-yarn4` is planned
- **THEN** the plan's declared context is 1,048,576 tokens and its rope scaling has factor 4 over 262,144

#### Scenario: A larger context without scaling is refused

- **WHEN** dense Qwen preparation is configured with 1,048,576 tokens of context and no rope scaling
- **THEN** it raises naming the 262,144-token ceiling

### Requirement: The dense Qwen recipe accepts images

The dense Qwen recipe SHALL prepare the checkpoint's vision encoder and accept up to four images per request through the Qwen image preparation and layout, with either proposal method or with speculation disabled. Image tokens SHALL take the three rotary positions the canonical model assigns, and text after an image SHALL continue from the canonical position. `--language-model-only` SHALL prepare the recipe without the encoder and reject images.

#### Scenario: Prepared image features match the canonical encoder

- **WHEN** the prepared encoder and `transformers`' vision model run the same patches with the checkpoint's vision weights
- **THEN** the merged features agree within a bound derived from BF16 arithmetic

#### Scenario: Positions follow the canonical model

- **WHEN** a prompt holds text, an image and more text
- **THEN** the decoder's rotary angles equal those of `transformers`' three-axis positions for every token, and decode continues from the canonical next position

#### Scenario: A text prompt is unchanged

- **WHEN** a prompt holds no image
- **THEN** every token's three positions equal its token index and the position offset is zero

#### Scenario: Language-only preparation

- **WHEN** `--language-model-only` is planned for a dense Qwen checkpoint
- **THEN** no vision graph or image storage is prepared, the state row has no position-offset region, and the core rejects images

#### Scenario: A core prepared with image input answers an image request

- **WHEN** a dense Qwen core exported without `--language-model-only` receives a chat request with an image
- **THEN** it reports image input in `/v1/models` and answers from the image's content

This scenario is unrun: the DaisyRose core served on GPU 0 was exported language-only, and `docs/TODO.md` lists the export with image input and the image request.

### Requirement: Existing cores and DFlash2 preparation are unchanged

A core exported before this capability SHALL load and serve without re-export. DFlash2 preparation of the recipe SHALL declare the same state-row layout, page regions and page cost as before; a plan with image input adds only the position-offset region to the state row.

#### Scenario: A DFlash2 layout is unchanged

- **WHEN** the continuation layout and page regions are computed for the default method
- **THEN** their names, shapes, dtypes and byte counts equal those computed before this capability existed

### Requirement: Qualification is stated from measurement

The documentation SHALL state DSpark's support as present qualification: the hardware, checkpoint, draft count, verification policy, sequence count and context at which accepted tokens per round were measured, the draft history precision that was prepared, and the page cost. Checks and measurements that were not run, including throughput and acceptance at fixed concurrency, SHALL be listed in `docs/TODO.md`.

#### Scenario: The measured configuration is documented

- **WHEN** a DaisyRose core is exported with DSpark at seven draft tokens and served at fixed width eight
- **THEN** `docs/model-preparation.md` records the accepted tokens per round measured on the served core, with its hardware, sequence count and context, beside the model card's SGLang figures, labelled as a different runtime and hardware, and `docs/TODO.md` lists the trials at fixed concurrency as unrun
