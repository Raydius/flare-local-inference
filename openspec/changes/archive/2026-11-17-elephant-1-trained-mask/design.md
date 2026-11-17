# Design — elephant-1-trained-mask

## Approach

Two stages with a gate between them. The first changes no serving code and ends in one operator decision; the second is built only on that decision and its tasks are fixed by it.

**The mask.** In the eight full-attention layers a token attends to every token of its own block and of every earlier block. A rendered decision prompt has two blocks: the shared prefix (system turn and state) and the question suffix. The vendor runtime builds it with Transformers' `create_causal_mask(block_sequence_ids=…)` (`transformers/masking_utils.py`, the mechanism PaliGemma and pi0 use), which the checkpoint's `serve.patch` feeds from `qev/readout.py`. The Gated DeltaNet layers are recurrent and stay causal. The answer is read at the last prompt token, so no decode step runs under the mask for a one-token read.

**Stage 1 — feasibility.**

1. *Which block matters.* Extend the reference side of `qualify/elephant1_decision_parity.py:256` to score the probe set under four masks in one GPU run of the vendor environment: trained, causal, causal prefix with bidirectional suffix, bidirectional prefix with causal suffix. Score the trained mask twice to give the bounds a noise floor. If a causal prefix with a bidirectional suffix meets 0.99 argmax agreement and 0.02 mean delta, the prefix keeps chunked causal prefill and today's prefix reuse, and only a block of a few hundred tokens needs noncausal attention.
2. *Which operator executes a block.* The paged prefill kernels are causal (`ignition/components/paged_attention.py:102`, `ignition/components/paged_tma.py:81`); the noncausal paged operator takes 16 queries per request; the dense variable-length operator is noncausal at any length over contiguous keys (`ignition/components/varlen_attention.py:28`). Plan and run that operator at this geometry (16 query heads, 4 KV heads, head dimension 256, BF16) for suffix queries over prefix keys gathered from pages plus the block's own keys, compare with a PyTorch reference, and record its scratch bytes.
3. *Whether a whole block fits.* Each block must pass each layer whole: layers 4 and later consume layer 3's output, and the Gated DeltaNet state advances in place per chunk (`ignition/recipes/qwen38_dense/decoder.py:362`). Measure a prefill graph's scratch at the block sizes stage 1.1 leaves in play, up to 16,384 tokens, against the headroom under 12GB.
4. *How the runtime keeps a block whole and where its boundary comes from.* Helios already keeps bidirectional spans inside one prefill submission (`helios/src/scheduler.rs:1859`), bounded by the prefill chunk size; today only a vision core's image layout produces them (see the feasibility finding). Establish whether an Elephant-1 block can use that mechanism with the boundary derived from tokens alone (the end of the rendered state block), which keeps prefix keys valid (`helios/src/cache/hash.rs:62`), or whether it needs a request value, a prefix-key change, or a block end as a checkpoint endpoint (`helios/src/scheduler/checkpoints.rs:90`).
5. *What approval the build needs.* Classify the result: recipe-local composition of existing operators and existing span handling, or a new captured-graph operation contract or shared prefill-policy change, which `AGENTS.md` reserves for an owner decision.

The stage writes its measurements under `qualify/results/` and a finding into this change, then asks the operator for one of: build the suffix block only, build both blocks, or stop.

**Stage 2 — build, both blocks (operator's choice, 2026-11-17).** The path is the one `## Feasibility finding` describes. A text core's contract declares the state-end marker; for a prompt holding it exactly once Helios produces a prefix span and a suffix span from the prompt's tokens and keeps each inside one prefill submission, and it refuses a prompt holding it more than once; the prefix span's end is a checkpoint boundary, which makes the prefix reusable across questions, and no cache tier resumes a request strictly inside one of its spans. These are shared-runtime and prepared-metadata changes, approved for this fork by its maintainer on 2026-11-17. Prompt logprobs stay null inside a text span, as inside an image span: each token of a bidirectional block has seen the token it would be scored against. The operator dropped the task that would have returned them on 2026-11-17; the decision readout uses generation logprobs at the answer position. In the recipe, the full-attention layers read the per-token span bindings and attend with the dense variable-length operator, widened with key lengths; the prefill chunk becomes 16,384 tokens; the recipe identity changes so records written by a causal core cannot be restored into it (`ignition/recipes/elephant1/bootstrap.py:111`). The core is re-exported through the console and its footprint re-measured on GPU 1; the parity driver's mask comparison is rerun against a BF16 forward using the served mask; the documentation states the measured result.

### Explicit Rejections

- Building before the mask ablation — rejected: the build for a suffix-only mask and for a whole-prefix mask differ by a shared-runtime change and a 16,384-token graph.
- Writing a block's keys chunk by chunk and attending afterwards — rejected: later layers and the recurrent state would already have advanced on outputs computed without the rest of the block.
- A decision-specific HTTP route or a per-model runtime flag for the mask — rejected: the repository forbids both; the boundary is prepared metadata or a typed input shared code consumes.
- Folding the NVFP4 precision pass into this change — rejected by scope; its comparison is against the BF16 forward this change redefines, so it follows.

## External Review

### Peer Review Verdict

Quorum: not achieved (weight 1/3, escalated to tier 2)
Certainty: 0.3
Caller Agreement: 0.75
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.9
- local-ai-qllm (qLLM) — ok, agreement 0.25
- google-gemini-rpa (models/gemini-flash-lite-latest) — ok, agreement 0.5
- deepseek-flash (deepseek-flash) — ok, agreement 0.25
- anthropic-claude-sonnet-5-5 (claude-sonnet-5-5) — error, tier budget elapsed while the call was in flight

Override: Ray Dollete | override | 2026-11-16

The operator asked for the review to be run again on 2026-11-16 after a first call with the same outcome (weight 1/3, certainty 0.3, caller agreement 0.7, the same peer past its deadline); the figures above are the second call's.

### Review Feedback

No consensus feedback: the quorum was not achieved, the response was withheld, and nothing from it is incorporated.

### Stage 2 review, 2026-11-17

Brief: the both-blocks build (span declaration by marker, span-end checkpoint, prompt logprobs in text spans, 16,384-token prefill with dense block attention) and five questions on the marker rule, prefix-cache soundness, the checkpoint at the prefix end, the single large submission, and model policy in shared runtime.

Quorum: not achieved (weight 2/3 at tier 2)
Certainty: 0.55
Caller Agreement: 0.95
Sources:
- claude-sonnet-5-5 — ok, agreement 0
- models/gemini-flash-lite-latest — ok, agreement 0.85
- qLLM — ok, agreement 0.8
- claude-opus-5-5 — error, tier budget elapsed while the call was in flight

The response was withheld, so no peer feedback is incorporated. The points below are the author's own answers to the brief, folded into the stage 2 tasks:

- *Marker rule.* The boundary is the index after the last occurrence of the declared marker sequence. A question that itself contains the marker moves the boundary away from the vendor's; the rule and that case are stated in the spec's qualification and tested with a two-marker prompt.
- *Cache soundness.* The span layout depends on the last marker in the whole prompt, so a checkpoint taken at a prompt end or terminal position can cover tokens a longer prompt would place in one larger prefix block. A request with spans therefore reuses only checkpoints that end at or before the start of its first span or exactly at a span end whose covered layout is the same, and a deeper hit inside a span falls back instead of failing the request (`helios/src/scheduler.rs:1873`).
- *Checkpoint at the prefix end.* Valid: prefix tokens never attend to the suffix and the recurrent layers are causal. The prefix submission ends exactly at the boundary, so prefix and suffix are separate submissions on a miss.
- *Gathered keys.* Earlier keys read from pages into a contiguous tensor are declared scratch, with no serving-time allocation.

Second call, same brief, run at the operator's request:

Quorum: not achieved (weight 1/3 at tier 2)
Certainty: 0.28
Caller Agreement: 0.95
Sources:
- claude-sonnet-5-5 — ok, agreement 0
- models/gemini-flash-lite-latest — error, network error
- qLLM — ok, agreement 0.85
- claude-opus-5-5 — ok, agreement 0

The response was withheld again. The two heaviest peers rated the author's answer at 0, so the Opus peer was then asked the same brief alone. Its verdict was sound-with-changes; this is one peer's feedback, not a consensus. Points beyond the author's four above, not yet folded into the tasks:

- *Cache key.* Hash the span boundaries clipped to a checkpoint's depth into the prefix key, and mark whether the depth closes a span, instead of restricting which checkpoints a spanned request may reuse.
- *Marker rule.* A marker in the question silently moves the boundary. Derive it from template structure (the first marker after a declared message start) or refuse a prompt whose declared region holds other than one marker; add adversarial prompts to the qualification.
- *Mixed masks in one submission.* Causal tokens can precede or follow a span in one submission; the dense operator has no causal rows, so the graph needs separate causal and span segments.
- *Prompt logprobs at 16,384 tokens.* Prompt scoring is sized by the prefill chunk (`ignition/components/prompt_logprobs.py:276`); its storage at a 16,384-token chunk is not in the feasibility arithmetic and is unmeasured.
- *Chunk size.* Raising the prefill chunk to 16,384 for every prompt blocks the other request and gives up 2,048-token chunks for causal prompts; a separate maximum span length would keep the chunk at 2,048. That is a further shared-runtime change.
- *Resume at an unaligned boundary.* The prefix end is rarely a multiple of 128 tokens; a resumed suffix needs the partial-page path and correct span distances.
- *Composition.* Text spans and image spans on one prompt are undefined; refuse overlap.

Override: Ray Dollete | override | 2026-11-17

Disposition of the single-peer points, decided with the override and carried by the stage 2 tasks:

- *Cache key* — taken as a lookup rule, not a key change: a request never resumes from a depth strictly inside one of its spans and falls back to a shallower checkpoint. With at most one marker per prompt, the layout over the tokens a checkpoint covers is fixed by those tokens at every depth that is not strictly inside a span, so existing keys stay byte-identical. The author's own reading; the peer proposed hashing the layout.
- *Marker rule* — taken: a prompt with exactly one occurrence of the declared marker has the two spans; none is causal; more than one is refused as an ambiguous block boundary. This replaces "last occurrence".
- *Mixed masks* — taken: with the rule above a spanned prompt has no causal prompt tokens, a submission holds one or both spans, and the graph attends each span over the request's keys up to that span's end; a submission with no span takes the causal path.
- *Prompt logprobs at 16,384 tokens* — taken: prompt-scoring storage joins the memory arithmetic and is measured by the export.
- *Chunk size* — declined: this model's prompts are spanned, a separate maximum span length is one more shared-runtime change, and no measurement yet shows the need. Prefill latency at 16,384 tokens is measured after export and the question reopens if it fails the operator.
- *Unaligned resume* — taken as tests.
- *Composition* — taken: a contract that declares both a text block marker and an image span is refused.

## Bomb-squad findings

- `helios/src/cache/hash.rs:62` — a prefix key hashes the recipe identity, the tokens and image conditioning; it does not carry a block boundary, so a boundary supplied as a request value would let two requests with equal leading tokens share state computed under different masks. Stage 1.4 decides the boundary with this in view.
- `helios/src/scheduler.rs:1866` — a cache hit that ends inside a declared span fails the request instead of falling back to an earlier checkpoint.
- `ignition/recipes/elephant1/bootstrap.py:111` — the recipe identity does not encode the mask, so a mask core would share its record namespace with the causal core. Stage 2 changes the identity.
- `ignition/recipes/qwen38_dense/decoder.py:362` — each prefill chunk runs all 32 layers, and the prepared Gated DeltaNet built here advances its state in place, so a block must pass each layer whole. Stage 1.3 measures whether it fits.
- `ignition/recipes/elephant1/bootstrap.py:146` — a request with more than one output token decodes causally after a bidirectional suffix; the one-token read is unaffected. Stage 2 states or refuses that case once the vendor behavior is known.
- `helios/src/scheduler/checkpoints.rs:90` — checkpoint endpoints are message boundaries and the prompt end; a Completions prompt publishes no checkpoint at the prefix block's end, so the prefix is not reused across questions today either.
- `helios/src/model.rs:419` — a span longer than the prefill chunk size is refused; a 16,384-token block needs a 16,384-token chunk and a longer capacity ladder (`ignition/components/capacity.py:47`).
- `helios/src/model.rs:288` — span membership is read as an image position, which suppresses prompt logprobs inside a span.
- Checked and safe: `ignition/components/paged_attention.py:136` raises instead of substituting causal attention; `ignition/recipes/qwen38_dense/decoder.py:283` writes a submission's keys before attending; `helios/src/scheduler/policy.rs:329` never splits a declared span. Not examined: `helios/src/offload/transfer.rs`, the engram code and the TLA+ models.

## Feasibility finding

Stage 1 result, 2026-11-17. No file under `ignition/`, `helios/`, `flare/` or `flare-cli/` changed.

**Which block matters: both.** Vendor runtime on the RTX PRO 6000, 159 answerable questions, each run compared with the first trained run (`qualify/results/elephant1-mask-feasibility/masks.json`):

| Mask | Argmax agreement | Mean abs. delta | Max abs. delta | Verdict (0.99 / 0.02) |
|---|---|---|---|---|
| trained, second run | 1.0000 | 0.0000 | 0.0000 | noise floor: none |
| causal | 0.9057 | 0.0320 | 0.428 | failed acceptance |
| suffix block only | 0.9434 | 0.0214 | 0.290 | failed acceptance |
| prefix block only | 0.9308 | 0.0234 | 0.478 | failed acceptance |

Each half recovers about a third of the causal deviation and neither reaches either bound. Only the full two-block mask can meet the `elephant-1-serving` requirement.

**Which operator executes a block: the dense variable-length one.** `b12x.attention.varlen` ran noncausally at 16 query heads, 4 key-value heads, head dimension 256, BF16, on the RTX 5060 Ti beside the serving core (`qualify/results/elephant1-mask-feasibility/operator.json`): 256 suffix queries over 4,352 keys, maximum absolute difference 5.3e-4 from FP32 PyTorch attention (reference magnitude 0.145), 16,384 bytes of scratch; one 16,384-token block over itself, 2.7e-4 (magnitude 0.073), 1,048,576 bytes of scratch. Inputs were seeded random values, not model activations. `PreparedVarlenAttention` passes one length tensor for queries and keys (`ignition/components/varlen_attention.py:27`); the suffix case needs it widened with an optional key-length input. The operator reads contiguous keys, so a block's keys are attended from the chunk's own key tensor (prefix block) or gathered from pages (prefix keys for the suffix).

**Whether a whole block fits: by arithmetic yes, unmeasured in a graph.** Declared decoder scratch, batch inputs and prefill pages for one row grow by 256, 769 and 1,795 MiB over today's 2,048-token storage at 4,096, 8,192 and 16,384 tokens; the headroom under 12GB beside the measured 9,608 MiB process is 2,680 MiB. At 16,384 tokens that leaves 885 MiB for what the arithmetic leaves out: each operator's private scratch and the preparation headroom that grows with chunk tokens (`ignition/planning.py:24`). Only an export measures it.

**Where the boundary comes from: the prompt's tokens.** On 160 of 160 rendered probes the token after the last `</state>` and blank line (token ids 510, 2374, 29, 271) equals the prefix length the vendor engine uses (`_prefix_and_suffix_ids`). No request value is needed, so two requests with equal leading tokens always have equal blocks over them and prefix keys (`helios/src/cache/hash.rs:74`: identity, tokens, image conditioning) stay valid once the recipe identity changes with the mask (`ignition/recipes/elephant1/bootstrap.py:111`).

**How the runtime keeps a block whole: the span mechanism fits, but nothing produces a text span.** Corrections to the approach and hunt above:

- The scheduler takes arbitrary token ranges, keeps each inside one submission, lets causal tokens precede and follow a span, and accepts adjacent spans and a span from the first to the last prompt token (`helios/src/scheduler.rs:1858`, `helios/src/scheduler/policy.rs:329`, `helios/src/scheduler/policy_tests.rs:402`).
- The only producer of spans is the DeepSeek image layout (`helios/src/frontend/images/layouts/deepseek.rs:149`). The contract declares a span as an image marker pair with two distinct tokens, legal only on a vision core (`helios/src/model/contract.rs:334`, `:385`); the text prompt path yields none (`helios/src/model/prompt.rs:103`). A block that starts at the first token or ends at the last cannot be declared.
- A captured graph already receives span geometry per token, the distance to the span's start and to its last token, through the `IMAGE_LEFT_CONTEXT` and `IMAGE_RIGHT_CONTEXT` bindings (`helios/src/model/packing.rs:617`, `ignition/preparation/contract.py:39`); `deepseek4` consumes them (`ignition/recipes/deepseek4/decoder.py:408`). There is no span start or length scalar.
- A span is limited to the prefill chunk size (`helios/src/model.rs:419`), which Helios allows up to the context; 2,048 is the recipe's default (`ignition/recipes/elephant1/recipe.py:48`), and the capacity ladder's top rung is the chunk size.
- Span membership suppresses prompt logprobs (`helios/src/model.rs:288`, consumed at `helios/src/model/packing.rs:551` and `helios/src/server/runtime.rs:1677`).
- A cache hit that ends strictly inside a span fails the request with no fallback (`helios/src/scheduler.rs:1873`, `helios/src/server/runtime.rs:1409`).
- No checkpoint is planned at a span boundary; spans only disqualify endpoints inside them (`helios/src/scheduler/checkpoints.rs:99`). A token-id Completions prompt checkpoints only at its end. The Messages protocol's `cache_control` hint is the one way to cut inside a user message (`helios/src/server/protocol/decode.rs:231`).
- Deriving the blocks inside the graph from the uploaded token ids, with no span, is unsound: a checkpoint cut, a chunk end or a cache hit can split a block (`helios/src/scheduler/policy.rs:331`).

**Path: suffix block only.** Fails acceptance by measurement (0.9434, 0.0214), so it cannot satisfy the delta spec; it would be a partial improvement over causal. It keeps the 2,048-token chunk and causal prefix prefill. Files: a text span declaration and producer in the prepared contract and Helios prompt path (`helios/proto/runtime_core.proto`, `helios/src/model/contract.rs`, `helios/src/model/prompt.rs`, `ignition/preparation/contract.py`); `ignition/components/varlen_attention.py` (key lengths); `ignition/recipes/elephant1/` (contract, a full-attention layer that gathers prefix keys from pages, recipe identity). Prefix keys need no change; checkpoint endpoints need none; suffix prompt logprobs become null. Classification: the graph side is recipe-local composition of an existing operator family; the span declaration is a prepared-metadata and shared-runtime change.

**Path: both blocks.** The only path that can meet the bound. Everything in the suffix path, plus: the prefill chunk and top graph capacity rise to 16,384 tokens (`ignition/recipes/elephant1/recipe.py:48`, `ignition/recipes/elephant1/bootstrap.py:136`), with the footprint above to be measured by export; a prompt's whole prefix prefills in one submission, during which the other request waits. To reuse a prefix across questions, which the delta spec's third scenario requires, a checkpoint endpoint at the prefix block's end is needed (`helios/src/scheduler/checkpoints.rs:90`), and a hit there lands on a span boundary, which the span check accepts. Prompt logprobs become null for the whole prompt unless the span reading in `helios/src/model.rs:288` is separated from image positions. Classification: the span declaration, the span-end checkpoint endpoint and the prompt-logprob change are shared-runtime and prepared-metadata changes; the attention composition and chunk size are recipe-local.

**Approval.** `AGENTS.md:354` reserves the upstream project owner's approval for a new captured-graph operation family or operation contract. None of the three shared changes is one: they must instead stay free of model-identity branches and per-model flags and act on prepared data (`AGENTS.md:336`, `AGENTS.md:367`), which a span declaration any text core can carry satisfies. The recipe's use of dense variable-length attention in place of causal paged attention is read here as a shape variant of an existing family. Ray Dollete approved the three changes for this fork on 2026-11-17; they need the upstream owner's review only if proposed upstream.

**Not settled by this stage.** The captured-graph footprint at 16,384 tokens; prefill latency of a 16,384-token submission on the RTX 5060 Ti; whether the vendor runtime decodes causally after the suffix for answers longer than one token.
