# Serve Elephant-1 with its trained attention mask, after a feasibility check

Tier: 3
Type: feature

## Stated Requirements

- "make the trained mask the next change, starting with a feasibility check before any build."

Assumed: the change serves Elephant-1's eight full-attention layers with the trained block-bidirectional mask; the 24 Gated DeltaNet layers stay causal.
Assumed: the feasibility check is the change's first work and gates every build task; an infeasible result ends the change with the finding recorded and nothing built.
Assumed: acceptance is the mask bound `elephant-1-serving` already states, 0.99 argmax agreement and 0.02 mean absolute probability delta against the vendor runtime, measured with `qualify/elephant1_decision_parity.py`.
Assumed: the 12GB envelope on GPU 1, the 16,384-token context and concurrency 2 still hold.
Assumed: a precision pass over the NVFP4 packing plan is out of scope; the export this change needs also measures the packer storage change that `docs/TODO.md` records.
Assumed: the block boundary reaches the server without a decision-specific HTTP route; how it is carried is a feasibility question.

## Why

The `elephant1` recipe runs every layer causally, and the measured cost is the largest defect of the served core: the BF16 checkpoint under causal attention agrees with the vendor runtime on 0.906 of argmax answers with a 0.032 mean probability delta, against bounds of 0.99 and 0.02 (`qualify/results/elephant1-decision-parity/stage-a-mask.json`). The checkpoint was trained with a two-block mask in its eight full-attention layers: every token attends to its whole block and to every earlier block, where the blocks are the shared prefix (system turn and state) and the question suffix. Until the core applies that mask its probabilities are not the model's, whatever the storage precision.

The mask does not fit the prepared path as it stands. A bidirectional block needs every token of the block at a layer before any of them can attend, while Flare prefills a prompt in chunks of at most 2,048 tokens through the whole layer stack. The paged attention kernels that prefill uses are causal; the noncausal ones cover at most 16 queries per request or work on dense, unpaged keys. The server also has no notion of where the prefix ends. Each of these may turn a recipe-local change into a new captured-graph operation contract or a shared-runtime change, which the repository allows only with an owner decision. The size of the build is therefore unknown, and this change settles it before building.

## What Changes

- A feasibility stage, with no change to serving code: measure in the vendor environment which parts of the mask the agreement depends on, establish which existing b12x attention operators can execute each block, measure whole-block prefill memory at 16,384 tokens against the 12GB envelope, and decide how the block boundary is carried. Its result is a written finding and one operator decision: the build path, or no build.
- If the operator chooses a build path: the `elephant1` recipe's full-attention layers execute the trained mask for a prompt with a declared block boundary, the prefix block's state stays reusable across questions about one state, and the core is re-exported and re-measured on GPU 1.
- The parity driver's mask comparison is rerun against a BF16 forward that uses the served mask, and `docs/model-preparation.md` and `docs/TODO.md` state the measured result.

Non-goals: the NVFP4 precision pass; vision; speculative decoding; a decision-specific HTTP route; any change that needs a new captured-graph operation family without the owner approval `AGENTS.md` requires.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

`elephant-1-serving`: the attention-mask requirement changes from causal attention in every layer, with its deviation measured, to the trained block mask in the full-attention layers; the decision-readout requirement gains the block boundary.

## Impact

- Feasibility stage: `qualify/elephant1_decision_parity.py` (mask variants on the reference side), a kernel and memory probe under `qualify/`, results under `qualify/results/`. No `ignition/`, `helios/` or `flare/` change.
- Build stage, bounds to be fixed by the feasibility finding: `ignition/recipes/elephant1/` and the attention composition it reuses from `ignition/recipes/qwen38_dense/decoder.py`; possibly `ignition/components/` for a block attention component; possibly prepared-contract inputs and Helios prefill scheduling, which are shared runtime and need an explicit decision.
- A new export on GPU 0 through the console, and a new footprint approval on GPU 1.

## Blast Radius

1. `ignition/recipes/qwen38_dense/decoder.py:229` — `FullAttention` builds `PreparedPagedAttention` with the default causal mask; the `elephant1` recipe reuses this decoder for its eight full-attention layers.
2. `ignition/recipes/qwen38_dense/decoder.py:227` — the paged KV writer stores a chunk's keys and values before attention; a bidirectional block needs the whole block's keys present first.
3. `ignition/components/paged_attention.py:102` — the TMA prefill path is taken only for causal attention.
4. `ignition/components/paged_attention.py:136` — noncausal paged attention raises unless the geometry is the paged-decode one.
5. `ignition/components/paged_tma.py:81` — the TMA prefill kernel is planned with `is_causal=True`.
6. grepped `MAX_Q_PER_REQ` in the installed `b12x/attention/paged_decode/_tuning.py` — 16 queries per request, the limit of the one noncausal paged operator.
7. `ignition/components/varlen_attention.py:28` — the dense variable-length attention component plans `causal=False`; it serves vision towers today and takes unpaged keys.
8. `ignition/recipes/elephant1/bootstrap.py:136` — prefill graphs are captured at `prefill_capacities(2048)`; a prompt longer than a chunk is prefilled in several steps.
9. `ignition/components/capacity.py:47` — the prefill capacity ladder, which `helios/src/scheduler.rs:84` must match; whole-block prefill changes what the ladder has to offer.
10. `ignition/recipes/elephant1/pool.py:27` — paged BF16 key and value regions for the eight layers; a dense block attention reads them as a gathered view or needs block storage.
11. `ignition/recipes/elephant1/contract.py:11` — the contract is the dense Qwen text contract; it declares no input that carries a block boundary.
12. `helios/model-contract.md:160` — length bindings such as `QUERY_LENGTH` are the existing way a request-dependent length reaches a captured graph.
13. `qualify/elephant1_decision_parity.py:256` — the reference side switches the vendor mask between its trained and causal forms; mask variants extend this switch.
14. `openspec/specs/elephant-1-serving/spec.md:53` — the main spec requires causal attention in every layer.
15. `docs/model-preparation.md:1607` — the Elephant-1 section states that the core runs every layer causally and is not equivalent to the vendor runtime.
16. grepped `block_sequence_ids` in the installed `transformers` — `masking_utils.py` builds the block mask the vendor runtime injects; PaliGemma and pi0 use the same argument.
17. grepped `causal=False` across `ignition/` — draft and vision paths only (`ignition/components/dflash2.py:275`, `ignition/components/varlen_attention.py:28`, recipe draft modules); no target-model prefill path is noncausal.
