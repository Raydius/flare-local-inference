# Design — add-qwen38-dense-dspark

## Approach

See proposal.md for motivation. The dense Qwen recipe gains a second proposal method, composed inside the recipe from mechanisms other recipes already execute. No kernel, Protobuf field or Helios code is added; the shared DFlash2 components, the Flash Next vision encoder and the `qwen` image profile gain optional arguments and an accepted `model_type` whose defaults keep their earlier behavior.

### Method name and limits

- `ignition/recipes/declaration.py`: `Speculation.method` and `Recipe.speculative_methods` accept `dspark`. `SpeculativeMethod` gains `max_draft_tokens: int | None = None`; `None` keeps the recipe's `max_draft_tokens`, so every existing declaration is unchanged.
- `ignition/cli/__init__.py`, `ignition/cli/preparation.py`, `ignition/cli/help.py`: `--speculative-method` offers `dspark`; the draft-token limit is read after the method resolves, from the method when it declares one.
- `ignition/cli/display.py`, `ignition/cli/serve.py`, `ignition/cli/config.py`: the plan report, the serve banner and the configuration template name DSpark when the plan's method is `dspark`.
- Why a method name and not a second recipe: the draft reads the same target layers, the same Gated DeltaNet rollback records and the same vocabulary head as DFlash2, and Kimi K3.1 Pro and Flash Next already carry two methods through `speculative_methods`.

### Recipe declaration

- `ignition/recipes/qwen38_dense/recipe.py` declares `speculative_methods={"dspark": SpeculativeMethod(...)}` with measured rank costs, `draft_bytes`, `draft_files=("config.json", "model.safetensors")` and `max_draft_tokens=7`, and the checkpoint `daisyrose-27b` with its pinned revision and measured byte counts.
- `prepare` selects the draft repository by method: the DFlash2 pin in `draft_config.py`, or the DSpark pin in the new `dspark_config.py`; a `--draft-model` folder replaces the DSpark download as it does for Flash Next (`ignition/recipes/qwen38/recipe.py:62`).
- DSpark plans refuse `--draft-head-quantization`, `--draft-body-quantization` and speculation studies with a usage error naming the method, as `ignition/recipes/kimi31/recipe.py:24` refuses draft-body conversion for its DSpark. `--online-mxfp8` is accepted and converts the target only; the DSpark draft keeps its checkpoint precision.

### Draft configuration and weights (new modules)

- `dspark_config.py`: the repository and revision pin; strict validation of the published `config.json` (architecture `DSparkDraftModel`, five `full_attention` layers, hidden 5120, 32 query and 8 key/value heads of 128, intermediate 17408, target layers 5, 19, 33, 47, 61, mask token 248070, block size 7, vanilla Markov rank 256, confidence head with Markov input, YaRN factor 32 over 8192 positions, `sliding_window` null) against the target's hidden size, vocabulary and layer count. It derives the rotary gain `0.1 * ln(factor) + 1` and the inverse frequencies through `ignition.components.rotary.interpolated_frequencies`.
- `dspark_weights.py`: per-layer fused query/key/value and output projections, norms and MLP; the fused context projection (`fc.weight`), `hidden_norm`, the stacked per-layer context key/value projection, the final norm, and the Markov embedding and head. The confidence tensors are checked for shape and not loaded. The weights dataclass is registered in `ignition/components/weights.py` beside the other draft weight declarations so the rebuild manifest names its tensors.
- The draft has no embedding or vocabulary head; it uses the target's, as the DFlash2 owner does (`ignition/recipes/qwen38_dense/dflash.py:24`).

### Draft forward and round (new module `dspark.py`)

- Context: the five captured target features are concatenated, projected, normalized, projected to every layer's keys and values, key-normalized, rotated with the YaRN frequencies and gain at the token's position, and written to that layer's draft pages at the accepted token's page slot. This is the shared DFlash2 context computation (`PreparedContext` in `ignition/components/dflash2.py`) with page slots in place of ring slots.
- Block: seven positions, the pending token followed by six mask tokens, embedded with the target embedding. Position `i` proposes draft `i + 1`, so the owner declares `block_size = 7` and `logit_offset = 0`, the convention of `ignition/recipes/kimi3/dspark.py:375`. The model card states seven proposals from a block of seven.
- Attention: each layer runs `PreparedPagedAttention` in `verify` mode, noncausal, unwindowed, width 7, over that layer's draft pages through the request's page table, with the committed partial page and the block's keys and values staged into private tail pages in the page dtype. The recipe composes this directly; Inkling 2's `DraftAttention` requires a numeric window and BF16 pages and is not imported.
- Markov: `ignition.recipes.kimi3.dspark.PreparedMarkov` is imported and bound through the shared round's Markov hook; it returns the FP32 bias the round expects and runs without a collective at one rank.
- Round: an owner beside `dflash.SpeculativePreparation` supplies the forwards to `PreparedParallelDraftRound`. Prefix selection stays on the host, so adaptive and fixed-width verification both prepare. The confidence head is not executed.

### Preparation, storage and capture

- `bootstrap.py`: `PreparationConfig` carries the method name. Weights, state layout, page layout, feature capture and the speculative owner branch on it. With `dflash2` every value passed downstream equals today's. Verification width above 8 is rejected under `dspark`.
- `pool.py`: under `dspark`, `page_specs` adds `draft{N}.key` and `draft{N}.value` regions for the five draft layers beside the target's, and the state row omits the DFlash2 rings. `contract.py` already declares every pool page region as paged key/value history on the shared page table, so Helios covers draft history in prefix reuse, snapshots, eviction and offload without a model-specific path.
- Page dtype: FP8 E4M3, the recipe's convention for target history and DFlash2 rings, when `paged_decode.supports` accepts the noncausal tail geometry at FP8; otherwise BF16, which Inkling 2 serves. Page cost is 5,505,024 bytes with FP8 and 6,815,744 bytes with BF16, against 4,194,304 today. The declared costs and the documentation record the prepared dtype.
- `decoder.py`: under `dspark`, feature capture keeps every position of a prefill chunk; the compact trailing-window capture stays DFlash2-only.
- Draft-disabled graphs write no draft pages, as they write no DFlash2 rings today; Helios serves one mode per process.

### Target rope scaling (operator addition during apply)

- `recipe.py` declares `daisyrose-27b-yarn4` with `context=1_048_576` and `RopeScaling(4., 262144)` and passes the plan checkpoint's scaling to `PreparationConfig`, which raises its context ceiling to the scaling's and requires the 262,144 anchor. Declaring one scaled checkpoint also admits `--rope-parameters` for the recipe's other checkpoints (`ignition/cli/preparation.py` `rope_scaled`), which is how a console profile of a checkpoint folder selects it.
- `ModelWeights.load` checks the anchor against the checkpoint's `max_position_embeddings` and records the scaling on `ModelConfig`; `decoder.py` builds its 32 inverse frequencies with `ignition.components.rotary.interpolated_frequencies` when a scaling is present. The preparation identity gains a `rope_scaling` entry only when present.
- Draft rotary is unchanged for both methods. The DSpark draft attends the whole request, so past 262,144 tokens its relative offsets exceed those its YaRN configuration covers; draft acceptance there is unmeasured and listed in `docs/TODO.md`.
- Task 7 exports and serves the scaled core, on the operator's instruction.

### Image input (operator addition during apply)

The checkpoint stores a 27-layer vision encoder (333 BF16 tensors under `model.visual.`, 921,460,192 bytes, output width 5120) that the dense recipe did not load. The Flash Next recipe serves images from the same encoder geometry with quantized projections and output width 2560, and Helios implements the Qwen image preparation and layout (`QWEN_IMAGE_PREPARATION`, `QWEN_IMAGE_LAYOUT`): it uploads patches, publishes features into a per-request arena slot, and uploads three position rows per prompt token. Reuse follows the repository's order — import, pass arguments, widen with a default:

- Encoder: `ignition.recipes.qwen38.vision.PreparedVision`, `VisionScratch`, `VisionMetadata` and `VisionWeights`, widened with an output width (default 2560), the rotary embedding class (default the Flash Next class), and unquantized projections when a loaded projection weight is a BF16 tensor. The Flash Next call sites pass none of them. Four `vision<patches>` graphs for 256, 1024, 4096 and 16384 patches, as that recipe prepares.
- Feature storage and injection: the shared `ignition.components.vision_state.VisionState` and `PrefillVisionInputs` with `hidden=5120`, and `ignition.components.vision_delivery.PreparedFeaturePublication`, as the Inkling2 recipe wires them into `PreparedBatch(validate_prefill=...)` and its decoder. A dense-recipe module imports the Flash Next upload fields (`VisionInputs`, `FEATURES`, `PATCHES` from `ignition/recipes/qwen38/vision_state.py`) and holds the generation that validates an upload, runs the encoder and publishes; every image recipe holds its own generation, because it binds the recipe's encoder metadata.
- Positions: Qwen image tokens carry three rotary positions (time, row, column), and text after an image continues from a position smaller than its token index. The dense decoder takes the three uploaded position rows in prefill, selects each rotary pair's axis by the checkpoint's interleaved `mrope_section` [11, 11, 10], and stores the offset between the next rotary position and the next token index in a `rope_delta` state-row region that decode and verification add to the token position. A text-only prompt has equal axes and a zero offset, which gives the positions the decoder computes today. The path exists only when vision is prepared: a language-only plan, the DFlash2 and DSpark layouts without vision, and the Elephant-1 recipe, which imports the dense decoder, keep their graphs and state rows.
- Contract and frontend: `contract.prepare` declares the vision inputs, image tokens (`image_token_id` 248056), the Qwen preparation and layout identifiers and three position dimensions when vision is prepared; `ignition/preparation/frontend.py` accepts the checkpoint's `model_type` `qwen3_5` for the `qwen` image profile. No Helios change: the identifiers, bindings and upload paths are the ones the Flash Next recipe uses.
- Declaration: `max_images=4`, `image_bytes`, `language_model_only=True`; the vision bytes join the device cost and the feature arena (image slots x 16384 x 5120 BF16) joins the fixed cost.

Draft positions stay token indices. Mixed prefill is not prepared; the Flash Next recipe refuses it with DFlash2 as well.

### Validation

- CPU: configuration validation, tensor mapping against the published header, method resolution through flag and configuration file, per-method limits, refusals, declarations.
- Reference parity: draft logits before and after the Markov bias against the published `DSparkDraftModel` on fixed inputs, which also fixes the rotary gain and the block convention.
- GPU: draft pages hold the same keys and values for one prefill and for chunked prefill, after resume, and after a round that rejects drafts; DFlash2 tests unchanged.
- Served: accepted tokens per round at seven drafts and fixed width eight on the code workload, at concurrency 1 and 8, for `daisyrose-27b`; a core exported before this change loads and serves.

## External Review

### Peer Review Verdict

Quorum: not achieved (weight 2/3, escalated to tier 2)
Certainty: 0.5
Caller Agreement: 0.5
Arbiter: ok
Sources:
- anthropic-claude-haiku-5-5 (claude-haiku-5-5) — ok, agreement 0.8
- local-ai-qllm (qLLM) — ok, agreement 0.4
- google-gemini-rpa (models/gemini-flash-lite-latest) — ok, agreement 0.7
- deepseek-deepseek-flash (deepseek-flash) — ok, agreement 0.5
- anthropic-claude-sonnet-5-5 (claude-sonnet-5-5) — error, agreement n/a (tier budget elapsed, still in flight)
Override: Ray Dollete | Override and continue | 2026-11-18

The operator directed one re-run on 2026-11-18 after a first call that also ended without quorum (weight 1/3, certainty 0.3, caller agreement 0.75, the same peer timed out); the block above is the re-run.

### Review Feedback

Consensus feedback recorded verbatim:
- none — both responses were withheld under the unmet quorum; nothing from them is incorporated.

## Bomb-squad findings

- `ignition/recipes/inkling2/dspark_state.py:41` and `ignition/recipes/inkling2/pool.py:68` — the only tail-page caller keeps BF16 draft pages and merges page rows with block keys without conversion — fixed: 5 (the Qwen draft stages its tail in the page dtype with an explicit conversion, and the page dtype follows the kernel's supported geometry)
- `ignition/recipes/inkling2/dspark.py:128` — Inkling 2's `DraftAttention` computes `window + block_size` for every layer, and its context needs feature-mix weights and a recurrent row — fixed: 5 (not imported; the recipe composes `PreparedPagedAttention` directly)
- `ignition/recipes/kimi31/dspark.py:33` — the Kimi K3.1 YaRN helpers read latent-attention keys (`qk_rope_head_dim`, `mscale`, `attn_factor`) absent from a GQA draft — fixed: 3 (frequencies from `components/rotary.interpolated_frequencies`, gain derived in `dspark_config.py`)
- `ignition/components/parallel_draft.py:268` — the round slices logits by the owner's `logit_offset`; block 7 with offset 1 yields six logits for seven drafts — fixed: 5 and 6 (offset 0 with block 7, confirmed by reference parity)
- `ignition/recipes/qwen38_dense/decoder.py:411` and `ignition/components/dflash2.py:54` — prefill capture keeps a trailing window and would drop earlier positions from full-context draft history — fixed: 4
- `ignition/planning.py:231` — speculation studies are admitted by recipe name and build DFlash2 owners — fixed: 4 (refused under `dspark`)
- `ignition/planning.py:474` and `ignition/recipes/qwen38_dense/weights.py:183` — draft conversions are validated for the recipe, not the method — fixed: 2 and 4 (refused under `dspark`)
- `ignition/cli/preparation.py:220`, `ignition/cli/help.py:22`, `ignition/recipes/qwen38_dense/bootstrap.py:89`, `ignition/recipes/qwen38_dense/pool.py:28` — the draft-token and width limits are recipe-wide at every layer — fixed: 1 and 4
- `ignition/cli/display.py:45` and `ignition/cli/serve.py:320` — a `dspark` plan would be labelled "MTP module" — fixed: 1
- `ignition/cli/display.py:45`, `ignition/cli/help.py:46`, `ignition/cli/preparation.py:255` — Kimi K3.1 Pro's unnamed default DSpark is labelled "MTP module", its `dflash2` alternative is not listed in help, and `--speculative-method dspark` is rejected for it — declined: naming Kimi K3.1 Pro's default method changes that recipe's declaration and recorded rebuild metadata, which this change does not touch
- `ignition/cli/preparation.py:131` and `ignition/recipes/qwen38_dense/recipe.py:25` — a method that declares draft files passes the `--draft-model` check while the dense recipe ignores the folder — fixed: 2
- `ignition/recipes/minimax_m3/draft.py:443` — MiniMax M3's Markov composition returns a BF16 bias and depends on its selector — fixed: 5 (the Kimi K3 composition is imported instead)
- `helios/src/core/conversion.rs:2717` — core combination does not compare the recorded method — declined: the dense recipe prepares one tensor-parallel size, so two dense cores are never combined
- `ignition/preparation/rebuild_manifest.py:279` — a weights dataclass missing from the allowlist in `ignition/components/weights.py` is walked whole — fixed: 3
- `ignition/components/paged_attention.py:127` — whether the paged-decode kernel accepts FP8, noncausal, unwindowed tail pages at this geometry was not determinable without b12x installed — fixed: 5 (checked at preparation; BF16 pages otherwise)

## Reframe

- Wrong assumption: task 7 treated the serving configuration as the change's to choose: "the concurrency the operator serves (at most 8)", a profile of its own name and template, and an image built from the branch wherever it stood. The corrections disproved each part: the operator serves 16 sequences, as `Qwen3.6`, with the Quinn chat template, from images on `deploy`.
- Corrected model: the served package is the operator's default console profile with only the change's subject replaced. That profile (`2026-11-10-flare-dflash2-mxfp8-a16-1794383054083-copy-261115_213526`, read through `GET /api/profiles`) carries `max_num_seqs` 16, `served_model_name` "Qwen3.6", `chat_template_path` `~/models/raydius/quinn-template/chat_template.jinja`, and the run fields `--generation-defaults`, `--prefix-cache-memory 48G`, `--scheduler-policy throughput` and `--kv-cache-limit memory`. The export differs from it in the model folder, `--recipe qwen38_dense`, the DSpark draft folder and method, the image, the native 262,144-token context, and the options that do not apply to the dense recipe or the DSpark draft (rope parameters, online MXFP8, draft head conversion, A16 decode, embedding-table storage). The branch is rebased onto `deploy` before `build.sh` (`ignition/recipes/qwen38_dense/recipe.py` and `bootstrap.py` are unchanged by the rebase; range-diff shows both commits `=`).
- Approach delta: the profile is derived from the default profile by a recorded field diff instead of being composed; run fields are copied; the image is built at the rebased commit e092796f; the operator's stated wishes (16 sequences, template, served name, no 4x package yet) are the done-when.
- Tasks reframed: task 7 survives with rewritten criteria and a `reframed:` line.
- Lesson: openspec/lessons.md#serve-from-the-operators-running-profile — the operator's default console profile specifies a served package; copy it and change only the change's subject.


## Reuse decision after verification (operator, 2026-11-19)

The verification audit (`verification.md`, Reused findings 1 to 3) found three recipe-local re-implementations of shared DFlash2 code: the weight loader and shape table in `dspark_weights.py` (shared: `ignition/components/dflash2_weights.py`, which has a `projector` switch), `rotate` in `dspark.py` (shared: `_rotary` in `ignition/components/dflash2.py`), and `PreparedContext` in `dspark.py` (shared: `PreparedContext` in the same component). The operator chose "Fix by reusing shared code": widen the shared components with optional arguments that default to today's behaviour, test on GPU 1, then re-export and re-approve the DaisyRose core.
