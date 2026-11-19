# Prepare and serve Qwen3.8-27B with a DSpark draft

Tier: 3
Type: feature

## Stated Requirements

- "The goal is DSpark support"
- "What would be required to create a recipe for a checkpoint based on Qwen3.8-27B that also ships with dspark drafts?"
- "draft: https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark"
- "This seems like a clear case for a method on qwen38_dense unless you think otherwise"
- "I was actually mistaken, this is the correct checkpoint model card: https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4"
- "We can make context scaling a followup, but I wanted to just validate the feasibility"
- "I want to go ahead and start prepping the 4x YaRN profile also." (2026-11-18, during apply)
- "do not package the 4x yet" / "first rebase onto deploy" (2026-11-19, during apply)
- "also use Chat Template `raydius/quinn-template/chat_template.jinja`" / "served model name should be Qwen3.6" (2026-11-19, the served profile)

Assumed: DSpark is a second learned proposal method of the dense Qwen recipe, selected with `--speculative-method dspark` or `[models."ALIAS"] speculative_method = "dspark"`; DFlash2 stays the recipe default.
Assumed: the draft is pinned to `RadixArk/Qwen3.8-27B-DSpark` revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`, and the target `RadixArk/DaisyRose-27B-NVFP4` revision `c4d85c5dfad123426555c22f4e24bb15381f0bdb` joins the recipe's checkpoints under the alias `daisyrose-27b`.
Assumed: DSpark prepares with every checkpoint of the dense Qwen recipe, because the draft reads target layers and a vocabulary that all of them share.
Assumed: the draft attends every accepted position of the request, as its configuration declares (`sliding_window: null`), through paged draft history stored as FP8 E4M3 like the recipe's target history, or as BF16 if the paged-attention kernel does not prepare that geometry at FP8.
Assumed: the draft's rotary embedding applies the attention factor that its YaRN configuration implies, `0.1 * ln(32) + 1`.
Assumed: the draft's confidence head is validated and not executed, as Kimi K3.1 Pro treats the same head; prefix selection stays on the host so fixed-width verification remains available.
Assumed: DSpark prepares at most seven draft tokens, the draft's `block_size`.
Assumed: the 4x YaRN profile is frequency-only rope interpolation of the target by four over its trained 262,144 tokens, declared as the checkpoint `daisyrose-27b-yarn4` (1,048,576 tokens), as `qwen3.8-flash-yarn4` declares it; the draft's rotary frequencies stay unscaled, the choice the archived change `2026-11-04-qwen38-dflash-rope-scaling` made, and acceptance past 262,144 tokens is a measurement, not a guarantee.
- "The base profile was just for feasibility, I have zero intention of every running it." (2026-11-19, during apply)

Assumed: the core this change exports and serves is the 4x YaRN profile, started only on the operator's instruction; the 262,144-token profile is not exported.

## Why

The dense Qwen recipe (`ignition/recipes/qwen38_dense/`) prepares one learned drafter, DFlash2, from `incoai/Qwen3.8-27B-DFlash2`. RadixArk publishes a DSpark draft for the same target architecture and recommends it for its `DaisyRose-27B` fine-tune; its model card reports mean accepted lengths of 3.3 to 4.1 tokens on code workloads at seven draft tokens and verification width eight under SGLang. Flare cannot prepare it: the speculative method names stop at `mtp` and `dflash2`, the recipe downloads only the DFlash2 draft, and its draft storage is a 2048-position ring where this draft attends the whole request.

The draft is the DFlash2 draft body with different attachments. Both have five noncausal GQA layers at hidden size 5120 with 32 query and 8 key/value heads, both read target layers 5, 19, 33, 47 and 61, and both use mask token 248070. The DSpark draft drops DFlash2's convolution and candidate selector and adds a rank-256 Markov head, a confidence head, YaRN rotary embedding, and unwindowed context. Flare already executes each of those pieces in another recipe, so the work is composition inside the dense Qwen recipe plus a method name.

`RadixArk/DaisyRose-27B-NVFP4` has the tensor names, quantization layout, chat template and preprocessor configuration of `nvidia/Qwen3.8-27B-NVFP4`, the checkpoint the recipe already pins. Supporting it is a checkpoint declaration.

## What Changes

- The speculative method name `dspark` is accepted by the recipe declaration, the command line, and the configuration file. A method may declare its own draft-token maximum.
- The dense Qwen recipe declares `dspark` as a second method with its own draft checkpoint, device costs and page geometry, and declares the `daisyrose-27b` checkpoint.
- New draft modules in `ignition/recipes/qwen38_dense/`: draft configuration and pin, draft weights, and the draft forward (context projection into paged history, noncausal block attention with YaRN rotary, Markov proposals).
- Dense Qwen preparation carries the selected method through its configuration, state layout, page layout, weights, decoder feature capture and graph catalog.
- Draft history under DSpark is paged storage addressed by the target's page table, so prefix reuse, snapshots, eviction and offload cover it through the shared paged-history mechanisms.
- Documentation states the method, its commands, its storage, and its qualification; unrun checks go to `docs/TODO.md`.

Non-goals: context scaling past 262,144 tokens; executing the confidence head or GPU prefix selection; `--draft-head-quantization`, `--draft-body-quantization` and `--online-mxfp8` conversion of the DSpark draft; tensor-parallel preparation; DSpark for the Flash Next recipe (`ignition/recipes/qwen38/`); any `helios/`, `flare/` or `flare-cli/` change.

## Capabilities

### New Capabilities

`qwen38-dense-dspark`: the DSpark proposal method of the dense Qwen recipe — method selection, the pinned draft, the DaisyRose checkpoint, full-context draft history, the refusals, and the qualification statements.

### Modified Capabilities

None.

## Impact

- `ignition/recipes/declaration.py`, `ignition/cli/__init__.py`, `ignition/cli/preparation.py`, `ignition/cli/help.py`, `ignition/cli/display.py`, `ignition/cli/serve.py`, `ignition/cli/config.py`, `ignition/planning.py`: the method name, the per-method draft-token maximum, and the refusals.
- `ignition/components/weights.py`: the draft weights declaration for the rebuild manifest.
- `ignition/recipes/qwen38_dense/`: `recipe.py`, `bootstrap.py`, `weights.py`, `pool.py`, `decoder.py`, `config.py`, and three new modules.
- `docs/mtp-design.md`, `docs/cli-design.md`, `docs/model-preparation.md`, `docs/TODO.md`.
- Tests under `tests/cli/`, `tests/recipes/` and `tests/reference/`.
- No Protobuf, core-format or Helios change: the draft's history uses the existing paged key/value region kind, and the rounds use the existing speculative graph contract. Cores exported before this change load unchanged.
- Page cost under DSpark rises from 4,194,304 bytes per 128-token page to 5,505,024 bytes with FP8 draft history (about 31%) or 6,815,744 bytes with BF16 (about 63%).

## Blast Radius

1. `ignition/recipes/declaration.py:167` — `Speculation` accepts only `mtp` and `dflash2`.
2. `ignition/recipes/declaration.py:329` — `Recipe` rejects any other `speculative_methods` key.
3. `ignition/recipes/declaration.py:224` — `SpeculativeMethod` carries costs and draft files, no draft-token maximum.
4. `ignition/cli/__init__.py:194` — `--speculative-method` choices.
5. `ignition/cli/preparation.py:220` — the draft-token limit is the recipe's single `max_draft_tokens`; `:255` gates the method on `speculative_methods`.
6. `ignition/cli/display.py:45` — the plan report names the source "DFlash2" or "MTP module".
7. `ignition/cli/serve.py:320` — the serve banner names DFlash2 proposals.
8. `ignition/cli/config.py:196` — the configuration template's `speculative_method` comment.
9. `ignition/recipes/qwen38_dense/recipe.py:22` — preparation downloads the DFlash2 draft for every speculative plan; `:58` lists the checkpoints; `:67` declares the default method.
10. `ignition/recipes/qwen38_dense/bootstrap.py:77` — `mtp: bool` stands for "DFlash2 prepared"; `:116` to `:121` validate DFlash2 options; `:202`, `:209`, `:221` and `:262` build weights, layout, pool and the speculative owner from that boolean.
11. `ignition/recipes/qwen38_dense/weights.py:181` — loads `dflash2_weights.DraftWeights` whenever drafting is on.
12. `ignition/recipes/qwen38_dense/pool.py:14` — `page_specs` lists target pages only and ignores its `mtp` argument; `:36` adds the DFlash2 context rings to the state row.
13. `ignition/recipes/qwen38_dense/decoder.py:411` — target feature capture is sized by the DFlash2 window with compact prefill; `:477` reads the draft's target layers.
14. `ignition/recipes/qwen38_dense/contract.py:27` — every pool page region becomes a paged key/value history region, which covers draft pages without edits.
15. `ignition/recipes/qwen38_dense/dflash.py:13` — the DFlash2 owner the DSpark owner sits beside.
16. `ignition/recipes/inkling2/dspark.py:126` — existing noncausal GQA block attention over paged draft history with a staged tail; the mechanism this change reuses.
17. `ignition/recipes/inkling2/pool.py:60` — existing draft page declaration beside target pages.
18. `ignition/recipes/kimi31/dspark.py:33` and `:53` — existing YaRN frequencies and rotary gain for a DSpark draft.
19. `ignition/recipes/kimi3/dspark.py:304` and `ignition/recipes/minimax_m3/draft.py:443` — existing Markov transition composition; `ignition/components/parallel_draft.py:166` consumes it.
20. `ignition/recipes/kimi31/dspark_weights.py:115` — existing shape table for the Markov and confidence tensors.
21. `ignition/components/parallel_draft.py:77` — the shared round every draft uses.
22. `helios/src/runtime_core/protobuf.rs:25` — the only native comparison against a method name; it resolves draft-body conversion for `dflash2`, and another name yields none, which is correct for DSpark.
23. `docs/mtp-design.md:6`, `docs/cli-design.md:450`, `docs/model-preparation.md:354` — the statements of supported drafters, the method flag, and the dense Qwen section.
24. `tests/cli/test_speculative_method.py:25`, `tests/cli/test_draft_checkpoints.py:162`, `tests/recipes/test_declarations.py:27`, `tests/cli/test_models.py:327` — tests that enumerate methods, draft downloads, aliases and capacities.
25. grepped `dflash2` across `flare/`, `flare-cli/src/` and `helios/src/` — only `helios/src/runtime_core/protobuf.rs`; no native site selects behavior for a new method name.
26. grepped `dspark` across `ignition/recipes/qwen38_dense/` and `ignition/recipes/qwen38/` — no sites.
27. grepped `lessons.md` across `openspec/` — no file; no recorded lesson applies.
