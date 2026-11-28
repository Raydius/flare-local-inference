Title: planning: let a checkpoint declare its own context ceiling

**Why.** A recipe's `context` is the only serving-context ceiling a plan reads, so every checkpoint a recipe resolves plans for the same longest sequence. A checkpoint whose own configuration serves a different length cannot be described. Qwen3.8 Flash Next trains 262,144 positions, and its [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) extends the same weights to 1,048,576 tokens by changing only the YaRN `rope_parameters` in `text_config`. Carrying the ceiling on the resolved `Checkpoint` lets the recipe that reads that configuration report it through the existing `inspect_replacement` hook, with no flag, no alias for the same weights, and no model knowledge in core.

**What changed.** `Checkpoint` gains two optional fields, `context` and `rope_parameters`, both `None` by default. Every site that read `recipe.context` now reads `checkpoint.context or recipe.context` (`Plan.declared_context`):

- `Plan.context`, `Plan.guaranteed_tokens` and `Plan.demand_pages`;
- the plan report's `checkpoint_context`, which also gains a `rope_parameters` entry only when the checkpoint sets one;
- the `context` passed to a recipe's `PreparationConfig`;
- the request body limit, the `--prefill-chunk-size` bound, and the context line of `ignition export MODEL --help`.

No Protobuf, Helios, CLI flag, or configuration key changes; a prepared core already carries its ceiling as `max_sequence_tokens`. The `qwen3.8-flash-yarn4` alias, the `--rope-parameters` flag, the shared interpolation helper, and every recipe file of the earlier revision are gone from this branch. The recipe side is local-inference-lab/hyperflare-recipes#68.

**Result.** A checkpoint that sets neither field plans and reports exactly as before. A checkpoint that sets `context` is planned, bounded, and reported at that ceiling. No recipe in this repository sets the fields.

**Evidence.** One commit, `3c73cd1d`, on `main` `74618652`; 7 files, +50 −10.

- `tests/cli/test_planning.py::test_checkpoint_context_replaces_the_recipe_ceiling` plans one fixture recipe with and without `context=1048576` and checks the ceiling, guaranteed tokens, demand pages, report entries, and prefill-chunk bound. It passes on `3c73cd1d` and is absent on `74618652`.
- `pytest tests/cli`: 126 passed and 5 failed on `3c73cd1d`, 125 passed and the same 5 failed on `74618652`. The 5 need a `flare_heliopause` build this host lacks.
- `tests/preparation/test_core_boundary.py` passes. `tests/preparation` as a whole was not run: it crashes the interpreter on this host on `74618652` as well.
- Served through this field, with the linked recipe change at `671b6609`, on one RTX PRO 6000 (SM120): a Qwen3.8 Flash Next checkpoint directory carrying the model card's YaRN block exported as `gb202-tp1-202611280514` and served with `max_model_len` 1048576; the pinned native checkpoint exported from the same image as `gb202-tp1-202611280608` and served with `max_model_len` 262144. The quality measurements on those two packages are in the recipes pull request.

Not measured: SM103 and tensor-parallel execution. The recipe refuses a scaled context on SM103.
