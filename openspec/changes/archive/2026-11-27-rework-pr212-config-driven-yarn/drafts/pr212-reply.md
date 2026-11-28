@yatesdr the branch is rebuilt on `main` `74618652` as one core commit, `3c73cd1d`, and the recipe half is local-inference-lab/hyperflare-recipes#68 (`671b6609`). The description above is rewritten. By item:

1. **Attention factor.** The recipe now calls the installed `ROPE_INIT_FUNCTIONS["yarn"]` for both the inverse frequencies and the attention factor, and multiplies the cos and sin tables by the factor where they are built. `interpolated_frequencies` and the copied reference formulas are gone. The tests compare gathered tables with `Qwen4ExpTextRotaryEmbedding` at positions through 1,048,575 with zero tolerance, and one case pins ramp bounds 14 and 22 and factor 1.1386. `attention_factor`, `mscale` and `mscale_all_dim` are passed to the reference instead of refused. The fused Q/K rotation CUDA test gains a YaRN case: 60 passed on an SM120 GPU.
2. **One general mechanism.** The alias and `--rope-parameters` are removed. The recipe reads `text_config.rope_parameters` from the checkpoint's `config.json`, as the model card gives it, in one validator that refuses any `rope_type` other than `default` or `yarn` and any key it does not implement. The ceiling reaches the plan through the two optional `Checkpoint` fields this pull request adds, and the plan report records `checkpoint_context` and the full `rope_parameters`, `beta_fast` and `beta_slow` included when the config sets them.
3. **SM103 scratch sizing.** The `workspace_context` hunk is removed; the SM103 score-workspace lines are byte-identical to recipes `main`. A scaled context on SM103 is refused at validation ("a rope-scaled context is not measured on SM103") and listed in `docs/TODO.txt` with its removal criterion. It is not implemented and not measured.
4. **Conflicts.** None against `74618652`; the branch holds no recipe-owned file.
5. **Evidence.** Rerun on the reworked heads, one RTX PRO 6000 (SM120), packages `gb202-tp1-202611280514` (YaRN config, 1,048,576) and `gb202-tp1-202611280608` (native, 262,144) from one image:
   - Export and serving: both export and serve; `/v1/models` reports 1048576 and 262144.
   - Passkey, YaRN package: 18/18 at 8k, 64k, 200k, 320k, 520k and 1,000k tokens, three depths.
   - Beyond retrieval, RULER multi-hop variable tracking (four hops), YaRN package: 18/18 on the same grid.
   - Short-context comparison at 8k, 64k and 200k, both tasks: YaRN 18/18, native 17/18, 17 of 18 greedy replies identical. The differing cell is tracking at 8k, depth 0.5, where the native package's reply was an explanation cut at 32 tokens.

   This shows no degradation on two short-reply tasks. It does not measure free-form generation quality across the extended range.
6. **Description.** Rewritten in the four-part order, without process statements.

On the Autobot review:

- **Predates the recipe split:** split as above; this branch touches `ignition/declaration.py`, `planning.py`, `parameters.py`, `preparation/bootstrap.py`, `cli/serve.py`, `cli/help.py` and one test.
- **Two mechanisms:** both are removed in favour of the checkpoint config.
- **Acceptance record:** the description and the recipes `docs/TODO.txt` now agree and name the heads.
- **Unbounded scaling:** the validator caps the ceiling at 1,048,576 and requires 64 alignment; `factor: 1000` is refused by name.
- **Identity omits `context`:** the ceiling is `factor` times a fixed original, both in `config.json`, which the identity already hashes; a test pins that native, factor-2 and factor-4 identities differ.
- **Docs:** `docs/model-preparation.md` in the recipes repository documents the scaled directory; no flag remains to document.
- **Evidence claims:** restated per selector and per head; no claim about b12x chunking remains.
