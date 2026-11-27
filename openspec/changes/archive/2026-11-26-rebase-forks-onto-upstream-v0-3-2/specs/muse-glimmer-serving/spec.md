## MODIFIED Requirements

### Requirement: The decoder matches the canonical forward
The prepared decoder SHALL compute each layer as `input_layernorm → attention → post_attention_layernorm → residual add → pre_feedforward_layernorm → SwiGLU → post_feedforward_layernorm → residual add` with centered (1 + w) RMSNorms, attention with scale-less q/k RMSNorm, query × `qk_scale_factor`, rope 500000 on sliding layers and no positional encoding on full layers, the sigmoid gate applied to the raw attended value (no output norm), FP8 E4M3 sliding rings of 2048 and 128-token paged full-layer histories, the normed embedding, and logits `20·tanh(logits × output_multiplier / 20)`. The attention SHALL run through the recipe's own module in the recipes repository (`hyperflare_recipes/recipes/muse_glimmer_nvfp4/attention.py`), whose forward takes metadata per call and whose separately stored gate projection multiplies the raw attended value; Muse Spark's attention module SHALL stay as upstream ships it.

#### Scenario: Reference parity
- **WHEN** a traced prompt longer than 2048 tokens runs through the prepared prefill and decode graphs and through `hyperflare_recipes.reference.muse_glimmer`
- **THEN** the softcapped logits agree within 1e-1 relative L2 with top-1 agreement of at least 0.98 and the reference argmax inside the prepared decode top-3, and the residual stream and final hidden states sit no farther from the FP8-history reference than twice the distance the reference itself moves when only its history precision changes (E4M3 to BF16), per layer and on the final hidden, on sliding and full layers across the window boundary

#### Scenario: Spark unchanged
- **WHEN** the Muse Glimmer recipe is present on a recipes-repository branch cut from an upstream tip
- **THEN** `git diff <upstream tip> -- hyperflare_recipes/recipes/muse_spark/attention.py tests/recipes/test_muse_spark_attention.py` is empty in the recipes repository, and that repository's TODO document carries the entry that folds Glimmer's attention back onto Spark's once Spark's attention takes metadata per call
