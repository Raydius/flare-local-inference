## MODIFIED Requirements

### Requirement: Text projections load from the checkpoint's ModelOpt NVFP4 metadata
Preparation SHALL read the `quantization_config` `config_groups`/`ignore` form (4-bit, group 16 weights) through the shared parser, after the recipe itself has confirmed the checkpoint declares 4-bit group-16 input activations; mark every `model.language_model.layers.*.{self_attn.{q,k,v,gate,o}_proj, mlp.{gate,up,down}_proj}` NVFP4 and leave `lm_head`, `embed_tokens`, norms and `model.vision_*` BF16; q/k/v load as one fused NVFP4 group, the attention gate as its own projection, mlp gate/up as one fused group; input scales stored in `model-inputscales.safetensors` resolve through the index.

#### Scenario: Fused groups honor calibration
- **WHEN** layer weights load
- **THEN** q/k/v share one `weight_scale_2`/`input_scale` pair, gate/up share one, and the attention gate keeps its own, with no `ValueError` from the fused-group check

#### Scenario: Gemma 4 unchanged
- **WHEN** Gemma 4 NVFP4 configuration is parsed through the lifted parser
- **THEN** the resulting layer map equals the map the private Gemma parser produced

#### Scenario: Gemma 4 weights-only configuration parses
- **WHEN** a ModelOpt NVFP4 `config_groups.group_0` declares 4-bit group-16 `weights` and no `input_activations`
- **THEN** the shared parser returns the layer map, as the private Gemma 4 parser did

#### Scenario: Glimmer refuses a checkpoint without W4A4 input activations
- **WHEN** a Muse Glimmer config's `config_groups.group_0.input_activations` is absent, not 4-bit, or not group 16
- **THEN** the recipe's config loader raises `ValueError` naming the input activations before any layer map is built
