## REMOVED Requirements

### Requirement: The recipe pins its own target and draft
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: Tapped target features feed the draft context
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: The draft ring is per-request state restored as plain state
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: A round proposes a DFlash2 block and verifies through Qwen's hybrid state
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: Declared capacities and costs match the prepared layout
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: A mixed round drafts with DFlash2 around one packed target forward
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: The draft runs on the shared DFlash2 component
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: Context writes respect liveness and recency
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: The prepared draft reproduces the DFlash2 reference under its storage contract
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: The folded package serves the measured baselines
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: The recipe declares a YaRN x4 checkpoint
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.

### Requirement: A scaled package serves and drafts past the native context
**Reason**: The `qwen38_dflash` recipe is retired; at 16 sequences and a 1M context it measured no better than the `qwen38` recipe with speculative method `dflash2` and could not hold a 1M prompt (`qualify/results/dflash2-paths-c16/README.md`).
**Migration**: Export Qwen3.8-Flash-Next DFlash2 packages with the `qwen38` recipe and speculative method `dflash2`: `ignition export MODEL --speculative-method dflash2`, where MODEL is an alias, a repository id, or a checkpoint directory (a directory also takes `--recipe qwen38`); no alias pins the QAD target. Add `--rope-parameters` for a 1,048,576-token context and `--draft-model` for a local draft folder. Packages already exported with the retired recipe keep loading.
