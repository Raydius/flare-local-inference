## Purpose

Serves Qwen3.8-Flash-Next with DFlash2 block drafts: a separate `qwen38_dflash` recipe exports a package whose speculative rounds propose up to seven tokens from the pinned `incoai/Qwen3.8-Flash-Next-DFlash2` draft, verified through Qwen's own hybrid-state verifier.

## ADDED Requirements

### Requirement: The recipe pins its own target and draft
`ignition export qwen3.8-flash-dflash2` SHALL prepare the `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` target at revision `dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` (the checkpoint the acceptance spike measured and the operator serves) from the hub cache, with the DFlash2 draft at revision `a22a1744…` fetched into the hub cache, on one GPU; the alias declares that repository itself, so neither it nor the qwen38 aliases share a repository, and bare repository or directory resolution keeps today's behavior.

#### Scenario: Alias export
- **WHEN** the operator runs `ignition export qwen3.8-flash-dflash2 --gpus 0` with the pinned target snapshot in the hub cache
- **THEN** preparation loads that target, downloads or reuses the pinned draft, and writes a package whose identity differs from the MTP recipe's

#### Scenario: Declared bytes match the checkpoint
- **WHEN** the declaration test inspects the pinned target's local safetensors index
- **THEN** the declared stored bytes equal the index's `total_size`, and the MTP aliases still resolve to the qwen38 recipe

### Requirement: Tapped target features feed the draft context
Every prefill and verification graph SHALL write, for each token, the mean over the four hyperconnection streams of the residual after completed layers 5, 15, 25, 35, and 45 into a `[tokens, 12800]` BF16 feature buffer — the same residual the decoder's layer trace records, as the prepared decoder computes it — and the draft context SHALL project those features into per-layer key/value records for the last 2048 positions of each request. The prepared decoder's residuals diverge from the canonical reference with depth (measured 0.03–0.06 relative at layer 5 up to 0.16–0.41 at layers 35 and 45 on the acceptance traces), so the tap is qualified by acceptance: DFlash2 acceptance measured on the prepared tap features SHALL stay within a few percent of the acceptance measured on canonical features and above the MTP draft's.

#### Scenario: Tap check on a traced prompt
- **WHEN** a traced prompt is prefilled through the compiled prepared decoder with the taps (`qualify/dflash2_tap_check.py`)
- **THEN** the feature buffer holds every prompt position, the ring slots are live for exactly the last 2048 positions, the per-layer error against the canonical trace is recorded, and the eager prepared decoder reports the same error profile

#### Scenario: Acceptance on prepared features
- **WHEN** the acceptance traces are re-prefilled through the prepared tapped decoder and measured with E4M3-rounded key/value
- **THEN** code_edit acceptance at widths 4 and 8 is within two percent of the canonical-feature measurement (3.744 / 6.394 against 3.760 / 6.537) and above MTP's (3.654 / 5.622)

#### Scenario: Prefill longer than the window
- **WHEN** a request's prompt chunk holds more than 2048 tokens
- **THEN** only its last 2048 positions are written to its ring, each at slot `position mod 2048`, and no two tokens of one chunk write the same slot

### Requirement: The draft ring is per-request state restored as plain state
Each state row SHALL hold five E4M3 key and value windows of 2048 positions × 8 heads × 128; snapshots and restores SHALL copy them unchanged (draft lag 0), the layout SHALL carry no MTP-only regions and report no draft ring capacity, and the qwen38 and qwen38_dense layouts SHALL stay byte-identical.

#### Scenario: Prefix restore resumes drafting
- **WHEN** a request resumes from a prefix snapshot taken after N consumed tokens
- **THEN** its next round drafts against the restored windows at positions up to N − 1 with no feedback residual or draft page table involved

### Requirement: A round proposes a DFlash2 block and verifies through Qwen's hybrid state
A speculative round of width w SHALL run one draft forward over `[pending, mask × 7]`, correct each of the first w − 1 positions' logits with the predecessor-conditioned selector, sample proposals with the shared draft sampler, substitute history copies when offered, verify through the Qwen target with GDN rollback records, commit accepted tokens with draft context lag 0 and the round's copy mask, and write the accepted tokens' features into the ring.

#### Scenario: Seven drafts accepted
- **WHEN** a width-8 round's seven proposals all pass verification
- **THEN** eight tokens are emitted, the GDN and QSA state advance by eight, and eight ring positions are written

#### Scenario: Width-one round
- **WHEN** the runtime selects width 1
- **THEN** no draft forward runs, the pending token is verified as ordinary decode, and its feature is written to the ring so a wider round can follow

#### Scenario: History copies
- **WHEN** the package is loaded with `--history-drafts` and a lane carries three copies for the round's consumed count
- **THEN** positions 0..2 propose the copies with q = 1 and the commit records mask 0b111

### Requirement: Declared capacities and costs match the prepared layout
The recipe SHALL declare `draft_tokens=7`, `token_capacity=128`, rank-1 costs whose `state_row_bytes` equals the prepared layout's row bytes, `mixed_prefill=False`, and TP1 only, rejecting `--gpus` with more than one device.

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** preparation fails before loading with a message naming the single-GPU restriction
