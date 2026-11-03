# qwen38-dflash2-speculation Specification

## Purpose

Serves Qwen3.8-Flash-Next with DFlash2 block drafts: a separate `qwen38_dflash` recipe exports a package whose speculative rounds propose up to seven tokens from the pinned `incoai/Qwen3.8-Flash-Next-DFlash2` draft, verified through Qwen's own hybrid-state verifier.

## Requirements

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
The recipe SHALL declare `draft_tokens=7`, `token_capacity=128`, rank-1 costs whose `state_row_bytes` equals the prepared layout's row bytes, `mixed_prefill=True`, and TP1 only, rejecting `--gpus` with more than one device.

#### Scenario: Multi-GPU rejected
- **WHEN** the operator passes `--gpus 0,1`
- **THEN** preparation fails before loading with a message naming the single-GPU restriction

#### Scenario: Mixed catalog matches the plan
- **WHEN** the operator exports `qwen3.8-flash-dflash2` with `--mixed-prefill`
- **THEN** the package holds the same mixed shapes the plan reports, for draft-enabled and draft-disabled modes, and the memory report lists them

#### Scenario: Mixed prefill is opt-in
- **WHEN** the operator exports without `--mixed-prefill`
- **THEN** the package holds no mixed graphs and the memory report lists no mixed shapes, as for qwen38

### Requirement: A mixed round drafts with DFlash2 around one packed target forward
When the plan selects mixed prefill, a mixed graph SHALL run the DFlash2 speculative round for its decode rows and prefill one prompt chunk in the same packed Qwen target forward. The packed forward SHALL tap the draft's target layers, and the round SHALL split the tapped features by role. It SHALL write the prompt chunk's last 2048 positions into the prompt request's rings, and the emitted verification columns into each decode request's rings. These are the same writes a separate prefill graph and speculative graph would make. The qwen38 MTP mixed round SHALL keep its call shape, mixed shapes, and state regions; its packed decoder keeps the `b*w+1` sample rows the role states gather into, and each speculation-enabled mixed signature carries the proposal depth (`width - 1`) the package binding requires.

#### Scenario: Mixed round equals separate graphs
- **WHEN** one prompt chunk and a speculative batch run through a DFlash mixed graph, and the same inputs run through a prefill graph followed by a speculative graph
- **THEN** commits, emitted tokens, and ring contents agree within the verification arithmetic's measured repeatability

#### Scenario: Prompt chunk longer than the window
- **WHEN** a mixed graph's prompt chunk holds more than 2048 tokens
- **THEN** only its last 2048 positions are written to the prompt request's ring, each at slot `position mod 2048`

#### Scenario: Layout without a residual region
- **WHEN** a DFlash prompt role is prepared with draft state maintained
- **THEN** no residual checkpoint is compiled, because the DFlash layout holds no `target_residual` region

#### Scenario: MTP mixed catalog keeps its shape
- **WHEN** the qwen38 recipe prepares its mixed catalog with MTP
- **THEN** `PreparedMixedRound` builds `PreparedPrefillAbsorption` and `PreparedSpeculativeRound` with the call shape used before this change, allocates no feature buffer, and the mixed shapes, state regions, and compiled residual checkpoints equal those prepared before; the packed decoder is built with `verification=False, sampler=False` and the mixed signatures carry `draft_tokens = width - 1`, so the package exports and loads

### Requirement: The draft runs on the shared DFlash2 component
The Qwen3.8-Flash-Next DFlash2 draft SHALL be prepared by the same shared DFlash2 component that serves Qwen3.8-27B and Muse Spark: its checkpoint loads through the shared loader with the shared tensor-inventory and BF16 checks (`ignition/components/dflash2_weights.py:37`, `:76`), its attention geometry is declared as the shared geometry record (`ignition/components/dflash2_geometry.py:11`), and the block forward, context writer, embedding, head, and candidate selector are the component's (`ignition/components/dflash2.py:44`, `:204`). The recipe SHALL add only what the component does not own: the Flash-Next checkpoint pin and configuration bounds, the hybrid-state ring regions, the feature rows that feed the component's context from Qwen's prefill, mixed-prompt, and verification states, the proposal-only round with the Qwen admission validator (`ignition/recipes/qwen38_dflash/speculative_model.py:69`), history copies (`ignition/components/parallel_draft.py:217`–`:220`), GDN commit, and the mixed prompt context (`ignition/recipes/qwen38_dflash/speculative_model.py:240`). No separate copy of the draft block SHALL exist in `ignition/components/`.

#### Scenario: One shared implementation
- **WHEN** `grep -rn "dflash_draft" ignition/ tests/ qualify/` runs after the change
- **THEN** it finds nothing, `ignition/recipes/nemotron_h/dflash.py` equals upstream's file, and the Flash-Next recipe imports the draft block, scratch, context, geometry, and loader from `ignition/components/dflash2*.py` only

#### Scenario: Flash-Next checkpoint contract
- **WHEN** a draft checkpoint lacks a tensor, carries a misshaped tensor, stores a non-BF16 tensor, or carries an unexpected tensor
- **THEN** the shared loader rejects it before opening the weights, for the Flash-Next geometry as it does for the 27B geometry

#### Scenario: Single GPU through the sharded path
- **WHEN** the recipe prepares on one GPU
- **THEN** the component's sharded embedding, head, and reductions run without a collective runtime and the recipe still refuses `world_size != 1` with the usage error it raises today (`ignition/recipes/qwen38_dflash/recipe.py:45`)

#### Scenario: History copies stay exclusive with vocabulary-parallel sampling
- **WHEN** the recipe's round is constructed with both history-copy buffers and vocabulary-parallel sampling
- **THEN** construction raises `ValueError("history copies and vocabulary-parallel sampling are exclusive")`

### Requirement: Context writes respect liveness and recency
The recipe SHALL present each prefill chunk, mixed prompt chunk, and verification round to the component's context writer so that a token's key/value is written only when the recipe marks it live — a prefill or prompt chunk's last 2048 positions (`ignition/recipes/qwen38_dflash/speculative_model.py:89`–`:101`), a verification round's accepted columns (`:104`–`:111`) — at ring slot `row × 2048 + position mod 2048`, and dead tokens (request id −1 or outside the live set) SHALL resolve to slot −1 so they never address another request's ring.

#### Scenario: Dead tokens
- **WHEN** a batch holds inactive rows whose request ids are −1
- **THEN** no ring slot of any live row is written by those tokens

#### Scenario: Prefill longer than the window
- **WHEN** a prefill chunk longer than 2048 tokens is written
- **THEN** only its last 2048 positions land in the request's rings, each at slot `position mod 2048`, and no two tokens of one chunk write the same slot

### Requirement: The prepared draft reproduces the DFlash2 reference under its storage contract
For the pinned Qwen3.8-Flash-Next-DFlash2 draft, one block through the shared component with key/value round-tripped through an E4M3 ring, followed by the candidate selector, SHALL match `ignition/reference/qwen38_dflash2.py`'s forward, candidates, and greedy walk with the reference's key/value rounded to E4M3 the same way, within the bounds recorded from the measured repeatability of that rounded reference (`tests/recipes/qwen38_dflash/test_draft_cuda.py:16`–`:49`: rings 2e-2, hidden and logits 5e-2 on real features; the fabricated-feature bound recorded separately), with the top-16 shortlist equal on real features and the walked tokens equal on real features except at a near-tie: a row whose walk leaves the reference's SHALL do so only at a position where the reference's own top-two corrected selector scores differ by no more than the recorded near-tie tolerance, the largest gap at which the reference's own walk flips under one-ulp noise on its features or block inputs (measured 0.1053 over 8 draws × 2 kinds on the twelve batch-4 cuts of trace chat-capacity-10c824cef65b8fb7, 2026-11-03; recorded as 0.11), and the prepared token SHALL be the reference's choice within that gap. The bounds and the tolerance SHALL NOT be relaxed to accommodate the component's arithmetic; a bound the component cannot meet is reported with the measured values.

#### Scenario: Parity on real tapped features
- **WHEN** the CUDA test runs on GPU 1 with the pinned draft and a trace of the target's tapped features
- **THEN** the component's hidden states, logits, and rings match the E4M3-rounded reference within the recorded per-output bounds, the top-16 shortlist equals the reference's on every row, every walk that leaves the reference's does so at a near-tie within the recorded tolerance with the gap printed, and ties resolve to the lowest token id in both

#### Scenario: Fabricated features
- **WHEN** no trace is available
- **THEN** the test runs on random features against the same E4M3-rounded reference and records that bound separately, skipping no assertion silently

### Requirement: The folded package serves the measured baselines
A package exported from the folded recipe SHALL serve the frozen speculation workload through `flare run` with clean transport and no failed requests, and its code_edit output throughput SHALL be compared against the pre-fold package's 329.1 tokens per second at concurrency 1 and 516.5 at concurrency 8 (`qualify/results/dflash2-served/summary.json`, vault DFlash2 Recipe Progress §12), with the drafter's offered and accepted token counts recorded beside them.

#### Scenario: Requalification
- **WHEN** `qualify.dflash2_served` runs trials `0:1:dflash2-adaptive` and `0:8:dflash2-adaptive` against the new entry
- **THEN** both trials report `transport_passed` with zero failed requests, and the medians and acceptance counts are recorded against the baselines; a median more than 5 % below its baseline is reported as a finding, never absorbed
