# Handoff — muse-glimmer-text-recipe

## Objective and location
Serve Muse Glimmer 30B NVFP4 (text + vision + DFlash drafter) on one GPU through a new `ignition/recipes/muse_glimmer/` recipe. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb`, branch `wt/muse-glimmer-dr-usb` at `58ac27be` (= `origin/deploy`). Change dir `openspec/changes/muse-glimmer-text-recipe/` — untracked, uncommitted; nothing else dirty.

## Stage
Planning complete (proposal, design, spec delta, tasks all lint-clean; Tier 3; peer-review quorum 3/3, certainty 0.95; bomb-squad recorded in design.md). No task started. Next: `/rad-apply`. Ready frontier: 1.1, 1.2, 1.3, 1.4, 2.1 (independent `owns:`).

## Owner's verbatim words
- Scope answer: "Everything" (text, vision and DFlash2 in one change); earlier "Why text only?" rejected a text-only scope.

## Facts the next session must not re-derive
- Measured on the HF checkpoint (all 52 layers): `input_scale` equal across q/k/v/gate and gate/up; `weight_scale_2` equal across q/k/v and across gate/up, but `gate_proj` ≠ q/k/v every layer → attention gate is its own NVFP4 projection (design.md "Measured").
- Glimmer has no attention output norm; Spark's `attention.py:179` norm becomes `output_norm=True` default (design.md).
- `components/varlen_attention.py` `window` is 1-D — vision window layers use window-ordered patches + `cu_window_seqlens` + `reverse_index` from Helios (design.md Vision), not MiMo's 2-D window.
- Live bug to fix in task 2.1: `helios/src/frontend/images/conditioned.rs:184-191` `_ => 588` misclassifies `PatchMerge` (1176).
- Assistant checkpoint `meta-models/Muse-Glimmer-30B-assistant@c676cce9` answered HTTP 200 unauthenticated; tensor list/shapes in proposal.md Blast Radius 15.
- Transformers sources for parity were fetched to `/tmp/glimmer-src/` (rev `1d9654e7`) and config/tokenizer/index to `/tmp/glimmer-cfg/`, `/tmp/glimmer-st/` — temporary; re-fetch if gone. No local HF snapshot of Glimmer exists yet (`ls /data/cache/huggingface/hub | grep -i glimmer` empty).
- Host GPU: RTX PRO 6000 Blackwell 96 GB (SM120) + RTX 5060 Ti 16 GB. Project venv: `/home/rjdollete/dev/flare-local-inference/.venv` has no transformers; `../dflash-s-e2vk/.venv` has it.

## Resume prompt
```
Read openspec/changes/muse-glimmer-text-recipe/handoff.md in /home/rjdollete/dev/flare-local-inference.worktrees/muse-glimmer-dr-usb, then run /rad-apply on muse-glimmer-text-recipe starting from the ready frontier (radflow-lint --ready-frontier).
```
