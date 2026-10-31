# Handoff — qwen38-dflash2-recipe

## Objective and location
DFlash2 change 2 of 3: the `qwen38_dflash` recipe (Qwen taps + separate recipe) for Qwen3.8-Flash-Next. Worktree `/home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk`, branch `wt/dflash-s-e2vk`, HEAD 861b791f. Change dir `openspec/changes/qwen38-dflash2-recipe/` (untracked, uncommitted; no commit was requested).

## Stage and stopping point
Propose complete 2026-10-29: proposal, design (peer review + bomb-squad transcribed), three spec deltas, tasks (8 tasks, `Concurrency: graph`) — all lint clean, `openspec validate` ok. No code edited yet; no task ticked.

## Next action
`/rad-apply qwen38-dflash2-recipe`. Ready frontier: 1.1, 2.1, 3.1, 7.1 (independent); 4.1 after 3.1; 5.1 after 1.1/2.1/3.1/4.1; 6.1 after 1.1/4.1; 8.1 is the operator leg.

## State that shapes apply
- GPU 0 (RTX PRO 6000) runs the owner's production Qwen instance — never stop it; it is the only GPU that fits the target, so task 8.1 (tap check + export + serve) waits for the owner. All other CUDA tests: `CUDA_VISIBLE_DEVICES=1` on the host `.venv` (torch 2.14.1+cu132, CUDA available).
- Checkpoints on host: draft `~/models/incoai/Qwen3.8-Flash-Next-DFlash2` (`--local-dir` layout, pinned a22a1744), target `~/models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD`, Nemotron + LiLiCorr under `~/models/nvidia/`.
- Peer review certainty 0.85 (< 0.9 floor, advisory) and caller agreement 0.6 are recorded in design.md; rejected point (BF16 rings) has its reason there.
- Parked, unrelated: Nemotron serves wrong text on this host (`openspec/debug/nemotron-serve-device-error.md`, H1–H18, cause = Inductor-compiled forwards, next experiment H19 written in the ledger). Blocks only the archived history-drafts change's task 1.6 served check. Leave as is.
- Model note: the propose session ran on Fable 5.1; AGENTS.md bars Claude Opus 5.5 from kernel/fusion work — the tasks avoid Triton edits by design (design item 5).

## Owner's verbatim words
- "Park the Nemotron ledger as-is, then propose the Qwen taps + qwen38_dflash recipe change." (chosen option, 2026-10-29)
- Carried from 2026-10-28: "I agree with this being a separate recipe." / "my usage is almost entirely code generation" / "typically the concurrency is <8"

## Resume prompt
```
Read openspec/changes/qwen38-dflash2-recipe/handoff.md first. Working location: worktree /home/rjdollete/dev/flare-local-inference.worktrees/dflash-s-e2vk, change qwen38-dflash2-recipe (planning complete, apply not started). Run /rad-apply qwen38-dflash2-recipe. Do not touch GPU 0 or the owner's Qwen instance; CUDA tests go to CUDA_VISIBLE_DEVICES=1.
```
