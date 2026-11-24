# Handoff — rebase deploy onto upstream ba42481a — DONE (2026-11-19)

Published: patch/00 a8994f59, 01 ca0291d2, 03 020052e9, 06 bf5ae6c0, 07 9e5d893c, deploy dd62eb13 (7c8dcb23 at the gate; dd62eb13 retires the #211 row) (each leased vs backup/pre-rebase-20261119-*), base/current = main = ba42481a (local + origin).
Gate fix: 7c8dcb23 adds upstream's RenderedPrompt::ordinary to the fork's block-end-marker test (Helios lib tests did not compile).
Evidence: ~/models/.cache/rebase-ba42481a/{rust-gate.log,rust-gate-rerun.log,rust-gate-rdv.log,gpu-gate.log,elephant-check.json}.
Open: vault notes / vllm-env build README not updated for 7c8dcb23; ~/LocalApps/vllm-env has the uncommitted gdn-head-ratio patch update; image tag is 02e4f749 (7c8dcb23 changes only a test).
