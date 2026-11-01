# Vetting phase (started 2026-10-29 02:35Z)

One file per candidate, written by an adversarial vetter. Schema per file:
1. Claim (one paragraph, source report + section)
2. Mechanism verification: file:line evidence; CPU test script + output if testable
3. Duplicate / ownership check: open PRs (number, title, state, files overlap), merged PRs, RFCs, maintainer statements
4. Minimum valuable PR: files, functions, estimated lines, config keys (default off?), tests (CPU / GPU), docs
5. Proof: exact benchmark config on the Modal harness (bench/README.md), metric, expected effect with arithmetic
6. Adversarial answers A1-A5 (research/VETTING-RUBRIC.md)
7. Gates G1-G5 pass/fail, scores S1-S5, verdict GO / GO-WITH-CHANGES / NO-GO, strongest reason

Candidate ids: K1 vlm-dedupe, K2 lmhead-rows, K3 oldlogprob-skip, K4 weightsync-fastpath, K5 gae-vectorized,
K6 v1-slice-at-source, K7 greso-prefilter, K8 group-affinity-routing, K9 over-sample-abort, K10 ref-bf16-residency,
K11 selective-ckpt, K12 v1-horizon-bug, K13 fa-kwargs-once, K14 fsdp-async-ckpt-save,
E1 lora-edge-refresh, E2 multi-adapter, E3 turn-rewards-episodes, E6 device-trajectory-ingest,
E4 self-healing-fleet (after report 10), E5 edge-runtime-entrypoint (after report 12).
