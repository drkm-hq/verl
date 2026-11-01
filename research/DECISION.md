# Decision: the contributions to build (2026-10-29 ~03:50Z)

(The E-series vetting notes on edge deployment that this decision cites are not included in this repository.)

Inputs: 14 research reports (research/00-13), 20 vetted candidates (research/vet/*.md), the sweeper's three
additions (S1-S3), the Modal harness baseline (bench/README.md: BASELINE-A 8xH100 Qwen3-1.7B GRPO n=8,
step 31.2 s = gen 12.8 + old_log_prob 3.5 + ref 3.4 + update_actor 8.7; GPU suite 503/0/8 on main).
the author's criteria: maximum speed/performance; stability with self-healing; enable RL at the edge (folded into a
big contribution or standalone beside one); novel and needed; nothing pushed until his review.

## Vetting outcome
GO-WITH-CHANGES: K3 old-log-prob skip (17/25), K2 loss-row-only LM head (15), K11 selective activation
checkpointing (15), K7 GRESO prefilter (15, multi-epoch only), S1 FP8 ref forward (17, certainty 2),
S2 agent-loop worker liveness (16), E4-narrowed in-flight TTL + re-issue + dead-worker redispatch,
E3 turn-level rewards + done + turn-LOO (edge), E1-min self-healing LoRA swap (needs #7496), E5 HTTP token
routes (no ask), K12 V1 loss-horizon bug (correctness), S3 deterministic V1 ordering (borderline).
NO-GO (evidence in the vet files): K1 VLM dedupe (6% capture, torch-2.9 artefact), K4 weight-sync bytes (3-6% of
a 2 s sync; #8110), K4b/E1 LoRA-disaggregated bug (true; duplicate of #7495/#7496), K5 vectorized GAE
(maintainers: "trivial"), K6 TQ slicing (1-2%, #8203), K8 group routing (#6122 unanswered), K9 straggler abort
(exists as colocate_async), K10 ref bf16 (#7182/#8189), K13 FA kwargs (0.4%), K14 async ckpt (#8056/#7955),
E2 multi-adapter (infeasible on default FSDP), E6 trajectory ingest (works outside verl; #8203).

## The pick
Three big contributions, one edge contribution beside them, one small starter.

1. P1 `[fsdp, perf] feat: project the LM head only for loss rows` (K2). Speed + memory on the DEFAULT FSDP
   path; bitwise-identical response log-probs/entropy/gradients (CPU proof on verl's real prepare_model_inputs/
   outputs); head FLOPs and (T,V) logits bytes drop by the prompt fraction (-10..-13% on the three training
   passes at a 6k/1k shape, -5..-8 GB per rank, step -4..-6%; Megatron twin #7686 measured -8..-16%). The
   long-observation / short-action agent loops of the edge use case are exactly the workloads it helps most.
   No open or merged PR does this on FSDP (#7370 fused-only, closed unreviewed). ~260 lines, opt-in flag.
2. P2 `[trainer, perf] feat: skip the old-log-prob pass when strictly on-policy` (K3). Exact (bitwise CPU
   proof incl. gradient accumulation) when ppo_epochs=1 and one mini-batch per sampled batch; removes one of
   five forward passes: old_log_prob 3.5 s -> 0 of a 31 s step (-11%) for the single-mini-batch configs that
   13/100 shipped scripts use (every Qwen3.5 / DeepSeek-V4 / gpt-oss / Qwen3-235B GRPO example). The legacy
   worker shipped the same shortcut (#3119, merged) and #6067 deleted it as collateral; #3996/#6280 invite a PR.
   Opt-in with hard validation; entropy under no_grad in the update pass; ~110 lines.
3. P3 `[trainer, rollout, cfg] feat: V1 steps never hang: in-flight prompt TTL with re-issue and dead
   agent-loop worker redispatch` (E4-narrowed + S2). Self-healing for the DEFAULT sync mode and for edge rollout
   fleets: today any AgentLoopWorker death leaves its groups `running` forever (reproduced on CPU with the real
   classes: sample() blocks forever, the next submit raises RayActorError, no actor sets max_restarts). The
   replica-side fence/restart is owned (#8168 / RFC #8380, which explicitly leave CPU workers out); this slice is
   unowned. ~230 source + ~250 test lines; Ray-local CPU chaos tests. Known risk: #8203 (V1 owner, BREAKING)
   rewrites the same files; keep the hooks minimal and document the rebase plan.
4. P4 (edge, standalone beside the big ones) `[rollout, algo, tool] feat: turn-level rewards, episode done and
   a turn-LOO advantage for multi-turn agent loops` (E3). Unlocks an observe->act->reward loop: tool step
   rewards are produced today but consumed by nothing; no `done`; a score on a masked tail token is dropped
   (bug #7 in 09). The exact policy-gradient check (400k groups) shows the per-turn-index baseline is biased and
   RFC #7244's LOO return-to-go is unbiased. ~170 source + ~220 test lines; draft #8305 overlaps only span plumbing.
5. P0 starter `[algo, trainer] fix: resolve the seq-mean-token-sum-norm horizon to response_length under V1`
   (K12) + the micro-batch invariance test suite. Confirmed through the real ppo_loss/value_loss (losses differ
   up to 48x between micro-batch splits in the Dr. GRPO mode; a constant factor restores exact invariance).
   ~150 lines incl. tests, quiet zone. Recommended as the submitter's FIRST submission: only 7 of the last 400 merges came
   from first-time authors, and a small, certain fix establishes contributor status before P1-P4.

Deferred (documented, not built): K11 selective checkpointing (-4% step, ~70 lines; next speed PR after P1),
S1 FP8 ref forward (roadmap P1, needs Hopper + numerics study), E1-min (after #7496), E5 HTTP routes, S3.

## Proof plan
P1: Modal long-prompt variant (6k filler prompt / 1k response) main vs branch, plus BASELINE-A for no-regression;
    GPU unit tests on the branch. P2: BASELINE-A with ppo_mini_batch_size=128 (k=1), main vs branch; GPU tests.
P3: Ray-local CPU chaos tests (kill a worker mid-generation: step completes; dead-worker redispatch), plus a
    Modal 2-GPU separate_async chaos run if the harness preset is added. P4: CPU tests (mocked tool loop with
    known per-turn rewards -> dense reward tensor -> advantages) and the estimator bias test; optional toy env run.
P0: the failing test on main, passing on the branch; CPU regression set green.
