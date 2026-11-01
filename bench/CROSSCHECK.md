# Independent cross-check of the A/B numbers (computed from bench/results/*/summary.json, steps 2..N means,
# ± = half the max-min spread over the runs in the group; 8xH100, verlai/verl:uv.cu130.dev5). 2026-10-29 05:5xZ.

## P1 loss-row-only LM head
| group | n | gen | old_log_prob | ref | update_actor | training side | step |
|---|---|---|---|---|---|---|---|
| long-prompt main | 2 | 17.02±0.04 | 8.06±0.03 | 6.59±0.00 | 25.86±0.06 | 40.52±0.10 | 62.16±0.07 |
| long-prompt branch, flag OFF | 1 | 16.61 | 8.06 | 6.55 | 26.00 | 40.61 | 61.87 |
| long-prompt branch, flag ON | 2 | 15.77±0.20 | 6.19±0.03 | 5.84±0.02 | 23.65±0.08 | 35.68±0.09 | 54.08±0.12 |
| BASELINE-A main | 2 | 12.58±0.19 | 3.52±0.03 | 3.41±0.02 | 8.82±0.11 | 15.74±0.11 | 31.25±0.10 |
| BASELINE-A branch, flag ON | 2 | 10.49±0.02 | 3.43±0.05 | 3.05±0.01 | 8.08±0.01 | 14.56±0.06 | 28.89±0.35 |
Reading: at 6k-token prompts the training side drops 12.0% (40.5 -> 35.7 s) and the step 13.0%; flag OFF
equals main (no-op). At 83-token GSM8K prompts the training side drops 7.5%; the gen difference there is node
noise (P1 does not touch generation). Spreads are tight (<=0.2 s) on these runs.

## P2 on-policy old-log-prob skip (single mini-batch preset)
| group | n | gen | old_log_prob | ref | update_actor | training side | step |
|---|---|---|---|---|---|---|---|
| main | 2 | 12.98±0.19 | 3.68±0.04 | 2.89±0.01 | 9.26±0.15 | 15.83±0.12 | 31.78±0.38 |
| branch, flag OFF | 3 | 12.19±1.99 | 3.88±0.64 | 2.89±0.41 | 8.85±0.65 | 15.62±1.67 | 32.46±5.03 |
| branch, flag ON | 5 | 13.24±2.08 | 0.00±0.00 | 3.89±0.64 | 10.90±1.36 | 14.79±2.01 | 31.94±4.49 |
Reading (honest): the old-log-prob pass is removed (3.7 s -> 0), but update_actor grows by ~1.0-1.7 s (the
entropy metric and the log-prob write-back now happen in the update pass) and the ref pass reads ~0.5-1.0 s
slower on the ON arms (not touched by the code; two ON runs landed on a slow node where gen and ref were slower
for every phase, and a warm-up effect may have shifted from the removed pass). Net training-side gain: -5% to
-7% on the means (15.6-15.8 -> 14.8 s); the step clock is inside the ±5 s rollout noise. This is less than the
vet's -11%-of-step estimate; the PR body must state the measured net, the update_actor increase and its cause.
Follow-up that would recover more: make the entropy metric optional under the skip (actor/entropy off by
default) so the update pass does not inherit the entropy cost.

## P3 self-healing (2xH100 separate_async, agent-loop worker SIGKILLed after step 2)
| arm | src | outcome |
|---|---|---|
| main x3 | 2ab35292 | crashed (rc=1) x3 |
| combined branch, knobs OFF | 5e6f6b74 | crashed |
| combined branch, knobs ON x2 | 5e6f6b74 | completed x2 |
| p3a [1/2] TTL only | 7f43e43e | crashed (expected: the only worker is dead, nothing to re-issue to; documented) |
| p3b [2/2] knobs OFF | e43e784c | crashed |
| p3b [2/2] knobs ON x2 | e43e784c | completed x2 |
| p3b [2/2] knobs ON, FINAL | f6846af0 | completed |
No-chaos arms (sep2-p3-on-1/2, sep2-p3b-on-1): step 3.8-4.1 s vs main 3.8-4.0 s = no regression.

## GPU unit suite (gpu_unit_tests.yml selection, 8 GPUs)
Count note: pytest's console line is authoritative: `503 passed, 8 skipped` over 511 test cases; the junit
`testsuite tests=529` attribute also counts 18 subtests, which is why an earlier draft here said 521. 0 failures either way.
main 503 passed / 0 failed / 8 skipped; p0 (6684917e) 503/0/8; p1 (4bcbbeb9) 503/0/8; p2 (1e45a508) 503/0/8;
later shas and p3b/p4b: see the benchmark agent's tables in bench/README.md and packages/*/bench/RESULTS.md.

# Review round 2 (2026-10-29 13:05Z ->): numbers read from bench/results/*/summary.json and metrics.jsonl by the bench agent

## P4 enabled-feature run (`gsm8k-tool` preset: Qwen3-1.7B, vLLM async, V1 sync + TransferQueue, ToolAgentLoop with the
## recovered Gsm8kTool, custom reward fn = packages/p4/bench/gsm8k_tool_reward.py)
| run | tree | GPUs | steps | rc | rows_with_turn_spans | val metadata_present / spans_ok | val acc@1 (steps) | critic/returns/mean | step s mean(2..N) |
|---|---|---|---|---|---|---|---|---|---|
| smoke-gsm8k-tool-p4b-on-2 (ENABLED) | 97a43c92 + bench_extras | 2 | 2 | 0 | n/a on that tree | 1.0 / 1.0 | 0.609 (2) | 0.48, 0.66 | 8.8 |
| smoke-gsm8k-tool-p4b-off-1 (CONTROL) | 97a43c92 + bench_extras | 2 | 2 | 0 | - | 0.0 / 0.0 | 0.619 (2) | -0.10, -0.02 | 13.2 |
| p4-enabled-r2-1 (ENABLED, step_reward correctness) | 64f8f604 + bench_extras (superseded if the sha moves) | 8 | 6 | 0 | 512 / 512 every step | 1.0 / 1.0 (x3) | 0.608, 0.626, 0.702 (0, 3, 6) | 0.91-1.09 | 17.14 |
| p4-control-r2-1 (CONTROL) | 64f8f604 + bench_extras | 8 | 6 | 0 | absent | 0.0 / 0.0 (x3) | 0.603, 0.630, 0.671 (0, 3, 6) | -0.04..-0.02 | 17.17 |
Reading: every stage of the enabled path is exercised on a real model and the flag-off arm is the stock path (no metadata,
outcome-only rewards). The 6-step accuracy difference (0.702 vs 0.671 at step 6, one seed, 1319 prompts) is NOT a learning
claim; the curves run on the final p4b sha. Timing: the estimator adds nothing measurable (`timing_s/adv` 0.32 vs 0.32 s).

## P3b chaos re-run (0d28f34f, "superseded": p3b moves again)
| run | outcome | steps | kill -> next step | recovery step | counters |
|---|---|---|---|---|---|
| chaos-p3b-r2-1 | completed rc=0 | 6/6 | 2.5 s | step 6 14.0 s | relaunched 1, orphaned 4+4, inflight_reissued 4 (dead-worker 4), exhausted 0 |
| chaos-p3b-r2-2 | completed rc=0 | 6/6 | 2.6 s | step 6 12.8 s | relaunched 1, orphaned 4+4, inflight_reissued 8 (dead-worker 8), exhausted 0 |
Cumulative knobs-ON survival after an agent-loop worker kill: 7/7 (main and knobs-OFF 0/6).

## P1 GPU identity (tests/workers/test_response_only_lm_head_on_gpu.py, 1x H100)
| run | tree | result | forward log_probs / entropy | sum_pi_squared | first failing gradient key |
|---|---|---|---|---|---|
| gpu-identity-p1-2 | d7d820aa (superseded by 407082ec) | 6 failed / 0 passed / 0 skipped, 235.8 s | bitwise identical (max_abs 0) in all 6 variants | max_rel 3.8e-6 | layer 0: fsdp 1.694e-02 (flat_param), fsdp2 2.201e-02 (input_layernorm), lora 1.676e-02 (down_proj.lora_B) vs 1e-2 |
| gpu-identity-p1-3 | 407082ec (FINAL) | 6 failed / 0 passed / 0 skipped, 246.9 s | identical | max_abs 1.08e-7, mean_delta 2.8e-10 | the same three values at layer 0 |
Reading: the kept-row LM head reproduces the forward quantities exactly; the gradient check fails at its first key on
both trees with identical values (deterministic), so this is a tolerance/metric design question for the P1 author, not
noise. Not patched by the bench agent (lead's instruction).

## Final shas (lead's list p0 9c99205c, p1 363470c1, p2 7f0c53f5, p3b 3f3d2c24, p4b 988a10bc), read from summary.json / metrics.jsonl
| run | tree | result |
|---|---|---|
| p4-enabled-r2-final | p4b 988a10bc | rc=0; rows_with_turn_spans 512 on 6/6 steps; critic/returns/mean 0.95-1.13; score 1.16-1.34; val metadata_present / spans_ok 1.0 x3; acc 0.599 / 0.633 / 0.694 at 0/3/6; step mean(2..6) 17.4 s |
| p4-control-r2-final | p4b 988a10bc | rc=0; metadata_present 0.0 x3; returns/mean -0.04..0; acc 0.608 / 0.627 / 0.684; step 17.8 s |
| p4-curve-enabled-s1 / control-s1 | p4b 988a10bc, 30 steps, data.seed=1 | acc every 5 steps: 0.588 0.685 0.732 0.763 0.789 0.819 0.826 vs 0.599 0.659 0.717 0.765 0.797 0.817 0.832; ENABLED val turn_count 1.44 -> 2.60, num_turns 5.2 vs 3.8; 20.6 vs 16.1 s/step. Same accuracy curve within noise; the dense reward buys more tool calls, not accuracy, on GSM8K in 30 steps (one seed; s2 running) |
| chaos-p3b-final-1 / -2 | p3b 3f3d2c24 | completed x2 (6/6), kill->next 2.6 s, recovery 6.0+10.0 s / 14.8 s, relaunched 1, orphaned 4+4 / 12+4, inflight_reissued 8 / 7, exhausted 0; no ttl_suspended key in the metrics |
| gpu-identity-p1-4 | p1 363470c1 | 2 passed (lora x2) / 4 failed (plain, gradient-checkpointing x fsdp, fsdp2); forward quantities bitwise identical; fp32-vs-bf16 flag-off log-probs mean 2.903e-02 max 4.309e-01; bf16 grad layer 0 1.694e-02 -> layer 23 2.199e-03, fp32 1.0-1.8e-05; fsdp fails the fp32 update rule (1.00-1.29e-04 > 1e-4), fsdp2 the bf16 update rule on q/k_proj biases (up to 1.39e-01 > 0.1) |
| gpu-tests-p0-r2 / p2-r2 | 9c99205c / 7f0c53f5 | 503 / 0 / 8, torchrun 2/2 (H200:8) |
| gpu-tests-p1-r2 | 407082ec (test file only differs from 363470c1) | 503 passed / 6 failed (the identity test file) / 8 skipped, torchrun 2/2 |
| gpu-tests-p3b-r2-2, gpu-tests-p4b-r2 | 3f3d2c24 / 988a10bc | running at the time of writing (bench/README.md table gets the counts) |
| p4-curve-enabled-s2 / control-s2 | p4b 988a10bc, data.seed=2 | acc every 5 steps: 0.597 0.682 0.732 0.764 0.792 0.820 0.810 vs 0.597 0.668 0.748 0.784 0.796 0.820 0.824. Pooled 2 seeds: ENABLED 0.593 0.684 0.732 0.763 0.790 0.820 0.818 vs CONTROL 0.598 0.663 0.733 0.774 0.796 0.818 0.828 -> no accuracy difference beyond +-2 points; rows_with_turn_spans 512 on all 60 ENABLED steps |
| gpu-tests-p4b-r2 | 988a10bc | 503 / 0 / 8, torchrun 2/2 (H200:8) |

# Review round 3 (2026-10-29 17:00Z ->): numbers read from bench/results/*/summary.json, metrics.jsonl and bench/analyze_runs.py

## P4 v2 arms on p4b e1052882 (first-correct reward, terminal done, v2 scorer; 30 steps, test_freq 5, 8x H100; two data seeds)
See packages/p4/bench/RESULTS.md "v2" for the per-run table. Pooled validation accuracy at steps 0/5/10/15/20/25/30:
A (outcome-only GRPO) 0.668 0.731 0.791 0.811 0.815 0.825 0.832; B (dense + reinforce_plus_plus) 0.663 0.752 0.795 0.806 0.812
0.818 0.822; C (dense + turn_rloo) 0.662 0.757 0.789 0.802 0.812 0.828 0.829. Repeated identical answers at step 30 (reviewer's
rule): A 2 / 4, B 16 / 4, C 18 / 4 of 1319 (round 2: 923 / 1085); step reward >= 2: 0 everywhere; every ENABLED C step has
512 span rows; metadata_present 1.0 on B/C validations, 0.0 on A. Reading: no arm separates from the others by more than the
seed-to-seed spread (1-2 points); the farming incentive is gone (repeats < 1.4 % vs 70-82 %); B vs C (same reward, different
baseline) differ by -0.4 .. +1.0 points pooled. Measured values only; no equivalence or superiority claim.

## P1 identity runs (test file redesigned three times in the round)
e7c0d616 `gpu-identity-p1-6` (H100:2): 8 passed, the 2 new two-rank cases crash in the loss (padded 256 vs 40, predicted by the
re-review). 1dbc0c07 `gpu-identity-p1-7`: 10 failed, all at the new line 537 (`baseline[key] - reference[key]` on a bool tensor)
before any measurement. Stock suite on 1dbc0c07 (`gpu-tests-p1-r3`) queued/running on H200 at the time of writing.

## P3b chaos (both superseded by later commits): 45690b0f and 64212367 each completed 6/6 after the kill (kill->next 2.7 s,
recovery step 13.8 / 14.4 s, relaunched 1, orphaned 4+4, inflight_reissued 4, exhausted 0; no TTL-budget key emitted).

## Stock suites: p4b e1052882 503 / 0 / 8 + torchrun 2/2 (H200:8); p2 e19e7949 and p1 1dbc0c07 running at the time of writing.

## Round-3 finals (p1 8c340e47, p2 e19e7949, p3b 1b397dfd, p4b e1052882; p0 unchanged)
| run | result |
|---|---|
| gpu-identity-p1-8 (8c340e47, H100:2) | 10 passed / 0 failed / 0 skipped, 320.5 s; max D_on/D_off 1.02 fsdp, 1.22 fsdp2, 1.11 lora, 1.01 / 1.13 two-rank; two-rank forward bitwise identical, sgd residual 0.217 ulp |
| gpu-tests-p1-r3-final (8c340e47) | console 513 / 0 / 8, junit 539 / 531 / 0 / 8, torchrun 2/2 |
| gpu-tests-p2-r3-final (e19e7949) | console 503 / 0 / 8, junit 529 / 521 / 0 / 8, torchrun 2/2 |
| gpu-tests-p4b-r3 (e1052882) | console 503 / 0 / 8, junit 529 / 521 / 0 / 8, torchrun 2/2 |
| chaos-p3b-final3-1 (1b397dfd) | completed 6/6, kill->next 2.3 s, recovery 6.2 + 11.5 s, relaunched 1, orphaned 4+4, inflight_reissued 7, exhausted 0 |
| gpu-tests-p3b-r3-final (1b397dfd) | console 503 / 0 / 8, junit 529 / 521 / 0 / 8, torchrun 2/2 |
Superseded but completed: gpu-identity-p1-6 (e7c0d616) 8/2, gpu-identity-p1-7 (1dbc0c07) 0/10, gpu-tests-p1-r3 (1dbc0c07) 502/11/8
(10 identity + 1 EADDRINUSE), chaos on 45690b0f and 64212367 completed 6/6, gpu-tests-p3b-r3 (64212367) 503/0/8.
