# Follow-up independent review of the updated verl contributions

Reviewed October 29, 2026. Source: `<repo>`. This review compares the updated branches with the heads in the first independent review, checks the earlier failures, examines new implementation and tests, and audits the recorded GPU evidence. No contribution source was changed and no GPU job was launched.

The updates resolve the original reproduced failures. The evidence is substantially stronger, especially the actual P1 engine comparisons and P4 enabled training runs. I still recommend four scoped changes before submission: reject W4A4 QAT with P2's skip, fix P3's zero-retry refill loop, make P1's numerical comparison reject non-finite results, and preserve or explicitly disclose P4's changed callback contract when disabled. P1 also needs an evidence-description correction. P4's experiment needs a better reward incentive before it can support an estimator-benefit claim.

## Reviewed versions and earlier findings

| Contribution | Current head | Assessment of previous concerns |
|---|---|---|
| P0 loss horizon | `9c99205c` | Finite-positive/type/int64 validation addressed; no new blocker found. |
| P1 LM-head rows | `cf539906` | PEFT probe, synchronization wording and K/T accounting fixed; actual feature-enabled GPU comparison added. |
| P2 old-log-prob skip | `7f0c53f5` | Effective overrides, loaded adapter dropout, jitter and standard module checks addressed; legacy trainer now rejects the option. One built-in mode remains unsafe. |
| P3a TTL recovery | `6445f21b` | Per-step budget now persists across sampling calls; exhausted/draining handling improved. Cap-zero refill interaction remains. |
| P3b worker relaunch | `3f3d2c24` | Empty validation fetch and discarded recovery counters fixed. Original empty-validation reproduction now passes. |
| P4a turn metadata | `d8c4b925` | Metadata reaches all five managers; strict span validation, real tokenizer tests and disabled span-work gating added. |
| P4b turn estimator | `988a10bc` | Mathematical qualifications and enabled GPU evidence added; no new defect confirmed in the intended enabled estimator path. |

All seven packaged patches match Git diffs against their stated bases. All seven source worktrees were clean. Full hashes, base hashes and patch hashes are in [reviewed-heads.tsv](reviewed-heads.tsv).

## R2-01 — P2: built-in W4A4 activation observers bypass both guards

**Medium priority; correctness of an accepted opt-in combination.**

Location: [engine_workers.py:793](<repo>/wt/p2/verl/workers/engine_workers.py:793). The mode-dependent operation is [qat/linear.py:356](<repo>/wt/p2/verl/utils/qat/linear.py:356).

With `actor.skip_old_log_prob_when_on_policy=true` and `actor.fsdp_config.qat.enable=true, mode=w4a4`, both config validation and the actual worker module check accept the actor. Verl's own `QATLinear` updates its activation quantization scale only in training mode. Identical weights and inputs can therefore use different quantizer parameters during the old eval pass and replacement train pass. This is a documented experimental built-in configuration, rather than hypothetical custom `forward` behavior.

The included CPU probe loads the unchanged QAT class definitions, substitutes a spy only for the CUDA quantizer, and exercises the real config and worker guards. Both accept it; the module walk finds no offenders. For identical input, the eval scale is **2688**, the train scale **1344**, and quantizer `global_amax` changes from **1 to 2**. This establishes different observer state and quantizer inputs; it is **not** an end-to-end GPU log-probability measurement.

**Suggested change:** explicitly reject enabled W4A4 QAT with this skip until observers are frozen consistently and equivalence is demonstrated. Add coverage at configuration and constructed-worker validation. There is no evidence here for rejecting W4A16, whose weight-only path lacks this activation observer branch. An explicit known-incompatible-mode guard is clearer than continually expanding a heuristic module-name scanner.

Evidence: [probe](evidence/p2/reproduce_qat_skip.py), [output](evidence/p2/qat-reproduction.log).

## R2-02 — P3a/P3b: zero retries still permits an endless refill loop

**Medium priority; hang in a supported non-default configuration.**

Locations: [P3a replay_buffer.py:475](<repo>/wt/p3a/verl/trainer/ppo/v1/replay_buffer.py:475), [P3b replay_buffer.py:514](<repo>/wt/p3b/verl/trainer/ppo/v1/replay_buffer.py:514).

With `max_inflight_reissues=0`, every first attempt is exhausted as soon as its TTL expires. Exhausted prompts bypass the per-step reissue budget and become failures. Async sampling, or sync sampling with `sync_refill_failed_groups=True`, then dispatches replacements. Each replacement starts at attempt 1 and repeats the cycle. No reissue is charged, so the budget never suspends this behavior.

The independent test runs the real sampling loops with an in-memory queue and simulated clock. Every healthy generation completes after two seconds; TTL is 0.5 seconds. On **both final heads**, cap-zero sync and async cases fail to return a batch after 20 simulated seconds: **34 dispatches, 31 completed generations**. Cap-three controls complete. This contradicts the PR body's unrestricted “never livelocks” claim.

**Suggested change:** bound TTL-triggered retirement/replacement as well as redispatch, or reject the zero-cap/refill combination. At the limit, retain healthy pending work or stop with an explicit diagnostic. Keep this full-loop regression test for both modes. Separately, the documented choice to wait indefinitely after a positive budget is spent could become an explicit configurable wait/abort policy; that is a design suggestion, not another reproduced defect.

Evidence: [test](evidence/p3/test_zero_retry_slow_refill_on_cpu.py), [P3a output](evidence/p3/final-p3a-and-zero-retry.log), [P3b output](evidence/p3/zero-retry-p3b.log), [detailed review](evidence/p3/report.md).

## R2-03 — P1: the GPU comparison can pass NaN results

**Medium priority; defect in the new correctness test, not a demonstrated model failure.**

Location: [test_response_only_lm_head_on_gpu.py:323](<repo>/wt/p1/tests/workers/test_response_only_lm_head_on_gpu.py:323).

The comparison uses `delta > tolerance` and similar scalar inequalities without first requiring finite values. Comparisons with NaN are false. Injecting NaNs into the enabled arm's log-probabilities, entropy, loss, gradients or SGD residual yields no failures, in both bf16 and fp32. Repeating the flag-off baseline does not catch a defect introduced only by flag-on.

Eleven independent failure-injection expectations fail against the actual imported helper. One models the engine's behavior when non-finite gradients cause an optimizer step to be skipped: the enabled arm has NaN gradients, unchanged parameters and zero updates, yet passes. This does **not** establish that the recorded GPU run contained NaNs; it establishes that the test can miss them.

**Suggested change:** require finite tensors in every arm and reference, including the fp32 comparison, losses, gradient norm and residuals, before tolerance checks. Retain small failure-injection tests for the comparison helper. Assert that the intended optimizer update actually occurred when gradients should be nonzero.

Evidence: [tests](evidence/p1/test_gpu_oracle_on_cpu.py), [output](evidence/p1/oracle-tests.log), [detailed review](evidence/p1/report.md).

## R2-04 — P4a/P4b: disabled mode changes custom callback inputs

**Medium priority, narrowly scoped backward compatibility; alternatively an explicitly accepted contract change.**

Location: [reward_manager/base.py:89](<repo>/wt/p4b/verl/experimental/reward_loop/reward_manager/base.py:89); DAPO calls this helper at [dapo.py:62](<repo>/wt/p4b/verl/experimental/reward_loop/reward_manager/dapo.py:62).

The shared helper unconditionally merges loop metadata and overwrites `num_turns`/`rollout_reward_scores`. DAPO/GDPO previously forwarded dataset `extra_info` without this merge; rate-limited also gains standardized fields. Existing custom reward callbacks can therefore change their output with `turn_level_rewards=False`.

The real DAPO reproduction supplies dataset `num_turns=7`, loop `__num_turns__=3`, and a callback returning that field. With the new feature off, the reward is **7 on main and 3 on current P4b**. Dataset `tool_rewards` is also replaced. The new enabled metadata contract is useful, but “flag off, tensors and metadata are bitwise what main produces” is too broad.

**Suggested change:** preserve the prior disabled callback contract, or separate and document the unconditional standardization as a deliberate behavior change, including collision precedence. Merely gating span bookkeeping does not establish unchanged reward behavior.

Evidence: [probe](evidence/p4/flag_off_reward_repro.py), [main](evidence/p4/flag-off-main.log), [current](evidence/p4/flag-off-p4b.log).

## P1 evidence and coverage suggestions

Correct the gradient/update distinction in the test docstring, PR body and round-two verdict. The raw final log reports worst actual bf16 gradient relative Frobenius difference **0.03851**; the quoted **0.1390** is the difference between *subtracted parameter updates*, which is affected by cancellation. The earlier run's failures name `update/...`, not `grad/...`. That supports replacing cancellation-sensitive update checks, but does not justify removing the direct gradient bound.

Keep a justified direct bf16 on/off gradient bound, with absolute floors for near-zero tensors, alongside the new fp32-distance check. Its factor-two rule is a relative-error guard rather than identity: the included finite example `truth=1, off=1.1, on=0.8` passes despite a 27.27% on/off difference. This characterizes the assertion, not an observed model error. Describe the actual measured agreement separately from the allowed acceptance region.

The recorded six-case GPU comparison really ran, but scope it precisely: one rank (FSDP1 explicitly switches to `NO_SHARD`), fresh zero-B LoRA with zero A gradients in the first backward, separate checkpointing and LoRA variants, and SGD. A nonzero/trained adapter, LoRA with checkpointing, and an actual multi-rank sharded comparison would strengthen it. Existing distributed performance runs remain useful smoke evidence. Bitwise forward equality on one H100/shape should be treated as an observation, not a portable requirement for every GPU/GEMM implementation.

Raw distinction: [final log:270](<repo>/bench/results/gpu-identity-p1-5/pytest.log:270) and [log:272](<repo>/bench/results/gpu-identity-p1-5/pytest.log:272). Additional references are in the detailed P1 report.

## P4 benchmark: fix the repeated-reward incentive

The enabled integration gap is closed: all six final/control and curve runs identify `988a10bc` and the same shipped-source digest, independently recomputed from current source plus declared benchmark extras. Raw metrics show 512 span-bearing rows each enabled training step and successful metadata delivery in validation.

However, `step_reward=correctness` awards +1 for **every** correct submission while `done_when_correct=False`. At step 30, parsing recorded tool-call JSON reveals:

| Run | Validation episodes | Episodes repeating an identical answer | Episodes with at least 2 step-reward points |
|---|---:|---:|---:|
| Enabled, data seed 1 | 1319 | 923 | 901 |
| Control, data seed 1 | 1319 | 2 | 0 |
| Enabled, data seed 2 | 1319 | 1085 | 1049 |
| Control, data seed 2 | 1319 | 3 | 0 |

This is repeated-reward farming encouraged by the benchmark. Pooled final accuracy is approximately **0.818 enabled versus 0.828 control**, with mean steps excluding step 1 **11.7–28.0% slower**. These observations do not establish statistical inferiority/equivalence or a general verdict on `turn_rloo`; they do warrant explicitly reporting the repeated submissions. Only `data.seed` differs between the two named seeds.

Award first-correct/delta-improvement reward, or terminate on correctness and adapt the outcome scorer to recognize terminal tool success. Then compare **dense reward + stock estimator** against **the same dense reward + turn_rloo**. The current outcome-only GRPO comparison changes both reward objective and estimator, so it cannot isolate estimator value. Replace “same curve within noise” with the measured values and “no demonstrated accuracy gain.” This is an experimental-design suggestion, not a defect in the estimator implementation.

Evidence: [analysis script](evidence/p4/analyze_runs.py), [audited counts](evidence/p4/audit.json), [detailed P4 report](evidence/p4/report.md).

## Validation and remaining limits

These are independently rerun local selections, not whole-suite totals:

| Selection | Result |
|---|---|
| P0 product and round-two reviewer tests | 104 passed, 2 expected failures |
| P1 product and PEFT reviewer tests | 33 passed |
| P2 new guard/reviewer selection | 25 passed after rerunning one test with local socket access |
| P2 existing skip/engine/entropy regression selection | 59 passed |
| P3a product/reviewer selection | 49 passed |
| P3b product/reviewer selection | 81 passed |
| Original P3b empty-validation reproduction, adapted to bind new real helpers | 1 passed |
| P4 product and reviewer selection, including real tokenizer | 165 passed |

The new diagnostic tests are separate: P1 has 11 failing expectations exposing NaN acceptance and one passing finite-rule characterization; P3 has two failing cap-zero expectations plus two passing cap-three controls **on each branch**. QAT and disabled-callback probes produce the differences described above. These failures are review findings, not failures of the shipped regression selections.

GPU records were inspected, not rerun. P1's six feature comparisons passed; its stock console tally is 509 passed/8 skipped (JUnit counts subtests differently). P3's final two chaos runs complete all six steps and cover worker-death recovery, but do not trigger TTL suspension or validation recovery. P4 proves the enabled Qwen/V1/naive-manager path; the GPU runs do not enable terminal `done_when_correct` behavior. Broader GPU/tokenizer/manager combinations remain unproven.

Small documentation cleanup: P3b's benchmark introduction still says the final changes have not run on Modal despite the final-run section. The final P3 stock-suite metadata says **8×H200**, while the PR bodies say H100; this does not change the separate two-H100 chaos-run description. Keep “time until next completed step” distinct from actual recovery latency. See the detailed P3 report for raw file references.

## Recommended next edits

1. Add the explicit P2 W4A4 incompatibility guard and tests.
2. Fix or reject P3 cap-zero refill behavior, using the provided full-loop regression.
3. Add P1 finiteness checks; correct gradient/update claims and retain a defensible direct gradient bound.
4. Decide and document P4 disabled callback compatibility; test metadata collisions.
5. Revise the P4 reward experiment before making any estimator-value claim.
6. Refresh stale evidence summaries. Keep the good separation now present between regression tests, enabled integration evidence, and performance measurements.

P0 has no new requested fix from this review. No production-code issue was confirmed in P1's new PEFT probe or P4's intended enabled estimator path. Those conclusions are bounded by the inspection and tests above, not a guarantee of correctness.
