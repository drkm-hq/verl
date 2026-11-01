# P4 enabled-feature GPU run (results below, bench agent, 2026-10-29)

Status: **DONE**: round-2 runs on 988a10bc (data-path proof; the v1 curves farmed the per-submission reward, disclosed in RESULTS.md), v2 arms A / B / C on e1052882 (RESULTS.md "v2": no demonstrated accuracy gain for either estimator, farming <= 1.4 %, `done` exercised on GPU). The 503-pass GPU unit suite is stock regression coverage (it excludes `*_on_cpu.py` and
`tests/experimental`, so none of P4's tests ran there); this run is the enabled-path evidence: generation ->
tool execution -> reward callback -> TransferQueue -> `turn_rloo` -> actor update, on a real model.

Harness: preset `gsm8k-tool` in `bench/verl_bench.py` (Qwen3-1.7B, vLLM async, V1 `sync` trainer, TransferQueue,
`multi_turn.enable=True`, hermes format, `max_assistant_turns=3`, 64 prompts x 8, one mini-batch, 6 steps,
8xH100). `--ship-extras gsm8k_tool` copies `bench/gsm8k_tool/` to `/workspace/verl/bench_extras/`:
`gsm8k_tool.py` (the demo tool verl removed in #6302, ported by the bench agent; a step reward per
`calc_gsm8k_reward` call), `gsm8k_tool_config.yaml`, `gsm8k_tool_reward.py` (identical to this directory's
copy: verl's strict GSM8K scorer plus the delivery diagnostics below). Data: `/hf-cache/data/gsm8k-tool`
from `examples/data_preprocess/gsm8k_tool_agent_loop.py` (`agent_name=tool_agent`, `tools_kwargs` carry the
ground truth to the tool).

Tree to run: the round-2 runs were on `contrib/p4b-turn-rloo` @ 988a10bc (contains p4a @ d8c4b925); the v2 arms (below) run on the round-3 heads p4b @ e1052882 (contains p4a @ 0d4cd8ca; both on main @ 2ab35292).

## Arms (hydra overrides)

Common, from the preset: `trainer.use_v1=True trainer.v1.trainer_mode=sync transfer_queue.enable=True
data.prompt_key=prompt data.return_raw_chat=True actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.calculate_log_probs=True
actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.multi_turn.tool_config_path=/workspace/verl/bench_extras/gsm8k_tool_config.yaml
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
reward.custom_reward_function.path=/workspace/verl/bench_extras/gsm8k_tool_reward.py
reward.custom_reward_function.name=compute_score` (reward manager: the default `naive`; `dapo` / `gdpo` deliver
the same metadata since p4a 7e4e8e4f).

- **ENABLED** (dense turn rewards + the estimator), add:
  `actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=turn_rloo algorithm.gamma=1.0`
- **CONTROL** (outcome reward, flag off), add: `algorithm.adv_estimator=grpo`
- optional third arm (dense rewards, stock estimator): `turn_level_rewards=True algorithm.adv_estimator=reinforce_plus_plus`

Knobs that matter with Qwen3 + vLLM + the gsm8k tool:

- `multi_turn.format=hermes`: Qwen3 emits `<tool_call>{json}</tool_call>` and EOS after it; `HermesToolParser`
  needs no extra stop tokens. `data.return_raw_chat=True` and `rollout.mode=async` are required by the agent loop.
- `max_assistant_turns=3` = two tool calls plus the final answer turn (the last allowed turn never executes its
  tools; p4a docs). Expect `turn_count` in {1, 2, 3} and `training/num_turns/mean` between 2 and 6.
- Tool config: the bench port's `step_reward: verl` is the removed demo's shape (0.0 on an improving submission,
  -0.05 otherwise) and gives near-zero `turn_rewards`; set `step_reward: correctness` (1.0 / 0.0 per submission) in
  the ENABLED arm so the dense row carries a visible per-turn signal. Keep `done_when_correct: false`: with
  `done` the final `#### <answer>` text is never generated and the strict outcome scorer reads 0.
- Qwen3 thinks before every turn; if `response_length/clip_ratio` is high (episodes cut by the 1024-token
  budget before the tool call), add `+data.apply_chat_template_kwargs.enable_thinking=False` (both arms).
- Validation metrics need at least one validation pass: `trainer.val_before_train=True` (default) or a
  `trainer.test_freq` within the 6 steps; `val-aux/*` carries the reward function's diagnostic keys.

## What proves each stage

Read `bench/results/<run>/metrics.jsonl` (train.log for the same keys). `N` = the validation `n`.

| stage | metric | ENABLED expects | CONTROL expects |
|---|---|---|---|
| multi-turn generation | `training/num_turns/mean` (V1), `val-aux/num_turns/mean` | > 1 | > 1 |
| tool executed, step rewards summed per turn | `val-aux/openai/gsm8k/turn_rewards_sum/mean@N`, `turn_count/mean@N` | non-zero, >= 1 | 0.0 (no metadata) |
| reward callback received the metadata | `val-aux/openai/gsm8k/metadata_present/mean@N`, `spans_ok/mean@N` | 1.0, 1.0 | 0.0, 0.0 |
| `episode_done` delivered | `val-aux/openai/gsm8k/episode_done/mean@N` | 0.0 (`done_when_correct: false`) | 0.0 |
| dense `rm_scores` row | `critic/score/mean` = outcome + step rewards | above the outcome rate | = outcome rate |
| TransferQueue -> span lift -> `turn_rloo` | `adv_estimator/rows_with_turn_spans` (p4b, V1 `_compute_advantage`) | 512 (= 64 x 8) each step | absent |
| `turn_rloo` ran (raw returns kept) | `critic/returns/mean` | > 0 | ~0 (grpo returns = advantages) |
| advantages reached the actor | `critic/advantages/max` / `min` non-zero, `actor/pg_loss`, `actor/grad_norm` finite | 6/6 steps | 6/6 steps |
| outcome scorer still works | `val-core/openai/gsm8k/acc/mean@N` | reported | reported |

What this run does not claim: a learning-curve comparison (6 steps, one seed), the GPU cost of the estimator's
`bs x turns` index build (compare `timing_s/adv` between arms, exploratory), or Continuous Token authors other
than Qwen's.

## Results

### Smoke on the reviewed head (p4b@97a43c92 + bench_extras, 2x H100, 2 steps, bs 16 x n 4, mini 16; tool `step_reward: verl`)

`bench/results/smoke-gsm8k-tool-p4b-on-2` (ENABLED, <modal-app>) and `smoke-gsm8k-tool-p4b-off-1` (CONTROL,
<modal-app>), both rc=0, validation after step 2 on the 1319-row test set (`trainer.test_freq=2`); the
first ENABLED attempt `smoke-gsm8k-tool-p4b-on-1` (<modal-app>, bench placeholder reward, no validation)
also completed. Full tables: bench/README.md "Review round 2: P4 enabled smoke".

| stage metric | ENABLED | CONTROL |
|---|---|---|
| `num_turns/mean` (max) step 1, 2 | 2.94 (4), 3.25 (6) | 2.72 (6), 3.13 (6) |
| `val-aux/.../turn_rewards_sum/mean@1`, `turn_count/mean@1` | -0.0010, 1.453 | 0.0, 0.0 |
| `val-aux/.../metadata_present/mean@1`, `spans_ok/mean@1` | 1.0, 1.0 | 0.0, 0.0 |
| `val-aux/.../episode_done/mean@1` | 0.0 | 0.0 |
| `critic/score/mean` step 1, 2 / `critic/rewards/min` | 0.531, 0.733 / 0.0, **-0.10** | 0.469, 0.719 / 0.0, 0.0 |
| `critic/returns/mean` step 1, 2 | 0.477, 0.659 | -0.099, -0.019 |
| `critic/advantages/min` / `max`, `actor/pg_loss`, `actor/grad_norm` | -1.05 / 1.05, 0.037 / 0.014, 0.910 / 0.810 | -1.5 / 1.5, 0.100 / 0.019, 1.463 / 1.272 |
| `val-core/openai/gsm8k/acc/mean@1` | 0.6088 | 0.6187 |
| `adv_estimator/rows_with_turn_spans` | not on 97a43c92 (added later) | - |

### 6-step arms on p4b@64f8f604 ("64f8f604, superseded if the sha moves"; 8x H100 80GB, 64 x 8, mini 64, `step_reward: correctness`, validation at steps 0/3/6)

`p4-enabled-r2-1` (<modal-app>, rc=0, wall 489.9 s) and `p4-control-r2-1` (<modal-app>, rc=0,
wall 496.8 s); `meta.json`: `src_head` 64f8f604, `digest_match=True`, `extras.json` lists the shipped tool/config/reward.
Validation = the 1319-row test set, greedy, n=1 (`val-core/openai/gsm8k/acc/mean@1`).

| stage metric | ENABLED (turn_level_rewards + turn_rloo) | CONTROL (grpo, flag off) |
|---|---|---|
| `num_turns/mean` steps 1..6 (max) | 3.00, 3.04, 3.15, 3.05, 3.20, 3.43 (6) | 2.96, 3.04, 3.02, 3.12, 3.07, 3.18 (6) |
| `val-aux/.../turn_rewards_sum/mean@1` at steps 0 / 3 / 6 | 0.452 / 0.524 / 0.703 | 0.0 / 0.0 / 0.0 |
| `val-aux/.../turn_count/mean@1` at 0 / 3 / 6 | 1.469 / 1.549 / 1.732 | 0.0 / 0.0 / 0.0 |
| `val-aux/.../metadata_present`, `spans_ok` (all three validations) | 1.0, 1.0 | 0.0, 0.0 |
| `val-aux/.../episode_done` | 0.0 | 0.0 |
| `critic/score/mean` steps 1..6 (dense row = outcome + step rewards) | 1.145, 1.188, 1.180, 1.107, 1.291, 1.338 (`critic/rewards/max` 3.0) | 0.639, 0.660, 0.668, 0.668, 0.695, 0.637 (= outcome rate) |
| `adv_estimator/rows_with_turn_spans` | **512 on every step** | absent |
| `critic/returns/mean` steps 1..6 | 0.981, 1.020, 0.999, 0.906, 1.093, 1.077 (> 0) | -0.043 .. -0.020 (= advantages) |
| `critic/advantages/min` / `max` | -2.00..-2.14 / 1.86..2.71 | -2.475 / 2.475 |
| `actor/pg_loss`, `actor/grad_norm` (6/6 finite) | 0.033..0.057, 0.43..0.62 | 0.020..0.043, 0.50..0.66 |
| `val-core/openai/gsm8k/acc/mean@1` at steps 0 / 3 / 6 | 0.6080 / 0.6262 / **0.7020** | 0.6027 / 0.6300 / 0.6710 |
| `timing_s/adv` mean(2..6) (exploratory) | 0.324 s | 0.316 s |
| `timing_s/step` mean(2..6), `gen`, `update_actor` | 17.14 s, 7.17, 4.27 | 17.17 s, 7.55, 3.85 |

Every row of the stage table above is satisfied. The 30-step curves and `gpu-tests-p4b-r2` run on the final p4b sha
only (lead, 14:30Z); the 6-step pair is repeated there as `p4-enabled-r2-final` / `p4-control-r2-final`.

### FINAL tree p4b@988a10bc (contains p4a@d8c4b925): 6-step arms, 8x H100, same config as above

`p4-enabled-r2-final` (<modal-app>, rc=0, wall 482.0 s) and `p4-control-r2-final` (<modal-app>,
rc=0, wall 558.0 s); `meta.json`: `src_head` 988a10bc, `digest_match=True`.

| stage metric | ENABLED | CONTROL |
|---|---|---|
| `num_turns/mean` steps 1..6 (max 6) | 3.00, 3.08, 3.09, 3.09, 3.31, 3.47 | 3.00, 2.97, 3.02, 2.97, 3.06, 3.20 |
| val `turn_rewards_sum` / `turn_count` at 0 / 3 / 6 | 0.433 / 0.525 / 0.697; 1.450 / 1.549 / 1.732 | 0 / 0 / 0; 0 / 0 / 0 |
| val `metadata_present`, `spans_ok` (x3) / `episode_done` | 1.0, 1.0 / 0.0 | 0.0, 0.0 / 0.0 |
| `critic/score/mean` 1..6 (`critic/rewards/max`) | 1.160, 1.227, 1.191, 1.160, 1.332, 1.340 (3.0) | 0.652, 0.688, 0.617, 0.648, 0.686, 0.625 (1.0) |
| `adv_estimator/rows_with_turn_spans` | **512 on every step** | absent |
| `critic/returns/mean` 1..6 | 1.006, 1.054, 1.009, 0.954, 1.127, 1.072 | -0.035, -0.004, -0.013, -0.007, -0.038, -0.031 |
| `critic/advantages` min / max | -2.14 / 2.29 | -2.475 / 2.475 |
| `actor/pg_loss`, `actor/grad_norm` (6/6 finite) | 0.024..0.048, 0.46..0.65 | 0.004..0.038, 0.51..0.62 |
| `val-core/openai/gsm8k/acc/mean@1` at 0 / 3 / 6 | 0.5989 / 0.6331 / 0.6937 | 0.6080 / 0.6270 / 0.6839 |
| `response_length/clip_ratio` 1..6 | 0.17, 0.15, 0.16, 0.17, 0.18, 0.22 | 0.16, 0.13, 0.18, 0.17, 0.17, 0.22 |
| `timing_s/adv` mean(2..6), `timing_s/step` mean(2..6) | 0.23 s, 17.4 s | 0.37 s, 17.8 s |

Every stage row holds on the final tree.

### Learning-curve A/B on 988a10bc: 30 steps, `trainer.test_freq=5`, `data.seed=1`, 8x H100 (practical-value evidence, not an identity proof)

`p4-curve-enabled-s1` (<modal-app>, rc=0, wall 1241.5 s, step mean(2..30) 20.6 s) vs `p4-curve-control-s1`
(<modal-app>, rc=0, wall 924.8 s, 16.1 s). Validation = greedy, n=1, 1319 prompts.

| step | ENABLED acc | CONTROL acc | ENABLED val turn_count / turn_rewards_sum / num_turns | CONTROL val num_turns | ENABLED train score | CONTROL train score |
|---|---|---|---|---|---|---|
| 0 | 0.5883 | 0.5989 | 1.435 / 0.420 / 2.87 | 2.91 | - | - |
| 5 | 0.6854 | 0.6588 | 1.702 / 0.672 / 3.40 | 3.11 | 1.44 (step 6) | 0.67 (step 6) |
| 10 | 0.7316 | 0.7172 | 1.848 / 0.812 / 3.70 | 3.38 | 1.76 (11) | 0.80 (11) |
| 15 | 0.7627 | 0.7650 | 1.869 / 0.818 / 3.74 | 3.56 | 1.65 (16) | 0.81 (16) |
| 20 | 0.7885 | 0.7968 | 1.876 / 0.816 / 3.75 | 3.64 | 1.84 (21) | 0.92 (21) |
| 25 | 0.8188 | 0.8173 | 1.923 / 0.857 / 3.85 | 3.75 | 1.75 (26) | 0.83 (26) |
| 30 | **0.8264** | **0.8324** | **2.602 / 1.519 / 5.20** | 3.79 | 2.02 (30) | 0.81 (30) |

Reading (one seed): the two arms reach the same validation accuracy (0.826 vs 0.832 at step 30, inside the 1319-prompt
noise; ENABLED is ahead at steps 5 and 10 by 2.7 / 1.4 points, behind by < 1 point from step 15 on). What the dense per-call
correctness reward changes is behaviour: the ENABLED policy makes more tool calls (validation `turn_count` 1.44 -> 2.60
generated turns, `turn_rewards_sum` 0.42 -> 1.52, `num_turns` 5.2 at step 30 vs 3.8 for CONTROL), which is exactly what a
+1.0 per correct submission rewards, and costs ~28 % longer steps (20.6 vs 16.1 s). No accuracy advantage is claimed from this
experiment. A second seed (`data.seed=2`: p4-curve-enabled-s2 / p4-curve-control-s2) was launched afterwards.

### Second seed (`data.seed=2`): p4-curve-enabled-s2 (<modal-app>, rc=0, wall 977.4 s, 17.6 s/step) vs p4-curve-control-s2 (<modal-app>, rc=0, wall 891.1 s, 15.7 s/step)

| step | ENABLED s2 acc | CONTROL s2 acc | ENABLED s2 val turn_count / turn_rewards_sum / num_turns | CONTROL s2 val num_turns |
|---|---|---|---|---|
| 0 | 0.5974 | 0.5974 | 1.453 / 0.438 / 2.91 | 2.93 |
| 5 | 0.6823 | 0.6679 | 1.697 / 0.666 / 3.40 | 3.17 |
| 10 | 0.7316 | 0.7483 | 1.842 / 0.797 / 3.68 | 3.52 |
| 15 | 0.7635 | 0.7839 | 1.853 / 0.811 / 3.71 | 3.71 |
| 20 | 0.7923 | 0.7961 | 1.864 / 0.817 / 3.73 | 3.75 |
| 25 | 0.8203 | 0.8196 | 1.997 / 0.937 / 3.99 | 3.79 |
| 30 | 0.8097 | 0.8241 | **2.703 / 1.616 / 5.41** | 3.80 |

`adv_estimator/rows_with_turn_spans` = 512 on all 30 steps of both ENABLED curves; `metadata_present` 1.0 on every ENABLED
validation and 0.0 on every CONTROL one.

**Pooled over the two seeds** (mean of s1 and s2, validation accuracy on 1319 prompts):

| step | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| ENABLED (turn_rloo + dense correctness per call) | 0.593 | 0.684 | 0.732 | 0.763 | 0.790 | 0.820 | 0.818 |
| CONTROL (grpo, outcome only) | 0.598 | 0.663 | 0.733 | 0.774 | 0.796 | 0.818 | 0.828 |
| difference (ENABLED - CONTROL), points | -0.5 | +2.0 | -0.1 | -1.1 | -0.6 | +0.1 | -1.0 |

Reading: the two arms learn GSM8K at the same rate (every pooled difference is within +-2 points, the seed-to-seed spread at a
fixed step is 1-3 points); the only consistent difference is behavioural: both ENABLED runs end with 2.6-2.7 generated turns
per validation trajectory (1.44-1.45 at step 0; CONTROL's `num_turns` grows 2.9 -> 3.8, i.e. ~1.9 generated turns), and the
ENABLED s2 accuracy dips 0.820 -> 0.810 between steps 25 and 30 exactly when its tool-call count jumps 2.0 -> 2.7. The dense
per-submission reward is a stronger incentive to call the tool again than to answer; it buys no accuracy on this task in 30
steps, and costs 10-28 % longer steps (longer trajectories). This is the practical-value evidence the review asked for; it is
not an identity proof and not a claim that `turn_rloo` is better or worse in general.

## v2 design (round 3, after the AI follow-up review): isolate the estimator

Why v2: the round-2 curves used `step_reward: correctness` (+1 for EVERY correct submission) with `done_when_correct: false`,
so the enabled arms learned to resubmit the same correct answer (at step 30: 923 / 1085 of 1319 validation episodes repeated an
identical answer, 901 / 1049 earned >= 2 step points; controls 2-3 / 0; pooled acc 0.818 vs 0.828; steps 11.7-28.0 % slower).
That comparison changed the reward objective and the estimator at once, so it could not attribute anything to `turn_rloo`.
No demonstrated accuracy gain; `data.seed` was the only difference between the two seeds.

Reward semantics, identical for every arm (the bench agent owns `bench/gsm8k_tool/gsm8k_tool.py` and the preset):

- tool config `step_reward: first_correct` (a correct submission earns +1.0 once per episode; later submissions earn 0, a
  non-improving one may keep the demo's -0.05), `done_when_correct: true` (`ToolResponse.done` ends the episode at the first
  correct submission; this also exercises `done` on GPU: expect `val-aux/.../episode_done/mean@1` > 0 in all arms).
- outcome scorer `packages/p4/bench/gsm8k_tool_reward.py` v2 (copy it to `bench/gsm8k_tool/`): the correctness of the LAST
  submitted answer parsed from the trajectory text (the last `<tool_call>` JSON `arguments.answer`), judged with the tool's
  own rule; an episode ended by `done` is scored by that same last submission; a trajectory with no submission falls back
  to the strict `#### <answer>` text rule (`reward_kwargs.fallback_to_final_text=false` to score it 0 instead; keep one
  setting for all arms). It also reports `submissions`, `distinct_submissions`, `repeated_submission` (the reviewer's
  `analyze_runs.py` rule: an episode repeats when its parsed answers exceed its distinct answers), `scored_by_submission`,
  `final_text_correct` and the p4a metadata keys.

Arms (preset `gsm8k-tool`; 30 steps, `trainer.test_freq=5`, two data seeds `data.seed=1` and `=2`; everything else equal):

| arm | what it isolates | overrides to add |
|---|---|---|
| A, outcome-only GRPO (control) | the dense reward (A vs B) | `actor_rollout_ref.rollout.multi_turn.turn_level_rewards=False algorithm.adv_estimator=grpo` |
| B, dense + stock estimator | the estimator (B vs C) | `actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=reinforce_plus_plus algorithm.gamma=1.0` |
| C, dense + turn_rloo | | `actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True algorithm.adv_estimator=turn_rloo algorithm.gamma=1.0` |

Why `reinforce_plus_plus` for B rather than `gae`: it consumes the dense row token-wise (the running return crosses the
observation tokens) without a value head, so B differs from C only in the baseline, whereas GAE would add a critic to train
(a second confound and extra GPU time per step). All three arms run the same tool config, so episode dynamics (termination
on the first correct submission) are identical; A ignores the step rewards (flag off), B and C train on the same dense row.

What proves each stage (per arm, from `metrics.jsonl`; validation keys are `val-aux/openai/gsm8k/<key>/mean@1`):

| stage | metric | A | B | C |
|---|---|---|---|---|
| multi-turn generation | `training/num_turns/mean`, val `num_turns` | > 1 | > 1 | > 1 |
| `done` on GPU | val `episode_done` | > 0 | > 0 | > 0 |
| metadata delivered | val `metadata_present`, `spans_ok` | 0.0 | 1.0, 1.0 | 1.0, 1.0 |
| dense row (first-correct) | `critic/score/mean` vs val `acc`; val `turn_rewards_sum` in [-0.05 k, 1.0] | = acc | > acc | > acc |
| no farming | val `repeated_submission`, `submissions` | ~0 | ~0 | ~0 |
| span lift | `adv_estimator/rows_with_turn_spans` | absent | absent (`reinforce_plus_plus` takes no spans) | 512 |
| estimator ran | `critic/returns/mean` | ~0 | REINFORCE++ returns | > 0 (raw return-to-go) |
| actor update | `actor/pg_loss`, `actor/grad_norm` finite | 30/30 | 30/30 | 30/30 |

Report per arm and seed: val `acc` at steps 0 / 5 / ... / 30 (pooled over the two seeds), the repeated-submission counts
from the reviewer's rule (`research/review-bundle/follow-up-review/evidence/p4/analyze_runs.py` on `val/30.jsonl`, and the
per-arm `repeated_submission` metric), `submissions` / `distinct_submissions`, `scored_by_submission`, `episode_done`,
`num_turns`, mean step time excluding step 1. State the measured values only; no equivalence or superiority claim.
