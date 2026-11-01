# P4: GPU unit tests on the branch (bench/README.md harness)

`gpu-tests-p4-1`: `--src <repo>/wt/p4` at `contrib/p4-turn-rewards@f9d52c4c` (9 commits
ahead of main@2ab35292, clean tree; `meta.json` `src_head` / `tree.digest_match=True`), 8x H100 80GB, exactly the
`.github/workflows/gpu_unit_tests.yml` job: `uv run --frozen --all-packages --extra sglang --extra megatron --extra ci
pytest -s <the workflow's 13 --ignore-glob filters> tests/` (without CI's `-x`, so counts are complete) plus the two
torchrun tests, with `/root/models/<org>/<name>` linked to the hf-cache snapshots the tests expect.

| run | tree | pytest (passed / failed / skipped, wall) | torchrun linear_cross_entropy_tp | torchrun megatron_kl_loss_tp | new failures vs main |
|-----|------|------------------------------------------|----------------------------------|------------------------------|----------------------|
| gpu-tests-main-2 (reference) | main@2ab35292 | 503 / 0 / 8, 24 min 59 s | rc=0 | rc=0 | - |
| gpu-tests-p4-1 | p4@f9d52c4c (9 commits, pre-split) | **503 / 0 / 8, 32 min 42 s** (rc=0) | rc=0 | rc=0 | **none** |
| gpu-tests-p4b-1 (final split tree; p4 final 912f79fb = p4b@97a43c92) | p4b@97a43c92 (contains p4a@f4651857) | **503 / 0 / 8, 26 min 37 s** (rc=0) | rc=0 | rc=0 | **none** |

P4 was split into p4a (`contrib/p4a-turn-signals`) and p4b (`contrib/p4b-turn-rloo`, stacked on p4a); the final
combined p4 (912f79fb) is the same tree as p4b@97a43c92, so `gpu-tests-p4b-1` is the final-tree result. Per-branch
files: `packages/p4a/bench/RESULTS.md`, `packages/p4b/bench/RESULTS.md`.

App <modal-app> (the launch registry (kept privately)). Artifacts: `bench/results/gpu-tests-p4-1/{pytest.log,pytest.xml,
summary.json,torchrun-*.log,command.txt}`; `summary.json["pytest"]["failed"]` lists any failing test ids.

## Review round 2: enabled-feature GPU run (2026-10-29)

The enabled end-to-end evidence (generation -> tool execution -> reward callback -> TransferQueue -> `turn_rloo` -> actor
update) is in `packages/p4/bench/ENABLED-RUN.md` "Results": smoke on p4b@97a43c92 (2x H100, ENABLED + CONTROL) and the
6-step ENABLED / CONTROL pair on p4b@64f8f604 (8x H100; "64f8f604, superseded if the sha moves"): `adv_estimator/
rows_with_turn_spans` 512/512 on every step, validation `metadata_present` / `spans_ok` 1.0 vs 0.0, `critic/returns/mean`
~1.0 vs ~0, val acc 0.608 -> 0.702 (ENABLED) vs 0.603 -> 0.671 (CONTROL) over 6 steps. Harness: `bench/README.md`
("gsm8k-tool preset"); raw runs: `bench/results/{smoke-gsm8k-tool-p4b-on-2,smoke-gsm8k-tool-p4b-off-1,p4-enabled-r2-1,p4-control-r2-1}`.

Final tree p4b@988a10bc (contains p4a@d8c4b925): `p4-enabled-r2-final` / `p4-control-r2-final` (6 steps, 8x H100) repeat every
stage row (rows_with_turn_spans 512 x6, returns/mean ~1.0 vs ~0, metadata_present 1.0 vs 0.0, acc 0.599 -> 0.694 vs 0.608 ->
0.684), and the 30-step learning curves `p4-curve-enabled-s1` / `p4-curve-control-s1` (data.seed=1, validation every 5 steps)
end at 0.826 vs 0.832 accuracy: no demonstrated accuracy gain (the per-submission reward was farmed; see the disclosure below), with the ENABLED policy making more tool calls (val turn_count
1.44 -> 2.60) at ~28 % longer steps. No accuracy advantage is claimed; tables in `packages/p4/bench/ENABLED-RUN.md`. Stock
GPU unit-test selection on 988a10bc: `gpu-tests-p4b-r2` (regression coverage; count in bench/README.md).

## Round-2 learning experiment on 988a10bc: disclosure (the AI follow-up review, 2026-10-29)

The 30-step ENABLED (`turn_level_rewards=True`, `turn_rloo`, tool `step_reward: correctness`, `done_when_correct: false`) vs
CONTROL (`grpo`, flag off) curves in `packages/p4/bench/ENABLED-RUN.md` show **no demonstrated accuracy gain**: final
validation accuracy 0.8264 vs 0.8324 (seed 1) and 0.8097 vs 0.8241 (seed 2), pooled 0.818 vs 0.828; mean step time excluding
step 1 +28.0 % (s1) and +11.7 % (s2) for ENABLED. The per-submission reward (+1 for EVERY correct `calc_gsm8k_reward` call)
was farmed: at step 30 the enabled arms repeated an identical answer in 923 (s1) and 1085 (s2) of 1319 validation episodes and
earned >= 2 step points in 901 / 1049 of them, the controls in 2-3 / 0 (`research/review-bundle/follow-up-review/evidence/p4/
analyze_runs.py`, `audit.json`). `data.seed` is the only difference between the two seeds. The comparison changed the reward
objective and the estimator at once, so it attributes nothing to `turn_rloo`; the v2 design (first-correct reward +
`done_when_correct`, arms A outcome-only GRPO / B dense + `reinforce_plus_plus` / C dense + `turn_rloo`, same outcome scorer)
is in ENABLED-RUN.md "v2 design". The round-2 runs remain valid as the end-to-end data-path proof (512 span rows per step,
metadata delivered, finite actor updates).

## v2 (review round 3): first-correct reward, terminal `done`, the estimator isolated

Round 2's curves rewarded every correct submission (`step_reward: correctness`, `done_when_correct: false`); the follow-up
review showed the enabled arms learned to resubmit the same correct answer (923 / 1085 of 1319 validation episodes at step 30,
controls 2-3), so the comparison measured reward farming, not the estimator. v2 (bench/gsm8k_tool/, owned by the bench
agent): `step_reward: first_correct` (+1.0 once per episode) and `done_when_correct: true` (`ToolResponse.done` ends the episode
at the first correct submission), one yaml for every arm; outcome scorer = `packages/p4/bench/gsm8k_tool_reward.py` v2
(correctness of the LAST submitted answer; `submissions`, `distinct_submissions`, `repeated_submission`, `scored_by_submission`,
`final_text_correct` diagnostics). Arms (30 steps, `test_freq=5`, `data.seed` 1 and 2): A outcome-only GRPO (flag off), B dense +
`reinforce_plus_plus` + `gamma=1.0`, C dense + `turn_rloo` + `gamma=1.0`; B vs C isolates the estimator, A vs B the reward.

Smokes of the new tool mode (2x H100, 2 steps, bs 16 x n 4, validation at step 2, 1319 prompts; `bench/analyze_runs.py` = the
reviewer's audit script):

| run | tree / scorer | acc | episode_done = turn_rewards_sum | spans_ok / metadata | turn_count | repeated_submission (metric) | repeated (audit) | step reward >= 2 |
|---|---|---|---|---|---|---|---|---|
| smoke-v2-C-1 / B-1 | 988a10bc, round-2 scorer | 0.246 / 0.297 (the strict `####` rule reads 0 on `done` episodes) | 0.451 / 0.412 | 1.0 / 1.0 | 1.02 | - | 0 / 0 | 0 / 0 |
| smoke-v2-C-2 / B-2 | 988a10bc, v2 scorer | 0.674 / 0.663 | 0.445 / 0.439 | 1.0 / 1.0 | 1.02 | 0.000 / 0.0015 | 0 / 2 | 0 / 0 |
| smoke-v2-C-3 | **e1052882** (round-3 tree), v2 scorer | 0.676 | 0.447 | 1.0 / 1.0 | 1.02 | 0.0008 | 1 | 0 |

`critic/rewards/max` 2.0 in every enabled smoke (outcome + the single step point land on the same last generated token); C
`adv_estimator/rows_with_turn_spans` 64/64; B has no span metric (reinforce_plus_plus takes none). The six arms
`p4-v2-{A,B,C}-s{1,2}` and `gpu-tests-p4b-r3` run on e1052882; their table follows below when they land.

### The six arms on p4b@e1052882 (30 steps, test_freq 5, 8x H100; meta.json digest_match=True, extras.json = the v2 tool/yaml/scorer; independent manifest digest matches)

| run (app) | arm | rc / wall s | val acc at steps 0 / 5 / 10 / 15 / 20 / 25 / 30 | mean step s (excl. 1) | val turn_count 0 -> 30 | val num_turns 0 -> 30 | repeated_submission (metric, step 30) | submissions / distinct (30) | episode_done = turn_rewards_sum (30) | critic/returns/mean (2..30) | rows_with_turn_spans |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p4-v2-A-s1 (<modal-app>) | A: outcome-only GRPO (flag off) | 0 / 912 | 0.6641 / 0.7248 / 0.7923 / 0.8074 / 0.8158 / 0.8234 / 0.8347 | 15.84 | - | 2.45 -> 2.96 | 0.0015 | 0.901 / 0.900 | - (flag off: not exported; episodes still end on done) | -0.050 | absent |
| p4-v2-A-s2 (<modal-app>) | A: outcome-only GRPO (flag off) | 0 / 864 | 0.6717 / 0.7369 / 0.7892 / 0.8143 / 0.8135 / 0.8256 / 0.8302 | 15.16 | - | 2.47 -> 2.95 | 0.0030 | 0.892 / 0.889 | - (flag off: not exported; episodes still end on done) | -0.046 | absent |
| p4-v2-B-s1 (<modal-app>) | B: first-correct dense + reinforce_plus_plus | 0 / 1152 | 0.6588 / 0.7544 / 0.8059 / 0.8082 / 0.8150 / 0.8165 / 0.8234 | 23.67 | 1.023 -> 1.070 | 2.47 -> 2.96 | 0.0121 | 0.901 / 0.886 | 0.8226 = 0.8226 | 1.415 | absent |
| p4-v2-B-s2 (<modal-app>) | B: first-correct dense + reinforce_plus_plus | 0 / 923 | 0.6679 / 0.7498 / 0.7847 / 0.8029 / 0.8089 / 0.8196 / 0.8211 | 16.15 | 1.019 -> 1.057 | 2.46 -> 2.93 | 0.0030 | 0.880 / 0.876 | 0.8211 = 0.8211 | 1.386 | absent |
| p4-v2-C-s1 (<modal-app>) | C: first-correct dense + turn_rloo | 0 / 879 | 0.6672 / 0.7528 / 0.7900 / 0.8112 / 0.8196 / 0.8302 / 0.8408 | 15.73 | 1.022 -> 1.084 | 2.46 -> 3.01 | 0.0136 | 0.931 / 0.914 | 0.8408 = 0.8408 | 1.416 | 512 x30 |
| p4-v2-C-s2 (<modal-app>) | C: first-correct dense + turn_rloo | 0 / 937 | 0.6573 / 0.7619 / 0.7870 / 0.7923 / 0.8044 / 0.8264 / 0.8180 | 16.52 | 1.023 -> 1.058 | 2.46 -> 2.93 | 0.0030 | 0.880 / 0.876 | 0.8180 = 0.8180 | 1.403 | 512 x30 |

Pooled over the two data seeds (mean of s1 and s2), validation accuracy on 1319 prompts:

| arm | 0 | 5 | 10 | 15 | 20 | 25 | 30 | mean step s |
|---|---|---|---|---|---|---|---|---|
| A outcome-only GRPO (flag off) | 0.6679 | 0.7309 | 0.7908 | 0.8108 | 0.8146 | 0.8245 | 0.8324 | 15.50 |
| B first-correct dense + reinforce_plus_plus | 0.6634 | 0.7521 | 0.7953 | 0.8055 | 0.8120 | 0.8180 | 0.8222 | 19.91 |
| C first-correct dense + turn_rloo | 0.6622 | 0.7574 | 0.7885 | 0.8017 | 0.8120 | 0.8283 | 0.8294 | 16.13 |
| B - A (points) | -0.5 | +2.1 | +0.5 | -0.5 | -0.3 | -0.6 | -1.0 | |
| C - A (points) | -0.6 | +2.7 | -0.2 | -0.9 | -0.3 | +0.4 | -0.3 | |
| C - B (points) | -0.1 | +0.5 | -0.7 | -0.4 | +0.0 | +1.0 | +0.7 | |

Reviewer's audit (`bench/analyze_runs.py` on `val/30.jsonl`, saved in `bench/results/p4-v2-audit.json`): repeated identical answer
A 2 / 4, B 16 / 4, C 18 / 4 of 1319 (s1 / s2; round 2: 923 / 1085), step reward >= 2: 0 in every arm, parsed tool calls
1161-1228 per validation, `span_rows` 512 on every C step and absent on A / B. B-s1's 23.7 s/step (wall 1152 s vs 864-937 s for
the other five) is node variance on an otherwise identical config: B-s2 runs at 16.2 s.

Reading (measured values only, one task, two seeds): with the farming incentive removed every arm learns at the same rate; the
pooled differences between arms stay within the seed-to-seed spread (1-2 points at a fixed step): B - A ranges -1.0 .. +2.1
points, C - A -0.9 .. +2.7, C - B -0.7 .. +1.0. The dense first-correct reward changes nothing visible in accuracy or in tool
use (val `turn_count` 1.02 -> 1.06-1.08 in B / C; A ends episodes on the same `done` rule and shows `num_turns` 2.95-2.96 vs
2.93-3.01); repeated submissions stay below 1.4 %. No accuracy gain is demonstrated for either estimator over the outcome-only
control, and `turn_rloo` is indistinguishable from `reinforce_plus_plus` on the same dense reward in this setting.
