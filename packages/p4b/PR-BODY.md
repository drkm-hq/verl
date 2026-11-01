### What does this PR do?

**[2/2]** of turn-level credit for multi-turn tool agent loops, stacked on **[1/2]** `[tool, rollout, cfg] feat:
ToolResponse.done, generated turn spans and a dense per-turn reward row` (branch `contrib/p4a-turn-signals`), which
produces the dense per-turn `rm_scores` row and `extra_fields["generated_turn_spans"]`.

Adds **`algorithm.adv_estimator: turn_rloo`** (`verl/trainer/ppo/core_algos.py`), RFC #7244's leave-one-out
return-to-go baseline per turn index. For sample `i` of a group sharing a prompt, every generated token of turn `t`
gets `G[i,t] - mean_{j != i, m_j > t} G[j,t]`: `G[i,t]` is the (per-turn `gamma`-discounted) return-to-go from turn
`t` and the mean runs over the other members that reached turn `t` (0 when none did). Tool-observation tokens get
0. Rows without spans are one turn, so on single-turn data it gives the same policy advantages as `rloo`; the
`returns` tensor differs (`turn_rloo` keeps the raw return-to-go, `rloo` returns its advantages), which changes
`critic/returns/*` and nothing in the actor update.

Why this baseline: subtracting the group mean of the *same turn's reward* (the MT-GRPO shape) is biased whenever the
policy decides whether a later turn happens, because members that stopped early contribute no later rewards while
the mean is taken over all members. The leave-one-out return-to-go mean over the members that *reached* the turn is
a constant w.r.t. the member's own action. `tests/trainer/ppo/test_turn_rloo_on_cpu.py` measures both against the
closed-form gradient of a two-turn MDP.

Conditions (documented): the unbiasedness argument holds for independent trajectories, `gamma = 1` and
`norm_adv_by_std=False` (the defaults; the exact-gradient test is that case). `norm_adv_by_std=True` divides by a
standard deviation that includes the sample's own return, so it is a variance-reduction heuristic (enumerated:
rewards `[0, 1, 4]`, probabilities `[0.2, 0.3, 0.5]`, groups of 3: `[-0.46, -0.39, 0.85]` unnormalised vs about
`[-0.245, -0.185, 0.430]` normalised, not a common rescaling). `gamma < 1` uses the discounted return-to-go without
the outer `gamma^t` weight (the `reinforce_plus_plus` convention), not the gradient of `E[sum gamma^t r_t]`
(two independent Bernoulli turns, `gamma` 0.5: `[0.25, 0.25]` vs `[0.25, 0.125]`), a heuristic. Outputs of one V1
multi-output session share `uid` and are correlated, outside the argument.

Plumbing, gated: `core_algos.adv_estimator_takes_turn_spans(name | fn)` is the one signature check. The V0
`compute_advantage` forwards `non_tensor_batch["generated_turn_spans"]` as `turn_spans` only to estimators that
declare the parameter (`**kwargs` alone does not opt in; the outermost callable's own signature decides, and a
`functools.wraps` wrapper opts in only when it takes `**kwargs` and the wrapped estimator declares `turn_spans`, so a
wrapper must expose what it forwards), and the V1 trainer selects and unwraps the TransferQueue
`extra_fields` column only for those estimators (every other estimator pays nothing for the column) and logs
`adv_estimator/rows_with_turn_spans`. `compute_turn_rloo_advantage` validates spans (ordered, non-negative,
non-empty, non-overlapping, within the row, integral offsets with floats and bools rejected by name, one entry per
row) and raises a `ValueError` naming the row: the loop clips its own spans, so a violation means an external
producer. Also fixes `TestRegisterAdvEst`, which left
`ADV_ESTIMATOR_REGISTRY` empty for the rest of a test session.

Answers RFC #7244 and the earlier asks #2540 / #3683. The author of RFC #7244 is welcome as a co-author of this PR
(offered here; no prior contact).

### Checklist Before Starting

- [x] Search for similar PRs (read-only searches re-run 2026-10-29 for "turn-level reward", "turn_rewards",
  "per-turn", "episode done", "turn_rloo", "generated_turn_spans"; no PR mentions `turn_rloo`):
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+turn-level+reward
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+%22turn_rewards%22+OR+%22generated_turn_spans%22
  - https://github.com/verl-project/verl/issues?q=%22turn-level%22+OR+%22per-turn%22+OR+GiGPO+OR+MT-GRPO
  - Coordination with #8305 (draft, luzy99, open, last updated 2026-10-21, +964/-3, 19 files): opt-in turn-level
    process rewards for V1 with user-supplied turn rewards, input validation, transport and actor-side advantage
    shaping after the outcome estimator; no estimator, `core_algos.py` untouched. The shared interface is the
    `generated_turn_spans` / `turn_rewards` names and the alignment rule ([1/2] reuses them). Proposal: one schema
    for the two fields (names, index alignment, clipping, validation), with the reward source and the consumer
    (an estimator here, shaping there) as separate switches; this PR rebases onto #8305's schema if it lands first.
  - #8203 (Begunner, BREAKING rewrite of `agent_loop_tq.py` / `trainer_base.py`): the trainer-side hunk here is 15
    lines in `_compute_advantage` and will be rebased after it lands. #7580 (weighted multi-output trajectories)
    touches `trainer_base.py` nearby. #8099 (RFC issue, optimizer-step accounting): orthogonal.
- [x] Format the PR title as `[{modules}] {type}: {description}` (checked with `tests/special_sanity/check_pr_title.py`).

### Test

CPU harness (macOS, no GPU; CI's `python_files=*_on_cpu.py`):
`PYTHONPATH=<tree> python -m pytest -c pytest-cpu.ini --asyncio-mode=auto <files>`, log attached as `tests.log`.

- `tests/trainer/ppo/test_turn_rloo_on_cpu.py` (27): exact policy-gradient check on a two-turn MDP whose policy
  controls termination, 20k groups of 8 (`turn_rloo` within 4 SE on both coordinates, per-turn-index normalisation
  off by `-p(1-p)pμ` as RFC #7244 predicts, >10 SE); tensor == scalar formula; no spans == `rloo` advantages;
  `rloo` returns vs raw returns on one-token rewards `[1, 3]`; the `norm_adv_by_std` enumeration above; the
  `gamma = 0.5` enumeration above; ragged turns + gamma by hand; thirteen malformed-span shapes raise naming the row
  (incl. float / bool / numpy-float / torch-bool offsets and a string row), a scalar column is named;
  `adv_estimator_takes_turn_spans` (names, `**kwargs`, `functools.wraps`); a transparent wrapper gets the spans and an
  incompatible one gets `rloo` behaviour instead of a crash; `compute_advantage` forwards the spans
  only to estimators that declare them (`grpo_vectorized` and a by-name estimator unchanged next to the column);
  dense rewards through GAE / REINFORCE++ reach the policy only on generated tokens.
- `tests/trainer/ppo/v1/test_compute_advantage_turn_spans_on_cpu.py` (6): the V1 trainer lifts the spans out of
  TransferQueue-shaped rows into `turn_rloo` with the row count metric; rows without spans keep `rloo` behaviour; a
  non-dict row is tolerated; `rloo` / `reinforce_plus_plus` / `grpo` never select `extra_fields`.
- `tests/experimental/agent_loop/test_turn_level_rewards_on_cpu.py` (+1): four real `ToolAgentLoop` episodes through
  `turn_rloo` equal hand-computed advantages. `tests/trainer/ppo/test_core_algos_on_cpu.py`: registry save/restore.
- Results @ e1052882 (`tests.log`): the files above pass (106); `tests/trainer` (477 / 1 skipped) and
  `tests/experimental` (110) groups on this tree, the full six-group CPU regression on 64f8f604 (1538 passed / 64
  skipped / 0 failed); `pre-commit` (ruff,
  ruff-format, mypy, license, docs time-info) passed on the changed files.
- GPU: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`,
  so it does not execute this PR's new tests) passed 503 / 0 / 8 plus both torchrun tests (rc=0) on 8xH100 (Modal,
  `verlai/verl:uv.cu130.dev5`) on this branch @ 988a10bc (the current head e1052882 is the same [2/2] diff rebased onto [1/2]'s round-3
  reward-manager contract commit), identical to main.
- Enabled-feature GPU evidence (this branch @ e1052882; tables and raw runs in `packages/p4/bench/ENABLED-RUN.md` and
  `packages/p4/bench/RESULTS.md`): verl's tool-calling GSM8K through the `ToolAgentLoop` (Qwen3-1.7B, vLLM async, V1
  trainer with TransferQueue, 8xH100, `max_assistant_turns=3`, a `calc_gsm8k_reward` tool, a `custom_reward_function`
  that scores the outcome and reports the metadata it receives). The data path holds end to end: generation -> tool
  execution -> reward callback with the metadata (`metadata_present` = `spans_ok` = 1.0 with the flag on, 0.0 off) ->
  TransferQueue -> `turn_rloo` (`adv_estimator/rows_with_turn_spans` 512 on every step, `critic/returns/mean` ~1.4 vs
  ~0) -> actor update, and `ToolResponse.done` ends episodes on GPU (`episode_done` = `turn_rewards_sum` 0.82-0.84 at
  step 30). A first experiment (per-submission correctness reward, no `done`, 30 steps, two seeds) showed no
  demonstrated accuracy gain (pooled 0.818 vs 0.828) and was invalid as an estimator test: the enabled arms farmed the
  reward by resubmitting the same correct answer (923 / 1085 of 1319 validation episodes). The redesigned experiment
  (a first-correct reward with `done_when_correct`, one outcome scorer, 30 steps, two data seeds, validation every 5
  steps) compares A outcome-only GRPO, B dense + `reinforce_plus_plus`, C dense + `turn_rloo`; pooled validation
  accuracy at steps 0 / 5 / 10 / 15 / 20 / 25 / 30: A 0.668 0.731 0.791 0.811 0.815 0.825 0.832, B 0.663 0.752 0.795
  0.806 0.812 0.818 0.822, C 0.662 0.757 0.789 0.802 0.812 0.828 0.829 (step 30 per seed: A 0.835 / 0.830, B 0.823 /
  0.821, C 0.841 / 0.818; repeated submissions <= 1.4 %, no episode with >= 2 step points; mean step 15.2-16.5 s in
  every arm except one B seed on a slow node, 23.7 s). All arms lie within the 1-2-point seed spread (B-A -1.0..+2.1,
  C-A -0.9..+2.7, C-B -0.7..+1.0 points): no demonstrated accuracy gain for either estimator, and `turn_rloo` is
  indistinguishable from `reinforce_plus_plus` on the same dense reward. This PR makes no algorithm-performance claim:
  GSM8K is a single-answer task where per-turn credit has nothing to assign; the capability is for environments with
  genuine per-step rewards and episode termination (the workcell-style loop in the motivation).

The GPU cost of the `bs x turns` index build was not isolated; in the 6-step runs `timing_s/adv` was 0.23-0.32 s per step
for `turn_rloo`, within the control's range (exploratory, one run per arm).

### API and Usage Example

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      turn_level_rewards: True   # [1/2]: the dense per-turn rm_scores row and generated_turn_spans
algorithm:
  adv_estimator: turn_rloo       # new
  gamma: 1.0                     # per-turn discount for turn_rloo (1.0 = RFC #7244's unbiased form)
```

A custom estimator opts into the spans by declaring a `turn_spans` parameter:
`def my_estimator(token_level_rewards, response_mask, index, turn_spans=None, config=None, **kwargs)`.
`compute_turn_rloo_advantage(..., norm_adv_by_std=False, epsilon=1e-6)` and `adv_estimator_takes_turn_spans` are
importable from `core_algos`. No existing API changes.

### Design & Code Changes

- `verl/trainer/ppo/core_algos.py`: `compute_turn_rloo_advantage` registered as `"turn_rloo"` (no enum change):
  segment reward of turn t = sum of `token_level_rewards` over `[s_t, s_{t+1})`, `G[i,t] = Σ_{t'≥t} γ^{t'-t} R[i,t']`,
  baseline = mean `G[j,t]` over the other members that reached t (0 if none), vectorised with `scatter_add_` /
  `index_add_`; `_validated_turn_spans`; `adv_estimator_takes_turn_spans`.
- `verl/trainer/ppo/ray_trainer.py` (+5): the signature-gated `turn_spans` kwarg.
- `verl/trainer/ppo/v1/trainer_base.py` (+15): `extra_fields` selected and lifted only for such estimators;
  `adv_estimator/rows_with_turn_spans`.
- `docs/advance/agent_loop.rst` (estimator paragraph, the conditions paragraph, that without `turn_level_rewards`
  no spans reach the trainer), `docs/algo/ppo.md`, `docs/examples/config.rst`.

Known limits (documented): n=1 groups and turns reached by one member only have baseline 0 (unbiased, higher
variance); rollout-correction-rejected rows still enter baselines, exactly as `rloo` today; a wrapper that does not
expose `turn_spans` (no explicit parameter, or `**kwargs` without `__wrapped__`) receives no spans and degrades to
`rloo` behaviour.

### AI assistance

Written with Claude (Anthropic) under the author's direction. Commits carry a `Co-authored-by` trailer. The design
follows RFC #7244; the exact-gradient test is a port of a standalone CPU study run during the design.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply pre-commit checks on changed files (`ruff`, `ruff-format`, `mypy`, docs time info, license header).
- [x] Add / Update the documentation.
- [x] Add unit tests: every new code path has a `*_on_cpu.py` test that runs in the existing CPU workflow.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.
