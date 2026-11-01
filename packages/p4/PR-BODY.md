> Combined round-1 record (branch contrib/p4-turn-rewards, re-pointed to the [2/2] head on 2026-10-29). The PR bodies to submit are packages/p4a/PR-BODY.md and packages/p4b/PR-BODY.md, which carry the round-2 fixes, wording and GPU evidence.

> Combined reference body (branch `contrib/p4-turn-rewards`). Superseded for submission by the split:
> packages/p4a/PR-BODY.md ([1/2] signals + writer) and packages/p4b/PR-BODY.md ([2/2] estimator), which also carry the
> review fixes (gated metadata export and `turn_spans` forwarding, reward-model validation, outcome fallback, warnings).

### What does this PR do?

Gives multi-turn tool agent loops per-turn credit, end to end, all opt-in:

1. **`ToolResponse.done`** (`verl/tools/schemas.py`): a tool returns `ToolResponse(text=..., done=True)` to end the
   episode. `ToolAgentLoop` appends the response as the final observation and terminates instead of generating another
   turn; `extra_fields["episode_done"]` records it. Until now a tool could only end an episode by *saying* so in text.
2. **Turn spans and per-turn rewards** (`verl/experimental/agent_loop/tool_agent_loop.py`): the loop records the
   `[start, end)` offsets of every generated assistant turn within `response_ids` (`extra_fields["generated_turn_spans"]`,
   aligned through every Continuous Token merge with the same rule as draft #8305) and sums the step rewards that
   `BaseTool.execute` already returns per assistant turn (`extra_fields["turn_rewards"]`, index-aligned with the spans).
   Today those step rewards are collected (`tool_rewards`) and consumed by nothing.
3. **Dense `rm_scores`** behind `actor_rollout_ref.rollout.multi_turn.turn_level_rewards` (default `False`): each
   turn's reward lands on the *last generated token* of its span and the episode score on the last generated token of
   the response, never on a tool-response token. GAE and REINFORCE++ carry returns across `response_mask == 0` spans
   but drop a reward that sits on one (#8367), so this placement is what makes them credit turns. With the flag off the
   writer is bitwise unchanged (`rm_scores[-1] = reward_score`).
4. **`algorithm.adv_estimator: turn_rloo`** (`verl/trainer/ppo/core_algos.py`): RFC #7244's leave-one-out
   return-to-go baseline per turn index, vectorised over the batch, falling back to exactly `rloo` on rows without spans.
   The spans are plumbed through `compute_advantage` (V0 and V1, which calls the same function) and lifted out of the
   TransferQueue row in the V1 trainer.

Answers RFC #7244 (and the earlier asks #2540 / #3683) and issue #8367's multi-turn case. Motivation: an
observe -> act -> reward loop (a device tool scoring every cycle) produces one reward per turn; verl can only train on
one scalar per episode today.

### Checklist Before Starting

- [x] Search for similar PRs. Query links:
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+turn-level+reward
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+%22turn_rewards%22+OR+%22generated_turn_spans%22
  - https://github.com/verl-project/verl/issues?q=%22turn-level%22+OR+%22per-turn%22+OR+GiGPO+OR+MT-GRPO
  - Related and how this differs:
    - #8305 (draft, luzy99): adds `generated_turn_spans` + user-supplied `turn_rewards` and shapes advantages *after* the
      outcome estimator (`process_reward.*`, V1 only). This PR reuses its field names and alignment rule so the two can
      stack; it differs in building `turn_rewards` from the tool step rewards the loop already has, writing them into
      `rm_scores`, adding `done`, and adding an estimator. No overlap in `core_algos.py` / `schemas.py`.
    - #8368 (chrisyifanjin): moves the single outcome score to the last `response_mask == 1` token. Complementary: this
      PR adopts the same placement rule inside the opt-in dense writer and leaves the default path to #8368.
    - RFC #7244 (lijuliana): the estimator implemented here; the RFC's reference fork is not public. Co-authorship is offered to the RFC author in this PR text.
    - #8099 (RFC issue, AWS agentcore): optimizer-step accounting for multi-output rollouts, orthogonal.
    - #8203 (Begunner, BREAKING rewrite of `agent_loop_tq.py` / `trainer_base.py`): the trainer-side hunk here is 9
      lines in `_compute_advantage` and will be rebased after it lands.
- [x] Format the PR title as `[{modules}] {type}: {description}` (checked with `tests/special_sanity/check_pr_title.py`).

### Test

CPU unit tests, all run on this branch (log attached as `tests.log`):

- `tests/trainer/ppo/test_turn_rloo_on_cpu.py` (new): exact policy-gradient check on a two-turn MDP whose policy
  controls termination, 20k groups of 8: `turn_rloo` matches the closed-form gradient within 4 SE on both coordinates,
  while the per-turn-index normalisation (MT-GRPO shape) is off by `-p(1-p)·p·μ` as RFC #7244 predicts (>10 SE); tensor
  implementation == scalar formula; no spans == `rloo` bitwise-close; hand-computed ragged-turn + gamma case; optional std
  normalisation; dense rewards through GAE and REINFORCE++ reach the policy only when placed on generated tokens.
- `tests/experimental/agent_loop/test_turn_level_rewards_on_cpu.py` (new): the real `ToolAgentLoop.run` with a scripted
  server/parser/merges and a stateful tool returning per-step rewards and `done` on its third call: generation stops on
  `done` (3 generate calls with `max_assistant_turns=8`), spans `[(0,3),(5,8),(10,13)]`, `turn_rewards`, `episode_done`;
  without `done` the loop runs until the model stops calling tools; dense `rm_scores` placement; flag-off byte identity;
  a group of four episodes through `turn_rloo` equals hand-computed advantages; the V0 `_postprocess` row writer.
- `tests/trainer/ppo/v1/test_compute_advantage_turn_spans_on_cpu.py` (new): the V1 trainer lifts
  `extra_fields["generated_turn_spans"]` out of the TransferQueue row into `turn_rloo`; rows without spans keep `rloo`.
- `tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py` (+1): the TQ row stores the dense `rm_scores`.
- `tests/experimental/agent_loop/test_tool_call_id_on_cpu.py`: fakes updated to the real `MergeResult` / new `AgentData` fields.
- Results on this branch (macOS CPU harness, `tests.log`): new tests 24 passed; `tests/trainer` 448 passed / 1 skipped;
  `tests/experimental` 80 passed; `tests/tools` 42 passed; `tests/utils` 473 passed / 23 skipped (the known
  macOS-only exclusions: megatron collection, `test_sandbox_on_cpu.py`).

- GPU: verl's `gpu_unit_tests.yml` selection on 8xH100 (Modal, `verlai/verl:uv.cu130.dev5`) passed 503 / failed 0 /
  skipped 8, plus both torchrun tests rc=0, on `contrib/p4b-turn-rloo` @ 97a43c92 (which contains p4a f4651857) and on the
  combined `contrib/p4-turn-rewards`; identical to main. Harness: `bench/README.md`; results: `packages/p4b/bench/RESULTS.md`.

No real-environment learning curve was measured: the evidence is the CPU tests plus the GPU unit suite. Planned experiment
(not part of this PR's evidence): a text grid-world workcell (8-turn episodes, step reward = -Δdistance, +1 on placement with
`done=True`), GRPO-outcome vs REINFORCE++-dense vs `turn_rloo`, steps-to-0.8 success.

### API and Usage Example

```python
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

class WorkcellTool(BaseTool):
    async def execute(self, instance_id, parameters, **kwargs):
        observation, step_reward, placed = await self.device.step(parameters)   # one control cycle
        # step_reward credits the assistant turn that issued this call; done=True ends the episode
        return ToolResponse(text=observation, done=placed), step_reward, {}
```

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True
      tool_config_path: workcell_tools.yaml
      turn_level_rewards: True        # new, default False
algorithm:
  adv_estimator: turn_rloo            # new; reinforce_plus_plus / gae also consume the dense rewards
  gamma: 1.0                          # turn-level discount for turn_rloo (1.0 = RFC #7244's form)
```

Exported per trajectory (always, flag or not): `extra_fields["generated_turn_spans"]`, `extra_fields["turn_rewards"]`,
`extra_fields["episode_done"]`; a `custom_reward_function` sees them in `extra_info`. New typed field
`AgentLoopOutput.turn_rewards` (None unless the flag is on). No existing API changes.

### Design & Code Changes

- `verl/tools/schemas.py`: `ToolResponse.done: bool = False`; `_call_tool` carries it through the rebuilt response.
- `verl/experimental/agent_loop/tool_agent_loop.py`: `AgentData.generated_turn_spans / turn_rewards / episode_done`;
  `_align_generated_turn_spans(agent_data, merge_result)` applied before each merge's aligned mask replaces the old one
  (trimmed suffix clips spans, an assistant merge appends `(retained + inserted, + appended_token_count)`; same rule as
  #8305); the turn's tool rewards are added to `turn_rewards[-1]`; `done` -> `TERMINATED` after the context merge; spans
  clipped to `response_length` at output.
- `verl/experimental/agent_loop/agent_loop.py`: `write_turn_level_rewards(...)` (one row, last-generated-token rule);
  `AgentLoopOutput.turn_rewards`; `as_dict` and `_postprocess` use it only when `turn_rewards` is set; the three new keys
  join `default_extra_keys` so batches from mixed agent loops keep one schema.
- `verl/trainer/ppo/core_algos.py`: `compute_turn_rloo_advantage` registered as `"turn_rloo"` (no enum change, per the
  note in `AdvantageEstimator`). Segment reward of turn t = sum of `token_level_rewards` over `[s_t, s_{t+1})`
  (segment 0 from token 0, last segment to the row end), `G[i,t] = Σ_{t'≥t} γ^{t'-t} R[i,t']`, baseline = mean `G[j,t]`
  over the other members that reached t (0 if none), broadcast on the turn's generated tokens, 0 elsewhere.
- `verl/trainer/ppo/ray_trainer.py` (+2) / `verl/trainer/ppo/v1/trainer_base.py` (+9): pass `turn_spans` when
  `generated_turn_spans` is present.
- `verl/workers/config/rollout.py`, `verl/trainer/config/rollout/rollout.yaml`, `_generated_*.yaml`: the flag.
- `docs/advance/agent_loop.rst`: new section; `docs/algo/ppo.md`, `docs/examples/config.rst`: estimator lists.

Why the LOO return-to-go baseline: with per-turn rewards the obvious baseline (subtract the group mean of the same
turn's reward, then sum) is biased whenever the policy decides whether a later turn happens, because members that
stopped early contribute no later rewards while the mean is taken over all members. The leave-one-out mean of the
return-to-go over members that *reached* the turn is a constant w.r.t. the member's own action, so it is unbiased; on
single-turn data it is exactly RLOO. The bias study is `tests/trainer/ppo/test_turn_rloo_on_cpu.py`.

Known limits (documented): n=1 groups and turns reached by one member only have baseline 0 (unbiased, higher variance);
GRPO / RLOO / REINFORCE++-baseline sum the dense row (episodic, as before); a wall-clock episode deadline is a follow-up.

### AI assistance

Written with Claude (Anthropic) under the author's direction. The design follows RFC
#7244 and reuses draft #8305's field names; the exact-gradient test is a port of a standalone CPU study run during the
design. Commits carry a `Co-authored-by` trailer.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply pre-commit checks on changed files (`ruff`, `ruff-format`, `mypy`, `autogen-trainer-cfg`, docs time info,
      license header).
- [x] Add / Update the documentation (`docs/advance/agent_loop.rst`, `docs/algo/ppo.md`, `docs/examples/config.rst`).
- [x] Add unit tests: all new code paths are covered by `*_on_cpu.py` tests that run in the existing CPU workflow.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.
