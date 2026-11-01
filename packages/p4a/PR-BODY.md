### What does this PR do?

**[1/2]** of turn-level credit for multi-turn tool agent loops; **[2/2]** is `[algo, trainer] feat: turn_rloo advantage
estimator with gated turn-span plumbing` (branch `contrib/p4b-turn-rloo`, stacked on this one). Everything here is
opt-in and usable alone with `reinforce_plus_plus` or `gae`:

1. **`ToolResponse.done`** (`verl/tools/schemas.py`): a tool returns `ToolResponse(text=..., done=True)` to end the
   episode; `ToolAgentLoop` appends the response as the final observation and terminates instead of generating again.
2. **Turn spans and per-turn rewards** (`verl/experimental/agent_loop/tool_agent_loop.py`): under the flag below the
   loop records the `[start, end)` offsets of every generated assistant turn within `response_ids` (aligned through
   every Continuous Token merge with the same rule as draft #8305) and sums the step rewards `BaseTool.execute`
   already returns per assistant turn (today collected in `tool_rewards` and consumed by nothing).
3. **Dense `rm_scores`** behind `actor_rollout_ref.rollout.multi_turn.turn_level_rewards` (default `False`): each
   turn's reward lands on the *last generated token* of its span and the episode score on the last generated token
   of the response, never on a tool-response token (GAE and REINFORCE++ drop a reward on a `response_mask == 0`
   token, #8367). The loop exports `extra_fields["generated_turn_spans"]`, `["turn_rewards"]`, `["episode_done"]`,
   and, with the flag on, every reward manager (`naive`, `dapo`, `gdpo`, `rate_limited`, `remote`) hands them to a
   `custom_reward_function` in `extra_info` through the new `RewardManagerBase.extra_info_for_reward_fn` (the loop's
   fields override dataset keys of the same name; the loop's list metadata is copied, so a reward function cannot edit
   the row; a non-mapping `extra_info` is handed over unchanged with one warning). With the flag off each manager keeps
   its previous callback contract, read from the config at init: `naive` / `remote` merge the loop's `tool_extra_fields`
   plus `num_turns` as before, `rate_limited` merges `tool_extra_fields` as before, `dapo` / `gdpo` hand the dataset's
   `extra_info` over untouched.
4. **Guard rails**: `validate_config` rejects the flag with a colocated reward model (it would overwrite the row;
   use a rule-based reward or `reward.reward_model.enable_resource_pool=True`); a turn reward that cannot be placed
   raises a `RuntimeWarning` instead of vanishing.

Compatibility: with the flag off the tensors are bitwise what main produces, no span bookkeeping runs, and the
custom-reward callback contract of every manager is unchanged (a callback reading `extra_info["num_turns"]` under
`dapo` gets the dataset's value, as on main); `done` works with the flag off too. Related: #8367 (the masked-tail drop), RFC #7244 / #2540 / #3683 (answered by
[2/2]).

### Checklist Before Starting

- [x] Search for similar PRs (read-only searches re-run 2026-10-29 for "turn-level reward", "turn_rewards",
  "per-turn", "episode done", "turn_rloo", "generated_turn_spans"):
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+turn-level+reward
  - https://github.com/verl-project/verl/pulls?q=is%3Apr+%22turn_rewards%22+OR+%22generated_turn_spans%22
  - https://github.com/verl-project/verl/issues?q=%22turn-level%22+OR+%22per-turn%22+OR+GiGPO+OR+MT-GRPO
  - Coordination with #8305 (draft, luzy99, open, last updated 2026-10-21, +964/-3, 19 files): opt-in turn-level
    process rewards for V1 with user-supplied turn rewards, input validation, transport and actor-side advantage
    shaping after the outcome estimator. The shared interface is the `generated_turn_spans` / `turn_rewards` names
    and the span-alignment rule, which this PR reuses so the two can stack. What differs: the reward source (tool
    step rewards the loop already has, written into `rm_scores`, vs. user-supplied rewards shaped into advantages),
    `done`, and the reward-manager contract. Proposal: converge on one schema for the two fields (names, index
    alignment, clipping at the response budget, validation) and keep the reward source and the consumer as separate
    switches; this PR rebases onto #8305's schema if it lands first.
  - #8368 (chrisyifanjin): the single outcome score on the last `response_mask == 1` token; this PR uses the same
    rule inside the opt-in writer and leaves the default path to #8368.
  - Nearby, not overlapping: #8156 (BaseTool lifecycle) and #6404 (per-assistant-turn cap) in `tool_agent_loop.py`;
    #7580 (weighted multi-output trajectories) in `agent_loop.py` / `agent_loop_tq.py`; #8203 (BREAKING
    `agent_loop_tq.py` / `trainer_base.py` rewrite, neither file touched here); #8099 (optimizer-step accounting).
- [x] Format the PR title as `[{modules}] {type}: {description}` (checked with `tests/special_sanity/check_pr_title.py`).

### Test

CPU harness (macOS, no GPU; CI's `python_files=*_on_cpu.py`):
`PYTHONPATH=<tree> python -m pytest -c pytest-cpu.ini --asyncio-mode=auto <files>`, log attached as `tests.log`.

- `tests/experimental/agent_loop/test_turn_level_rewards_on_cpu.py` (13): the real `ToolAgentLoop.run` with a scripted
  server / parser / merges and a stateful tool: generation stops on `done`, spans `[(0,3),(5,8),(10,13)]`,
  `turn_rewards`, `episode_done`, dense placement, flag-off byte identity of tensors and metadata, the V0 row writer,
  both warnings, and the response-budget boundary (step reward and `done` kept, observation absent).
- `tests/experimental/agent_loop/test_turn_spans_real_tokenizer_on_cpu.py` (2): the loop's own Continuous Token
  merges on `QwenContinuousTokenBuilder` over `~/models/Qwen/Qwen2.5-0.5B` with the real Hermes parser. Generation
  stops at `<|im_end|>`, the context merge inserts the newline boundary token: the spans equal the runs of generated
  tokens and decode to the generated text, the dense rewards land on each turn's `<|im_end|>`, `done` keeps the
  terminal observation, the budget boundary drops it while keeping the step reward and `done`.
- `tests/experimental/reward_loop/test_reward_manager_extra_info_on_cpu.py` (20): with the flag on the three keys plus
  `num_turns` reach a custom reward function under all five managers; the batch's dicts and lists are not mutated; a
  non-mapping `extra_info` is passed through with one warning naming its type; the review's probe (dataset `num_turns`
  7 vs loop 3) over the five managers in both flag states: off keeps each manager's main contract (7 under `dapo` /
  `gdpo` / `rate_limited`, the dataset's own dict under `dapo` / `gdpo`), on delivers the merged contract (3).
- `tests/trainer/test_turn_level_rewards_config_on_cpu.py` (5), `tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py`
  (+1), `tests/experimental/agent_loop/test_tool_call_id_on_cpu.py` (fakes carry the new fields).
- Results @ 0d4cd8ca (`tests.log`): the files above 62 passed; the full CPU regression ran on the stacked tree (see
  [2/2]'s `tests.log`); `pre-commit` (ruff, ruff-format, mypy, license, docs time-info) passed on the changed files.
- GPU: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`,
  so it does not execute this PR's new tests) passed 503 / 0 / 8 plus both torchrun tests (rc=0) on 8xH100 (Modal,
  `verlai/verl:uv.cu130.dev5`) on the stacked tree p4b@988a10bc (containing this branch @ d8c4b925; the current head 0d4cd8ca adds the round-3
  reward-manager contract commit, CPU-tested), identical to main.
- Enabled-feature GPU evidence (run on the stacked tree `contrib/p4b-turn-rloo` @ e1052882, which contains this branch @ 0d4cd8ca; tables and raw runs in `packages/p4/bench/ENABLED-RUN.md` and
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

Not exercised: Continuous Token authors other than Qwen's (GLM / Gemma / Kimi junction trims are covered by scripted
`MergeResult`s with 0-2 removed and 0-2 inserted tokens).

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
      max_assistant_turns: 9          # K steps need K + 1: the last allowed turn never runs its tools
      turn_level_rewards: True        # new, default False
algorithm:
  adv_estimator: reinforce_plus_plus  # or gae; both read the dense rewards ([2/2] adds turn_rloo)
```

New: `AgentLoopOutput.turn_rewards` (None unless the flag is on), `RewardManagerBase.extra_info_for_reward_fn`
(static). No existing API changes.

### Design & Code Changes

- `verl/tools/schemas.py`: `ToolResponse.done: bool = False`; `_call_tool` carries it through the rebuilt response.
- `verl/experimental/agent_loop/tool_agent_loop.py`: `AgentData.generated_turn_spans / turn_rewards / episode_done`;
  `_align_generated_turn_spans(agent_data, merge_result)` before each merge's aligned mask replaces the old one, only
  under the flag (trimmed suffix clips spans, an assistant merge appends `(retained + inserted, + appended)`); the
  turn's tool rewards added to `turn_rewards[-1]`; `done` -> `TERMINATED` after the context merge; spans clipped to
  `response_length` at output; the three keys exported only under the flag.
- `verl/experimental/agent_loop/agent_loop.py`: `write_turn_level_rewards(...)` (one row, last-generated-token rule,
  `fallback` index); `AgentLoopOutput.turn_rewards`; dense rows in `as_dict` / `_postprocess`; the schema columns
  join `default_extra_keys` only in turn-level mode.
- `verl/experimental/reward_loop/reward_manager/{base,naive,dapo,gdpo,limited,remote}.py`:
  `extra_info_for_reward_fn` (the one hunk outside the agent loop).
- `verl/utils/config.py`: `_validate_turn_level_rewards_config`; `verl/workers/config/rollout.py`,
  `verl/trainer/config/rollout/rollout.yaml`, `_generated_*.yaml`: the flag.
- `docs/advance/agent_loop.rst`: the section, incl. the reward-manager contract and the response-budget boundary.

Known limits (documented): a context merge that would reach `response_length` is not applied (as before), so that
turn's step reward and `done` are kept while its tool response is absent from `response_ids`; metrics that sum
`rm_scores` (`critic/score/*`, `val-core/*/reward`, DAPO's group filter) report the return instead of the outcome
when the flag is on, validation included; the last allowed assistant turn never runs its tools; a wall-clock
episode deadline is a follow-up.

### AI assistance

Written with Claude (Anthropic) under the author's direction. Commits carry a `Co-authored-by` trailer.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply pre-commit checks on changed files (`ruff`, `ruff-format`, `mypy`, `autogen-trainer-cfg`, docs time
      info, license header).
- [x] Add / Update the documentation (`docs/advance/agent_loop.rst`).
- [x] Add unit tests: every new code path has a `*_on_cpu.py` test that runs in the existing CPU workflow.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.
