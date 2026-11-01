# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""R5: loop edge paths: done on the last allowed turn, truncation mid-turn, the context-merge budget hit,
tool errors, parallel calls with done, and the V0 row writer when a response has no generated token."""

from types import SimpleNamespace

import pytest
import torch

from scripted_loop import GEN, OBS, StepTool, make_loop, mask_runs, run_episode
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput


async def test_done_on_turn_one_of_a_max_turns_one_loop_never_runs_the_tool():
    tool = StepTool([1.0], done_on=[0])
    loop = make_loop([tool], tool_call_turns=1, max_assistant_turns=1)
    out = await run_episode(loop)
    # max_assistant_turns terminates BEFORE tool extraction: the tool is never called, done never fires.
    assert tool.calls == 0 and out.extra_fields["episode_done"] is False
    assert out.extra_fields["generated_turn_spans"] == [(0, GEN)] and out.extra_fields["turn_rewards"] == [0.0]


async def test_truncation_mid_turn_clips_the_span_and_keeps_the_outcome_on_a_generated_token():
    tool = StepTool([0.25, -0.5, 1.0], done_on=[2])
    loop = make_loop([tool], tool_call_turns=3, response_length=GEN + OBS + 2)  # turn 2 cut after 2 tokens
    out = await run_episode(loop)
    assert len(out.response_ids) == len(out.response_mask) == GEN + OBS + 2
    assert out.extra_fields["generated_turn_spans"] == [(0, GEN), (GEN + OBS, GEN + OBS + 2)]
    assert out.extra_fields["turn_rewards"] == [0.25, 0.0]  # turn 2's tool never ran
    out.reward_score = 1.0
    rm = out.as_dict()["rm_scores"]
    assert rm[GEN - 1].item() == pytest.approx(0.25) and rm[-1].item() == pytest.approx(1.0)
    assert out.response_mask[-1] == 1


async def test_context_merge_budget_hit_keeps_the_step_reward_but_drops_the_observation():
    tool = StepTool([0.25, -0.5, 1.0], done_on=[2])
    loop = make_loop([tool], tool_call_turns=3, response_length=2 * GEN + OBS + 1)  # 2nd observation overflows
    out = await run_episode(loop)
    assert tool.calls == 2
    assert out.response_mask == [1] * GEN + [0] * OBS + [1] * GEN  # the 2nd observation never entered
    assert out.extra_fields["generated_turn_spans"] == mask_runs(out.response_mask)
    assert out.extra_fields["turn_rewards"] == [0.25, -0.5]  # credited although its observation is absent
    assert out.extra_fields["episode_done"] is False


async def test_unknown_tool_and_tool_exception_count_as_zero_reward_turns():
    class Boom(StepTool):
        async def execute(self, instance_id, parameters, **kwargs):
            raise RuntimeError("device offline")

    tool = Boom([0.0, 0.0], name="boom")
    loop = make_loop([tool], tool_call_turns=2)
    out = await run_episode(loop)
    assert out.extra_fields["turn_rewards"] == [0.0, 0.0, 0.0]
    assert out.extra_fields["tool_rewards"] == [0.0, 0.0]
    assert out.extra_fields["episode_done"] is False
    assert out.extra_fields["generated_turn_spans"] == mask_runs(out.response_mask)


async def test_parallel_calls_sum_rewards_and_any_done_ends_the_episode():
    a = StepTool([0.5, 0.5], done_on=[1], name="a")
    b = StepTool([2.0, 2.0], name="b")
    loop = make_loop([a, b], tool_call_turns=4, max_parallel_calls=2, calls_per_turn=2)
    out = await run_episode(loop)
    assert a.calls == 2 and b.calls == 2 and out.extra_fields["episode_done"] is True
    assert out.extra_fields["turn_rewards"] == [2.5, 2.5]
    assert out.extra_fields["generated_turn_spans"] == mask_runs(out.response_mask)


def _row(reward_score, turn_rewards, spans, mask, attention):
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[0, 7, 8]]),
        response_ids=torch.tensor([[11, 12, 13, 14, 15, 0]]),
        input_ids=torch.tensor([[0, 7, 8, 11, 12, 13, 14, 15, 0]]),
        position_ids=torch.arange(9).unsqueeze(0),
        response_mask=torch.tensor([mask]),
        attention_mask=torch.tensor([attention]),
        reward_score=reward_score,
        turn_rewards=turn_rewards,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"generated_turn_spans": spans, "turn_rewards": turn_rewards or [], "episode_done": False},
    )


def test_v0_row_with_no_generated_token_puts_the_outcome_where_the_default_path_does():
    """A 5-token response whose tokens are all mask-0 (6-wide padded row): the default path writes the
    score on the last VALID token (index 4); the dense path must not move it onto the pad column."""
    mask, attention = [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0]
    worker = SimpleNamespace(reward_loop_worker_handles=None)
    dense = AgentLoopWorker._postprocess(worker, [_row(1.0, [], [], mask, attention)]).batch["rm_scores"][0]
    scalar = AgentLoopWorker._postprocess(worker, [_row(1.0, None, [], mask, attention)]).batch["rm_scores"][0]
    assert scalar.tolist() == [0, 0, 0, 0, 1.0, 0]
    assert dense.tolist() == scalar.tolist(), f"dense put the outcome at {dense.nonzero().flatten().tolist()}"
