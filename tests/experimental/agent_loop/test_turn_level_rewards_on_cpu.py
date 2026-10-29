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

"""ToolAgentLoop end to end on CPU: turn spans, per-turn tool rewards, ``ToolResponse.done`` and the dense
``rm_scores`` written by ``rollout.multi_turn.turn_level_rewards``.

The LLM server, tokenizer and Continuous Token merges are scripted; the loop, the tool dispatch and the
reward writer are the real code.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.tokenizer.continuous_token import MergeResult
from verl.workers.rollout.replica import TokenOutput

GENERATED_TOKENS = 3  # tokens per assistant turn
OBSERVATION_TOKENS = 2  # tokens per tool response


class ScriptedWorkcellTool:
    """A stateful tool whose k-th call returns step reward ``rewards[k]`` and ``done`` on the last one."""

    def __init__(self, rewards: list[float], done_on_last: bool):
        self.name = "step"
        self.tool_schema = SimpleNamespace(
            model_dump=lambda **kwargs: {"type": "function", "function": {"name": "step", "parameters": {}}}
        )
        self.rewards = rewards
        self.done_on_last = done_on_last
        self.calls = 0

    async def create(self, create_kwargs=None):
        return "instance", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        reward = self.rewards[self.calls]
        self.calls += 1
        done = self.done_on_last and self.calls == len(self.rewards)
        return ToolResponse(text=f"obs {parameters['k']}", done=done), reward, {}

    async def release(self, instance_id):
        pass


class ScriptedServer:
    """Returns ``GENERATED_TOKENS`` fresh token ids per call and counts the calls."""

    def __init__(self):
        self.calls = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls += 1
        base = 1000 * self.calls
        return TokenOutput(token_ids=[base + i for i in range(GENERATED_TOKENS)], log_probs=None)


class ScriptedToolParser:
    """Every turn up to ``tool_call_turns`` calls the tool once; later turns call nothing."""

    stop_token_ids: list[int] = []

    def __init__(self, tool_call_turns: int):
        self.tool_call_turns = tool_call_turns
        self.turn = 0

    async def extract_tool_calls(self, response_ids, tools):
        self.turn += 1
        if self.turn <= self.tool_call_turns:
            return "", [FunctionCall(name="step", arguments=f'{{"k": {self.turn}}}')]
        return "final answer", []


def make_loop(tool: ScriptedWorkcellTool, *, turn_level_rewards: bool, max_assistant_turns: int = 8) -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.max_user_turns = None
    loop.max_assistant_turns = max_assistant_turns
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 256
    loop.tool_response_truncate_side = "middle"
    loop.turn_level_rewards = turn_level_rewards
    loop.prompt_length = 16
    loop.response_length = 64
    loop.tools = {tool.name: tool}
    loop.tool_schemas = [tool.tool_schema.model_dump()]
    loop.tool_parser = ScriptedToolParser(tool_call_turns=len(tool.rewards))
    loop.tool_parser_name = "hermes"
    loop.processor = None
    loop.server_manager = ScriptedServer()

    async def process_multi_modal_info(messages):
        return {}

    async def ct_build_initial_tokens(messages, **kwargs):
        return [101, 102, 103]

    async def ct_merge_assistant_token(runtime_token_ids, assistant_token_ids, response_mask, *args, **kwargs):
        merge = MergeResult(
            token_ids=list(runtime_token_ids) + list(assistant_token_ids),
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )
        return merge, list(response_mask) + [1] * len(assistant_token_ids), None

    async def ct_merge_context_msg(previous_messages, updated_messages, runtime_token_ids, response_mask, *a, **kw):
        observation = [500 + len(runtime_token_ids), 501 + len(runtime_token_ids)]
        merge = MergeResult(
            token_ids=list(runtime_token_ids) + observation, appended_token_count=len(observation), kind="context"
        )
        return merge, list(response_mask) + [0] * OBSERVATION_TOKENS, None

    loop.process_multi_modal_info = process_multi_modal_info
    loop._get_mm_processor_kwargs = lambda audios: None
    loop._assert_mm_supported = lambda has_multi_modal: None
    loop.ct_build_initial_tokens = ct_build_initial_tokens
    loop.ct_merge_assistant_token = ct_merge_assistant_token
    loop.ct_merge_context_msg = ct_merge_context_msg
    return loop


async def run_episode(rewards: list[float], *, turn_level_rewards: bool, done: bool = True) -> tuple[Any, Any]:
    tool = ScriptedWorkcellTool(rewards, done_on_last=done)
    loop = make_loop(tool, turn_level_rewards=turn_level_rewards)
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "place the part"}])
    return loop, output


def _expected_mask(turns: int, trailing_observation: bool) -> list[int]:
    mask = []
    for t in range(turns):
        mask += [1] * GENERATED_TOKENS
        if t < turns - 1 or trailing_observation:
            mask += [0] * OBSERVATION_TOKENS
    return mask


@pytest.mark.asyncio
async def test_done_ends_the_episode_with_turn_spans_and_turn_rewards():
    loop, output = await run_episode([0.25, -0.5, 1.0], turn_level_rewards=True)

    assert loop.server_manager.calls == 3, "generation stopped on done, not on max_assistant_turns=8"
    assert output.response_mask == _expected_mask(3, trailing_observation=True)
    assert output.extra_fields["generated_turn_spans"] == [(0, 3), (5, 8), (10, 13)]
    assert output.extra_fields["turn_rewards"] == [0.25, -0.5, 1.0]
    assert output.extra_fields["tool_rewards"] == [0.25, -0.5, 1.0]
    assert output.extra_fields["episode_done"] is True
    assert output.turn_rewards == [0.25, -0.5, 1.0]
    assert output.num_turns == 7
    assert len(output.response_ids) == 15


@pytest.mark.asyncio
async def test_without_done_the_loop_keeps_generating_until_the_model_stops_calling_tools():
    loop, output = await run_episode([0.25, -0.5, 1.0], turn_level_rewards=True, done=False)

    assert loop.server_manager.calls == 4
    assert output.extra_fields["episode_done"] is False
    assert output.response_mask == _expected_mask(4, trailing_observation=False)
    assert output.extra_fields["generated_turn_spans"] == [(0, 3), (5, 8), (10, 13), (15, 18)]
    assert output.extra_fields["turn_rewards"] == [0.25, -0.5, 1.0, 0.0]

    output.reward_score = 2.0
    rm_scores = output.as_dict()["rm_scores"]
    expected = torch.zeros(18)
    expected[2], expected[7], expected[12], expected[17] = 0.25, -0.5, 1.0, 2.0
    torch.testing.assert_close(rm_scores, expected)


@pytest.mark.asyncio
async def test_dense_rm_scores_land_on_the_last_generated_token_of_each_turn():
    _, output = await run_episode([0.25, -0.5, 1.0], turn_level_rewards=True)
    output.reward_score = 1.0  # what the reward loop sets after run()

    fields = output.as_dict()

    expected = torch.zeros(15)
    expected[2], expected[7], expected[12] = 0.25, -0.5, 1.0 + 1.0  # outcome joins the final turn's reward
    torch.testing.assert_close(fields["rm_scores"], expected)
    assert fields["rm_scores"][13:].tolist() == [0.0, 0.0], "nothing on the trailing observation (mask 0)"
    assert "turn_rewards" not in fields
    assert fields["extra_fields"]["generated_turn_spans"] == [(0, 3), (5, 8), (10, 13)]


@pytest.mark.asyncio
async def test_flag_off_keeps_the_default_scalar_placement_byte_identical():
    _, output = await run_episode([0.25, -0.5, 1.0], turn_level_rewards=False)
    output.reward_score = 1.0

    fields = output.as_dict()

    assert output.turn_rewards is None
    expected = torch.zeros(15)
    expected[-1] = 1.0  # today's rule: last response token, even though it is an observation token
    assert torch.equal(fields["rm_scores"], expected)
    assert set(fields["extra_fields"]) == {"turn_scores", "tool_rewards"}, "no new metadata keys with the flag off"
    assert set(fields) == {
        "prompts",
        "responses",
        "response_mask",
        "rm_scores",
        "num_turns",
        "metrics",
        "extra_fields",
        "multi_modal_data",
        "mm_processor_kwargs",
        "mm_processor_output",
    }
    assert fields["extra_fields"]["tool_rewards"] == [0.25, -0.5, 1.0]  # exported as before


def test_v0_postprocess_writes_dense_rm_scores_only_for_rows_with_turn_rewards():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput

    def row(reward_score, turn_rewards, spans):
        mask = torch.tensor([[1, 1, 0, 1, 0, 0]])  # two generated turns, a trailing observation, one pad token
        attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0]])  # prompt left-padded (3), response right-padded (6)
        return _InternalAgentLoopOutput(
            prompt_ids=torch.tensor([[0, 7, 8]]),
            response_ids=torch.tensor([[11, 12, 13, 14, 15, 0]]),
            input_ids=torch.tensor([[0, 7, 8, 11, 12, 13, 14, 15, 0]]),
            position_ids=torch.arange(9).unsqueeze(0),
            response_mask=mask,
            attention_mask=attention,
            reward_score=reward_score,
            turn_rewards=turn_rewards,
            num_turns=4,
            metrics=AgentLoopMetrics(),
            extra_fields={"generated_turn_spans": spans, "turn_rewards": turn_rewards or [], "episode_done": True},
        )

    worker = SimpleNamespace(reward_loop_worker_handles=None)
    dense = row(1.0, [0.5, 0.25], [(0, 2), (3, 4)])
    scalar = row(1.0, None, [(0, 2), (3, 4)])

    batch = AgentLoopWorker._postprocess(worker, [dense, scalar])

    rm_scores = batch.batch["rm_scores"]
    assert rm_scores[0].tolist() == pytest.approx([0.0, 0.5, 0.0, 1.25, 0.0, 0.0])
    assert rm_scores[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # unchanged: last attention token
    assert list(batch.non_tensor_batch["generated_turn_spans"]) == [[(0, 2), (3, 4)], [(0, 2), (3, 4)]]
    assert list(batch.non_tensor_batch["episode_done"]) == [True, True]


def test_v0_postprocess_adds_the_turn_level_columns_only_when_the_flag_is_on():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput

    def single_turn_row():
        return _InternalAgentLoopOutput(
            prompt_ids=torch.tensor([[7, 8]]),
            response_ids=torch.tensor([[11, 12, 0]]),
            input_ids=torch.tensor([[7, 8, 11, 12, 0]]),
            position_ids=torch.arange(5).unsqueeze(0),
            response_mask=torch.tensor([[1, 1, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            reward_score=1.0,
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )

    new_keys = {"turn_rewards", "generated_turn_spans", "episode_done"}
    off = SimpleNamespace(reward_loop_worker_handles=None, rollout_config=None)
    assert not new_keys & set(AgentLoopWorker._postprocess(off, [single_turn_row()]).non_tensor_batch)
    on = SimpleNamespace(
        reward_loop_worker_handles=None,
        rollout_config=SimpleNamespace(multi_turn=SimpleNamespace(turn_level_rewards=True)),
    )
    columns = AgentLoopWorker._postprocess(on, [single_turn_row()]).non_tensor_batch
    assert new_keys <= set(columns) and all(columns[key][0] is None for key in new_keys)


def test_v0_row_without_generated_tokens_puts_the_outcome_on_the_last_valid_token():
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput

    row = _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[7, 8]]),
        response_ids=torch.tensor([[11, 12, 13, 0]]),
        input_ids=torch.tensor([[7, 8, 11, 12, 13, 0]]),
        position_ids=torch.arange(6).unsqueeze(0),
        response_mask=torch.tensor([[0, 0, 0, 0]]),  # three observation tokens, one pad column
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
        reward_score=1.0,
        turn_rewards=[],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"generated_turn_spans": []},
    )
    worker = SimpleNamespace(reward_loop_worker_handles=None, rollout_config=None)
    rm_scores = AgentLoopWorker._postprocess(worker, [row]).batch["rm_scores"]
    assert rm_scores[0].tolist() == [0.0, 0.0, 1.0, 0.0], "the default path's position (last valid token), not the pad"


def test_writer_warns_when_a_turn_has_no_generated_token():
    from verl.experimental.agent_loop.agent_loop import write_turn_level_rewards

    rm_scores, mask = torch.zeros(6), torch.tensor([1, 1, 0, 0, 0, 0])
    with pytest.warns(RuntimeWarning, match="turn 1 has no generated token"):
        write_turn_level_rewards(rm_scores, mask, [(0, 2), (2, 4)], [0.5, 1.0], reward_score=None)
    assert rm_scores.tolist() == [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]


def test_alignment_warns_when_a_merge_removes_a_rewarded_turn():
    from verl.experimental.agent_loop.tool_agent_loop import _align_generated_turn_spans

    agent_data = SimpleNamespace(
        response_mask=[1, 1, 1, 0, 1, 1], generated_turn_spans=[(0, 3), (4, 6)], turn_rewards=[0.5, 1.0]
    )
    with pytest.warns(RuntimeWarning, match="whole generated turn"):
        _align_generated_turn_spans(
            agent_data, MergeResult(token_ids=[], appended_token_count=0, kind="context", removed_prefix_token_count=3)
        )
    assert agent_data.generated_turn_spans == [(0, 3)] and agent_data.turn_rewards == [0.5]
