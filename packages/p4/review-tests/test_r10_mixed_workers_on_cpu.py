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
"""R10 (fix D1, column half): with the flag on, a worker chunk holding only single-turn rows must still carry the
three schema columns, otherwise ``DataProto.concat`` across workers fails (``list_of_dict_to_dict_of_list``
asserts later chunks add no keys). Also the hazard of a worker subclass without ``rollout_config``."""

from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput

NEW_KEYS = {"turn_rewards", "generated_turn_spans", "episode_done"}


def _row(tool_agent: bool, flag: bool):
    extra = {"turn_scores": [], "tool_rewards": [0.5] if tool_agent else []}
    turn_rewards = None
    if tool_agent and flag:
        extra.update({"generated_turn_spans": [(0, 2)], "turn_rewards": [0.5], "episode_done": True})
        turn_rewards = [0.5]
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[7, 8]]),
        response_ids=torch.tensor([[11, 12, 0]]),
        input_ids=torch.tensor([[7, 8, 11, 12, 0]]),
        position_ids=torch.arange(5).unsqueeze(0),
        response_mask=torch.tensor([[1, 1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        reward_score=1.0,
        turn_rewards=turn_rewards,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra,
    )


def _worker(flag, with_attr=True):
    kwargs = {"reward_loop_worker_handles": None}
    if with_attr:
        kwargs["rollout_config"] = SimpleNamespace(multi_turn=SimpleNamespace(turn_level_rewards=flag))
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize("flag", [False, True])
def test_single_turn_chunk_first_then_tool_agent_chunk_concat(flag):
    single = AgentLoopWorker._postprocess(_worker(flag), [_row(False, flag), _row(False, flag)])
    tool = AgentLoopWorker._postprocess(_worker(flag), [_row(True, flag), _row(True, flag)])
    merged = DataProto.concat([single, tool])  # the single-turn chunk defines the key set
    assert len(merged) == 4
    assert (NEW_KEYS <= set(merged.non_tensor_batch)) is flag
    if flag:
        assert list(merged.non_tensor_batch["generated_turn_spans"]) == [None, None, [(0, 2)], [(0, 2)]]
        assert merged.batch["rm_scores"][2].tolist() == [0.0, 1.5, 0.0]  # dense: 0.5 turn reward + outcome


def test_worker_without_rollout_config_breaks_cross_worker_concat_with_the_flag_on():
    """Hazard, not a defect of the shipped worker (AgentLoopWorker sets rollout_config at agent_loop.py:331):
    a custom worker lacking the attribute silently skips the schema columns and the concat assertion fires."""
    single = AgentLoopWorker._postprocess(_worker(True, with_attr=False), [_row(False, True)])
    tool = AgentLoopWorker._postprocess(_worker(True, with_attr=False), [_row(True, True)])
    assert not NEW_KEYS & set(single.non_tensor_batch)
    with pytest.raises(AssertionError, match="not present in the keys of the first dictionary"):
        DataProto.concat([single, tool])


def test_flag_off_tool_agent_rows_have_no_new_columns_even_when_mixed():
    merged = DataProto.concat(
        [
            AgentLoopWorker._postprocess(_worker(False), [_row(True, False)]),
            AgentLoopWorker._postprocess(_worker(False), [_row(False, False)]),
        ]
    )
    assert not NEW_KEYS & set(merged.non_tensor_batch)
    assert merged.batch["rm_scores"].tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
