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
"""R2: span alignment when the Continuous Token context merge edits the junction, which the branch's
own tests never exercise: GLM removes the trailing generated boundary token (removed_prefix_token_count=1),
Gemma inserts <|tool_response> (inserted_token_ids), and the pathological case of a merge that removes
more than the whole last turn (the reward must not vanish silently)."""

import warnings

import pytest
import torch

from scripted_loop import GEN, StepTool, make_loop, mask_runs, run_episode
from verl.experimental.agent_loop.agent_loop import write_turn_level_rewards


@pytest.mark.parametrize("removed,inserted", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)])
async def test_spans_equal_mask_runs_under_junction_edits(removed, inserted):
    tool = StepTool([0.25, -0.5, 1.0], done_on=[2])
    loop = make_loop([tool], tool_call_turns=3, removed_on_context=removed, inserted_on_context=inserted)
    out = await run_episode(loop)
    spans = out.extra_fields["generated_turn_spans"]
    assert spans == mask_runs(out.response_mask), (spans, out.response_mask)
    assert out.extra_fields["turn_rewards"] == [0.25, -0.5, 1.0]
    assert len(out.response_ids) == len(out.response_mask)
    out.reward_score = 2.0
    rm = out.as_dict()["rm_scores"]
    expected = torch.zeros(len(out.response_mask))
    for (s, e), r in zip(spans, [0.25, -0.5, 1.0], strict=True):
        expected[e - 1] += r
    expected[spans[-1][1] - 1] += 2.0
    torch.testing.assert_close(rm, expected)
    assert all(out.response_mask[i] == 1 for i in rm.nonzero().flatten().tolist())


async def test_merge_that_removes_the_whole_turn_warns_and_drops_its_reward():
    """Pathological: a context merge that strips GEN+1 trailing tokens removes the entire last assistant
    span and, with it, the turn's step reward. Accepted behaviour after fix D7: the loop WARNS (RuntimeWarning)
    and drops; the mask and spans stay consistent and tool_rewards still records every step reward.
    Pre-fix this was silent (the original version of this test asserted conservation and failed)."""
    tool = StepTool([0.25, -0.5, 1.0], done_on=[2])
    loop = make_loop([tool], tool_call_turns=3, removed_on_context=GEN + 1)
    with pytest.warns(RuntimeWarning, match="whole generated turn"):
        out = await run_episode(loop)
    assert out.extra_fields["generated_turn_spans"] == mask_runs(out.response_mask)
    assert out.extra_fields["tool_rewards"] == [0.25, -0.5, 1.0]
    assert sum(out.extra_fields["turn_rewards"]) != pytest.approx(sum(tool.rewards))  # dropped, not conserved


def test_writer_warns_and_drops_a_reward_whose_span_has_no_generated_token():
    rm = torch.zeros(6)
    mask = torch.tensor([1, 1, 0, 0, 0, 0])
    with pytest.warns(RuntimeWarning, match="no generated token"):
        write_turn_level_rewards(rm, mask, [(0, 2), (2, 4)], [0.5, 1.0], reward_score=None)
    assert rm.tolist() == [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a zero reward on an unplaceable span is not worth a warning
        write_turn_level_rewards(torch.zeros(6), mask, [(0, 2), (2, 4)], [0.5, 0.0], reward_score=None)
