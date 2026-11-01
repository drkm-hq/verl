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
"""R2b (producer side of the strict-validation risk): for EVERY ``response_length`` from 1 past the full
episode, with context merges that trim 0-2 runtime tokens and insert 0-2 boundary tokens, the spans the real
``ToolAgentLoop`` exports are ordered, non-empty, non-overlapping, within ``[0, len(response_ids)]``, are exactly
the runs of ``response_mask == 1``, stay index-aligned with ``turn_rewards``, and the dense ``rm_scores`` row
lands only on generated tokens. Both budget boundaries (mid-generation, at the context merge) occur in the sweep.

Uses the round-1 review harness ``packages/p4/review-tests/scripted_loop.py``.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "p4" / "review-tests"))

from scripted_loop import GEN, OBS, StepTool, make_loop, mask_runs, run_episode  # noqa: E402

TURNS = 4  # three tool-calling turns plus the final answer
LENGTHS = range(1, TURNS * GEN + (TURNS - 1) * OBS + 6)
JUNCTIONS = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (2, 2), (2, 1)]


def strict(spans, length):
    """The invariants [2/2]'s ``_validated_turn_spans`` enforces, restated here so [1/2] is checked on its own."""
    previous_end = 0
    for start, end in spans:
        assert type(start) is int and type(end) is int, spans
        assert previous_end <= start < end <= length, (spans, length)
        previous_end = end


async def episode(response_length, removed, inserted, *, flag=True, done_on=(2,)):
    tool = StepTool([0.5, 0.25, 1.0], done_on=done_on)
    loop = make_loop([tool], tool_call_turns=3, turn_level_rewards=flag, response_length=response_length,
                     removed_on_context=removed, inserted_on_context=inserted)
    return tool, loop, await run_episode(loop)


@pytest.mark.parametrize("removed,inserted", JUNCTIONS)
async def test_exported_spans_pass_the_strict_invariants_at_every_response_length(removed, inserted):
    boundaries = {"mid_generation": 0, "at_merge": 0, "complete": 0}
    _, _, unbounded = await episode(10**6, removed, inserted)
    full_len = len(unbounded.response_ids)
    for response_length in LENGTHS:
        tool, loop, out = await episode(response_length, removed, inserted)
        spans, rewards, mask = out.extra_fields["generated_turn_spans"], out.extra_fields["turn_rewards"], out.response_mask
        n = len(out.response_ids)
        assert 1 <= n <= response_length and len(mask) == n
        strict(spans, n)
        assert spans == mask_runs(mask), f"L={response_length}: spans are not the runs of generated tokens"
        assert rewards == out.turn_rewards and len(rewards) == len(spans)
        assert out.extra_fields["episode_done"] is (tool.calls == 3)

        out.reward_score = 2.0
        rm_scores = out.as_dict()["rm_scores"]
        assert torch.all(rm_scores[torch.tensor(mask) == 0] == 0), "rewards only on generated tokens"
        assert rm_scores.sum().item() == pytest.approx(sum(rewards) + 2.0), "no reward dropped by the clip"

        done = out.extra_fields["episode_done"]
        if done and mask[-1] == 0:  # the terminal observation was applied: the complete episode
            boundaries["complete"] += 1
            assert n == full_len and tool.calls == 3
        elif done:  # done, but the terminal merge hit the budget: the documented exception
            boundaries["at_merge"] += 1
            assert spans[-1][1] == n and rewards[-1] == 1.0, "the step reward and done survive, the observation does not"
        elif tool.calls < len(spans):  # the last assistant turn was cut by the budget before its tool ran
            boundaries["mid_generation"] += 1
            assert spans[-1][1] == n == response_length and rewards[-1] == 0.0
        # else: a non-terminal context merge hit the budget (n < response_length, not done)
    assert boundaries["mid_generation"] and boundaries["complete"], boundaries
    # the done-at-merge boundary needs the merge to grow the trajectory (inserted + observation > trimmed)
    assert (boundaries["at_merge"] > 0) is (inserted + OBS - removed > 0), boundaries


@pytest.mark.parametrize("removed,inserted", JUNCTIONS)
async def test_flag_off_exports_no_turn_metadata_at_any_response_length(removed, inserted):
    for response_length in LENGTHS:
        tool, loop, out = await episode(response_length, removed, inserted, flag=False)
        assert out.turn_rewards is None
        assert not {"generated_turn_spans", "turn_rewards", "episode_done"} & set(out.extra_fields)
        assert out.extra_fields["tool_rewards"] == [0.5, 0.25, 1.0][: tool.calls]
        out.reward_score = 2.0
        assert out.as_dict()["rm_scores"].tolist()[-1] == 2.0, "main's scalar placement on the last token"


async def test_a_terminal_observation_that_exactly_fills_the_budget_is_dropped_not_kept():
    """The documented rule is ``>=``: a merge whose last token would sit exactly at response_length - 1 is still
    dropped. Pin it so a later ``>`` change (which would keep an in-budget observation) is a visible decision."""
    full = TURNS * GEN + (TURNS - 1) * OBS
    tool, loop, exact = await episode(3 * GEN + 3 * OBS, 0, 0)  # 3 turns + 3 observations fill it exactly
    assert exact.extra_fields["episode_done"] is True and exact.response_mask[-1] == 1
    assert len(exact.response_ids) == 3 * GEN + 2 * OBS, "the terminal observation is not applied"
    tool, loop, roomy = await episode(3 * GEN + 3 * OBS + 1, 0, 0)
    assert roomy.response_mask[-OBS:] == [0] * OBS, "one more token of budget keeps it"
    assert full > len(roomy.response_ids)
