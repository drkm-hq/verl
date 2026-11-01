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
"""R9 (fix D1, both halves): the real ToolAgentLoop -> V0 ``_postprocess`` -> ``compute_advantage`` chain with
the flag OFF and ON, for ``grpo_vectorized``, a by-name estimator without ``**kwargs``, ``rloo`` and (where the
tree has it) ``turn_rloo``. Runs on [1/2] alone (no estimator, no kwarg hunk) and on [2/2]."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos as core_algos
from scripted_loop import StepTool, make_loop, run_episode
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_advantage

HAS_TURN_RLOO = "turn_rloo" in core_algos.ADV_ESTIMATOR_REGISTRY
RESPONSE_LENGTH = 24


def _internal(output, reward_score):
    """Right-pad one AgentLoopOutput the way _run_agent_loop does (prompt left-padded to 4)."""
    prompt = torch.tensor([[0] + output.prompt_ids])
    resp = output.response_ids + [0] * (RESPONSE_LENGTH - len(output.response_ids))
    mask = output.response_mask + [0] * (RESPONSE_LENGTH - len(output.response_mask))
    attn = [0] + [1] * len(output.prompt_ids) + [1] * len(output.response_ids) + [0] * (
        RESPONSE_LENGTH - len(output.response_ids)
    )
    return _InternalAgentLoopOutput(
        prompt_ids=prompt,
        response_ids=torch.tensor([resp]),
        input_ids=torch.cat([prompt, torch.tensor([resp])], dim=1),
        position_ids=torch.arange(prompt.size(1) + RESPONSE_LENGTH).unsqueeze(0),
        response_mask=torch.tensor([mask]),
        attention_mask=torch.tensor([attn]),
        reward_score=reward_score,
        turn_rewards=output.turn_rewards,
        num_turns=output.num_turns,
        metrics=AgentLoopMetrics(),
        extra_fields=dict(output.extra_fields),
    )


async def _batch(flag: bool):
    rows = []
    for i, rewards in enumerate([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]):
        loop = make_loop([StepTool(rewards, done_on=[2])], tool_call_turns=3, turn_level_rewards=flag)
        rows.append(_internal(await run_episode(loop), reward_score=float(i % 2)))
    worker = SimpleNamespace(
        reward_loop_worker_handles=None,
        rollout_config=SimpleNamespace(multi_turn=SimpleNamespace(turn_level_rewards=flag)),
    )
    batch = AgentLoopWorker._postprocess(worker, rows)
    batch.non_tensor_batch["uid"] = np.array(["p"] * 4, dtype=object)
    batch.batch["token_level_rewards"] = batch.batch["rm_scores"]
    return batch


def _plain_rloo(token_level_rewards, response_mask, index, config=None):
    return core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("name", ["grpo_vectorized", "plain_rloo", "rloo", "reinforce_plus_plus"])
async def test_estimators_without_turn_spans_survive_the_flag(monkeypatch, flag, name):
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "plain_rloo", _plain_rloo)
    batch = await _batch(flag)
    assert ("generated_turn_spans" in batch.non_tensor_batch) is flag, "column only under the flag"
    out = compute_advantage(batch, adv_estimator=name, config=SimpleNamespace(gamma=1.0, lam=1.0))
    assert out.batch["advantages"].shape == (4, RESPONSE_LENGTH)
    if flag:  # dense rows: the reward sum is the return, not just the outcome
        assert torch.allclose(batch.batch["rm_scores"].sum(-1), torch.tensor([2.0, 1.0, 1.0, 4.0]))
    else:
        assert torch.allclose(batch.batch["rm_scores"].sum(-1), torch.tensor([0.0, 1.0, 0.0, 1.0]))


@pytest.mark.skipif(not HAS_TURN_RLOO, reason="[1/2] has no turn_rloo")
@pytest.mark.parametrize("flag", [False, True])
async def test_turn_rloo_sees_spans_only_under_the_flag(flag):
    batch = await _batch(flag)
    rloo = compute_advantage(batch, adv_estimator="rloo", config=SimpleNamespace(gamma=1.0)).batch["advantages"]
    turn = compute_advantage(batch, adv_estimator="turn_rloo", config=SimpleNamespace(gamma=1.0)).batch[
        "advantages"
    ]
    if flag:
        assert not torch.allclose(turn, rloo), "spans reached turn_rloo: per-turn credit differs from RLOO"
        # episode 3: G = [4, 3, 2]; LOO baselines over the other three: turn 1 (2+1+1)/3, turn 3 (1+1+0)/3
        assert turn[3, 0].item() == pytest.approx(4.0 - 4.0 / 3) and turn[3, 10].item() == pytest.approx(2.0 - 2.0 / 3)
    else:
        torch.testing.assert_close(turn, rloo)  # no column, no spans: exactly RLOO
