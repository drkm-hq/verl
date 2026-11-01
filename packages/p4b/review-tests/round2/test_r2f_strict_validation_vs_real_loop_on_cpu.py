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
"""R2f (the key risk of the round): the strict span validation that replaced the round-1 past-end tolerance,
driven by the REAL ``ToolAgentLoop`` over every response budget (scripted merges that trim / insert 0-2 junction
tokens) through both trainer paths: V0 ``_postprocess`` -> ``compute_advantage("turn_rloo")`` on rows padded to
``response_length``, and the V1 ``extra_fields`` lift -> ``turn_rloo`` on ``as_dict`` rows; plus the real Qwen
tokenizer trajectories at every budget in three paddings. A legitimately truncated trajectory must never raise.

Uses ``packages/p4/review-tests/scripted_loop.py`` and the p4a round-2 Qwen harness.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

PACKAGES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGES / "p4" / "review-tests"))
sys.path.insert(0, str(PACKAGES / "p4a" / "review-tests" / "round2"))

import verl.trainer.ppo.v1.trainer_base as trainer_base_module  # noqa: E402
from scripted_loop import GEN, OBS, StepTool, make_loop, run_episode  # noqa: E402
from test_r2c_real_tokenizer_budget_sweep_on_cpu import MODEL_PATH, StepTool as QwenTool  # noqa: E402
from test_r2c_real_tokenizer_budget_sweep_on_cpu import make_loop as make_qwen_loop  # noqa: E402
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput  # noqa: E402
from verl.trainer.ppo.core_algos import compute_turn_rloo_advantage  # noqa: E402
from verl.trainer.ppo.ray_trainer import compute_advantage  # noqa: E402
from verl.trainer.ppo.v1.trainer_base import PPOTrainer  # noqa: E402

PATTERNS = [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
LENGTHS = range(1, 4 * GEN + 3 * OBS + 4)
JUNCTIONS = [(0, 0), (2, 0), (0, 2), (1, 1), (2, 2)]
PROMPT = [{"role": "user", "content": "Place the part."}]


def internal(out, reward_score, width):
    resp = out.response_ids + [0] * (width - len(out.response_ids))
    mask = out.response_mask + [0] * (width - len(out.response_mask))
    prompt = torch.tensor([[0] + out.prompt_ids])
    attn = [0] + [1] * len(out.prompt_ids) + [1] * len(out.response_ids) + [0] * (width - len(out.response_ids))
    return _InternalAgentLoopOutput(
        prompt_ids=prompt, response_ids=torch.tensor([resp]), input_ids=torch.cat([prompt, torch.tensor([resp])], 1),
        position_ids=torch.arange(prompt.size(1) + width).unsqueeze(0), response_mask=torch.tensor([mask]),
        attention_mask=torch.tensor([attn]), reward_score=reward_score, turn_rewards=out.turn_rewards,
        num_turns=out.num_turns, metrics=AgentLoopMetrics(), extra_fields=dict(out.extra_fields),
    )


async def episodes(response_length, removed, inserted):
    outs = []
    for rewards in PATTERNS:
        loop = make_loop([StepTool(rewards, done_on=[2])], tool_call_turns=3, response_length=response_length,
                         removed_on_context=removed, inserted_on_context=inserted)
        outs.append(await run_episode(loop))
    return outs


@pytest.mark.parametrize("removed,inserted", JUNCTIONS)
async def test_v0_postprocess_to_turn_rloo_never_raises_on_a_truncated_trajectory(removed, inserted):
    worker = SimpleNamespace(reward_loop_worker_handles=None,
                             rollout_config=SimpleNamespace(multi_turn=SimpleNamespace(turn_level_rewards=True)))
    for response_length in LENGTHS:
        outs = await episodes(response_length, removed, inserted)
        batch = AgentLoopWorker._postprocess(worker, [internal(o, float(i % 2), response_length) for i, o in enumerate(outs)])
        batch.non_tensor_batch["uid"] = np.array(["p"] * 4, dtype=object)
        batch.batch["token_level_rewards"] = batch.batch["rm_scores"]
        out = compute_advantage(batch, adv_estimator="turn_rloo", config=SimpleNamespace(gamma=1.0, lam=1.0))
        adv, ret, mask = out.batch["advantages"], out.batch["returns"], out.batch["response_mask"]
        assert adv.shape == (4, response_length) and torch.isfinite(adv).all()
        assert torch.all(adv[mask == 0] == 0) and torch.all(ret[mask == 0] == 0)
        for i, o in enumerate(outs):  # G[i, 0] is the whole row's reward sum (gamma 1)
            first = o.extra_fields["generated_turn_spans"][0][0]
            assert ret[i, first].item() == pytest.approx(batch.batch["rm_scores"][i].sum().item())


class _Trainer(PPOTrainer):
    def on_sample_end(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass


class _Meta:
    def __init__(self, keys):
        self.keys, self.partition_id = keys, "train"

    def __len__(self):
        return len(self.keys)


def run_v1(monkeypatch, outs, reward_scores):
    rows = []
    for out, score in zip(outs, reward_scores, strict=True):
        out.reward_score = score
        d = out.as_dict()
        rows.append((d["response_mask"].float(), d["rm_scores"], out.extra_fields))
    stored, metrics = {}, {}

    def kv_batch_get(keys, partition_id, select_fields):
        cols = {
            "uid": NonTensorStack.from_list([NonTensorData("p") for _ in rows]),
            "response_mask": torch.nested.as_nested_tensor([m for m, _, _ in rows], layout=torch.jagged),
            "rm_scores": torch.nested.as_nested_tensor([s for _, s, _ in rows], layout=torch.jagged),
            "extra_fields": NonTensorStack.from_list([NonTensorData(e) for *_, e in rows]),
        }
        return TensorDict({k: v for k, v in cols.items() if k in select_fields}, batch_size=[len(rows)])

    def kv_batch_put(keys, partition_id, fields):
        stored.update(fields)
        return _Meta(keys)

    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", kv_batch_get)
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", kv_batch_put)
    trainer = object.__new__(_Trainer)
    trainer.config = OmegaConf.create({"algorithm": {"adv_estimator": "turn_rloo", "gamma": 1.0, "lam": 1.0,
                                                     "use_kl_in_reward": False}, "actor_rollout_ref": {"rollout": {"n": 4}}})
    trainer._compute_advantage(_Meta([f"p_0_{i}" for i in range(len(rows))]), metrics=metrics)
    return stored, metrics


@pytest.mark.parametrize("removed,inserted", JUNCTIONS)
async def test_v1_lift_to_turn_rloo_never_raises_on_a_truncated_trajectory(monkeypatch, removed, inserted):
    for response_length in LENGTHS:
        outs = await episodes(response_length, removed, inserted)
        stored, metrics = run_v1(monkeypatch, outs, [0.0, 1.0, 0.0, 1.0])
        assert metrics["adv_estimator/rows_with_turn_spans"] == 4
        adv = stored["advantages"]
        assert adv.is_nested
        for i, o in enumerate(outs):
            row = adv[i]
            assert row.shape == (len(o.response_ids),) and torch.isfinite(row).all()
            assert torch.all(row[torch.tensor(o.response_mask) == 0] == 0)


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason=f"{MODEL_PATH} is not available")
async def test_real_qwen_trajectories_pass_turn_rloo_at_every_budget_in_three_paddings():
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(str(MODEL_PATH))
    full = await make_qwen_loop(tokenizer, QwenTool([0.25, 1.0], True), 4096).run({}, raw_prompt=PROMPT)
    for response_length in range(1, len(full.response_ids) + 3):
        out = await make_qwen_loop(tokenizer, QwenTool([0.25, 1.0], True), response_length).run({}, raw_prompt=PROMPT)
        out.reward_score = 1.0
        d = out.as_dict()
        spans = np.empty(1, dtype=object)
        spans[0] = out.extra_fields["generated_turn_spans"]
        for width in (len(out.response_ids), response_length, 4096):
            rewards, mask = torch.zeros(1, width), torch.zeros(1, width)
            rewards[0, : d["rm_scores"].numel()] = d["rm_scores"]
            mask[0, : d["response_mask"].numel()] = d["response_mask"].float()
            adv, ret = compute_turn_rloo_advantage(rewards, mask, np.array(["g"], dtype=object), turn_spans=spans)
            assert torch.isfinite(adv).all() and torch.all(adv[mask == 0] == 0)
            assert ret[0, spans[0][0][0]].item() == pytest.approx(rewards.sum().item())
