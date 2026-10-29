# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Every reward manager hands the agent loop's ``tool_extra_fields`` to the custom reward function in
``extra_info`` (``generated_turn_spans`` / ``turn_rewards`` / ``episode_done`` under
``rollout.multi_turn.turn_level_rewards``), through ``RewardManagerBase.extra_info_for_reward_fn``, as a copy."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.experimental.reward_loop.reward_manager.remote import RemoteRewardManager

TOOL_EXTRA_FIELDS = {
    "turn_scores": [],
    "tool_rewards": [2.0],
    "generated_turn_spans": [(0, 2)],
    "turn_rewards": [2.0],
    "episode_done": True,
}
DATASET_EXTRA_INFO = {"split": "train", "index": 7}


def _batch() -> DataProto:
    """One sample, the way ``AgentLoopWorker._compute_score`` builds it for the reward loop."""
    return DataProto(
        batch=TensorDict(
            {"responses": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}, batch_size=[1]
        ),
        non_tensor_batch={
            "data_source": np.array(["toy"]),
            "reward_model": np.array([{"ground_truth": "x"}], dtype=object),
            "extra_info": np.array([dict(DATASET_EXTRA_INFO)], dtype=object),
            "tool_extra_fields": np.array([dict(TOOL_EXTRA_FIELDS)], dtype=object),
            "__num_turns__": np.array([3]),
        },
    )


def _manager(cls, compute_score):
    config = OmegaConf.create({"reward": {}, "trainer": {"experiment_name": "p4-check"}})
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "final response")
    if cls is RemoteRewardManager:
        # its __init__ starts Ray actors; run its run_single with an in-process stand-in for the worker pool
        manager = object.__new__(cls)
        RewardManagerBase.__init__(manager, config, tokenizer, compute_score)
        manager.is_async_reward_score = False
        manager.reward_router_address = None
        manager.reward_model_tokenizer = None

        class _Handle:
            @staticmethod
            async def remote(**kwargs):
                return compute_score(**kwargs)

        manager.choose_reward_worker = lambda: SimpleNamespace(compute_score=_Handle)
        return manager
    if cls is RateLimitedRewardManager:
        for attr in ("_semaphore", "_rpm_limiter", "_tpm_limiter"):
            setattr(cls, attr, None)
        cls._class_initialized = False
    return cls(config, tokenizer, compute_score)


@pytest.mark.parametrize(
    "cls", [NaiveRewardManager, DAPORewardManager, GDPORewardManager, RateLimitedRewardManager, RemoteRewardManager]
)
@pytest.mark.asyncio
async def test_custom_reward_function_sees_the_turn_level_metadata_under_every_manager(cls):
    captured = {}

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        captured.update(extra_info)
        return float(extra_info.get("episode_done", False))

    data = _batch()
    result = await _manager(cls, compute_score).run_single(data)

    assert result["reward_score"] == 1.0, f"{cls.__name__} did not deliver episode_done"
    for key in ("generated_turn_spans", "turn_rewards", "episode_done", "tool_rewards"):
        assert captured[key] == TOOL_EXTRA_FIELDS[key], key
    assert captured["num_turns"] == 3
    assert captured["rollout_reward_scores"] == {}
    assert captured["split"] == "train" and captured["index"] == 7
    # the reward function got a copy: the batch's own dicts are untouched
    assert data.non_tensor_batch["extra_info"][0] == DATASET_EXTRA_INFO
    assert data.non_tensor_batch["tool_extra_fields"][0] == TOOL_EXTRA_FIELDS


@pytest.mark.asyncio
async def test_list_valued_tool_fields_are_copied_so_a_reward_function_cannot_edit_the_row(caplog):
    def mutating(data_source, solution_str, ground_truth, extra_info, **kwargs):
        extra_info["generated_turn_spans"].append((9, 9))
        extra_info["turn_rewards"][0] = -1.0
        extra_info["new_key"] = 1
        return 1.0

    data = _batch()
    await _manager(NaiveRewardManager, mutating).run_single(data)
    assert data.non_tensor_batch["tool_extra_fields"][0] == TOOL_EXTRA_FIELDS
    assert data.non_tensor_batch["extra_info"][0] == DATASET_EXTRA_INFO


@pytest.mark.parametrize("cls", [DAPORewardManager, NaiveRewardManager])
@pytest.mark.asyncio
async def test_a_non_mapping_extra_info_is_passed_through_unchanged_with_one_warning(cls, caplog):
    """main's DAPO handed a ``str`` extra_info to the reward function untouched; so does every manager now."""
    from verl.experimental.reward_loop.reward_manager import base

    seen = []

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        seen.append(extra_info)
        return 1.0

    data = _batch()
    data.non_tensor_batch["extra_info"] = np.array(['{"split": "train"}'], dtype=object)
    base._NON_MAPPING_EXTRA_INFO_WARNED.clear()
    with caplog.at_level("WARNING", logger=base.logger.name):
        for _ in range(2):
            result = await _manager(cls, compute_score).run_single(data)
    assert result["reward_score"] == 1.0 and seen == ['{"split": "train"}', '{"split": "train"}']
    warnings_ = [r for r in caplog.records if "not a mapping" in r.getMessage()]
    assert len(warnings_) == 1 and "str" in warnings_[0].getMessage(), "one warning per type, naming it"


def test_extra_info_for_reward_fn_without_metadata_or_dataset_extra_info():
    data = _batch()
    data.non_tensor_batch.pop("tool_extra_fields")
    data.non_tensor_batch["extra_info"] = np.array([None], dtype=object)
    extra_info = RewardManagerBase.extra_info_for_reward_fn(data[0])
    assert extra_info == {"num_turns": 3, "rollout_reward_scores": {}}
