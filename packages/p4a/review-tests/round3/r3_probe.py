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
"""Differential probe (not a test): builds the five reward-loop managers with the flag OFF (no key in the config)
and prints, per manager, the ``extra_info`` the callback received, whether it is the dataset's own dict object, and
the dataset dict after the call. Runs on main (no helper at all) and on the branch; ``test_r3a_*`` invokes it in a
subprocess with ``PYTHONPATH=<tree>`` for each tree and compares the two reports."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
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

DATASET = {"num_turns": 7, "tool_rewards": [123.0], "rollout_reward_scores": {"dataset": 1}, "keep": "me"}
LOOP = {"turn_scores": [], "tool_rewards": [2.0], "extra": 1}
MANAGERS = [NaiveRewardManager, DAPORewardManager, GDPORewardManager, RateLimitedRewardManager, RemoteRewardManager]


def batch() -> DataProto:
    ntb = {
        "data_source": np.array(["toy"]),
        "reward_model": np.array([{"ground_truth": "x"}], dtype=object),
        "__num_turns__": np.array([3]),
        "reward_scores": np.array([{"acc": 1.0}], dtype=object),
    }
    for key, value in (("extra_info", DATASET), ("tool_extra_fields", LOOP)):
        column = np.empty(1, dtype=object)
        column[0] = json.loads(json.dumps(value))
        ntb[key] = column
    tensors = TensorDict({"responses": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}, batch_size=[1])
    return DataProto(batch=tensors, non_tensor_batch=ntb)


def build(cls, compute_score):
    config = OmegaConf.create({"reward": {}, "trainer": {"experiment_name": "r3"}})  # no flag key: off
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "r")
    if cls is RemoteRewardManager:  # its __init__ starts Ray actors: in-process stand-in for the worker pool
        obj = object.__new__(cls)
        RewardManagerBase.__init__(obj, config, tokenizer, compute_score)
        obj.is_async_reward_score, obj.reward_router_address, obj.reward_model_tokenizer = False, None, None

        class _Handle:
            @staticmethod
            async def remote(**kwargs):
                return compute_score(**kwargs)

        obj.choose_reward_worker = lambda: SimpleNamespace(compute_score=_Handle)
        return obj
    if cls is RateLimitedRewardManager:
        for attr in ("_semaphore", "_rpm_limiter", "_tpm_limiter"):
            setattr(cls, attr, None)
        cls._class_initialized = False
    return cls(config, tokenizer, compute_score)


async def main():
    report = {}
    for cls in MANAGERS:
        seen = {}

        def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
            seen["obj"] = extra_info
            return 1.0

        data = batch()
        dataset_obj = data.non_tensor_batch["extra_info"][0]
        try:
            await build(cls, compute_score).run_single(data)
            error = None
        except Exception as e:  # noqa: BLE001 - the report carries the type
            error = type(e).__name__
        received = seen.get("obj")
        report[cls.__name__] = {
            "error": error,
            "received": received if isinstance(received, dict) else repr(received),
            "is_dataset_object": received is dataset_obj,
            "dataset_after": dataset_obj,
        }
    print(json.dumps(report, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else repr(o)))


if __name__ == "__main__":
    asyncio.run(main())
