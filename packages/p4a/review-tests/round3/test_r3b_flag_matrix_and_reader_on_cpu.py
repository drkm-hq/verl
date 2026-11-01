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
"""R3b (fix 0d4cd8ca): ``turn_level_rewards_enabled`` on absent / null / MISSING / struct / string / dict /
namespace / generated-yaml configs; every manager under {key absent, False, True, reward-only config}; identity and
mutation with dapo / gdpo off (the dataset's own object, as on main); the merge precedence; the rate-limited
manager's legacy ``__call__`` batch path under both flag states."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import turn_level_rewards_enabled as enabled
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

GENERATED = f"{verl.__path__[0]}/trainer/config/_generated_ppo_trainer.yaml"
PATH = "actor_rollout_ref.rollout.multi_turn.turn_level_rewards"


def cfg(value):
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"multi_turn": {"turn_level_rewards": value}}}})


def test_reader_defaults_to_false_on_every_absent_or_unresolvable_shape():
    for config in (
        OmegaConf.create({"reward": {}}),
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"multi_turn": None}}}),
        OmegaConf.create({"actor_rollout_ref": {"rollout": None}}),
        OmegaConf.create({"actor_rollout_ref": None}),
        cfg("???"),
        cfg(None),
        {"reward": {}},
        {},
        SimpleNamespace(reward={}),
        SimpleNamespace(actor_rollout_ref=None),
        None,
    ):
        assert enabled(config) is False, config
    struct = OmegaConf.create({"reward": {}})
    OmegaConf.set_struct(struct, True)
    assert enabled(struct) is False
    struct_off = cfg(False)
    OmegaConf.set_struct(struct_off, True)
    assert enabled(struct_off) is False and enabled(cfg(True)) is True
    assert enabled({"actor_rollout_ref": {"rollout": {"multi_turn": {"turn_level_rewards": True}}}}) is True
    assert enabled(SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(multi_turn=SimpleNamespace(turn_level_rewards=True))))) is True


def test_reader_on_the_generated_trainer_config_and_on_truthy_strings():
    generated = OmegaConf.load(GENERATED)
    assert isinstance(OmegaConf.select(generated, PATH), bool) and enabled(generated) is False
    OmegaConf.update(generated, PATH, True)
    assert enabled(generated) is True
    # LOW: a string value is read with bool(): "false" / "False" count as on, consistent with ToolAgentLoop's own
    # `if self.turn_level_rewards:`; the composed config holds a real bool, so only a quoted CLI override gets here
    assert enabled(cfg("false")) is True and enabled(cfg("False")) is True and enabled(cfg(0)) is False


DATASET = {"num_turns": 7, "tool_rewards": [123.0], "rollout_reward_scores": {"dataset": 1}, "keep": "me"}
LOOP = {"turn_scores": [], "tool_rewards": [2.0], "num_turns": "loop-field", "extra": 1}


def batch(n=1) -> DataProto:
    ntb = {
        "data_source": np.array(["toy"] * n),
        "reward_model": np.array([{"ground_truth": "x"}] * n, dtype=object),
        "__num_turns__": np.array([3] * n),
        "reward_scores": np.array([{"acc": 1.0}] * n, dtype=object),
    }
    for key, value in (("extra_info", DATASET), ("tool_extra_fields", LOOP)):
        column = np.empty(n, dtype=object)
        for i in range(n):
            column[i] = {**value, "row": i} if key == "extra_info" else dict(value)
        ntb[key] = column
    tensors = TensorDict({"responses": torch.tensor([[1, 2]] * n), "attention_mask": torch.tensor([[1, 1, 1]] * n)}, batch_size=[n])
    return DataProto(batch=tensors, non_tensor_batch=ntb)


def build(cls, compute_score, config_shape):
    raw = {"reward": {}, "trainer": {"experiment_name": "r3b"}}
    if config_shape in (True, False):
        raw["actor_rollout_ref"] = {"rollout": {"multi_turn": {"turn_level_rewards": config_shape}}}
    elif config_shape == "reward-only":
        raw = {"reward": {}, "trainer": {"experiment_name": "r3b"}}
    if cls is RateLimitedRewardManager:
        for attr in ("_semaphore", "_rpm_limiter", "_tpm_limiter"):
            setattr(cls, attr, None)
        cls._class_initialized = False
    return cls(OmegaConf.create(raw), SimpleNamespace(decode=lambda *args, **kwargs: "r"), compute_score)


@pytest.mark.parametrize("config_shape", ["absent", False, True, "reward-only"], ids=["absent", "off", "on", "reward-only"])
@pytest.mark.parametrize("cls", [NaiveRewardManager, DAPORewardManager, GDPORewardManager, RateLimitedRewardManager])
async def test_each_manager_contract_under_every_config_shape(cls, config_shape):
    seen = {}

    def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        seen["obj"] = extra_info
        extra_info["touched"] = True  # a callback that mutates what it was given
        num_turns = extra_info["num_turns"]
        return float(num_turns) if isinstance(num_turns, int | np.integer) else -1.0

    data = batch()
    stored = data.non_tensor_batch["extra_info"][0]
    result = await build(cls, compute_score, config_shape).run_single(data)
    got, on = seen["obj"], config_shape is True
    if on or cls is NaiveRewardManager:  # merged: loop fields override dataset keys, the rollout fields override both
        assert result["reward_score"] == 3.0 and got["tool_rewards"] == [2.0] and got["extra"] == 1 and got["keep"] == "me"
        assert got["rollout_reward_scores"] == {"acc": 1.0} and got is not stored and "touched" not in stored
    elif cls is RateLimitedRewardManager:  # tool fields merged (a loop field named num_turns wins, as on main), no rollout fields
        assert result["reward_score"] == -1.0
        assert got["tool_rewards"] == [2.0] and got["num_turns"] == "loop-field" and got["rollout_reward_scores"] == {"dataset": 1}
        assert got is not stored and "touched" not in stored
    else:  # dapo / gdpo: the dataset's own object; the callback's mutation lands on it, as on main
        assert result["reward_score"] == 7.0 and got is stored and stored["touched"] is True
        assert got["tool_rewards"] == [123.0] and "extra" not in got
        if cls is GDPORewardManager:
            assert stored["experiment_name"] == "r3b", "gdpo writes experiment_name into the dataset dict, as on main"


def test_rate_limited_legacy_batch_call_follows_the_same_contract_in_both_flag_states():
    """The legacy ``__call__(data, return_dict)`` path scores every row through ``run_single``."""
    for flag, expected_num_turns in ((False, 7), (True, 3)):
        seen = []

        def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
            seen.append(extra_info)
            return float(extra_info["num_turns"]) if not isinstance(extra_info["num_turns"], str) else -1.0

        loop_fields = dict(LOOP)
        loop_fields.pop("num_turns")
        data = batch(n=2)
        data.non_tensor_batch["tool_extra_fields"] = np.array([dict(loop_fields), dict(loop_fields)], dtype=object)
        out = build(RateLimitedRewardManager, compute_score, flag)(data, return_dict=True)
        assert out["reward_tensor"][:, 1].tolist() == [expected_num_turns] * 2 and len(seen) == 2  # last valid token
        assert [info["row"] for info in seen] == [0, 1] and all(info["tool_rewards"] == [2.0] for info in seen)
        assert all(("rollout_reward_scores" in info and info["rollout_reward_scores"] == {"acc": 1.0}) is flag for info in seen)
