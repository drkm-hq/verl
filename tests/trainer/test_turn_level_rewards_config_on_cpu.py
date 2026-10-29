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

"""``rollout.multi_turn.turn_level_rewards`` is rejected with a colocated reward model, which would overwrite the
dense per-turn ``rm_scores`` row after rollout."""

import pytest
from omegaconf import OmegaConf

from verl.utils.config import _validate_turn_level_rewards_config


def _config(turn_level_rewards: bool, rm_enable: bool, rm_resource_pool: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"multi_turn": {"enable": True, "turn_level_rewards": turn_level_rewards}}
            },
            "reward": {"reward_model": {"enable": rm_enable, "enable_resource_pool": rm_resource_pool}},
        }
    )


@pytest.mark.parametrize(
    "turn_level_rewards,rm_enable,rm_resource_pool",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_accepted_combinations(turn_level_rewards, rm_enable, rm_resource_pool):
    _validate_turn_level_rewards_config(_config(turn_level_rewards, rm_enable, rm_resource_pool))


def test_colocated_reward_model_is_rejected():
    with pytest.raises(ValueError, match="turn_level_rewards=True needs the reward computed in the agent loop"):
        _validate_turn_level_rewards_config(_config(True, True, False))


def test_config_without_multi_turn_section_is_accepted():
    _validate_turn_level_rewards_config(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {}}, "reward": {"reward_model": {"enable": True}}})
    )
