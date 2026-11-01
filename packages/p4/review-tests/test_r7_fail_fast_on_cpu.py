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
"""R7 (fix D2): the fail-fast validation on the REAL composed ``ppo_trainer`` config (not a hand-built dict), and
its equivalence with the condition under which the agent loop has no reward handles
(``RewardLoopManager.reward_loop_worker_handles``), including the legacy ``enable_resource_pool: null``."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl
from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.utils.config import _validate_turn_level_rewards_config, validate_config

CONFIG_DIR = str(Path(verl.__file__).resolve().parent / "trainer" / "config")  # the tree under test


def _real_config(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="ppo_trainer", overrides=list(overrides))


def _handles_are_none(cfg):
    """Evaluate the property's own logic on a stub manager: None means the trainer scores after rollout."""
    stub = SimpleNamespace(config=cfg, reward_loop_workers=["worker"])
    return RewardLoopManager.reward_loop_worker_handles.fget(stub) is None


@pytest.mark.parametrize(
    "flag,enable,pool,rejected",
    [(True, True, False, True), (True, True, True, False), (True, False, False, False), (False, True, False, False)],
)
def test_real_config_paths_exist_and_validation_matches_the_handles_condition(flag, enable, pool, rejected):
    cfg = _real_config(
        f"actor_rollout_ref.rollout.multi_turn.turn_level_rewards={flag}",
        f"reward.reward_model.enable={enable}",
        f"reward.reward_model.enable_resource_pool={pool}",
    )
    assert cfg.actor_rollout_ref.rollout.multi_turn.turn_level_rewards is flag
    if rejected:
        with pytest.raises(ValueError, match="needs the reward computed in the agent loop"):
            _validate_turn_level_rewards_config(cfg)
    else:
        _validate_turn_level_rewards_config(cfg)
    # the rejection fires exactly when the agent loop would have no reward handles
    assert (flag and _handles_are_none(cfg)) == rejected


def test_legacy_null_resource_pool_counts_as_colocated():
    cfg = _real_config(
        "actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True",
        "reward.reward_model.enable=True",
        "reward.reward_model.enable_resource_pool=null",
    )
    assert _handles_are_none(cfg)
    with pytest.raises(ValueError):
        _validate_turn_level_rewards_config(cfg)


def test_full_validate_config_reaches_the_new_check():
    cfg = _real_config(
        "actor_rollout_ref.rollout.multi_turn.turn_level_rewards=True",
        "reward.reward_model.enable=True",
        "reward.reward_model.enable_resource_pool=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "data.train_batch_size=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    )
    with pytest.raises(ValueError, match="turn_level_rewards"):
        validate_config(cfg, use_reference_policy=False, use_critic=False)


def test_flag_off_is_never_rejected_whatever_the_reward_setup():
    for enable, pool in [(True, False), (True, True), (False, False)]:
        cfg = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"multi_turn": {"turn_level_rewards": False}}},
                "reward": {"reward_model": {"enable": enable, "enable_resource_pool": pool}},
            }
        )
        _validate_turn_level_rewards_config(cfg)
