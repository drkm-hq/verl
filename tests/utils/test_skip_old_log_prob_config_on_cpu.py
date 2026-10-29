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
"""Config validation of ``actor.skip_old_log_prob_when_on_policy``: the skip is exact only for one
optimizer step per sampled batch at the sampling weights with nothing reading ``old_log_probs``
before the update; every other configuration must be rejected loudly, naming the condition."""

import copy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.utils.config import _validate_skip_old_log_prob_config, validate_config

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"


@pytest.fixture(scope="module")
def on_policy_config():
    """The shipped ppo_trainer config with the flag on and the single-mini-batch shape."""
    overrides = [
        "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
        "data.train_batch_size=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.n=2",
        "algorithm.adv_estimator=grpo",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="ppo_trainer", overrides=overrides)


def _with(config, **updates):
    config = copy.deepcopy(config)
    for key, value in updates.items():
        OmegaConf.update(config, key.replace("__", "."), value, force_add=True)
    return config


def test_single_mini_batch_on_policy_config_is_accepted(on_policy_config):
    _validate_skip_old_log_prob_config(on_policy_config)
    # the shipped default rollout_correction group is metrics-only and therefore compatible
    assert on_policy_config.algorithm.rollout_correction.rollout_is is None
    # and the check is wired into the entry-point validation
    validate_config(on_policy_config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"actor_rollout_ref.actor.ppo_epochs": 2}, "actor.ppo_epochs=2"),
        (
            {"actor_rollout_ref.actor.ppo_mini_batch_size": 4},
            "actor.ppo_mini_batch_size=4 != data.train_batch_size / parameter_sync_step = 8",
        ),
        ({"actor_rollout_ref.actor.shuffle": True}, "actor.shuffle=True"),
        ({"actor_rollout_ref.actor.calculate_sum_pi_squared": True}, "actor.calculate_sum_pi_squared=True"),
        ({"algorithm.use_kl_in_reward": True}, "algorithm.use_kl_in_reward=True"),
        ({"algorithm.rollout_correction.bypass_mode": True}, "rollout_correction has bypass_mode"),
        ({"algorithm.rollout_correction.rollout_is": "token"}, "rollout_correction has bypass_mode"),
        ({"algorithm.rollout_correction.rollout_rs": "token"}, "rollout_correction has bypass_mode"),
        ({"algorithm.adv_estimator": "optimal_token_baseline"}, "adv_estimator=optimal_token_baseline"),
        ({"algorithm.adv_estimator": "tir_optimal_token_baseline"}, "adv_estimator=tir_optimal_token_baseline"),
        ({"trainer.v1.trainer_mode": "separate_async"}, "trainer_mode=separate_async"),
        ({"trainer.use_v1": False}, "trainer.use_v1=False"),
        ({"distillation.enabled": True}, "distillation.enabled=True"),
        ({"actor_rollout_ref.actor.engine.router_replay.mode": "R2"}, "router_replay.mode=R2"),
    ],
)
def test_each_inexact_condition_is_rejected_by_name(on_policy_config, updates, expected):
    config = _with(on_policy_config, **updates)
    with pytest.raises(ValueError, match="requires a strictly on-policy update") as excinfo:
        _validate_skip_old_log_prob_config(config)
    assert expected in str(excinfo.value)


def test_every_violated_condition_is_listed_at_once(on_policy_config):
    config = _with(
        on_policy_config, **{"actor_rollout_ref.actor.ppo_epochs": 3, "actor_rollout_ref.actor.shuffle": True}
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_skip_old_log_prob_config(config)
    message = str(excinfo.value)
    assert "actor.ppo_epochs=3" in message and "actor.shuffle=True" in message


def test_separate_async_sample_batch_uses_parameter_sync_step(on_policy_config):
    """train_batch_size is spread over parameter_sync_step local updates: the mini-batch condition is
    stated against the per-update sampled batch, not the global train_batch_size."""
    config = _with(
        on_policy_config,
        **{"trainer.v1.trainer_mode": "separate_async", "trainer.v1.separate_async.parameter_sync_step": 4},
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_skip_old_log_prob_config(config)
    assert "actor.ppo_mini_batch_size=8 != data.train_batch_size / parameter_sync_step = 2" in str(excinfo.value)


def test_flag_off_never_validates(on_policy_config):
    config = _with(
        on_policy_config,
        **{"actor_rollout_ref.actor.skip_old_log_prob_when_on_policy": False, "actor_rollout_ref.actor.ppo_epochs": 2},
    )
    _validate_skip_old_log_prob_config(config)
