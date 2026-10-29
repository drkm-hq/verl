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
"""The legacy ``RayPPOTrainer`` never implements ``actor.skip_old_log_prob_when_on_policy``. A run that builds it
with the flag set (``main_ppo_v0.py`` with the shipped default ``trainer.use_v1=true`` passes ``validate_config``)
must fail at construction, naming the flag and ``trainer.use_v1``, instead of silently keeping the old-log-prob
pass."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import _validate_skip_old_log_prob_config

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"


def test_legacy_trainer_refuses_the_flag_it_does_not_implement():
    overrides = [
        "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
        "data.train_batch_size=8",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.n=2",
        "algorithm.adv_estimator=grpo",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    assert config.trainer.use_v1  # the shipped default: the config-level validator accepts this config
    _validate_skip_old_log_prob_config(config)

    with pytest.raises(ValueError, match="trainer.use_v1=True") as excinfo:
        RayPPOTrainer(config=config, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
    assert "skip_old_log_prob_when_on_policy=True" in str(excinfo.value)

    # with the flag off the constructor gets past the check (and stops later on the empty role mapping, as before)
    config.actor_rollout_ref.actor.skip_old_log_prob_when_on_policy = False
    with pytest.raises(AssertionError, match="role_worker_mapping"):
        RayPPOTrainer(config=config, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)
