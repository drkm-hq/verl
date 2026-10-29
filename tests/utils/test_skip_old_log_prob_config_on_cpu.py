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

from verl.utils import config as config_module
from verl.utils.config import _hf_config_dropouts, _validate_skip_old_log_prob_config, validate_config

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
        ({"actor_rollout_ref.actor.strategy": "megatron"}, "actor.strategy=megatron"),
        ({"actor_rollout_ref.actor.strategy": "veomni"}, "actor.strategy=veomni"),
        ({"actor_rollout_ref.actor.strategy": "torchtitan"}, "actor.strategy=torchtitan"),
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


def test_megatron_engine_is_rejected_from_its_own_config_group():
    """The Megatron data path is untested with the skip (and dynamic context parallel drops the training
    outputs): the composed megatron config is rejected by strategy, with or without dynamic CP."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true",
                "data.train_batch_size=8",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.rollout.n=2",
            ],
        )
    assert config.actor_rollout_ref.actor.strategy == "megatron"
    with pytest.raises(ValueError, match="actor.strategy=megatron"):
        _validate_skip_old_log_prob_config(config)
    with pytest.raises(ValueError, match="actor.strategy=megatron"):
        _validate_skip_old_log_prob_config(
            _with(config, **{"actor_rollout_ref.actor.megatron.dynamic_context_parallel": True})
        )


def test_dropout_in_the_model_config_is_rejected(on_policy_config, monkeypatch):
    monkeypatch.setattr(config_module, "_hf_config_dropouts", lambda model_config: [("attention_dropout", 0.1)])
    with pytest.raises(ValueError, match="attention_dropout=0.1"):
        _validate_skip_old_log_prob_config(on_policy_config)
    # a dropout-free config passes; an unloadable config only warns (the shipped default path does not exist here)
    monkeypatch.setattr(config_module, "_hf_config_dropouts", lambda model_config: [])
    _validate_skip_old_log_prob_config(on_policy_config)
    monkeypatch.setattr(config_module, "_hf_config_dropouts", lambda model_config: None)
    _validate_skip_old_log_prob_config(on_policy_config)


def test_hf_config_dropouts_reads_a_saved_config(tmp_path):
    from transformers import Qwen3Config

    Qwen3Config(attention_dropout=0.1, num_hidden_layers=1, hidden_size=8, intermediate_size=8).save_pretrained(
        tmp_path / "with_dropout"
    )
    Qwen3Config(num_hidden_layers=1, hidden_size=8, intermediate_size=8).save_pretrained(tmp_path / "no_dropout")
    assert _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "with_dropout")})) == [
        ("attention_dropout", 0.1)
    ]
    assert _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "no_dropout")})) == []
    assert _hf_config_dropouts(OmegaConf.create({"path": str(tmp_path / "missing")})) is None
    assert _hf_config_dropouts(OmegaConf.create({"path": None})) is None
