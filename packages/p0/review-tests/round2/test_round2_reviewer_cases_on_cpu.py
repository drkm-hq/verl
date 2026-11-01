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
"""Round 2 (the submitter's P0 findings) re-attacked through the public ``validate_config`` on e0a7eb22: the explicit
factor, the inferred horizon, which key the error names, and the launcher orders that reach the resolver."""

import logging

import pytest
from round2_common import ACTOR_KEY, CRITIC_KEY, HORIZON_KEY, NORM, RESPONSE_LENGTH, RL, validated

from verl.utils.config import omega_conf_to_dataclass, validate_config

FACTOR = "actor_rollout_ref.actor.loss_scale_factor"


# N1: the submitter's nan / +inf plus the spellings a sign check, a float check or an isfinite check alone would let through.
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.0", "0.0", "1e400", "1e-400", "false", "[16]", "'16'"])
def test_explicit_factor_rejected_naming_the_actor_key(model_dir, value):
    with pytest.raises(ValueError, match=ACTOR_KEY) as e:
        validated(model_dir, NORM, RL, f"{FACTOR}={value}")
    assert "got " in str(e.value)


# N2: the submitter's 0 / negative, plus the float, bool and string spellings Hydra can deliver.
@pytest.mark.parametrize("value", ["0", "-16", "2.5", "true", "'2048'"])
def test_inferred_horizon_rejected_naming_the_rollout_key(model_dir, value):
    with pytest.raises(ValueError, match=HORIZON_KEY) as e:
        validated(model_dir, NORM, RL, f"actor_rollout_ref.rollout.response_length={value}")
    assert "got " in str(e.value) and f"set {FACTOR} explicitly" in str(e.value)


def test_data_max_response_length_zero_is_the_same_rejection(model_dir):
    """The common user spelling (``data.max_response_length=0``) reaches the resolver through ``oc.select``."""
    with pytest.raises(ValueError, match=HORIZON_KEY):
        validated(model_dir, NORM, "data.max_response_length=0")


def test_critic_explicit_factor_names_the_critic_key(model_dir):
    with pytest.raises(ValueError, match=CRITIC_KEY) as e:
        validated(model_dir, NORM, RL, "critic.loss_scale_factor=nan")
    assert "got nan" in str(e.value)


def test_critic_alone_in_norm_mode_names_the_critic_key(model_dir):
    with pytest.raises(ValueError, match=CRITIC_KEY):
        validated(model_dir, RL, "critic.loss_agg_mode=seq-mean-token-sum-norm", "critic.loss_scale_factor=-1")


def test_actor_is_named_before_the_critic_when_both_are_invalid(model_dir):
    with pytest.raises(ValueError, match=ACTOR_KEY):
        validated(model_dir, NORM, RL, f"{FACTOR}=inf", "critic.loss_scale_factor=nan")


def test_unused_critic_is_not_validated(model_dir):
    """``use_critic=False``: the critic section is skipped (its factor is never consumed); the actor is resolved."""
    config = validated(model_dir, NORM, RL, "critic.loss_scale_factor=nan", use_critic=False)
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH


@pytest.mark.parametrize("resolve_first", [False, True], ids=["main_ppo-order", "v0-and-one_step_off-order"])
def test_null_factor_resolved_in_both_launcher_orders(model_dir, resolve_first):
    config = validated(model_dir, NORM, RL, resolve_first=resolve_first)
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH
    assert config.critic.loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.actor_rollout_ref.actor).loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.critic).loss_scale_factor == RESPONSE_LENGTH


@pytest.mark.parametrize("resolve_first", [False, True], ids=["main_ppo-order", "v0-and-one_step_off-order"])
def test_explicit_factor_kept_and_inherited_in_both_launcher_orders(model_dir, resolve_first):
    config = validated(model_dir, NORM, RL, f"{FACTOR}=4096", resolve_first=resolve_first)
    assert config.actor_rollout_ref.actor.loss_scale_factor == 4096
    assert config.critic.loss_scale_factor == 4096


def test_resolve_first_order_still_rejects(model_dir):
    with pytest.raises(ValueError, match=ACTOR_KEY):
        validated(model_dir, NORM, RL, f"{FACTOR}=nan", resolve_first=True)


def test_legacy_trainer_flag_does_not_bypass_the_resolver(model_dir):
    config = validated(model_dir, NORM, RL, "trainer.use_v1=false")
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH


def test_megatron_trainer_config_resolves_and_rejects(model_dir):
    config = validated(model_dir, NORM, RL, config_name="ppo_megatron_trainer")
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH
    assert config.critic.loss_scale_factor == RESPONSE_LENGTH
    with pytest.raises(ValueError, match=ACTOR_KEY):
        validated(model_dir, NORM, RL, f"{FACTOR}=nan", config_name="ppo_megatron_trainer")
    with pytest.raises(ValueError, match=HORIZON_KEY):
        validated(model_dir, NORM, "data.max_response_length=-1", config_name="ppo_megatron_trainer")


def test_second_validation_accepts_the_resolved_integer(model_dir):
    """main_ppo.main validates, then main_ppo_v0.TaskRunner.run validates the same config again."""
    config = validated(model_dir, NORM, RL)
    validate_config(config, use_reference_policy=False, use_critic=True)
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH


def test_unset_response_length_warns_and_keeps_the_old_fallback(model_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="verl.utils.config"):
        config = validated(model_dir, NORM, RL, "actor_rollout_ref.rollout.response_length=null")
    assert config.actor_rollout_ref.actor.loss_scale_factor is None
    assert omega_conf_to_dataclass(config.actor_rollout_ref.actor).loss_scale_factor is None
    assert "response_length is unset" in caplog.text
