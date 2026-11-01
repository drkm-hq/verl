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
"""Adversarial review: edge semantics of ``_resolve_loss_scale_factor`` through the public ``validate_config``."""

import dataclasses
import os

import pytest
import torch
from omegaconf import OmegaConf
from review_common import RESPONSE_LENGTH, UNEVEN, WHOLE, accumulate, compose_config, make_nested_batch

from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.metric import AggregationType

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")), reason="needs the model snapshot"
)
NORM = "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"


def _validated(*extra, use_critic=True):
    config = compose_config(*extra)
    validate_config(config, use_reference_policy=False, use_critic=use_critic)
    return config


def test_critic_explicit_factor_differs_from_actor_and_both_are_kept():
    config = _validated(NORM, "critic.loss_scale_factor=7")
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH
    assert config.critic.loss_scale_factor == 7
    assert omega_conf_to_dataclass(config.critic).loss_scale_factor == 7


def test_rollout_response_length_wins_over_data_max_response_length():
    """The horizon is ``rollout.response_length`` (the V0 padded width, agent_loop.py:793-807), not
    ``data.max_response_length``, when a user overrides the former."""
    config = _validated(NORM, "actor_rollout_ref.rollout.response_length=32")
    assert config.data.max_response_length == RESPONSE_LENGTH
    assert config.actor_rollout_ref.actor.loss_scale_factor == 32
    assert omega_conf_to_dataclass(config.critic).loss_scale_factor == 32


def test_critic_keeps_the_interpolation_and_use_critic_false_still_inherits():
    """NOTES.md claims the critic becomes a plain int 'in both cases'. It does not: when the actor is resolved
    first the critic's ``oc.select`` is left in place (it merely resolves to the actor's int), and with
    ``use_critic=False`` the critic section is skipped yet still inherits the actor's value through the
    interpolation. Harmless (the dataclass and ``OmegaConf.resolve`` both yield the int), but the note is wrong."""
    config = _validated(NORM)
    assert OmegaConf.is_interpolation(config.critic, "loss_scale_factor")
    assert config.critic.loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.critic).loss_scale_factor == RESPONSE_LENGTH
    OmegaConf.resolve(config)  # main_ppo.py:149 before the workers are built
    assert not OmegaConf.is_interpolation(config.critic, "loss_scale_factor")
    assert config.critic.loss_scale_factor == RESPONSE_LENGTH

    config = _validated(NORM, use_critic=False)
    assert config.critic.loss_scale_factor == RESPONSE_LENGTH


def test_validate_config_is_idempotent():
    config = _validated(NORM)
    validate_config(config, use_reference_policy=False, use_critic=True)
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH


@pytest.mark.parametrize("mode", ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean"])
@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_other_modes_ignore_the_factor_at_the_loss_level(kind, mode):
    """Stronger than the branch's config-level check: even if the factor WERE set under another mode (the critic
    inherits the actor's int when the actor runs Dr. GRPO and the critic does not), loss and gradient are
    bit-identical to factor None, because ``agg_loss`` reads it only in the norm branch (core_algos.py:1189)."""
    config = compose_config(f"actor_rollout_ref.actor.loss_agg_mode={mode}")
    validate_config(config, use_reference_policy=False, use_critic=True)
    section = config.actor_rollout_ref.actor if kind == "actor" else config.critic
    base = omega_conf_to_dataclass(section)
    assert base.loss_scale_factor is None
    forced = dataclasses.replace(base, loss_scale_factor=RESPONSE_LENGTH)

    data, model_output, seq_lens = make_nested_batch()
    for splits in (WHOLE, UNEVEN):
        loss_a, grad_a, _ = accumulate(kind, base, data, model_output, seq_lens, splits)
        loss_b, grad_b, _ = accumulate(kind, forced, data, model_output, seq_lens, splits)
        assert torch.equal(loss_a, loss_b)
        assert torch.equal(grad_a, grad_b)


def test_actor_norm_with_critic_token_mean_inherits_an_unused_factor():
    """Actor Dr. GRPO, critic explicitly token-mean: the critic's factor becomes 16 by inheritance. Under V1 this
    changes nothing (value_loss already aggregates with SUM because batch_num_tokens is set)."""
    config = _validated(NORM, "critic.loss_agg_mode=token-mean")
    critic_config = omega_conf_to_dataclass(config.critic)
    assert critic_config.loss_agg_mode == "token-mean"
    assert critic_config.loss_scale_factor == RESPONSE_LENGTH
    data, model_output, seq_lens = make_nested_batch()
    _, _, metrics = accumulate("critic", critic_config, data, model_output, seq_lens, WHOLE)
    assert metrics["critic/vf_loss"].aggregation == AggregationType.SUM


def test_metric_aggregation_flips_only_on_the_test_only_path():
    """losses.py:85-93 / :210-218 switch pg_loss / vf_loss metric aggregation to SUM when the factor is not None.
    The engine always sets batch_num_tokens (fsdp/transformer_impl.py:738, megatron:807), so SUM already applied;
    the flip is observable only with the test-only ``batch_num_tokens=None, global_batch_size=None`` inputs."""
    config = _validated(NORM)
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    data, model_output, seq_lens = make_nested_batch()
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    _, _, metrics = accumulate("actor", actor_config, data, model_output, seq_lens, WHOLE)
    assert metrics["actor/pg_loss"].aggregation == AggregationType.SUM
    none_config = dataclasses.replace(actor_config, loss_scale_factor=None)
    _, _, metrics = accumulate("actor", none_config, data, model_output, seq_lens, WHOLE)
    assert metrics["actor/pg_loss"].aggregation == AggregationType.MEAN
