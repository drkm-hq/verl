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
"""Adversarial review: the 'V0 is identical in output' claim at the loss level, and the Megatron config path."""

import dataclasses
import os

import pytest
import torch
from review_common import RESPONSE_LENGTH, compose_config, make_dense_batch

from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.utils.losses import ppo_loss, value_loss

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")), reason="needs the model snapshot"
)
NORM = "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_v0_dense_inputs_give_the_same_loss_with_factor_none_and_response_length(kind):
    """Legacy trainer layout (``left_right_2_no_padding``: dense response fields padded to ``rollout.response_length``,
    ``max_response_len`` carried): ``loss_mask.shape[-1]`` is the full width, so the resolved factor changes nothing.
    This is the branch's 'V0 identical' claim, proven through the real loss functions with dense inputs."""
    config = compose_config(NORM)
    validate_config(config, use_reference_policy=False, use_critic=True)
    section = config.actor_rollout_ref.actor if kind == "actor" else config.critic
    resolved = omega_conf_to_dataclass(section)
    assert resolved.loss_scale_factor == RESPONSE_LENGTH
    legacy = dataclasses.replace(resolved, loss_scale_factor=None)  # what main's config carries

    data, model_output, _ = make_dense_batch()
    fn = ppo_loss if kind == "actor" else value_loss
    key = "log_probs" if kind == "actor" else "values"
    losses = []
    for cfg in (resolved, legacy):
        leaf = model_output.clone().requires_grad_(True)
        loss, _ = fn(cfg, {key: leaf}, data)
        loss.backward()
        losses.append((loss.detach(), leaf.grad.clone()))
    assert torch.equal(losses[0][0], losses[1][0])
    assert torch.equal(losses[0][1], losses[1][1])


def test_v0_width_narrower_than_response_length_would_change():
    """Boundary of the claim: if a legacy-layout batch were ever padded to less than ``rollout.response_length``
    (no in-tree path does; agent_loop.py:793-807 pads to it), the resolved factor would scale the loss by
    width / response_length. Documented so the owner can state the precondition in the PR body."""
    config = compose_config(NORM)
    validate_config(config, use_reference_policy=False, use_critic=True)
    resolved = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    legacy = dataclasses.replace(resolved, loss_scale_factor=None)
    data, model_output, _ = make_dense_batch(width=12)
    loss_resolved, _ = ppo_loss(resolved, {"log_probs": model_output.clone()}, data)
    loss_legacy, _ = ppo_loss(legacy, {"log_probs": model_output.clone()}, data)
    torch.testing.assert_close(loss_resolved * RESPONSE_LENGTH / 12, loss_legacy)


def test_megatron_trainer_config_is_resolved_too():
    """``ppo_megatron_trainer`` composes a McoreActorConfig; the resolution runs before the dataclass is built and
    the Megatron engine shares ``ppo_loss`` (engine_workers.py:687), so the same constant reaches it."""
    pytest.importorskip("verl.workers.config.engine")
    try:
        config = compose_config(NORM, config_name="ppo_megatron_trainer")
        validate_config(config, use_reference_policy=False, use_critic=True)
    except ModuleNotFoundError as e:  # megatron-core is not installable on macOS
        pytest.skip(f"megatron config path needs {e.name}")
    assert config.actor_rollout_ref.actor.strategy == "megatron"
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.actor_rollout_ref.actor).loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.critic).loss_scale_factor == RESPONSE_LENGTH
