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
"""Adversarial review: what the validate_config placement does NOT cover.

Every test here PASSES on the branch: each one documents a path on which the resolved factor never arrives, so
the loss is still normalized by the micro-batch's own padded width. They are evidence for the placement discussion
in REVIEW.md, not regressions of the branch.
"""

import os

import pytest
import torch
from review_common import UNEVEN, WHOLE, accumulate, compose_config, make_nested_batch

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, CriticConfig

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")), reason="needs the model snapshot"
)


def test_worker_config_built_without_validate_config_keeps_the_micro_batch_horizon():
    """A launcher that composes ``ppo_trainer`` and builds the worker dataclasses without calling
    ``validate_config`` (recipe TaskRunners such as flowrl/entropy/atropos, custom scripts, library users) gets
    ``loss_scale_factor=None`` and the split-dependent loss the PR fixes. Same inputs as the branch test:
    loss ratio 8.996x for split [1, 2, 5]."""
    config = compose_config("actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm")
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)  # engine_workers.py:633 shape
    assert actor_config.loss_scale_factor is None

    data, model_output, seq_lens = make_nested_batch()
    whole_loss, whole_grad, _ = accumulate("actor", actor_config, data, model_output, seq_lens, WHOLE)
    split_loss, split_grad, _ = accumulate("actor", actor_config, data, model_output, seq_lens, UNEVEN)

    ratio = (split_loss / whole_loss).item()
    assert abs(ratio - 8.996) < 0.01, ratio  # the bug is intact on this path
    assert not torch.allclose(split_grad, whole_grad)


def test_dataclass_defaults_are_unchanged():
    """The fix lives outside the dataclasses: ``ActorConfig`` / ``CriticConfig`` constructed directly (unit tests,
    library users, the fully_async rollouter's own config handling) keep ``loss_scale_factor=None`` under the
    Dr. GRPO mode, so ``agg_loss`` still falls back to ``loss_mask.shape[-1]`` there."""
    actor = ActorConfig(
        strategy="fsdp", rollout_n=1, loss_agg_mode="seq-mean-token-sum-norm", ppo_micro_batch_size_per_gpu=1
    )
    critic = CriticConfig(strategy="fsdp", loss_agg_mode="seq-mean-token-sum-norm", ppo_micro_batch_size_per_gpu=1)
    assert actor.loss_scale_factor is None
    assert critic.loss_scale_factor is None


def test_hydra_default_is_still_null_in_the_generated_yaml():
    """``_generated_ppo_trainer.yaml`` (what ``python -m verl.trainer.main_ppo --cfg job`` and the docs show)
    still says ``loss_scale_factor: null``; the resolution is invisible until ``validate_config`` runs. A user
    reading the generated config cannot tell what normalizer a run will use."""
    config = compose_config("actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm")
    assert config.actor_rollout_ref.actor.loss_scale_factor is None
    assert config.critic.loss_scale_factor is None
