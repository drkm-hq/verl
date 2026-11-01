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
"""Shared helpers for the round-2 adversarial review tests of contrib/p0-loss-horizon (reviewer-owned, never on the
branch). Run from the worktree with PYTHONPATH=<repo>/wt/p0 so ``verl`` is the branch.
The configs are built the way the launchers build them (compose ``ppo_trainer`` / ``ppo_megatron_trainer``, then
``validate_config``); the model path is a stub directory holding a tiny ``config.json`` (no snapshot needed)."""

import json
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl.trainer.config as trainer_config
from verl.utils.config import validate_config

CONFIG_DIR = os.path.dirname(trainer_config.__file__)
RESPONSE_LENGTH = 16
RL = f"data.max_response_length={RESPONSE_LENGTH}"
NORM = "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm"
ACTOR_KEY = r"actor_rollout_ref\.actor\.loss_scale_factor must be a finite positive number"
CRITIC_KEY = r"critic\.loss_scale_factor must be a finite positive number"
HORIZON_KEY = r"actor_rollout_ref\.rollout\.response_length must be a positive integer"

_HF_CONFIG = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "vocab_size": 128,
    "max_position_embeddings": 64,
}


def make_stub_model(path):
    (path / "config.json").write_text(json.dumps(_HF_CONFIG))
    return str(path)


def base_overrides(model_dir):
    return [
        f"actor_rollout_ref.model.path={model_dir}",
        f"critic.model.path={model_dir}",
        "+critic.model.load_tokenizer=false",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "critic.ppo_micro_batch_size_per_gpu=1",
    ]


def composed(model_dir, *overrides, config_name="ppo_trainer"):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=[*base_overrides(model_dir), *overrides])


def validated(model_dir, *overrides, use_critic=True, config_name="ppo_trainer", resolve_first=False):
    """``resolve_first`` mimics main_ppo_v0.TaskRunner.run and experimental/one_step_off_policy/main_ppo.py, which
    call ``OmegaConf.resolve(config)`` before ``validate_config`` (main_ppo.main validates the live interpolations)."""
    config = composed(model_dir, *overrides, config_name=config_name)
    if resolve_first:
        OmegaConf.resolve(config)
    validate_config(config, use_reference_policy=False, use_critic=use_critic)
    return config
