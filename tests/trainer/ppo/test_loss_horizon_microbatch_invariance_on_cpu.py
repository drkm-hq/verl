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
"""CPU tests: the engine loss functions are invariant to the micro-batch split on nested (V1) inputs, and
``validate_config`` resolves the "seq-mean-token-sum-norm" normalizer to ``rollout.response_length``.

The V1 trainer hands ``ppo_loss`` / ``value_loss`` nested (jagged) micro-batches. ``losses.py`` pads each
micro-batch to its own max response length and ``agg_loss`` divides "seq-mean-token-sum-norm" by
``loss_mask.shape[-1]`` when ``loss_scale_factor`` is None, so the horizon used to change with the split: with
the inputs below the accumulated actor loss was 8.996x (split [1, 2, 5]) and 48.2x (singles) the
whole-mini-batch loss and the critic loss 1.155x / 2.076x, with the gradients off accordingly, while the
equal-width split [4, 4] and the other four modes were exact. ``validate_config`` now resolves a null factor to
``actor_rollout_ref.rollout.response_length`` for the actor and the critic, which restores the documented
constant horizon. These tests build the worker configs the way the trainer does (compose ``ppo_trainer``,
``validate_config``, ``omega_conf_to_dataclass``) and drive the real loss functions. No GPU or model weights are
needed; ``validate_config`` reads the model config and tokenizer from the CI snapshot under ``~/models``.
"""

import functools
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

import verl.trainer.config as trainer_config
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.utils.losses import ppo_loss, value_loss

_CONFIG_DIR = os.path.dirname(trainer_config.__file__)
# config.json + tokenizer only, no weights: the snapshot tests/workers/config/test_model_config_on_cpu.py uses too
_MODEL_PATH = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")
_RESPONSE_LENGTH = 16  # the configured rollout.response_length; every response below is shorter
_PROMPT_LENS = [4, 3, 5, 2, 6, 3, 4, 5]
_RESP_LENS = [12, 1, 3, 7, 12, 2, 9, 5]
_BATCH = len(_RESP_LENS)
_MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]
_WHOLE = [list(range(_BATCH))]
_SPLITS = {
    "uneven_1_2_5": [[0], [1, 2], [3, 4, 5, 6, 7]],  # padded widths 12, 3, 12
    "even_4_4": [[0, 1, 2, 3], [4, 5, 6, 7]],  # padded widths 12, 12
    "singles": [[i] for i in range(_BATCH)],  # padded width = each response length
}

pytestmark = pytest.mark.skipif(not os.path.isdir(_MODEL_PATH), reason=f"needs the CI model snapshot at {_MODEL_PATH}")


@functools.cache
def _validated_config(mode, factor=None, loss_mode="vanilla", critic_mode=None):
    """Compose ``ppo_trainer`` and run ``validate_config`` on it, as ``main_ppo`` does before launching."""
    overrides = [
        f"data.max_response_length={_RESPONSE_LENGTH}",
        f"actor_rollout_ref.model.path={_MODEL_PATH}",
        f"critic.model.path={_MODEL_PATH}",
        f"actor_rollout_ref.actor.loss_agg_mode={mode}",
        f"actor_rollout_ref.actor.loss_scale_factor={'null' if factor is None else factor}",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={loss_mode}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "critic.ppo_micro_batch_size_per_gpu=1",
    ]
    if critic_mode is not None:
        overrides.append(f"critic.loss_agg_mode={critic_mode}")
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    validate_config(config, use_reference_policy=False, use_critic=True)
    return config


@functools.cache
def _worker_configs(mode, loss_mode="vanilla"):
    """The dataclasses the workers build from the validated config (engine_workers.py / v1/trainer_base.py)."""
    config = _validated_config(mode, loss_mode=loss_mode)
    return omega_conf_to_dataclass(config.actor_rollout_ref.actor), omega_conf_to_dataclass(config.critic)


def _nested(tensors):
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def _make_batch(seed=0):
    """A mini-batch as the V1 trainer stores it: nested per-row tensors plus the global batch info the engine
    assigns (``dp_size`` / ``batch_num_tokens`` in ``transformer_impl.py``, ``global_batch_size`` in the trainer)."""
    g = torch.Generator().manual_seed(seed)

    def rand(n):
        return torch.randn(n, generator=g)

    prompts = [torch.randint(10, 100, (p,), generator=g) for p in _PROMPT_LENS]
    responses = [torch.randint(10, 100, (r,), generator=g) for r in _RESP_LENS]
    response_mask = [torch.ones(r, dtype=torch.int64) for r in _RESP_LENS]
    response_mask[0][4:7] = 0  # a masked tool span inside the longest response
    data = TensorDict(
        {
            "prompts": _nested(prompts),
            "responses": _nested(responses),
            "response_mask": _nested(response_mask),
            "old_log_probs": _nested([-1.5 + 0.5 * rand(r) for r in _RESP_LENS]),
            "advantages": _nested([rand(r) for r in _RESP_LENS]),
            "values": _nested([rand(r) for r in _RESP_LENS]),
            "returns": _nested([rand(r) for r in _RESP_LENS]),
        },
        batch_size=[_BATCH],
    )
    num_tokens = int(sum(int(m.sum()) for m in response_mask))
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=num_tokens, global_batch_size=_BATCH)
    seq_lens = torch.tensor([p + r for p, r in zip(_PROMPT_LENS, _RESP_LENS, strict=True)])
    model_output = -1.2 + 0.4 * rand(int(seq_lens.sum()))  # one flat leaf (log-probs / values) for all rows
    return data, model_output, seq_lens


def _accumulate(kind, config, data, model_output, seq_lens, splits):
    """Sum of the per-micro-batch losses and the accumulated gradient on the flat model output, the way the
    engine's per-micro-batch ``loss.backward()`` produces them (no rescaling, ``fsdp/transformer_impl.py``)."""
    leaf = model_output.clone().requires_grad_(True)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens.cumsum(0)])
    total = torch.zeros(())
    for rows in splits:
        micro_batch = tu.index_select_tensor_dict(data, torch.tensor(rows))
        pieces = torch.cat([leaf[offsets[i] : offsets[i + 1]] for i in rows])
        row_offsets = torch.cat([torch.zeros(1, dtype=torch.long), seq_lens[rows].cumsum(0)])
        output = torch.nested.nested_tensor_from_jagged(pieces, row_offsets)
        if kind == "actor":
            loss, _ = ppo_loss(config, {"log_probs": output}, micro_batch)
        else:
            loss, _ = value_loss(config, {"values": output}, micro_batch)
        loss.backward()
        total = total + loss.detach()
    return total, leaf.grad.clone()


@pytest.mark.parametrize("split", list(_SPLITS))
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_loss_and_gradient_invariant_to_micro_batch_split(kind, mode, split):
    """Summing the per-micro-batch losses (and accumulating their gradients) gives the whole-mini-batch loss and
    gradient for every ``loss_agg_mode`` and every split, actor and critic. For "seq-mean-token-sum-norm" this
    holds because the resolved ``loss_scale_factor`` gives every micro-batch the same horizon; without the
    resolution the uneven splits fail (loss ratios 8.996x / 48.2x actor, 1.155x / 2.076x critic)."""
    actor_config, critic_config = _worker_configs(mode)
    config = actor_config if kind == "actor" else critic_config
    data, model_output, seq_lens = _make_batch()

    whole_loss, whole_grad = _accumulate(kind, config, data, model_output, seq_lens, _WHOLE)
    split_loss, split_grad = _accumulate(kind, config, data, model_output, seq_lens, _SPLITS[split])

    torch.testing.assert_close(split_loss, whole_loss)
    torch.testing.assert_close(split_grad, whole_grad)


@pytest.mark.parametrize("split", ["whole", *_SPLITS])
@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_seq_mean_token_sum_norm_horizon_is_response_length(kind, split):
    """The Dr. GRPO normalizer is the configured ``rollout.response_length`` (16 here), not the padded width of
    the micro-batch at hand (12, 3 or less): "seq-mean-token-sum-norm" equals "seq-mean-token-sum" divided by
    ``response_length`` for every micro-batch, in loss and gradient."""
    index = 0 if kind == "actor" else 1
    norm_config = _worker_configs("seq-mean-token-sum-norm")[index]
    sum_config = _worker_configs("seq-mean-token-sum")[index]
    data, model_output, seq_lens = _make_batch()
    splits = _WHOLE if split == "whole" else _SPLITS[split]

    norm_loss, norm_grad = _accumulate(kind, norm_config, data, model_output, seq_lens, splits)
    sum_loss, sum_grad = _accumulate(kind, sum_config, data, model_output, seq_lens, splits)

    torch.testing.assert_close(norm_loss, sum_loss / _RESPONSE_LENGTH)
    torch.testing.assert_close(norm_grad, sum_grad / _RESPONSE_LENGTH)


def test_validate_config_resolves_null_factor_to_response_length():
    """``loss_scale_factor: null`` under "seq-mean-token-sum-norm" becomes ``rollout.response_length`` for the
    actor and the critic, and the worker dataclasses carry the resolved constant."""
    config = _validated_config("seq-mean-token-sum-norm")
    assert config.actor_rollout_ref.actor.loss_scale_factor == _RESPONSE_LENGTH
    assert config.critic.loss_scale_factor == _RESPONSE_LENGTH

    actor_config, critic_config = _worker_configs("seq-mean-token-sum-norm")
    assert actor_config.loss_scale_factor == _RESPONSE_LENGTH
    assert critic_config.loss_scale_factor == _RESPONSE_LENGTH


def test_validate_config_keeps_an_explicit_factor():
    config = _validated_config("seq-mean-token-sum-norm", factor=4096)
    assert config.actor_rollout_ref.actor.loss_scale_factor == 4096
    assert config.critic.loss_scale_factor == 4096


@pytest.mark.parametrize("mode", [mode for mode in _MODES if mode != "seq-mean-token-sum-norm"])
def test_validate_config_leaves_other_modes_untouched(mode):
    config = _validated_config(mode)
    assert config.actor_rollout_ref.actor.loss_scale_factor is None
    assert config.critic.loss_scale_factor is None


def test_validate_config_resolves_the_critic_on_its_own():
    """The critic inherits the actor's mode by default but may select Dr. GRPO normalization alone."""
    config = _validated_config("token-mean", critic_mode="seq-mean-token-sum-norm")
    assert config.actor_rollout_ref.actor.loss_scale_factor is None
    assert config.critic.loss_scale_factor == _RESPONSE_LENGTH


@pytest.mark.parametrize(
    "loss_mode",
    [
        pytest.param(
            "geo_mean",
            marks=pytest.mark.xfail(
                strict=True,
                reason="geo_mean takes a micro-batch-local torch.mean and bypasses agg_loss (issue #7913, PR #7916)",
            ),
        ),
        pytest.param(
            "kl_cov",
            marks=pytest.mark.xfail(
                strict=True, reason="kl_cov selects its top-k covariance tokens per micro-batch (issue #7829, PR #7830)"
            ),
        ),
    ],
)
def test_known_non_invariant_policy_losses(loss_mode):
    """Policy losses known not to be micro-batch invariant, kept as strict xfails so the suite flips when their
    fixes land. Not touched here."""
    actor_config, _ = _worker_configs("token-mean", loss_mode=loss_mode)
    data, model_output, seq_lens = _make_batch()

    whole_loss, whole_grad = _accumulate("actor", actor_config, data, model_output, seq_lens, _WHOLE)
    split_loss, split_grad = _accumulate("actor", actor_config, data, model_output, seq_lens, _SPLITS["uneven_1_2_5"])

    torch.testing.assert_close(split_loss, whole_loss)
    torch.testing.assert_close(split_grad, whole_grad)
