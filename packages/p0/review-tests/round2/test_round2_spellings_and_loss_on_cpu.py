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
"""Round 2: the spellings Hydra / OmegaConf can deliver for the factor and the horizon (interpolations, env strings,
floats, denormals, huge ints), and whether an accepted factor is really consumed by the V1 loss functions."""

import os
import sys
from pathlib import Path

import pytest
import torch
from round2_common import ACTOR_KEY, HORIZON_KEY, NORM, RESPONSE_LENGTH, RL, validated

from verl.utils.config import omega_conf_to_dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the round-1 helpers (nested V1 batch, accumulate)
from review_common import UNEVEN, accumulate, make_nested_batch  # noqa: E402

FACTOR = "actor_rollout_ref.actor.loss_scale_factor"
HORIZON = "actor_rollout_ref.rollout.response_length"


@pytest.fixture(autouse=True)
def _env():
    os.environ["P0_ROUND2_RL"] = "2048"
    yield
    os.environ.pop("P0_ROUND2_RL", None)


@pytest.mark.parametrize("spelling", ["${data.max_response_length}", "'${data.max_response_length}'"])
def test_factor_given_as_an_interpolation_is_accepted_as_the_resolved_int(model_dir, spelling):
    config = validated(model_dir, NORM, RL, f"{FACTOR}={spelling}")
    assert config.actor_rollout_ref.actor.loss_scale_factor == RESPONSE_LENGTH
    assert omega_conf_to_dataclass(config.actor_rollout_ref.actor).loss_scale_factor == RESPONSE_LENGTH


def test_factor_from_an_env_string_is_rejected_with_the_string_shown(model_dir):
    with pytest.raises(ValueError, match=ACTOR_KEY) as e:
        validated(model_dir, NORM, RL, f"{FACTOR}='${{oc.env:P0_ROUND2_RL}}'")
    assert "got '2048'" in str(e.value)


def test_horizon_from_a_decoded_env_var_is_accepted(model_dir):
    config = validated(model_dir, NORM, "data.max_response_length='${oc.decode:${oc.env:P0_ROUND2_RL}}'")
    assert config.actor_rollout_ref.actor.loss_scale_factor == 2048


def test_horizon_from_an_undecoded_env_string_is_rejected(model_dir):
    with pytest.raises(ValueError, match=HORIZON_KEY) as e:
        validated(model_dir, NORM, f"{HORIZON}='${{oc.env:P0_ROUND2_RL}}'")
    assert "got '2048'" in str(e.value)


def test_integral_float_horizon_is_rejected_by_design(model_dir):
    """Author's documented choice (NOTES.md): 2048.0 is rejected like 2.5; a relax must change this test."""
    with pytest.raises(ValueError, match=HORIZON_KEY):
        validated(model_dir, NORM, "data.max_response_length=2048.0")


@pytest.mark.parametrize("value", ["2048.0", "5e-324", "1e308"])
def test_any_finite_positive_float_factor_is_accepted(model_dir, value):
    config = validated(model_dir, NORM, RL, f"{FACTOR}={value}")
    assert config.actor_rollout_ref.actor.loss_scale_factor == float(value)


def test_huge_int_factor_is_rejected_at_validation_since_9c99205c(model_dir):
    """D1 (LOW) on e0a7eb22: an int above float max skipped ``math.isfinite`` and ``ppo_loss`` raised OverflowError
    on the first micro-batch. Retired by the lead after 9c99205c bounded int factors to 1..2**63-1: both
    10**400 and 2**63 are now rejected at validation (torch wraps a Python int as int64, so 2**63 overflowed too)."""
    for value in (10**400, 2**63):
        with pytest.raises(ValueError, match=ACTOR_KEY):
            validated(model_dir, NORM, RL, f"{FACTOR}={value}")


def test_huge_int_factor_should_be_rejected_at_validation(model_dir):  # D1 fixed at 9c99205c
    with pytest.raises(ValueError, match=ACTOR_KEY):
        validated(model_dir, NORM, RL, f"{FACTOR}={10**400}")


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_explicit_float_factor_is_consumed_by_the_v1_loss(model_dir, kind):
    """The branch test only checks the stored value; here 0.5 drives the real loss on nested inputs across an
    uneven split: norm(0.5) == seq-mean-token-sum / 0.5 for the loss and the accumulated gradient."""
    norm = validated(model_dir, NORM, RL, f"{FACTOR}=0.5")
    plain = validated(model_dir, "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum", RL)
    section = "actor_rollout_ref.actor" if kind == "actor" else "critic"
    norm_cfg = omega_conf_to_dataclass(norm[section.split(".")[0]][section.split(".")[-1]] if "." in section else norm[section])
    plain_cfg = omega_conf_to_dataclass(plain[section.split(".")[0]][section.split(".")[-1]] if "." in section else plain[section])
    assert norm_cfg.loss_scale_factor == 0.5 and plain_cfg.loss_scale_factor is None
    data, model_output, seq_lens = make_nested_batch()
    norm_loss, norm_grad, _ = accumulate(kind, norm_cfg, data, model_output, seq_lens, UNEVEN)
    plain_loss, plain_grad, _ = accumulate(kind, plain_cfg, data, model_output, seq_lens, UNEVEN)
    torch.testing.assert_close(norm_loss, plain_loss / 0.5)
    torch.testing.assert_close(norm_grad, plain_grad / 0.5)
