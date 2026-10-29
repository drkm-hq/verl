# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Failure injection against the comparison helper of the GPU identity test; no GPU needed.

The helper is imported from ``test_response_only_lm_head_on_gpu.py`` the way an external oracle
uses it. NaN in any inspected tensor must be reported, in bf16 and in fp32, as must a skipped
optimizer step (non-finite gradients, unchanged parameters, zero update); the direct bf16 gradient
bound rejects a 27 % on/off difference that the relative-error guard alone accepts, while a
difference within the baseline's own error passes (the guard is not an identity statement); the
direct bound has an absolute floor for near-zero tensors.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_gpu_test_module():
    path = Path(__file__).with_name("test_response_only_lm_head_on_gpu.py")
    spec = importlib.util.spec_from_file_location("response_only_lm_head_gpu_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu = _load_gpu_test_module()
GRAD = "grad/model.layers.0.weight"
KEYS = ["forward/log_probs", "train/entropy", "loss/ppo", GRAD, "sgd/residual_ulps", "sgd/grad_norm"]


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("key", KEYS)
def test_nan_in_the_enabled_arm_is_reported(precision, key):
    reference = {key: torch.tensor([1.0])}
    enabled = {key: torch.tensor([float("nan")])}
    truth = {key: torch.tensor([1.0])} if precision == "bf16" else None
    failures = gpu._compare("injected NaN", enabled, reference, precision, truth, verbose=False)
    assert failures and "non-finite" in failures[0], f"{key}: NaN accepted in {precision}"


@pytest.mark.parametrize("where", ["reference", "truth"])
def test_nan_in_the_reference_or_the_truth_is_reported(where):
    finite, nan = torch.tensor([1.0]), torch.tensor([float("nan")])
    reference = {GRAD: nan if where == "reference" else finite}
    truth = {GRAD: nan if where == "truth" else finite}
    failures = gpu._compare("injected NaN", {GRAD: finite}, reference, "bf16", truth, verbose=False)
    assert failures and "non-finite" in failures[0]


def _arms_after_one_step(enabled_grad, enabled_after, enabled_residual, enabled_grad_norm):
    before, grad = torch.tensor([1.0]), torch.tensor([0.1])
    after = before - gpu.LR * grad
    reference = {
        "before/model.weight": before,
        "grad/model.weight": grad,
        "param/model.weight": after,
        "update/model.weight": after - before,
        "sgd/residual_ulps": torch.tensor(0.2),
        "sgd/grad_norm": torch.tensor(0.1),
    }
    enabled = {
        "before/model.weight": before.clone(),
        "grad/model.weight": enabled_grad,
        "param/model.weight": enabled_after,
        "update/model.weight": enabled_after - before,
        "sgd/residual_ulps": torch.tensor(enabled_residual),
        "sgd/grad_norm": torch.tensor(enabled_grad_norm),
    }
    return enabled, reference


def test_a_skipped_optimizer_step_on_non_finite_gradients_is_reported():
    """The engine skips the step on non-finite gradients: unchanged parameters, zero update, NaN residual."""
    enabled, reference = _arms_after_one_step(
        torch.tensor([float("nan")]), torch.tensor([1.0]), float("nan"), float("nan")
    )
    text = "\n".join(gpu._compare("skipped step", enabled, reference, "bf16", reference, verbose=False))
    assert "non-finite" in text and "zero update" in text


def test_a_skipped_optimizer_step_with_finite_gradients_is_reported():
    """Finite gradients but no step: the update is zero, the residual is huge and the grad norm is zero."""
    enabled, reference = _arms_after_one_step(torch.tensor([0.1]), torch.tensor([1.0]), 840.0, 0.0)
    text = "\n".join(gpu._compare("skipped step", enabled, reference, "bf16", reference, verbose=False))
    assert "zero update" in text and "param_after != param_before - lr * grad" in text and "step skipped" in text


def test_the_direct_gradient_bound_rejects_what_the_guard_alone_accepts():
    truth, reference = {GRAD: torch.tensor([1.0])}, {GRAD: torch.tensor([1.1])}
    # on 0.8: D_on 0.2 <= 2 * D_off 0.1 passes the guard, but 0.3 > 0.1 * |g_off| fails the direct bound
    rejected = gpu._compare("27 % apart", {GRAD: torch.tensor([0.8])}, reference, "bf16", truth, verbose=False)
    assert rejected and "gradients" in rejected[0]
    # on 1.05: 4.5 % apart, within the baseline's own error: the guard is a relative-error guard, not identity
    assert not gpu._compare("within the error", {GRAD: torch.tensor([1.05])}, reference, "bf16", truth, verbose=False)


def test_the_direct_bound_has_an_absolute_floor_for_near_zero_tensors():
    big, tiny = "grad/model.layers.0.big", "grad/model.layers.0.tiny"
    reference = {big: torch.ones(4), tiny: torch.full((4,), 1e-9)}
    enabled = {big: torch.ones(4), tiny: torch.full((4,), 1e-9) + 1e-8}  # 1000 % relative, 2e-8 below 1e-6 * |big|
    assert not gpu._compare("floor", enabled, reference, "bf16", None, verbose=False)
    enabled[big] = torch.ones(4) * 1.2  # 20 % on a tensor of the layer's scale is rejected
    assert gpu._compare("20 % apart", enabled, reference, "bf16", None, verbose=False)


def test_an_ulp_sized_bf16_forward_difference_passes_and_a_row_shift_fails():
    rows = -torch.rand(6, 256, generator=torch.Generator().manual_seed(0)) * 20
    rows[:, 128:] = 0  # padding, zero in both arms
    shifted = rows.clone()
    shifted[0, 0] += 0.125  # one bf16 ulp of a logit of 20 on one row
    assert not gpu._compare(
        "one ulp", {"forward/log_probs": shifted}, {"forward/log_probs": rows}, "bf16", verbose=False
    )
    shifted = rows - 0.03  # every row shifted: the batch mean catches it
    assert gpu._compare(
        "shifted", {"forward/log_probs": shifted * (rows != 0)}, {"forward/log_probs": rows}, "bf16", verbose=False
    )
