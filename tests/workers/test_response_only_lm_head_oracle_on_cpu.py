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
direct bound never hides a negated tensor; a key without an fp32 reference, an empty shard and all-zero
response rows are handled as the docstring says.
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


def test_the_direct_bound_never_hides_a_negated_tensor():
    big, tiny = "grad/model.layers.0.mlp.up_proj.weight", "grad/model.layers.0.input_layernorm.weight"
    reference = {big: torch.full((4,), 500.0), tiny: torch.full((4,), 2.5e-4)}  # |big| = 1e3, |tiny| = 5e-4
    enabled = {big: reference[big].clone(), tiny: -reference[tiny]}  # the tiny gradient points the other way
    assert gpu._compare("negated", enabled, reference, "bf16", None, verbose=False)
    enabled[tiny] = reference[tiny] * 1.05  # 5 % on a tensor far below the layer's scale: within the bound
    assert not gpu._compare("tiny 5 %", enabled, reference, "bf16", None, verbose=False)
    enabled[tiny] = torch.zeros(4) + 1e-13  # below the absolute term: no gradient at all, compared as such
    reference[tiny] = torch.zeros(4)
    assert not gpu._compare("no gradient", enabled, reference, "bf16", None, verbose=False)


def test_a_key_without_an_fp32_reference_is_reported():
    reference, enabled = {GRAD: torch.tensor([1.0])}, {GRAD: torch.tensor([1.01])}
    failures = gpu._compare("missing truth", enabled, reference, "bf16", {}, verbose=False)
    assert failures and "no such key" in failures[0]


def test_an_empty_shard_is_skipped_not_crashed():
    reference = {"grad/model.layers.0.scale": torch.empty(0), "sgd/grad_norm": torch.tensor(1.0)}
    enabled = {key: value.clone() for key, value in reference.items()}
    assert not gpu._compare("empty shard", enabled, reference, "bf16", None, verbose=False)


def _rows(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rows = -torch.rand(6, 256, generator=torch.Generator().manual_seed(seed)) * 20
    mask = torch.zeros(6, 256, dtype=torch.bool)
    for i, length in enumerate(gpu.RESPONSE_LENS):
        mask[i, :length] = True
    return rows * mask, mask


def test_all_zero_response_rows_are_reported_not_treated_as_padding():
    rows, mask = _rows()
    zeros = torch.zeros_like(rows)
    assert gpu._compare("all zero", {"forward/log_probs": zeros}, {"forward/log_probs": zeros.clone()}, verbose=False)
    with_mask = {"forward/log_probs": zeros, "mask/response": mask}
    assert gpu._compare("all zero, masked", with_mask, dict(with_mask), verbose=False)
    assert not gpu._compare(
        "padding only",
        {"forward/log_probs": rows, "mask/response": mask},
        {"forward/log_probs": rows.clone(), "mask/response": mask},
        verbose=False,
    )


def test_an_ulp_sized_bf16_forward_difference_passes_and_a_row_shift_fails():
    rows, mask = _rows()
    shifted = rows.clone()
    shifted[0, 0] += 0.125  # one bf16 ulp of a logit of 20 on one row
    reference = {"forward/log_probs": rows, "mask/response": mask}
    assert not gpu._compare("one ulp", {"forward/log_probs": shifted, "mask/response": mask}, reference, verbose=False)
    shifted = (rows - 0.03) * mask  # every response row shifted: the batch mean catches it
    assert gpu._compare("shifted", {"forward/log_probs": shifted, "mask/response": mask}, reference, verbose=False)


def test_the_determinism_baseline_handles_boolean_and_empty_tensors():
    reference = {
        "mask/response": torch.tensor([[True, False]]),
        "grad/model.layers.0.scale": torch.empty(0),
        "sgd/grad_norm": torch.tensor(1.0),
        "forward/log_probs": torch.tensor([[-1.0, 0.0]]),
    }
    gpu._assert_bitwise("baseline", {key: value.clone() for key, value in reference.items()}, reference)
    flipped = {key: value.clone() for key, value in reference.items()}
    flipped["mask/response"][0, 1] = True
    with pytest.raises(AssertionError, match="mask/response differs"):
        gpu._assert_bitwise("baseline", flipped, reference)
    shifted = {key: value.clone() for key, value in reference.items()}
    shifted["forward/log_probs"][0, 0] = -1.5
    with pytest.raises(AssertionError, match="max_abs=5.000e-01"):
        gpu._assert_bitwise("baseline", shifted, reference)


@pytest.mark.parametrize("rank, world_size", [(0, 1), (0, 2), (1, 2)])
def test_the_response_mask_matches_the_rank_local_batch(rank, world_size):
    batch = gpu._make_batch(512, rank, world_size)
    mask = gpu._response_mask(batch, "cpu")
    lengths = batch["responses"].offsets().diff().tolist()
    assert mask.shape == (len(lengths), max(gpu.RESPONSE_LENS)) and mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == lengths
    assert batch["max_response_len"] == max(lengths)
