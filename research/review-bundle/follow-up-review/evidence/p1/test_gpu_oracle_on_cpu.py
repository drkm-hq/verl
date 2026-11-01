"""Checks the actual P1 GPU comparison helper without spending GPU time.

The NaN cases should be rejected; failures demonstrate holes in the helper.
The finite distance case characterizes the documented acceptance region.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

GPU_TEST = Path('<repo>/wt/p1/tests/workers/test_response_only_lm_head_on_gpu.py')
spec = importlib.util.spec_from_file_location('p1_gpu_oracle', GPU_TEST)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


@pytest.mark.parametrize('precision', ['bf16', 'fp32'])
@pytest.mark.parametrize('key', ['forward/log_probs', 'train/entropy', 'loss/ppo', 'grad/model.layers.0.weight', 'sgd/residual_ulps'])
def test_nan_in_enabled_arm_must_fail(precision, key):
    reference = {key: torch.tensor([1.0])}
    enabled = {key: torch.tensor([float('nan')])}
    truth = {key: torch.tensor([1.0])} if precision == 'bf16' else None
    failures = gpu._compare('injected NaN', enabled, reference, precision, truth)
    assert failures, f'{key} NaN silently accepted for {precision}'


def test_failed_backward_skipped_step_must_fail():
    """Mimics FSDPEngine.optimizer_step's behavior when the enabled arm has NaN grads."""
    before = torch.tensor([1.0])
    good_grad = torch.tensor([0.1])
    good_after = before - gpu.LR * good_grad
    reference = {
        'before/model.weight': before,
        'grad/model.weight': good_grad,
        'param/model.weight': good_after,
        'update/model.weight': good_after - before,
        'sgd/residual_ulps': torch.tensor(0.2),
    }
    enabled = {
        'before/model.weight': before.clone(),
        'grad/model.weight': torch.tensor([float('nan')]),
        'param/model.weight': before.clone(),
        'update/model.weight': torch.tensor([0.0]),
        'sgd/residual_ulps': torch.tensor(float('nan')),
    }
    failures = gpu._compare('NaN backward, optimizer skipped', enabled, reference, 'bf16', reference)
    assert failures, 'A failed enabled-arm backward and skipped optimizer step silently pass'


def test_distance_rule_is_not_identity():
    """A 27% on/off gradient difference can satisfy the distance-to-reference rule."""
    key = 'grad/model.layers.0.weight'
    reference = {key: torch.tensor([1.1])}
    truth = {key: torch.tensor([1.0])}
    enabled = {key: torch.tensor([0.8])}
    assert not gpu._compare('finite acceptance region', enabled, reference, 'bf16', truth)
    assert (enabled[key] - reference[key]).norm() / reference[key].norm() > 0.27
