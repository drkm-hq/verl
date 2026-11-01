# Round-3 adversarial tests for contrib/p1-lmhead-rows @ e7c0d616 (not part of the branch).
# Target: the finiteness gate and the direct bf16 gradient bound of _compare in
# tests/workers/test_response_only_lm_head_on_gpu.py. Each test encodes a claim of that module's docstring
# or of NOTES.md; a failing test is a gap in the claim, not a product defect.
import importlib.util
import os

import pytest
import torch

import verl

GPU_TEST = os.path.join(os.path.dirname(verl.__file__), "..", "tests", "workers", "test_response_only_lm_head_on_gpu.py")
GRAD = "grad/model.layers.0.weight"


@pytest.fixture(scope="module")
def gpu():
    spec = importlib.util.spec_from_file_location("gpu_identity_test", GPU_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare(gpu, enabled, reference, truth=None, precision="bf16"):
    return gpu._compare("round3", enabled, reference, precision, truth, verbose=False)


@pytest.mark.parametrize(
    "key, bad",
    [
        ("sgd/grad_norm", torch.tensor(float("nan"))),  # 0-d, the shape _run_arm records
        ("sgd/residual_ulps", torch.tensor(float("inf"))),
        ("forward/log_probs", torch.full((6, 256), float("-inf"))),
        ("grad/model.layers.3.input_layernorm.weight", torch.tensor([1.0, float("nan"), 1.0])),
    ],
)
def test_the_gate_reports_non_finite_values_of_every_shape(gpu, key, bad):
    good = torch.zeros_like(bad) + 0.5
    failures = _compare(gpu, {key: bad}, {key: good}, {key: good})
    assert failures and "non-finite" in failures[0], failures


def test_the_gate_covers_a_truth_tensor_that_is_missing(gpu):
    """Claim: every inspected tensor of every arm and of the truth is finite and the guard runs on every bf16
    gradient. A key absent from the truth is skipped silently by ``key in truth``: no guard, no finiteness."""
    reference, enabled = {GRAD: torch.tensor([1.0])}, {GRAD: torch.tensor([1.09])}  # within the direct bound
    failures = _compare(gpu, enabled, reference, truth={})
    assert failures, "a gradient without an fp32 truth passed without the relative-error guard"


def test_the_direct_bound_floor_cannot_hide_a_negated_tensor(gpu):
    """Claim: the direct bound ``|g_on - g_off| <= 0.1 |g_off| + 1e-6 G_layer`` is the identity bound. The floor is
    relative to the layer's largest gradient norm, so a tensor below ~5e-6 of it can be negated and pass."""
    big, tiny = "grad/model.layers.0.mlp.up_proj.weight", "grad/model.layers.0.input_layernorm.weight"
    reference = {big: torch.full((4,), 500.0), tiny: torch.full((4,), 2.5e-4)}  # |big| = 1e3, |tiny| = 5e-4
    enabled = {big: reference[big].clone(), tiny: -reference[tiny]}  # the tiny gradient points the other way
    failures = _compare(gpu, enabled, reference, truth=None)
    assert failures, "a negated gradient tensor passed the direct bound through the layer floor"


def test_empty_shards_do_not_crash_the_comparison(gpu):
    """FSDP2 gives a rank an empty shard when a parameter's dim 0 is smaller than the world size (not the case for
    Qwen2.5-0.5B at two ranks, but for any model with a size-1 leading dimension). The gate must not raise."""
    reference = {"grad/model.layers.0.scale": torch.empty(0), "sgd/grad_norm": torch.tensor(1.0)}
    enabled = {key: value.clone() for key, value in reference.items()}
    assert not _compare(gpu, enabled, reference, truth=None)


def test_both_arms_zero_on_every_response_row_is_reported(gpu):
    """Claim: the forward rows are compared per element. Rows that are zero in both arms are treated as padding,
    so two arms that both scatter zeros onto every response row compare as identical."""
    rows = torch.zeros(6, 256)
    failures = _compare(gpu, {"forward/log_probs": rows}, {"forward/log_probs": rows.clone()}, truth=None)
    assert failures, "all-zero response rows in both arms passed the forward comparison"
