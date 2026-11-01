# Round-3 adversarial tests for contrib/p1-lmhead-rows @ e7c0d616 (not part of the branch).
# Target: the two-rank cases of tests/workers/test_response_only_lm_head_on_gpu.py: "rank r takes sequences r::2 of
# the same frozen batch" must run the real PPO loss on every rank, and the loss normalisation (dp_size,
# batch_num_tokens, global_batch_size) must make the two half batches equivalent to the whole one.
import importlib.util
import os

import pytest
import torch

import verl
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss

GPU_TEST = os.path.join(os.path.dirname(verl.__file__), "..", "tests", "workers", "test_response_only_lm_head_on_gpu.py")
VOCAB = 151936
KEYS = ("log_probs", "entropy", "sum_pi_squared")


@pytest.fixture(scope="module")
def gpu():
    spec = importlib.util.spec_from_file_location("gpu_identity_test", GPU_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actor_config(strategy="fsdp"):
    return ActorConfig(
        strategy=strategy,
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=-1,
        entropy_coeff=0.01,
        calculate_entropy=True,
        calculate_sum_pi_squared=True,
    )


def _full_packed(gpu):
    """One seeded packed vector per key for the whole frozen batch, keyed by the full batch's sequence segments."""
    full = gpu._make_batch(VOCAB)
    offsets = full["input_ids"].offsets().tolist()
    generator = torch.Generator().manual_seed(7)
    vectors = {key: torch.randn(offsets[-1], generator=generator) for key in KEYS}
    vectors["log_probs"] = -vectors["log_probs"].abs() - 1.0  # log-probs are negative; ratio against frozen 0 stays sane
    return full, offsets, vectors


def _model_output(batch, offsets, vectors, kept):
    """The packed model output of ``batch`` (sequences ``kept`` of the full batch, in that order) as nested tensors."""
    segments = {key: torch.cat([vectors[key][offsets[i] : offsets[i + 1]] for i in kept]) for key in KEYS}
    return {key: torch.nested.nested_tensor_from_jagged(value, batch["input_ids"].offsets()) for key, value in segments.items()}


def _engine_fields(batch, dp_size):
    """What forward_backward_batch injects before calling the loss: the global token count and the DP size."""
    tu.assign_non_tensor(batch, sp_size=1, dp_size=dp_size, batch_num_tokens=sum(gpu_response_lens := [40, 128, 9, 64, 1, 256]))
    return batch


@pytest.mark.parametrize("rank", [0, 1])
def test_the_ppo_loss_runs_on_each_rank_half_of_the_frozen_batch(gpu, rank):
    """Claim (docstring, NOTES): the two-rank cases run the same five-engine flow on each rank's half.

    ``ppo_loss`` pads ``old_log_probs`` / ``advantages`` / ``response_mask`` with ``to_padded_tensor()`` (the local
    maximum) and ``log_prob`` with ``no_padding_2_padding`` (``max_response_len``, 256 on every rank)."""
    full, offsets, vectors = _full_packed(gpu)
    kept = list(range(6))[rank::2]
    half = _engine_fields(gpu._make_batch(VOCAB, rank, 2), dp_size=2)
    assert half["responses"].offsets().diff().tolist() == [[40, 9, 1], [128, 64, 256]][rank]
    loss, _ = ppo_loss(_actor_config(), _model_output(half, offsets, vectors, kept), half)
    assert torch.isfinite(loss)


def test_the_two_half_losses_average_to_the_whole_batch_loss(gpu):
    """FSDP averages the two ranks' gradients, so (loss_rank0 + loss_rank1) / 2 must equal the one-rank loss of the
    whole batch for the same per-sequence model outputs (dp_size 2, global batch_num_tokens 498, global batch 6).
    The half batches get their local ``max_response_len`` so that the comparison is not blocked by the crash above."""
    full, offsets, vectors = _full_packed(gpu)
    whole = ppo_loss(_actor_config(), _model_output(full, offsets, vectors, list(range(6))), _engine_fields(full, 1))[0]
    halves = []
    for rank in (0, 1):
        kept = list(range(6))[rank::2]
        half = _engine_fields(gpu._make_batch(VOCAB, rank, 2), dp_size=2)
        local_max = int(half["responses"].offsets().diff().max())
        tu.assign_non_tensor(half, max_response_len=local_max, max_response_length=local_max)
        halves.append(ppo_loss(_actor_config(), _model_output(half, offsets, vectors, kept), half)[0])
    print(f"whole={whole.item():.6f} halves={[h.item() for h in halves]} mean={(sum(halves) / 2).item():.6f}")
    torch.testing.assert_close(sum(halves) / 2, whole, rtol=1e-5, atol=1e-6)


def test_consumer_view_pads_both_arms_to_the_same_width_on_every_rank(gpu):
    """The on/off comparison itself (``_consumer_view``) pads to ``max_response_len`` on both arms: consistent."""
    full, offsets, vectors = _full_packed(gpu)
    for rank in (0, 1):
        half = gpu._make_batch(VOCAB, rank, 2)
        view = gpu._consumer_view(_model_output(half, offsets, vectors, list(range(6))[rank::2]), half)
        assert all(v.shape == (3, 256) for v in view.values())
