# Adversarial review tests for contrib/p2-oldlogprob-skip (every registered policy loss).
"""ppo_loss under on_policy_skip_old_log_prob for every registered loss mode (except bypass_mode, which the
gate rejects): the loss must compute without an old_log_probs field, its gradient must equal the gradient
with old_log_probs = log_prob.detach() handed in as a field (bitwise), the ratio metrics must be exactly 0
where the mode reports them, and WITHOUT the flag a missing old_log_probs must still raise."""

import pytest
import torch
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

P, RESP_LENS = 4, [6, 3, 5, 4]
MODES = [
    "vanilla",
    "dppo_tv",
    "dppo_kl",
    "gspo",
    "sapo",
    "gpg",
    "clip_cov",
    "kl_cov",
    "geo_mean",
    "dro",
    "cispo",
]


def _nested(rows):
    return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)


def _config(mode):
    kwargs = {"loss_mode": mode}
    if mode == "dro":
        kwargs["dro_beta"] = 0.1
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=2,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        entropy_coeff=0.0,
        use_kl_loss=False,
        policy_loss=PolicyLossConfig(**kwargs),
    )


def _inputs(with_old, on_policy):
    torch.manual_seed(0)
    lens = [P + r for r in RESP_LENS]
    leaf = (-torch.rand(sum(lens))).requires_grad_(True)
    offsets = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])
    full = torch.nested.nested_tensor_from_jagged(leaf * 1.0, offsets)
    rows = {
        "prompts": [torch.randint(0, 50, (P,)) for _ in RESP_LENS],
        "responses": [torch.randint(0, 50, (r,)) for r in RESP_LENS],
        "response_mask": [torch.ones(r, dtype=torch.long) for r in RESP_LENS],
        "advantages": [torch.randn(r) for r in RESP_LENS],
    }
    tensors = {k: _nested(v) for k, v in rows.items()}
    if with_old:
        # the response slice of the detached log-probs, as _store_actor_log_prob would store it
        tensors["old_log_probs"] = _nested(
            row[P - 1 : P - 1 + r].detach().clone()
            for row, r in zip(full.unbind(), RESP_LENS, strict=True)
        )
    data = tu.get_tensordict(tensor_dict=tensors)
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=sum(RESP_LENS),
        global_batch_size=len(RESP_LENS),
    )
    if on_policy:
        tu.assign_non_tensor(data, on_policy_skip_old_log_prob=True)
    return leaf, full, data


@pytest.mark.parametrize("mode", MODES)
def test_skip_matches_explicit_detached_old_log_probs_bitwise(mode):
    config = _config(mode)
    leaf_a, full_a, data_a = _inputs(with_old=False, on_policy=True)
    loss_a, metrics_a = ppo_loss(config, {"log_probs": full_a}, data_a)
    (grad_a,) = torch.autograd.grad(loss_a, leaf_a)

    leaf_b, full_b, data_b = _inputs(with_old=True, on_policy=False)
    loss_b, metrics_b = ppo_loss(config, {"log_probs": full_b}, data_b)
    (grad_b,) = torch.autograd.grad(loss_b, leaf_b)

    assert torch.equal(loss_a.detach(), loss_b.detach()), mode
    assert torch.equal(grad_a, grad_b), mode
    assert grad_a.abs().sum() > 0, f"{mode}: no gradient at all"
    for key in ("actor/pg_clipfrac", "actor/ppo_kl"):
        if key in metrics_a:
            assert metrics_a[key].aggregate() == 0.0, (mode, key)


@pytest.mark.parametrize("mode", ["vanilla", "gspo"])
def test_without_the_flag_a_missing_field_still_raises(mode):
    _, full, data = _inputs(with_old=False, on_policy=False)
    with pytest.raises(KeyError):
        ppo_loss(_config(mode), {"log_probs": full}, data)


def test_flag_false_in_data_takes_the_normal_path():
    leaf, full, data = _inputs(with_old=True, on_policy=False)
    tu.assign_non_tensor(data, on_policy_skip_old_log_prob=False)
    loss, _ = ppo_loss(_config("vanilla"), {"log_probs": full}, data)
    assert loss.requires_grad
