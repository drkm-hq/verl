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

"""Tests for the ``turn_rloo`` advantage estimator (RFC #7244).

The soundness check is an exact policy-gradient comparison on a tabular two-turn MDP in which the
policy decides whether the second turn happens at all (stop vs. continue). RFC #7244 shows that the
natural per-turn-index normalisation (the MT-GRPO shape) is biased there by the accumulated group
mean of the intermediate reward, while a leave-one-out return-to-go baseline over the group members
that reached the turn is unbiased. Both are measured against the closed-form gradient.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_reinforce_plus_plus_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_turn_rloo_advantage,
    get_adv_estimator_fn,
)

# Trajectory layout of the tabular MDP in response coordinates: two generated tokens per turn and one
# observation token after turn 1. Rewards sit on the last generated token of each turn.
#   continue: tokens [t1 t1 obs t2 t2], mask [1 1 0 1 1], spans [(0, 2), (3, 5)]
#   stop:     tokens [t1 t1 pad pad pad], mask [1 1 0 0 0], spans [(0, 2)]
MU, R_STOP, N_GROUPS, GROUP_SIZE = 1.0, 0.6, 20_000, 8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _simulate(rng, th_c, th_a):
    """Sample groups of the two-turn MDP. Every array has shape (N_GROUPS, GROUP_SIZE)."""
    p, q = _sigmoid(th_c), _sigmoid(th_a)
    cont = rng.random((N_GROUPS, GROUP_SIZE)) < p
    good = rng.random((N_GROUPS, GROUP_SIZE)) < q
    r_i1 = np.where(cont, MU, 0.0)
    r_o = np.where(cont, good.astype(float), R_STOP)
    g_c = cont.astype(float) - p  # d log pi(c | s1) / d th_c
    g_a = np.where(cont, good.astype(float) - q, 0.0)  # d log pi(a | s2) / d th_a, 0 if never reached
    return cont, r_i1, r_o, g_c, g_a


def _loo_mean(x, valid):
    xv = np.where(valid, x, 0.0)
    s = xv.sum(axis=1, keepdims=True) - xv
    c = valid.sum(axis=1, keepdims=True) - valid
    return np.where(c > 0, s / np.maximum(c, 1), 0.0)


def _turn_index_normalised(cont, r_i1, r_o):
    """The biased estimator: return-to-go minus the per-turn-index group means of the raw rewards."""
    all_valid = np.ones_like(cont, dtype=bool)
    mu_i1, mu_o = _loo_mean(r_i1, all_valid), _loo_mean(r_o, all_valid)
    a1 = (r_i1 + r_o) - np.where(cont, mu_i1 + mu_o, mu_o)
    a2 = np.where(cont, r_o, 0.0) - mu_o
    return a1, a2


def _as_batch(cont, r_i1, r_o):
    """Lay the sampled trajectories out as (token_level_rewards, response_mask, index, turn_spans)."""
    n = cont.size
    rewards = np.zeros((n, 5))
    mask = np.zeros((n, 5))
    spans = np.empty(n, dtype=object)
    cont_f, r_i1_f, r_o_f = cont.reshape(-1), r_i1.reshape(-1), r_o.reshape(-1)
    mask[:, :2] = 1
    mask[cont_f, 3:] = 1
    rewards[cont_f, 1] = r_i1_f[cont_f]
    rewards[cont_f, 4] = r_o_f[cont_f]
    rewards[~cont_f, 1] = r_o_f[~cont_f]
    for i in range(n):
        spans[i] = [(0, 2), (3, 5)] if cont_f[i] else [(0, 2)]
    index = np.repeat(np.arange(cont.shape[0]), cont.shape[1])
    return torch.tensor(rewards, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), index, spans


def test_turn_rloo_is_registered():
    assert get_adv_estimator_fn("turn_rloo") is compute_turn_rloo_advantage


def test_turn_rloo_matches_exact_policy_gradient_where_turn_index_normalisation_is_biased():
    rng = np.random.default_rng(0)
    th_c, th_a = 0.0, 0.0
    p, q = _sigmoid(th_c), _sigmoid(th_a)
    exact = np.array([p * (1 - p) * (MU + q - R_STOP), p * q * (1 - q)])
    cont, r_i1, r_o, g_c, g_a = _simulate(rng, th_c, th_a)

    rewards, mask, index, spans = _as_batch(cont, r_i1, r_o)
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans, config=SimpleNamespace(gamma=1.0))
    a1 = adv[:, 1].numpy().reshape(cont.shape)  # advantage of the turn-1 decision
    a2 = adv[:, 4].numpy().reshape(cont.shape)  # advantage of the turn-2 decision (0 for stoppers)

    def estimate(adv1, adv2):
        per_group = np.stack([(adv1 * g_c).mean(axis=1), (adv2 * g_a).mean(axis=1)], axis=1)
        return per_group.mean(axis=0), per_group.std(axis=0, ddof=1) / np.sqrt(N_GROUPS)

    est, se = estimate(a1, a2)
    assert np.all(np.abs(est - exact) < 4 * se), f"turn_rloo biased: est={est} exact={exact} se={se}"

    # The same data through the per-turn-index normalisation misses the termination gradient by
    # -p(1-p) * E[r_I] = -p(1-p) * p * MU (RFC #7244), many standard errors away: the test has the
    # power to tell the two estimators apart.
    b1, b2 = _turn_index_normalised(cont, r_i1, r_o)
    est_b, se_b = estimate(b1, b2)
    assert abs(est_b[0] - exact[0]) > 10 * se_b[0]
    assert abs((est_b[0] - exact[0]) - (-p * (1 - p) * p * MU)) < 4 * se_b[0]


def test_turn_rloo_tensor_matches_scalar_formula_and_zeros_observation_tokens():
    rng = np.random.default_rng(1)
    cont, r_i1, r_o, _, _ = _simulate(rng, 0.3, -0.2)
    cont, r_i1, r_o = cont[:16], r_i1[:16], r_o[:16]
    rewards, mask, index, spans = _as_batch(cont, r_i1, r_o)
    adv, returns = compute_turn_rloo_advantage(
        rewards, mask, index, turn_spans=spans, config=SimpleNamespace(gamma=1.0)
    )

    g1 = r_i1 + r_o
    g2 = np.where(cont, r_o, 0.0)
    a1 = g1 - _loo_mean(g1, np.ones_like(cont, dtype=bool))
    a2 = g2 - _loo_mean(g2, cont)
    np.testing.assert_allclose(adv[:, 0].numpy(), a1.reshape(-1), atol=1e-5)
    np.testing.assert_allclose(adv[:, 1].numpy(), a1.reshape(-1), atol=1e-5)
    np.testing.assert_allclose(adv[:, 4].numpy(), np.where(cont, a2, 0.0).reshape(-1), atol=1e-5)
    assert torch.all(adv[:, 2] == 0) and torch.all(returns[:, 2] == 0)  # the observation token
    np.testing.assert_allclose(returns[:, 0].numpy(), g1.reshape(-1), atol=1e-6)
    np.testing.assert_allclose(returns[:, 4].numpy(), np.where(cont, g2, 0.0).reshape(-1), atol=1e-6)


def test_turn_rloo_without_spans_equals_rloo():
    torch.manual_seed(0)
    mask = (torch.rand(12, 9) > 0.3).float()
    rewards = torch.randn(12, 9) * mask
    index = np.array(["a"] * 5 + ["b"] * 6 + ["c"])  # a singleton group keeps its raw score in both
    adv_rloo, _ = compute_rloo_outcome_advantage(rewards.clone(), mask, index)
    adv_turn, _ = compute_turn_rloo_advantage(rewards.clone(), mask, index)
    torch.testing.assert_close(adv_turn, adv_rloo)
    none_spans = np.empty(12, dtype=object)
    adv_none, _ = compute_turn_rloo_advantage(rewards.clone(), mask, index, turn_spans=none_spans)
    torch.testing.assert_close(adv_none, adv_rloo)


def test_turn_rloo_gives_rloo_policy_advantages_but_keeps_raw_returns():
    """One-token rewards 1 and 3: both estimators give advantages [-2, 2]; rloo hands its advantages back as
    ``returns`` while turn_rloo keeps the raw return-to-go (return metrics differ, the actor gradient does not)."""
    rewards, mask, index = torch.tensor([[1.0], [3.0]]), torch.ones(2, 1), np.array(["g", "g"], dtype=object)
    adv_rloo, ret_rloo = compute_rloo_outcome_advantage(rewards.clone(), mask, index)
    adv_turn, ret_turn = compute_turn_rloo_advantage(rewards.clone(), mask, index)
    torch.testing.assert_close(adv_turn, torch.tensor([[-2.0], [2.0]]))
    torch.testing.assert_close(adv_rloo, adv_turn)
    torch.testing.assert_close(ret_rloo, torch.tensor([[-2.0], [2.0]]))
    torch.testing.assert_close(ret_turn, torch.tensor([[1.0], [3.0]]))


def test_norm_adv_by_std_is_a_variance_heuristic_not_a_rescaled_gradient():
    """Single turn, three actions with rewards [0, 1, 4] and probabilities [0.2, 0.3, 0.5], groups of 3, every
    joint outcome enumerated: the unnormalised expected score-gradient is the exact gradient p_k (r_k - E[r])
    = [-0.46, -0.39, 0.85]; dividing by the group std (which contains the sample's own return) gives
    [-0.245, -0.185, 0.430], not a common rescaling of it."""
    rewards_by_action, probs = torch.tensor([0.0, 1.0, 4.0]), torch.tensor([0.2, 0.3, 0.5])
    grads = {norm: torch.zeros(3, dtype=torch.float64) for norm in (False, True)}
    for actions in itertools.product(range(3), repeat=3):
        weight = probs[list(actions)].prod().double()
        rewards = rewards_by_action[list(actions)].reshape(3, 1)
        for norm in (False, True):
            adv, _ = compute_turn_rloo_advantage(rewards, torch.ones(3, 1), np.array(["g"] * 3), norm_adv_by_std=norm)
            for i, a in enumerate(actions):
                score = -probs.double()
                score[a] += 1.0  # d log pi(a) / d logits
                grads[norm] += weight * score * adv[i, 0].double() / 3
    exact = (probs * (rewards_by_action - (probs * rewards_by_action).sum())).double()
    torch.testing.assert_close(grads[False], exact, atol=1e-6, rtol=0)
    reviewer = torch.tensor([-0.46, -0.39, 0.85], dtype=torch.float64)
    torch.testing.assert_close(grads[False], reviewer, atol=1e-6, rtol=0)
    reviewer_normalised = torch.tensor([-0.2453, -0.1847, 0.4300], dtype=torch.float64)
    torch.testing.assert_close(grads[True], reviewer_normalised, atol=2e-4, rtol=0)
    ratio = grads[True] / grads[False]
    assert ratio.max() / ratio.min() > 1.05, f"{grads[True].tolist()} is not a rescaling of {grads[False].tolist()}"


def test_gamma_below_one_is_the_discounted_return_to_go_heuristic():
    """Two independent Bernoulli turns (p = 0.5, reward = action, logit parameters) with gamma 0.5: the
    implemented expected score-gradient is [0.25, 0.25], the gradient of E[r1 + 0.5 r2] is [0.25, 0.125]."""

    def expected_gradient(gamma):
        grad, mask = torch.zeros(2, dtype=torch.float64), torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.float32)
        spans = np.empty(1, dtype=object)
        spans[0] = [(0, 2), (3, 5)]
        for a1, a2 in itertools.product((0, 1), repeat=2):
            rewards = torch.tensor([[0.0, float(a1), 0.0, 0.0, float(a2)]])  # [t1 t1 obs t2 t2]
            adv, _ = compute_turn_rloo_advantage(
                rewards, mask, np.array(["g"]), turn_spans=spans, config=SimpleNamespace(gamma=gamma)
            )
            score = torch.tensor([a1 - 0.5, a2 - 0.5], dtype=torch.float64)  # d log pi / d logit
            grad += 0.25 * score * torch.tensor([adv[0, 0].item(), adv[0, 3].item()], dtype=torch.float64)
        return grad

    torch.testing.assert_close(expected_gradient(1.0), torch.tensor([0.25, 0.25], dtype=torch.float64))
    torch.testing.assert_close(expected_gradient(0.5), torch.tensor([0.25, 0.25], dtype=torch.float64))
    assert not torch.allclose(expected_gradient(0.5), torch.tensor([0.25, 0.125], dtype=torch.float64))


def test_turn_rloo_hand_computed_with_ragged_turn_counts_and_gamma():
    # group g: rows 0, 1 have three turns, row 2 has one; group h: a single row (baseline 0 everywhere)
    #   tokens: [t1 t1 o t2 t2 o t3 t3]; the turn reward sits on the last generated token of the turn
    mask = torch.tensor(
        [[1, 1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )
    turn_rewards = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0], [0.5, 0.5]]
    spans = np.empty(4, dtype=object)
    rewards = torch.zeros(4, 8)
    for i, rs in enumerate(turn_rewards):
        spans[i] = [(0, 2), (3, 5), (6, 8)][: len(rs)]
        for (_, end), r in zip(spans[i], rs, strict=False):
            rewards[i, end - 1] = r
    index = np.array(["g", "g", "g", "h"])
    gamma = 0.5

    def ret(rs, t):
        return sum(gamma ** (k - t) * r for k, r in enumerate(rs) if k >= t)

    returns = [[ret(rs, t) for t in range(len(rs))] for rs in turn_rewards]
    expected = torch.zeros(4, 8)
    expected_returns = torch.zeros(4, 8)
    for i, rs in enumerate(turn_rewards):
        for t in range(len(rs)):
            others = [returns[j][t] for j in range(4) if j != i and index[j] == index[i] and len(turn_rewards[j]) > t]
            baseline = sum(others) / len(others) if others else 0.0
            start, end = spans[i][t]
            expected[i, start:end] = returns[i][t] - baseline
            expected_returns[i, start:end] = returns[i][t]

    adv, got_returns = compute_turn_rloo_advantage(
        rewards, mask, index, turn_spans=spans, config=SimpleNamespace(gamma=gamma)
    )
    torch.testing.assert_close(adv, expected)
    torch.testing.assert_close(got_returns, expected_returns)
    # turn 1 of the three-turn rows is baselined against each other only; row 2's lone turn 1 against both
    assert adv[0, 0].item() == pytest.approx(returns[0][0] - (returns[1][0] + returns[2][0]) / 2)
    assert adv[0, 6].item() == pytest.approx(2.0 - 0.0)  # row 2 never reached turn 3
    assert adv[3, 0].item() == pytest.approx(returns[3][0])  # singleton group: baseline 0


def test_turn_rloo_std_normalisation_is_optional():
    mask = torch.ones(3, 2)
    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]])
    index = np.array(["g"] * 3)
    spans = np.empty(3, dtype=object)
    for i in range(3):
        spans[i] = [(0, 2)]
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)
    torch.testing.assert_close(adv[:, 0], torch.tensor([1.0 - 4.0, 3.0 - 3.0, 5.0 - 2.0]))
    adv_norm, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans, norm_adv_by_std=True)
    std = torch.tensor([1.0, 3.0, 5.0]).std()
    torch.testing.assert_close(adv_norm[:, 0], adv[:, 0] / (std + 1e-6))


def test_dense_turn_rewards_reach_gae_and_reinforce_plus_plus_only_on_generated_tokens():
    """Regression for the placement rule of ``rollout.multi_turn.turn_level_rewards``: both estimators carry a
    reward across observation spans when it sits on a generated token, and drop it when it sits on a
    ``response_mask == 0`` token (issue #8367), which is why the writer uses the turn's last generated token."""
    mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
    rewards = torch.zeros(1, 14)
    rewards[0, 2], rewards[0, 6], rewards[0, 13] = 1.0, 0.0, 1.0  # turn rewards 1, 0, 1 on last generated tokens
    expected_return = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    generated = mask[0].bool()

    _, rpp_returns = compute_reinforce_plus_plus_outcome_advantage(rewards, mask, config=SimpleNamespace(gamma=1.0))
    torch.testing.assert_close(rpp_returns[0][generated], expected_return)
    _, gae_returns = compute_gae_advantage_return(rewards, torch.zeros_like(rewards), mask, gamma=1.0, lam=1.0)
    torch.testing.assert_close(gae_returns[0][generated], expected_return)

    misplaced = torch.zeros(1, 14)
    misplaced[0, 4] = 1.0  # the same reward on an observation token is lost by both
    _, rpp_lost = compute_reinforce_plus_plus_outcome_advantage(misplaced, mask, config=SimpleNamespace(gamma=1.0))
    _, gae_lost = compute_gae_advantage_return(misplaced, torch.zeros_like(misplaced), mask, gamma=1.0, lam=1.0)
    assert rpp_lost[0][generated].sum() == 0 and gae_lost[0][generated].sum() == 0


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([(0, 3), (2, 4)], id="overlap"),
        pytest.param([(2, 4), (0, 2)], id="reversed-order"),
        pytest.param([(0, 2), (2, 2)], id="empty"),
        pytest.param([(3, 2)], id="start-after-end"),
        pytest.param([(-1, 2)], id="negative"),
        pytest.param([(0, 2), (3, 9)], id="past-the-row-end"),
        pytest.param([(0, 2), (9, 11)], id="starting-past-the-row-end"),
        pytest.param([(0, 2), (3,)], id="not-a-pair"),
    ],
)
def test_malformed_spans_raise_naming_the_row(bad):
    """External producers can hand over spans the loop never would; they must not silently distort the grouping."""
    rewards, mask = torch.zeros(2, 4), torch.ones(2, 4)
    index = np.array(["g", "g"], dtype=object)
    spans = np.empty(2, dtype=object)
    spans[0], spans[1] = [(0, 2), (2, 4)], bad
    with pytest.raises(ValueError, match=r"turn_spans\[1\]"):
        compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)
    spans[1] = [(0, 2), (2, 4)]
    compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)  # the well-formed batch passes
    with pytest.raises(ValueError, match="3 rows for a batch of 2"):
        compute_turn_rloo_advantage(rewards, mask, index, turn_spans=np.empty(3, dtype=object))


def test_adv_estimator_takes_turn_spans_reads_the_signature():
    import functools

    from verl.trainer.ppo.core_algos import adv_estimator_takes_turn_spans

    assert adv_estimator_takes_turn_spans("turn_rloo") and adv_estimator_takes_turn_spans(compute_turn_rloo_advantage)
    for name in ("rloo", "grpo", "gae", "reinforce_plus_plus", "grpo_vectorized"):
        assert not adv_estimator_takes_turn_spans(name), name

    def kwargs_only(token_level_rewards, response_mask, index, config=None, **kwargs):
        return None

    @functools.wraps(compute_turn_rloo_advantage)
    def wrapped(*args, **kwargs):
        return compute_turn_rloo_advantage(*args, **kwargs)

    assert not adv_estimator_takes_turn_spans(kwargs_only), "**kwargs alone does not opt in"
    assert adv_estimator_takes_turn_spans(wrapped), "inspect.signature follows __wrapped__"


def _spans_batch(with_spans_column: bool, spans=None):
    from tensordict import TensorDict

    from verl import DataProto

    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    non_tensor = {"uid": np.array(["a", "a", "b", "b"], dtype=object)}
    if with_spans_column:
        column = np.empty(4, dtype=object)
        column[:] = spans if spans is not None else [None] * 4
        non_tensor["generated_turn_spans"] = column
    batch = TensorDict({"token_level_rewards": rewards, "response_mask": mask}, batch_size=[4])
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


def test_compute_advantage_passes_turn_spans_only_to_estimators_that_declare_them(monkeypatch):
    """A `generated_turn_spans` column must not break estimators without a `turn_spans` parameter or `**kwargs`."""
    import verl.trainer.ppo.core_algos as core_algos
    from verl.trainer.ppo.ray_trainer import compute_advantage

    def plain_rloo(token_level_rewards, response_mask, index, config=None):
        return compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)

    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "plain_rloo", plain_rloo)
    for name in ("grpo_vectorized", "plain_rloo", "rloo"):
        expected = compute_advantage(_spans_batch(False), adv_estimator=name).batch["advantages"]
        got = compute_advantage(_spans_batch(True), adv_estimator=name).batch["advantages"]
        torch.testing.assert_close(got, expected)

    spans = [[(0, 3), (3, 4)], [(0, 2), (3, 4)], [(0, 2)], [(0, 4)]]
    out = compute_advantage(_spans_batch(True, spans), adv_estimator="turn_rloo").batch["advantages"]
    assert out[1, 0].item() == pytest.approx(0.5 - 1.0) and out[1, 3].item() == pytest.approx(0.5 - 0.0)
