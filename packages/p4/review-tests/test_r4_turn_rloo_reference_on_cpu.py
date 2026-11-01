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
"""R4: the tensor ``turn_rloo`` against an independent scalar reference on random ragged groups:
group sizes 1..4, 1..4 turns per row, rewards on arbitrary tokens (generated, observation, pad),
gamma in {1, 0.9, 0.5}, with and without std normalisation; plus the exact RLOO reduction."""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage, compute_turn_rloo_advantage


def _random_case(rng, bs_groups, length=12):
    rows, index = [], []
    for g in range(bs_groups):
        n = int(rng.integers(1, 5))
        for _ in range(n):
            turns = int(rng.integers(1, 5))
            spans, pos, mask = [], 0, np.zeros(length)
            for _t in range(turns):
                gen = int(rng.integers(1, 3))
                if pos + gen > length - 1:
                    break
                spans.append((pos, pos + gen))
                mask[pos : pos + gen] = 1
                pos += gen + int(rng.integers(0, 3))  # 0..2 observation tokens
            rewards = np.where(rng.random(length) < 0.4, rng.normal(size=length), 0.0)
            rows.append((spans, mask, rewards))
            index.append(f"g{g}")
    return rows, np.array(index, dtype=object)


def _reference(rows, index, gamma, norm):
    """Scalar RFC #7244 form: R_t = sum of rewards in [s_t, s_{t+1}) (segment 0 from token 0),
    G_t = sum_{t'>=t} gamma^(t'-t) R_t', A_t = G_t - mean over OTHER same-group rows that reached t."""
    length = len(rows[0][1])
    G = []
    for spans, _, rewards in rows:
        R = []
        for t, (s, _e) in enumerate(spans):
            lo = 0 if t == 0 else s
            hi = spans[t + 1][0] if t + 1 < len(spans) else length
            R.append(rewards[lo:hi].sum())
        G.append([sum(gamma ** (k - t) * R[k] for k in range(t, len(R))) for t in range(len(R))])
    adv = np.zeros((len(rows), length))
    for i, (spans, mask, _) in enumerate(rows):
        for t, (s, e) in enumerate(spans):
            others = [G[j][t] for j in range(len(rows)) if j != i and index[j] == index[i] and len(G[j]) > t]
            a = G[i][t] - (np.mean(others) if others else 0.0)
            if norm:
                allg = [G[j][t] for j in range(len(rows)) if index[j] == index[i] and len(G[j]) > t]
                if len(allg) > 1:
                    a = a / (np.std(allg, ddof=1) + 1e-6)
            adv[i, s:e] = a * mask[s:e]
    return adv


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("gamma", [1.0, 0.9, 0.5])
@pytest.mark.parametrize("norm", [False, True])
def test_tensor_matches_scalar_reference(seed, gamma, norm):
    rng = np.random.default_rng(seed)
    rows, index = _random_case(rng, bs_groups=4)
    rewards = torch.tensor(np.stack([r for _, _, r in rows]), dtype=torch.float32)
    mask = torch.tensor(np.stack([m for _, m, _ in rows]), dtype=torch.float32)
    spans = np.empty(len(rows), dtype=object)
    for i, (s, _, _) in enumerate(rows):
        spans[i] = s
    cfg = type("C", (), {"gamma": gamma})()
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans, config=cfg, norm_adv_by_std=norm)
    np.testing.assert_allclose(adv.numpy(), _reference(rows, index, gamma, norm), atol=1e-4, rtol=1e-4)


def test_no_spans_equals_rloo_including_singleton_groups():
    torch.manual_seed(3)
    mask = (torch.rand(10, 7) > 0.3).float()
    rewards = torch.randn(10, 7)  # rewards also on mask-0 tokens: both estimators sum the full row
    index = np.array(["a"] * 4 + ["b"] * 5 + ["c"], dtype=object)
    ref, _ = compute_rloo_outcome_advantage(rewards.clone(), mask, index)
    got, _ = compute_turn_rloo_advantage(rewards.clone(), mask, index)
    torch.testing.assert_close(got, ref)


def test_spans_given_as_a_2d_numpy_array_or_lists_of_lists():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 2.0], [0.0, 0.5, 0.0, 0.0]])
    mask = torch.ones(2, 4)
    index = np.array(["g", "g"], dtype=object)
    as_lists = np.empty(2, dtype=object)
    as_lists[0], as_lists[1] = [[0, 2], [2, 4]], [[0, 2], [2, 4]]
    a, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=as_lists)
    b, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=np.array([[(0, 2), (2, 4)]] * 2))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[:, 0], torch.tensor([3.0 - 0.5, 0.5 - 3.0]))


def test_spans_past_the_row_end_are_not_counted_as_reached():
    """A clipped-away span (start >= length) must not enter the baseline as a zero return."""
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 4.0], [0.0, 1.0, 0.0, 4.0]])
    mask = torch.ones(3, 4)
    index = np.array(["g"] * 3, dtype=object)
    spans = np.empty(3, dtype=object)
    spans[0], spans[1], spans[2] = [(0, 2), (9, 11)], [(0, 2), (2, 4)], [(0, 2), (2, 4)]
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)
    # turn 2 of rows 1 and 2: each has G=4; the only other member that reached turn 2 is the other one -> 0
    assert adv[1, 2].item() == pytest.approx(0.0) and adv[2, 2].item() == pytest.approx(0.0)
