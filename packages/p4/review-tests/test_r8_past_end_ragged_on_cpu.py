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
"""R8 (fix D5): the past-end span filter inside random ragged groups with gamma < 1 and std normalisation,
checked against the scalar reference computed on the filtered spans; spans that straddle the row end stay."""

import numpy as np
import pytest
import torch

from test_r4_turn_rloo_reference_on_cpu import _random_case, _reference
from verl.trainer.ppo.core_algos import compute_turn_rloo_advantage


def _inject_past_end(rows, rng, length):
    """Append 1-2 spans starting at/after the row end to ~half the rows (what a stale producer might emit)."""
    injected = []
    for spans, mask, rewards in rows:
        spans = list(spans)
        if rng.random() < 0.5:
            spans += [(length + int(rng.integers(0, 4)), length + 5)] * int(rng.integers(1, 3))
        injected.append((spans, mask, rewards))
    return injected


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
@pytest.mark.parametrize("gamma,norm", [(1.0, False), (0.9, False), (0.9, True), (0.5, True)])
def test_past_end_spans_are_ignored_in_ragged_groups(seed, gamma, norm):
    rng = np.random.default_rng(seed)
    length = 12
    rows, index = _random_case(rng, bs_groups=4, length=length)
    dirty = _inject_past_end(rows, rng, length)
    assert any(len(d[0]) > len(c[0]) for d, c in zip(dirty, rows, strict=True)), "nothing injected"
    rewards = torch.tensor(np.stack([r for _, _, r in dirty]), dtype=torch.float32)
    mask = torch.tensor(np.stack([m for _, m, _ in dirty]), dtype=torch.float32)
    spans = np.empty(len(dirty), dtype=object)
    for i, (s, _, _) in enumerate(dirty):
        spans[i] = s
    cfg = type("C", (), {"gamma": gamma})()
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans, config=cfg, norm_adv_by_std=norm)
    # the reference sees only the real spans (the clean rows): past-end spans must change nothing
    np.testing.assert_allclose(adv.numpy(), _reference(rows, index, gamma, norm), atol=1e-4, rtol=1e-4)


def test_span_straddling_the_row_end_is_kept_and_clipped():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]])
    mask = torch.ones(2, 4)
    index = np.array(["g", "g"], dtype=object)
    spans = np.empty(2, dtype=object)
    spans[0], spans[1] = [(0, 2), (3, 9)], [(0, 2), (3, 9)]  # second span runs past the end: still a turn
    adv, returns = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)
    torch.testing.assert_close(returns[:, 3], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(adv[:, 3], torch.tensor([2.0 - 0.0, 0.0 - 2.0]))


def test_row_with_only_past_end_spans_falls_back_to_one_turn():
    """All spans filtered -> the row is treated as a single segment [0, L) (the no-span rule), not as 'reached
    nothing'. Unreachable from the loop (it clips before export); documented behaviour for other producers."""
    rewards = torch.tensor([[0.0, 3.0], [0.0, 1.0]])
    mask = torch.ones(2, 2)
    index = np.array(["g", "g"], dtype=object)
    spans = np.empty(2, dtype=object)
    spans[0], spans[1] = [(5, 7)], [(0, 2)]
    adv, _ = compute_turn_rloo_advantage(rewards, mask, index, turn_spans=spans)
    torch.testing.assert_close(adv, torch.tensor([[2.0, 2.0], [-2.0, -2.0]]))
