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
"""R1: ``compute_advantage`` now forwards ``turn_spans`` to EVERY else-branch estimator whenever the
``generated_turn_spans`` column exists, and the V0 ``_postprocess`` adds that column (None-filled) to
every agent-loop batch via ``default_extra_keys``. An estimator without ``**kwargs`` must still work."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import verl.trainer.ppo.core_algos as core_algos
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage


def _batch(with_spans_column: bool, spans=None):
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    batch = TensorDict({"token_level_rewards": rewards, "response_mask": mask}, batch_size=[4])
    non_tensor = {"uid": np.array(["a", "a", "b", "b"], dtype=object)}
    if with_spans_column:
        col = np.empty(4, dtype=object)
        col[:] = spans if spans is not None else [None] * 4  # what every V0 single-turn batch now carries
        non_tensor["generated_turn_spans"] = col
    return DataProto(batch=batch, non_tensor_batch=non_tensor)


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    # tests/trainer/ppo/test_core_algos_on_cpu.py::TestRegisterAdvEst empties the global registry; pin what we use.
    monkeypatch.setitem(
        core_algos.ADV_ESTIMATOR_REGISTRY, "grpo_vectorized", core_algos.compute_grpo_vectorized_outcome_advantage
    )
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "rloo", core_algos.compute_rloo_outcome_advantage)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "turn_rloo", core_algos.compute_turn_rloo_advantage)


def test_grpo_vectorized_still_works_when_the_spans_column_is_none_filled():
    """A V0 single-turn agent-loop batch with ``adv_estimator=grpo_vectorized`` (flag OFF)."""
    expected = compute_advantage(_batch(with_spans_column=False), adv_estimator="grpo_vectorized").batch["advantages"]
    got = compute_advantage(_batch(with_spans_column=True), adv_estimator="grpo_vectorized").batch["advantages"]
    torch.testing.assert_close(got, expected)


def test_user_registered_estimator_without_var_kwargs_still_works(monkeypatch):
    """Third-party estimators registered by name (the enum docstring invites this) may lack ``**kwargs``."""

    def plain_rloo(token_level_rewards, response_mask, index, config=None):
        return core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)

    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "plain_rloo", plain_rloo)
    expected = compute_advantage(_batch(with_spans_column=False), adv_estimator="plain_rloo").batch["advantages"]
    got = compute_advantage(_batch(with_spans_column=True), adv_estimator="plain_rloo").batch["advantages"]
    torch.testing.assert_close(got, expected)


def test_rloo_unchanged_by_the_spans_column():
    expected = compute_advantage(_batch(with_spans_column=False), adv_estimator="rloo").batch["advantages"]
    got = compute_advantage(_batch(with_spans_column=True), adv_estimator="rloo").batch["advantages"]
    torch.testing.assert_close(got, expected)


def test_turn_rloo_receives_spans_through_compute_advantage():
    # group a: row 0 segments [0,3)=1, [3,4)=0 -> G=[1,0]; row 1 segments [0,3)=0, [3,4)=0.5 -> G=[0.5,0.5]
    spans = [[(0, 3), (3, 4)], [(0, 2), (3, 4)], [(0, 2)], [(0, 4)]]
    out = compute_advantage(_batch(True, spans), adv_estimator="turn_rloo", config=None).batch["advantages"]
    assert out.shape == (4, 4)
    assert out[1, 0].item() == pytest.approx(0.5 - 1.0) and out[1, 3].item() == pytest.approx(0.5 - 0.0)
