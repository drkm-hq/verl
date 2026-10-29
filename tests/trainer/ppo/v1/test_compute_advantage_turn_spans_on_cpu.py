# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""The V1 trainer lifts ``extra_fields["generated_turn_spans"]`` out of the TransferQueue row so that
``turn_rloo`` sees the turn spans; rows without spans keep working; estimators that do not declare
``turn_spans`` never fetch the column."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

import verl.trainer.ppo.v1.trainer_base as trainer_base_module
from verl.trainer.ppo.core_algos import compute_turn_rloo_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _Trainer(PPOTrainer):
    """Concrete shell: only ``_compute_advantage`` runs, on the attributes the test sets."""

    def on_sample_end(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass


class _BatchMeta:
    def __init__(self, keys):
        self.keys = keys
        self.partition_id = "train"

    def __len__(self):
        return len(self.keys)


def _nested(rows, dtype):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


def _run_compute_advantage(monkeypatch, rows, adv_estimator="turn_rloo"):
    """``rows``: (uid, response_mask, rm_scores, extra_fields) per sample, response lengths may differ.

    Returns the fields written back, the metrics and the fields the trainer selected from the queue.
    """
    stored, metrics, selected = {}, {}, []

    def kv_batch_get(keys, partition_id, select_fields):
        selected.extend(select_fields)
        columns = {
            "uid": NonTensorStack.from_list([NonTensorData(uid) for uid, *_ in rows]),
            "response_mask": _nested([m for _, m, _, _ in rows], torch.float32),
            "rm_scores": _nested([s for _, _, s, _ in rows], torch.float32),
            "extra_fields": NonTensorStack.from_list([NonTensorData(ef) for *_, ef in rows]),
        }
        return TensorDict({k: v for k, v in columns.items() if k in select_fields}, batch_size=[len(rows)])

    def kv_batch_put(keys, partition_id, fields):
        stored.update(fields)
        return _BatchMeta(keys)

    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", kv_batch_get)
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", kv_batch_put)

    trainer = object.__new__(_Trainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv_estimator, "gamma": 1.0, "lam": 1.0, "use_kl_in_reward": False},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer._compute_advantage(_BatchMeta([f"u{i}_0_0" for i in range(len(rows))]), metrics=metrics)
    return stored, metrics, selected


def test_v1_compute_advantage_passes_generated_turn_spans_to_turn_rloo(monkeypatch):
    # two members of one prompt group, dense rewards on the last generated token of each turn
    rows = [
        ("g", [1, 1, 0, 1, 1], [0.0, 1.0, 0.0, 0.0, 2.0], {"generated_turn_spans": [(0, 2), (3, 5)]}),
        ("g", [1, 1, 0, 1], [0.0, 0.0, 0.0, 1.0], {"generated_turn_spans": [(0, 2), (3, 4)]}),
    ]
    stored, metrics, selected = _run_compute_advantage(monkeypatch, rows)

    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 0, 1, 0]], dtype=torch.float32)
    spans = np.empty(2, dtype=object)
    spans[0], spans[1] = [(0, 2), (3, 5)], [(0, 2), (3, 4)]
    expected, _ = compute_turn_rloo_advantage(rewards, mask, np.array(["g", "g"]), turn_spans=spans)
    assert stored["advantages"].is_nested
    torch.testing.assert_close(stored["advantages"][0], expected[0])
    torch.testing.assert_close(stored["advantages"][1], expected[1, :4])
    assert stored["advantages"][0][0].item() == pytest.approx(3.0 - 1.0)  # turn-1 return-to-go minus the other's
    assert stored["advantages"][0][3].item() == pytest.approx(2.0 - 1.0)  # turn 2 likewise
    assert "extra_fields" in selected
    assert metrics["adv_estimator/rows_with_turn_spans"] == 2


def test_v1_compute_advantage_without_spans_keeps_rloo_behaviour(monkeypatch):
    rows = [
        ("g", [1, 1, 1], [0.0, 0.0, 3.0], {"tool_rewards": []}),
        ("g", [1, 1], [0.0, 1.0], {"tool_rewards": []}),
    ]
    stored, metrics, _ = _run_compute_advantage(monkeypatch, rows)
    torch.testing.assert_close(stored["advantages"][0], torch.tensor([2.0, 2.0, 2.0]))
    torch.testing.assert_close(stored["advantages"][1], torch.tensor([-2.0, -2.0]))
    assert metrics["adv_estimator/rows_with_turn_spans"] == 0


def test_v1_compute_advantage_tolerates_a_row_whose_extra_fields_is_not_a_dict(monkeypatch):
    rows = [
        ("g", [1, 1, 0, 1, 1], [0.0, 1.0, 0.0, 0.0, 2.0], {"generated_turn_spans": [(0, 2), (3, 5)]}),
        ("g", [1, 1], [0.0, 1.0], None),
    ]
    stored, metrics, _ = _run_compute_advantage(monkeypatch, rows)
    # row 1 has no spans: one turn [0, 2) with return 1; row 0's turn-1 return is 3
    torch.testing.assert_close(stored["advantages"][1], torch.tensor([1.0 - 3.0, 1.0 - 3.0]))
    torch.testing.assert_close(stored["advantages"][0][:2], torch.tensor([3.0 - 1.0, 3.0 - 1.0]))
    assert metrics["adv_estimator/rows_with_turn_spans"] == 1


@pytest.mark.parametrize("adv_estimator", ["rloo", "reinforce_plus_plus", "grpo"])
def test_v1_compute_advantage_does_not_fetch_extra_fields_for_estimators_without_turn_spans(monkeypatch, adv_estimator):
    """The disabled path: no ``extra_fields`` round trip through the queue, no span lift, no metric."""
    rows = [
        ("g", [1, 1, 1], [0.0, 0.0, 3.0], {"generated_turn_spans": [(0, 3)]}),
        ("g", [1, 1], [0.0, 1.0], {"generated_turn_spans": [(0, 2)]}),
    ]
    stored, metrics, selected = _run_compute_advantage(monkeypatch, rows, adv_estimator=adv_estimator)
    assert "extra_fields" not in selected
    assert "adv_estimator/rows_with_turn_spans" not in metrics
    assert stored["advantages"].is_nested and stored["advantages"][0].shape == (3,)
