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
"""R6: the V1 ``_compute_advantage`` lift of ``extra_fields["generated_turn_spans"]`` with a synthetic padding
row (padding_utils copies the first real row's extra_fields, so its spans outrun its 1-token response),
rows from mixed agent loops, a non-dict extra_fields row, and the real TQ-side packing of the row."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

import verl.trainer.ppo.core_algos as core_algos
import verl.trainer.ppo.v1.trainer_base as trainer_base_module
from verl.trainer.ppo.core_algos import compute_turn_rloo_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "turn_rloo", compute_turn_rloo_advantage)


class _Trainer(PPOTrainer):
    def on_sample_end(self, *a, **k):
        pass

    def on_step_end(self, *a, **k):
        pass


class _Meta:
    def __init__(self, keys):
        self.keys, self.partition_id = keys, "train"

    def __len__(self):
        return len(self.keys)


def _nested(rows, dtype):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


def _run(monkeypatch, rows):
    stored = {}

    def kv_batch_get(keys, partition_id, select_fields):
        return TensorDict(
            {
                "uid": NonTensorStack.from_list([NonTensorData(u) for u, *_ in rows]),
                "response_mask": _nested([m for _, m, _, _ in rows], torch.float32),
                "rm_scores": _nested([s for _, _, s, _ in rows], torch.float32),
                "extra_fields": NonTensorStack.from_list([NonTensorData(e) for *_, e in rows]),
            },
            batch_size=[len(rows)],
        )

    def kv_batch_put(keys, partition_id, fields):
        stored.update(fields)
        return _Meta(keys)

    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", kv_batch_get)
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", kv_batch_put)
    trainer = object.__new__(_Trainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "turn_rloo", "gamma": 1.0, "lam": 1.0, "use_kl_in_reward": False},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer._compute_advantage(_Meta([f"u{i}_0_0" for i in range(len(rows))]), metrics={})
    return stored


REAL = [
    ("g", [1, 1, 0, 1, 1], [0.0, 1.0, 0.0, 0.0, 2.0], {"generated_turn_spans": [(0, 2), (3, 5)], "episode_done": True}),
    ("g", [1, 1, 0, 1], [0.0, 0.0, 0.0, 1.0], {"generated_turn_spans": [(0, 2), (3, 4)], "episode_done": False}),
]


def test_padding_row_copied_from_the_first_real_row_does_not_crash_or_change_real_rows(monkeypatch):
    base = _run(monkeypatch, REAL)
    pad = ("pad0123", [0], [0.0], dict(REAL[0][3]))  # padding_utils: template extra_fields, 1-token response
    with_pad = _run(monkeypatch, REAL + [pad])
    torch.testing.assert_close(with_pad["advantages"][0], base["advantages"][0])
    torch.testing.assert_close(with_pad["advantages"][1], base["advantages"][1])
    assert with_pad["advantages"][2].tolist() == [0.0]


def test_mixed_rows_with_and_without_spans(monkeypatch):
    rows = REAL + [("h", [1, 1, 1], [0.0, 0.0, 3.0], {"tool_rewards": []}), ("h", [1, 1], [0.0, 1.0], {})]
    stored = _run(monkeypatch, rows)
    torch.testing.assert_close(stored["advantages"][2], torch.tensor([2.0, 2.0, 2.0]))
    torch.testing.assert_close(stored["advantages"][3], torch.tensor([-2.0, -2.0]))


def test_non_dict_extra_fields_row_is_tolerated(monkeypatch):
    """_compute_metrics guards ``isinstance(extra_field, dict)``; the new lift does not."""
    rows = REAL + [("h", [1, 1], [0.0, 1.0], None)]
    stored = _run(monkeypatch, rows)
    assert stored["advantages"][2].shape == (2,)


def test_tq_side_packing_keeps_spans_as_tuples_through_nontensorstack():
    fields = [
        {"rm_scores": torch.tensor([0.0, 0.5]), "extra_fields": {"generated_turn_spans": [(0, 2)], "episode_done": True}},
        {"rm_scores": torch.tensor([1.0]), "extra_fields": {"generated_turn_spans": [], "episode_done": False}},
    ]
    td = list_of_dict_to_tensordict(fields)
    # td["extra_fields"] is a tensordict LinkedList (no .tolist()); td.get(...) is the NonTensorStack (has it).
    # trainer_base._compute_advantage relies on .tolist() after pop(); list(...) works on both forms.
    assert not hasattr(td["extra_fields"], "tolist") and hasattr(td.get("extra_fields"), "tolist")
    rows = td.get("extra_fields").tolist()
    assert rows == list(td["extra_fields"])
    assert rows[0]["generated_turn_spans"] == [(0, 2)] and rows[1]["generated_turn_spans"] == []
    assert isinstance(rows[0]["generated_turn_spans"][0], tuple)
    assert np.array([r["episode_done"] for r in rows]).tolist() == [True, False]
