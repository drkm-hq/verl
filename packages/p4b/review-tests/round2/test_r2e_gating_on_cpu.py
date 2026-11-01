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
"""R2e (round-2 finding 2 and the D1 gate): ``adv_estimator_takes_turn_spans`` on partials, enums, callable
classes, ufuncs, a ``functools.wraps`` wrapper with an incompatible own signature and a plain ``**kwargs`` wrapper;
V0 ``compute_advantage`` forwarding to custom registered estimators; the V1 ``_compute_advantage`` fetch / lift /
metric for custom estimators, ragged span rows, no-span rows and an unknown estimator."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

import verl.trainer.ppo.core_algos as core_algos
import verl.trainer.ppo.v1.trainer_base as trainer_base_module
from verl import DataProto
from verl.trainer.ppo.core_algos import adv_estimator_takes_turn_spans as takes
from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage as rloo_fn
from verl.trainer.ppo.core_algos import compute_turn_rloo_advantage as turn_fn
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

SPANS = [[(0, 2), (3, 5)], [(0, 2), (3, 4)], None]
CFG = SimpleNamespace(gamma=1.0, lam=1.0)


def recording(takes_spans):
    calls = []
    if takes_spans:

        def est(token_level_rewards, response_mask, index, turn_spans=None, config=None, **kwargs):
            calls.append({"turn_spans": turn_spans, **kwargs})
            return turn_fn(token_level_rewards, response_mask, index, turn_spans=turn_spans, config=config)

    else:

        def est(token_level_rewards, response_mask, index, config=None, **kwargs):
            calls.append(dict(kwargs))
            return rloo_fn(token_level_rewards, response_mask, index)

    return est, calls


class NoSpans:
    def __call__(self, token_level_rewards, response_mask, index, config=None, **kwargs):
        return rloo_fn(token_level_rewards, response_mask, index)


class WithSpans:
    def __call__(self, token_level_rewards, response_mask, index, turn_spans=None, config=None, **kwargs):
        return turn_fn(token_level_rewards, response_mask, index, turn_spans=turn_spans, config=config)


def test_signature_probe_on_the_callable_shapes_a_user_can_register():
    assert takes(functools.partial(turn_fn, norm_adv_by_std=True))
    assert not takes(functools.partial(rloo_fn, epsilon=1e-5))
    assert takes(WithSpans()) and not takes(NoSpans())
    assert not takes(np.add), "a ufunc: no turn_spans, no exception"
    for name in core_algos.ADV_ESTIMATOR_REGISTRY:
        assert takes(name) is (name == "turn_rloo"), name
    assert not takes(core_algos.AdvantageEstimator.RLOO) and not takes(core_algos.AdvantageEstimator.GAE)
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        takes("no_such_estimator")


def column(rows):
    spans = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        spans[i] = row
    return spans


def v0_batch():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    data = DataProto(batch=TensorDict({"token_level_rewards": rewards, "response_mask": mask}, batch_size=[3]))
    data.non_tensor_batch["uid"] = np.array(["g"] * 3, dtype=object)
    data.non_tensor_batch["generated_turn_spans"] = column(SPANS)
    return data


@pytest.mark.parametrize("takes_spans", [False, True])
def test_v0_forwards_the_column_only_to_estimators_that_declare_turn_spans(monkeypatch, takes_spans):
    est, calls = recording(takes_spans)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_custom", est)
    out = compute_advantage(v0_batch(), adv_estimator="r2e_custom", config=CFG)
    assert out.batch["advantages"].shape == (3, 5)
    if takes_spans:
        spans = calls[0]["turn_spans"]
        assert list(spans[0]) == SPANS[0] and list(spans[1]) == SPANS[1] and spans[2] is None
    else:
        assert "turn_spans" not in calls[0]


def test_v0_wraps_wrapper_whose_own_signature_cannot_take_the_kwarg_gets_no_spans(monkeypatch):
    """LOW 5, closed by 988a10bc: the outermost callable's own signature decides. A ``functools.wraps`` wrapper
    without ``**kwargs`` is not asked for spans (rloo behaviour, no TypeError); a transparent ``**kwargs``
    wrapper over an estimator declaring ``turn_spans`` is, and gets them."""

    @functools.wraps(turn_fn)
    def narrow(token_level_rewards, response_mask, index, config=None):
        return turn_fn(token_level_rewards, response_mask, index, config=config)

    @functools.wraps(turn_fn)
    def transparent(*args, **kwargs):
        return turn_fn(*args, **kwargs)

    @functools.wraps(rloo_fn)
    def transparent_over_rloo(*args, **kwargs):
        return rloo_fn(*args, **kwargs)

    assert not takes(narrow) and takes(transparent) and not takes(transparent_over_rloo)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_narrow", narrow)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_transparent", transparent)
    rloo = compute_advantage(v0_batch(), adv_estimator="rloo", config=CFG).batch["advantages"]
    turn = compute_advantage(v0_batch(), adv_estimator="turn_rloo", config=CFG).batch["advantages"]
    torch.testing.assert_close(compute_advantage(v0_batch(), adv_estimator="r2e_narrow", config=CFG).batch["advantages"], rloo)
    torch.testing.assert_close(compute_advantage(v0_batch(), adv_estimator="r2e_transparent", config=CFG).batch["advantages"], turn)
    assert not torch.allclose(turn, rloo)


def test_v0_plain_kwargs_wrapper_without_wraps_degrades_to_rloo_silently(monkeypatch):
    def wrapper(*args, **kwargs):
        return turn_fn(*args, **kwargs)

    assert not takes(wrapper)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_plain", wrapper)
    plain = compute_advantage(v0_batch(), adv_estimator="r2e_plain", config=CFG).batch["advantages"]
    rloo = compute_advantage(v0_batch(), adv_estimator="rloo", config=CFG).batch["advantages"]
    turn = compute_advantage(v0_batch(), adv_estimator="turn_rloo", config=CFG).batch["advantages"]
    torch.testing.assert_close(plain, rloo)
    assert not torch.allclose(turn, rloo), "the spans did reach turn_rloo itself"


class _Trainer(PPOTrainer):
    def on_sample_end(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass


class _Meta:
    def __init__(self, keys):
        self.keys, self.partition_id = keys, "train"

    def __len__(self):
        return len(self.keys)


def nested(rows, dtype):
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=dtype) for r in rows], layout=torch.jagged)


def run_v1(monkeypatch, rows, adv_estimator):
    stored, metrics, selected = {}, {}, []

    def kv_batch_get(keys, partition_id, select_fields):
        selected.extend(select_fields)
        cols = {
            "uid": NonTensorStack.from_list([NonTensorData(u) for u, *_ in rows]),
            "response_mask": nested([m for _, m, _, _ in rows], torch.float32),
            "rm_scores": nested([s for _, _, s, _ in rows], torch.float32),
            "extra_fields": NonTensorStack.from_list([NonTensorData(e) for *_, e in rows]),
        }
        return TensorDict({k: v for k, v in cols.items() if k in select_fields}, batch_size=[len(rows)])

    def kv_batch_put(keys, partition_id, fields):
        stored.update(fields)
        return _Meta(keys)

    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_get", kv_batch_get)
    monkeypatch.setattr(trainer_base_module.tq, "kv_batch_put", kv_batch_put)
    trainer = object.__new__(_Trainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": adv_estimator, "gamma": 1.0, "lam": 1.0, "use_kl_in_reward": False},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer._compute_advantage(_Meta([f"u{i}_0_0" for i in range(len(rows))]), metrics=metrics)
    return stored, metrics, selected


ROWS = [
    ("g", [1, 1, 0, 1, 1], [0.0, 1.0, 0.0, 0.0, 2.0], {"generated_turn_spans": [(0, 2), (3, 5)]}),
    ("g", [1, 1, 0, 1], [0.0, 0.0, 0.0, 1.0], {"tool_rewards": []}),
    ("g", [1, 1], [0.0, 3.0], None),
]


def test_v1_custom_estimator_with_turn_spans_gets_the_lift_and_an_int_metric(monkeypatch):
    est, calls = recording(True)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_custom", est)
    stored, metrics, selected = run_v1(monkeypatch, ROWS, "r2e_custom")
    assert "extra_fields" in selected and metrics == {"adv_estimator/rows_with_turn_spans": 1}
    assert type(metrics["adv_estimator/rows_with_turn_spans"]) is int
    spans = calls[0]["turn_spans"]
    assert list(spans[0]) == [(0, 2), (3, 5)] and spans[1] is None and spans[2] is None
    assert stored["advantages"].is_nested


def test_v1_custom_estimator_without_turn_spans_never_fetches_extra_fields(monkeypatch):
    est, calls = recording(False)
    monkeypatch.setitem(core_algos.ADV_ESTIMATOR_REGISTRY, "r2e_custom", est)
    stored, metrics, selected = run_v1(monkeypatch, ROWS, "r2e_custom")
    assert "extra_fields" not in selected and "adv_estimator/rows_with_turn_spans" not in metrics
    assert "turn_spans" not in calls[0]


def test_v1_turn_rloo_with_no_span_rows_lifts_nothing_logs_zero_and_equals_rloo(monkeypatch):
    rows = [(u, m, s, {"tool_rewards": []}) for u, m, s, _ in ROWS]
    stored, metrics, selected = run_v1(monkeypatch, rows, "turn_rloo")
    assert "extra_fields" in selected and metrics["adv_estimator/rows_with_turn_spans"] == 0
    rloo, _, _ = run_v1(monkeypatch, rows, "rloo")
    torch.testing.assert_close(stored["advantages"].to_padded_tensor(0.0), rloo["advantages"].to_padded_tensor(0.0))


def test_v1_unknown_estimator_fails_in_the_gate_before_the_queue_fetch(monkeypatch):
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        run_v1(monkeypatch, ROWS, "no_such_estimator")
