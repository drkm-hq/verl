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
"""The update pass hands back its per-token model outputs when ``return_model_output`` is set
(``actor.skip_old_log_prob_when_on_policy``): ``train_batch`` keeps the engine's ``model_output``
instead of popping it, and ``train_mini_batch`` returns one row per input sample, in input order,
next to the metrics. Driven with mocked engines, as test_engine_workers_mini_batch_trace_on_cpu.py.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric.utils import Metric
from verl.workers.engine_workers import TrainingWorker


def _nested(rows):
    return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)


def _engine(**overrides):
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: True,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_group=lambda: None,
    )
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def _worker(engine):
    worker = SimpleNamespace(
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={},
        loss_fn=lambda: None,
        profiler=MagicMock(),
    )

    def _postprocess_output(output, *, global_token_num, delta_time, forward_only, images_seqlens):
        # the tail of TrainingWorker._postprocess_output (the metric all-gather and memory stats need a device)
        return tu.get_tensordict(
            tensor_dict=output.pop("model_output", {}), non_tensor_dict={"metrics": output["metrics"]}
        )

    worker._postprocess_output = _postprocess_output
    return worker


@pytest.mark.parametrize("return_model_output", [False, True])
def test_train_batch_keeps_model_output_only_when_asked(return_model_output):
    rows = [-torch.rand(5), -torch.rand(3)]
    engine = _engine(
        train_batch=lambda data, loss_function: {
            "model_output": {"log_probs": _nested(rows)},
            "loss": [0.5],
            "metrics": {"pg_loss": [1.0]},
        }
    )
    worker = _worker(engine)
    worker.profiler = None
    data = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(data, global_token_num=[5, 3])
    if return_model_output:
        tu.assign_non_tensor(data, return_model_output=True)

    output = TrainingWorker.train_batch(worker, data)

    assert tu.get(output, "metrics")["pg_loss"] == [1.0]
    assert ("log_probs" in output.keys()) is return_model_output
    if return_model_output:
        assert output.batch_size[0] == 2
        assert torch.equal(output["log_probs"].values(), torch.cat(rows))
    else:
        assert len(output.batch_size) == 0


def _mini_batch_outputs(rows, pg_losses):
    return [
        tu.get_tensordict(
            tensor_dict={"log_probs": _nested(chunk)},
            non_tensor_dict={"metrics": {"pg_loss": [Metric(aggregation="mean", value=pg_loss)]}},
        )
        for chunk, pg_loss in zip(rows, pg_losses, strict=True)
    ]


def test_train_mini_batch_returns_row_aligned_model_output(monkeypatch):
    rows = [-torch.rand(4), -torch.rand(6), -torch.rand(5), -torch.rand(3)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter([TensorDict({}, batch_size=[2])] * 2))
    worker = _worker(_engine())
    worker.train_batch = MagicMock(side_effect=_mini_batch_outputs([rows[:2], rows[2:]], [1.0, 3.0]))
    data = TensorDict({}, batch_size=[4])
    tu.assign_non_tensor(data, mini_batch_size=2, return_model_output=True)

    output = TrainingWorker.train_mini_batch(worker, data)

    assert output.batch_size[0] == 4
    assert [row.shape[0] for row in output["log_probs"].unbind()] == [4, 6, 5, 3]
    assert torch.equal(output["log_probs"].values(), torch.cat(rows))
    # the metrics still travel next to the outputs, one entry per mini-batch as before
    assert tu.get(output, "metrics")["pg_loss"] == [1.0, 3.0]


def test_train_mini_batch_without_the_flag_returns_metrics_only(monkeypatch):
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter([TensorDict({}, batch_size=[2])] * 2))
    worker = _worker(_engine())
    worker.train_batch = MagicMock(
        side_effect=[
            tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {"pg_loss": [Metric("mean", value=v)]}})
            for v in (1.0, 3.0)
        ]
    )
    data = TensorDict({}, batch_size=[4])
    tu.assign_non_tensor(data, mini_batch_size=2)

    output = TrainingWorker.train_mini_batch(worker, data)

    assert len(output.batch_size) == 0
    assert tu.get(output, "metrics")["pg_loss"] == [1.0, 3.0]


@pytest.mark.parametrize("extra", [{"epochs": 2}, {"dataloader_kwargs": {"shuffle": True}}])
def test_train_mini_batch_refuses_model_output_when_rows_would_not_align(monkeypatch, extra):
    rows = [-torch.rand(4), -torch.rand(6), -torch.rand(5), -torch.rand(3)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter([TensorDict({}, batch_size=[2])] * 2))
    worker = _worker(_engine())
    worker.train_batch = MagicMock(side_effect=_mini_batch_outputs([rows[:2], rows[2:]], [1.0, 3.0]))
    data = TensorDict({}, batch_size=[4])
    tu.assign_non_tensor(data, mini_batch_size=2, return_model_output=True, **extra)

    with pytest.raises(AssertionError, match="input order"):
        TrainingWorker.train_mini_batch(worker, data)


def _recording_worker():
    """train_batch stand-in that records the rows of every optimizer step (one per call)."""
    worker = _worker(_engine())
    steps = []

    def train_batch(mini_batch_td):
        steps.append(mini_batch_td["row"].tolist())
        rows = [-torch.rand(3) for _ in range(mini_batch_td.batch_size[0])]
        return tu.get_tensordict(
            tensor_dict={"log_probs": _nested(rows)},
            non_tensor_dict={"metrics": {"pg_loss": [Metric(aggregation="mean", value=1.0)]}},
        )

    worker.train_batch = train_batch
    worker.steps = steps
    return worker


def _rows(num_rows, **non_tensor):
    data = TensorDict({"row": torch.arange(num_rows)}, batch_size=[num_rows])
    tu.assign_non_tensor(data, epochs=1, dataloader_kwargs={"shuffle": False}, **non_tensor)
    return data


def test_two_mini_batches_under_the_skip_are_refused_before_any_step():
    """The second mini-batch would be trained after the first optimizer step, with ratio 1 and no clipping;
    the worker refuses the whole update (real tu.make_iterator)."""
    worker = _recording_worker()
    data = _rows(4, mini_batch_size=2, return_model_output=True, on_policy_skip_old_log_prob=True)

    with pytest.raises(AssertionError, match="one optimizer step"):
        TrainingWorker.train_mini_batch(worker, data)
    assert worker.steps == []


def test_one_mini_batch_under_the_skip_is_accepted():
    worker = _recording_worker()
    data = _rows(4, mini_batch_size=4, return_model_output=True, on_policy_skip_old_log_prob=True)

    output = TrainingWorker.train_mini_batch(worker, data)

    assert worker.steps == [[0, 1, 2, 3]]
    assert output.batch_size[0] == 4


def test_two_mini_batches_without_the_skip_still_run():
    worker = _recording_worker()
    data = _rows(4, mini_batch_size=2)

    TrainingWorker.train_mini_batch(worker, data)

    assert worker.steps == [[0, 1], [2, 3]]
