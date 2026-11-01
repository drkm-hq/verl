# Adversarial review tests for contrib/p2-oldlogprob-skip (worker-side guard).
"""train_mini_batch must refuse to run the on-policy skip over more than one mini-batch: the second
mini-batch runs after an optimizer step, so `old_log_prob = log_prob.detach()` is no longer the sampling
log-prob and the clipping safety net is silently gone. The branch only asserts epochs == 1 and no shuffle
(engine_workers.py:375-378); this test asks for the missing condition with the real tu.make_iterator."""

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


def _worker():
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: True,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_group=lambda: None,
    )
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
    steps = []

    def train_batch(mini_batch_td):
        # one optimizer step per call (engine/base.py:125-128); record which rows it saw
        steps.append(mini_batch_td["row"].tolist())
        rows = [-torch.rand(3) for _ in range(mini_batch_td.batch_size[0])]
        return tu.get_tensordict(
            tensor_dict={"log_probs": _nested(rows)},
            non_tensor_dict={
                "metrics": {"pg_loss": [Metric(aggregation="mean", value=1.0)]}
            },
        )

    worker.train_batch = train_batch
    worker.steps = steps
    return worker


def _data(rows, **non_tensor):
    data = TensorDict({"row": torch.arange(rows)}, batch_size=[rows])
    tu.assign_non_tensor(data, **non_tensor)
    return data


def test_two_mini_batches_under_the_skip_must_be_refused():
    worker = _worker()
    data = _data(
        4,
        mini_batch_size=2,
        epochs=1,
        dataloader_kwargs={"shuffle": False},
        return_model_output=True,
        on_policy_skip_old_log_prob=True,
    )
    with pytest.raises(AssertionError):
        TrainingWorker.train_mini_batch(worker, data)
    assert worker.steps == [[0, 1]] or worker.steps == [], (
        "no second optimizer step may run under the skip"
    )


def test_two_mini_batches_currently_run_two_optimizer_steps_and_return_aligned_rows():
    """What the branch does today (documents the silent path if the previous test fails)."""
    worker = _worker()
    data = _data(
        4,
        mini_batch_size=2,
        epochs=1,
        dataloader_kwargs={"shuffle": False},
        return_model_output=True,
        on_policy_skip_old_log_prob=True,
    )
    try:
        output = TrainingWorker.train_mini_batch(worker, data)
    except AssertionError:
        pytest.skip("guard present")
    assert worker.steps == [[0, 1], [2, 3]]
    assert output.batch_size[0] == 4  # row alignment holds, exactness does not


def test_one_mini_batch_is_accepted():
    worker = _worker()
    data = _data(
        4,
        mini_batch_size=4,
        epochs=1,
        dataloader_kwargs={"shuffle": False},
        return_model_output=True,
        on_policy_skip_old_log_prob=True,
    )
    output = TrainingWorker.train_mini_batch(worker, data)
    assert worker.steps == [[0, 1, 2, 3]] and output.batch_size[0] == 4
