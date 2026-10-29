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
"""``PPOTrainer._validate`` when agent loop workers die during validation (in-memory TransferQueue stand-in).

Mirrors the independent review's reproduction: the real ``_validate`` / ``_val_metrics_update`` /
``_consume_recovery_metrics`` and the real ``ReplayBuffer`` run against an in-memory transport whose
``kv_batch_get`` keeps the installed client's guard (an empty key list raises ``ValueError``), so the trainer
must never fetch, decode or clear for a validation batch without rows. A dead worker is simulated by a
manager that never reports some (or all) prompts and a liveness probe that reports every in-flight prompt
as orphaned.
"""

import logging
from collections import Counter, defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.agent_loop_tq import _chunk_uids
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

NESTED_FIELDS = ("prompts", "responses")
TENSOR_FIELDS = ("rm_scores", "num_turns")


class MemoryTQ:
    """Just enough of the TransferQueue API for ``_validate`` and the replay buffer: tags and fields per key."""

    def __init__(self):
        self.tags: dict[str, dict[str, dict]] = defaultdict(dict)
        self.fields: dict[str, dict[str, dict]] = defaultdict(dict)
        self.batch_get_calls: list[list[str]] = []
        self.clear_calls: list[list[str]] = []

    def kv_put(self, key, partition_id, tag=None, fields=None, **kwargs):
        if tag:
            self.tags[partition_id].setdefault(str(key), {}).update(tag)
        if fields:
            self.fields[partition_id].setdefault(str(key), {}).update(fields)

    def kv_batch_put(self, keys, partition_id, tags, fields=None, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            self.kv_put(key, partition_id, tag)

    def kv_list(self, partition_id=None):
        data = deepcopy(self.tags)
        return data if partition_id is None else {partition_id: data.get(partition_id, {})}

    def kv_clear(self, keys, partition_id):
        self.clear_calls.append(list(keys))
        for key in keys:
            self.tags[partition_id].pop(key, None)
            self.fields[partition_id].pop(key, None)

    def kv_batch_get(self, keys, partition_id, select_fields=None):
        if isinstance(keys, list) and len(keys) < 1:
            raise ValueError("Received an empty list as keys.")  # the installed client's input guard
        self.batch_get_calls.append(list(keys))
        rows = [self.fields[partition_id][key] for key in keys]
        out = {}
        for name in select_fields:
            values = [row[name] for row in rows]
            if name in NESTED_FIELDS:
                out[name] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            elif name in TENSOR_FIELDS:
                out[name] = torch.stack(values)
            else:
                out[name] = np.array(values, dtype=object)
        return out

    def install(self, monkeypatch) -> None:
        for name in ("kv_put", "kv_batch_put", "kv_list", "kv_clear", "kv_batch_get"):
            monkeypatch.setattr(trainer_base.tq, name, getattr(self, name))


class Workers:
    """Agent loop manager stand-in: the first ``healthy_per_batch`` prompts of every dispatched batch get one
    finished session with a row; the others never report (their worker is dead)."""

    def __init__(self, memory: MemoryTQ, healthy_per_batch: int = 0, score: float = 1.0):
        self.memory = memory
        self.healthy_per_batch = healthy_per_batch
        self.score = score
        self.dispatched: list[list[str]] = []
        self.recovery_metrics: Counter = Counter()

    def generate_sequences(self, batch) -> None:
        uids = _chunk_uids(batch)
        self.dispatched.append(uids)
        for uid in uids[: self.healthy_per_batch]:
            self.memory.kv_put(
                f"{uid}_0_0",
                "val",
                tag={"is_prompt": False, "seq_len": 5, "global_steps": 1},
                fields={
                    "uid": uid,
                    "prompts": torch.tensor([5, 6, 7]),
                    "responses": torch.tensor([8, 9]),
                    "rm_scores": torch.tensor([0.0, self.score]),
                    "num_turns": torch.tensor(1),
                    "reward_model": {"ground_truth": "42"},
                    "data_source": "test",
                    "extra_fields": {"reward_extra_info": {"acc": self.score}},
                },
            )
            self.memory.kv_put(uid, "val", tag={"status": "finished"})


class Recorder:
    def __init__(self, result=None, raises: bool = False):
        self.calls: list[tuple] = []
        self.result = result
        self.raises = raises

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises:
            raise AssertionError("must not be called for a validation batch without rows")
        return args[0] if self.result == "first_arg" else self.result


def _decode(ids, skip_special_tokens=True) -> str:
    return " ".join(str(int(token)) for token in ids if int(token) != 0)


def _make_trainer(workers: Workers, batches, n_val: int = 1, colocated: bool = False):
    """Only what ``_validate`` touches, with the real ``_validate`` / ``_val_metrics_update`` /
    ``_consume_recovery_metrics`` bound to it and the logging / dump sinks recorded."""
    trainer = SimpleNamespace(
        global_steps=1,
        val_dataloader=[
            {"raw_prompt": list(prompts), "index": list(range(len(prompts))), "data_source": ["test"] * len(prompts)}
            for prompts in batches
        ],
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": n_val}}},
                "trainer": {"log_val_generations": 1, "validation_data_dir": "recorded"},
            }
        ),
        agent_loop_manager=workers,
        replay_buffer=ReplayBuffer(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=0.001,
            reissue_fn=lambda *args: [],
            liveness_fn=lambda uids: set(uids),  # every prompt still in flight at the probe lost its worker
            liveness_period_s=0,
        ),
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=None if colocated else [object()]),
        checkpoint_manager=SimpleNamespace(sleep_replicas=Recorder(), update_weights=Recorder()),
        tokenizer=SimpleNamespace(pad_token_id=0, decode=_decode),
        _compute_reward_colocate=Recorder(raises=True),
        _maybe_log_val_generations=Recorder(),
        _dump_generations=Recorder(),
    )
    for name in ("_validate", "_val_metrics_update", "_consume_recovery_metrics"):
        setattr(trainer, name, getattr(PPOTrainer, name).__get__(trainer))
    return trainer


def _setup(monkeypatch, batches, healthy_per_batch: int = 0, **trainer_kwargs):
    memory = MemoryTQ()
    memory.install(monkeypatch)
    workers = Workers(memory, healthy_per_batch=healthy_per_batch)
    return memory, workers, _make_trainer(workers, batches, **trainer_kwargs)


def test_every_group_of_a_batch_loses_its_worker_with_a_remote_reward_loop(monkeypatch, caplog):
    """The review's probe, now passing: both prompts of the batch are owned by a dead worker, the probe fails
    them in place, ``sample`` returns no rows and ``_validate`` completes without fetching, decoding, dumping
    or clearing an empty key list; the failed count comes back with the validation metrics. No session of the
    data source materialised, so nothing defines ``acc`` and no accuracy is reported for it."""
    caplog.set_level(logging.WARNING, logger=trainer_base.__name__)
    memory, workers, trainer = _setup(monkeypatch, batches=(["a", "b"],))
    metrics = trainer._validate()
    assert metrics == {"validation/inflight_dead_worker_failed": 2}
    assert memory.batch_get_calls == [] and [] not in memory.clear_calls and memory.tags["val"] == {}
    assert trainer._compute_reward_colocate.calls == [] and trainer._dump_generations.calls == []
    assert trainer._maybe_log_val_generations.calls == [((), {"inputs": [], "outputs": [], "scores": []})]
    assert any("no rows to score" in record.message for record in caplog.records)


def test_every_group_lost_with_a_colocated_reward_model_skips_the_reward_pass(monkeypatch):
    memory, workers, trainer = _setup(monkeypatch, batches=(["a"],), colocated=True)
    assert trainer._validate() == {"validation/inflight_dead_worker_failed": 1}
    assert trainer._compute_reward_colocate.calls == [] and memory.batch_get_calls == []
    assert trainer.checkpoint_manager.sleep_replicas.calls == []
    assert trainer.checkpoint_manager.update_weights.calls == []


@pytest.mark.parametrize("colocated", [False, True])
def test_mixed_batch_decodes_the_healthy_row_and_scores_the_failed_session_zero(monkeypatch, colocated):
    """One group finished, one lost its worker: the healthy row is fetched, decoded, logged and dumped with
    aligned lists, and the failed session counts as ``acc = 0`` for its data source."""
    memory, workers, trainer = _setup(monkeypatch, batches=(["a", "b"],), healthy_per_batch=1, colocated=colocated)
    trainer._compute_reward_colocate = Recorder(result="first_arg")
    metrics = trainer._validate()
    row = f"{workers.dispatched[0][0]}_0_0"
    assert metrics["validation/inflight_dead_worker_failed"] == 1
    assert metrics["val-core/test/acc/mean@1"] == pytest.approx(0.5), "healthy 1.0 and the failed session 0.0"
    assert metrics["val-aux/test/acc_success_only/mean@1"] == pytest.approx(1.0)
    assert metrics["val-aux/test/reward/mean@1"] == pytest.approx(1.0)
    assert metrics["val-aux/num_turns/mean"] == 1
    assert memory.batch_get_calls == [[row], [row]] and [] not in memory.clear_calls and memory.tags["val"] == {}
    assert len(trainer._compute_reward_colocate.calls) == (1 if colocated else 0)
    assert len(trainer.checkpoint_manager.update_weights.calls) == (1 if colocated else 0)
    assert trainer._maybe_log_val_generations.calls == [
        ((), {"inputs": ["5 6 7"], "outputs": ["8 9"], "scores": [1.0]})
    ]
    ((), dump) = trainer._dump_generations.calls[0]
    assert dump["inputs"] == ["5 6 7"] and dump["outputs"] == ["8 9"] and dump["gts"] == ["42"]
    assert dump["scores"] == [1.0] and dump["reward_extra_infos_dict"] == {"reward": [1.0], "acc": [1.0], "uid": [row]}


def test_single_dead_worker_owning_every_batch_counts_each_batch_and_drains_the_relaunch(monkeypatch):
    """One worker owned every validation chunk: each batch comes back empty, the failed counts accumulate
    across batches, and the manager's relaunch counters raised during validation are drained into the
    validation metrics (namespaced ``validation/``) instead of being lost or logged with the next step."""
    memory, workers, trainer = _setup(monkeypatch, batches=(["a", "b"], ["c"]))
    workers.recovery_metrics.update({"dead_workers_relaunched": 1, "prompts_orphaned_by_dead_workers": 3})
    assert trainer._validate() == {
        "validation/inflight_dead_worker_failed": 3,
        "validation/dead_workers_relaunched": 1,
        "validation/prompts_orphaned_by_dead_workers": 3,
    }
    assert workers.recovery_metrics == Counter(), "drained by the validation, not double-counted by the step"
    workers.recovery_metrics["dead_workers_relaunched"] = 2
    assert trainer._consume_recovery_metrics() == {"rollout/dead_workers_relaunched": 2}


class ScheduledWorkers(Workers):
    """``healthy_per_batch`` taken per dispatch from ``schedule`` (0 = the chunk's worker is dead)."""

    def __init__(self, memory: MemoryTQ, schedule, score: float = 1.0):
        super().__init__(memory, healthy_per_batch=0, score=score)
        self.schedule = list(schedule)

    def generate_sequences(self, batch) -> None:
        self.healthy_per_batch = self.schedule.pop(0) if self.schedule else 0
        super().generate_sequences(batch)


class LiveWorkersFailingEverySession(Workers):
    """Upstream failure shape: the worker is alive and marks every group ``failure`` without a row."""

    def generate_sequences(self, batch) -> None:
        uids = _chunk_uids(batch)
        self.dispatched.append(uids)
        for uid in uids:
            self.memory.kv_put(uid, "val", tag={"status": "failure"})


def _with_real_wandb_logger(trainer, generations_to_log: int = 1):
    """The real ``_maybe_log_val_generations`` and a real ``ValidationGenerationsLogger`` building real wandb
    tables; ``wandb.run`` is None, so nothing is sent."""
    import wandb

    from verl.utils.tracking import ValidationGenerationsLogger

    assert wandb.run is None
    trainer.config.trainer.logger = ["wandb"]
    trainer.config.trainer.log_val_generations = generations_to_log
    trainer.validation_generations_logger = ValidationGenerationsLogger()
    trainer._maybe_log_val_generations = PPOTrainer._maybe_log_val_generations.__get__(trainer)
    return trainer


@pytest.mark.parametrize("schedule", [[0, 1], [1, 0]])
def test_generations_logger_survives_an_all_lost_validation_before_or_after_a_healthy_one(monkeypatch, schedule):
    """Stock ``trainer.logger: [..., wandb]`` with ``log_val_generations > 0``: an all-lost validation logs no
    generations (one warning) and leaves the wandb table as it was, so the healthy validation before or after
    it keeps its column count."""
    memory = MemoryTQ()
    memory.install(monkeypatch)
    workers = ScheduledWorkers(memory, schedule)
    trainer = _with_real_wandb_logger(_make_trainer(workers, batches=(["a"],)))
    results = [trainer._validate(), trainer._validate()]
    lost, healthy = (results[0], results[1]) if schedule == [0, 1] else (results[1], results[0])
    assert lost == {"validation/inflight_dead_worker_failed": 1}
    assert healthy["val-core/test/acc/mean@1"] == pytest.approx(1.0)
    table = trainer.validation_generations_logger.validation_table
    assert len(table.data) == 1 and table.columns == ["step", "input_1", "output_1", "score_1"]


def test_generations_logger_pads_a_partially_lost_validation_to_the_table_width(monkeypatch):
    """Fewer samples than before (one of two groups lost) pad their row with None; more samples widen the
    table and pad the earlier rows."""
    memory = MemoryTQ()
    memory.install(monkeypatch)
    workers = ScheduledWorkers(memory, [2, 1, 2])
    trainer = _with_real_wandb_logger(_make_trainer(workers, batches=(["a", "b"],)), generations_to_log=4)
    trainer._validate()
    partial = trainer._validate()
    trainer._validate()
    assert partial["validation/inflight_dead_worker_failed"] == 1
    table = trainer.validation_generations_logger.validation_table
    assert len(table.columns) == 1 + 3 * 2 and len(table.data) == 3
    assert table.data[1][4:] == [None, None, None] and None not in table.data[0] and None not in table.data[2]


def test_live_worker_failing_every_session_without_rows_reports_counts_not_a_dead_worker(monkeypatch, caplog):
    """A live worker that marks every group ``failure`` without rows reaches the same guard: the warning is
    neutral and ``_validate`` returns the counts instead of the empty dict ``fit()`` rejects."""
    caplog.set_level(logging.WARNING, logger=trainer_base.__name__)
    memory = MemoryTQ()
    memory.install(monkeypatch)
    workers = LiveWorkersFailingEverySession(memory)
    trainer = _make_trainer(workers, batches=(["a", "b"], ["c"]))
    trainer.replay_buffer.liveness_fn = lambda uids: set()  # every worker answers the probe
    metrics = trainer._validate()
    assert metrics == {"validation/prompts_expected": 3, "validation/prompts_without_rows": 3}
    assert memory.batch_get_calls == [] and memory.tags["val"] == {}
    assert not any("lost its agent loop worker" in record.message for record in caplog.records)
    assert sum("has no rows to score" in record.message for record in caplog.records) == 2
