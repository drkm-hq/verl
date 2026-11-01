"""Round-2 review helpers for P3b (reviewer-owned; never part of the branch).

Loads the author's in-memory ``_validate`` harness from the worktree on PYTHONPATH
(``tests/trainer/ppo/v1/test_validate_recovery_on_cpu.py``; that directory has no ``__init__.py``)
and adds worker stand-ins the author's tests do not have: a per-dispatch health schedule, a worker
that writes some session rows and then dies, and a live worker whose sessions all fail.
"""

import importlib.util
import sys
from pathlib import Path

import torch

import verl

HARNESS = Path(verl.__file__).resolve().parents[1] / "tests/trainer/ppo/v1/test_validate_recovery_on_cpu.py"


def load_builder_harness():
    name = "builder_validate_harness"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = load_builder_harness()


class ScheduledWorkers(H.Workers):
    """``healthy_per_batch`` taken per dispatch from ``schedule`` (0 = the whole chunk's worker is dead)."""

    def __init__(self, memory, schedule, score: float = 1.0):
        super().__init__(memory, healthy_per_batch=0, score=score)
        self.schedule = list(schedule)

    def generate_sequences(self, batch) -> None:
        self.healthy_per_batch = self.schedule.pop(0) if self.schedule else 0
        super().generate_sequences(batch)


class PartialThenDeadWorkers(H.Workers):
    """Writes session 0's row for every prompt, then dies before the other sessions and the status."""

    def generate_sequences(self, batch) -> None:
        uids = H._chunk_uids(batch)
        self.dispatched.append(uids)
        for uid in uids:
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


class LiveWorkersWhoseSessionsAllFail(H.Workers):
    """Upstream failure shape: the worker is alive and marks every group ``failure`` without a row."""

    def generate_sequences(self, batch) -> None:
        uids = H._chunk_uids(batch)
        self.dispatched.append(uids)
        for uid in uids:
            self.memory.kv_put(uid, "val", tag={"status": "failure"})


def make_trainer(monkeypatch, workers_cls, batches, n_val: int = 1, colocated: bool = False, **worker_kwargs):
    memory = H.MemoryTQ()
    memory.install(monkeypatch)
    workers = workers_cls(memory, **worker_kwargs)
    trainer = H._make_trainer(workers, batches, n_val=n_val, colocated=colocated)
    return memory, workers, trainer
