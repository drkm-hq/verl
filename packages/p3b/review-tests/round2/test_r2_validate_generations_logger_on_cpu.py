"""R2-B1 (HIGH): an all-lost validation batch now survives ``_validate``, but the real generations logger
does not survive it.

The author's tests stub ``_maybe_log_val_generations``. The real one hands ``samples`` (empty after an
all-lost validation) to ``ValidationGenerationsLogger._log_generations_to_wandb``, which derives the table
columns from ``len(samples)``: an all-lost validation makes a ``["step"]`` table, and the next healthy
validation (or a healthy one before it) rebuilds the table with a different column count, which wandb's
``Table`` rejects with ``ValueError``. Reachable with the stock ``trainer.logger: ["console", "wandb"]``
plus ``log_val_generations > 0`` (set by 16 example scripts) and ``relaunch_dead_workers=True``.

Expected (the guarantee under test): both validations complete. These two tests FAIL on 0d28f34f.
"""

import pytest
import wandb

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.tracking import ValidationGenerationsLogger

from _r2_common import ScheduledWorkers, make_trainer


def _with_real_wandb_table_logger(trainer):
    """Bind the real ``_maybe_log_val_generations`` and a real logger; wandb.run is None so nothing is sent."""
    assert wandb.run is None
    trainer.config.trainer.logger = ["wandb"]
    trainer.config.trainer.log_val_generations = 1
    trainer.validation_generations_logger = ValidationGenerationsLogger()
    trainer._maybe_log_val_generations = PPOTrainer._maybe_log_val_generations.__get__(trainer)
    return trainer


def test_r2_b1_all_lost_validation_then_healthy_validation_keeps_logging(monkeypatch):
    memory, workers, trainer = make_trainer(monkeypatch, ScheduledWorkers, batches=(["a"],), schedule=[0, 1])
    _with_real_wandb_table_logger(trainer)

    first = trainer._validate()  # every group lost its worker: zero samples -> a ["step"] table
    assert first == {"validation/inflight_dead_worker_failed": 1}

    second = trainer._validate()  # the worker is back: one healthy sample -> 4 columns against 1-column rows
    assert second["val-core/test/acc/mean@1"] == pytest.approx(1.0)


def test_r2_b1_healthy_validation_then_all_lost_validation_keeps_logging(monkeypatch):
    memory, workers, trainer = make_trainer(monkeypatch, ScheduledWorkers, batches=(["a"],), schedule=[1, 0])
    _with_real_wandb_table_logger(trainer)

    first = trainer._validate()
    assert first["val-core/test/acc/mean@1"] == pytest.approx(1.0)

    second = trainer._validate()  # all lost: a 1-element row against the 4-column table
    assert second == {"validation/inflight_dead_worker_failed": 1}
