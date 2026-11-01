"""R2-B2..B5: ``_validate`` edge cases around the empty-batch guard (real ``_validate`` / ``_val_metrics_update``,
the author's in-memory TransferQueue with the client's empty-key guard).

B2/B3 (expected PASS): the all-lost batch is the FIRST or the LAST of several; list alignment, the dump, the
colocated sleep/wake bracket and the accuracy denominator must all hold. B4 (expected PASS): a dead worker
that wrote one of two sessions of a group: the missing session scores 0. B5 (LOW, characterization, FAILS):
a LIVE worker whose sessions all fail without rows reaches the same guard; the warning blames a dead worker
and ``_validate`` returns ``{}`` (which ``fit()`` asserts on for ``val_before_train``).
"""

import logging

import pytest

from verl.trainer.ppo.v1 import trainer_base

from _r2_common import H, LiveWorkersWhoseSessionsAllFail, PartialThenDeadWorkers, ScheduledWorkers, make_trainer


@pytest.mark.parametrize("colocated", [False, True])
def test_r2_b2_first_batch_all_lost_then_healthy_last_batch(monkeypatch, colocated):
    memory, workers, trainer = make_trainer(
        monkeypatch, ScheduledWorkers, batches=(["a", "b"], ["c"]), schedule=[0, 1], colocated=colocated
    )
    trainer._compute_reward_colocate = H.Recorder(result="first_arg")
    metrics = trainer._validate()
    row = f"{workers.dispatched[1][0]}_0_0"
    assert metrics["validation/inflight_dead_worker_failed"] == 2
    assert metrics["val-core/test/acc/mean@1"] == pytest.approx(1 / 3), "c = 1.0, a and b = 0.0"
    assert metrics["val-aux/test/acc_success_only/mean@1"] == pytest.approx(1.0)
    assert memory.batch_get_calls == [[row], [row]] and [] not in memory.clear_calls and memory.tags["val"] == {}
    assert trainer._maybe_log_val_generations.calls == [((), {"inputs": ["5 6 7"], "outputs": ["8 9"], "scores": [1.0]})]
    ((), dump) = trainer._dump_generations.calls[0]
    assert dump["reward_extra_infos_dict"]["uid"] == [row] and dump["scores"] == [1.0]
    assert len(trainer.checkpoint_manager.sleep_replicas.calls) == (1 if colocated else 0)
    assert len(trainer.checkpoint_manager.update_weights.calls) == (1 if colocated else 0)


def test_r2_b3_healthy_first_batch_then_all_lost_last_batch(monkeypatch):
    memory, workers, trainer = make_trainer(monkeypatch, ScheduledWorkers, batches=(["a"], ["b", "c"]), schedule=[1, 0])
    metrics = trainer._validate()
    row = f"{workers.dispatched[0][0]}_0_0"
    assert metrics["validation/inflight_dead_worker_failed"] == 2
    assert metrics["val-core/test/acc/mean@1"] == pytest.approx(1 / 3)
    assert memory.batch_get_calls == [[row], [row]] and memory.tags["val"] == {}
    ((), dump) = trainer._dump_generations.calls[0]
    assert dump["reward_extra_infos_dict"]["uid"] == [row]


def test_r2_b4_dead_worker_wrote_one_of_two_sessions_the_missing_session_scores_zero(monkeypatch):
    memory, workers, trainer = make_trainer(monkeypatch, PartialThenDeadWorkers, batches=(["a"],), n_val=2)
    metrics = trainer._validate()
    row = f"{workers.dispatched[0][0]}_0_0"
    assert metrics["validation/inflight_dead_worker_failed"] == 1
    assert metrics["val-core/test/acc/mean@2"] == pytest.approx(0.5), "session 0 = 1.0, session 1 never ran = 0.0"
    assert metrics["val-aux/test/acc_success_only/mean@1"] == pytest.approx(1.0)
    assert memory.batch_get_calls == [[row], [row]] and memory.tags["val"] == {}, "the partial row is scored once"


def test_r2_b5_live_worker_failing_every_session_without_rows_is_not_reported_as_a_dead_worker(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=trainer_base.__name__)
    memory, workers, trainer = make_trainer(monkeypatch, LiveWorkersWhoseSessionsAllFail, batches=(["a", "b"],))
    trainer.replay_buffer.liveness_fn = lambda uids: set()  # every worker answers the probe
    metrics = trainer._validate()
    assert memory.batch_get_calls == [] and memory.tags["val"] == {}, "the guard is hit: no rows, no empty fetch"
    assert metrics, "fit() asserts `val_metrics` is non-empty for val_before_train; {} trips it"
    assert not any("lost its agent loop worker" in r.message for r in caplog.records), "no worker died here"
