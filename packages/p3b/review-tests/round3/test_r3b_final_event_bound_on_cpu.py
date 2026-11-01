"""R3b-final E1..E4 for P3b (1b397dfd): the per-poll hold-back meets the liveness probe, and the death-event bound.

E1 (PASS): a held-back group whose worker is then reported dead is failed at once (dead-owned groups are outside
the hold-back) and the step completes padded from the healthy group. E2 (PASS, by design): the same when the held
group was the step's only group: no live producer, the all-failed RuntimeError. E3 (PASS): a probe that keeps
reporting a retired uid (stale) adds no event and no discard (TTL above the generation time, so only the death acts). E4 (PASS):
a train death reported while the val partition polls counts nothing under val and one event under train.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "p3a" / "review-tests" / "round3"))
from _r3_loop_common import PARTITION, PREFIX, Sim  # noqa: E402


def _dead_after(sim, when: float):
    """A probe that reports the hung prompts dead once the simulated clock passes ``when``."""
    return lambda uids: (set(uids) & sim.hung) if sim.now >= when else set()


def test_r3b_e1_held_back_group_reported_dead_is_failed_and_the_step_pads(monkeypatch):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=0, refill=False, batch_size=2, liveness_fn=_dead_after(sim, 101.0), liveness_period_s=0)
    sim.dispatch(1, hung=True)  # held back at t=100.6 with the healthy group, reported dead from t=101.0
    sim.dispatch(1)
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 1 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1
    assert metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1, "the healthy slow group was held back meanwhile"
    assert rb._dead_worker_events_in_step[PARTITION] == 1 and rb._ttl_discards_in_step[PARTITION] == 0


def test_r3b_e2_held_back_only_group_reported_dead_ends_in_the_all_failed_runtime_error(monkeypatch):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=0, refill=False, batch_size=1, liveness_fn=_dead_after(sim, 101.0), liveness_period_s=0)
    sim.dispatch(1, hung=True)
    with pytest.raises(RuntimeError, match="no materializable trajectories"):
        rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert rb._dead_worker_events_in_step[PARTITION] == 1 and sim.now - sim.start < 2.0


def test_r3b_e3_stale_probe_repeating_a_retired_uid_adds_no_event(monkeypatch):
    sim = Sim(monkeypatch)
    sim.dispatch(1)
    original = next(iter(sim.prompt_tags()))
    # the original uid is reported dead on every probe, even after it was retired by its re-issue
    rb = sim.buffer("sync", cap=3, refill=True, batch_size=1, ttl=10.0, liveness_fn=lambda uids: {original}, liveness_period_s=0)
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert len(batch.keys) == 1 and sim.counts["reissues"] == 1 and sim.counts["dispatches"] == 2
    assert rb._dead_worker_events_in_step[PARTITION] == 1 and rb._dead_worker_discards_in_step[PARTITION] == 1
    assert metrics[f"{PREFIX}/inflight_dead_worker_reissued"] == 1


def test_r3b_e4_a_train_death_seen_by_the_val_poll_counts_only_under_train(monkeypatch):
    sim = Sim(monkeypatch)
    sim.dispatch(1, hung=True)  # the train prompt's worker is dead; its re-issue lands on a live worker
    dead = next(iter(sim.prompt_tags()))
    rb = sim.buffer("sync", cap=3, refill=True, batch_size=1, liveness_fn=lambda uids: {dead} & set(uids), liveness_period_s=0)
    sim.put("v1", "val", {"is_prompt": True, "status": "running", "global_steps": 1, "dispatched_at": sim.now, "attempt": 1})
    rb._begin_ttl_budget("val", 1)
    rb._sync_metadata_from_transfer_queue()
    assert rb._recover_stuck_inflight("val", {}, reissue_budget=rb._reissue_budget(1)) is False
    assert rb._dead_worker_events_in_step.get("val", 0) == 0 and "v1" in rb._inflight_uids("val")
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert len(batch.keys) == 1 and rb._dead_worker_events_in_step[PARTITION] == 1
    assert rb._dead_worker_events_in_step.get("val", 0) == 0
