"""Re-verification with real AgentLoopManagerTQ + stub actors (p3b only; skipped on p3a)."""

import sys

import pytest
import ray

from _review2_common import HAS_RELAUNCH

pytestmark = pytest.mark.skipif(not HAS_RELAUNCH, reason="this branch has no dead-worker relaunch (p3a)")

if HAS_RELAUNCH:
    from _review_common import SampleThread, make_batch, make_manager, make_rb, make_trainer, prompt_statuses, submit_val, teardown, wait_until
    from verl.trainer.ppo.v1 import agent_loop_tq


def _kill(manager, index: int):
    ray.kill(manager.agent_loop_workers[index], no_restart=True)


def test_relaunch_streak_cap_raises_and_a_served_dispatch_resets_it(ray_tq, monkeypatch):
    """N3. Three consecutive probe relaunches of worker 1 without a served dispatch exceed MAX=2 and raise; a
    served dispatch in between resets the streak."""
    monkeypatch.setattr(agent_loop_tq, "MAX_WORKER_RELAUNCHES", 2)
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    try:
        for expected in (1, 2):
            _kill(manager, 1)
            assert manager.reap_dead_workers(set()) == set()
            assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == expected
        _kill(manager, 1)
        with pytest.raises(RuntimeError, match=r"worker 1 was re-created 2 times without serving a dispatch"):
            manager.reap_dead_workers(set())
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 2
    finally:
        teardown(manager)

    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    try:
        _kill(manager, 1)
        manager.reap_dead_workers(set())  # streak 1
        batch, uids = make_batch(4)
        trainer._submit_batch_to_rollout(batch)  # served by both workers: streaks reset
        wait_until(lambda: all(prompt_statuses().get(u) == "finished" for u in uids), what="4 finished")
        assert manager._relaunch_streak[1] == 0
        for _ in range(2):
            _kill(manager, 1)
            manager.reap_dead_workers(set())  # streak 1, 2: no error
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 3
    finally:
        teardown(manager)


def test_val_dead_worker_fails_val_groups_in_place_and_train_orphans_are_handed_over_later(ray_tq):
    """N4. Worker 1 holds one train prompt and one val prompt and dies during validation: the val group is
    failed in place (same uid, no regeneration), the worker is relaunched, and the train prompt it owned is
    re-issued under a fresh uid at the next train poll (owner map survives the val probe)."""
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    try:
        ray.get(manager.agent_loop_workers[1].set_mode.remote("stuck"))
        tb, tuids = make_batch(2)
        trainer._submit_batch_to_rollout(tb)
        vb, vuids = make_batch(2, validate=True)
        submit_val(manager, vb)
        wait_until(lambda: sorted(prompt_statuses().get(u) for u in tuids) == ["finished", "running"], what="train 1+1")
        wait_until(lambda: sorted(prompt_statuses("val").get(u) for u in vuids) == ["finished", "running"], what="val 1+1")
        stuck_train = next(u for u in tuids if prompt_statuses()[u] == "running")
        stuck_val = next(u for u in vuids if prompt_statuses("val")[u] == "running")
        assert manager._prompt_owner[stuck_train] == 1 and manager._prompt_owner[stuck_val] == 1
        dead = manager.agent_loop_workers[1]
        _kill(manager, 1)

        rb = make_rb(trainer, ttl=None, liveness=True)
        rb.liveness_period_s = 0.2
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="val", batch_size=2)).result_or_raise(15.0)
        # The val sample consumed both groups under their own uids (prompt keys are cleared on materialisation):
        # nothing was regenerated, and only the finished val row materialised.
        assert {key.split("_")[0] for key in out.keys} == set(vuids) - {stuck_val}, "only the finished val row materializes"
        assert set(prompt_statuses("val")) == set(), "no fresh val uid was registered"
        assert metrics["validation/inflight_dead_worker_failed"] == 1
        assert manager.agent_loop_workers[1] is not dead and manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
        assert stuck_train in manager._orphaned_uids, "the train orphan waits for the train poll"
        assert prompt_statuses()[stuck_train] == "running", "train prompts are untouched by the val poll"

        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2)).result_or_raise(15.0)
        assert len({key.split("_")[0] for key in out.keys}) == 2 and stuck_train not in {k.split("_")[0] for k in out.keys}
        assert metrics["training/inflight_dead_worker_reissued"] == 1 and metrics["training/inflight_reissued"] == 1
        assert manager._orphaned_uids == set()
    finally:
        teardown(manager)


def test_unavailable_retry_at_dispatch_must_not_spawn_the_same_uid_twice(ray_tq, monkeypatch):
    """N5. If the dispatch RPC executed but its reply came back as ``ActorUnavailableError`` (Ray: the task may
    have run), ``_recover_dispatch`` re-sends the same chunk to the same live worker: two attempts of one uid on
    the same keys. Expected: a uid is spawned once (clear + fresh uid, or kill-and-relaunch), never twice."""
    monkeypatch.setattr(agent_loop_tq, "DISPATCH_RETRY_INTERVAL_SECONDS", 0.01)
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    real_get, state = ray.get, {"armed": True}

    def get_then_lose_the_reply(ref, *args, **kwargs):
        result = real_get(ref, *args, **kwargs)  # the dispatch really executed on the worker
        caller = sys._getframe(1)
        from_dispatch = caller.f_code.co_name == "generate_sequences" and caller.f_globals.get("__name__") == agent_loop_tq.__name__
        if state["armed"] and from_dispatch:
            state["armed"] = False
            raise ray.exceptions.ActorUnavailableError("simulated transient unavailability", None)
        return result

    monkeypatch.setattr(agent_loop_tq.ray, "get", get_then_lose_the_reply)
    try:
        batch, uids = make_batch(2)
        trainer._submit_batch_to_rollout(batch)
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        wait_until(lambda: all(prompt_statuses().get(u) == "finished" for u in uids), what="2 finished")
        received = [ray.get(worker.received.remote()) for worker in manager.agent_loop_workers]
        counts = {u: sum(r.count(u) for r in received) for u in uids}
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 0, "nothing died"
        assert all(c == 1 for c in counts.values()), f"a uid was spawned twice on a live worker: {counts}"
    finally:
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        teardown(manager)
