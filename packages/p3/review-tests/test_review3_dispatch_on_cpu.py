"""Re-check of the R-A fix (f6846af0): a chunk is never dispatched twice; actor-error dispatches park prompts."""

import sys

import pytest
import ray

from _review2_common import HAS_RELAUNCH

pytestmark = pytest.mark.skipif(not HAS_RELAUNCH, reason="this branch has no dead-worker relaunch (p3a)")

if HAS_RELAUNCH:
    from _review_common import SampleThread, make_batch, make_manager, make_rb, make_trainer, partition_keys, prompt_statuses, submit_val, teardown, wait_until
    from verl.trainer.ppo.v1 import agent_loop_tq


def _lost_reply_hook(real_get, state, convert_dead=False):
    """First ``ray.get`` issued from ``generate_sequences`` comes back as ``ActorUnavailableError``: either after
    the call really executed (reply lost), or in place of the ``ActorDiedError`` of a dead actor (state unknown)."""

    def hook(ref, *args, **kwargs):
        caller = sys._getframe(1)
        from_dispatch = caller.f_code.co_name == "generate_sequences" and caller.f_globals.get("__name__") == agent_loop_tq.__name__
        try:
            result = real_get(ref, *args, **kwargs)
        except ray.exceptions.ActorDiedError:
            if state["armed"] and from_dispatch and convert_dead:
                state["armed"] = False
                raise ray.exceptions.ActorUnavailableError("simulated: reply lost, actor state unknown", None) from None
            raise
        if state["armed"] and from_dispatch:
            state["armed"] = False
            raise ray.exceptions.ActorUnavailableError("simulated: reply lost after execution", None)
        return result

    return hook


def _counts(manager, uids):
    received = [ray.get(worker.received.remote()) for worker in manager.agent_loop_workers]
    return {u: sum(r.count(u) for r in received) for u in uids}


def _setup(monkeypatch, mode: str, delay: float = 0.0, kill0: bool = False, convert_dead: bool = False):
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    for worker in manager.agent_loop_workers:
        ray.get(worker.set_mode.remote(mode, delay))
    if kill0:
        ray.kill(manager.agent_loop_workers[0], no_restart=True)  # cursor 0: chunk 0 goes to worker 0
    real_get = ray.get
    monkeypatch.setattr(agent_loop_tq.ray, "get", _lost_reply_hook(real_get, {"armed": True}, convert_dead))
    return manager, trainer, real_get


def test_lost_reply_after_execution_slow_attempt_leaves_exactly_one_live_group(ray_tq, monkeypatch):
    """N8. The dispatch executed on the live worker 0 (3 s generation) but its reply was lost: the chunk is parked,
    never re-sent; the probe re-issues it once under a fresh uid; the old attempt's late rows are swept."""
    manager, trainer, real_get = _setup(monkeypatch, "slow", delay=3.0)
    try:
        batch, uids = make_batch(2)
        trainer._submit_batch_to_rollout(batch)
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        assert manager.recovery_metrics["rollout/prompts_orphaned_at_dispatch"] == 1
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 0, "the worker answered the second-opinion ping"
        parked = next(iter(manager._orphaned_uids))
        assert parked in uids
        rb = make_rb(trainer, ttl=None, liveness=True)
        rb.liveness_period_s = 0.2
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2)).result_or_raise(20.0)
        sampled = {key.split("_")[0] for key in out.keys}
        assert parked not in sampled and len(sampled) == 2 and (set(uids) - {parked}) <= sampled
        assert metrics["training/inflight_reissued"] == 1, "parked prompt re-issued exactly once across several probes"
        fresh = next(iter(sampled - set(uids)))
        counts = _counts(manager, [*uids, fresh])
        assert counts[parked] == 1 and counts[fresh] == 1, f"no uid was ever dispatched twice: {counts}"
        rb._sync_metadata_from_transfer_queue()
        rb._sweep_retired_attempts("train")  # the next poll would do this
        assert parked not in prompt_statuses(), "the retired attempt's late status write is swept"
        assert f"{parked}_0_0" not in partition_keys("train"), "the retired attempt's late row is swept"
        assert manager._orphaned_uids == set()
    finally:
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        teardown(manager)


def test_lost_reply_after_execution_fast_attempt_is_not_reissued(ray_tq, monkeypatch):
    """N8b. Same lost reply, but the attempt finishes before the first probe: the parked uid leaves the parked set
    at the reap and is sampled as is (no regeneration, no duplicate)."""
    manager, trainer, real_get = _setup(monkeypatch, "finish")
    try:
        batch, uids = make_batch(2)
        trainer._submit_batch_to_rollout(batch)
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        assert len(manager._orphaned_uids) == 1
        wait_until(lambda: all(prompt_statuses().get(u) == "finished" for u in uids), what="2 finished")
        rb = make_rb(trainer, ttl=None, liveness=True)
        rb.liveness_period_s = 0.2
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2)).result_or_raise(20.0)
        assert {key.split("_")[0] for key in out.keys} == set(uids)
        assert metrics.get("training/inflight_reissued", 0) == 0
        assert manager._orphaned_uids == set() and _counts(manager, uids) == {u: 1 for u in uids}
    finally:
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        teardown(manager)


def test_unavailable_reply_from_a_dead_actor_fails_the_second_opinion_ping_and_relaunches(ray_tq, monkeypatch):
    """N11. Worker 0 is dead; its dispatch reply is reported as unavailable (state unknown). The ping fails with
    ActorDiedError, the worker is re-created, the chunk is parked and re-issued once under a fresh uid."""
    manager, trainer, real_get = _setup(monkeypatch, "finish", kill0=True, convert_dead=True)
    dead = manager.agent_loop_workers[0]
    try:
        batch, uids = make_batch(2)
        trainer._submit_batch_to_rollout(batch)
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1 and manager.agent_loop_workers[0] is not dead
        assert manager.recovery_metrics["rollout/prompts_orphaned_at_dispatch"] == 1
        rb = make_rb(trainer, ttl=None, liveness=True)
        rb.liveness_period_s = 0.2
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2)).result_or_raise(20.0)
        assert len({key.split("_")[0] for key in out.keys}) == 2 and metrics["training/inflight_reissued"] == 1
        parked = next(u for u in uids if u not in {k.split("_")[0] for k in out.keys})
        assert _counts(manager, [parked])[parked] == 0, "the dead worker never executed it; no re-send either"
    finally:
        monkeypatch.setattr(agent_loop_tq.ray, "get", real_get)
        teardown(manager)


def test_dead_worker_at_a_validation_dispatch_fails_that_val_chunk_in_place(ray_tq):
    """N9 (documents the trade-off). A validation chunk dispatched to a dead worker is parked, then failed in
    place by the val probe (validation is never regenerated): that chunk counts as failure for the step."""
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    try:
        ray.kill(manager.agent_loop_workers[0], no_restart=True)
        vb, vuids = make_batch(2, validate=True)
        submit_val(manager, vb)
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
        assert manager.recovery_metrics["rollout/prompts_orphaned_at_dispatch"] == 1
        rb = make_rb(trainer, ttl=None, liveness=True)
        rb.liveness_period_s = 0.2
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="val", batch_size=2)).result_or_raise(20.0)
        assert len({key.split("_")[0] for key in out.keys}) == 1, "only the live worker's val group has rows"
        assert metrics["validation/inflight_dead_worker_failed"] == 1
    finally:
        teardown(manager)
