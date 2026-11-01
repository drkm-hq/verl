"""Reviewer tests with real AgentLoopManagerTQ + stub worker actors on Ray local (part B)."""

import ray
import torch
import transfer_queue as tq

from _review_common import SampleThread, make_batch, make_manager, make_rb, make_trainer, prompt_statuses, submit_val, teardown, wait_until
from verl.trainer.ppo.v1 import replay_buffer as replay_buffer_module


def _finish_running_by_hand(partition: str) -> None:
    for key, status in prompt_statuses(partition).items():
        if status in ("pending", "running"):
            tq.kv_put(
                key=f"{key}_0_0",
                partition_id=partition,
                fields={"input_ids": torch.tensor([1, 2, 3])},
                tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
            )
            tq.kv_put(key=key, partition_id=partition, tag={"status": "finished"})


def test_two_workers_dying_at_once_are_both_relaunched_and_their_prompts_reissued(ray_tq, monkeypatch):
    """R7 (expected PASS). Workers 1 and 2 of 3 die with 2 stuck prompts each; one probe finds both."""
    monkeypatch.setattr(replay_buffer_module, "WORKER_LIVENESS_PERIOD_SECONDS", 0.2)
    manager = make_manager(num_workers=3, relaunch=True)
    trainer = make_trainer(manager)
    try:
        for index in (1, 2):
            ray.get(manager.agent_loop_workers[index].set_mode.remote("stuck"))
        batch, uids = make_batch(6)
        trainer._submit_batch_to_rollout(batch)
        wait_until(lambda: sorted(prompt_statuses().get(u) for u in uids) == ["finished"] * 2 + ["running"] * 4, what="2 finished + 4 stuck")
        dead = [manager.agent_loop_workers[1], manager.agent_loop_workers[2]]
        for worker in dead:
            ray.kill(worker, no_restart=True)

        rb = make_rb(trainer, ttl=None, liveness=True)
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=6)).result_or_raise(25.0)
        assert len({key.split("_")[0] for key in out.keys}) == 6
        assert metrics["training/inflight_reissued"] == 4
        assert metrics["training/inflight_dead_worker_reissued"] == 4
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 2
        assert manager.recovery_metrics["rollout/prompts_orphaned_by_dead_workers"] == 4
        for index in (1, 2):
            assert manager.agent_loop_workers[index] not in dead
            assert ray.get(manager.agent_loop_workers[index].ping.remote()) is True
    finally:
        teardown(manager)


def test_ttl_shorter_than_a_legitimate_generation_livelocks_on_healthy_workers(ray_tq):
    """R8. Every generation takes 1.2 s on alive workers; ``inflight_ttl_s=0.4``. Each attempt expires before it
    finishes, is retired and re-issued from scratch, forever: the step never completes, the retired set grows,
    and the old attempts keep running (nothing cancels them). On main this configuration simply works."""
    manager = make_manager(num_workers=2)
    trainer = make_trainer(manager)
    for worker in manager.agent_loop_workers:
        ray.get(worker.set_mode.remote("slow", 1.2))
    batch, uids = make_batch(2)
    trainer._submit_batch_to_rollout(batch)
    wait_until(lambda: all(prompt_statuses().get(u) == "running" for u in uids), what="2 running")
    reissues: list[list[str]] = []

    def counting_reissue(partition_id: str, stuck_uids: list[str]) -> int:
        reissues.append(list(stuck_uids))
        return trainer._reissue_inflight_prompts(partition_id, uids=stuck_uids)

    rb = make_rb(trainer, ttl=0.4, reissue_fn=counting_reissue)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2))
    sampler.start()
    sampler.join(5.0)
    try:
        assert len(reissues) >= 3, f"expected repeated re-issues, got {reissues}"
        assert len(rb._retired_uids["train"]) >= 3
        assert not sampler.is_alive(), f"livelock: {sum(map(len, reissues))} prompts re-issued in 5 s; the step never completes"
    finally:
        for worker in manager.agent_loop_workers:
            ray.get(worker.set_mode.remote("finish"))
        sampler.join(5.0)
        _finish_running_by_hand("train")
        sampler.join(3.0)
        teardown(manager)


def test_validation_partition_still_hangs_on_a_dead_worker_with_both_knobs_on(ray_tq, monkeypatch):
    """R9. ``_recover_stuck_inflight`` returns early for ``val`` (replay_buffer.py), so neither the TTL nor the
    probe runs during ``_validate``'s ``sample("val")``: a worker that dies mid-validation blocks validation
    forever and is not even relaunched (the docs say it is relaunched "at its next dispatch", which never comes)."""
    monkeypatch.setattr(replay_buffer_module, "WORKER_LIVENESS_PERIOD_SECONDS", 0.2)
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    ray.get(manager.agent_loop_workers[1].set_mode.remote("stuck"))
    batch, uids = make_batch(2, validate=True)
    submit_val(manager, batch)
    wait_until(lambda: sorted(prompt_statuses("val").get(u) for u in uids) == ["finished", "running"], what="val: 1 finished + 1 stuck")
    ray.kill(manager.agent_loop_workers[1], no_restart=True)

    rb = make_rb(trainer, ttl=0.5, liveness=True)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id="val", batch_size=2))
    sampler.start()
    sampler.join(4.0)
    try:
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 0, "no probe runs for val"
        assert not sampler.is_alive(), "validation sample hangs on a dead worker even with TTL + relaunch on"
    finally:
        _finish_running_by_hand("val")
        sampler.join(3.0)
        teardown(manager)
