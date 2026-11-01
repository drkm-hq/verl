"""Reviewer tests with real AgentLoopManagerTQ + stub worker actors on Ray local (part A)."""

import ray
import torch
import transfer_queue as tq

from _review_common import SampleThread, make_batch, make_manager, make_rb, make_trainer, prompt_statuses, teardown, wait_until
from verl.trainer.ppo.v1 import replay_buffer as replay_buffer_module


def _finish_running_by_hand(partition: str = "train") -> None:
    for key, status in prompt_statuses(partition).items():
        if status in ("pending", "running"):
            tq.kv_put(
                key=f"{key}_0_0",
                partition_id=partition,
                fields={"input_ids": torch.tensor([1, 2, 3])},
                tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
            )
            tq.kv_put(key=key, partition_id=partition, tag={"status": "finished"})


def test_reissue_of_a_small_batch_goes_back_to_worker_0_so_a_wedged_worker_0_livelocks(ray_tq):
    """R4. ``generate_sequences`` chunks the re-issue batch from index 0 (agent_loop_tq.py:277-280); a batch
    smaller than ``num_workers`` always lands on the lowest indices. If worker 0 is the wedged one (alive, never
    writes a terminal status), its prompt is re-issued to it again every TTL, forever: the step never completes
    and nothing cancels the retired attempts. Expected on a correct branch: the re-issue reaches another worker."""
    manager = make_manager(num_workers=2)
    trainer = make_trainer(manager)
    ray.get(manager.agent_loop_workers[0].set_mode.remote("stuck"))
    batch, uids = make_batch(2)
    trainer._submit_batch_to_rollout(batch)
    wait_until(lambda: sorted(prompt_statuses().get(u) for u in uids) == ["finished", "running"], what="1 finished + 1 stuck")
    reissues: list[list[str]] = []

    def counting_reissue(partition_id: str, stuck_uids: list[str]) -> int:
        reissues.append(list(stuck_uids))
        return trainer._reissue_inflight_prompts(partition_id, uids=stuck_uids)

    rb = make_rb(trainer, ttl=0.5, reissue_fn=counting_reissue)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=2))
    sampler.start()
    sampler.join(6.0)
    try:
        w0 = ray.get(manager.agent_loop_workers[0].received.remote())
        w1 = ray.get(manager.agent_loop_workers[1].received.remote())
        assert len(reissues) >= 3, f"expected repeated TTL re-issues, got {reissues}"
        assert len(w1) == 1 and len(w0) == 1 + len(reissues), f"every re-issue went to worker 0: w0={len(w0)} w1={len(w1)}"
        assert not sampler.is_alive(), f"livelock: {len(reissues)} re-issues in 6 s, retired={len(rb._retired_uids['train'])}, step never completes"
    finally:
        _finish_running_by_hand()
        sampler.join(3.0)
        teardown(manager)


def test_prompt_owner_map_grows_without_bound_on_the_default_path(ray_tq):
    """R5. ``_prompt_owner[uid] = index`` is recorded for every dispatched prompt (agent_loop_tq.py:285-286) and
    only pruned inside ``reap_dead_workers``, which is wired only when ``relaunch_dead_workers`` is on. With the
    knob off (the default) the map grows by one entry per prompt for the whole run."""
    manager = make_manager(num_workers=2, relaunch=False)
    trainer = make_trainer(manager)
    try:
        all_uids = []
        for _ in range(3):
            batch, uids = make_batch(4)
            trainer._submit_batch_to_rollout(batch)
            all_uids += uids
        wait_until(lambda: all(prompt_statuses().get(u) == "finished" for u in all_uids), what="12 finished")
        # Every group is terminal; the trainer would even have sampled and cleared them by now.
        tq.kv_clear(keys=list(tq.kv_list(partition_id="train").get("train", {}).keys()), partition_id="train")
        assert len(manager._prompt_owner) == 0, f"default path keeps {len(manager._prompt_owner)} owner entries forever"
    finally:
        teardown(manager)


def test_dead_worker_half_terminated_chunk_reissues_only_the_inflight_prompt(ray_tq, monkeypatch):
    """R6 (expected PASS). Worker 1 finished one of its two prompts and then died: the probe re-issues only the
    in-flight one; the finished group keeps its original rows and is not regenerated."""
    monkeypatch.setattr(replay_buffer_module, "WORKER_LIVENESS_PERIOD_SECONDS", 0.2)
    manager = make_manager(num_workers=2, relaunch=True)
    trainer = make_trainer(manager)
    try:
        ray.get(manager.agent_loop_workers[1].set_mode.remote("finish_first_only"))
        batch, uids = make_batch(4)
        trainer._submit_batch_to_rollout(batch)
        expected = {uids[0]: "finished", uids[1]: "finished", uids[2]: "finished", uids[3]: "running"}
        wait_until(lambda: {u: prompt_statuses().get(u) for u in uids} == expected, what="3 finished + 1 running")
        dead = manager.agent_loop_workers[1]
        ray.kill(dead, no_restart=True)

        rb = make_rb(trainer, ttl=None, liveness=True)
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id="train", batch_size=4)).result_or_raise(20.0)
        sampled = {key.split("_")[0] for key in out.keys}
        assert len(sampled) == 4 and uids[2] in sampled and uids[3] not in sampled
        assert f"{uids[2]}_0_0" in out.keys, "the dead worker's finished group was sampled as is"
        assert metrics["training/inflight_reissued"] == 1
        assert metrics["training/inflight_dead_worker_reissued"] == 1
        assert manager.recovery_metrics["rollout/prompts_orphaned_by_dead_workers"] == 1
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
        assert manager.agent_loop_workers[1] is not dead and ray.get(manager.agent_loop_workers[1].ping.remote())
    finally:
        teardown(manager)
