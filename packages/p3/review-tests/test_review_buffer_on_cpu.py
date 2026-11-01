"""Reviewer tests at the replay-buffer / trainer level (real TransferQueue, no worker actors)."""

import time

import torch
import transfer_queue as tq
from tensordict import TensorDict

from _review_common import SampleThread, make_rb, make_trainer, partition_keys, uid
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.utils import MetricsAggregator


class FakeManager:
    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


def _submit(partition: str, key: str, status: str, dispatched_at: float, global_steps: int = 1) -> None:
    batch = TensorDict({"index": torch.tensor([0])}, batch_size=[1])
    batch["uid"] = [key]
    batch["raw_prompt"] = [f"prompt-{key}"]
    tag = {"is_prompt": True, "status": status, "global_steps": global_steps, "dispatched_at": dispatched_at}
    tq.kv_batch_put(keys=[key], partition_id=partition, tags=[tag], fields=batch)


def _finish_by_hand(partition: str, key: str, global_steps: int = 1) -> None:
    tq.kv_put(
        key=f"{key}_0_0",
        partition_id=partition,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": global_steps},
    )
    tq.kv_put(key=key, partition_id=partition, tag={"status": "finished"})


def _clear(partition: str) -> None:
    keys = list(partition_keys(partition))
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition)


def test_terminal_flip_in_the_reissue_window_loses_the_group(ray_tq):
    """R1. A TTL-expired prompt whose (slow, alive) worker writes ``finished`` between the buffer's snapshot and
    the trainer's own ``kv_list`` is NOT re-issued (correct) but IS retired (replay_buffer.py
    ``_recover_stuck_inflight``: ``self._retired_uids |= stuck`` regardless of what ``reissue_fn`` did), so the
    next poll's sweep deletes the finished group. Expected on a correct branch: the group is sampled."""
    partition = f"review-{uid()}"
    stuck = uid()
    _submit(partition, stuck, "running", dispatched_at=time.time() - 100)
    trainer = make_trainer(FakeManager(), "sync")
    calls: list[list[str]] = []

    def racing_reissue(partition_id: str, uids: list[str]) -> int:
        calls.append(list(uids))
        _finish_by_hand(partition_id, stuck)  # lands in the window; the real method then sees a terminal group
        return trainer._reissue_inflight_prompts(partition_id, uids=uids)

    rb = make_rb(trainer, "sync", ttl=1.0, reissue_fn=racing_reissue)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1))
    sampler.start()
    sampler.join(3.0)
    try:
        assert calls == [[stuck]]
        assert trainer.agent_loop_manager.batches == [], "nothing was re-issued (the group had finished)"
        assert not sampler.is_alive(), (
            f"finished group lost: retired={rb._retired_uids[partition]}, keys left in TQ={partition_keys(partition)}"
        )
        assert [key.split("_")[0] for key in sampler.result[0].keys] == [stuck]
    finally:
        _finish_by_hand(partition, stuck)
        sampler.join(2.0)
        _clear(partition)


def test_inflight_reissued_is_averaged_not_summed_across_iterations():
    """R2. ``MetricsAggregator`` (utils.py:41-56) sums only a fixed list; ``training/inflight_reissued`` and
    ``training/inflight_dead_worker_reissued`` fall through to ``weighted_avg`` (utils.py:103), so with
    ``parameter_sync_step > 1`` the logged value is the sample-weighted mean over iterations, not the count."""
    agg = MetricsAggregator()
    agg.add_step_metrics(
        {"training/inflight_reissued": 4, "training/inflight_dead_worker_reissued": 2, "training/rollout_failure/evicted_samples": 3},
        sample_count=16,
    )
    agg.add_step_metrics(
        {"training/inflight_reissued": 0, "training/inflight_dead_worker_reissued": 0, "training/rollout_failure/evicted_samples": 0},
        sample_count=16,
    )
    out = agg.get_aggregated_metrics()
    assert out["training/rollout_failure/evicted_samples"] == 3, "control: an existing counter is summed"
    assert out["training/inflight_reissued"] == 4, f"counter was averaged: {out['training/inflight_reissued']}"
    assert out["training/inflight_dead_worker_reissued"] == 2


def test_reissued_prompt_survives_checkpoint_and_retired_late_write_persists_as_garbage(ray_tq, tmp_path):
    """R3. Async mode: a mid-run re-issue (fresh uid, rows persisted) survives save/load and is re-issued once under
    its own uid on restart (dataloader untouched, nothing double-trained). Also documents that a retired
    attempt's late tag-only write made before any sweep is checkpointed and never swept after a restart
    (``_retired_uids`` is process memory only)."""
    partition = f"review-{uid()}"
    stuck = uid()
    _submit(partition, stuck, "running", dispatched_at=time.time() - 100, global_steps=3)
    trainer = make_trainer(FakeManager(), "separate_async", global_steps=7)
    rb = make_rb(trainer, "separate_async", ttl=1.0)
    try:
        rb._sync_metadata_from_transfer_queue()
        assert rb._recover_stuck_inflight(partition, {}) is True
        (dispatched,) = trainer.agent_loop_manager.batches
        (new_uid,) = list(dispatched["uid"])
        assert new_uid != stuck and stuck not in partition_keys(partition)

        # The retired attempt wakes up while the trainer is in its update phase (no poll, no sweep yet).
        _finish_by_hand(partition, stuck)
        ckpt = str(tmp_path / "transfer_queue")
        tq.save_checkpoint(ckpt, metadata={"global_steps": 7})
        _clear(partition)
        tq.load_checkpoint(ckpt)

        # Restart: a fresh trainer re-issues exactly one group, under the fresh uid, from the persisted row.
        trainer2 = make_trainer(FakeManager(), "separate_async", global_steps=8)
        assert trainer_base._count_tq_prompt_groups(partition) == 1
        assert trainer2._reissue_inflight_prompts(partition) == 1
        (restart_batch,) = trainer2.agent_loop_manager.batches
        assert list(restart_batch["uid"]) == [new_uid]
        assert list(restart_batch["raw_prompt"]) == [f"prompt-{stuck}"]

        # The garbage keys of the retired attempt came back with the checkpoint and nobody sweeps them now.
        rb2 = make_rb(trainer2, "separate_async", ttl=1.0)
        rb2._sync_metadata_from_transfer_queue()
        assert rb2._recover_stuck_inflight(partition, {}) is False
        assert {stuck, f"{stuck}_0_0"} <= partition_keys(partition), "documented low-severity leak"
        assert stuck in rb2.partitions[partition], "a non-prompt key named like a prompt uid"
    finally:
        _clear(partition)
