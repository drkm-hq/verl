"""Independent P3 review reproducer: real trainer/replay methods, in-memory TQ transport.

The empty-key rejection calls the installed TransferQueue client's real input validation.
Run with PYTHONPATH pointing at wt/p3b and the existing .venv-cpu interpreter.
"""
import asyncio
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

from omegaconf import OmegaConf
from transfer_queue.client import AsyncTransferQueueClient

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def test_validation_completes_when_all_dispatched_groups_lose_their_worker(monkeypatch):
    storage = defaultdict(dict)

    def put(key, partition_id, tag, **kwargs):
        storage[partition_id].setdefault(str(key), {}).update(tag)

    def batch_put(keys, partition_id, tags, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            put(key, partition_id, tag)

    def clear(partition_id, keys):
        for key in keys:
            storage[partition_id].pop(key, None)

    def batch_get(keys, partition_id, **kwargs):
        # Exercise the actual installed library's guard, without creating its network transport.
        assert keys == [], "This reproducer expects every validation group to have failed before writing rows"
        return asyncio.run(AsyncTransferQueueClient.async_kv_retrieve_meta.__wrapped__(
            SimpleNamespace(), keys=keys, partition_id=partition_id
        ))

    monkeypatch.setattr(trainer_base.tq, "kv_put", put)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", batch_put)
    monkeypatch.setattr(trainer_base.tq, "kv_list", lambda *args, **kwargs: deepcopy(storage))
    monkeypatch.setattr(trainer_base.tq, "kv_clear", clear)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", batch_get)

    rb = ReplayBuffer(
        trainer_mode="sync", trainer_config={}, max_off_policy_threshold=None,
        max_off_policy_strategy="drop", sampler_kwargs={}, poll_interval=0.001,
        reissue_fn=lambda *args: [], liveness_fn=lambda uids: set(uids), liveness_period_s=0,
    )
    trainer = SimpleNamespace(
        global_steps=1,
        val_dataloader=[{"raw_prompt": ["a"], "index": [0], "data_source": ["test"]}],
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}}, "trainer": {"log_val_generations": 0, "validation_data_dir": None}}),
        agent_loop_manager=SimpleNamespace(generate_sequences=lambda batch: None),
        replay_buffer=rb,
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[object()]),
    )
    # P3 promises that failed-in-place validation groups allow _validate to complete.
    for name in ("_maybe_log_val_generations", "_val_metrics_update", "_consume_recovery_metrics"):
        setattr(trainer, name, getattr(PPOTrainer, name).__get__(trainer))
    assert PPOTrainer._validate(trainer) == {"validation/inflight_dead_worker_failed": 1}
