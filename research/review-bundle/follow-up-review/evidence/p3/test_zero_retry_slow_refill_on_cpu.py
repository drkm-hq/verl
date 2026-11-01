"""Independent full-sampling-loop check of the zero-retry budget edge.

All generations are healthy: each finishes two simulated seconds after dispatch.
The TTL is 0.5 seconds. A virtual clock bounds the test without Ray or wall-clock waits.
"""
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from verl.trainer.ppo.v1 import replay_buffer as module
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
@pytest.mark.parametrize("cap", [0, 3])
def test_healthy_slow_replacements_eventually_complete_under_step_budget(monkeypatch, mode, cap):
    now = [100.0]
    tags = defaultdict(dict)
    pending_finishes = []
    counts = {"dispatches": 0, "completions": 0}

    def put(key, partition_id, tag, **kwargs):
        tags[partition_id].setdefault(key, {}).update(tag)

    def batch_put(keys, partition_id, tags, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            put(key, partition_id, tag)

    def clear(keys, partition_id):
        for key in keys:
            tags[partition_id].pop(key, None)

    def dispatch(count=1, attempt=1):
        for _ in range(count):
            uid = uuid4().hex
            put(uid, "train", {"is_prompt": True, "status": "running", "global_steps": 1,
                               "attempt": attempt, "dispatched_at": now[0]})
            pending_finishes.append((now[0] + 2.0, uid))
            counts["dispatches"] += 1
        return count

    def reissue(partition, uids):
        for uid in uids:
            attempt = tags[partition][uid]["attempt"]
            clear([uid], partition)
            dispatch(attempt=attempt + 1)
        return list(uids)

    def sleep(duration):
        now[0] += duration
        while pending_finishes and pending_finishes[0][0] <= now[0]:
            _, uid = pending_finishes.pop(0)
            put(f"{uid}_0_0", "train", {"seq_len": 3, "global_steps": 1, "status": "success"})
            put(uid, "train", {"status": "finished"})
            counts["completions"] += 1
        if now[0] >= 120:
            pytest.fail(f"Sampling livelocked for 20 simulated seconds despite healthy 2-second generations: {counts}")

    for name, fn in {"kv_put": put, "kv_batch_put": batch_put, "kv_clear": clear,
                     "kv_list": lambda *args, **kwargs: deepcopy(tags)}.items():
        monkeypatch.setattr(module.tq, name, fn)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], sleep=sleep))
    cls = ReplayBuffer if mode == "sync" else ReplayBufferAsync
    rb = cls(trainer_mode=mode, trainer_config={}, max_off_policy_threshold=None,
             max_off_policy_strategy="drop", sampler_kwargs={}, poll_interval=0.1,
             refill_fn=dispatch, train_batch_size=1, gen_batch_size=1,
             sync_refill_failed_groups=(mode == "sync"), inflight_ttl_s=0.5,
             max_inflight_reissues=cap, reissue_fn=reissue)
    dispatch()
    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert len(batch.keys) == 1
