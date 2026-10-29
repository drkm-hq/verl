# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full-sampling-loop check of the TTL step budget with healthy-but-slow generations (independent review, round 3).

Every generation finishes two simulated seconds after its dispatch; the TTL is 0.5 s, so every attempt expires
before it finishes. With ``max_inflight_reissues=0`` every first attempt is exhausted on its first expiry and the
refill path (``ReplayBufferAsync``, or sync with ``sync_refill_failed_groups``) dispatches a replacement that
starts at attempt 1 again: without the budget charge on exhaustion failures this cycled forever. The budget bounds
the TTL-induced discards of the step; once spent the loop waits and the pending healthy work completes. A
virtual clock bounds the test without Ray or wall-clock waits.
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
            put(
                uid,
                "train",
                {
                    "is_prompt": True,
                    "status": "running",
                    "global_steps": 1,
                    "attempt": attempt,
                    "dispatched_at": now[0],
                },
            )
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

    for name, fn in {
        "kv_put": put,
        "kv_batch_put": batch_put,
        "kv_clear": clear,
        "kv_list": lambda *args, **kwargs: deepcopy(tags),
    }.items():
        monkeypatch.setattr(module.tq, name, fn)
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], sleep=sleep))
    cls = ReplayBuffer if mode == "sync" else ReplayBufferAsync
    rb = cls(
        trainer_mode=mode,
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.1,
        refill_fn=dispatch,
        train_batch_size=1,
        gen_batch_size=1,
        sync_refill_failed_groups=(mode == "sync"),
        inflight_ttl_s=0.5,
        max_inflight_reissues=cap,
        reissue_fn=reissue,
    )
    dispatch()
    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert len(batch.keys) == 1
