"""Round-3 review harness (reviewer-owned): the REAL sampling loops (sync ``sample``, async
``wait_for_sampleable`` + ``sample``) over an in-memory TransferQueue with a virtual clock, after the submitter's
``test_zero_retry_slow_refill_on_cpu.py``. Every generation is healthy and finishes ``gen_time`` simulated
seconds after its dispatch; the TTL is 0.5 s. ``refill_fn`` / ``reissue_fn`` do what the trainer does
(fresh uid, attempt + 1). ``sleep`` advances the clock, completes due generations and bounds the test.
"""

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from verl.trainer.ppo.v1 import replay_buffer as module
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

PARTITION = "train"
PREFIX = ReplayBuffer._metrics_prefix(PARTITION)
TTL = 0.5


class Sim:
    def __init__(self, monkeypatch, gen_time: float = 2.0, deadline: float = 20.0, max_dispatches: int = 400):
        self.max_dispatches = max_dispatches  # a cycle that never sleeps (re-issue -> continue) is bounded here
        self.now = 100.0
        self.start = 100.0
        self.gen_time = gen_time
        self.deadline = deadline
        self.tags: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending: list[tuple[float, str]] = []
        self.hung: set[str] = set()  # prompts whose writer never finishes (dead worker / poison prompt)
        self.poison: set[str] = set()  # subset of hung whose re-issues hang too
        self.counts = {"dispatches": 0, "completions": 0, "reissues": 0}
        self.rb: ReplayBuffer | None = None
        for name, fn in {"kv_put": self.put, "kv_batch_put": self.batch_put, "kv_clear": self.clear,
                         "kv_list": self.kv_list}.items():
            monkeypatch.setattr(module.tq, name, fn)
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: self.now, sleep=self.sleep))

    # --- TransferQueue stand-in ---
    def put(self, key, partition_id, tag=None, fields=None, **kwargs):
        if tag:
            self.tags[partition_id].setdefault(str(key), {}).update(tag)

    def batch_put(self, keys, partition_id, tags, fields=None, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            self.put(key, partition_id, tag)

    def clear(self, keys, partition_id):
        for key in keys:
            self.tags[partition_id].pop(key, None)

    def kv_list(self, partition_id=None):
        return deepcopy(self.tags)

    # --- what the trainer does ---
    def dispatch(self, count: int = 1, attempt: int = 1, hung: bool = False, poison: bool = False) -> int:
        for _ in range(count):
            uid = uuid4().hex
            self.put(uid, PARTITION, {"is_prompt": True, "status": "running", "global_steps": 1,
                                      "attempt": attempt, "dispatched_at": self.now})
            if poison:
                self.poison.add(uid)
            if hung or poison:
                self.hung.add(uid)
            else:
                self.pending.append((self.now + self.gen_time, uid))
            self.counts["dispatches"] += 1
            if self.counts["dispatches"] > self.max_dispatches:
                self.fail(f"more than {self.max_dispatches} dispatches without the clock advancing")
        return count

    def reissue(self, partition_id, uids):
        for uid in uids:
            attempt = self.tags[partition_id][uid]["attempt"]
            self.clear([uid], partition_id)
            # a poison prompt hangs on every attempt; a prompt whose worker died finishes on a live worker
            self.dispatch(attempt=attempt + 1, poison=uid in self.poison)
            self.counts["reissues"] += 1
        return list(uids)

    # --- the clock ---
    def sleep(self, duration: float) -> None:
        self.now += duration
        while self.pending and self.pending[0][0] <= self.now:
            _, uid = self.pending.pop(0)
            self.put(f"{uid}_0_0", PARTITION, {"seq_len": 3, "global_steps": 1, "status": "success"})
            self.put(uid, PARTITION, {"status": "finished"})
            self.counts["completions"] += 1
        if self.now - self.start >= self.deadline:
            self.fail(f"sampling did not return within {self.deadline} simulated seconds")

    def fail(self, why: str) -> None:
        charged = dict(self.rb._ttl_discards_in_step) if self.rb is not None else "n/a"
        pytest.fail(f"{why} despite healthy {self.gen_time}-second generations: {self.counts}, "
                    f"TTL discards charged per step: {charged}")

    def prompt_tags(self) -> dict[str, dict]:
        return {k: t for k, t in self.tags[PARTITION].items() if t.get("is_prompt")}

    def buffer(self, mode: str, cap: int, refill: bool, batch_size: int = 1, budget: int | None = None, ttl: float = TTL, **extra):
        cls = ReplayBuffer if mode == "sync" else ReplayBufferAsync
        self.rb = cls(
            trainer_mode=mode, trainer_config={}, max_off_policy_threshold=None, max_off_policy_strategy="drop",
            sampler_kwargs={}, poll_interval=0.1, refill_fn=self.dispatch, train_batch_size=batch_size,
            gen_batch_size=1, sync_refill_failed_groups=refill, inflight_ttl_s=ttl, max_inflight_reissues=cap,
            max_inflight_reissues_per_step=budget, reissue_fn=self.reissue, **extra,
        )
        return self.rb
