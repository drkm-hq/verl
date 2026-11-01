"""Round-2 review helpers for the P3a per-step TTL budget (reviewer-owned; never part of the branch).

An in-memory TransferQueue patched into ``verl.trainer.ppo.v1.replay_buffer.tq`` (the ``transfer_queue``
module object, so the patch covers every importer), a ``reissue_fn`` that does what the trainer's
``_reissue_inflight_prompts(uids=...)`` does (old uid cleared, fresh uid pending with ``attempt + 1``), and
one-poll drivers around the real ``ReplayBuffer._recover_stuck_inflight``. Runs on p3a and on p3b.
"""

import time
from collections import defaultdict
from copy import deepcopy
from uuid import uuid4

from verl.trainer.ppo.v1 import replay_buffer as rb_module
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

PARTITION = "train"
PREFIX = ReplayBuffer._metrics_prefix(PARTITION)


class MemoryTQ:
    def __init__(self):
        self.tags: dict[str, dict[str, dict]] = defaultdict(dict)
        self.cleared: list[list[str]] = []

    def kv_put(self, key, partition_id, tag=None, fields=None, **kwargs):
        if tag:
            self.tags[partition_id].setdefault(str(key), {}).update(tag)

    def kv_batch_put(self, keys, partition_id, tags, fields=None, **kwargs):
        for key, tag in zip(keys, tags, strict=True):
            self.kv_put(key, partition_id, tag)

    def kv_list(self, partition_id=None):
        data = deepcopy(self.tags)
        return data if partition_id is None else {partition_id: data.get(partition_id, {})}

    def kv_clear(self, keys, partition_id):
        self.cleared.append(list(keys))
        for key in keys:
            self.tags[partition_id].pop(key, None)

    def install(self, monkeypatch) -> None:
        for name in ("kv_put", "kv_batch_put", "kv_list", "kv_clear"):
            monkeypatch.setattr(rb_module.tq, name, getattr(self, name))

    # --- helpers for the tests ---
    def submit(self, age_s: float, attempt: int = 1, status: str = "running") -> str:
        uid = uuid4().hex
        tag = {"is_prompt": True, "status": status, "global_steps": 1, "dispatched_at": time.time() - age_s, "attempt": attempt}
        self.kv_batch_put(keys=[uid], partition_id=PARTITION, tags=[tag])
        return uid

    def prompt_tags(self) -> dict[str, dict]:
        return {k: t for k, t in self.tags[PARTITION].items() if t.get("is_prompt")}

    def failures(self) -> list[str]:
        return [k for k, t in self.prompt_tags().items() if t.get("status") == "failure"]

    def inflight(self) -> set[str]:
        return {k for k, t in self.prompt_tags().items() if t.get("status") in ("pending", "running")}


def make_reissuer(memory: MemoryTQ, log: list[str]):
    """What the trainer does for a mid-run re-issue: clear the old uid, register a fresh uid at attempt + 1."""

    def reissue(partition_id, uids):
        retired = []
        for uid in uids:
            tag = memory.tags[partition_id].get(uid)
            if tag is None or tag.get("status") not in ("pending", "running"):
                continue
            memory.kv_clear(keys=[uid], partition_id=partition_id)
            memory.kv_batch_put(
                keys=[uuid4().hex],
                partition_id=partition_id,
                tags=[{"is_prompt": True, "status": "pending", "global_steps": 1, "dispatched_at": time.time(), "attempt": tag.get("attempt", 1) + 1}],
            )
            retired.append(uid)
            log.append(uid)
        return retired

    return reissue


def make_buffer(memory: MemoryTQ, log: list[str], **overrides) -> ReplayBuffer:
    kwargs = dict(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.001,
        inflight_ttl_s=1.0,
        reissue_fn=make_reissuer(memory, log),
    )
    kwargs.update(overrides)
    return ReplayBuffer(**kwargs)


def poll(rb: ReplayBuffer, metrics: dict, batch_size: int, draining: bool = False, global_steps: int | None = None) -> bool:
    """One iteration of what sync ``sample`` / async ``wait_for_sampleable`` do before selecting.

    ``global_steps`` opens the step's TTL budget the way the loops do (round 3: the budget is per step, so two
    polls with the same ``global_steps`` share it and a new one reopens it); ignored on trees without it.
    """
    if global_steps is not None and hasattr(rb, "_begin_ttl_budget"):
        rb._begin_ttl_budget(PARTITION, global_steps)
    rb._sync_metadata_from_transfer_queue()
    return rb._recover_stuck_inflight(
        PARTITION, metrics, draining=draining, reissue_budget=rb._reissue_budget(batch_size)
    )
