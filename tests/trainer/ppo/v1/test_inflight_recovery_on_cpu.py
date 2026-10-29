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
"""In-flight prompt recovery in the V1 replay buffer (TTL re-issue, re-issue cap, retired-attempt sweep).

These run against a real (CPU-only) TransferQueue, mirroring ``test_replay_buffer_on_cpu.py`` and
``test_reissue_inflight_on_cpu.py``. Only the agent loop worker that runs a prompt writes its terminal
status; if that worker dies or hangs, the prompt stays ``running`` forever and, on ``main``, ``sample``
never returns. With ``inflight_ttl_s`` the buffer re-issues such prompts mid-run
through ``PPOTrainer._reissue_inflight_prompts(partition_id, uids)`` and keeps polling.
"""

import logging
import threading
import time
import uuid

import pytest
import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

POLL_INTERVAL = 0.05
# Metric keys are prefixed per ReplayBuffer._metrics_prefix; test partitions are not the literal "train".
PREFIX = ReplayBuffer._metrics_prefix("test")


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()
    ray.shutdown()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    # uid must not contain "_" because trajectory keys are "{uid}_{session}_{index}".
    return uuid.uuid4().hex


def _submit_prompt(
    partition_id: str, uid: str, status: str, global_steps: int, dispatched_at: float | None, attempt: int = 1
) -> None:
    """Mirror ``_submit_batch_to_rollout`` with recovery enabled: prompt rows as fields plus the status tag."""
    batch = tu.get_tensordict({"uid": [uid], "raw_prompt": [f"prompt-for-{uid}"], "index": torch.tensor([0])})
    tag = {"is_prompt": True, "status": status, "global_steps": global_steps, "attempt": attempt}
    if dispatched_at is not None:
        tag["dispatched_at"] = dispatched_at
    tq.kv_batch_put(keys=[uid], partition_id=partition_id, tags=[tag], fields=batch)


def _add_trajectory(partition_id: str, uid: str, session_id: int = 0, global_steps: int = 1) -> str:
    key = f"{uid}_{session_id}_0"
    tq.kv_put(
        key=key,
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": global_steps},
    )
    return key


def _finish(partition_id: str, uid: str, global_steps: int = 1) -> None:
    """What a live worker does: write the group's trajectories, then its terminal status."""
    _add_trajectory(partition_id, uid, global_steps=global_steps)
    tq.kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})


def _partition_items(partition_id: str) -> dict:
    return tq.kv_list(partition_id=partition_id).get(partition_id, {})


def _clear_partition(partition_id: str) -> None:
    keys = list(_partition_items(partition_id).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _make_rb(trainer_mode: str = "sync", **kwargs) -> ReplayBuffer:
    replay_buffer_cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync
    defaults = dict(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=None if trainer_mode == "sync" else 8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=POLL_INTERVAL,
    )
    return replay_buffer_cls(**{**defaults, **kwargs})


class FakeAgentLoopManager:
    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


def _make_trainer_stub(trainer_mode: str = "sync", global_steps: int = 3, recovery_enabled: bool = True):
    """Bind the real submit/re-issue methods to a stub exposing only what they use."""
    stub = type("Stub", (), {})()
    stub.trainer_mode = trainer_mode
    stub.global_steps = global_steps
    stub.inflight_recovery_enabled = recovery_enabled
    stub.agent_loop_manager = FakeAgentLoopManager()
    stub._reissue_inflight_prompts = PPOTrainer._reissue_inflight_prompts.__get__(stub)
    stub._submit_batch_to_rollout = PPOTrainer._submit_batch_to_rollout.__get__(stub)
    return stub


class SampleThread(threading.Thread):
    """Run the blocking ``sample`` (or ``wait_for_sampleable``) so the test can bound the wait."""

    def __init__(self, fn):
        super().__init__(daemon=True)
        self.fn = fn
        self.result = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.fn()
        except Exception as e:  # surfaced through result_or_raise
            self.error = e

    def result_or_raise(self, timeout: float = 15.0):
        if self.ident is None:
            self.start()
        self.join(timeout)
        assert not self.is_alive(), "sampling did not return in time"
        if self.error is not None:
            raise self.error
        return self.result


# ----------------------------------------------------------------------------------------------------------
# _expired_inflight_keys and the dispatched_at tag
# ----------------------------------------------------------------------------------------------------------


def test_expired_inflight_keys_uses_dispatched_at_and_ttl(tq_init, partition_id):
    now = 1_000.0
    old_running, old_pending, fresh_running, untagged, old_finished = (_uid() for _ in range(5))
    _submit_prompt(partition_id, old_running, "running", 1, dispatched_at=now - 30)
    _submit_prompt(partition_id, old_pending, "pending", 1, dispatched_at=now - 30)
    _submit_prompt(partition_id, fresh_running, "running", 1, dispatched_at=now - 1)
    _submit_prompt(partition_id, untagged, "running", 1, dispatched_at=None)
    _submit_prompt(partition_id, old_finished, "finished", 1, dispatched_at=now - 30)
    try:
        rb = _make_rb(inflight_ttl_s=10, reissue_fn=lambda partition, uids: 0)
        rb._sync_metadata_from_transfer_queue()
        # Only non-terminal prompts with a dispatched_at tag older than the TTL expire.
        assert rb._expired_inflight_keys(partition_id, now=now) == {old_running, old_pending}
        assert rb._expired_inflight_keys(partition_id, now=now - 25) == set()
        # Validation prompts are never re-issued, and a null TTL disables expiry.
        assert rb._expired_inflight_keys("val", now=now) == set()
        rb_off = _make_rb()
        rb_off._sync_metadata_from_transfer_queue()
        assert rb_off._expired_inflight_keys(partition_id, now=now) == set()
    finally:
        _clear_partition(partition_id)


def test_dispatched_at_survives_the_running_flip(tq_init, partition_id):
    """The worker flips the status with a tag-only put; TransferQueue merges tags, so the stamp survives."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "pending", 1, dispatched_at=123.5)
    try:
        tq.kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        rb = _make_rb()
        rb._sync_metadata_from_transfer_queue()
        assert uid in rb.running_keys[partition_id]
        assert rb.prompt_dispatched_at[partition_id][uid] == 123.5
    finally:
        _clear_partition(partition_id)


def test_sync_submit_persists_rows_and_stamps_dispatched_at_only_when_recovery_is_enabled(monkeypatch):
    batch = tu.get_tensordict({"uid": [_uid(), _uid()], "raw_prompt": ["a", "b"], "index": torch.tensor([0, 1])})
    tu.assign_non_tensor_data(batch, "global_steps", 3)
    puts = []
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))

    before = time.time()
    stub = _make_trainer_stub(trainer_mode="sync", recovery_enabled=True)
    assert stub._submit_batch_to_rollout(batch) == 2
    assert set(puts[-1]["fields"].keys()) == {"uid", "raw_prompt", "index"}
    assert all(before <= tag["dispatched_at"] <= time.time() for tag in puts[-1]["tags"])
    assert all(tag["status"] == "pending" and tag["global_steps"] == 3 for tag in puts[-1]["tags"])

    # Default sync path: tag-only registration, exactly as before.
    stub = _make_trainer_stub(trainer_mode="sync", recovery_enabled=False)
    assert stub._submit_batch_to_rollout(batch) == 2
    assert "fields" not in puts[-1]
    assert "dispatched_at" in puts[-1]["tags"][0]


# ----------------------------------------------------------------------------------------------------------
# _reissue_inflight_prompts(partition_id, uids): the mid-run path
# ----------------------------------------------------------------------------------------------------------


def test_reissue_uids_restarts_under_fresh_uids_and_clears_old_rows(tq_init, partition_id):
    stuck, sibling = _uid(), _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    _submit_prompt(partition_id, sibling, "running", 1, dispatched_at=time.time())
    partial_key = _add_trajectory(partition_id, stuck, session_id=0)
    try:
        stub = _make_trainer_stub(trainer_mode="sync", global_steps=5)
        before = time.time()
        assert stub._reissue_inflight_prompts(partition_id, uids=[stuck]) == [stuck]

        items = _partition_items(partition_id)
        # The retired attempt's prompt key and partial trajectory are gone; the sibling is untouched.
        assert stuck not in items and partial_key not in items
        assert items[sibling]["status"] == "running"
        new_uids = [key for key, tag in items.items() if tag.get("is_prompt") and key not in (stuck, sibling)]
        assert len(new_uids) == 1
        new_tag = items[new_uids[0]]
        assert new_tag["status"] == "pending" and new_tag["global_steps"] == 5
        assert new_tag["attempt"] == 2, "a re-issue counts as the next attempt"
        assert before <= new_tag["dispatched_at"] <= time.time()
        # The new attempt was dispatched with the original prompt row under its fresh uid, and the row is
        # persisted so a later re-issue can find it again. The dataloader was not consulted.
        (dispatched,) = stub.agent_loop_manager.batches
        assert list(dispatched["uid"]) == new_uids
        assert list(dispatched["raw_prompt"]) == [f"prompt-for-{stuck}"]
        assert dispatched["global_steps"] == 5
        stored = tq.kv_batch_get(keys=new_uids, partition_id=partition_id)
        assert list(stored["raw_prompt"]) == [f"prompt-for-{stuck}"]
    finally:
        _clear_partition(partition_id)


def test_reissue_uids_skips_prompts_that_already_turned_terminal(tq_init, partition_id):
    """A terminal flip that races the buffer's snapshot must not restart a finished group."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "finished", 1, dispatched_at=time.time() - 100)
    try:
        stub = _make_trainer_stub(trainer_mode="sync")
        assert stub._reissue_inflight_prompts(partition_id, uids=[uid]) == []
        assert stub.agent_loop_manager.batches == []
        assert _partition_items(partition_id)[uid]["status"] == "finished"
    finally:
        _clear_partition(partition_id)


def test_reissue_without_uids_keeps_the_sync_early_return(tq_init, partition_id):
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", 1, dispatched_at=time.time() - 100)
    try:
        assert _make_trainer_stub(trainer_mode="sync")._reissue_inflight_prompts(partition_id) == 0
    finally:
        _clear_partition(partition_id)


# ----------------------------------------------------------------------------------------------------------
# The sampling loops
# ----------------------------------------------------------------------------------------------------------


class RecordingReissuer:
    """Stand in for the trainer: retire the uids and produce finished replacements, as a live worker would."""

    def __init__(self, partition_id: str, global_steps: int = 1):
        self.partition_id = partition_id
        self.global_steps = global_steps
        self.calls: list[list[str]] = []
        self.replacements: list[str] = []

    def __call__(self, partition_id: str, uids: list[str]) -> int:
        assert partition_id == self.partition_id
        self.calls.append(list(uids))
        tq.kv_clear(keys=list(uids), partition_id=partition_id)
        for _ in uids:
            new_uid = _uid()
            _submit_prompt(partition_id, new_uid, "running", self.global_steps, dispatched_at=time.time())
            _finish(partition_id, new_uid, self.global_steps)
            self.replacements.append(new_uid)
        return len(uids)


def test_sync_sample_reissues_expired_prompt_and_reports_metrics(tq_init, partition_id):
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, reissue_fn=reissuer)
        batch, metrics = SampleThread(
            lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        ).result_or_raise()
        assert reissuer.calls == [[stuck]]
        assert [key.split("_")[0] for key in batch.keys] == reissuer.replacements
        assert metrics[f"{PREFIX}/inflight_reissued"] == 1
        assert metrics[f"{PREFIX}/inflight_expired_age_max"] >= 100
    finally:
        _clear_partition(partition_id)


def test_async_wait_for_sampleable_reissues_expired_prompt(tq_init, partition_id):
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id, global_steps=1)
    try:
        rb = _make_rb(trainer_mode="separate_async", inflight_ttl_s=1, reissue_fn=reissuer)
        sampleable, metrics = SampleThread(
            lambda: rb.wait_for_sampleable(global_steps=1, partition_id=partition_id, target_count=1)
        ).result_or_raise()
        assert reissuer.calls == [[stuck]]
        assert sampleable == set(reissuer.replacements)
        assert metrics[f"{PREFIX}/inflight_reissued"] == 1
    finally:
        _clear_partition(partition_id)


def test_retired_attempt_late_writes_are_swept(tq_init, partition_id):
    """A slow-not-dead attempt that writes after being re-issued cannot leave rows behind or resurrect
    its prompt key: its uid is retired, and the buffer clears whatever it still writes on the next poll."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, reissue_fn=reissuer)
        SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)).result_or_raise()
        assert rb._retired_uids[partition_id] == {stuck}

        # The retired attempt wakes up: a trajectory row and a tag-only terminal status on the old key.
        late_key = _add_trajectory(partition_id, stuck, session_id=1)
        tq.kv_put(key=stuck, partition_id=partition_id, tag={"status": "finished"})
        assert {late_key, stuck} <= set(_partition_items(partition_id))

        rb._sync_metadata_from_transfer_queue()
        assert rb._sweep_retired_attempts(partition_id) == 2
        assert not {late_key, stuck} & set(_partition_items(partition_id))
        assert late_key not in rb.partitions[partition_id]
    finally:
        _clear_partition(partition_id)


def test_ttl_off_keeps_the_pre_existing_blocking_behaviour(tq_init, partition_id):
    """Documented baseline: without a TTL a prompt with no terminal status blocks sampling until one is
    written (on main this is forever when the writer is dead)."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    rb = _make_rb()  # inflight_ttl_s=None
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1))
    try:
        sampler.start()
        sampler.join(1.0)
        assert sampler.is_alive(), "sample returned without a terminal status"
        # Only a terminal status unblocks it (what a live worker would write).
        _finish(partition_id, stuck)
        batch, metrics = sampler.result_or_raise()
        assert [key.split("_")[0] for key in batch.keys] == [stuck]
        assert f"{PREFIX}/inflight_reissued" not in metrics
    finally:
        _clear_partition(partition_id)


# ----------------------------------------------------------------------------------------------------------
# Review fixes: the re-issue window race, the re-issue cap, validation, the DAPO drain, restart orphans
# ----------------------------------------------------------------------------------------------------------


def test_terminal_flip_in_the_reissue_window_is_sampled_not_retired(tq_init, partition_id):
    """A slow-but-alive worker finishes the expired group after the buffer's snapshot but before the trainer
    re-issues it: nothing is re-issued, nothing is retired, and the finished group is sampled."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    stub = _make_trainer_stub(trainer_mode="sync")
    calls: list[list[str]] = []

    def racing_reissue(pid: str, uids: list[str]):
        calls.append(list(uids))
        _finish(partition_id, stuck)  # lands in the window
        return stub._reissue_inflight_prompts(pid, uids=uids)

    try:
        rb = _make_rb(inflight_ttl_s=1, reissue_fn=racing_reissue)
        batch, metrics = SampleThread(
            lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        ).result_or_raise()
        assert calls == [[stuck]]
        assert stub.agent_loop_manager.batches == []
        assert [key.split("_")[0] for key in batch.keys] == [stuck]
        assert rb._retired_uids[partition_id] == set()
        assert metrics.get(f"{PREFIX}/inflight_reissued", 0) == 0
    finally:
        _clear_partition(partition_id)


def test_exhausted_reissues_become_a_failure_group_that_the_refill_path_replaces(tq_init, partition_id):
    """After ``max_inflight_reissues`` attempts an expired prompt is not re-issued again: it is marked
    ``failure`` under a fresh uid (old rows cleared, old uid retired) and the existing failure eviction +
    refill replaces it, so the step completes instead of looping."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100, attempt=4)
    _add_trajectory(partition_id, stuck, session_id=0)
    reissuer = RecordingReissuer(partition_id)
    refills: list[int] = []

    def refill_fn(count: int) -> int:
        refills.append(count)
        for _ in range(count):
            new_uid = _uid()
            _submit_prompt(partition_id, new_uid, "running", 1, dispatched_at=time.time())
            _finish(partition_id, new_uid)
        return count

    try:
        rb = _make_rb(
            trainer_mode="separate_async",
            inflight_ttl_s=1,
            max_inflight_reissues=3,
            reissue_fn=reissuer,
            refill_fn=refill_fn,
        )
        sampleable, metrics = SampleThread(
            lambda: rb.wait_for_sampleable(global_steps=1, partition_id=partition_id, target_count=1)
        ).result_or_raise()
        assert reissuer.calls == [], "no 4th re-issue"
        assert refills == [1], "the failure group was evicted and refilled once"
        assert len(sampleable) == 1 and stuck not in sampleable
        assert metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1
        assert metrics[f"{PREFIX}/rollout_failure/evicted_samples"] == 1
        assert stuck in rb._retired_uids[partition_id]
        assert not any(key.startswith(stuck) for key in _partition_items(partition_id))
    finally:
        _clear_partition(partition_id)


def test_draining_clears_expired_prompts_instead_of_regenerating(tq_init, partition_id):
    """Sync DAPO drain: in-flight groups will be discarded as surplus, so an expired one is cleared, not re-issued."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, reissue_fn=reissuer)
        rb._sync_metadata_from_transfer_queue()
        metrics: dict = {}
        assert rb._recover_stuck_inflight(partition_id, metrics, draining=True) is True
        assert reissuer.calls == []
        assert metrics[f"{PREFIX}/inflight_discarded_while_draining"] == 1
        assert stuck not in _partition_items(partition_id) and stuck in rb._retired_uids[partition_id]
    finally:
        _clear_partition(partition_id)


def test_restart_reissue_drops_rows_without_a_prompt_group(tq_init, partition_id):
    """Checkpoint restore: rows no prompt refers to any more (a retired attempt's late write, the batch trained
    right before the save) are dropped by the restart-time re-issue instead of living on as garbage."""
    inflight, retired = _uid(), _uid()
    _submit_prompt(partition_id, inflight, "running", 1, dispatched_at=time.time())
    late_row = _add_trajectory(partition_id, retired, session_id=0)
    tq.kv_put(key=retired, partition_id=partition_id, tag={"status": "finished"})  # tag-only, not a prompt
    try:
        stub = _make_trainer_stub(trainer_mode="separate_async", global_steps=9)
        assert stub._reissue_inflight_prompts(partition_id) == 1
        items = _partition_items(partition_id)
        assert late_row not in items and retired not in items
        assert items[inflight]["status"] == "pending" and items[inflight]["attempt"] == 1
    finally:
        _clear_partition(partition_id)


def test_bool_ttl_is_rejected():
    with pytest.raises(AssertionError, match="inflight_ttl_s"):
        _make_rb(inflight_ttl_s=True, reissue_fn=lambda partition, uids: [])
    with pytest.raises(AssertionError, match="max_inflight_reissues"):
        _make_rb(inflight_ttl_s=1, max_inflight_reissues=True, reissue_fn=lambda partition, uids: [])


def test_dispatch_rotates_the_start_worker_across_calls(tq_init):
    """A re-issue (or refill) batch smaller than the worker count is not pinned to worker 0: the start index
    rotates, so a prompt stuck on an alive-but-silent worker reaches another worker on a later re-issue."""
    import ray
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

    @ray.remote
    class RecordingWorker:
        def __init__(self, *args):
            self.seen: list[str] = []

        async def generate_sequences(self, batch) -> None:
            self.seen.extend(str(uid) for uid in batch["uid"])

        def received(self) -> list[str]:
            return list(self.seen)

    class RecordingManager(AgentLoopManagerTQ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = RecordingWorker

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 2}}, "model": {}}})
    manager = RecordingManager.create(config, None)
    try:
        full, _ = _batch_of(["a", "b"])
        manager.generate_sequences(full)
        for uid in ("c", "d", "e"):
            manager.generate_sequences(_batch_of([uid])[0])
        received = [ray.get(worker.received.remote()) for worker in manager.agent_loop_workers]
        assert received == [["a", "c", "e"], ["b", "d"]]
    finally:
        for worker in manager.agent_loop_workers:
            ray.kill(worker, no_restart=True)


def _batch_of(uids: list[str]):
    batch = tu.get_tensordict(
        {"uid": uids, "raw_prompt": [f"p-{uid}" for uid in uids], "index": torch.arange(len(uids))}
    )
    tu.assign_non_tensor_data(batch, "global_steps", 1)
    return batch, uids


def _inflight_prompt_uids(partition_id: str) -> list[str]:
    """Prompts registered but not terminal (a re-issued attempt is ``pending`` until a worker flips it)."""
    return [
        key
        for key, tag in _partition_items(partition_id).items()
        if tag.get("is_prompt") and tag.get("status") in ("pending", "running")
    ]


@pytest.mark.parametrize("trainer_mode", ["sync", "separate_async"])
def test_every_replacement_slow_is_bounded_by_the_step_budget_then_the_ttl_waits(
    tq_init, partition_id, trainer_mode, caplog
):
    """Every generation exceeds the TTL: the original, every re-issue (real ``_reissue_inflight_prompts``,
    ``attempt + 1``) and every refill that replaces an exhausted prompt (whose attempt starts again at 1, so
    the per-prompt cap alone would regenerate forever). The per-step budget bounds the TTL-induced discards
    (re-issues and exhaustion failures together); once spent the in-flight generation is waited for (one
    warning, the metric) as ``main`` would, and the step completes as soon as it finishes."""
    caplog.set_level(logging.WARNING, logger=ReplayBuffer.__module__)
    stub = _make_trainer_stub(trainer_mode=trainer_mode, global_steps=1)
    refills: list[str] = []

    def refill_fn(count: int) -> int:
        for _ in range(count):
            new_uid = _uid()
            _submit_prompt(partition_id, new_uid, "running", 1, dispatched_at=time.time())  # attempt 1 again
            refills.append(new_uid)
        return count

    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    try:
        rb = _make_rb(
            trainer_mode=trainer_mode,
            inflight_ttl_s=0.3,
            max_inflight_reissues=1,
            max_inflight_reissues_per_step=3,
            sync_refill_failed_groups=trainer_mode == "sync",
            gen_batch_size=1 if trainer_mode == "sync" else None,
            refill_fn=refill_fn,
            reissue_fn=stub._reissue_inflight_prompts,
        )
        if trainer_mode == "sync":
            thread = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1))
        else:
            thread = SampleThread(
                lambda: rb.wait_for_sampleable(global_steps=1, partition_id=partition_id, target_count=1)
            )
        thread.start()
        deadline = time.time() + 15
        while not any("fails nothing more this step" in record.message for record in caplog.records):
            assert thread.is_alive() and time.time() < deadline, "the TTL was never suspended"
            time.sleep(POLL_INTERVAL)
        # Budget 3 spent: re-issue of the original (1), its exhausted second attempt failed (2), the refill
        # re-issued (3); the refill's second attempt stays in flight past its expiry instead of being failed.
        time.sleep(0.5)
        inflight = _inflight_prompt_uids(partition_id)
        assert len(inflight) == 1 and inflight[0] not in rb._retired_uids[partition_id]
        assert inflight[0] not in refills, "the re-issued attempt of the refill, not a new refill"
        assert len(stub.agent_loop_manager.batches) == 2, "re-issue dispatches"
        assert len(refills) == 1
        _finish(partition_id, inflight[0], 1)  # the honest generation finishes late
        result, metrics = thread.result_or_raise()
        if trainer_mode == "sync":
            assert [key.split("_")[0] for key in result.keys] == inflight
        else:
            assert result == set(inflight)
        assert metrics[f"{PREFIX}/inflight_reissued"] == 2
        assert metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1, "re-issues + exhaustion failures == budget 3"
        assert metrics[f"{PREFIX}/rollout_failure/evicted_samples"] == 1
        assert metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1
        suspensions = [record for record in caplog.records if "fails nothing more this step" in record.message]
        assert len(suspensions) == 1 and "max_inflight_reissues_per_step=3" in suspensions[0].message
    finally:
        _clear_partition(partition_id)


def test_step_budget_caps_a_poll_to_the_oldest_expired_prompts(tq_init, partition_id):
    """Three prompts expire in one poll with two re-issues left in the budget: the two oldest are re-issued,
    the third waits and is never re-issued later in the same call."""
    now = time.time()
    uids = [_uid() for _ in range(3)]
    for age, uid in zip((300, 200, 100), uids, strict=True):
        _submit_prompt(partition_id, uid, "running", 1, dispatched_at=now - age)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, max_inflight_reissues_per_step=2, reissue_fn=reissuer)
        rb._sync_metadata_from_transfer_queue()
        metrics: dict = {}
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=rb._reissue_budget(3)) is True
        assert reissuer.calls == [sorted(uids[:2])]
        assert metrics[f"{PREFIX}/inflight_reissued"] == 2 and metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1
        rb._sync_metadata_from_transfer_queue()
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=2) is False
        assert reissuer.calls == [sorted(uids[:2])] and uids[2] in rb.running_keys[partition_id]
    finally:
        _clear_partition(partition_id)


def test_step_budget_defaults_to_the_per_prompt_cap_plus_one_times_the_steps_prompt_count():
    reissue_fn = lambda partition, uids: []  # noqa: E731
    assert _make_rb(inflight_ttl_s=1, max_inflight_reissues=3, reissue_fn=reissue_fn)._reissue_budget(16) == 64
    rb = _make_rb(inflight_ttl_s=1, max_inflight_reissues=3, reissue_fn=reissue_fn, train_batch_size=8)
    assert rb._reissue_budget(16) == 32, "the step's prompt count, not the sampling call's batch size"
    assert _make_rb(inflight_ttl_s=1, max_inflight_reissues=0, reissue_fn=reissue_fn)._reissue_budget(16) == 16
    assert _make_rb(inflight_ttl_s=1, max_inflight_reissues_per_step=5, reissue_fn=reissue_fn)._reissue_budget(16) == 5
    assert _make_rb(inflight_ttl_s=1, max_inflight_reissues_per_step=0, reissue_fn=reissue_fn)._reissue_budget(16) == 0
    for bad in (True, -1, 1.5):
        with pytest.raises(AssertionError, match="max_inflight_reissues_per_step"):
            _make_rb(inflight_ttl_s=1, max_inflight_reissues_per_step=bad, reissue_fn=reissue_fn)


def test_zero_max_inflight_reissues_fails_an_expired_prompt_on_its_first_expiry(tq_init, partition_id):
    """``max_inflight_reissues=0`` means never re-issue: the step budget resolves to 0 and must not make the
    TTL inert; the expired prompt is failed under a fresh uid at once."""
    stuck = _uid()
    _submit_prompt(partition_id, stuck, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, max_inflight_reissues=0, reissue_fn=reissuer)
        rb._begin_ttl_budget(partition_id, 1)
        rb._sync_metadata_from_transfer_queue()
        metrics: dict = {}
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=rb._reissue_budget(4)) is True
        assert reissuer.calls == [] and stuck in rb._retired_uids[partition_id]
        assert (
            metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1 and f"{PREFIX}/inflight_ttl_suspended" not in metrics
        )
        assert rb._ttl_discards_in_step[partition_id] == 1, "the failure is charged against the step budget"
        assert [k for k, t in _partition_items(partition_id).items() if t.get("status") == "failure"]
    finally:
        _clear_partition(partition_id)


def test_exhaustion_failures_are_charged_against_the_budget_and_the_drain_still_discards(tq_init, partition_id):
    """A re-issue and the failure that ends an exhausted prompt both spend the step budget: with budget 2 the
    re-issue and one exhaustion go through, a second exhausted prompt is held back (suspension reported) and
    stays in flight, while the sync DAPO drain still discards an expired in-flight group (never charged)."""
    first, exhausted, held, drained = _uid(), _uid(), _uid(), _uid()
    _submit_prompt(partition_id, first, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, max_inflight_reissues=1, max_inflight_reissues_per_step=2, reissue_fn=reissuer)
        rb._begin_ttl_budget(partition_id, 1)
        rb._sync_metadata_from_transfer_queue()
        metrics: dict = {}
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=2) is True
        assert reissuer.calls == [[first]] and rb._ttl_discards_in_step[partition_id] == 1
        _submit_prompt(partition_id, exhausted, "running", 1, dispatched_at=time.time() - 100, attempt=2)
        rb._sync_metadata_from_transfer_queue()
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=2) is True
        assert exhausted in rb._retired_uids[partition_id] and rb._ttl_discards_in_step[partition_id] == 2
        assert metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1
        assert f"{PREFIX}/inflight_ttl_suspended" not in metrics, "nothing held back yet"
        _submit_prompt(partition_id, held, "running", 1, dispatched_at=time.time() - 100, attempt=2)
        rb._sync_metadata_from_transfer_queue()
        assert rb._recover_stuck_inflight(partition_id, metrics, reissue_budget=2) is False
        assert held in rb.running_keys[partition_id] and held not in rb._retired_uids[partition_id]
        assert metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1
        _submit_prompt(partition_id, drained, "running", 1, dispatched_at=time.time() - 100)
        rb._sync_metadata_from_transfer_queue()
        assert rb._recover_stuck_inflight(partition_id, metrics, draining=True, reissue_budget=2) is True
        assert drained not in _partition_items(partition_id) and drained in rb._retired_uids[partition_id]
        assert metrics[f"{PREFIX}/inflight_discarded_while_draining"] == 2, "held and drained, both discarded"
        assert reissuer.calls == [[first]] and rb._ttl_discards_in_step[partition_id] == 2
    finally:
        _clear_partition(partition_id)


def test_step_budget_is_shared_by_the_sampling_calls_of_one_step_and_reset_by_the_next(tq_init, partition_id):
    """``parameter_sync_step > 1``: the sampling calls of one ``global_steps`` share the allowance; the next
    step opens a fresh one (through the real ``sample`` entry point)."""
    older, newer, later = _uid(), _uid(), _uid()
    _submit_prompt(partition_id, older, "running", 1, dispatched_at=time.time() - 200)
    _submit_prompt(partition_id, newer, "running", 1, dispatched_at=time.time() - 100)
    reissuer = RecordingReissuer(partition_id)
    try:
        rb = _make_rb(inflight_ttl_s=1, max_inflight_reissues_per_step=1, reissue_fn=reissuer)
        batch, metrics = SampleThread(
            lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1)
        ).result_or_raise()
        assert reissuer.calls == [[older]] and metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1
        # Second call of the same step: the allowance is spent, `newer` waits until its worker finishes it.
        thread = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition_id, batch_size=1))
        thread.start()
        time.sleep(0.3)
        _finish(partition_id, newer, 1)
        batch, _ = thread.result_or_raise()
        assert [key.split("_")[0] for key in batch.keys] == [newer] and reissuer.calls == [[older]]
        # Next step: a fresh allowance.
        _submit_prompt(partition_id, later, "running", 2, dispatched_at=time.time() - 100)
        SampleThread(lambda: rb.sample(global_steps=2, partition_id=partition_id, batch_size=1)).result_or_raise()
        assert reissuer.calls == [[older], [later]]
    finally:
        _clear_partition(partition_id)
