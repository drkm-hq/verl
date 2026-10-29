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
"""Dead or stuck agent loop workers, end to end on Ray local mode.

Real ``AgentLoopManagerTQ`` (with stub worker actors that write TransferQueue statuses the way
``AgentLoopWorkerTQ`` does), real ``ReplayBuffer`` and the real ``PPOTrainer`` submit / re-issue / reap
methods bound to a stub trainer. A worker is killed with ``ray.kill`` (actor death) or told to stay silent
(the engine-RPC-hang class), and each test bounds its wait so a regression shows as a failure, not a hang.
"""

import asyncio
import threading
import time
import uuid

import pytest
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import replay_buffer as replay_buffer_module
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.agent_loop_tq import (
    MAX_WORKER_RELAUNCHES,
    WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD,
    AgentLoopManagerTQ,
)
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

POLL_INTERVAL = 0.05
# _submit_batch_to_rollout registers prompts in "train"; every test clears the partition it used.
PARTITION = "train"


@pytest.fixture(scope="module")
def ray_tq():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
    tq.init()
    yield
    tq.close()
    ray.shutdown()


@ray.remote
class StubWorkerTQ:
    """Stands in for ``AgentLoopWorkerTQ``: marks each prompt ``running``, then (unless stuck) writes one
    trajectory row and the terminal ``finished`` status. Takes the manager's construction arguments."""

    def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
        tq.init()
        self.mode = "finish"
        self.delay = 0.0
        self.seen: list[str] = []
        self.background: set = set()

    def set_mode(self, mode: str, delay: float = 0.0) -> None:
        # "finish": normal; "stuck": never writes a terminal status; "stuck_once": only for the next batch;
        # "slow": finishes ``delay`` seconds later in a background task, like a long generation.
        self.mode, self.delay = mode, delay

    def received(self) -> list[str]:
        return list(self.seen)

    async def ping(self) -> bool:
        return True

    async def _finish(self, uid: str, partition: str, step: int) -> None:
        await tq.async_kv_put(
            key=f"{uid}_0_0",
            partition_id=partition,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "seq_len": 3, "global_steps": step},
        )
        await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "finished"})

    async def _finish_later(self, uid: str, partition: str, step: int) -> None:
        await asyncio.sleep(self.delay)
        await self._finish(uid, partition, step)

    async def generate_sequences(self, batch) -> None:
        validate = bool(batch["validate"]) if "validate" in batch.keys() else False
        partition = "val" if validate else PARTITION
        step = int(batch["global_steps"])
        uids = [str(u) for u in batch["uid"]]
        self.seen.extend(uids)
        for uid in uids:
            await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "running"})
            if self.mode in ("stuck", "stuck_once"):
                continue
            if self.mode == "slow":
                task = asyncio.create_task(self._finish_later(uid, partition, step))
                self.background.add(task)
                task.add_done_callback(self.background.discard)
                continue
            await self._finish(uid, partition, step)
        if self.mode == "stuck_once":
            self.mode = "finish"


class StubManagerTQ(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = StubWorkerTQ


def _make_manager(num_workers: int = 2, relaunch: bool = False) -> StubManagerTQ:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"agent": {"num_workers": num_workers, "relaunch_dead_workers": relaunch}},
                "model": {},
            }
        }
    )
    return StubManagerTQ.create(config, None)


def _make_trainer(manager, global_steps: int = 1):
    """Bind the real submit / re-issue / reap methods to a stub exposing only what they use."""
    stub = type("Stub", (), {})()
    stub.trainer_mode = "sync"
    stub.global_steps = global_steps
    stub.inflight_recovery_enabled = True
    stub.agent_loop_manager = manager
    for name in ("_reissue_inflight_prompts", "_submit_batch_to_rollout", "_reap_dead_workers"):
        setattr(stub, name, getattr(PPOTrainer, name).__get__(stub))
    return stub


def _make_rb(trainer, ttl: float | None = None, liveness: bool = False, **kwargs) -> ReplayBuffer:
    return ReplayBuffer(
        **{
            **dict(
                trainer_mode="sync",
                trainer_config={},
                max_off_policy_threshold=None,
                max_off_policy_strategy="drop",
                sampler_kwargs={},
                poll_interval=POLL_INTERVAL,
                inflight_ttl_s=ttl,
                reissue_fn=trainer._reissue_inflight_prompts if (ttl is not None or liveness) else None,
                liveness_fn=trainer._reap_dead_workers if liveness else None,
                liveness_period_s=0.2 if liveness else None,
            ),
            **kwargs,
        }
    )


def _batch(num_prompts: int, global_steps: int = 1):
    uids = [uuid.uuid4().hex for _ in range(num_prompts)]
    batch = tu.get_tensordict(
        {"uid": uids, "raw_prompt": [f"prompt-{uid}" for uid in uids], "index": torch.arange(num_prompts)}
    )
    tu.assign_non_tensor_data(batch, "global_steps", global_steps)
    return batch, uids


def _prompt_statuses(partition: str = PARTITION) -> dict[str, str]:
    items = tq.kv_list(partition_id=partition).get(partition, {})
    return {key: tag["status"] for key, tag in items.items() if tag.get("is_prompt")}


def _wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"timed out waiting for {what}")


def _teardown(manager) -> None:
    for partition in (PARTITION, "val"):
        keys = list(tq.kv_list(partition_id=partition).get(partition, {}).keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
    for worker in manager.agent_loop_workers:
        ray.kill(worker, no_restart=True)


class SampleThread(threading.Thread):
    """Run the blocking ``ReplayBuffer.sample`` so the test can bound the wait."""

    def __init__(self, rb: ReplayBuffer, batch_size: int, partition: str = PARTITION):
        super().__init__(daemon=True)
        self.rb = rb
        self.batch_size = batch_size
        self.partition = partition
        self.result = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.rb.sample(global_steps=1, partition_id=self.partition, batch_size=self.batch_size)
        except Exception as e:  # surfaced through result_or_raise
            self.error = e

    def result_or_raise(self, timeout: float = 30.0):
        if self.ident is None:
            self.start()
        self.join(timeout)
        assert not self.is_alive(), "sample did not return in time"
        if self.error is not None:
            raise self.error
        return self.result


def _submit_half_stuck(manager, trainer, mode: str = "stuck"):
    """Dispatch 4 prompts over 2 workers; worker 1 leaves its 2 prompts ``running`` forever."""
    ray.get(manager.agent_loop_workers[1].set_mode.remote(mode))
    batch, uids = _batch(4)
    trainer._submit_batch_to_rollout(batch)
    _wait_until(
        lambda: sorted(_prompt_statuses().get(uid) for uid in uids) == ["finished", "finished", "running", "running"],
        what="2 finished + 2 running prompts",
    )
    return uids


def test_ttl_reissues_prompts_of_a_silently_stuck_worker(ray_tq):
    """A worker that never writes a terminal status (engine RPC hang): with inflight_ttl_s the two stuck
    prompts are re-issued under fresh uids and the step completes with 4 groups."""
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    try:
        uids = _submit_half_stuck(manager, trainer, mode="stuck_once")
        started = time.time()
        batch, metrics = SampleThread(_make_rb(trainer, ttl=1.0), batch_size=4).result_or_raise()
        elapsed = time.time() - started

        assert len(batch.keys) == 4
        sampled_uids = {key.split("_")[0] for key in batch.keys}
        stuck_uids = {uid for uid in uids if uid not in sampled_uids}
        assert len(stuck_uids) == 2, "the two stuck prompts were retired and replaced by fresh uids"
        assert metrics["training/inflight_reissued"] == 2
        assert metrics["training/inflight_dead_worker_reissued"] == 0
        assert metrics["training/inflight_expired_age_max"] >= 1.0
        assert elapsed < 15.0
        assert not (set(_prompt_statuses()) & stuck_uids), "retired prompt keys are gone"
    finally:
        _teardown(manager)


def test_ttl_off_stuck_worker_blocks_sampling(ray_tq):
    """Pre-existing behaviour, kept when the TTL is null: sampling blocks until a terminal status is
    written, which a stuck worker never does. The test unblocks it by hand instead of hanging."""
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    sampler = SampleThread(_make_rb(trainer, ttl=None), batch_size=4)
    try:
        uids = _submit_half_stuck(manager, trainer, mode="stuck")
        sampler.start()
        sampler.join(1.5)
        assert sampler.is_alive(), "sample returned without the stuck prompts' terminal status"

        for uid, status in _prompt_statuses().items():
            if uid in uids and status == "running":
                tq.kv_put(
                    key=f"{uid}_0_0",
                    partition_id=PARTITION,
                    fields={"input_ids": torch.tensor([1, 2, 3])},
                    tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
                )
                tq.kv_put(key=uid, partition_id=PARTITION, tag={"status": "finished"})
        batch, metrics = sampler.result_or_raise()
        assert {key.split("_")[0] for key in batch.keys} == set(uids)
        assert "training/inflight_reissued" not in metrics
    finally:
        _teardown(manager)


def test_killed_worker_is_relaunched_at_redispatch_and_its_prompts_reissued(ray_tq):
    """Actor death mid-generation with the TTL on and relaunch on: the stuck prompts expire, the re-issue
    dispatch hits the dead worker, which is re-created, and the step completes."""
    manager = _make_manager(relaunch=True)
    trainer = _make_trainer(manager)
    try:
        _submit_half_stuck(manager, trainer)
        dead = manager.agent_loop_workers[1]
        ray.kill(dead, no_restart=True)

        batch, metrics = SampleThread(_make_rb(trainer, ttl=1.0, liveness=False), batch_size=4).result_or_raise()

        assert len(batch.keys) == 4
        assert metrics["training/inflight_reissued"] == 2
        assert manager.agent_loop_workers[1] is not dead
        assert ray.get(manager.agent_loop_workers[1].ping.remote()) is True
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
        # The 2-row re-issue batch is chunked over both workers; the dead one's chunk was re-dispatched.
        assert manager.recovery_metrics["rollout/prompts_redispatched"] == 1
    finally:
        _teardown(manager)


def test_killed_worker_with_relaunch_off_raises_instead_of_hanging(ray_tq, caplog):
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    try:
        _submit_half_stuck(manager, trainer)
        ray.kill(manager.agent_loop_workers[1], no_restart=True)

        with pytest.raises(ray.exceptions.RayActorError):
            SampleThread(_make_rb(trainer, ttl=1.0), batch_size=4).result_or_raise()
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 0
        assert any(
            "worker 1 died" in record.message and "relaunch_dead_workers" in record.message for record in caplog.records
        )
    finally:
        _teardown(manager)


def test_liveness_probe_reissues_dead_worker_prompts_without_ttl(ray_tq, monkeypatch):
    """relaunch_dead_workers alone: the probe finds the dead worker within one period, its in-flight
    prompts are re-issued immediately (no TTL needed) and the worker is re-created."""
    monkeypatch.setattr(replay_buffer_module, "WORKER_LIVENESS_PERIOD_SECONDS", 0.2)
    manager = _make_manager(relaunch=True)
    trainer = _make_trainer(manager)
    try:
        _submit_half_stuck(manager, trainer)
        dead = manager.agent_loop_workers[1]
        ray.kill(dead, no_restart=True)

        batch, metrics = SampleThread(_make_rb(trainer, ttl=None, liveness=True), batch_size=4).result_or_raise()

        assert len(batch.keys) == 4
        assert metrics["training/inflight_reissued"] == 2
        assert metrics["training/inflight_dead_worker_reissued"] == 2
        assert "training/inflight_expired_age_max" not in metrics
        assert manager.agent_loop_workers[1] is not dead
        assert manager.recovery_metrics["rollout/prompts_orphaned_by_dead_workers"] == 2
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
    finally:
        _teardown(manager)


def test_dispatch_to_dead_worker_relaunches_and_redispatches(ray_tq):
    manager = _make_manager(relaunch=True)
    trainer = _make_trainer(manager)
    try:
        dead = manager.agent_loop_workers[1]
        ray.kill(dead, no_restart=True)
        batch, uids = _batch(4)
        trainer._submit_batch_to_rollout(batch)  # on main: RayActorError here

        _wait_until(lambda: all(_prompt_statuses().get(uid) == "finished" for uid in uids), what="4 finished")
        assert manager.agent_loop_workers[1] is not dead
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
        assert manager.recovery_metrics["rollout/prompts_redispatched"] == 2
        assert {manager._prompt_owner[uid] for uid in uids} == {0, 1}  # recorded only with relaunch on
        # Ownership is pruned to what is still in flight on the next reap, and live workers are not touched.
        assert manager.reap_dead_workers(inflight_uids=set()) == set()
        assert manager._prompt_owner == {}
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
    finally:
        _teardown(manager)


def test_dispatch_to_dead_worker_with_relaunch_off_names_worker_and_prompts(ray_tq, caplog):
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    try:
        ray.kill(manager.agent_loop_workers[1], no_restart=True)
        batch, uids = _batch(4)
        with pytest.raises(ray.exceptions.RayActorError):  # the default path keeps the original exception type
            trainer._submit_batch_to_rollout(batch)
        logged = [record.message for record in caplog.records if "worker 1 died and 2 prompts" in record.message]
        assert logged and any(uid in logged[0] for uid in uids[2:])
    finally:
        _teardown(manager)


def test_reissue_rotates_away_from_a_wedged_worker(ray_tq):
    """A re-issue batch smaller than the worker count is dispatched from a rotating start index, so a prompt
    stuck on an alive-but-silent worker 0 reaches worker 1 within a couple of TTLs instead of looping."""
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    try:
        ray.get(manager.agent_loop_workers[0].set_mode.remote("stuck"))
        batch, uids = _batch(2)
        trainer._submit_batch_to_rollout(batch)
        _wait_until(lambda: sorted(_prompt_statuses().get(u) for u in uids) == ["finished", "running"], what="1 stuck")

        out, metrics = SampleThread(_make_rb(trainer, ttl=0.5), batch_size=2).result_or_raise()
        assert len(out.keys) == 2
        assert 1 <= metrics["training/inflight_reissued"] <= 2
        received_by_1 = ray.get(manager.agent_loop_workers[1].received.remote())
        assert len(received_by_1) == 2, "the stuck prompt's re-issue reached worker 1"
    finally:
        _teardown(manager)


def test_ttl_below_generation_time_fails_the_group_after_max_reissues_and_refill_replaces_it(ray_tq):
    """A TTL shorter than a healthy generation does not livelock: after ``max_inflight_reissues`` the prompt
    becomes a ``failure`` group and sync failure refill replaces it; the late rows of the retired attempts
    are swept."""
    manager = _make_manager(relaunch=False)
    trainer = _make_trainer(manager)
    refills: list[int] = []

    def refill_fn(count: int) -> int:
        refills.append(count)
        for _ in range(count):
            uid = uuid.uuid4().hex
            tq.kv_put(key=uid, partition_id=PARTITION, tag={"is_prompt": True, "status": "running", "global_steps": 1})
            tq.kv_put(
                key=f"{uid}_0_0",
                partition_id=PARTITION,
                fields={"input_ids": torch.tensor([1, 2, 3])},
                tag={"is_prompt": False, "seq_len": 3, "global_steps": 1},
            )
            tq.kv_put(key=uid, partition_id=PARTITION, tag={"status": "finished"})
        return count

    try:
        for worker in manager.agent_loop_workers:
            ray.get(worker.set_mode.remote("slow", 1.5))
        batch, uids = _batch(2)
        trainer._submit_batch_to_rollout(batch)
        _wait_until(lambda: all(_prompt_statuses().get(u) == "running" for u in uids), what="2 running")

        rb = _make_rb(
            trainer,
            ttl=0.4,
            max_inflight_reissues=2,
            max_inflight_reissues_per_step=8,  # above 2 x 2: the per-prompt cap must fire before the step budget
            sync_refill_failed_groups=True,
            gen_batch_size=1,
            refill_fn=refill_fn,
        )
        out, metrics = SampleThread(rb, batch_size=2).result_or_raise()
        assert len({key.split("_")[0] for key in out.keys}) == 2
        assert metrics["training/inflight_reissued"] == 4, "2 prompts x 2 allowed re-issues"
        assert metrics["training/inflight_reissue_exhausted"] == 2
        assert refills == [2] or refills == [1, 1]
        assert len(rb._retired_uids[PARTITION]) == 6
        time.sleep(2.0)  # let the retired attempts finish and write late
        rb._sync_metadata_from_transfer_queue()
        rb._sweep_retired_attempts(PARTITION)
        assert not any(
            key.split("_")[0] in rb._retired_uids[PARTITION] for key in tq.kv_list(PARTITION).get(PARTITION, {})
        )
    finally:
        _teardown(manager)


def test_validation_dead_worker_is_relaunched_and_its_val_groups_fail(ray_tq):
    """A worker that dies while a validation batch is in flight is found by the probe during ``sample("val")``,
    relaunched, and its validation groups are marked ``failure`` in place so validation completes."""
    manager = _make_manager(relaunch=True)
    trainer = _make_trainer(manager)
    try:
        ray.get(manager.agent_loop_workers[1].set_mode.remote("stuck"))
        batch, uids = _batch(2)
        tu.assign_non_tensor_data(batch, "validate", True)
        tq.kv_batch_put(keys=uids, partition_id="val", tags=[trainer_base._pending_prompt_tag(1) for _ in uids])
        manager.generate_sequences(batch)
        _wait_until(
            lambda: sorted(_prompt_statuses("val").get(u) for u in uids) == ["finished", "running"], what="val stuck"
        )
        dead = manager.agent_loop_workers[1]
        ray.kill(dead, no_restart=True)

        rb = _make_rb(trainer, ttl=0.5, liveness=True)
        out, metrics = SampleThread(rb, batch_size=2, partition="val").result_or_raise()
        assert [key.split("_")[0] for key in out.keys] == [uids[0]], "the finished val group, as is"
        assert metrics["validation/inflight_dead_worker_failed"] == 1
        assert not any(tag.get("is_prompt") for tag in tq.kv_list("val").get("val", {}).values()), "nothing regenerated"
        assert manager.agent_loop_workers[1] is not dead
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 1
    finally:
        _teardown(manager)


def test_unavailable_probes_must_repeat_and_relaunches_are_capped(ray_tq):
    """``ActorUnavailableError`` is transient: only consecutive probes mark a worker dead (a successful ping
    resets the streak). Re-creating one index more than ``MAX_WORKER_RELAUNCHES`` times in a row raises."""
    manager = _make_manager(relaunch=True)
    try:
        ray.get([worker.ping.remote() for worker in manager.agent_loop_workers])  # actors up before probing
        unavailable = ray.exceptions.ActorUnavailableError("raylet hiccup", None)
        for _ in range(WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD - 1):
            assert manager._probe_marks_dead(1, unavailable) is False
        assert manager._probe_marks_dead(1, unavailable) is True
        assert manager._probe_marks_dead(1, unavailable) is False, "the streak restarts after a verdict"
        manager.reap_dead_workers(inflight_uids=set())  # a successful ping resets the streak
        assert manager._unavailable_streak[1] == 0
        assert manager.reap_dead_workers(inflight_uids=set()) == set()
        assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == 0

        manager._relaunch_streak[1] = MAX_WORKER_RELAUNCHES
        with pytest.raises(RuntimeError, match="giving up"):
            manager._relaunch_worker(1, RuntimeError("replacement keeps dying"))
        assert ray.get(manager.agent_loop_workers[1].ping.remote()) is True, "the live worker was not replaced"
    finally:
        _teardown(manager)
