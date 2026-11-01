"""Shared helpers for the reviewer's adversarial tests (real ReplayBuffer / AgentLoopManagerTQ / TransferQueue)."""

import asyncio
import threading
import time
import uuid

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, _pending_prompt_tag
from verl.utils import tensordict_utils as tu

POLL_INTERVAL = 0.05


def uid() -> str:
    return uuid.uuid4().hex


@ray.remote
class ReviewWorker:
    """Stub for AgentLoopWorkerTQ with controllable behaviour per prompt.

    modes: "finish" (write one row + finished), "stuck" (running forever), "slow" (finish after ``delay`` s in a
    background task, like a long generation), "finish_first_only" (first uid of each chunk finishes, the rest
    stay running). Writes to the partition the batch's ``validate`` flag selects, like the real worker.
    """

    def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
        tq.init()
        self.mode = "finish"
        self.delay = 0.0
        self.seen: list[str] = []
        self.background = set()

    def set_mode(self, mode: str, delay: float = 0.0) -> None:
        self.mode, self.delay = mode, delay

    def received(self) -> list[str]:
        return list(self.seen)

    async def ping(self) -> bool:
        return True

    async def _finish(self, key: str, partition: str, step: int) -> None:
        await tq.async_kv_put(
            key=f"{key}_0_0",
            partition_id=partition,
            fields={"input_ids": torch.tensor([1, 2, 3])},
            tag={"is_prompt": False, "seq_len": 3, "global_steps": step},
        )
        await tq.async_kv_put(key=key, partition_id=partition, tag={"status": "finished"})

    async def _finish_later(self, key: str, partition: str, step: int) -> None:
        await asyncio.sleep(self.delay)
        await self._finish(key, partition, step)

    async def generate_sequences(self, batch) -> None:
        validate = bool(batch["validate"]) if "validate" in batch.keys() else False
        partition = "val" if validate else "train"
        step = int(batch["global_steps"])
        uids = [str(u) for u in batch["uid"]]
        self.seen.extend(uids)
        for i, key in enumerate(uids):
            await tq.async_kv_put(key=key, partition_id=partition, tag={"status": "running"})
            if self.mode == "stuck" or (self.mode == "finish_first_only" and i > 0):
                continue
            if self.mode == "slow":
                task = asyncio.create_task(self._finish_later(key, partition, step))
                self.background.add(task)
                task.add_done_callback(self.background.discard)
                continue
            await self._finish(key, partition, step)


class ReviewManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ReviewWorker


def make_manager(num_workers: int = 2, relaunch: bool = False) -> ReviewManager:
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"agent": {"num_workers": num_workers, "relaunch_dead_workers": relaunch}}, "model": {}}}
    )
    return ReviewManager.create(config, None)


def make_trainer(manager, trainer_mode: str = "sync", global_steps: int = 1):
    stub = type("Stub", (), {})()
    stub.trainer_mode = trainer_mode
    stub.global_steps = global_steps
    stub.inflight_recovery_enabled = True
    stub.agent_loop_manager = manager
    for name in ("_reissue_inflight_prompts", "_submit_batch_to_rollout", "_reap_dead_workers"):
        setattr(stub, name, getattr(PPOTrainer, name).__get__(stub))
    return stub


def make_rb(trainer, trainer_mode: str = "sync", ttl=None, liveness: bool = False, reissue_fn=None) -> ReplayBuffer:
    cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync
    if reissue_fn is None and (ttl is not None or liveness):
        reissue_fn = trainer._reissue_inflight_prompts
    return cls(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=None if trainer_mode == "sync" else 8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=POLL_INTERVAL,
        inflight_ttl_s=ttl,
        reissue_fn=reissue_fn,
        liveness_fn=trainer._reap_dead_workers if liveness else None,
    )


def make_batch(num_prompts: int, global_steps: int = 1, validate: bool = False):
    uids = [uid() for _ in range(num_prompts)]
    batch = tu.get_tensordict({"uid": uids, "raw_prompt": [f"prompt-{u}" for u in uids], "index": torch.arange(num_prompts)})
    tu.assign_non_tensor_data(batch, "global_steps", global_steps)
    if validate:
        tu.assign_non_tensor_data(batch, "validate", True)
    return batch, uids


def submit_val(manager, batch, global_steps: int = 1) -> None:
    """What PPOTrainer._validate does: tag-only registration in ``val`` and a dispatch."""
    tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=[_pending_prompt_tag(global_steps) for _ in range(len(batch))])
    manager.generate_sequences(batch)


def prompt_statuses(partition: str = "train") -> dict[str, str]:
    items = tq.kv_list(partition_id=partition).get(partition, {})
    return {key: tag["status"] for key, tag in items.items() if tag.get("is_prompt")}


def partition_keys(partition: str) -> set[str]:
    return set(tq.kv_list(partition_id=partition).get(partition, {}).keys())


def wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"timed out waiting for {what}")


def teardown(manager, partitions=("train", "val")) -> None:
    for partition in partitions:
        keys = list(partition_keys(partition))
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
    for worker in getattr(manager, "agent_loop_workers", []):
        ray.kill(worker, no_restart=True)


class SampleThread(threading.Thread):
    def __init__(self, fn):
        super().__init__(daemon=True)
        self.fn, self.result, self.error = fn, None, None

    def run(self) -> None:
        try:
            self.result = self.fn()
        except Exception as e:  # surfaced by result_or_raise
            self.error = e

    def result_or_raise(self, timeout: float = 30.0):
        if self.ident is None:
            self.start()
        self.join(timeout)
        assert not self.is_alive(), "sampling did not return in time"
        if self.error is not None:
            raise self.error
        return self.result
