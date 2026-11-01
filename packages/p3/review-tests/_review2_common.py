"""Helpers for the re-verification suite (runs on p3a, which has no liveness kwargs, and on p3b)."""

import inspect
import time

import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer

POLL_INTERVAL = 0.05
HAS_RELAUNCH = hasattr(PPOTrainer, "_reap_dead_workers")


class FakeManager:
    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


def make_trainer2(manager, trainer_mode: str = "sync", global_steps: int = 1):
    stub = type("Stub", (), {})()
    stub.trainer_mode = trainer_mode
    stub.global_steps = global_steps
    stub.inflight_recovery_enabled = True
    stub.agent_loop_manager = manager
    names = ["_reissue_inflight_prompts", "_submit_batch_to_rollout"] + (["_reap_dead_workers"] if HAS_RELAUNCH else [])
    for name in names:
        setattr(stub, name, getattr(PPOTrainer, name).__get__(stub))
    return stub


def make_rb2(trainer_mode: str = "sync", **overrides) -> ReplayBuffer:
    """Construct the buffer with only the kwargs this branch's __init__ accepts."""
    cls = ReplayBuffer if trainer_mode == "sync" else ReplayBufferAsync
    kwargs = dict(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=None if trainer_mode == "sync" else 8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=POLL_INTERVAL,
    )
    kwargs.update(overrides)
    accepted = inspect.signature(cls.__init__).parameters
    unknown = set(kwargs) - set(accepted)
    assert not unknown, f"this branch's ReplayBuffer does not accept {unknown}"
    return cls(**kwargs)


def submit_prompt(partition: str, key: str, status: str, dispatched_at: float, global_steps: int = 1, attempt: int = 1) -> None:
    batch = TensorDict({"index": torch.tensor([0])}, batch_size=[1])
    batch["uid"] = [key]
    batch["raw_prompt"] = [f"prompt-{key}"]
    tag = {"is_prompt": True, "status": status, "global_steps": global_steps, "dispatched_at": dispatched_at, "attempt": attempt}
    tq.kv_batch_put(keys=[key], partition_id=partition, tags=[tag], fields=batch)


def add_row(partition: str, key: str, session: int = 0, index: int = 0, global_steps: int = 1) -> str:
    row = f"{key}_{session}_{index}"
    tq.kv_put(
        key=row,
        partition_id=partition,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": global_steps},
    )
    return row


def finish_by_hand(partition: str, key: str, global_steps: int = 1) -> None:
    add_row(partition, key, global_steps=global_steps)
    tq.kv_put(key=key, partition_id=partition, tag={"status": "finished"})


def items(partition: str) -> dict:
    return tq.kv_list(partition_id=partition).get(partition, {})


def prompt_uids(partition: str) -> dict[str, dict]:
    return {key: tag for key, tag in items(partition).items() if tag.get("is_prompt")}


def clear(partition: str) -> None:
    keys = list(items(partition).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition)


def wait_until(predicate, timeout: float = 10.0, what: str = "condition") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"timed out waiting for {what}")
