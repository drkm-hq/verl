"""Re-verification: the re-issue cap turns a prompt into a failure group that the refill paths replace once."""

import time
import uuid

from _review_common import SampleThread
from _review2_common import FakeManager, clear, finish_by_hand, items, make_rb2, make_trainer2, submit_prompt


def _uid() -> str:
    return uuid.uuid4().hex


class CountingRefill:
    """Stands in for ``_add_prompts_to_generate``: registers ``n`` fresh finished groups and counts the calls."""

    def __init__(self, partition: str):
        self.partition, self.calls, self.uids = partition, [], []

    def __call__(self, num_prompts: int) -> int:
        self.calls.append(num_prompts)
        for _ in range(num_prompts):
            new = _uid()
            submit_prompt(self.partition, new, "running", dispatched_at=time.time())
            finish_by_hand(self.partition, new)
            self.uids.append(new)
        return num_prompts


def test_cap_reached_in_sync_mode_fails_the_group_and_refill_replaces_it_exactly_once(ray_tq):
    """N2. attempt=2 with max_inflight_reissues=1 is exhausted: no re-issue, a fresh failure group, evicted by
    the sync failure path and replaced by exactly one refill; the step completes with the replacement."""
    partition = "train"
    clear(partition)
    stuck = _uid()
    submit_prompt(partition, stuck, "running", dispatched_at=time.time() - 100, attempt=2)
    trainer = make_trainer2(FakeManager(), "sync")
    refill = CountingRefill(partition)
    rb = make_rb2(
        "sync",
        inflight_ttl_s=1.0,
        max_inflight_reissues=1,
        reissue_fn=trainer._reissue_inflight_prompts,
        refill_fn=refill,
        sync_refill_failed_groups=True,
        gen_batch_size=1,
    )
    try:
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1)).result_or_raise(10.0)
        assert trainer.agent_loop_manager.batches == [], "an exhausted prompt is not re-issued"
        assert refill.calls == [1], f"exactly one replacement, got {refill.calls}"
        assert [key.split("_")[0] for key in out.keys] == refill.uids
        assert metrics["training/inflight_reissue_exhausted"] == 1
        assert metrics["training/rollout_failure/evicted_samples"] == 1
        assert metrics.get("training/inflight_reissued", 0) == 0
        assert stuck not in items(partition) and rb._retired_uids[partition] == {stuck}
        assert not [k for k, t in items(partition).items() if t.get("is_prompt")], "old, failure and refill groups all consumed"
    finally:
        clear(partition)


def test_cap_reached_in_async_mode_is_evicted_and_refilled_once(ray_tq):
    """N2b. Same in ``ReplayBufferAsync.wait_for_sampleable``: failure groups are evicted and refilled by default."""
    partition = "train"
    clear(partition)
    stuck, other = _uid(), _uid()
    submit_prompt(partition, stuck, "running", dispatched_at=time.time() - 100, attempt=3)
    submit_prompt(partition, other, "running", dispatched_at=time.time())
    finish_by_hand(partition, other)
    trainer = make_trainer2(FakeManager(), "separate_async")
    refill = CountingRefill(partition)
    rb = make_rb2(
        "separate_async", inflight_ttl_s=1.0, max_inflight_reissues=2, reissue_fn=trainer._reissue_inflight_prompts, refill_fn=refill
    )
    try:
        sampleable, metrics = SampleThread(lambda: rb.wait_for_sampleable(global_steps=1, partition_id=partition, target_count=2)).result_or_raise(10.0)
        assert trainer.agent_loop_manager.batches == []
        assert refill.calls == [1]
        assert sampleable == {other, *refill.uids}
        assert metrics["training/inflight_reissue_exhausted"] == 1
        assert metrics["training/rollout_failure/evicted_samples"] == 1
        assert stuck not in items(partition)
    finally:
        clear(partition)


def test_cap_zero_means_no_reissue_at_all(ray_tq):
    """N2c. max_inflight_reissues=0: the first expiry already fails the group (attempt 1 > 0)."""
    partition = "train"
    clear(partition)
    stuck = _uid()
    submit_prompt(partition, stuck, "running", dispatched_at=time.time() - 100)
    trainer = make_trainer2(FakeManager(), "sync")
    refill = CountingRefill(partition)
    rb = make_rb2(
        "sync", inflight_ttl_s=1.0, max_inflight_reissues=0, reissue_fn=trainer._reissue_inflight_prompts, refill_fn=refill,
        sync_refill_failed_groups=True, gen_batch_size=1,
    )
    try:
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1)).result_or_raise(10.0)
        assert trainer.agent_loop_manager.batches == [] and refill.calls == [1]
        assert metrics["training/inflight_reissue_exhausted"] == 1
        assert [key.split("_")[0] for key in out.keys] == refill.uids
    finally:
        clear(partition)
