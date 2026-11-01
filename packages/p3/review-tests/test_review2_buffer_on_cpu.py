"""Re-verification: the D1 window race and repeated re-issues (p3a and p3b, real TransferQueue)."""

import time
import uuid

import transfer_queue as tq

from _review_common import SampleThread
from _review2_common import FakeManager, add_row, clear, finish_by_hand, items, make_rb2, make_trainer2, prompt_uids, submit_prompt, wait_until
from verl.trainer.ppo.v1 import trainer_base


def _uid() -> str:
    return uuid.uuid4().hex


def test_terminal_write_inside_the_trainer_window_is_swept_and_the_fresh_attempt_is_the_only_group(ray_tq, monkeypatch):
    """N1. The slow worker writes rows + ``finished`` for the old uid AFTER the trainer's own ``kv_list`` (so the
    trainer still re-issues it) and BEFORE the trainer clears the old keys. Correct outcome: the old uid is
    retired, its late rows are swept, the fresh attempt is the only live group, nothing is lost or doubled."""
    partition = "train"
    clear(partition)
    stuck = _uid()
    submit_prompt(partition, stuck, "running", dispatched_at=time.time() - 100)
    trainer = make_trainer2(FakeManager(), "sync")
    real_get, state = trainer_base.tq.kv_batch_get, {"armed": True}

    def late_write_then_get(**kwargs):
        if state["armed"]:  # exactly once, inside the window
            state["armed"] = False
            finish_by_hand(partition, stuck)
        return real_get(**kwargs)

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", late_write_then_get)
    rb = make_rb2("sync", inflight_ttl_s=1.0, reissue_fn=trainer._reissue_inflight_prompts)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1))
    sampler.start()
    try:
        wait_until(lambda: len(trainer.agent_loop_manager.batches) == 1, what="the fresh attempt to be dispatched")
        (new_uid,) = list(trainer.agent_loop_manager.batches[0]["uid"])
        assert new_uid != stuck and prompt_uids(partition)[new_uid]["attempt"] == 2
        wait_until(lambda: not {stuck, f"{stuck}_0_0"} & set(items(partition)), what="late rows of the retired uid swept")
        assert rb._retired_uids[partition] == {stuck}
        finish_by_hand(partition, new_uid)  # what the new attempt's worker does
        out, metrics = sampler.result_or_raise(10.0)
        assert [key.split("_")[0] for key in out.keys] == [new_uid]
        assert metrics["training/inflight_reissued"] == 1
        assert not prompt_uids(partition), "exactly one group existed and it was consumed"
    finally:
        clear(partition)


def test_two_consecutive_reissues_leave_exactly_one_live_group(ray_tq):
    """N7. A prompt whose first and second attempts both expire: two re-issues, attempt 3 is the single live
    group, both retired uids are gone, and the step samples the group once."""
    partition = "train"
    clear(partition)
    first = _uid()
    submit_prompt(partition, first, "running", dispatched_at=time.time() - 100)
    add_row(partition, first)  # a partial row of the first attempt
    trainer = make_trainer2(FakeManager(), "sync")
    rb = make_rb2("sync", inflight_ttl_s=0.5, max_inflight_reissues=3, reissue_fn=trainer._reissue_inflight_prompts)
    sampler = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1))
    sampler.start()
    try:
        wait_until(lambda: len(trainer.agent_loop_manager.batches) == 1, what="first re-issue")
        (second,) = list(trainer.agent_loop_manager.batches[0]["uid"])
        add_row(partition, second)  # the second attempt also writes a partial row and then stalls
        wait_until(lambda: len(trainer.agent_loop_manager.batches) == 2, timeout=10.0, what="second re-issue")
        (third,) = list(trainer.agent_loop_manager.batches[1]["uid"])
        assert len({first, second, third}) == 3
        wait_until(lambda: set(prompt_uids(partition)) == {third}, what="a single live prompt group")
        assert prompt_uids(partition)[third]["attempt"] == 3
        wait_until(lambda: not {f"{first}_0_0", f"{second}_0_0"} & set(items(partition)), what="old rows swept")
        finish_by_hand(partition, third)
        out, metrics = sampler.result_or_raise(10.0)
        assert [key.split("_")[0] for key in out.keys] == [third] and len(out.keys) == 1
        assert metrics["training/inflight_reissued"] == 2
        assert rb._retired_uids[partition] == {first, second}
    finally:
        clear(partition)


def test_reissue_fn_returning_an_int_still_retires_only_what_left_the_registry(ray_tq):
    """N1b. Compatibility: a ``reissue_fn`` that returns a count (old contract) must not retire a group that
    turned terminal in the window; the buffer's own post-reissue registry check is sufficient."""
    partition = "train"
    clear(partition)
    stuck, calls = _uid(), []
    submit_prompt(partition, stuck, "running", dispatched_at=time.time() - 100)

    def int_reissue(partition_id: str, uids: list[str]) -> int:
        calls.append(list(uids))
        finish_by_hand(partition_id, stuck)  # turned terminal; nothing is re-issued
        return 0

    rb = make_rb2("sync", inflight_ttl_s=1.0, reissue_fn=int_reissue)
    try:
        out, metrics = SampleThread(lambda: rb.sample(global_steps=1, partition_id=partition, batch_size=1)).result_or_raise(10.0)
        assert calls == [[stuck]]
        assert [key.split("_")[0] for key in out.keys] == [stuck]
        assert rb._retired_uids[partition] == set() and metrics.get("training/inflight_reissued", 0) == 0
    finally:
        clear(partition)
