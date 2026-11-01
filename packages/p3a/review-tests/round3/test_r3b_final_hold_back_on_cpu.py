"""R3b-final F1/F2: the per-poll hold-back (2bd5c040) under two more shapes, sync padding mode, real ``sample`` loop.

F1 (PASS, by design): the only other group of the step is itself exhausted and never finishes either (a dead
writer beside a poison prompt): both are held back for ever, nothing is failed, nothing charged.
F2 (PASS, documented limit): an explicit ``max_inflight_reissues_per_step`` already spent by the re-issues when
the hold-back lifts: the budget suspension (not the hold-back) keeps the dead writer unfailed after the healthy
group finished, so the step waits as without a TTL; the default allowance ``(cap + 1) x B`` always leaves room
for the failures (see test_r3b_h1).
"""

import pytest

from _r3_loop_common import PARTITION, Sim


def test_r3b_f1_two_groups_that_never_finish_are_held_back_for_ever_and_charged_nothing(monkeypatch):
    sim = Sim(monkeypatch, deadline=10.0)
    rb = sim.buffer("sync", cap=0, refill=False, batch_size=2)
    sim.dispatch(1, hung=True)
    sim.dispatch(1, poison=True)
    with pytest.raises(pytest.fail.Exception, match="did not return within 10.0 simulated seconds"):
        rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert sim.counts["dispatches"] == 2 and not [t for t in sim.prompt_tags().values() if t["status"] == "failure"]
    assert rb._ttl_discards_in_step[PARTITION] == 0


def test_r3b_f2_a_budget_spent_by_reissues_keeps_the_dead_writer_unfailed_after_the_lift_by_design(monkeypatch):
    sim = Sim(monkeypatch, deadline=12.0)
    rb = sim.buffer("sync", cap=3, refill=False, batch_size=2, budget=6)  # exactly the 3 re-issues of both prompts
    sim.dispatch(1, poison=True)
    sim.dispatch(1)
    with pytest.raises(pytest.fail.Exception, match="did not return within 12.0 simulated seconds"):
        rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert sim.counts["reissues"] == 6 and sim.counts["completions"] >= 1, "the healthy group did finish"
    assert rb._ttl_discards_in_step[PARTITION] == 6, "the budget, not the hold-back, holds the dead writer"
