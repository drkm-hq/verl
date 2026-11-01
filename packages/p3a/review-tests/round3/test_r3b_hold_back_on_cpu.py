"""R3b-H1..H3: the padding-mode hold-back (e598929b, ``keep_last_finishable``) on the real sync ``sample`` loop.

H1 (expected to hold, FAILS if the hold-back blocks recovery): B = 2, one prompt whose writer is dead (cap 0: never
re-issued) or a poison prompt that hangs on every attempt (cap 3), the other healthy but slower than the TTL.
Once the healthy group has finished, a finished group exists, so the per-prompt cap must be allowed to fail the
dead one and the step must complete padded. On e598929b ``_hold_back_last_groups`` marked the whole step
budget spent, so the step hung on the dead writer; 2bd5c040 re-evaluates the hold-back per poll and charges nothing,
so the dead writer is failed under the normal budget once the other group finished (PASS expected). H2 (PASS, by design): the only in-flight group of a step on a live
worker that never finishes is held back forever rather than failed. H3 (PASS): refill mode is unchanged.
"""

import pytest

from _r3_loop_common import PARTITION, PREFIX, Sim


@pytest.mark.parametrize("cap", [0, 3])
def test_r3b_h1_dead_writer_is_failed_and_padded_once_the_other_group_finished(monkeypatch, cap):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=cap, refill=False, batch_size=2)
    sim.dispatch(1, hung=True, poison=bool(cap))  # its writer is dead (cap 0) / it hangs on every attempt (cap 3)
    sim.dispatch(1)  # healthy, finishes 2 s after each dispatch
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 1 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1, (
        f"the healthy row trained on and the dead group padded: {sim.counts}, {metrics}"
    )
    # round 3b final (2bd5c040): the hold-back charges nothing; the re-issues and the one failure after the lift do
    assert rb._ttl_discards_in_step[PARTITION] == 2 * cap + 1 and metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1


def test_r3b_h2_sole_hung_live_group_is_held_back_forever_by_design(monkeypatch):
    sim = Sim(monkeypatch, deadline=10.0)
    rb = sim.buffer("sync", cap=0, refill=False, batch_size=1)
    sim.dispatch(1, hung=True)
    with pytest.raises(pytest.fail.Exception, match="did not return within 10.0 simulated seconds"):
        rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert sim.counts["dispatches"] == 1 and not [t for t in sim.prompt_tags().values() if t["status"] == "failure"]
    assert rb._ttl_discards_in_step[PARTITION] == 0, "round 3b final: the hold-back charges nothing (re-evaluated per poll)"


def test_r3b_h3_refill_mode_still_fails_and_replaces_the_exhausted_prompt(monkeypatch):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=0, refill=True, batch_size=1)
    sim.dispatch(1, hung=True)
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert len(batch.keys) == 1 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1 and sim.counts["dispatches"] == 2
