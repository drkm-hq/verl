"""R2-A1..A5: the per-step TTL re-issue budget against the real ``ReplayBuffer._recover_stuck_inflight``.

Round-2 history: on 951bc3f4 the budget filter was applied to the whole expired set before the exhausted split
and before the DAPO ``draining`` branch (A1, A2, A5), was per sampling call (A4) and reported a suspension for
exhausted prompts it then failed (A3); all five failed. 6445f21b fixed them; d834b6b3 (round 3, the submitter's R2-02)
then CHARGED exhaustion failures against the budget, so A2 and A3 were rewritten here to the charged semantics:
A2 an exhausted prompt is failed while the step budget allows it, held back (waited for) once it is spent, and
failed again when the next ``global_steps`` reopens the budget; A3 the suspension metric fires only when an
expired prompt is actually held back. A1, A4, A5 are unchanged. Expected on d834b6b3 / 45690b0f: 5 PASS.
"""

from _r2_budget_common import PARTITION, PREFIX, MemoryTQ, make_buffer, poll


def test_r2_a1_max_inflight_reissues_zero_fails_an_expired_prompt_instead_of_waiting(monkeypatch):
    memory, log = MemoryTQ(), []
    memory.install(monkeypatch)
    rb = make_buffer(memory, log, max_inflight_reissues=0)  # per-step budget defaults to (0 + 1) x batch
    uid = memory.submit(age_s=100.0, attempt=1)
    metrics: dict = {}
    poll(rb, metrics, batch_size=4)
    assert log == [], "max_inflight_reissues=0: never regenerate"
    assert uid not in memory.inflight() and len(memory.failures()) == 1, (
        f"expected the expired prompt to be marked failure on its first expiry; metrics={metrics}"
    )


def test_r2_a2_exhaustion_failures_are_charged_and_held_back_once_the_budget_is_spent(monkeypatch):
    memory, log = MemoryTQ(), []
    memory.install(monkeypatch)
    rb = make_buffer(memory, log, max_inflight_reissues=1, max_inflight_reissues_per_step=1)
    first = memory.submit(age_s=100.0, attempt=1)
    metrics: dict = {}
    assert poll(rb, metrics, batch_size=2, global_steps=1) and log == [first], "the one discard the budget allows"
    exhausted = memory.submit(age_s=100.0, attempt=2)  # attempt > cap: a failure, which is a charged discard too
    assert poll(rb, metrics, batch_size=2, global_steps=1) is False, "budget spent: held back, wait as without a TTL"
    assert exhausted in memory.inflight() and memory.failures() == [] and log == [first]
    assert metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1 and rb._ttl_discards_in_step[PARTITION] == 1
    assert poll(rb, {}, batch_size=2, global_steps=2), "the next global_steps reopens the budget"
    assert exhausted not in memory.inflight() and len(memory.failures()) == 1
    assert rb._ttl_discards_in_step[PARTITION] == 1, "the failure is charged to the new step"


def test_r2_a3_suspension_is_reported_only_when_an_expired_prompt_is_held_back(monkeypatch):
    memory, log = MemoryTQ(), []
    memory.install(monkeypatch)
    rb = make_buffer(memory, log, max_inflight_reissues=1, max_inflight_reissues_per_step=2)
    memory.submit(age_s=200.0, attempt=2)
    memory.submit(age_s=100.0, attempt=2)
    metrics: dict = {}
    assert poll(rb, metrics, batch_size=2, global_steps=1)
    assert log == [] and len(memory.failures()) == 2 and memory.inflight() == set()
    assert metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 2
    assert f"{PREFIX}/inflight_ttl_suspended" not in metrics, f"both fit the budget, nothing held back: {metrics}"
    held = memory.submit(age_s=100.0, attempt=2)
    assert poll(rb, metrics, batch_size=2, global_steps=1) is False
    assert held in memory.inflight() and metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1 and log == []


def test_r2_a4_the_budget_bounds_a_step_not_only_one_sampling_call(monkeypatch):
    memory, log = MemoryTQ(), []
    memory.install(monkeypatch)
    rb = make_buffer(memory, log, max_inflight_reissues=3, max_inflight_reissues_per_step=1)
    memory.submit(age_s=100.0, attempt=1)
    memory.submit(age_s=100.0, attempt=1)
    first_call: dict = {}  # sample(mini-batch 1) of a step with parameter_sync_step=2
    poll(rb, first_call, batch_size=1, global_steps=1)
    assert len(log) == 1 and first_call[f"{PREFIX}/inflight_ttl_suspended"] == 1
    second_call: dict = {}  # sample(mini-batch 2) of the same step: fresh eviction_metrics, same step budget
    poll(rb, second_call, batch_size=1, global_steps=1)
    assert len(log) <= 1, f"max_inflight_reissues_per_step=1 but the step re-issued {len(log)} prompts"


def test_r2_a5_dapo_drain_keeps_discarding_expired_groups_after_the_budget_is_spent(monkeypatch):
    memory, log = MemoryTQ(), []
    memory.install(monkeypatch)
    rb = make_buffer(memory, log, max_inflight_reissues=3, max_inflight_reissues_per_step=1)
    memory.submit(age_s=100.0, attempt=1)
    metrics: dict = {}
    assert poll(rb, metrics, batch_size=1, global_steps=1) and len(log) == 1, "speculative phase: the one allowed re-issue"
    stuck = memory.submit(age_s=100.0, attempt=1)
    poll(rb, metrics, batch_size=1, draining=True, global_steps=1)  # has_enough_samples -> draining: surplus discarded
    assert len(log) == 1, "a drain never regenerates"
    assert stuck not in memory.inflight() and metrics.get(f"{PREFIX}/inflight_discarded_while_draining", 0) == 1, (
        f"the drain waited for an expired group it would discard; metrics={metrics}"
    )
