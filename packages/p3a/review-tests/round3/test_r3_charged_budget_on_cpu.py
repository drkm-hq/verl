"""R3-A1..A3: the charged TTL step budget (d834b6b3) against the real sampling loops with a virtual clock.

A1 (expected to hold per docs/PR body "degrades to main's behaviour", "pending healthy work completes and the
step returns"): sync mode WITHOUT ``sync_refill_failed_groups`` (the default padding path), every generation
healthy but slower than the TTL. The budget ``(cap + 1) x train_batch_size`` is exactly enough to fail every
prompt once, nothing replaces a failure in padding mode, and the step ends in the pre-existing all-failed
``RuntimeError`` after cap + 1 expiries instead of waiting; FAILS on d834b6b3 / 45690b0f (cap 0 and cap 3).
A2 (PASS): async with two batches in flight (warm-up), cap 0: one budget of discards per step, then the step
completes; every step pays it again while the TTL is below the generation time.
A3 (PASS; rewritten for round 3b): an explicit ``max_inflight_reissues_per_step=0`` is rejected (it used to be
accepted and turn the TTL off) and ``inflight_ttl_suspended`` is reduced as a per-step max.
"""

import pytest

from _r3_loop_common import PARTITION, PREFIX, Sim


@pytest.mark.parametrize("cap", [0, 3])
def test_r3_a1_sync_padding_mode_waits_for_healthy_slow_generations_instead_of_failing_the_step(monkeypatch, cap):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=cap, refill=False, batch_size=2)
    sim.dispatch(2)
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 2, f"both healthy generations should be trained on: {sim.counts}, {metrics}"


def test_r3_a2_async_warmup_cap_zero_pays_one_budget_of_discards_per_step_then_completes(monkeypatch):
    sim = Sim(monkeypatch)
    rb = sim.buffer("separate_async", cap=0, refill=True, batch_size=2)
    sim.dispatch(4)  # num_warmup_batches=2: twice the step's prompts in flight
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 2 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 2
    assert metrics[f"{PREFIX}/inflight_ttl_suspended"] == 1 and rb._ttl_discards_in_step[PARTITION] == 2
    assert sim.counts["dispatches"] == 6, "2 healthy generations discarded and replaced, 2 waited for"
    batch, metrics = rb.sample(global_steps=2, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 2 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 2, "paid again next step"
    assert sim.counts["dispatches"] == 8 and sim.now - sim.start < 20


def test_r3_a3_explicit_budget_zero_is_rejected_and_the_suspension_flag_is_a_per_step_max(monkeypatch):
    """Round 3b (e598929b): 0 is rejected at construction (inflight_ttl_s=null is the way to disable the TTL) and
    ``training/inflight_ttl_suspended`` is reduced as a max over a step's sampling calls (0/1 per step)."""
    from verl.trainer.ppo.v1.utils import MetricsAggregator

    sim = Sim(monkeypatch)
    with pytest.raises(AssertionError, match="positive integer or null"):
        sim.buffer("sync", cap=3, refill=True, batch_size=1, budget=0)
    aggregator = MetricsAggregator()
    for _ in range(3):  # parameter_sync_step=3: every mini-batch call held an expired prompt back
        aggregator.add_step_metrics({f"{PREFIX}/inflight_ttl_suspended": 1}, sample_count=1)
    assert aggregator.get_aggregated_metrics()[f"{PREFIX}/inflight_ttl_suspended"] == 1
