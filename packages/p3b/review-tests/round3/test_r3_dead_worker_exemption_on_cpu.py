"""R3-B1..B4 for P3b (45690b0f). B1 (rewritten for round 3b, PASS): the dead-worker exemption used to be unbounded (a probe that keeps reporting
the replacement's worker dead drove re-issue -> exhaustion failure -> refill forever, nothing charged); 64212367
bounds the dead-worker discards per step at (cap + 1) x train_batch_size and raises a RuntimeError beyond.
B2 (PASS): the manager keeps ``rollout/`` keys and the drain re-keys them per prefix. B3 (PASS): the real wandb
table survives shrinking/growing sample counts and an empty validation logs no row. B4 (LOW, FAILS): the
row-less counts are reported only when no other validation metric exists, so a validation with both a dead
worker's batch and a live worker's row-less batch drops the live count.
"""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "p3a" / "review-tests" / "round3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round2"))
from _r2_common import LiveWorkersWhoseSessionsAllFail, ScheduledWorkers, make_trainer  # noqa: E402
from _r3_loop_common import PARTITION, Sim  # noqa: E402

from verl.trainer.ppo.v1.trainer_base import PPOTrainer  # noqa: E402
from verl.utils.tracking import ValidationGenerationsLogger  # noqa: E402


@pytest.mark.parametrize(("cap", "allowance", "reissues", "dispatches"), [(0, 1, 0, 2), (3, 4, 4, 6)])
def test_r3_b1_dead_owned_replacements_are_bounded_per_step_and_fail_loudly(monkeypatch, cap, allowance, reissues, dispatches):
    """Round 3b (64212367): a probe that reports every replacement's worker dead now ends in a RuntimeError naming
    the counts once the dead-worker discards of the step exceed (cap + 1) x train_batch_size; re-issues and
    exhaustion failures of dead-owned prompts both count, refills do not."""
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=cap, refill=True, batch_size=1, liveness_fn=lambda uids: set(uids), liveness_period_s=0)
    sim.dispatch(1)
    with pytest.raises(RuntimeError, match="agent-loop workers keep dying") as excinfo:
        rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert rb._dead_worker_discards_in_step[PARTITION] == allowance + 1
    assert f"{allowance + 1} prompts of step 1" in str(excinfo.value) and f"allowance of {allowance}" in str(excinfo.value)
    # cap re-issues then a failure + refill of the first prompt (= the allowance), then the replacement's first discard
    assert sim.counts["reissues"] == reissues and sim.counts["dispatches"] == dispatches, sim.counts
    assert rb._ttl_discards_in_step[PARTITION] == 0, "none of it is charged to the TTL budget"


def test_r3_b2_manager_counters_keep_rollout_keys_and_the_drain_rekeys_per_prefix():
    trainer = SimpleNamespace(agent_loop_manager=SimpleNamespace(recovery_metrics=Counter()))
    consume = PPOTrainer._consume_recovery_metrics.__get__(trainer)
    counters = trainer.agent_loop_manager.recovery_metrics
    counters.update({"rollout/dead_workers_relaunched": 1, "rollout/prompts_orphaned_at_dispatch": 4})
    assert consume() == {"rollout/dead_workers_relaunched": 1, "rollout/prompts_orphaned_at_dispatch": 4}
    assert counters == Counter()
    counters.update({"rollout/dead_workers_relaunched": 2, "custom_count": 1})
    assert consume(prefix="validation") == {"validation/dead_workers_relaunched": 2, "validation/custom_count": 1}
    assert consume() == {}


def test_r3_b3_wandb_generations_table_survives_shrinking_and_growing_sample_counts():
    assert wandb.run is None
    logger = ValidationGenerationsLogger()
    logger._log_generations_to_wandb([("p1", "r1", 1.0), ("p2", "r2", 0.0)], 1, wandb)
    logger._log_generations_to_wandb([], 2, wandb)  # all-lost validation: no row
    logger._log_generations_to_wandb([("p3", "r3", 1.0)], 3, wandb)  # shrink: padded
    logger._log_generations_to_wandb([("a", "b", 1.0)] * 3, 4, wandb)  # grow: earlier rows padded
    table = logger.validation_table
    assert len(table.columns) == 1 + 3 * 3 and [row[0] for row in table.data] == [1, 3, 4]
    assert all(len(row) == len(table.columns) for row in table.data)


def test_r3_b4_live_worker_rowless_batch_is_still_counted_next_to_a_dead_worker_batch(monkeypatch):
    class Mixed(ScheduledWorkers):
        def generate_sequences(self, batch):
            if not self.schedule:  # second batch: a live worker whose sessions all fail without rows
                LiveWorkersWhoseSessionsAllFail.generate_sequences(self, batch)
                return
            super().generate_sequences(batch)

    memory, workers, trainer = make_trainer(monkeypatch, Mixed, batches=(["a"], ["b", "c"]), schedule=[0])
    probe = {"calls": 0}

    def liveness(uids):  # the first batch's worker is dead, the second batch's worker answers
        probe["calls"] += 1
        return set(uids) if probe["calls"] == 1 else set()

    trainer.replay_buffer.liveness_fn = liveness
    metrics = trainer._validate()
    assert metrics["validation/inflight_dead_worker_failed"] == 1
    assert metrics.get("validation/prompts_without_rows") == 3, f"the live worker's 2 row-less prompts vanished: {metrics}"
