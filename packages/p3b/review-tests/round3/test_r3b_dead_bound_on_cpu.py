"""R3b-D1..D3: the per-step dead-worker bound (64212367) and the manager's per-step relaunch cap.

D1 (expected to hold, FAILS): the bound is (cap + 1) x train_batch_size, but a dead worker's in-flight share scales
with the prompts in flight (async warm-up / DAPO prefetch, few agent-loop workers, a node loss), not with the
step's batch. ONE death that orphans more than the allowance (cap 0 and two prompts in flight; cap 3 and five) is
fully recoverable, yet the single poll that re-issues or fails them trips the bound and raises.
D2 (PASS): the hold-back never applies to a dead-owned group: padding mode, the dead worker's group is failed at
once and the step completes padded from the healthy slow group.
D3 (PASS): the manager caps re-creations of one worker index within a step regardless of served dispatches
(``_relaunches_in_step``), and a new ``global_steps`` handed to ``reap_dead_workers`` resets the count.
"""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "p3a" / "review-tests" / "round3"))
from _r3_loop_common import PARTITION, PREFIX, Sim  # noqa: E402

from verl.trainer.ppo.v1 import agent_loop_tq  # noqa: E402
from verl.trainer.ppo.v1.agent_loop_tq import MAX_WORKER_RELAUNCHES, AgentLoopManagerTQ  # noqa: E402


class OneDeath:
    """The first probe reports every in-flight prompt dead (one worker owned them all); later probes see no death."""

    def __init__(self):
        self.calls = 0

    def __call__(self, uids):
        self.calls += 1
        return set(uids) if self.calls == 1 else set()


@pytest.mark.parametrize(("cap", "inflight"), [(0, 2), (3, 5)])
def test_r3b_d1_one_death_orphaning_more_than_the_allowance_is_recovered_not_raised(monkeypatch, cap, inflight):
    sim = Sim(monkeypatch)
    rb = sim.buffer("sync", cap=cap, refill=True, batch_size=1, liveness_fn=OneDeath(), liveness_period_s=0)
    sim.dispatch(inflight)  # warm-up / prefetch: more prompts in flight than the step's batch, one worker owned them
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=1)
    assert len(batch.keys) == 1, f"a single recoverable death: {sim.counts}, {metrics}"
    assert rb._dead_worker_discards_in_step[PARTITION] == inflight


def test_r3b_d2_hold_back_never_applies_to_a_dead_owned_group(monkeypatch):
    sim = Sim(monkeypatch)
    probe = OneDeath()
    rb = sim.buffer("sync", cap=0, refill=False, batch_size=2, liveness_fn=lambda uids: probe(uids) & sim.hung, liveness_period_s=0)
    sim.dispatch(1, hung=True)  # its worker is dead: reported by the first probe
    sim.dispatch(1)  # healthy, slower than the TTL
    batch, metrics = rb.sample(global_steps=1, partition_id=PARTITION, batch_size=2)
    assert len(batch.keys) == 1 and metrics[f"{PREFIX}/inflight_reissue_exhausted"] == 1
    assert f"{PREFIX}/inflight_ttl_suspended" in metrics, "the healthy slow group is held back, the dead one failed"


def test_r3b_d3_manager_caps_relaunches_per_step_regardless_of_service_and_resets_per_step(monkeypatch):
    def stub_worker(*args, **kwargs):
        return SimpleNamespace(ping=SimpleNamespace(remote=lambda: object()))

    manager = object.__new__(AgentLoopManagerTQ)
    manager.agent_loop_workers = [stub_worker()]
    manager.relaunch_dead_workers = True
    manager.recovery_metrics = Counter()
    manager._relaunch_streak, manager._relaunches_in_step, manager._unavailable_streak = Counter(), Counter(), Counter()
    manager._relaunch_step, manager._prompt_owner, manager._orphaned_uids = None, {}, set()
    monkeypatch.setattr(AgentLoopManagerTQ, "_launch_worker", lambda self, index, node_ids=None: stub_worker())
    monkeypatch.setattr(agent_loop_tq.ray, "kill", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_loop_tq.ray, "wait", lambda refs, **kwargs: ([], []))
    manager.reap_dead_workers(set(), global_steps=1)  # the trainer's probe opens step 1
    for _ in range(MAX_WORKER_RELAUNCHES):
        manager._relaunch_worker(0, RuntimeError("boom"))
        manager._relaunch_streak[0] = 0  # the re-created worker served a dispatch before dying again
    with pytest.raises(RuntimeError, match="within training step 1"):
        manager._relaunch_worker(0, RuntimeError("boom"))
    manager.reap_dead_workers(set(), global_steps=2)  # a new step resets the per-step count
    manager._relaunch_worker(0, RuntimeError("boom"))
    assert manager.recovery_metrics["rollout/dead_workers_relaunched"] == MAX_WORKER_RELAUNCHES + 1
