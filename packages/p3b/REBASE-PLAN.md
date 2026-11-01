# P3b [2/2] rebase plan onto #8203

Refreshed 2026-10-29 for the final code (1b397dfd on p3a 2bd5c040). #8203's status and the viability argument are in
`packages/p3a/REBASE-PLAN.md` (open, not draft, `REVIEW_REQUIRED`, last updated 2026-10-23, CI partly red: `sgl`,
`vllm_ascend` and four more); the same conclusion holds here: landing before it is viable, re-targeting costs about
one to two days because the trainer-side hooks disappear into the actor.

| hunk | lands on #8203 as |
|---|---|
| `AgentLoopWorkerTQ.ping` | unchanged (#8203 rewrites `_run_prompt` into `_run_session` but keeps the class and its constructor) |
| `_launch_worker` / `_init_agent_loop_workers` in the TQ manager | unchanged |
| `generate_sequences` per-dispatch `ray.get` + owner map | re-apply on #8203's rotated loop: `self._prompt_owner[uid] = (start + i) % num_workers` for chunk `i` (#8203's `_next_worker`) |
| `_recover_dispatch`, `_relaunch_worker`, `_probe_marks_dead`, `reap_dead_workers`, counters (`rollout/...` keys, re-keyed by the trainer's drain) | textual |
| `ReplayBuffer` `liveness_fn` / `liveness_period_s`, `_probe_dead_owned`, `_fail_groups(fresh_uid=False)`, dead-worker counters | `StreamingDataloader`: no callable needed, the actor owns `self.agent_loop_manager` -> call `reap_dead_workers` directly from the lead pass (`_coordinator_step` before `_refill_to_watermark`), hand it `pending|running` of every partition; validation groups failed in place in `poll_once("val", ...)`; `_fail_groups` also clears `{uid}_{session_id}` markers |
| the step budget (re-issues + exhaustion failures) applied to the expired set minus the dead-owned prompts, which never increment the TTL counter; the probes that report dead workers are counted per step as death events and bounded by the same allowance (RuntimeError beyond it); the padding hold-back skips dead-owned groups (e9129696, a3ad633f, 1b397dfd) | same lines in the actor's `_recover_stuck_inflight` |
| `reap_dead_workers(inflight_uids, global_steps)` + the manager's per-step relaunch count (64212367) | the actor calls `reap_dead_workers` from its lead pass with the request's `global_steps` |
| `PPOTrainer._reap_dead_workers`, `_consume_recovery_metrics(prefix)`, `fit()` drain | disappear: the actor's `_pending_metrics` already flows into the sample result; the manager's counters are added there, namespaced by the partition of the request (`rollout/` for train, `validation/` for val) |
| `_validate`: empty-batch guard (neutral warning, row-less counts), sampling metrics summed over batches, validation drain, prompt counts when nothing else was produced (08209a98, 3f3d2c24); `_maybe_log_val_generations` returning without samples | `_validate` survives in `trainer_base.py` under #8203 (not rewritten; only its sampling call changes to the actor RPC): the guard and the accumulation apply textually after that call; the drain becomes "the actor's `poll_once("val")` metrics carry the relaunch counters", so the separate drain disappears |
| `rollout.yaml` / `AgentLoopConfig.relaunch_dead_workers` | unchanged |
| `verl/utils/tracking.py` `_log_generations_to_wandb` tolerance (3f3d2c24) | unchanged: #8203 does not touch `tracking.py` |
| tests | `test_dead_worker_recovery_on_cpu.py` keeps its stub actor and scenarios; the sampler thread becomes `ray.get(actor.sample.remote(...))` with a timeout; `test_validate_recovery_on_cpu.py` keeps its in-memory transport and stubs the actor's `sample` instead of `ReplayBuffer.sample` |

Duplicate-work check (read-only `gh` searches, 2026-10-29: "in-flight", "ttl", "dead worker", "relaunch", "agent
loop worker", "liveness", "RayActorError"; issues: "dead worker" / "agent loop worker" / liveness / "in-flight"):
no PR implements worker liveness, relaunch or a mid-run re-issue. New since the first check: #6641 `[trainer] fix:
fail closed on incomplete agent-loop rollouts` (open, updated 2026-10-22) is adjacent (it raises on incomplete
rollouts instead of recovering them; see PR-BODY.md for the boundary) and #8407 (isolate the standalone rollout
Python runtime) is unrelated despite matching "RayActorError". #8168, #8380, #7910, #8411 as before.
