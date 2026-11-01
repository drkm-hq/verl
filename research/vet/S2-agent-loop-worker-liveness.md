# S2 — Agent-loop worker liveness: a dead `AgentLoopWorkerTQ` fails its groups instead of hanging the step

Vetter: sweeper. Clone `2ab35292`. Not on the shortlist; complements E4 (replica self-healing: GPU replicas),
which does not cover CPU agent-loop workers. Source: report 10 §5 (row "Agent-loop worker"), §7 G3/G4, §10 #3;
report 12 §3.3 ("nothing catches actor death").

## 1. Claim
In every V1 trainer mode the death of one `AgentLoopWorkerTQ` actor (OOM-kill, node loss, uncaught exception in
the actor process) leaves every prompt group that worker was generating in TransferQueue with `status: running`
forever. In the DEFAULT `sync` mode `ReplayBuffer.sample` then polls forever (the step never completes); in the
async modes the groups leak a slot (`drop`) or block (`wait`); and the next dispatch crashes the trainer with
`RayActorError`. Nothing in verl restarts, probes or fences agent-loop workers. The candidate is an opt-in
liveness probe in `AgentLoopManagerTQ` that marks a dead worker's in-flight groups `failure` (so the existing
failure/padding/refill paths finish the step), relaunches the worker, and is polled by the replay buffer.

## 2. Mechanism verification (file:line in the clone)
- Dispatch is fire-and-forget: `AgentLoopManagerTQ.generate_sequences` chunks the batch over the workers and
  `ray.get`s only the dispatch RPC (`verl/trainer/ppo/v1/agent_loop_tq.py:248-262`); the worker spawns one
  background task per prompt (`:61-110`, `asyncio.create_task` at `:103-110`).
- Terminal status is written only by the worker itself: `_run_prompt` puts `running` (`:115`), then
  `finished`/`failure` after `_settle_session_tasks` (`:49-51`, `:136-148`, `:153`). A dead actor never writes it.
- Sync sampler: `ReplayBuffer.sample` is a `while True` loop (`verl/trainer/ppo/v1/replay_buffer.py:449-503`)
  that rebuilds state from `tq.kv_list()` each poll (`_sync_metadata_from_transfer_queue`, `:201-235`), evicts
  only TERMINAL groups (`_terminal_eviction_reasons` for sync returns no stale set, `:317-334`), waits while
  `len(sampleable) < batch_size` (`:462`) and sleeps `poll_interval` (`:408-409`). A `running` group with no
  writer keeps `has_enough_samples` false forever. Async: `_stale_terminal_keys` only covers terminal keys
  (`:531-539`), so in-flight groups have no TTL either.
- Restart-only re-issue: `_reissue_inflight_prompts` returns 0 in sync mode (`trainer_base.py:913`) and is
  called once at `fit()` start, not mid-run (`:911-947`).
- Crash on the next step: `_submit_batch_to_rollout` (`trainer_base.py:1497-1512`) calls
  `generate_sequences`, whose `ray.get([...worker.generate_sequences.remote(chunk)...])` (`agent_loop_tq.py:256-261`)
  raises `RayActorError` for a dead handle, inside `prepare_step`.
- No actor in `verl/` sets `max_restarts` (grep: 0 hits); workers are created with a name and
  `NodeAffinitySchedulingStrategy(soft=True)` only (`verl/experimental/agent_loop/agent_loop.py:1239-1258`).
- Probe primitive: Ray 2.55.1 exposes `ActorHandle.__ray_ready__` (`ray/actor.py:2657` in `.venv-cpu`); a
  `ray.wait` on it returns a ref that raises `RayActorError` on `ray.get` if the actor is dead.
- CPU-testable today: `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py` drives the real `ReplayBuffer`
  against a real TransferQueue on Ray local (`tq_init` fixture), has a thread-based block/unblock test
  (`test_sample_blocks_until_enough_then_unblocks`, line 459) and the failure-padding path
  (`test_sync_failure_without_trajectories_uses_padding_path_by_default`, line 408).

## 3. Duplicate / ownership check (dumps 2026-10-28; GitHub read-only)
- `dupcheck.py --kw RayActorError --kw "worker died" --kw "worker death" --kw liveness --kw "health check"
  --kw "agent loop worker" --path verl/trainer/ppo/v1/agent_loop_tq.py --merged`: 22 open hits, none implements
  agent-loop worker liveness. #8168 (tongyx361, 19 d, CONFLICTING): `check_health/snapshot/shutdown/restart`
  for standalone vLLM replicas; its text leaves "traffic fencing, weight restoration and publication of
  replacement handles" to applications and has no agent-loop scope. #8203 (Begunner, 16 d, [BREAKING]
  streaming dispatch): body has no timeout/TTL/dead/liveness handling (grep of the PR body); it rewrites
  `agent_loop_tq.py`/`replay_buffer.py` dispatch, so it is a rebase risk, not a duplicate. #8024 (dafu-wu:
  all-empty group = `failure`) is adjacent status semantics. #7366, #8112, #8113: unrelated.
- RFC #8380 (`verl/ft/`, HaoWang113/ETOgaosion/tongyx361): scope = "process-level faults with node alive and
  topology unchanged" (trainer in-place restart, rollout engine replace); CPU-worker loss is not listed.
  Roadmap #8186 FT bullets: replica restart, trainer restart, sandbox recover. None covers this.
- Merged (2026-07-16..): no title match. Related merged: #7082 (eviction/refill of stale/DAPO/failed groups),
  #7037 (streaming dataloader + async checkpoint recovery) — both Begunner; they built the paths this reuses.
- Owners: `verl/experimental` CODEOWNERS wuxibin89/ArronHZG; V1 sampler Begunner; FT tag owners above.
  Ask-first on #8380 (never posted by this campaign; needs the submitter).

## 4. Minimum valuable PR (opt-in, default off)
- `verl/trainer/ppo/v1/agent_loop_tq.py` (`AgentLoopManagerTQ`, ~+70): record `self._owner[uid] = worker_idx`
  in `generate_sequences` (the chunk→worker mapping is local, `:256-261`); new `reap_dead_workers(timeout_s)`:
  `ray.wait([w.__ray_ready__.remote() for w in workers], num_returns=len(workers), timeout=timeout_s)`; a ref
  that is done but raises `RayActorError` on `ray.get` marks the worker dead (timeouts count as alive); for each
  dead worker, every owned uid whose TQ tag is `pending`/`running` gets its partial trajectory keys cleared
  (the same clear `_reissue_inflight_prompts` does, `trainer_base.py:933-939`) and
  `tq.kv_put(key=uid, partition_id=..., tag={"status": "failure"})`; the worker index is relaunched through a
  `_launch_worker(i)` factored out of `_init_agent_loop_workers` (`agent_loop.py:1239-1258`); metrics
  `rollout/dead_workers`, `rollout/groups_failed_by_dead_workers`.
- `verl/trainer/ppo/v1/replay_buffer.py` (~+15): ctor kwarg `liveness_fn=None` (same injection pattern as
  `refill_fn`, `:139`, `:162`); `_wait_for_next_poll` calls it every `liveness_period_s`. The failed groups then
  take the existing paths: `sync_refill_failed_groups=True` → `refill_fn(failed_count)` (`:466-467`); default →
  padding rows (`:408` test), so the step completes.
- `verl/trainer/ppo/v1/trainer_base.py:_build_replay_buffer` (`:150-166`, ~+8): pass
  `liveness_fn=self.agent_loop_manager.reap_dead_workers` when enabled. Config (`ppo_trainer.yaml` sampler block
  `:292-318` + dataclass, +12): `trainer.v1.sampler.worker_liveness: {enable: false, period_s: 30, timeout_s: 10}`;
  regenerate `_generated_*.yaml` (autogen hook).
- Tests (~150): `tests/trainer/ppo/v1/test_worker_liveness_on_cpu.py` on Ray local (`tq.init()`): a stub
  `@ray.remote class StubWorker` whose `generate_sequences` tags its prompts `running` and never finishes; manager
  with two stubs (monkeypatched `_init_agent_loop_workers`); `ray.kill(stub0)`; `reap_dead_workers()` → stub0's
  uids are `failure`, stub1's still `running`, index 0 holds a fresh actor; `ReplayBuffer.sample(...)` with the
  `liveness_fn` returns within one poll instead of blocking (thread + join timeout, as line 459 does); with
  `sync_refill_failed_groups=True` the refill count equals the failed count.
- Docs: `docs/advance/v1_async_trainer.md` "Worker loss" (+25). Total ≈ 105 code + 150 tests + 25 docs.

## 5. Proof
- CPU: the test above (functional, no GPU).
- Modal (bench/README.md harness), 2×H100, V1 `sync` GRPO, Qwen2.5-1.5B, GSM8K, `rollout.agent.num_workers=4`,
  B=256 prompts, n=4: at step 3 SIGKILL one `ray::AgentLoopWorkerTQ` process. Baseline (flag off): `timing_s/gen`
  never completes; the replay-buffer debug line shows `running` stuck at 64 (`replay_buffer.py:411-417`); kill the
  job after 10 min. With the flag: step 3 completes within `period_s + timeout_s + poll_interval` (≤ 42 s with the
  defaults) of the kill, `rollout/groups_failed_by_dead_workers = 64` (= B / num_workers, i.e. ≤ 25 % of one
  batch, once), steps 4+ run at the pre-kill `timing_s/gen` with the relaunched worker, reward curve within seed
  noise. Metrics maintainers accept: `timing_s/gen`, `timing_s/step`, `training/rollout_failure/*`.

## 6. Adversarial answers
- A1 (why not merged): the FT line has named owners (#8380) who may want this inside `verl/ft/`; Begunner's #8203
  moves dispatch into a `StreamingDataloader` actor that would own the uid→worker map, so a PR against today's
  manager can be told to wait/rebase. Mitigation: keep the probe + fail-groups logic in the manager (kept by #8203),
  the sampler hook to one kwarg, and offer it on #8380's thread as the unowned "CPU worker" half.
- A2 (gain disappears): no existing path recovers a dead agent-loop worker, so nothing cancels it; the effect only
  shows when a worker dies (hang → completed step). It is a stability change, not a throughput one.
- A3 (correctness risk): a slow-but-alive worker must not be reaped — only `RayActorError` counts, `ray.wait`
  timeouts are "alive". Races: `_run_prompt` publishes terminal status only after all sessions settle
  (`agent_loop_tq.py:134-148`), and the writer is dead, so no late write follows the failure mark; a worker that
  died after writing `finished` leaves terminal groups untouched. Regressions would be caught by the 41 tests in
  `test_replay_buffer_on_cpu.py` (failure/refill/padding semantics) and `test_reissue_inflight_on_cpu.py`.
- A4 (who else): tongyx361 (#8168, #8112, #8113), HaoWang113 (#8380), Begunner (#8203), sunnylovestiramisu (#7910,
  standby trainer workers + NCCL re-rendezvous). None touches agent-loop workers.
- A5 (half scope): detection + fail-groups only, no relaunch (~70 lines + 1 test): turns the hang into a completed
  step; the next dispatch still fails, but loudly and checkpoint-resumable (`resume_mode: auto`).

## 7. Gates, scores, verdict
G1 pass (no open/merged PR; #8168/#8380 out of scope by their own text). G2 pass-with-ask (FT heading owned; this
half unclaimed). G3 pass (3 files + test, no new deps). G4 pass (CPU test; Modal kill drill; `timing_s/gen`).
G5 pass (DEFAULT sync mode hangs on any worker death today; the flag is opt-in but the failure is on the default path).
S1 3 (saves whole runs; edge-critical where CPU workers die routinely; no throughput) · S2 3 (#6583-class run losses,
FT roadmap heading; no issue names agent-loop workers) · S3 4 (code-level certainty; CPU-reproducible hang) ·
S4 3 (small, additive; contested files #8203/#8201; ask-first) · S5 3 (NeMo-RL/Miles have engine FT; nobody has
CPU-worker FT). Total 16 vs shortlist median ≈ 14-15.
Verdict: GO-WITH-CHANGES — ask on #8380 first; land the half-scope slice if #8203 is close; rebase onto #8203 if it
merges first. Strongest reason: it is the only unowned self-healing gap that is general (datacenter and edge).
