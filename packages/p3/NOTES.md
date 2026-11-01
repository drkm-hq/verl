# P3 author notes — in-flight prompt TTL + re-issue + dead agent-loop worker redispatch (V1)

Branch `contrib/p3-v1-never-hang` in `<repo>/wt/p3` (base main @ 2ab35292).
Venv: `.venv-cpu` with `PYTHONPATH=wt/p3` (verified: `verl.__file__` -> wt/p3; ray 2.55.1; TransferQueue 0.1.11).

## Facts established before coding (2026-10-28)
- TQ tag merge: `kv_batch_put(tags={..., dispatched_at})` then `kv_put(tag={"status": "running"})` keeps
  `dispatched_at` (probe: temporary directory/tq_probe.py). So a tag stamped at submit survives the worker's flip at
  `agent_loop_tq.py:115`. The existing code already relies on this (`global_steps` is read from running keys).
- A tag-only `kv_put` on a *cleared* key recreates it with only that tag (no `is_prompt`), i.e. a late terminal
  write from a retired attempt turns into a non-prompt garbage key. => retired attempts need a sweep.
- `kv_batch_get` on a key without fields raises `ValueError("Some fields are not ready ...")`: sync mode must
  persist prompt fields at submit when mid-run re-issue is enabled (async already persists them).
- `batch["uid"] = [new_uid, ...]` re-keys a TensorDict NonTensorStack column; `list(batch["uid"])` yields str.
- `tq.init()` calls `ray.init` when needed; `tq.close()` does NOT shut Ray down (tests must `ray.shutdown()`).
- `ActorUnavailableError` and `ActorDiedError` both subclass `ray.exceptions.RayActorError`.
- `trainer.v1.sampler` has no dataclass (plain DictConfig read via `self.config.trainer.v1.sampler`);
  `rollout.agent` has `verl/workers/config/rollout.py:AgentLoopConfig` -> the dataclass change is there only.
- Validation (`trainer_base.py:_validate`) registers `val` prompts tag-only (no fields) and keys its accounting
  (`expected_acc_counts`) by uid; re-issuing val prompts under fresh uids would break that accounting.
  => TTL/re-issue cover the `train` partition only; dead workers are still relaunched at any dispatch.

## Design decisions (file:line filled in as the hunks land)
1. `dispatched_at` lives in the prompt tag (one float next to `global_steps`), stamped in
   `_submit_batch_to_rollout`, `_reissue_inflight_prompts` and `_validate`. Cheaper than a manager-side dict:
   the tag dict is sent anyway, it survives the running flip, it is checkpointed with the TQ snapshot, and the
   replay buffer already reads every prompt tag each poll. Stamped unconditionally (a float per prompt; no
   behaviour change when the feature is off).
2. Mid-run re-issue uses a FRESH uid per re-issued prompt. The retired attempt may be slow-not-dead (engine RPC
   hang, #6583 class) and would otherwise write trajectories/terminal status onto the same keys as the new
   attempt (keys are `{uid}_{session}_{index}`), merging two attempts into one group. With a fresh uid the old
   attempt's late writes land on retired keys which the buffer sweeps. No worker-side change needed (the vet's
   `attempt`-tag alternative would have required a TQ read per trajectory write in the worker).
   Checkpoint-restore re-issue (`uids=None`) keeps uids: the old process is gone, no late writes possible.
3. Prompt fields are persisted in sync mode only when recovery is enabled (`inflight_ttl_s` set or
   `relaunch_dead_workers`), so the default sync path is byte-identical to main.
4. Dead workers are RE-CREATED, not `max_restarts`-restarted: `AgentLoopWorker.__init__` is a pure function of
   (config, clients, handles) so a Ray restart would rebuild state correctly, but (a) in-flight background
   tasks are lost either way and still need the TTL/reaper, (b) during a Ray restart RPCs raise
   `ActorUnavailableError` transiently and a probe would misread it, (c) explicit re-creation keeps detection,
   relaunch and metrics in one place and is Ray-version independent. The old handle is `ray.kill`ed best-effort
   so a zombie cannot come back and write statuses.
5. Liveness probe = `ray.wait` on `worker.ping.remote()` with a short timeout; only `RayActorError` counts as
   dead (a timeout is "busy, alive"). Probe period is a module constant tunable by env var
   (`VERL_WORKER_LIVENESS_PERIOD_SECONDS`, pattern of `VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS`) rather
   than a third config key.
6. Metrics: replay-buffer side `training/inflight_reissued`, `training/inflight_expired_age_max`,
   `training/inflight_dead_worker_reissued` (returned with the sample metrics); manager side
   `rollout/dead_workers_relaunched`, `rollout/prompts_redispatched` (Counter drained once per step).

## Hunk map (branch contrib/p3-v1-never-hang, 11 commits ec0ebe4a..5e6f6b74)
- Config: `verl/trainer/config/ppo_trainer.yaml:312` `inflight_ttl_s: null`; `rollout/rollout.yaml:262`
  `relaunch_dead_workers: false`; `verl/workers/config/rollout.py:76` dataclass field; `_generated_*.yaml`
  regenerated with `scripts/generate_trainer_config.sh` (its own up-to-date check: "All good" after commit).
- `verl/experimental/agent_loop/agent_loop.py:1245-1265` `_alive_cpu_node_ids` + `_launch_worker(index, node_ids)`
  factored out of `_init_agent_loop_workers` (same name/affinity/args; init behaviour unchanged).
- `verl/trainer/ppo/v1/agent_loop_tq.py`: `:42` probe timeout constant; `:45` `_chunk_uids`; `:69` `ping`;
  `:251-255` manager state; `:265-288` `generate_sequences` per-dispatch `ray.get` + owner map;
  `:290-301` `_recover_dispatch` (raise with worker index + uids when the flag is off);
  `:303-313` `_relaunch_worker`; `:315-347` `reap_dead_workers`.
- `verl/trainer/ppo/v1/replay_buffer.py`: `:31` probe period; `:170-172` ctor kwargs; `:223-230` state;
  `:259` `dispatched_at` read; `:303` `_expired_inflight_keys`; `:318` `_sweep_retired_attempts`;
  `:334` `_recover_stuck_inflight`; hooks at `:577` (sync `sample`) and `:702` (`wait_for_sampleable`).
- `verl/trainer/ppo/v1/trainer_base.py`: `:115` `_prompt_fields`; `:120` `_pending_prompt_tag` (module level
  so the existing stub-bound tests keep working: the first version was a method and broke 5 tests);
  `:176-196` wiring in `_build_replay_buffer` (`OmegaConf.select` for the agent key: partial configs in
  `test_trainer_base_on_cpu.py` have no `actor_rollout_ref`); `:514` recovery metrics drained in `fit()`;
  `:707` `_consume_recovery_metrics`; `:716` `_reap_dead_workers`; `:950-1008` `_reissue_inflight_prompts`;
  `:1129` validation stamp; `:1554-1570` `_submit_batch_to_rollout`.
- Docs: `docs/advance/v1_async_trainer.md:277` "Self-healing: in-flight TTL and worker relaunch";
  "Last updated" bumped to 10/28/2026 (check-docs-time-info hook).
- Tests: `tests/trainer/ppo/v1/test_inflight_recovery_on_cpu.py` (11 unit tests, real TQ) and
  `tests/trainer/ppo/v1/test_dead_worker_recovery_on_cpu.py` (7 chaos tests, Ray local, stub worker actors).

## Deviations from the brief (with reasons)
- Metric names: `training/inflight_reissued` and `training/inflight_expired_age_max` as asked, plus
  `training/inflight_dead_worker_reissued` and manager counters `rollout/dead_workers_relaunched`,
  `rollout/prompts_redispatched`, `rollout/prompts_orphaned_by_dead_workers`.
- No `max_restarts` (decision 4). No `ray.wait` probe at dispatch time: a dispatch RPC to a dead actor already
  raises immediately, a probe would only add latency; the probe lives in the sampling poll (reaper).
- Validation partition excluded from TTL/re-issue (fact 7 above); dead workers are still relaunched at the
  next dispatch, and the probe runs only while train sampling polls.
- The liveness period/timeout are env-tunable module constants, not config keys (two keys as asked).
- The "one-time warning when the TTL fires repeatedly" fires after the cumulative re-issue count reaches
  4x the size of the current stuck set (`replay_buffer.py:_recover_stuck_inflight`).
- `_submit_batch_to_rollout` persists prompt rows in sync mode only when recovery is enabled; the default
  sync path still registers tag-only (asserted by `test_sync_submit_persists_rows_and_stamps_dispatched_at_only_when_recovery_is_enabled`).
  `dispatched_at` itself is stamped unconditionally (one float per prompt tag).

## Review round (2026-10-29): fixes D1-D12 applied, branch split
- Combined reference: `contrib/p3-v1-never-hang` @ 39f95929, 15 commits (10 original + 5 fix commits
  898d4317 8918ce3c 1958cf64 900883df 39f95929); product +519/-25 incl. docstrings; tests 29 new.
  Mapping of every defect to its commit and its half: REVIEW.md section 8. Numbers: tests.log (after-fixes block).
- Split (as recommended): packages/p3a/ = [1/2] `contrib/p3a-inflight-ttl` (TTL, cap, rotation, D1 D2 D3 D7
  D10 D11 D12), packages/p3b/ = [2/2] `contrib/p3b-worker-relaunch` stacked on p3a (probe, relaunch,
  validation, D4 D5 D6 D8 D9). p3b's tree = this branch's tree + p3a's rotation unit test. Each package has
  PR-TITLE, PR-BODY ([1/2]/[2/2]), branch.txt, diff.patch, tests.log, CHECKLIST.md, NOTES.md, REBASE-PLAN.md.
- Reviewer suite (unmodified) after the fixes: 6/9; R3 / R4 / R9 fail only at characterization lines that
  pin the pre-fix behaviour (REVIEW.md section 8, tests.log).
- `verl/experimental/agent_loop/agent_loop.py` restored to main; `_launch_worker` moved into the TQ manager.

- Addendum: rotation unit test added as commit 2f7d9c9b on the combined branch too, so p3b (e43e784c) == this tree.

- R-A (2026-10-29): dispatch recovery never re-sends a chunk; parked uids flow through the fresh-uid re-issue; second-opinion ping on unavailable; commit f6846af0 (p3b) / f6366f48 (combined). `rollout/prompts_redispatched` -> `rollout/prompts_orphaned_at_dispatch`.

- R-H docs clause (2026-10-29): commit d0da57de on p3b, cherry-picked as 97f76904 on the combined branch (dead worker at a validation dispatch: chunk never re-sent, val groups failed in place, acc = 0, worker re-created).

## Review round 2 (2026-10-29)
The combined branch contrib/p3-v1-never-hang was re-pointed from 97f76904 to 3f3d2c24 (= contrib/p3b-worker-relaunch,
which stacks on contrib/p3a-inflight-ttl @ 6445f21b). The round-2 findings, fixes and verdicts live in
packages/p3a and packages/p3b (REVIEW.md section 10, REVIEW-ROUND2-VERDICT.md, review-tests/round2/). diff.patch
here is the combined diff against main at 3f3d2c24; REVIEW.md / bench/RESULTS.md in this directory are the round-1
combined records (the chaos re-run on 3f3d2c24 is recorded in packages/p3b/bench/RESULTS.md).

## Review round 3 (2026-10-29, the independent follow-up)
Combined branch re-pointed to 45690b0f (= contrib/p3b-worker-relaunch on contrib/p3a-inflight-ttl @ d834b6b3): every
TTL-induced discard is charged against the per-step budget (R2-02, the zero-cap refill livelock); the submitter's full-loop
test is in the branch. Round-3 records: packages/p3a and packages/p3b REVIEW.md section 11, REVIEW-ROUND3-VERDICT.md.
