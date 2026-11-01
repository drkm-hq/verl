# P3 adversarial review — `contrib/p3-v1-never-hang` (11 commits ec0ebe4a..5e6f6b74 on main @ 2ab35292)

Reviewer: Claude (Fable 5.1), 2026-10-29. Read-only on the worktree; every finding is cited file:line on the
branch. Harness: `.venv-cpu` (Python 3.12.14, ray 2.55.1, TransferQueue 0.1.11, torch 2.13.0 CPU),
`PYTHONPATH=wt/p3`, `pytest -c pytest-cpu.ini --asyncio-mode=auto`. Reviewer tests live in
`packages/p3/review-tests/` (not in the worktree).

## 1. Verdict: NEEDS-FIXES

The design (fresh-uid re-issue, retired-attempt sweep, re-creation instead of `max_restarts`, probe with
timeout-is-alive) is sound and the dead-worker paths work as claimed (R3, R6, R7 pass; author's 18 tests
and both regression groups pass on my re-run). It must not merge as is because of one data-loss defect and
one livelock in the TTL path, both in `_recover_stuck_inflight`:

- **D1** a TTL-expired group whose slow-but-alive worker finishes it in the window between the buffer's
  snapshot and the trainer's re-issue is retired without being re-issued, and the next poll's sweep deletes
  the finished group (R1 fails: the step then blocks on a batch that can never fill).
- **D3** a TTL below one generation (or a re-issue routed back to a wedged worker 0) re-issues forever
  with no cap, no abort of the old attempt and no failure path (R4, R8 fail): "never hang" becomes a
  livelock with the GPU burning. On main the same configuration simply trains.

Plus D2 (the new counters are averaged, not summed, under `parameter_sync_step > 1`), D4 (the owner map
grows without bound on the DEFAULT path) and D5 (validation still hangs on a dead worker, and the docs say
otherwise). All five fixes are small (about 40 lines total); none changes the design.

## 2. Defects

| id | sev | where | failing input | fix |
|---|---|---|---|---|
| D1 | HIGH | `replay_buffer.py:364-365` with `trainer_base.py:967-973` | TTL-expired `running` prompt; its worker writes `finished` after the buffer's `kv_list` snapshot but before `_reissue_inflight_prompts`'s own `kv_list` (the window includes the probe's `ray.wait` of up to 2 s and the sweep's `kv_clear`). `reissue_fn` correctly re-issues nothing, `_retired_uids \|= stuck` retires it anyway, `_sweep_retired_attempts` (`:318-332`) deletes the prompt key and every trajectory next poll. Group lost: not trained, not re-issued, dataloader item consumed; in sync mode the batch can never fill. R1. | Retire only what was actually re-issued: have `_reissue_inflight_prompts` return the list of old uids it retired (keep the `int` for `uids=None`, or add `_reissue_inflight_uids(...) -> list[str]` and wrap), then `self._retired_uids[partition_id] |= set(reissued_uids)` and count `len(reissued_uids)`. |
| D2 | MED | `verl/trainer/ppo/v1/utils.py:41-56`, `:103` | async mode with `parameter_sync_step=2`: iteration 1 re-issues 4, iteration 2 re-issues 0 -> `training/inflight_reissued` logged as 2 (sample-weighted mean). Same for `inflight_dead_worker_reissued`. R2. `inflight_expired_age_max` is fine ("max" keyword). | Add `training/inflight_reissued`, `training/inflight_dead_worker_reissued` (and the `validation/` twins) to the `"sum"` list; extend `test_metrics_aggregator_on_cpu.py`. |
| D3 | HIGH | `replay_buffer.py:334-385` (no cap, no failure path); `agent_loop_tq.py:277-280` (chunks from index 0) | (a) every generation takes 1.2 s, `inflight_ttl_s=0.4`: each attempt expires before finishing, is retired and re-issued from scratch; 8 re-issues in 5 s, `_retired_uids` grows, the step never completes, the old attempts keep running (R8). (b) 2 workers, worker 0 wedged (alive, silent): the 1-row re-issue batch chunks to index 0 every TTL, forever (R4). Real-world trigger for (a): any TTL below the step's full generation phase, since `running` is stamped before the engine queue wait (`agent_loop_tq.py:127`); the docs example 600 s is below typical long-CoT steps. | (1) Carry `attempt` in the prompt tag (`_pending_prompt_tag(global_steps, attempt=1)`; re-issue copies `attempt+1`); when `attempt > max_inflight_reissues` (new ctor kwarg, default 2, read from `trainer.v1.sampler`), write `failure` on the prompt instead of re-issuing so the existing failure/padding/refill paths finish the step, and log once per prompt. (2) Rotate the dispatch start index in `generate_sequences` (as #8203 does with `_next_worker`), or dispatch a re-issue to `(owner + 1) % num_workers`. (3) Docs: the TTL is dispatch-to-terminal including queueing; set it above `timing_s/gen` max of a healthy run, not "above the longest generation". The 4x warning (`:376-383`) should key on the per-prompt attempt count, not a cumulative total. |
| D4 | MED | `agent_loop_tq.py:287-288` (record), `:336` (only prune) | `relaunch_dead_workers=False` (default): `_prompt_owner` gains one entry per dispatched prompt for the whole run and is never pruned because `reap_dead_workers` is wired only with the knob on. 12 entries after 3 batches of 4 with everything terminal and cleared (R5); ~100 B per prompt, unbounded. Default path is not byte-identical in memory. | Record owners only `if self.relaunch_dead_workers`; also prune in `generate_sequences` against uids being retired, or cap by dropping uids whose prompt key is gone. |
| D5 | MED | `replay_buffer.py:340` (`partition_id == "val"` early return) + docs line "a worker that dies during validation is relaunched at its next dispatch" | both knobs on; a worker dies while a val batch is in flight: `_validate`'s `sample("val")` polls forever, no probe runs, the worker is not relaunched (R9: `dead_workers_relaunched == 0`, sampler alive after 4 s). | Run the probe for `val` too and mark the dead worker's orphaned val prompts `failure` (tag-only put) rather than re-issuing: `_validate` already tolerates failure groups via `expected_acc_counts`. Fix the docs sentence either way. |
| D6 | LOW | `agent_loop_tq.py:292-299` | flag off, dead worker at dispatch: `RuntimeError` instead of `RayActorError`. Default-path exception type change, not stated in the PR body. | Keep the type (`logger.error(msg); raise`) or state the change in the body. |
| D7 | LOW | `replay_buffer.py:226`, `:365` | `_retired_uids` is never pruned (one uid per re-issue, harmless) and is process memory only: a retired attempt's late write made before any sweep is checkpointed and restored as a non-prompt key nobody sweeps (R3 documents it). | On restart-time re-issue (`uids=None`) drop non-prompt keys whose uid has no prompt key; prune `_retired_uids` once swept and older than 2x TTL. |
| D8 | LOW-MED | `agent_loop_tq.py:303-313`, `:338-346` | replacement worker's `__init__` fails repeatedly: relaunch every probe period forever, prompts re-issued each time; on the dispatch path (`:300`) a bare `RayActorError` escapes. | `max_worker_relaunches` per index (e.g. 5) then `RuntimeError`; log each relaunch with the cause. |
| D9 | LOW | `agent_loop_tq.py:333` | `ActorUnavailableError` (subclass of `RayActorError`, transient by Ray's definition) kills and replaces a healthy worker and re-issues everything it owned. | Count only `ActorDiedError`, or require N consecutive unavailable probes. |
| D10 | LOW | `replay_buffer.py:190` | `isinstance(x, int \| float)` accepts `True`. | `and not isinstance(x, bool)`. |
| D11 | LOW | `test_inflight_recovery_on_cpu.py:43-47` | module fixture does `tq.close()` without `ray.shutdown()` (same as the sibling files); Ray lives until process exit when the file runs alone. | Mirror the dead-worker fixture. |
| D12 | LOW | `replay_buffer.py:577` in sync DAPO drain | a prompt expiring during `draining` is regenerated although it will be discarded as surplus. | Skip re-issue when `draining` (clear instead). |

## 4. Merge-process gaps and split proposal

Compliance, checked against verl's PR template and `tests/special_sanity/check_pr_title.py`:

| item | state |
|---|---|
| Title | valid (`check_pr_title.py`: modules trainer, rollout, cfg; type feat). No `[BREAKING]` needed, but see the exception-type change under defects. |
| Similar-PR links | 5 search links plus #8168 / #8380 / #8203 / #7910 / #7045 / #6583 named; the ownership boundary paragraph names #8168 and RFC #8380 explicitly. OK. |
| Test section | describes the 18 tests and the regression counts, but the pytest command itself is only in `packages/p3/tests.log`: paste the two commands into the PR body (maintainers run them). |
| "Chaos proof" table | both result cells are `TBD`. A PR body must not ship with TBD cells: either run the Modal chaos run before posting or cut the section to a one-line "GPU chaos run: pending, will attach". |
| AI statement | present ("AI assistance: ..."), plus the `🤖 Generated with Claude Code` footer. OK. |
| pre-commit | ruff / ruff-format / mypy re-run by the reviewer on the 7 changed `.py` files: see section 6. `scripts/generate_trainer_config.sh` -> "All good" (re-run). `Last updated` bumped. |
| Docs | section exists; three statements are wrong or misleading (TTL guidance, validation claim, metric aggregation): see defects D3, D5, D2. |
| CODEOWNERS | `verl/experimental/agent_loop/agent_loop.py` (wuxibin89 / ArronHZG) is touched only for the 12-line `_launch_worker` factor-out. Re-implementing `_launch_worker` inside `AgentLoopManagerTQ` (duplicating the `options(...)` block) would keep the whole diff under `verl/trainer/ppo/v1/` + config, one owner fewer. Worth it for a first-time author. |
| Size | 322 product `.py` lines (+40 deleted) over 5 product files, 357 with docs. Over the ~300 guideline and two concerns in one PR. |

Split proposal (improves merge odds; each half is additive and independently testable):

- **[1/2] `[trainer, cfg] feat: in-flight prompt TTL with mid-run re-issue (V1)`** — `replay_buffer.py`
  (ctor kwargs `inflight_ttl_s` / `reissue_fn`, `_expired_inflight_keys`, `_sweep_retired_attempts`,
  `_recover_stuck_inflight`, the two loop hooks), `trainer_base.py` (`_pending_prompt_tag`, `_prompt_fields`,
  persist-when-enabled, fresh-uid `_reissue_inflight_prompts(uids=...)`, wiring), `ppo_trainer.yaml` +
  generated yamls, the `MetricsAggregator` sum-list entries, docs, `test_inflight_recovery_on_cpu.py`.
  About 190 product lines. A dead worker still raises `RayActorError` at the re-issue dispatch (loud, as on
  main); the TTL alone already turns the sync-mode hang into a crash-or-recover.
- **[2/2] `[rollout, trainer, cfg] feat: relaunch dead agent loop workers and re-issue their prompts (V1)`** —
  `agent_loop_tq.py` (`ping`, per-dispatch `ray.get`, `_recover_dispatch`, `_relaunch_worker`,
  `reap_dead_workers`, owner map, counters), `_launch_worker` (in the TQ manager, see CODEOWNERS above),
  `rollout.yaml` + `AgentLoopConfig`, the `liveness_fn` kwarg and `_reap_dead_workers` wiring,
  `_consume_recovery_metrics`, `test_dead_worker_recovery_on_cpu.py`. About 140 lines.

Both halves should be announced in #8203's thread with the rebase map (REBASE-PLAN.md is realistic: #8203's
`AgentLoopManagerTQ.generate_sequences` rotates with `self._next_worker` and `(start + i) % num_workers`
(its diff lines 4519-4532), `_persist_prompt_fields` is a class attribute of `StreamingDataloader`, and
`tq_schema.prompt_tag(status, global_steps)` exists; the table's targets are correct). #8203 itself is
MERGEABLE but REVIEW_REQUIRED with only a coderabbit review, last commit 2026-10-23, and 4 red CI jobs
(e2e megatron-qwen3, sgl, two Ascend jobs): it is not landing this week, so landing [1/2] first is the
better bet than waiting.

## 3. Semantic risks that need the Modal chaos run (or a real engine)

- **SR1 TTL measures dispatch-to-terminal, queueing included.** `running` is written before the engine queue
  wait (`agent_loop_tq.py:127`), and in sync mode every prompt of the batch is dispatched at once, so the last
  groups finish at the END of the generation phase. The docs example `inflight_ttl_s=600` and the chaos
  preset's `120` are below typical long-CoT step generation times: healthy runs would re-issue their tail
  (R8 shows the degenerate case: TTL below one generation never completes). The chaos run must include a
  healthy control with the TTL on, asserting `training/inflight_reissued == 0` for all 20 steps, and report
  `timing_s/gen` max as the floor for any TTL recommendation.
- **SR2 Retired attempts are never aborted.** Re-issue clears rows and retires the uid but the worker's
  background tasks for the retired prompt keep running (`agent_loop_tq.py:118-122`): engine requests, tool
  calls and reward computation continue, their output is swept. For a false positive this doubles GPU work;
  for a genuinely hung engine RPC (#6583) every re-issue leaks one hung task (and its engine slot) per
  attempt. Only an engine run shows this (server in-flight counts, `agent_loop/num_preempted`, KV usage).
  A worker-side `cancel(uids)` that cancels the asyncio tasks and aborts the engine requests is the proper
  follow-up; the PR body should name it.
- **SR3 `ActorUnavailableError` counts as dead** (`agent_loop_tq.py:333`, MRO verified on ray 2.55.1).
  Ray raises it for transient conditions (raylet/GCS hiccups, a node briefly unreachable). The reaper then
  `ray.kill`s a healthy worker and re-issues everything it owned. Chaos run: `SIGSTOP` a worker for 30 s,
  restart its raylet, and record which error class the probe sees; consider `ActorDiedError` only, or three
  consecutive unavailable probes.
- **SR4 Unbounded relaunch.** A worker whose `__init__` fails on the replacement node (tokenizer path,
  tool config, OOM at construction) is re-created every probe period forever on the reaper path
  (`agent_loop_tq.py:338-346`) while its prompts are re-issued each time; the dispatch path raises a bare
  `RayActorError` from `_recover_dispatch` (`:300`) when the replacement dies before serving. Make one
  construction fail in the chaos run and watch `rollout/dead_workers_relaunched`.
- **SR5 Relaunch latency.** `AgentLoopWorker.__init__` loads tokenizer/processor/tools (tens of seconds);
  `_recover_dispatch` waits for it inside the sampling loop (`ray.get` at `:300`). Measure the step-time
  spike; the PR body's "<= probe period + one generation" claim omits the construction time.
- **SR6 Sync-mode persisted prompt rows.** With a knob on, sync mode persists every prompt's tensor
  columns in TransferQueue (`trainer_base.py:1557-1565`; multi-modal rows included). Bounded by one batch
  and cleared at `_materialize_batch`, but measure the TQ storage delta on the preset.
- **SR7 Mid-step relaunch during `update_weights`** (separate_async): a replacement worker created while
  the checkpoint engine is in `abort_replicas` / `update_weights` sees the same `llm_client` and load
  balancer; nothing here fences it, by design (#8168's area). Confirm a kill during a sync step does not
  deadlock the sync.

## 5. Adversarial tests and results

Location: `packages/p3/review-tests/` (`conftest.py`, `_review_common.py`, `test_review_buffer_on_cpu.py`,
`test_review_workers_a_on_cpu.py`, `test_review_workers_b_on_cpu.py`); full output in
`packages/p3/review-tests/results.log`. Run:

    cd wt/p3 && PYTHONPATH=$PWD TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
      ../../.venv-cpu/bin/python -m pytest -c ../../pytest-cpu.ini --rootdir ../../packages/p3/review-tests \
      -q --asyncio-mode=auto -p no:cacheprovider ../../packages/p3/review-tests

Real `ReplayBuffer` / `ReplayBufferAsync`, real `AgentLoopManagerTQ` with a stub worker actor
(`ReviewWorker`: modes finish / stuck / slow / finish_first_only; writes `train` or `val` by the batch's
`validate` flag, like the real worker), the real `PPOTrainer` submit / re-issue / reap methods, one Ray
local cluster per session with `ray.shutdown()` in the fixture; `pgrep raylet|gcs_server` empty after both
runs. Tests that demonstrate a defect are written to FAIL on the branch: the assertion states the expected
behaviour, the message states what happened.

| test | scenario | result on the branch | defect |
|---|---|---|---|
| R1 `test_terminal_flip_in_the_reissue_window_loses_the_group` | TTL-expired prompt whose worker finishes it between the buffer snapshot and the trainer's re-issue | FAIL: `finished group lost: retired={uid}, keys left in TQ=set()`; the sampler never returns | D1 |
| R2 `test_inflight_reissued_is_averaged_not_summed_across_iterations` | `MetricsAggregator` over two `_step_once` iterations (4 then 0) | FAIL: `counter was averaged: 2.0` (control: `rollout_failure/evicted_samples` summed to 3) | D2 |
| R3 `test_reissued_prompt_survives_checkpoint_and_retired_late_write_persists_as_garbage` | async re-issue, `tq.save_checkpoint`, load, restart-time re-issue | PASS: one group, fresh uid, same prompt row, dispatched once; the retired attempt's late keys come back as unswept garbage | verifies; D7 |
| R4 `test_reissue_of_a_small_batch_goes_back_to_worker_0_so_a_wedged_worker_0_livelocks` | 2 workers, worker 0 alive but silent, TTL 0.5 s | FAIL: 12 re-issues in 6 s, all 13 dispatches of that prompt to worker 0, `retired=12`, step never completes | D3(b) |
| R5 `test_prompt_owner_map_grows_without_bound_on_the_default_path` | `relaunch_dead_workers=False`, 3 batches of 4, all terminal and cleared | FAIL: `default path keeps 12 owner entries forever` | D4 |
| R6 `test_dead_worker_half_terminated_chunk_reissues_only_the_inflight_prompt` | worker 1 finished 1 of its 2 prompts, then `ray.kill`; probe every 0.2 s | PASS: exactly 1 re-issued, the finished group sampled with its original row, 1 relaunch | verifies |
| R7 `test_two_workers_dying_at_once_are_both_relaunched_and_their_prompts_reissued` | 3 workers, kill 1 and 2 with 2 stuck prompts each | PASS: 4 re-issued in one probe, both replaced and pinging, 6 groups sampled | verifies |
| R8 `test_ttl_shorter_than_a_legitimate_generation_livelocks_on_healthy_workers` | both workers alive, every generation 1.2 s, TTL 0.4 s | FAIL: 22 prompts re-issued in 5 s (11 rounds), the 4x warning fired once, step never completes; on main this configuration trains | D3(a) |
| R9 `test_validation_partition_still_hangs_on_a_dead_worker_with_both_knobs_on` | val batch in flight, worker 1 killed, TTL 0.5 s + probe 0.2 s | FAIL: sampler alive after 4 s, `dead_workers_relaunched == 0` | D5 |

Not reproducible on CPU, reasoned only: `ActorUnavailableError` on a live worker (D9 / SR3), abort of the
retired attempt's engine requests (SR2), relaunch latency (SR5).

Author's suite re-run by me on the branch, same harness: `test_inflight_recovery_on_cpu.py` +
`test_dead_worker_recovery_on_cpu.py` 18 passed (105.7 s); `tests/trainer` 456 passed, 1 skipped
(135.3 s); `tests/experimental` 74 passed (65.6 s); counts match `packages/p3/tests.log`. pre-commit
`ruff`, `ruff-format`, `mypy`, `check-docstrings`, `check-naming-conventions`, `check-license` on the 7
changed `.py` files: Passed. `scripts/generate_trainer_config.sh`: All good. `check_pr_title.py`: valid.
Worktree `wt/p3` and the main clone are clean after the review (no edits, no commits, nothing pushed).

## 6. What I verified correct (reading, plus the tests in section 5)

- `dispatched_at` survives the worker's tag-only `running` flip (TQ merges tags; author test and R3).
- Fresh-uid re-issue prevents two attempts from merging into one group: `_materialize_batch` gathers by
  uid prefix from the poll snapshot (`replay_buffer.py:519-531`), the old uid's prompt key and rows are
  cleared before the new dispatch (`trainer_base.py:979-998`), and every poll of both loops sweeps whatever
  the retired uid still writes before selection (`:334-341`, hooks at `:577` and `:702`). A late write
  arriving between snapshot and materialisation is not in the snapshot, so it cannot be sampled.
- The sweep cannot delete the new attempt: `uuid4().hex` has no `_`, so `key.split("_")[0]` is exactly the
  retired uid and the new uid never matches.
- Dataloader accounting: re-issue reads rows from TransferQueue only (`trainer_base.py:977`); nothing is
  fetched from the dataloader, nothing is trained twice (old rows cleared), `index`/`__rollout_n__` columns
  travel with the persisted row so `sample_index` and per-prompt `n` are unchanged. Checkpoint save/load of
  a re-issued prompt restores exactly one group under the fresh uid and the restart-time re-issue dispatches
  it once (R3). Dumps, DAPO tables and validation key by uid per attempt and need no change.
- Group integrity: a re-issue regenerates all `n` sessions (the worker's `_run_prompt` spawns `n` tasks
  from the persisted row); partial rows of the retired attempt are cleared first.
- Re-creation: `_launch_worker` builds the same actor class with the same (config, llm_client,
  teacher_client, reward handles), soft node affinity over alive CPU nodes, and a fresh
  `agent_loop_worker_{i}_{uuid}` name (no named-actor collision). Prompts of the dead worker that already
  reached a terminal status are not regenerated (owner map intersected with the in-flight snapshot; R6).
  Two simultaneous deaths are handled in one probe (R7). The old handle is `ray.kill`ed best-effort.
  `agent_loop_workers` is not referenced anywhere else in V1 (the V0 `fully_async_rollouter` keeps its own
  list). All buffer/manager calls run on the driver thread in every V1 mode (no threads in the async
  trainers), so the list mutation in `_relaunch_worker` cannot race a dispatch.
- Flag-off path: no probe, no loop, `reissue_fn`/`liveness_fn` are `None`, `_recover_stuck_inflight`
  returns at its first line; sync submit stays tag-only. The default path still changes in three small ways
  (D4 owner-map growth, D6 exception type, the extra `dispatched_at` float in every prompt tag): the PR
  body's "behaviour unchanged" needs that footnote. Checkpoints from main (no `dispatched_at`) never expire.
- Tests: real `ReplayBuffer`/`ReplayBufferAsync`/`AgentLoopManagerTQ`/TransferQueue on Ray local with stub
  worker actors that write statuses the way the real worker does; the "TTL-off baseline" test is sound
  (bounded join, then a hand-written terminal status releases the thread). Author's 18 tests and both
  regression groups re-run by me: see section 5.
- Config: yaml + dataclass + four regenerated yamls consistent (`generate_trainer_config.sh` -> All good);
  `OmegaConf.select` keeps partial test configs working; `AgentLoopConfig.get` exists on `BaseConfig`.
- REBASE-PLAN.md checked against `gh pr diff 8203`: targets exist and the rotation note is right.

## 7. Answers to the author's open questions

1. **Probe period/timeout: env vs config.** Keep the two env constants for now (verl precedent:
   `VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS`), but make the period a `ReplayBuffer` ctor kwarg
   (`liveness_period_s=WORKER_LIVENESS_PERIOD_SECONDS`) so tests stop monkeypatching the module and a future
   config key is a one-line wiring. Do not add a third config key in this PR.
2. **TTL for `val`.** A TTL is not what validation needs; liveness is (reviewer test R9: validation blocks
   forever on a dead worker even with both knobs on, and the worker is not relaunched because no dispatch
   follows). Minimal fix: run the probe for `val` too and mark a dead worker's orphaned val prompts
   `failure` instead of re-issuing them. `_validate` already tolerates failure groups
   (`expected_acc_counts` + `acc=0` padding in `metric_utils.process_validation_metrics`), uids stay stable,
   nothing is persisted. About 15 lines, no accounting change.
3. **Before #8203 or re-target.** Land before, as the [1/2]/[2/2] split above; both halves are additive and
   the map onto `StreamingDataloader` is verified line-for-line. Announce in #8203's thread so Begunner can
   carry the hunks if #8203 wins the race.
4. **`RuntimeError` vs bare `RayActorError` with the flag off.** Keep the original type: nothing in verl
   catches `RayActorError`, but a wrapper (Ray job driver, user launcher) may, and changing the exception
   type on the default path contradicts "defaults unchanged". `logger.error(<the same message>)` then a bare
   `raise` gives the same diagnosability. If you keep `RuntimeError`, say so in the PR body as a behaviour
   change.

## 8. Fixes applied (author, 2026-10-29) — on `contrib/p3-v1-never-hang` as new commits, then split

| defect | fix | commit (p3) | where it lives in the split |
|---|---|---|---|
| D1 window race | `_recover_stuck_inflight` re-syncs and intersects with the in-flight set right before acting; `_reissue_inflight_prompts(uids=...)` returns the old uids it actually retired and the buffer retires only those (`reissued = to_reissue - registered_prompt_uids`, intersected with the returned list) | 8918ce3c | p3a |
| D2 averaged counters | the five recovery counters added to `MetricsAggregator`'s `"sum"` list; `test_metrics_aggregator_on_cpu.py` extended | 1958cf64 | p3a (TTL keys) / p3b (dead-worker keys) |
| D3(a) livelock | `attempt` in the prompt tag (`_pending_prompt_tag(global_steps, attempt)`, re-issue stamps `attempt + 1`); `trainer.v1.sampler.max_inflight_reissues` (default 3) after which the prompt becomes a `failure` group under a fresh uid (`_fail_groups` / `_fail_exhausted`), so padding / `sync_refill_failed_groups` / async failure refill finish the step; per-prompt warning on a second re-issue instead of the cumulative 4x warning; docs: TTL = dispatch-to-terminal incl. queueing, set above `timing_s/gen` max | 8918ce3c, 39f95929 | p3a |
| D3(b) worker 0 | `generate_sequences` assigns chunks from a rotating start index (`_dispatch_cursor`) | 898d4317 | p3a (rotation only) |
| D4 owner map | recorded only with `relaunch_dead_workers` on; pruned to the in-flight set on every reap; orphaned uids moved to `_orphaned_uids` until consumed | 898d4317 | p3b |
| D5 validation | the probe also runs for `val` (the buffer hands over every partition's in-flight set); orphaned val groups are marked `failure` in place, the worker is relaunched; docs corrected | 8918ce3c, 39f95929 | p3b |
| D6 exception type | flag off: `logger.error(<worker, uids, hint>)` then `raise error` (the original `RayActorError`) | 898d4317 | p3b |
| D7 checkpoint garbage | restart-time re-issue (`uids=None`) drops restored non-prompt rows whose uid has no prompt key | 8918ce3c | p3a |
| D8 unbounded relaunch | `_relaunch_streak[index]` reset by a served dispatch; beyond `MAX_WORKER_RELAUNCHES` (env, 5) `_relaunch_worker` raises `RuntimeError` | 898d4317 | p3b |
| D9 unavailable | dispatch path retries an `ActorUnavailableError` `WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD` times; the probe counts only consecutive unavailable results (`_probe_marks_dead`), `ActorDiedError` is immediate | 898d4317 | p3b |
| D10 bool TTL | `not isinstance(x, bool)` for `inflight_ttl_s` and `max_inflight_reissues` | 8918ce3c | p3a |
| D11 fixture | unit fixture calls `ray.shutdown()` | 900883df | p3a |
| D12 DAPO drain | sync `sample` passes `draining`; expired prompts are cleared and retired, not regenerated (`inflight_discarded_while_draining`) | 8918ce3c | p3a |
| §7.1 period kwarg | `ReplayBuffer(liveness_period_s=...)`, falling back to the module constant at call time | 8918ce3c | p3b |
| merge process | `_launch_worker` lives in `AgentLoopManagerTQ` (`verl/experimental/agent_loop/agent_loop.py` back to main); PR bodies carry the exact pytest commands; chaos table stays TBD; split into p3a [1/2] / p3b [2/2] | 898d4317 + branches | both |

Reviewer suite after the fixes (unmodified, run on p3 and on p3b): R1, R2, R5, R8 now pass; R6, R7 still pass;
R3 / R4 / R9 fail only at their characterization lines (134 / 47 / 104) that pin the pre-fix behaviour the
requested fixes remove (garbage swept, re-issue rotated to worker 1, probe relaunched during validation); the
intent in each docstring is covered by `test_restart_reissue_drops_rows_without_a_prompt_group`,
`test_reissue_rotates_away_from_a_wedged_worker` and
`test_validation_dead_worker_is_relaunched_and_its_val_groups_fail`.

## 9. Re-verification (reviewer, 2026-10-29, after the author's fixes and the p3a / p3b split)

Read-only, same harness. Branch heads at the start of this pass: `contrib/p3-v1-never-hang` 39f95929,
`contrib/p3a-inflight-ttl` 7f43e43e, `contrib/p3b-worker-relaunch` a2ce4020 (p3a is an ancestor of p3b).

### 9.1 Branch identity and the standalone claim

- `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` was NOT empty at the start: one test hunk,
  `tests/trainer/ppo/v1/test_inflight_recovery_on_cpu.py` (+46: `test_dispatch_rotates_the_start_worker_across_calls`
  + `_batch_of`) present on p3a/p3b and missing on the combined branch. Product code identical. The lead
  says the author is adding it to the combined branch; the check is re-run at the end of this section.
- p3a stands alone: its product diff vs main is `replay_buffer.py` (+185), `trainer_base.py` (+96),
  `agent_loop_tq.py` (+14: `_dispatch_cursor` rotation only), `utils.py` (+3 sum-list keys), the two sampler
  keys, generated yamls, docs. `git grep` on p3a for `liveness|relaunch|reap_dead|_prompt_owner|RayActorError|
  orphan|ping` in `verl/` finds nothing of this feature; `ReplayBuffer.__init__` has no `liveness_fn` /
  `liveness_period_s` (verified: p3b's diff on p3a adds them). `verl/experimental/agent_loop/agent_loop.py` is
  identical to main on both branches (`_launch_worker` lives in `AgentLoopManagerTQ`).

### 9.2 D1: is the window really closed? (read on p3b @ a2ce4020 = p3 @ 39f95929 product code)

Order in `_recover_stuck_inflight` (`replay_buffer.py:396-479`): snapshot -> sweep -> `expired` /
`dead_owned` (probe, up to 2 s) -> **re-sync and `stuck &= inflight`** -> `reissue_fn(to_reissue)` ->
**re-sync, `reissued = to_reissue - registered_prompt_uids`, `&= claimed`** -> retire `reissued`. Inside the
trainer (`trainer_base.py:970-1028`): T1 `kv_list` filters to pending/running; T2 `kv_batch_get`; T3 clears
the rows known at T1; T4 clears the old prompt keys and registers the fresh uids; T5 dispatch; returns the old
uids it retired. A terminal write from the slow-but-alive worker can land:

| when | what happens | outcome |
|---|---|---|
| before the re-sync | dropped from `stuck` | group sampled (R1 now passes) |
| between the re-sync and T1 | T1 sees it terminal: not re-issued, not in `claimed`; still registered at the final re-sync | not retired, sampled (N1b with an int-returning `reissue_fn`) |
| between T1 and T4 (rows and/or status) | T4 clears the prompt key anyway; rows written after T1 survive T3 but their uid is retired (cleared at T4, in `claimed`) | swept next poll; the fresh attempt is the only group (N1) |
| after T4 | recreates a non-prompt key under the retired uid | swept |

No window leaves a terminal group retired, and no window leaves two live attempts. The `& claimed` guard also
makes the buffer robust to a `reissue_fn` that returns a count. Verdict: closed. Cost: one extra `kv_list`
per recovery pass (only when something is stuck).

### 9.3 D3: the cap -> failure -> refill path

`attempt` rides in the prompt tag (`_pending_prompt_tag(global_steps, attempt)`, original = 1; a re-issue
stamps `attempt + 1`, `trainer_base.py:1013`). `exhausted = attempt > max_inflight_reissues`
(`replay_buffer.py:437`), so `max_inflight_reissues = k` allows k re-issues and the (k+1)-th expiry calls
`_fail_groups(fresh_uid=True)` (`:375-395`): a tag-only `failure` prompt under a fresh uid, the old attempt
cleared and retired. From there the pre-existing paths apply: async evicts failure groups and refills one
(`_terminal_eviction_reasons`, `wait_for_sampleable`), sync with `sync_refill_failed_groups` evicts and
refills one, default sync pads (the group is sampled with no rows). Consequence worth one docs sentence: a
capped prompt is **not trained in this epoch** (replaced by a different dataloader prompt, or padded); the
docs say "padding in sync mode, or replacement" which is close enough. `_fail_groups` runs after the
post-reissue re-sync, so a group that finished in the meantime is not failed. Rotation (`agent_loop_tq.py`
`_dispatch_cursor`) advances by `len(chunks)`, so a full batch leaves the mapping unchanged and a
sub-`num_workers` batch moves the start; a wedged worker gets its prompt back with probability 1/N per
re-issue, bounded by the cap. Verdict: closed, with the docs caveat above.

### 9.4 Residual findings (file:line on p3b @ a2ce4020)

| id | sev | where | finding | fix |
|---|---|---|---|---|
| R-A | LOW-MED (p3b) | `agent_loop_tq.py:346-354` `_recover_dispatch` | On `ActorUnavailableError` the chunk is re-sent to the SAME live handle up to 3 times. Ray defines this error as "state unknown, the task may have executed": when the first dispatch did run, the retry spawns a second attempt of every uid in the chunk on the same keys (`{uid}_{s}_{i}`), the two attempts overwrite each other's rows, and the second one keeps writing after the group was sampled and its prompt key cleared: rows the trainer is reading downstream can change under it, and the late status write leaves a garbage key nobody sweeps (the uid is never retired). Reproduced with a lost-reply hook (N5). | Do not retry at dispatch: treat an unavailable dispatch like a dead one (`ray.kill` the handle, relaunch, re-dispatch) or, cleaner, leave the chunk `pending`, add its uids to `_orphaned_uids` and let the buffer re-issue them under fresh uids at the next probe (the trainer path clears partial rows). Keep the 3-strike rule for the read-only probe. |
| R-B | LOW (p3b) | `agent_loop_tq.py:367-369` | The relaunch re-dispatch reuses the chunk's uids. Reachable only if the dead attempt wrote rows before the reply was lost (same class as R-A); with `ActorDiedError` the reply either arrived (no re-dispatch) or the tasks never ran. | Covered by the R-A fix (fresh uids via the buffer). |
| R-C | LOW (both) | `agent_loop_tq.py` rotation | Default-path change: after any sub-`num_workers` dispatch (DAPO / async refill), chunk i no longer goes to worker i. Behaviourally equivalent (workers are symmetric; #8203 does the same), but the PR body says "default paths unchanged". | One sentence in both PR bodies. |
| R-D | LOW (p3a) | `replay_buffer.py` val exclusion | In TTL-only mode a dead worker during validation still blocks `_validate` (no probe in p3a). p3a's docs say "the TTL does not apply to the val partition" and that relaunch is a separate change, which is honest; the [1/2] PR body should state "validation can still hang on a dead worker until [2/2]". | PR-body sentence; or apply the TTL to `val` as fail-in-place (`_fail_groups(fresh_uid=False)` already exists). |
| R-E | LOW (p3b) | `agent_loop_tq.py:349` | The dispatch-path retry sleeps 3 x 1 s inside `prepare_step` / the sampling loop. Fine, but goes away with R-A. | - |
| R-F | NIT (both) | `_fail_groups(fresh_uid=True)` | The fresh failure prompt has no `dispatched_at`; harmless (never in flight). | - |

### 9.5 Runs (reviewer, same harness; logs in `packages/p3/review-tests/results-*.log` and the session temporary directory)

Branch heads at the end of this pass: combined `contrib/p3-v1-never-hang` 2f7d9c9b (gained the rotation test
as a new commit), p3a 7f43e43e (unchanged), p3b e43e784c (rebuilt on p3a; tree byte-identical to the a2ce4020
I read). `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` is now empty. Product code did not
move during the pass. One earlier `tests/trainer` run in `wt/p3b` printed nothing: the author's process
cleanup killed my pytest (pids 17741/17749, disclosed by the author); it was re-run and is reported below.

| run | p3a 7f43e43e | p3b e43e784c |
|---|---|---|
| author's `tests/trainer` | 455 passed, 1 skipped (235 s) | 468 passed, 1 skipped (297 s, re-run) |
| author's `tests/experimental` | 74 passed | 74 passed |
| original review suite R1-R9, unmodified | not applicable by construction: every test builds `ReplayBuffer(liveness_fn=...)` / binds `_reap_dead_workers`, which p3a does not have (TypeError / AttributeError, 9 "failed"); R2 fails on p3a only at the p3b-only key `training/inflight_dead_worker_reissued` | 6 passed (R1, R2, R5, R6, R7, R8), 3 failed: R3 at line 134, R4 at line 47, R9 at line 104 |
| new suite N1, N1b, N7 (`test_review2_buffer_on_cpu.py`) | 3 passed | 3 passed |
| new suite N2, N2b, N2c (`test_review2_cap_on_cpu.py`) | 3 passed | 3 passed |
| new suite N3, N4, N5 (`test_review2_workers_on_cpu.py`) | 3 skipped (no relaunch on p3a) | N3 passed; N4 passed (after fixing my own assertion, see below); **N5 failed** |

Judgement of the three R-suite failures on p3b (legitimate behaviour change vs hidden regression):

- **R3** (`test_review_buffer_on_cpu.py:134`): fails at `assert {stuck, f"{stuck}_0_0"} <= keys`, the line that
  *documented* the checkpoint garbage; the restart-time re-issue now drops restored rows without a prompt
  group (D7). Everything before it (fresh uid survives save/load, re-issued once, same prompt row) still
  holds. Legitimate change.
- **R4** (`test_review_workers_a_on_cpu.py:47`): fails at `assert len(reissues) >= 3` with two re-issues: the
  second one rotated to worker 1 and the step completed within the 6 s window (D3b). The docstring's
  expectation ("the re-issue reaches another worker") is exactly what happened. Legitimate change.
- **R9** (`test_review_workers_b_on_cpu.py:104`): fails at `assert dead_workers_relaunched == 0` ("no probe
  runs for val"): the probe ran during validation, relaunched the worker and failed the val group in place
  (D5). Legitimate change. No regression hides behind any of the three: the pre-fix defects they pinned
  (R1, R2, R5, R8) now pass and the two verification tests (R6, R7) still pass.

New adversarial cases (all Ray local, real buffer / manager / trainer methods):

| test | scenario | p3b result | meaning |
|---|---|---|---|
| N1 `test_terminal_write_inside_the_trainer_window_is_swept_and_the_fresh_attempt_is_the_only_group` | the slow worker finishes the old uid after the trainer's `kv_list` and before its clears (hooked at `kv_batch_get`) | PASS | old uid retired and swept, fresh attempt (attempt 2) is the only group, sampled once |
| N1b `test_reissue_fn_returning_an_int_still_retires_only_what_left_the_registry` | count-returning `reissue_fn`, group turns terminal in the window | PASS | nothing retired, group sampled |
| N7 `test_two_consecutive_reissues_leave_exactly_one_live_group` | attempts 1 and 2 both expire | PASS | attempt 3 is the single live group, both old uids swept, `inflight_reissued == 2` |
| N2 / N2b / N2c (`sync` + `sync_refill_failed_groups`, async, cap = 0) | cap reached | PASS | no re-issue, one fresh failure group, evicted, refill called exactly once with 1, step completes with the replacement; `inflight_reissue_exhausted == 1`, `rollout_failure/evicted_samples == 1` |
| N3 `test_relaunch_streak_cap_raises_and_a_served_dispatch_resets_it` | three probe relaunches with no dispatch in between, MAX=2; then a served dispatch | PASS | `RuntimeError` "re-created 2 times without serving a dispatch"; the served dispatch resets the streak |
| N4 `test_val_dead_worker_fails_val_groups_in_place_and_train_orphans_are_handed_over_later` | worker 1 owns one train and one val prompt and dies during validation | PASS (second run) | val group failed under its own uid, only the finished val row materialises, `validation/inflight_dead_worker_failed == 1`, worker relaunched; the train orphan is kept in `_orphaned_uids` and re-issued at the next train poll (`inflight_dead_worker_reissued == 1`). My first run asserted the val status after `sample`, which clears prompt keys on materialisation: a test bug, fixed. |
| N5 `test_unavailable_retry_at_dispatch_must_not_spawn_the_same_uid_twice` | the dispatch executes, its reply comes back as `ActorUnavailableError` (hooked `ray.get` from `generate_sequences`) | **FAIL**: `a uid was spawned twice on a live worker: {uid0: 2, uid1: 1}` | R-A confirmed: `_recover_dispatch` re-sends the chunk to the live handle; two attempts of one uid on the same keys |

Not run: GPU / Modal. Ray hygiene: `ray.shutdown()` in the session fixture; `pgrep raylet|gcs_server` empty
at the end; the two processes seen mid-pass (pid 18031's cluster) belonged to another session and were left
alone. Worktrees `wt/p3`, `wt/p3a`, `wt/p3b` and the main clone clean; nothing pushed.

### 9.6 Verdicts

- **p3a `contrib/p3a-inflight-ttl` 7f43e43e: MERGE-READY-PENDING-CHAOS-PROOF.** D1 closed (9.2, N1/N1b/N7),
  D3 closed (9.3, N2*), D2/D7/D10/D11/D12 verified, regression groups green. Residuals R-C and R-D are
  PR-body sentences; the docs should say that a capped prompt is not trained in this epoch. The healthy
  control chaos run (TTL on, `training/inflight_reissued == 0` on every step) is still the missing proof.
- **p3b `contrib/p3b-worker-relaunch` e43e784c: NEEDS-FIXES (one small fix, R-A).** D4/D5/D6/D8/D9 verified
  (N3, N4, R-suite), regression groups green, but the dispatch-path retry on `ActorUnavailableError`
  (`agent_loop_tq.py:346-354`) can spawn a second attempt of the same uids on a live worker (N5), which is
  the one failure mode the fresh-uid design exists to exclude, and it sits on the transient-network path the
  feature targets. Fix: no re-dispatch of the same chunk after an unavailable reply; kill + relaunch +
  re-dispatch, or park the uids in `_orphaned_uids` for the buffer's fresh-uid re-issue. With that, p3b is
  merge-ready pending the kill chaos run.

## 10. Re-verification fix (author, 2026-10-29): R-A on [2/2], cherry-picked here as f6366f48

See `packages/p3b/REVIEW.md` "Fix applied": no chunk is dispatched twice on any recovery path; actor-error
dispatches park their prompts for the fresh-uid re-issue; `ActorUnavailableError` gets one second-opinion ping
and the consecutive-failure rule; a dead worker's owned prompts are parked at dispatch time too. Combined branch
f6366f48 (17 commits, no rewrite): `tests/trainer` 470 passed / 1 skipped, `tests/experimental` 74,
`test_review2_*` 9/9 (`logs/review2-after-ra.log`). p3a residual wording (R-C, R-D, capped-prompt clause) applied
to `packages/p3a/PR-BODY.md`; the docs clause is also in the [2/2] docs commit.

## 11. Re-check of the R-A fix (reviewer, 2026-10-29 02:xx)

Heads: p3b `contrib/p3b-worker-relaunch` f6846af0 (3 commits on p3a 7f43e43e, unchanged), combined
`contrib/p3-v1-never-hang` f6366f48; `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` empty.
Commit f6846af0 touches `agent_loop_tq.py` (+90/-49), `test_dead_worker_recovery_on_cpu.py`, docs.

### 11.1 The fix, read in context (`agent_loop_tq.py` on f6846af0)

- `generate_sequences`: a `RayActorError` reply goes to `_recover_dispatch` and then `continue`: no streak
  reset, no owner record for that chunk. A served dispatch resets both the relaunch and the unavailable
  streak of its index.
- `_recover_dispatch` never re-sends: flag off -> log + re-raise the original error (unchanged); flag on ->
  for `ActorUnavailableError` one second-opinion `ping` bounded by `DISPATCH_RETRY_INTERVAL_SECONDS`
  (`_confirm_dead_after_unavailable`: answered -> streak reset, "suspect"; timed out -> busy/suspect; actor
  error -> the probe's consecutive-failure rule, `ActorDiedError` final); dead -> `_orphan_worker` (park every
  owned uid, relaunch with the streak cap). In every flag-on case the chunk's uids are parked in
  `_orphaned_uids` and `rollout/prompts_orphaned_at_dispatch` counts them (`prompts_redispatched` is gone).
- `reap_dead_workers` returns the parked set pruned to the in-flight union, so the buffer re-issues parked
  prompts under fresh uids through the trainer path (old rows cleared), and a parked prompt whose attempt
  finished on its own leaves the set untouched. Both unavailable outcomes (the call executed / it did not)
  and the dead-actor case converge on exactly one live attempt per prompt; the cost is at most one probe
  period of delay (default 10 s) and, when a lost-reply attempt was in fact running past the next probe,
  one discarded generation. No path dispatches a chunk twice. Verdict on R-A/R-B: closed.

### 11.2 New cases (`packages/p3/review-tests/test_review3_dispatch_on_cpu.py`, logs `results-p3b-recheck*.log`)

| test | scenario | p3b f6846af0 |
|---|---|---|
| N8 `test_lost_reply_after_execution_slow_attempt_leaves_exactly_one_live_group` | dispatch executed on live worker 0 (3 s generation), reply lost (hooked `ray.get` from `generate_sequences`) | PASS: not re-sent, no relaunch, parked, re-issued exactly once under a fresh uid across several probes, old attempt's late row and status swept, each uid dispatched once |
| N8b `test_lost_reply_after_execution_fast_attempt_is_not_reissued` | same, attempt finishes before the first probe | PASS: parked uid leaves the set at the reap, sampled as is, `inflight_reissued == 0` |
| N11 `test_unavailable_reply_from_a_dead_actor_fails_the_second_opinion_ping_and_relaunches` | worker 0 dead, its reply reported unavailable | PASS: ping fails -> relaunched once, chunk parked, re-issued once, the dead worker never executed it |
| N9 `test_dead_worker_at_a_validation_dispatch_fails_that_val_chunk_in_place` | val chunk dispatched to a dead worker | PASS (documents the trade-off): the chunk is failed in place by the val probe, `validation/inflight_dead_worker_failed == 1` |

Also re-run on f6846af0: `test_review2_*` 9/9 (N5 now passes: no uid spawned twice), `tests/trainer`
470 passed / 1 skipped, `tests/experimental` 74 passed. Ray: none left; trees clean.

### 11.3 Residuals

| id | sev | where | finding | fix |
|---|---|---|---|---|
| R-H | LOW | validation dispatch to a dead worker (N9) | that chunk's validation groups are failed in place, never regenerated, and `expected_acc_counts` counts them as `acc = 0` for the step: a biased val score once per such event. Consistent with "validation prompts are never regenerated", but the docs bullet does not say that a dead worker at a validation *dispatch* costs the chunk. | one clause in the docs bullet; or re-dispatch the same chunk only for `ActorDiedError` during validation (a dead actor cannot write late; val rows are not trained on) |
| R-I | NIT | `_relaunch_streak` reset only by a served dispatch to that index | with rotation a relaunched index may wait several small batches for a dispatch; full batches dispatch to every worker each step, so only workloads whose every batch is smaller than `num_workers` could accumulate a false streak | note only |

### 11.4 Verdict

**p3b `contrib/p3b-worker-relaunch` f6846af0: MERGE-READY-PENDING-CHAOS-PROOF.** R-A/R-B closed by
design (no second dispatch on any path) and by test (N5, N8, N8b, N11); D4-D9 unchanged and still verified;
regression groups green. Remaining before posting: the R-H docs clause, the PR-body sentences from 9.4
(R-C), the metric rename in the PR body (`rollout/prompts_orphaned_at_dispatch`), and the Modal kill run
(one `AgentLoopWorkerTQ` SIGKILL mid-step, plus a healthy control with both knobs on).
