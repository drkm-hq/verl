> Combined round-1 record (branch contrib/p3-v1-never-hang, re-pointed to the [2/2] head on 2026-10-29). The PR bodies to submit are packages/p3a/PR-BODY.md and packages/p3b/PR-BODY.md, which carry the round-2 fixes, wording and GPU evidence.

> **Combined reference only (2026-10-29).** After the adversarial review this work is submitted as two stacked PRs:
> `packages/p3a/` **[1/2]** `[trainer, cfg] feat: in-flight prompt TTL with fresh-uid re-issue (V1)` and
> `packages/p3b/` **[2/2]** `[rollout, trainer, cfg] feat: dead agent-loop worker detection and relaunch (V1)`.
> Their bodies carry the exact pytest commands and the post-fix numbers; this body predates the fixes (D1-D12, REVIEW.md section 8).

### What does this PR do?

Makes V1 TransferQueue training steps never hang on a lost rollout writer, opt-in, in every trainer mode.

Only the `AgentLoopWorkerTQ` that runs a prompt group writes its terminal status (`agent_loop_tq.py`, `_run_prompt`). If that Ray actor dies (node loss, OOM kill, crash) or its generation silently never returns (an engine RPC that hangs, a tool that never answers; #6583 is this class), the group stays `running` forever: `ReplayBuffer.sample` (sync, the default mode) and `ReplayBufferAsync.wait_for_sampleable` poll until a terminal status appears, `_reissue_inflight_prompts` only runs at `fit()` start and returns 0 in sync mode, and the next `generate_sequences` to the dead worker raises `RayActorError` (no actor in verl sets `max_restarts`). Reproduced on CPU with the real classes before writing this (see Test).

Two knobs, both default off, behaviour unchanged unless enabled:

- `trainer.v1.sampler.inflight_ttl_s` (default `null`): prompts carry a `dispatched_at` tag; a `pending`/`running` training prompt older than the TTL is re-issued mid-run under a fresh uid (old rows discarded), and sampling keeps polling.
- `actor_rollout_ref.rollout.agent.relaunch_dead_workers` (default `false`): a dead worker is re-created with the same construction arguments and placement rule and its chunk re-dispatched; a liveness probe from the sampling loop re-issues the in-flight prompts a dead worker owned without waiting for the TTL. With the flag off a dead worker now raises a `RuntimeError` naming the worker and the stuck uids instead of a bare `RayActorError`.

Boundary with the fault-tolerance line: rollout replica health, `update_weights` fencing and load-balancer membership are #8168 (standalone replica lifecycle APIs, client-side engine-failure replay) and RFC #8380 (`verl/ft/`, process-level faults on a live node) and are not touched. Both explicitly leave CPU agent-loop workers and in-flight prompt liveness out; this PR is that half. #7910 is V0-only standby trainer workers. #8203 (streaming dataloader actor, BREAKING) moves the host module; every hunk here is additive and maps onto `StreamingDataloader` (see "Design"); happy to rebase onto it if it lands first.

### Checklist Before Starting

- [x] Search for similar PRs. Query links: https://github.com/verl-project/verl/pulls?q=is%3Apr+inflight+ttl , https://github.com/verl-project/verl/pulls?q=is%3Apr+RayActorError , https://github.com/verl-project/verl/pulls?q=is%3Apr+max_restarts , https://github.com/verl-project/verl/pulls?q=is%3Apr+%22agent+loop+worker%22 , https://github.com/verl-project/verl/issues?q=is%3Aissue+liveness+OR+%22health+check%22+rollout . Nearest: #8168, #8380, #8203, #7910, #7045 (first-rendezvous timeout), #6583 (engine RPC hang in long agent loops): none implements a TTL, mid-run re-issue or agent-loop worker relaunch.
- [x] Format the PR title as `[{modules}] {type}: {description}` (checked with `tests/special_sanity/check_pr_title.py`). No `[BREAKING]`: no CLI/config/signature changes; `_reissue_inflight_prompts` gains an optional `uids=None` argument, `ReplayBuffer` gains three optional kwargs (default `None`), custom samplers get neither.

### Test

CPU, Ray local mode, real `TransferQueue`, real `ReplayBuffer`/`ReplayBufferAsync`, real `AgentLoopManagerTQ` (stub worker actors that write statuses the way `AgentLoopWorkerTQ` does), the real `PPOTrainer` submit / re-issue / reap methods:

- `tests/trainer/ppo/v1/test_inflight_recovery_on_cpu.py` (11): `_expired_inflight_keys` (TTL, missing tag, `val` excluded), `dispatched_at` surviving the worker's `running` flip, sync submit persisting rows only with recovery on (default path unchanged), `_reissue_inflight_prompts(uids=...)` restarting under fresh uids with old rows cleared and the dataloader untouched, skipping a group that turned terminal, the sync early return kept for `uids=None`, sync `sample` and async `wait_for_sampleable` re-issuing an expired prompt with metrics, the retired-attempt sweep, the TTL-off baseline (blocks until a terminal status, unblocked by hand), and the liveness hook.
- `tests/trainer/ppo/v1/test_dead_worker_recovery_on_cpu.py` (7): a silently stuck worker + TTL (step completes, 2 re-issued), TTL off (documented: blocks; bounded wait), `ray.kill` mid-generation with TTL + relaunch (worker re-created at the re-issue dispatch, step completes), with relaunch off (`RuntimeError` naming worker 1 and the uids, cause `RayActorError`), relaunch without TTL (liveness probe re-issues the 2 orphaned prompts), a dead worker hit at dispatch time (relaunched + chunk re-dispatched, owner map pruned on reap), and the flag-off error message. Each test 2-6 s; the file 49 s with Ray startup; `ray.shutdown()` in the fixture.
- Regression on this branch: `tests/trainer` 456 passed / 1 skipped / 0 failed (main: 438 + 18 new), `tests/experimental` 74 passed. Commands and timings in the package `tests.log`.

Reproductions of the failure on `main` (same harness): `research/vet/E4/repro_replay_wait_hang.py` (sync `sample` and async `wait_for_sampleable` alive after 3-4 s with one `running` prompt; only a hand-written terminal status unblocks them) and `repro_worker_death.py` (`ActorDiedError` at the next `generate_sequences`, 0.00 s).

### Chaos proof

Modal, 2x H100 80GB in one container (`verlai/verl:uv.cu130.dev5`), V1 `separate_async` with 1 trainer GPU + 1 standalone vLLM rollout GPU, Qwen3-0.6B GRPO n=4, `train_batch_size=16`, `ppo_mini_batch_size=8`, `parameter_sync_step=2`, prompt 512 / response 512, 6 steps, `agent.num_workers=4`, `checkpoint_engine.backend=nccl`; preset `separate-async-2gpu --chaos` (`bench/README.md`): one alive `AgentLoopWorkerTQ` is SIGKILLed (`ray.kill(no_restart=True)`) right after step 2 completes. Knobs ON = `inflight_ttl_s=60`, `max_inflight_reissues=3`, `relaunch_dead_workers=True`; OFF = the defaults. Independent cross-check: `bench/CROSSCHECK.md`; the benchmark agent's tables: `packages/p3/bench/RESULTS.md`, `packages/p3a/bench/RESULTS.md`, `packages/p3b/bench/RESULTS.md` (they agree).

| arm | tree | outcome |
|---|---|---|
| `main` x3 | 2ab35292 | **crashed** (rc=1) 3 of 3: the next dispatch hits the dead handle, `ActorDiedError` in `AgentLoopManagerTQ.generate_sequences` ~7 s after the kill; the 4 groups the dead worker owned are lost |
| combined branch, knobs OFF | 5e6f6b74 | **crashed** (rc=1, same path: defaults unchanged) |
| combined branch, knobs ON x2 | 5e6f6b74 | **completed** 2 of 2 (rc=0, 6/6 steps): step 5 13.9 s / 24.5 s vs 4.2 s / 7.4 s steady state with `rollout/dead_workers_relaunched=1` and the 4 orphaned groups recovered; `training/inflight_reissued=0` (the dead-worker path claimed them before the 60 s TTL) |
| [1/2] p3a, TTL only | 7f43e43e | **crashed** (rc=1, expected: every batch dispatch touches every worker handle and the dead one raises ~7 s after the kill, before any TTL can fire; the TTL half alone cannot survive the loss of a worker, see `packages/p3a/PR-BODY.md`) |
| [2/2] p3b, knobs OFF | e43e784c | **crashed** (rc=1, same `ActorDiedError` path) |
| [2/2] p3b, knobs ON x2 | e43e784c | **completed** 2 of 2 (rc=0, 6/6): step 5 13.4 s / 14.3 s with `dead_workers_relaunched=1`, 4 groups recovered |
| [2/2] p3b, knobs ON, FINAL | f6846af0 | **completed** (rc=0, 6/6, wall 273 s): step 5 6.1 s (`dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`), step 6 10.8 s (`training/inflight_reissued=8` = `inflight_dead_worker_reissued=8`: both parked sets re-issued under fresh uids) |

Score: `main` 0/3 survive the kill, knobs OFF 0/2, TTL only 0/1, TTL + relaunch 5/5 (pre-split x2, p3b x2, final x1). The recovery costs one slow step (the 10 s liveness-probe period plus the worker relaunch, then one fresh-uid re-issue on the final tree) and nothing afterwards; the batch size stays 16 and no sample is dropped (`off_policy/trajectory_staleness/mean` dips to 0.75 / 0.5 on steps 5-6 as the recovered groups arrive a version later).

No-chaos arms (knobs ON, no kill): step 3.8-4.1 s on the branch vs 3.8-4.0 s on `main` (mean of steps 2..6, two runs per arm) = no regression; the training-side sum is within the run-to-run spread (+2.5 % on the pre-split tree, inside +-3..4 %).

GPU unit suite (`gpu_unit_tests.yml` selection, 8x H100, main reference 503 / 0 / 8): pre-split combined tree `p3@5e6f6b74` 503 passed / 0 failed / 8 skipped, torchrun 2/2 rc=0; split tree `p3b@e43e784c` (`gpu-tests-p3b-1`) **503 / 0 / 8**, 33 min 20 s, torchrun 2/2 rc=0, no new failures; the final sha `f6846af0` (`gpu-tests-p3b-2`, same tree as the combined `f6366f48`) pending, listed as running in `packages/p3b/bench/RESULTS.md`.

Limits of this evidence: the Modal harness did not populate its recovery-latency fields (`step_reached`, `recovery_s`), so the recovery cost is read from per-step `timing_s/step` in `metrics.jsonl` only; detection latency for a silently vanished node and multi-node behaviour remain unmeasured; every arm is one container with a single rollout replica.

### API and Usage Example

```bash
python3 -m verl.trainer.main_ppo ... \
    trainer.use_v1=True \
    trainer.v1.sampler.inflight_ttl_s=600 \
    actor_rollout_ref.rollout.agent.relaunch_dead_workers=True
# tune the probe cadence without a config change:
#   VERL_WORKER_LIVENESS_PERIOD_SECONDS=10 VERL_WORKER_LIVENESS_PROBE_TIMEOUT_SECONDS=2
```

New metrics: `training/inflight_reissued`, `training/inflight_expired_age_max`, `training/inflight_dead_worker_reissued` (with the sampling metrics), `rollout/dead_workers_relaunched`, `rollout/prompts_orphaned_at_dispatch`, `rollout/prompts_orphaned_by_dead_workers` (once per step). A warning is logged when the TTL fires repeatedly (TTL shorter than legitimate generation).

### Design & Code Changes

- `trainer_base.py`: `_submit_batch_to_rollout` stamps `dispatched_at` in the prompt tag (TransferQueue merges tags, so it survives the worker's `running` flip) and persists the prompt rows in sync mode when recovery is on (async already did). `_reissue_inflight_prompts(partition_id, uids=None)`: `uids=None` is the unchanged restart-time path; with `uids`, only those still in flight are restarted, each under a FRESH uid with its old trajectories cleared, so a slow-not-dead attempt that writes late can never merge into the new group (keys are `{uid}_{session}_{index}`); the dataloader is not consulted. The buffer gets `reissue_fn` and, with relaunch on, `liveness_fn`; relaunch counters are drained once per step.
- `replay_buffer.py`: `_expired_inflight_keys`, `_sweep_retired_attempts` (clears whatever a retired attempt still writes, including a tag-only status put that would otherwise recreate the key), `_recover_stuck_inflight` called on every poll of sync `sample` and async `wait_for_sampleable` right after the metadata sync; the loop re-polls after a re-issue. Validation prompts are not re-issued (their rows are not persisted).
- `agent_loop_tq.py`: `AgentLoopWorkerTQ.ping`; `AgentLoopManagerTQ.generate_sequences` does one `ray.get` per dispatch and records `uid -> worker index`; `_recover_dispatch` (relaunch + re-dispatch, or raise with a clear message), `_relaunch_worker` (re-create via `AgentLoopManager._launch_worker`, `ray.kill` the old handle best-effort so a zombie cannot come back), `reap_dead_workers` (`ray.wait` on `ping` with a short timeout; only `RayActorError` counts as dead, a timeout is a busy worker). Workers are re-created rather than `max_restarts`-restarted: `__init__` is a pure function of its arguments so either works, but in-flight background tasks are lost either way, a Ray restart surfaces as transient `ActorUnavailableError` that a probe would misread, and explicit re-creation keeps detection, relaunch and metrics in one place.
- `agent_loop.py`: `_launch_worker(index, node_ids)` factored out of `_init_agent_loop_workers` (same name, affinity and arguments).
- Config: the two keys, `AgentLoopConfig.relaunch_dead_workers`, regenerated `_generated_*.yaml`. Docs: "Self-healing: in-flight TTL and worker relaunch" in `docs/advance/v1_async_trainer.md`.
- On #8203: the buffer hunks move to `StreamingDataloader._advance_request` / `poll_once` (same insertion point after `_sync_metadata_from_transfer_queue`), the stamp into `tq_schema.prompt_tag`, the re-issue into the actor's existing `_reissue_inflight_prompts`; the two callables disappear because the actor owns the manager. Manager hunks are unchanged apart from the rotated worker index.

AI assistance: the design was vetted and the code, tests and docs were written with Claude (Fable 5.1) in Claude Code, from CPU reproductions of the failure with verl's real classes; the submitter reviewed every hunk and ran the tests on the branch.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply [pre-commit checks](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md#code-linting-and-formatting): ruff, ruff-format, mypy, check-license, check-docs-time-info, check-docstrings, check-naming-conventions, validate-structure, autogen-trainer-cfg all pass on the changed files.
- [x] Add / Update [the documentation](https://github.com/verl-project/verl/tree/main/docs).
- [x] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/verl-project/verl/tree/main/.github/workflows) to cover all the code: both files are `*_on_cpu.py` under `tests/trainer/ppo/v1/`, collected by `cpu_unit_tests.yml`.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
