# P3 self-healing fleet: Modal GPU evidence (bench/README.md harness)

Preset `separate-async-2gpu` (bench/verl_bench.py): V1 `trainer_mode=separate_async`, 1 trainer GPU + 1 standalone vLLM
replica GPU (H100 80GB x2, one container), Qwen3-0.6B, GRPO n=4, `train_batch_size=16`, `ppo_mini_batch_size=8`,
`parameter_sync_step=2`, prompt 512 / response 512, 6 steps, `agent.num_workers=4`, `checkpoint_engine.backend=nccl`,
`hybrid_engine=False`, overrides mirroring `tests/special_e2e/run_v1_separate_async.sh`. `--chaos`: after step 2
completes, `ray.kill(no_restart=True)` of one alive `AgentLoopWorkerTQ` (`agent_loop_worker_0_*`, 4 alive, 3 survive),
within 0.1 s of the step line (pre-armed killer); caps: 1200 s wall, 600 s with no step after the kill = `hung`.
Flags ON = `trainer.v1.sampler.inflight_ttl_s=60 actor_rollout_ref.rollout.agent.relaunch_dead_workers=True`;
OFF = `inflight_ttl_s=null relaunch_dead_workers=False` (the branch's defaults). Full commands: `bench/results/<run>/command.txt`.

## Headline: outcome after killing an agent-loop worker

| run | tree | chaos | outcome | rc | steps reached / 6 | kill at | kill -> next completed step | kill -> exit | wall |
|-----|------|-------|---------|----|-------------------|---------|-----------------------------|--------------|------|
| sep2-chaos-main-1 | main@2ab35292 | kill after step 2 (landed after step 4: 9 s killer start-up, see note) | **crashed** | 1 | 5 | +265.4 s | 4.2 s (step 5 was in flight) | 6.0 s | 271 s |
| sep2-chaos-main-2 | main@2ab35292 | kill after step 2 (landed after step 4: 10 s GCS listing) | **crashed** | 1 | 4 | +341.8 s | 3.2 s (in-flight step) | 8.6 s | 350 s |
| sep2-chaos-main-3 | main@2ab35292 | kill 0.0 s after step 2 | **crashed** | 1 | 4 | +254.3 s | 2.8 s (steps 3-4 were in flight) | 6.9 s | 261 s |
| sep2-chaos-p3-off-1 | contrib/p3-v1-never-hang@5e6f6b74 (flags OFF) | kill 0.0 s after step 2 | **crashed** | 1 | 4 | +262.2 s | 2.4 s (in-flight) | 7.2 s | 269 s |
| sep2-chaos-p3-on-1 | contrib/p3-v1-never-hang@5e6f6b74 (flags ON) | kill 0.0 s after step 2 | **completed** | 0 | 6 | +271.1 s | 2.8 s; recovery step 5: 13.9 s vs 4.2 s steady state | n/a (finished) | 300 s |
| sep2-chaos-p3-on-2 | contrib/p3-v1-never-hang@5e6f6b74 (flags ON) | kill 0.0 s after step 2 | **completed** | 0 | 6 | +431.7 s | 3.3 s; recovery step 5: 24.5 s vs 7.1-7.6 s steady state on this (slower) host | n/a (finished) | 476 s |
| sep2-p3-on-1 | p3@5e6f6b74 (flags ON) | none | completed | 0 | 6 | - | - | - | 277 s |
| sep2-p3-on-2 | p3@5e6f6b74 (flags ON) | none | completed | 0 | 6 | - | - | - | 247 s |
| sep2-main-2 | main@2ab35292 | none | completed | 0 | 6 | - | - | - | 272 s |
| sep2-main-1 | main@2ab35292 | none | completed | 0 | 6 | - | - | - | 276 s |

Main / flags-OFF failure (identical traceback in all four crashes): the next dispatch hits the dead actor,
`trainer_base.fit -> step -> trainer_separate_async.prepare_step:288 -> trainer_base.prepare_step:1527 ->
_add_batch_to_generate:1524 -> _submit_batch_to_rollout:1512 -> agent_loop_tq.AgentLoopManagerTQ.generate_sequences:257`
-> `ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task. class_name: AgentLoopWorkerTQ`,
re-raised by `ray.get` in `main_ppo.run_ppo:97`, rc=1 about 7 s after the kill. The steps that complete between the kill
and the crash are the ones already generated (steps 3-4 of 16-prompt batches), the 4 prompt groups the dead worker owned
are lost. (The "hung" variant of `drop`-leaked slots would appear on a larger batch where the trainer waits for them;
on this 16-prompt preset the crash wins the race every time: 4/4.)

Flags ON (sep2-chaos-p3-on-1), per-step `metrics.jsonl`: steps 1-4 timing_s/step 19.0 / 4.4 / 4.3 / 4.2 (the kill lands
after step 2; steps 3-4 train on prompts already generated), step 5 **13.9 s** with `rollout/dead_workers_relaunched=1`
and `rollout/prompts_redispatched=4` (the 4 groups the dead worker owned, re-dispatched to the relaunched worker;
`training/inflight_reissued` stayed 0: the dead-worker path caught them before the 60 s TTL), step 6 6.3 s, rc=0.
`off_policy/trajectory_staleness/mean` 1.0 -> 0.75 (step 5) -> 0.5 (step 6) as the redispatched groups come in a version
later. Cost of the recovery: ~10 s on one step (liveness probe period 10 s + worker relaunch), nothing afterwards.
Second run (sep2-chaos-p3-on-2, same tree, same kill point): same shape - steps 2-4 7.4 / 7.1 / 7.6 s, step 5 24.5 s
with `rollout/dead_workers_relaunched=1`, `rollout/prompts_redispatched=4`, step 6 9.2 s, rc=0, 6/6 steps. Score:
main 0/3 survived, flags OFF 0/1, flags ON 2/2.

## No-regression (flags ON, no chaos) vs main - training-side timings, mean of steps 2..6 (s)

| run | tree | old_log_prob | update_actor | train sum | step | gen (noise, see README) | peak mem GB |
|-----|------|--------------|--------------|-----------|------|-------------------------|-------------|
| sep2-main-1 | main@2ab35292 | 1.853 | 1.239 | 3.09 | 3.78 | 0.019 | 13.80 |
| sep2-p3-on-1 | p3@5e6f6b74 ON | 2.010 | 1.401 | 3.41 | 4.13 | 0.036 | 13.74 |
| sep2-main-2 | main@2ab35292 | 2.022 | 1.236 | 3.26 | 3.98 | 0.019 | 13.93 |
| sep2-p3-on-2 | p3@5e6f6b74 ON | 1.784 | 1.321 | 3.11 | 3.79 | 0.035 | 13.97 |
| **main mean +- spread** | | 1.94 +- 0.08 | 1.24 +- 0.00 | 3.18 +- 0.08 | 3.88 +- 0.10 | | 13.87 |
| **p3 ON mean +- spread** | | 1.90 +- 0.11 | 1.36 +- 0.04 | 3.26 +- 0.15 | 3.96 +- 0.17 | | 13.86 |

(separate_async: `timing_s/gen` is the trainer's wait for sampleable groups, not generation; the standalone replica
generates continuously.) Two runs per arm: the ON arm's training-side sum is +2.5 % and step time +2 % vs main, inside
the run-to-run spread of either arm (+-3..4 %); the 10 s liveness probe and the TTL sweep are not visible at this scale.

## Final split branches (p3a TTL, p3b relaunch) - same preset, same kill point

| run | tree | knobs | outcome | steps / 6 | kill -> next step | recovery metrics |
|-----|------|-------|---------|-----------|-------------------|------------------|
| sep2-chaos-p3b-on-1 / -2 | p3b@e43e784c (contains p3a@7f43e43e) | TTL 60 + max_inflight_reissues 3 + relaunch | **completed x2** (rc=0) | 6 / 6 | 2.8 / 2.9 s | step 5 13.4 / 14.3 s: `dead_workers_relaunched=1`, `prompts_redispatched=4`, `inflight_reissued=0` |
| **sep2-chaos-p3b-on-final** | **p3b@f6846af0 (FINAL)** | TTL 60 + reissues 3 + relaunch | **completed** (rc=0, wall 273 s) | 6 | 2.6 s | step 5 6.1 s (`dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`), step 6 10.8 s (`inflight_reissued=8` via the fresh-uid path) |
| sep2-chaos-p3b-off-1 | p3b@e43e784c | both OFF | **crashed** (rc=1, same `ActorDiedError` path) | 4 | 2.7 s | - |
| sep2-chaos-p3a-ttl-1 | p3a@7f43e43e | TTL only | **crashed** (rc=1): the dead-handle dispatch raises ~7 s after the kill, before any TTL can fire | 4 | 2.8 s | - |
| sep2-p3b-on-1 | p3b@e43e784c | ON, no chaos | completed | 6 | - | train sum 3.54 s, step 4.17 s (main 3.78 / 3.98) |

Full tables and discussion: `packages/p3a/bench/RESULTS.md`, `packages/p3b/bench/RESULTS.md`.

## GPU unit tests

`gpu-tests-p3-1` on p3@5e6f6b74 (the pre-split branch, 10 commits), 8x H100, the `gpu_unit_tests.yml` selection:
**503 passed / 0 failed / 8 skipped in 33 min 18 s** (rc=0; main reference `gpu-tests-main-2`: 503 / 0 / 8), torchrun
2/2 rc=0. The final split tree is tested by `gpu-tests-p3b-1` (p3b@e43e784c) - see `packages/p3b/bench/RESULTS.md`.

Apps (all stopped by id, see the launch registry (kept privately)): sep2-main-1 <modal-app>, sep2-chaos-main-1
<modal-app>, sep2-chaos-main-2 <modal-app>, sep2-chaos-main-3 <modal-app>,
sep2-chaos-p3-on-1 <modal-app>, sep2-chaos-p3-off-1 <modal-app>, sep2-p3-on-1 <modal-app>.
Results: `bench/results/<run>/{summary.json,metrics.jsonl,train.log,harness.log,command.txt}`.
