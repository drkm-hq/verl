# P3b worker relaunch (stacked on P3a in-flight TTL): Modal GPU evidence (bench/README.md harness)

Trees: `contrib/p3b-worker-relaunch@e43e784c` (2 commits ahead of main@2ab35292; contains `contrib/p3a-inflight-ttl@7f43e43e`)
and the final proof tree `f6846af0` (3 commits; its product code equals the reviewed head d0da57de, which only adds a docs
clause). The review-round-2 commits (p3a 951bc3f4 / 6445f21b, p3b 833d0444 / 08209a98 / 3f3d2c24) change product code again (per-step
TTL budget, validation empty-batch handling, validation recovery counters, generations-logger tolerance) and WERE re-run on Modal:
`chaos-p3b-final-1` / `-2` (completed 2/2) and the stock GPU unit-test selection `gpu-tests-p3b-r2-2` (8x H200) on 3f3d2c24,
sections at the end of this file. The round-3 commits (p3a d834b6b3 / e598929b / 2bd5c040, p3b e9129696 / 702201d5 / a3ad633f / 1b397dfd: exhaustion failures
charged against the TTL step budget, padding-mode hold-back, dead-worker bounds on death events, counter keys kept) change `replay_buffer.py` and
`agent_loop_tq.py` again and have not been run on Modal. Preset `separate-async-2gpu`: V1 `trainer_mode=separate_async`,
1 trainer GPU + 1 standalone vLLM replica GPU (H100 80GB x2), Qwen3-0.6B, GRPO n=4, `train_batch_size=16`,
`ppo_mini_batch_size=8`, `parameter_sync_step=2`, prompt 512 / response 512, 6 steps, `agent.num_workers=4`,
`checkpoint_engine.backend=nccl`, `hybrid_engine=False`, overrides mirroring `tests/special_e2e/run_v1_separate_async.sh`.
`--chaos`: after step 2 completes, `ray.kill(no_restart=True)` of one alive `AgentLoopWorkerTQ` (4 alive, 3 survive) within
0.1 s of the step line; caps 1200 s wall / 600 s with no step after the kill (= `hung`).
Knobs ON = `trainer.v1.sampler.inflight_ttl_s=60 trainer.v1.sampler.max_inflight_reissues=3
actor_rollout_ref.rollout.agent.relaunch_dead_workers=True`; OFF = `inflight_ttl_s=null relaunch_dead_workers=False`
(the defaults). Full commands: `bench/results/<run>/command.txt`.

## Headline: outcome after killing an agent-loop worker

| run | tree | chaos | outcome | rc | steps / 6 | kill at | kill -> next completed step | recovery metrics |
|-----|------|-------|---------|----|-----------|---------|-----------------------------|------------------|
| sep2-chaos-main-1 / -2 / -3 | main@2ab35292 | kill after step 2 (main-1/-2 landed after step 4) | **crashed x3** (rc=1, `ActorDiedError` in `AgentLoopManagerTQ.generate_sequences`, ~7 s after the kill) | 1 | 5 / 4 / 4 | +265 / +342 / +254 s | 2.4-4.2 s (in-flight step only) | - |
| sep2-chaos-p3-on-1 / -2 (pre-split p3@5e6f6b74, TTL 60 + relaunch) | p3@5e6f6b74 ON | yes | **completed x2** | 0 | 6 / 6 | +271 / +432 s | 2.8 / 3.3 s | step 5: 13.9 s (vs 4.2) and 24.5 s (vs 7.4): `rollout/dead_workers_relaunched=1`, `rollout/prompts_redispatched=4`, `training/inflight_reissued=0` |
| sep2-chaos-p3-off-1 (pre-split, knobs OFF) | p3@5e6f6b74 OFF | yes | **crashed** | 1 | 4 | +262 s | 2.4 s | - |
| sep2-chaos-p3b-on-1 | p3b@e43e784c ON | yes | **completed** | 0 | 6 | +255.6 s | 2.8 s | steps 2-4 4.4 / 3.9 / 3.8 s, **step 5 13.4 s** with `rollout/dead_workers_relaunched=1`, `rollout/prompts_redispatched=4` (`training/inflight_reissued=0`), step 6 6.8 s |
| sep2-chaos-p3b-on-2 | p3b@e43e784c ON | yes | **completed** | 0 | 6 | +281.0 s | 2.9 s | steps 2-4 4.5 / 4.1 / 4.3 s, **step 5 14.3 s** with relaunched=1, redispatched=4, step 6 6.1 s |
| **sep2-chaos-p3b-on-final** | **p3b@f6846af0 (3 commits, FINAL)** ON | yes | **completed** | 0 | 6 | +248.1 s | 2.6 s | steps 2-4 4.2 / 3.9 / 3.9 s, **step 5 6.1 s** (`rollout/dead_workers_relaunched=1`, `rollout/prompts_orphaned_by_dead_workers=4`, `rollout/prompts_orphaned_at_dispatch=4`), **step 6 10.8 s** (`training/inflight_reissued=8`, `training/inflight_dead_worker_reissued=8`: the orphaned uids re-issued under fresh uids by the final dispatch-retry path), wall 273 s |
| sep2-chaos-p3b-off-1 | p3b@e43e784c OFF | yes | **crashed** (same `ActorDiedError` in `AgentLoopManagerTQ.generate_sequences`) | 1 | 4 | +267.6 s | 2.7 s | - |
| sep2-chaos-p3a-ttl-1 | p3a@7f43e43e TTL only (`inflight_ttl_s=60`, `max_inflight_reissues=3`; no relaunch knob on that tree) | yes | **crashed** | 1 | 4 | +235.8 s | 2.8 s | the next dispatch lands on the dead handle ~7 s after the kill, before a 60 s TTL could re-issue anything |
| sep2-p3b-on-1 | p3b@e43e784c ON | no | completed | 0 | 6 | - | - | train sum 3.54 s (old_log_prob 2.26 + update_actor 1.28), step 4.17 s: main 3.78 / 3.98, pre-split p3 4.13 / 3.79 - inside the 2-GPU spread, no regression |

Score: main 0/3 survive the kill, knobs OFF 0/2, TTL only 0/1 (expected: the last-resort path of [1/2] alone cannot survive a dead
dispatch handle), TTL + relaunch **5/5** (pre-split p3 x2, p3b e43e784c x2, final f6846af0 x1). On the four pre-final completed
runs `training/inflight_reissued` was 0 (the chunk was re-dispatched under the same uids, `rollout/prompts_redispatched=4`); on
f6846af0 the parked chunk went through the fresh-uid re-issue (`training/inflight_reissued=8` = `inflight_dead_worker_reissued`,
`inflight_reissue_exhausted=0`, `rollout/prompts_orphaned_at_dispatch=4`, `rollout/prompts_orphaned_by_dead_workers=4`). The
recovery costs ~10 s on one step (10 s liveness-probe period + relaunch) and nothing afterwards; the 4 prompt groups the
dead worker owned are re-dispatched to the relaunched worker, so the batch size stays 16 and no sample is dropped
(`off_policy/trajectory_staleness/mean` dips to 0.75 / 0.5 on steps 5-6 as those groups arrive a version later).
Sha note: the branch was rebuilt once on p3a@7f43e43e (a2ce4020 -> e43e784c, one test added, product code unchanged);
every arm above ran at e43e784c.
| sep2-main-1 / -2 | main@2ab35292 | no | completed | 0 | 6 | - | - | train sum 3.09 / 3.26 s, step 3.78 / 3.98 s |

## GPU unit tests

| run | tree | pytest | torchrun | new failures |
|-----|------|--------|----------|--------------|
| gpu-tests-main-2 (reference) | main@2ab35292 | 503 / 0 / 8 | 2/2 rc=0 | - |
| gpu-tests-p3-1 (pre-split) | p3@5e6f6b74 | 503 / 0 / 8, 33 min 18 s | 2/2 rc=0 | none |
| gpu-tests-p3b-1 | p3b@e43e784c (2 commits) | **503 / 0 / 8**, 33 min 20 s, rc=0 | 2/2 rc=0 | none |
| gpu-tests-p3b-2 | p3b@f6846af0 (3 commits, final; combined p3 f6366f48 = same tree) | **503 / 0 / 8**, 26 min 57 s, rc=0 | 2/2 rc=0 | none |

Apps: see the launch registry (kept privately) (stopped by id when each run finished). Artifacts: `bench/results/<run>/`.

## Review round 2: chaos re-run on the re-reviewed tree (p3b@0d28f34f, 2026-10-29) -- "0d28f34f, superseded": p3b moves again (budget placement + wandb-table fixes pending); the outcome stays a valid signal for the relaunch path

Same preset and knobs-ON string as above (`separate-async-2gpu --chaos`, `inflight_ttl_s=60 max_inflight_reissues=3
relaunch_dead_workers=True`), detached worktree at `contrib/p3b-worker-relaunch@0d28f34f` ("validation completes when every
group of a batch lost its worker; log its recovery counters"), 2x H100, `meta.json` `digest_match=True`.

| run | app | outcome | rc | steps / 6 | kill (after step 2) | kill -> next step | steps 3-6 (s) | recovery counters |
|-----|-----|---------|----|-----------|---------------------|-------------------|---------------|-------------------|
| chaos-p3b-r2-1 | <modal-app> | **completed** | 0 | 6 | `agent_loop_worker_0_512d1da5` at +268.8 s, pid gone 0.1 s later, 4 alive before | 2.5 s | 4.18 / 4.42 / 4.09 / **14.04** | step 5 `rollout/dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`; step 6 `training/inflight_reissued=4`, `inflight_dead_worker_reissued=4`, `inflight_reissue_exhausted=0`; wall 293.7 s |
| chaos-p3b-r2-2 | <modal-app> | **completed** | 0 | 6 | `agent_loop_worker_0_064fe184` at +255.2 s, pid gone 0.1 s | 2.6 s | 3.91 / 3.95 / 4.16 / **12.85** | step 5 relaunched=1, orphaned 4 + 4; step 6 `inflight_reissued=8`, `inflight_dead_worker_reissued=8`, exhausted 0; wall 280.5 s; train.log: "Agent loop worker 0 is dead (ActorDiedError); re-creating it (attempt 1/5)" |

Score on this tree: 2/2 completed (cumulative knobs-ON: 7/7 across p3 5e6f6b74 x2, p3b e43e784c x2, f6846af0 x1, 0d28f34f x2;
main and knobs-OFF 0/6). Steady-state step 3.9-4.4 s as before; the recovery cost is one ~13-14 s step (the liveness probe
period + relaunch + the re-issue of the orphaned prompts under fresh uids), nothing afterwards. Stock GPU unit-test selection on
0d28f34f: `gpu-tests-p3b-r2` was withdrawn from the queue when the hold arrived; the final p3b sha gets the chaos arm and the suite again.

## Review round 2: chaos arms on the FINAL tree (p3b@3f3d2c24, contains p3a@6445f21b; 2026-10-29)

Same preset and knobs-ON string, detached worktree `wt/bench-p3b-final`, 2x H100, `digest_match=True`; the two arms ran in
parallel containers.

| run | app | outcome | rc | steps / 6 | kill (after step 2) | kill -> next step | steps 3-6 (s) | recovery counters |
|-----|-----|---------|----|-----------|---------------------|-------------------|---------------|-------------------|
| chaos-p3b-final-1 | <modal-app> | **completed** | 0 | 6 | `agent_loop_worker_0_5243fe6c` at +250.4 s, pid gone 0.1 s, 4 alive before | 2.6 s | 3.87 / 3.89 / **6.04** / **9.97** | step 5 `rollout/dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`; step 6 `training/inflight_reissued=8`, `inflight_dead_worker_reissued=8`, `inflight_reissue_exhausted=0`; wall 274.9 s |
| chaos-p3b-final-2 | <modal-app> | **completed** | 0 | 6 | `agent_loop_worker_0_72f6e487` at +260.2 s, pid gone 0.1 s | 2.6 s | 3.80 / 3.87 / **14.76** / 3.73 | step 5 (all in one step this time) relaunched=1, `prompts_orphaned_by_dead_workers=12`, `prompts_orphaned_at_dispatch=4`, `inflight_reissued=7`, `inflight_dead_worker_reissued=7`, exhausted 0; wall 286.1 s |

No `ttl_suspended` counter appears in either run's metrics.jsonl (the keys emitted on this tree are the six above); if the
tree logs it only when non-zero, the TTL path was never suspended here. Score on the final tree: 2/2 completed; cumulative
knobs-ON 9/9 (main and knobs-OFF 0/6). Stock GPU unit-test selection on 3f3d2c24: `gpu-tests-p3b-r2` (regression coverage).

## Review round 3: chaos arm on p3b@45690b0f ("45690b0f, superseded": the round-3 re-review adds two MEDIUM fixes in the TTL hold-back and the dead-worker bound)

`chaos-p3b-r3-1` (app <modal-app>, 2x H100, detached worktree `wt/bench-p3b-r3`, knobs ON as before; the TTL
budget path changed and the dead-worker path was re-expressed on this tree): **completed** rc=0, 6/6 steps, wall 291.5 s;
kill of `agent_loop_worker_0_2b2474fb` at +266.3 s (after step 2), kill -> next completed step 2.7 s; steps 3-6 3.99 / 4.09 /
**3.82** / **13.82** s; step 5 `rollout/dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`,
`prompts_orphaned_at_dispatch=4`; step 6 `training/inflight_reissued=4`, `inflight_dead_worker_reissued=4`,
`inflight_reissue_exhausted=0`. The metrics carry no ttl / suspend / budget / hold-back key on this tree either. Cumulative
knobs-ON survival 10/10. The stock suite `gpu-tests-p3b-r3` on 45690b0f is superseded with it; both repeat on the final p3b sha.

## Review round 3: chaos arm on p3b@64212367 ("64212367, superseded": two small MEDIUMs follow in the round-3b code)

`chaos-p3b-r3-1` on 64212367 (app <modal-app>, 2x H100, `wt/bench-p3b-r3b`, knobs ON; "bound the dead-worker
exemption per step and fail loudly; count row-less validation prompts beside dead-worker failures"): **completed** rc=0, 6/6,
wall 306.5 s; kill of `agent_loop_worker_0_6f1eb3e6` at +280.3 s (after step 2), kill -> next step 2.7 s; steps 3-6 3.91 / 4.31 /
**4.06** / **14.36** s; step 5 `rollout/dead_workers_relaunched=1`, `prompts_orphaned_by_dead_workers=4`,
`prompts_orphaned_at_dispatch=4`; step 6 `training/inflight_reissued=4`, `inflight_dead_worker_reissued=4`,
`inflight_reissue_exhausted=0`; no ttl / suspend / budget / hold / exempt key in the metrics. Cumulative knobs-ON 11/11. The
45690b0f result is kept as `bench/results/chaos-p3b-r3-1-45690b0f-superseded/`. `gpu-tests-p3b-r3` on 64212367 was running
when the supersession arrived and finishes under the same label; the final p3b sha gets both runs again.

## Review round 3: chaos arm on the FINAL tree p3b@1b397dfd (contains p3a@2bd5c040; "bound dead-worker recovery by death events per step, not by orphaned prompts")

`chaos-p3b-final3-1` (app <modal-app>, 2x H100, detached worktree `wt/bench-p3b-final3`, knobs ON
`inflight_ttl_s=60 max_inflight_reissues=3 relaunch_dead_workers=True`, `digest_match=True`): **completed** rc=0, 6/6 steps,
wall 292.0 s; kill of `agent_loop_worker_0_de5e5232` at +266.8 s (after step 2), pid gone 0.1 s later, kill -> next completed
step **2.3 s**; steps 3-6 4.13 / 4.33 / **6.24** / **11.45** s; step 5 `rollout/dead_workers_relaunched=1`,
`prompts_orphaned_by_dead_workers=4`, `prompts_orphaned_at_dispatch=4`; step 6 `training/inflight_reissued=7`,
`inflight_dead_worker_reissued=7`, `inflight_reissue_exhausted=0`. As on every earlier tree, the metrics carry no ttl /
suspend / budget / hold / exempt / death-event key (the new per-step death-event bound is not surfaced as a metric, or only when
hit). Round-3 chaos score: 45690b0f, 64212367 (both superseded) and 1b397dfd all completed; cumulative knobs-ON 12/12 since
round 1 (main and knobs-OFF 0/6). Stock suite on 1b397dfd: `gpu-tests-p3b-r3-final` (app <modal-app>, 8x H200): console 503 passed / 0 failed /
8 skipped in 30 min 25 s, junit 529 / 521 / 0 / 8, torchrun 2/2 rc=0 (regression coverage).
