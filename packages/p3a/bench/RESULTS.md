# P3a in-flight prompt TTL: Modal GPU evidence (bench/README.md harness)

Tree: `contrib/p3a-inflight-ttl@7f43e43e` (1 commit ahead of main@2ab35292) at the time of these runs; the review-round-2 commits 951bc3f4 (per-step re-issue budget) and 6445f21b (budget on the re-issuable set only, per step) were covered by the re-run of the [2/2] tree 3f3d2c24 (`chaos-p3b-final-1`/`-2` completed 2/2, `gpu-tests-p3b-r2-2` on 8x H200; see `packages/p3b/bench/RESULTS.md`); the round-3 commits d834b6b3 (exhaustion failures charged against the budget), e598929b (padding-mode hold-back, positive budget) and 2bd5c040 (hold-back charges nothing) change `replay_buffer.py` again and have not been run on Modal. Its keys: `trainer.v1.sampler.inflight_ttl_s`
(default null = off) and `trainer.v1.sampler.max_inflight_reissues` (default 3). The stacked P3b branch
(`contrib/p3b-worker-relaunch@e43e784c`) adds `actor_rollout_ref.rollout.agent.relaunch_dead_workers`; the "both knobs"
arms and the GPU unit tests ran on P3b (which contains this branch) - see `packages/p3b/bench/RESULTS.md`.

Preset `separate-async-2gpu` (1 trainer GPU + 1 standalone vLLM GPU, Qwen3-0.6B, 16 prompts x 4, 512/512, 6 steps,
4 agent-loop workers) with `--chaos`: `ray.kill` of one `AgentLoopWorkerTQ` right after step 2 (details in
bench/README.md "Presets").

## TTL-only arm (this branch alone): does re-issue turn the crash into a completed run?

| run | tree | knobs | outcome | rc | steps / 6 | kill -> next completed step | recovery metrics |
|-----|------|-------|---------|----|-----------|-----------------------------|------------------|
| sep2-chaos-main-1 / -2 / -3 | main@2ab35292 | - | **crashed x3** | 1 | 5 / 4 / 4 | in-flight step only | - |
| sep2-chaos-p3a-ttl-1 | p3a@7f43e43e | `inflight_ttl_s=60 max_inflight_reissues=3` (no relaunch knob exists on this tree) | **crashed** | 1 | 4 | 2.8 s (in-flight step only) | none: rc=1 ~7 s after the kill, `ray.exceptions.ActorDiedError` in `AgentLoopManagerTQ.generate_sequences` (the same path as main) |
| sep2-chaos-p3b-on-1 / -2 | p3b@e43e784c | TTL 60 + relaunch | **completed x2** | 0 | 6 / 6 | 2.8 / 2.9 s | step 5 13.4 / 14.3 s: `rollout/dead_workers_relaunched=1`, `rollout/prompts_redispatched=4`, `training/inflight_reissued=0` |

Why the TTL alone cannot turn this failure into a completed step: the trainer's next `prepare_step` dispatches the next
prompt chunk round-robin across all 4 worker handles within one step (~4 s) of the kill, and the call on the dead handle
raises `ActorDiedError` immediately - 7 s after the kill, long before a 60 s TTL (or any TTL: the dispatch happens
every step) could classify the dead worker's 4 prompt groups as expired and re-issue them to the 3 survivors. The TTL
is the remedy for the other failure shape - a live worker that never reports a terminal status (silent hang, dropped
result) - where nothing raises and sampling would otherwise block forever; P3b's relaunch (dead-worker detection by
liveness probe, re-dispatch of its orphaned prompts, replacement actor) is what turns this kill into a 10 s hiccup.
On the four completed runs of the pre-final trees (p3 5e6f6b74 x2, p3b e43e784c x2) `training/inflight_reissued` was 0: the
relaunched worker got the dead worker's chunk re-dispatched under the same uids (`rollout/prompts_redispatched=4`). On the
final tree f6846af0, whose dispatch path parks a chunk instead of re-sending it, the orphaned prompts went through the
fresh-uid re-issue: `training/inflight_reissued=8` (= `inflight_dead_worker_reissued`), `inflight_reissue_exhausted=0`.

## GPU unit tests

Run on the stacked P3b tree (contains P3a): `gpu-tests-p3b-1` - see `packages/p3b/bench/RESULTS.md`; the pre-split
combined branch `gpu-tests-p3-1` (p3@5e6f6b74): 503 passed / 0 failed / 8 skipped, torchrun 2/2 rc=0.
