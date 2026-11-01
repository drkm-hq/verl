# 01 — What the verl maintainers say they want (roadmap + RFC demand study)

Scope: verl-project/verl at main `2ab35292` (2026-10-27), dumps taken 2026-10-28
(`gh/issues_open.json` 641 issues, `gh/prs_open.json` 774 PRs, `gh/prs_merged_recent.json`
400 PRs merged 2026-07-16 .. 2026-10-28). Read-only study; nothing was posted to GitHub.

Conventions: `#N` = issue or PR number on verl-project/verl. Reaction counts come from
`gh api repos/verl-project/verl/issues/N` (the dump's issue-level `reactionGroups` is empty
for every issue, so the dump cannot be used for reactions; comment-level reactions in the
dump are populated). "Claimed" means an open PR exists whose title/body/files match the item.
Guesses are marked (guess).

## 1. Label inventory (open issues)

Labels are sparse on this repo: 177 `bug`, 9 `call for contribution`, 7 `roadmap`,
7 `enhancement`, 5 `rollout`, 4 `help wanted`, 3 `RFC`, 2 `good first issue`, 1 `speed`,
0 `high priority`, 0 `status: accepted`. Most RFCs carry no label at all; 65 open issues
have `[RFC]`/`[roadmap]`/`[Tracking]`/`[Feature Proposal]` in the title. The maintainers'
real planning surface is the quarterly roadmap issue (#8186), not labels.
CONTRIBUTING.md (`CONTRIBUTING.md:13-19`) points contributors at the `good first issue`,
`call for contribution`, `RFC` and `roadmap` label queries.

### 1.1 Issues carrying the labels of interest

| # | label(s) | age | comments | title | status read |
|---|---|---|---|---|---|
| #708 | call for contribution, roadmap | 585d | 14 | [Project] deepseek R1 infrastructure | historical (2025-Q1); DeepSeek 671B landed via Megatron EP (#1467) |
| #400 | enhancement, call for contribution | 607d | 6 | TP rollout + FSDP / TP actor | historical; resharding solved by hybrid engine |
| #144 | enhancement, good first issue, call for contribution | 638d | 6 | Additional memory optimization features | historical (activation offload, etc. mostly landed) |
| #20 | question, megatron, call for contribution | 705d | 6 | non-RmPad vs RmPad model interchangeable? | historical |
| #2388 | call for contribution, roadmap | 477d | 3 | [roadmap] verl Q3 (2025) development | 34 reactions (+1 21); see §2.3 |
| #155 | enhancement, call for contribution | 637d | 3 | llama3 rope_scaling | historical, done upstream |
| #97 | help wanted, call for contribution | 653d | 3 | Fused CE loss integration | historical; Liger `USE_LIGER=1` exists (README "Key Features") |
| #1388 | bug, good first issue, call for contribution | 541d | 2 | FSDP2 `model_rmpad` CI test unstable | stale CI flake |
| #609 | call for contribution | 591d | 1 | resume-from-checkpoint for SFT | likely done (guess; SFT trainer has checkpoint code) |
| #6414 | help wanted | 160d | 1 | separate check of weight-conversion logic (zh) | overlaps #8252 weight checker |
| #6830 | bug, help wanted | 125d | 0 | `process_validation_metrics()` crashes on None-filled sparse `reward_extra_info` | small bug, open |
| #1757 | help wanted | 516d | 0 | vllm cannot load model after megatron training | stale |
| #384 | speed | 609d | 7 | Startup times are slow | stale but the only `speed`-labelled issue |
| #5755 | roadmap | 215d | 21 | [RFC] Multi-modal Generation RL 2026Q2 Roadmap | verl-omni side (separate repo) |
| #3624 | roadmap | 397d | 8 | verl next steps | 76 reactions (+1 53, eyes 16); see §2.2 |
| #5972 | roadmap | 199d | 2 | [roadmap] low precision support | see §2.5 |
| #5042 | roadmap | 274d | 2 | [roadmap] verl TensorRT-LLM rollout roadmap | TRT-LLM rollout exists (`docs/workers/trtllm_worker.rst`) |
| #2618 | roadmap, rollout | 466d | 1 | [agentic RL] multi-turn rollout and agent loop tracking | 23 reactions; mostly landed (agent loop, tool calls) |
| #658 | RFC | 588d | 17 | Load Balancing for rollout phase | see §3 |
| #8324 | enhancement, megatron, RFC | 5d | 1 | [RFC] Modern Megatron integration | maintainer-owned (ISEEKYAN); see §3 |
| #1063 | RFC | 561d | 1 | verl auto resharding design by generator | stale |

Takeaway: every `call for contribution` / `good first issue` / `help wanted` item is 125-705
days old and either done or stale. The labels are not maintained; demand has to be read from
#8186 (26Q4 roadmap, 17 days old, 10 comments, actively answered by @wuxibin89) and from the
unlabelled `[RFC]` issues.

### 1.2 High-signal unlabelled planning issues (reactions from the API)

| # | +1 / total | comments | title |
|---|---|---|---|
| #3624 | 53 / 76 | 8 | verl next steps (vermouth1992, 2025-09) |
| #2388 | 21 / 34 | 3 | [roadmap] verl Q3 development (eric-haibin-lin, 2025-07) |
| #5790 | 22 / 27 | 9 | [RFC] Agent Abstractions and Trajectory Gateway |
| #2618 | 15 / 23 | 1 | [agentic RL] agent loop tracking |
| #22 | 17 / 17 | 7 | [Roadmap] veRL Development Roadmap (2024-11) |
| #5375 | 13 / 13 | 6 | [RFC] Context Compression into Agentic Training |
| #6401 | 6 / 8 | 4 | [RFC] Prefix-Tree Shared Attention for Multi-Turn RL |
| #5400 | 0 / 7 (rocket 7) | 4 | [RFC] PPOTrainer with TransferQueue Integration |
| #5306 | 0 / 6 | 5 | [roadmap] TorchTitan Integration (landed: `verl/workers/engine/torchtitan/`) |
| #4318 | 5 / 5 | 19 | [RFC] Reward Loop (landed: `verl/experimental/reward_loop/`) |
| #324 | 5 / 5 | 1 | Parallel the reference model compute and reward_fn calling |
| #6577 | 3 / 5 | 0 | [RFC] Dynamo KV-aware routing / KVBM |
| #658 | 3 / 3 | 17 | Load Balancing for rollout phase |
| #7561 | 2 / 2 | 2 | [RFC] Tree Prefix Reuse for Multi-Trajectories Training |
| #6266 | 2 / 2 | 2 | [RFC] NCCL Suspend/Resume in colocated mode |
| #8186 | 0 / 0 | 10 | [roadmap] verl 26Q4 roadmap (wuxibin89, 2026-10-10) |

The 26Q4 roadmap has zero reactions but is where contributors are actually claiming work
(4 distinct claim comments in 17 days) and where @wuxibin89 answers within days.

## 2. Roadmap line items with status

Four roadmap generations exist. Only #8186 (26Q4) is live; the others are kept open as
"high-level trackers" by the Codex bot (@Kirrito-k423, "scheduled Codex GPT5.5 issue
processing task", 455 comments on open issues, last 2026-08-05).

### 2.1 #8186 — verl 26Q4 roadmap (wuxibin89, 2026-10-10, updated 2026-10-27)

Owner tags in the issue: Common/VeOmni @wuxibin89 (+@FoolPlayer @amenomori), Megatron
@ISEEKYAN @wuxibin89, FT @HaoWang113 @ETOgaosion @tongyx361.

| Area | Item | Prio | Status (evidence) |
|---|---|---|---|
| LP Common | Train/rollout quantizer parity | P0 | DONE: #8287 #8306 #8308 merged 2026-10-19..21 by wuxibin89 |
| LP Common | Unified low-precision recipe config across FSDP/VeOmni/Megatron/Automodel | P1 | OPEN, unclaimed. @VikingDeng proposed it (comment 2026-10) then withdrew: "#8287 already covers most of what I proposed"; wuxibin89 did not assign. Config still lives in 6 places (`rollout.quantization`, `fsdp_config.qat`, `megatron.qat`, `megatron.override_transformer_config.fp8*`, veomni `qat_implementation`, automodel `enable_fp8`). |
| LP Common | Low-precision inference-only forwards (ref, teacher, old_log_prob) | P1 | OPEN, unclaimed. No open PR matches (search "inference-only / ref fp8"); FSDP engine `transformer_impl.py` has no fp8/quant forward path for the ref role (grep). Speed item. |
| LP Common | Low-precision comm | P1 | CLAIMED by @amenomori; VeOmni node-local ReduceScatter merged #8330 (2026-10-22). FSDP2 float8 all-gather part still open. |
| LP Common | Batch-invariant / deterministic train+rollout kernels | P1/P2 | CLAIMED by @cr-gao: #8025 merged 2026-10-22 ("raw log-prob mismatch metrics"), #8026 open (true on-policy). @0z5a deferred to cr-gao. |
| LP Megatron | Modern Megatron integration (#8324) | P0 | MAINTAINER-OWNED (ISEEKYAN): fork prototype ISEEKYAN/verl#23/#25, Core PRs NVIDIA/Megatron-LM#8673/#8674. Do not touch. |
| LP Megatron | DeepSeek-V4 FP4->FP8 lossless path | P0 | MAINTAINER-OWNED; related drafts #7895 (sophiayyya, upd 36d), #7688 (ISEEKYAN). |
| LP Megatron | Blackwell MXFP8 E2E | P1 | DONE: #8127 merged 2026-10-22 (wengeezhang, NVIDIA). |
| LP Megatron | Blackwell NVFP4 training | P1 | CLAIMED: #7971 (zhangyimi, open, upd 0d). |
| LP Megatron | FP8 optimizer states; MoE W4A8 | P2 | OPEN, unclaimed (only #6992 draft W4A8 QAT *simulation* on FSDP, upd 36d). |
| LP Megatron | MXFP4 QAT recipe | (asked by wuxibin89) | CLAIMED by @0z5a: #8426 draft + #8425 (2026-10-27). |
| LP VeOmni | DeepSeek-V4/V4.1 QAT E2E | P0 | MAINTAINER-OWNED (wuxibin89/FoolPlayer). |
| LP VeOmni | torchao low-precision compute + float8/MXFP8 FSDP2 all-gather | P1 | OPEN, unclaimed. Only stale WIPs #4325 (FSDP2 FP8 training, 334d) and #1490 (533d). Speed item, but needs Blackwell/Hopper and torchao expertise. |
| LP VeOmni | torchao fake-quant QAT for NVFP4/MXFP4 | P1 | OPEN, unclaimed (guess: wuxibin89 will do it alongside V4.1 QAT). |
| Model support | DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, GLM-5.3-Flash | - | Partially claimed: #7483 HollowMan6 LoRA for DSV4/4.1/GLM5.3 (upd 0d), #7749 aoshen02 Qwen3.8-Flash-Next LoRA (upd 36d), #8403 ISEEKYAN DSV4.1 LoRA recipe. Full-weight support is maintainer work. |
| Rollout async | Token-level interrupt + persist KV cache across weight switch, no re-prefill | P0 | CLAIMED: #8273 cr-gao "preserve requests across weight updates" (vLLM `pause_generation(mode="keep")`, 17 files, GPU-tested, opened 2026-10-18). cr-gao also announced it on #8186. |
| Rollout async | Sample-level dispatch once #completed >= group size | P1 | CLAIMED by maintainer Begunner: #8203 `StreamingDataloader` actor (39 files, [BREAKING], GPU-verified on 64 GPUs). |
| Weight refit | Delta shard for quantized `(weight, scale)` | - | IN PROGRESS by ChangyiYang (#7060 item 3, branch not yet PR'd; #7676 "preserve DSv4 FP8 delta fidelity" open, upd 36d). wuxibin89 raised the block-sharding problem on discussion #7273. |
| Weight refit | NCCL M2N reshard | - | CLAIMED: NVIDIA chain #7432 #7552 #8087 #8088 (ss16118, dongha-yoon); competing #8395 `sharded_nixl` (rafiw, 2026-10-26, 235B 21.7s -> 6.35s) and #7108 Mooncake P2P. Crowded. |
| Rollout perf | Load balance: session-affinity routing for multi-turn agents | - | PARTLY DONE: #8179 merged 2026-10-10 (pin each multi-turn session to one vLLM DP rank, aoshen02); #7613 merged (randomized least-loaded tie-break). Open: #6533 aoshen02 draft, #7435 ralovets, #6940 KV-aware LB draft, #6122 group-level LB, #2629 skewness scheduler. Crowded. |
| Rollout perf | Centralized kv-cache pooling | - | OPEN. Mooncake-store connector documented (`docs/perf/rollout_kv_offload.md`), #8323 (daegyu94, salt external KV by weight version, GPU-tested on 3 engines), #6990 HanlinDu extra-prefix-cache (LMCache, upd 36d). |
| Rollout perf | kv-cache offload | - | OPEN: #7277/#7621 FlexKV (DylanChen-NV, NVIDIA, upd 36d, stalled). |
| FT | Rollout replica restart without interrupting trainer | - | CLAIMED: #8168 tongyx361 (maintainer) replica lifecycle APIs + `next_weight_sync` recovery; RFC #8380 HaoWang113 proposes `verl/ft/`. Also #7910 sunnylovestiramisu (Google, standby actors, re-rendezvous; upd 36d, no maintainer reply visible). |
| FT | Trainer restart without interrupting rollout | - | RFC #8380 (0 comments after 2 days); wuxibin89 noted on #7273 that M2N must support dynamic groups for this. |
| FT | Sandbox recover on restart | - | OPEN, unclaimed, no RFC. |
| Trainer | Deprecate `verl/experimental/fully_async_policy|one_step_off_policy` to verl-recipe | - | OPEN maintainer housekeeping; CI already migrated (#7357 merged). Unglamorous. |
| Trainer | Remove V0 trainer `verl/trainer/ppo/ray_trainer.py` | - | OPEN; `main_ppo.py:190-193` says "deprecated, will be removed in v0.9.0". 1823-line file. Housekeeping; a contributor PR here would conflict with every open PR touching ray_trainer.py. |
| Accelerators | Ascend (#8304), AMD (#6702 #6668 #7050 #8146 done), TPU (plugin done) | - | Vendor-owned. |

### 2.2 #6985 — verl 26Q3 roadmap (closed) and what carried over

Megatron mbridge dropped (#6951) and DeepSeek V4 landed. Carried into Q4 unfinished:
MXFP8/NVFP4 (#5972), megatron-lite polish (#7056), GLM 5.2 (landed later per Ascend docs),
delta weight sync (#7060), DeviceMesh p2p remap (#7552), **NCCL suspend/resume** (listed as a
maintainer item in Q3, PR #6408 by xiefan46 labelled "status: review in process", no human
activity since 2026-06-05 — stalled), KV-cache-aware scheduling (#6940 draft), Dynamo integration (recipe
PR verl-recipe#110), workload-aware scheduling (ThunderAgent, verl-recipe#126), Mooncake
distributed KV store (docs landed), remove legacy `RayPPOTrainer` (still open).

### 2.3 #3624 "verl next steps" (vermouth1992, 2025-09, 76 reactions) — the architecture thesis

Five services: Rollout Engine (server mode, done), Model Engine (done:
`verl/workers/engine/{fsdp,megatron,torchtitan,veomni,automodel}`), Weight Transfer Engine
(done: `verl/checkpoint_engine/` with naive/nccl/hccl/nixl/kimi/mooncake + delta_sync),
Agent Loop (done, `verl/experimental/agent_loop/`; uni-agent spun out), Data Transfer System
(done: TransferQueue is the V1 data plane, `docs/data/transfer_queue.md`). Top comment
(zhongwen-xu, 39 +1): "Be a library, not a framework." Unanswered user asks on the thread:
an OpenAI-gym-style env demo (Andcircle, 2025-12) and EPP/external-router integration docs.

### 2.4 #2388 — Q3 2025 roadmap (34 reactions): items still open

- "use tensordict and nested-tensor to remove padding and replace DataProto": partly done —
  NeoProto is the default V0 DataProto (#7557 merged 2026-08-31); V1 uses TransferQueue.
- "distributed data pool" -> TransferQueue (done).
- "SOTA multimodal VLM RL recipe (call for contribution)": moved to verl-recipe / verl-omni.
- "better dataset schema", "document how to add a new HF model", "model support status page"
  (#5389 tracks it): documentation debt, still open.
- "performance tuning and reference throughput benchmark across [model, size, seqlen,
  hardware, role]": never done in-tree (only `docs/algo/baseline.md` reward baselines). No PR.
- "async compute score" (#1584 via #1882 P1): done by Reward Loop (#4318).
- "request generation workload load balance among DP ranks" (#1722/#1818): see §2.1 load balance.

### 2.5 #5972 low-precision roadmap (sophiayyya, NVIDIA)

Rollout FP8 (vLLM/SGLang/TRT-LLM) done; MXFP8 rollout done (#7186 draft superseded by
#8306/#8308); NVFP4 QAT done (FSDP #5190, Megatron #5254); NVFP4 real quant #7971 open;
MXFP8 train #8127 done; NVFP4 train open. Kirrito-k423's suggested additions (FP8 KV cache,
FP8 for FSDP2, per-layer precision selection, cross-backend consistency) were not adopted by
sophiayyya except "I am running FP8 experiments on Blackwell".

## 3. RFC status table

"Maint." = first maintainer to respond (COLLABORATOR association). "Bot" = @Kirrito-k423's
scheduled Codex triage comment (Huawei Ascend engineer runs it; treat as non-binding).
PR `upd` dates of 2026-09-21 are a repo-wide mass update, not activity.

| RFC | Author | +1/total | Maint. response | PR / claim | Verdict |
|---|---|---|---|---|---|
| #5790 Agent Abstractions + Trajectory Gateway | zackcxb (2026-03) | 22/27 | wuxibin89 linked related asks (#4216, #5737, #5833) same week; no accept/reject | Author said draft PR "by tomorrow" (2026-04-07); gateway now lives in uni-agent (uni-agent#185 references "Gateway session"); rLLM offers `rllm-model-gateway` | DONE ELSEWHERE (uni-agent). Avoid. |
| #4318 Reward Loop | yyDing1 (maintainer) | 5/5 | self | Landed: `verl/experimental/reward_loop/`, docs `advance/reward_loop.rst` | DONE |
| #7839 backend-neutral WorkerGroup runtime (Ray + Monarch) | DarkHighness + zw0610 (maintainer) | 0 | none on RFC; PR #7840 APPROVED by zw0610 (100 files, targets `neo` branch) | #7840 | MAINTAINER-OWNED (zw0610). Avoid single_controller. |
| #8388 Isolate rollout Python runtimes | ZihaoW123 (2026-10-26) | 0 | wuxibin89 next day: "pretty easy ... with ray+uv" | #8407 Levius-Fubuki (2026-10-28, 20 files) | CLAIMED, maintainers lukewarm |
| #8324 Modern Megatron integration | ISEEKYAN (maintainer, NVIDIA) | 0 | self; Core PRs NVIDIA/Megatron-LM#8673/#8674 | fork ISEEKYAN/verl#23/#25 | MAINTAINER-OWNED |
| #7561 Tree Prefix Reuse (Megatron, AReaL-DTA style) | 1903943257 (2026-08) | 2/2 | wuxibin89 replied only with a link to #6689 | none; Jackie2049 "working on a similar topic" | DEFERRED to #6689 |
| #6401 Prefix-Tree Shared Attention (MAGI) | arvyanh (2026-05) | 6/8 | Bot review; Begunner reviewed PR #6689 (3 reviews, asked for unit tests + Flex e2e); 5.5k lines, conflicts | #6689 (upd 2026-09-21 mass) | STALLED 4 months |
| #8299 Shared-prefix packing for FSDP (trunk+branches) | DeeproChoudhury (2026-10-20) | 1/1 | none yet (8 days) | #8300 (1.7k lines, 60% tests, 2.7x on 2k prompt; only coderabbit + self reviews) | CLAIMED, fresh, waiting |
| #5375 Context compression in agentic training | ZhentaoFan (2026-02) | 13/13 | wuxibin89 (+3): "We're indeed going to support" via multi-output AgentLoop; huaiyizhao (+3) wants generic multi-trajectory | #5636 (reviewed by ArronHZG), verl-recipe#107; multi-output rows landed (#5969) | STALLED; bot asked maintainers 2026-08-05 to decide; no answer |
| #7579 Crash-durable replay buffer behind TransferQueue | savaresejeremy (2026-08) | 0 | Begunner asked a pointed consistency question (recovered buffer is newer than model ckpt); author answered; nothing since 2026-08-30 | prototype branch, 117 files, no PR | STALLED on a design decision (in-tree vs TransferQueue) |
| #8099 Stable optimizer-step counts for multi-output rollouts | luyuzhe111 (AWS, 2026-10-02) | 1/1 | none; eteluna (+1) has a ~100-line prototype | related #6025 (guillemgt, wuxibin89 commented), #8201 FZYsheep | CLAIMED-ish, unmerged family |
| #7712 Selected-token log-probs incl. fused kernels | eteluna (2026-09) | 0 | none (cc'd vermouth1992, zhtmike, wuxibin89) | narrowed to one engine output; WenzheWang offered the kernel slice; no PR | UNANSWERED, niche |
| #6266 NCCL suspend/resume in colocated mode | xiefan46 (2026-05) | 2/2 | ETOgaosion reviewed PR positively ("tremendous release of memory"); wuxibin89: "Hold until v0.8 release" (v0.8 since shipped) | #6408 (status: review in process), blocked on docker NCCL>=2.29.7 (#6621) | STALLED on infra + review, was on the Q3 roadmap |
| #8252 Weight checker for weight update | shiyuan680 (2026-10) | 0 | wuxibin89: use `scripts/rollout_logprob_diff.py` + QAT doc | none | REDIRECTED (effectively declined) |
| #8380 FT for separate_async | HaoWang113 (2026-10-26) | 0 | none yet; listed on #8186 with @HaoWang113 @ETOgaosion @tongyx361 | builds on #8168 (tongyx361) | CLAIMED by its authors + maintainers |
| #7060 delta_sharded tracking | ChangyiYang (2026-07) | 0 | wuxibin89 engaged (placement note); ChangyiYang: "need @wuxibin89's approval before implementation" (M2N composition) | #7676 fp8 delta (open), #7223 PP (merged), #7869 tests | OWNED by ChangyiYang; fp8 delta in progress |
| #5972 low precision roadmap | sophiayyya (NVIDIA, maintainer) | 0 | self | many merged | VENDOR-OWNED |
| #2618 agent loop tracking | wuxibin89 | 15/23 | self | mostly landed; `multi_turn` config move + tool cleanup still open | housekeeping |
| #658 rollout load balancing / partial rollout | uygnef (2025-03) | 3/3 | wuxibin89, eric-haibin-lin, zhaochenyang20 engaged in 2025; partial rollout shipped in fully_async + V1 (`FullyAsyncLLMServerClient`); least-in-flight LB in `llm_server.py:46` | #4349 recipe PR still open (reviewed by ccclyu) | MOSTLY DONE; umbrella kept open |
| #324 parallel ref + reward | Question406 (2025-02) | 5/5 | Bot only (2026-06): "still relevant, keep it conservative/configurable" | none | OPEN but low value: reward is already async (Reward Loop); old_log_prob -> ref -> values at `v1/trainer_base.py:577-597` share one GPU pool in colocated mode, so overlap only helps disaggregated ref |
| #5400 TransferQueue PPO trainer | wuxibin89 | 0/7 | vermouth1992: new config system; zw0610: abstract the data plane, do not add to single_controller | landed as V1 trainer | DONE (V1) |
| #7414 Mooncake as direct rollout data backend | zxpdemonio | 0 | wuxibin89: "We don't want Mooncake exposed to trainer directly ... behind TransferQueue" | #7415 draft | DECLINED as designed |
| #6577 Dynamo KV-aware routing / KVBM | sophiayyya (NVIDIA) | 3/5 | self | verl-recipe#110, #126 | VENDOR (recipe repo) |
| #6383 KV-cache-aware request load balancer | zhaizhiqiangA (2026-05) | 0 | Bot only | #6940 draft (from uni-agent) | STALLED, on Q3 roadmap |
| #6537 Generic RemoteBackend (Snowflake) | sfc-gh-kganesan (2026-05) | 0 | none (0 comments in 5 months) | Snowflake fork | IGNORED |
| #7269 Declarative topology API | ETOgaosion (maintainer) | 0 | wuxibin89: "confirm first with @zw0610" | #7531 bridge (zhang-cheng-hao) | MAINTAINER-OWNED, gated on zw0610 |
| #6225 Colocated checkpoint engine (Qwen team) | JackAlgorithm | 0 | wuxibin89 pushed back with CUDA-IPC numbers (DSV3.1 120s/80s); asked "where did you observe this"; no follow-up after 2026-06 | none | STALE |
| #5311 Megatron weight transfer speed | ISEEKYAN | 0 | self | superseded by Megatron-Bridge migration, delta, M2N | VENDOR |
| #7405 Fail-fast degenerate-run detector | nataliekung (2026-08) | 0 | none | none (author has 7 open PRs) | UNANSWERED, small |
| #7244 Per-turn reward, unbiased turn-level baseline | lijuliana (2026-08) | 0 | none; prior asks #2540 #3683 "closed without an answer" | fork; related #8305 draft turn-level process rewards (luzy99) | UNANSWERED, algorithmic |
| #4791 Suffix decoding + LSTM draft | vx120 | 4/4 | Bot; author: "implemented in verl-SpeCo" | - | DONE ELSEWHERE |
| #8077 C-GRPO adaptive group size (49-70% fewer rollouts) | AryaFayyazi (NeurIPS'26) | 0 | none (28 days); author asks "recipe or trainer option?" | none | UNANSWERED; author wants to implement |
| #8009 Score centering (bypass mode) | cr-gao | 0 | merged (`08b83872`, `7e987524`) | - | DONE |
| #5306 TorchTitan integration | acisseJZhong (Meta, maintainer) | 0/6 | self | `verl/workers/engine/torchtitan/` landed; PP, multimodal, critic open | VENDOR (Meta) |
| discussion #7273 NCCL-M2N transport | achidiac-nv (NVIDIA) | - | ISEEKYAN: FSDP raw-shard path first; wuxibin89: how are quantized (weight, scale) pairs handled; FT needs dynamic groups | #7432 #7552 #8087 #8088 | VENDOR-OWNED |

Pattern: an RFC gets a same-week reply from @wuxibin89 only when it touches rollout/weight
sync/low precision (his areas) or conflicts with an existing design (Mooncake, weight checker).
RFCs on trainer internals (optimizer steps, selected-token logits, replay buffer, per-turn
reward) sit unanswered for months unless @Begunner engages. Nothing is ever formally
"accepted"; acceptance is a merged PR.

## 4. Ranked unclaimed or stalled opportunities (speed, or novel-and-needed)

Ranking weights: maintainer demand (on #8186 or said by a maintainer) > measured speed > fit
for a local, opt-in, test-heavy PR that one owner can review > low collision with open PRs.
Sizes: S < 300 lines, M 300-1500, L > 1500. "Why unclaimed" is my read of the evidence.

| # | Item | Demand evidence | Claim status | Size | Risk | Why unclaimed |
|---|---|---|---|---|---|---|
| 1 | **Low-precision (FP8/MXFP8) inference-only forwards for ref / old_log_prob / teacher** on the FSDP engine: cast the frozen ref (and the no-grad old_log_prob pass) to the rollout's quantization format (same quantizer as #8287 export) so `ref` and `old_log_prob` phases run FP8 GEMMs and read 2x less weight memory; report mismatch KL vs bf16. | #8186 LP Common **P1** "low-precision inference-only forwards"; roadmap background: "Cheaper rollout and inference-only forwards (ref, teacher, old_log_prob): less memory, less memory traffic, faster GEMM"; the V1 step timer has separate `old_log_prob` / `ref` phases (`verl/trainer/ppo/v1/trainer_base.py:586-591`). | UNCLAIMED. No open PR matches; FSDP engine has only bf16 autocast (`verl/workers/engine/fsdp/transformer_impl.py:169,266,1580-1586`) and QAT fake-quant for training, not a real low-precision forward. | M-L | Needs Hopper/Blackwell (torchao float8 or TE); numerics: ref log-probs drift changes the KL term, must gate with mismatch metrics (#8025 landed the metric). | Nobody picked it: it is listed under wuxibin89's own "Common" column, and the obvious owners (NVIDIA) went for Megatron MXFP8 training first. |
| 2 | **Response-only LM-head projection on the FSDP engine** (project `lm_head` and run the fused log-prob/entropy kernel only on loss-bearing rows; restore layout afterwards). Megatron twin exists (#7686). | Speed for agentic / long-prompt RL: prompt + tool tokens pay a `[T, V=152k]` projection they never use. #7686 reports the Megatron gain on 8K/16K prompts; #8299's table shows prompt:response ratios of 3-16x in agentic workloads. FSDP path projects every token: rmpad branch `transformer_impl.py:1380-1450` (`logits_rmpad` over all `total_nnz` tokens), padded branch `:1494` (`logits = output.logits`); fused path `verl/models/transformers/dense_common.py:71-137` hands all `hidden_states` rows to `FusedLinearForPPO` and ignores the `logits_to_keep` argument it accepts (`dense_common.py:84`). | UNCLAIMED on FSDP: #7370 (Zhaoyi-Tian) was **closed by its author 2026-09-20 with zero maintainer reviews** ("assigned reviewers have had limited review activity"), not rejected. #7686 (Megatron) open, reviewed by tardis-key. | M | Must keep `no_padding_2_padding` row contract (`verl/workers/utils/losses.py:59`), entropy/`sum_pi_squared`/score-centering paths, Ulysses SP gather order; gains vanish when prompts are short. | Unglamorous and the first attempt died of review starvation; FSDP has no dedicated owner since vermouth1992 went quiet. |
| 3 | **NCCL communicator suspend/resume in colocated mode** (free 7-10 GB/rank of idle Megatron/FSDP NCCL buffers during rollout, 10 GB at EP=8; NCCL >= 2.29.7 native API). | Was a maintainer line item on the Q3 roadmap (#6985 "NCCL suspend/resume to release communicator memory"); RFC #6266 (2 +1) with measurements; ETOgaosion on PR #6408: "A tremendous release of memory!"; stable Dockerfiles already pin `nvidia-nccl-cu13>=2.29.7` "for ncclCommSuspend / ncclCommResume" (`docker/Dockerfile.stable.vllm:190-193`, `docker/Dockerfile.stable.sglang:112-115`). | STALLED: #6408 (xiefan46) last human activity 2026-06-05; wuxibin89 said "Hold until v0.8 release" (v0.8 shipped), then silence; blocker PR #6621 closed unmerged but the image fix landed anyway. No code on main (grep `ncclCommSuspend` = 0 hits). | M | Duplicating a living PR is bad form; the right move is a rebase of #6408's approach (Megatron `parallel_state` walk, engine `suspend_nccl_comms()` API) plus FSDP coverage, and offering it to xiefan46. Correctness: must not suspend checkpoint-engine groups (KIMI engine uses `dist.new_group`). | Review starvation; the author depends on docker/NCCL plumbing owned by ETOgaosion. |
| 4 | **Unified low-precision recipe config** (one typed `low_precision` block: train fmt + QAT + rollout fmt, expanding into the six existing keys, fail-fast on train/rollout mismatch). | #8186 LP Common **P1**, explicitly bulleted by wuxibin89 with the six keys; VikingDeng measured the silent BF16-train/FP8-rollout mismatch (`rollout_probs_diff_mean` 0.0058 -> 0.031, no warning). | UNCLAIMED: VikingDeng withdrew ("#8287 covers most ... I'll drop this item") but #8287 only unified rollout quantization + QAT path, not the cross-backend recipe block; wuxibin89 never said it was done. | M | Config churn touches `_generated_*.yaml`, `verl/workers/config/`, four engines; needs CPU tests only. Not a speed win by itself; it is the thing that makes items 1 and 5 usable. | Unglamorous plumbing; the one volunteer talked himself out of it. |
| 5 | **torchao float8 compute + float8 FSDP2 all-gather in verl's FSDP engine** (real FP8 GEMMs with FP32 master in the model, quantize-before-gather comm). | #8186 VeOmni **P1** "torchao low-precision compute: float8 (tensorwise/rowwise), MXFP8 on Blackwell; low-precision FSDP2 all-gather"; Kirrito-k423 asked for "FP8 for FSDP2" on #5972. Training speedup is the only route with "yes" in the roadmap's table. | UNCLAIMED: only dead WIPs #4325 (334d) and #1490 (533d). Roadmap text says it "is to come from torchao" under VeOmni, which wuxibin89/FoolPlayer own; verl's own FSDP engine is not mentioned, so an FSDP-engine implementation is open. | L | High: Hopper+ only, torchao version pinning, interaction with QAT fake-quant, LoRA, SP; convergence proof needed (train curves). May be seen as VeOmni-reserved. | Hard, needs GPUs for weeks, and the owners intend to do the VeOmni half themselves. |
| 6 | **Time-to-first-step: cut the fixed vLLM init + initial weight sync block** (~175-184 s for a 3B LoRA run per nataliekung's breakdown on #384; e.g. skip the initial sync when rollout loaded the same checkpoint, overlap engine init with trainer init, persist compile artifacts). | #384 is the only `speed`-labelled issue (609d, 2 +1, "known problem" per vermouth1992); maintainers keep touching it (#8167 tongyx361 "restore default vLLM compile caching"). | UNCLAIMED: nataliekung posted the profile (2026-08-13), no PR. | S-M | Correctness of skipping the first sync (LoRA, resume, `dummy` load_format); measurable only on real clusters. | Nobody owns "startup"; it is cross-cutting (Ray, vLLM, checkpoint engine). |
| 7 | **Ref-model cross-step prompt KV cache** (frozen ref: cache prompt prefill KV keyed by prompt hash; 13-16x ref speedup at 16K shared prompts, 0.9x on short GSM8K). | Agentic long-system-prompt workloads; same demand driver as #6401/#8299 (prefix sharing). | STALLED: #7292 (0hujun, "[wip]", NPU-tested, **0 reviewers**, last touch 2026-09-21 mass update). | M | Memory for cached KV; only pays with long shared prompts; must compose with prefix packing (#8300) rather than compete. | WIP quality, no CI, author never asked for review. |
| 8 | **FP8 optimizer states (Megatron) / MoE W4A8** | #8186 LP Megatron **P2**. | UNCLAIMED (only #6992 draft W4A8 *simulation* on FSDP, upd 36d). | S-M (optimizer states: expose Megatron-Core precision-aware optimizer + checkpoint tests) | Checkpoint format compatibility; P2 means low maintainer urgency. | Low priority, Megatron-only, needs Hopper. |
| 9 | **Sandbox recover on restart** (FT item: re-create tool sandboxes / env state when a rollout replica or agent loop restarts). | #8186 FT third bullet, unowned; #6583 production loss motivates FT. | UNCLAIMED, no RFC, no PR. | M | Scope undefined; depends on #8380's `verl/ft/` landing; sandbox-fusion specific. | Vague; everyone else is on the two trainer/rollout restart items. |
| 10 | **Selected-token log-probs under fused kernels** (`model_output["selected_token_logprobs"]` via `log p(id) = log p(label) + z_id - z_label`, no `[T,V]` materialization; 9.6 GB -> 0.9 GB at 6.8k tokens on H200). | RFC #7712 with H200 numbers; needed by verl-genrank / rerankers / calibration losses (Qwen3-Reranker style). | UNANSWERED by maintainers; eteluna "happy to open a draft PR", WenzheWang offered the kernel slice; no PR after 7 weeks. | S-M | Niche; `dense_common` only at first; must round `z` like each backend. | Maintainers have not replied, so the authors are waiting. Novel-and-needed for ranking RL, not speed. |
| 11 | **Workload-aware (predicted-length) rollout routing** (route by predicted outstanding tokens per replica, not in-flight count). | Q3 roadmap "workload awareness scheduling, e.g. ThunderAgent" (#6985), never done; #658 shows 2.3x rank-time imbalance. | Announced by mitukx on #6985 (2026-10-06) as a plan with no PR; adjacent PRs #6122, #2629, #6940. | M | Crowded router space; measurable only with a replay harness; wuxibin89 reviews LB changes closely. | Many half-finished routers; nobody shipped a benchmark. |
| 12 | **Reference throughput benchmark matrix** across [model, size, seqlen, hardware, role] with a reproducible harness (`tests/special_e2e` style) and a docs page. | #2388 line item never done; #3624 asks for "reference throughput benchmark ... for better disaggregated resource allocation"; #5389 model-support page (12 +1) is empty since 2026-04. | UNCLAIMED. | M (mostly scripts + docs) | Needs the hardware; numbers rot quickly. | Not code; nobody gets credit for it. |
| 13 | **Deterministic zero-advantage row skipping on the V1 trainer** (drop rows whose advantage is exactly 0 before `update_actor`; 1.7x `update_actor` at high accuracy in #5838). | #5838 measured -40.9% update_actor time; V1 already filters whole degenerate *groups* via DAPO filter (#7082, #7792), but not individual zero-advantage rows under GRPO when some groups are mixed. | STALLED: #5838 targets the V0 `dp_actor.py` path (deprecated), reviewed only by gemini, 0 human comments since April. A V1 version is unclaimed. | S | Changes optimizer-step accounting (interacts with #8099/#6025); KL/entropy terms on skipped rows are lost. | Written against the path the maintainers are deleting. |
| 14 | **Fail-fast detector for degenerate runs** (persistent zero advantage variance / zero grad_norm). | RFC #7405, 0 comments; saves whole wasted runs. | UNCLAIMED but trivially small; author (nataliekung) has 7 open PRs and may do it. | S | None technical; acceptance risk only. | Too small to be noticed. |

Not ranked but worth knowing: #8077 C-GRPO (49-70% fewer rollouts) is the largest "speed"
number on the tracker, but the author wants to build it and asked the maintainers two design
questions 28 days ago with no reply; a recipe in verl-recipe is the expected home.

### 4.1 Recommended top 3 for local implementation

1. **#2 response-only LM-head projection (FSDP engine)**: unclaimed after the author of #7370 gave
   up, measurable with one GPU, opt-in flag, contained in `verl/workers/engine/fsdp/` +
   `verl/models/transformers/dense_common.py` + `verl/utils/kernel/linear_cross_entropy.py`,
   and the Megatron twin (#7686) gives a reference design and a reviewer (tardis-key) who has
   already thought about it. Aim at wuxibin89; cite #7686 and #8299's prompt/response table.
2. **#1 FP8 inference-only forwards for ref/old_log_prob** behind the quantizer from #8287:
   it is the P1 item on the live roadmap that nobody took, it is pure speed, and it composes with
   #2 (fewer rows *and* cheaper GEMMs). Needs a Hopper box for the numbers; ship CPU tests for the
   config/expansion and an H100 table of `ref` / `old_log_prob` phase time + mismatch KL.
3. **#4 unified low-precision recipe config**: the cheapest P1 on the roadmap, zero GPU needed for
   tests, and it is the natural config surface for #1. Risk is only review bandwidth. If the
   human reviewers want fewer config changes, fold the minimal `fail fast on train/rollout fmt
   mismatch` check into #1 instead.

Runner-up: **#3 NCCL suspend/resume** has the biggest memory number (10 GB/rank) and explicit
maintainer enthusiasm, but a living PR exists; do it only as a rebase that credits #6408.

## 6. Things to avoid

Duplicates / already claimed (an open PR with an active author exists):
- Token-level interrupt + KV persistence across weight updates -> #8273 (cr-gao).
- Sample-level dispatch / streaming dataloader -> #8203 (Begunner, maintainer, BREAKING).
- Rollout Python runtime isolation -> #8407 (Levius-Fubuki) after RFC #8388.
- Replica restart / FT -> #8168 (tongyx361) + RFC #8380 (HaoWang113, ETOgaosion, tongyx361).
- NCCL M2N reshard / no-all-gather weight sync -> #7432 #7552 #8087 #8088 (NVIDIA), #8395
  sharded_nixl (rafiw), #7108 Mooncake P2P. Three vendors are competing here.
- Delta sync for quantized weights -> ChangyiYang (#7060 item 3, #7676).
- Batch-invariant training kernels -> #8026 (cr-gao); MXFP4 QAT recipe -> #8425/#8426 (0z5a);
  NVFP4 training -> #7971 (zhangyimi); low-precision comm (VeOmni) -> amenomori.
- Shared-prefix packing on FSDP -> #8300 (DeeproChoudhury, 8 days old, test-heavy). Prefix tree
  on Megatron -> #6689 (arvyanh, stalled) and TPR #7561. Do not open a third prefix-sharing PR;
  if we touch this area it must compose with #8300's `verl/workers/utils/prefix_packing.py` plan.
- KV-cache-aware / sticky-session load balancing -> #6940, #6533 (aoshen02 + wuxibin89 reviewed),
  #7435, #6122, #2629; workload-aware routing announced by mitukx (2026-10-06, no PR yet).
- External KV cache salting -> #8323 (daegyu94, wuxibin89/Begunner/aoshen02 already commenting).
- Context compression -> #5636 + verl-recipe#107 (ZhentaoFan, ArronHZG reviewing), and uni-agent.
- Stable optimizer-step counts / grouped mini-batches -> #6025 (guillemgt), #8201 (FZYsheep),
  eteluna's prototype (#8099). Weighted multi-output rows -> #7580 (dafu-wu).
- Crash-durable replay buffer -> savaresejeremy's 117-file branch (#7579); persistable replay
  buffer -> #2466 (old).
- C-GRPO -> AryaFayyazi wants to write the recipe (#8077). Per-turn rewards -> #7244 fork, #8305.

Areas the maintainers reserved for themselves (owner tags on #8186, or explicit statements):
- Modern Megatron integration (#8324, ISEEKYAN) and anything under `verl/models/mcore/` or
  `verl/workers/engine/megatron/` that changes forward/MTP/router-replay structure.
- DeepSeek-V4/V4.1 QAT and the FP4->FP8 lossless path (wuxibin89, FoolPlayer, ISEEKYAN).
- `verl/single_controller/` (zw0610; Runtime/Monarch #7840 approved) - wuxibin89: "confirm first
  with @zw0610".
- V1 trainer control flow (`verl/trainer/ppo/v1/trainer_base.py`, replay buffer) - Begunner is
  rewriting dispatch in #8203 [BREAKING]; any trainer-loop change will conflict.
- Removing the V0 trainer and moving `fully_async_policy` / `one_step_off_policy` to verl-recipe -
  housekeeping the maintainers listed for themselves; wuxibin89 closes V0-only fixes ("v0 trainer
  is deprecated and will be removed", #7349).
- TransferQueue data plane: anything that exposes a storage backend to the trainer directly is
  rejected (#7414: "hided behind TransferQueue").
- New config systems: vermouth1992 demanded a from-scratch unified config for V1 (#5400); do not
  add parallel config trees.

Design decisions that are still open (a PR here will wait on a maintainer call):
- in-tree vs TransferQueue for durable buffers (#7579); recipe vs trainer option for adaptive group
  size (#8077); whether verl wants a Hopper path for MXFP4 (0z5a's question on #8186, unanswered).
