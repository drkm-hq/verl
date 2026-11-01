# 02 — User-reported performance pain in verl (open + recently closed issues)

Scope: agent 2 of 10. Mines `gh/issues_open.json` (641 open issues, dumped 2026-10-28),
`gh/prs_open.json` (774 open PRs), `gh/prs_merged_recent.json` (400 PRs merged 2026-07-16 → 2026-10-28),
six closed-issue searches, and the read-only clone at `verl` (main @ `2ab35292`, 2026-10-28).
"Status on main" is checked against the **V1 trainer**, which is now the default path
(`trainer.use_v1: true`; `verl/trainer/ppo/v1/trainer_base.py` and siblings, worker
`verl/workers/engine_workers.py`, engines under `verl/workers/engine/`). The V0
`verl/trainer/ppo/ray_trainer.py` is slated for removal in the 26Q4 roadmap (#8186) and is cited only
where the code is shared (`core_algos.py`, `seqlen_balancing.py`, `metric_utils.py`, engines) or where a
V0-only finding is the only evidence. The old `verl/workers/actor/dp_actor.py` and `fsdp_workers.py`
referenced by many issues no longer exist on main.

All numbers below are quoted from issue threads (cited by `#number`) or from the clone (cited as
`path:line`). Anything inferred rather than measured is marked **(guess)**.

## 1. Scan method and top-60 table

### 1.1 Method

Script: `scan2.py` (temporary directory). For every open issue it builds a *prose* view of title + body +
comments: fenced code blocks are stripped, lines of the form `a.b.c=value` (Hydra CLI dumps) and
traceback lines are dropped. This matters: the naive scan (`scan.py`, first pass) matched 441/641
issues because config dumps contain `log_prob_micro_batch_size`, `param_offload`, `use_dynamic_bsz`
etc.; the prose-only scan matched 360/641 and ranked actual complaints first.

Vocabulary (regex, weight): slow/slowdown (3), throughput (3), tokens/s / tok/s / samples/s (3),
bottleneck (3), idle (3), bubble (3), GPU/SM utilization (2), long-tail / straggler / tail latency (4),
weight sync / update_weights / resharding / weight transfer (3), OOM / out of memory / memory peak /
leak / logits (2), rollout/generation time or "rollout takes" (3), log-prob + time/memory words within
80 chars (3), reward + time/serial words (3), dataloader slow (2), serialization / pickle / object
store / spill (3), ray overhead / dispatch overhead / single-controller overhead (3), scaling with
nodes/GPUs or "multi-node … no improvement" (2), MFU / TFLOPs (3), "takes N minutes|hours", "N s per
step", "hours per step", step time, wall-clock (3), padding waste (2), dynamic batch (1), seqlen /
load (im)balance (3), packing / shared prefix / prefix cache (2), async / agent loop / one-step-off /
disaggregated / colocated (1), partial rollout / oversample / replay buffer / abort (2), hang / stuck /
deadlock (1), wake-up / sleep / offload-onload with time words / startup / cold start / warm-up (2),
performance / speed / faster / efficiency / regression (1), multi-turn + slow/memory/retokenize (2),
CPU-bound / GIL / single-thread (3), NCCL/collective timeout (1). Backend tags (vllm, sglang,
megatron, fsdp, moe, lora, vlm, npu) carry weight 0 and only annotate.

Score = Σ weight × (3·min(title hits,2) + 1·min(body hits,3) + 0.3·min(comment hits,5))
      + 2.5·issue reactions + 0.5·comment reactions + 0.8·min(comments,15)
      + 4·e^(−days since update/150) + 2·e^(−days since creation/300).

Caveats discovered while reading threads:

* **Closed ≠ fixed.** An automated triage process closes any issue with no non-bot activity for 6 months (#3160, #1418, #1857
  all closed this way with the perf problem untouched). Closed-issue searches were therefore used only
  to find *additional evidence*, never to mark a theme resolved.
* The same bot also leaves long analytical comments on open issues; these are **not** maintainer
  confirmations. Only comments from `wuxibin89`, `vermouth1992`, `eric-haibin-lin`, `PeterSH6`,
  `ETOgaosion`, `yyDing1`, `sophiayyya`, `ISEEKYAN`, `zhaochenyang20`, `ArronHZG`, `HollowMan6`,
  `aoshen02`, `tardis-key`, `wucong25` were treated as maintainer responses.
* Roadmap/RFC issues (#3624 76 reactions, #2388 34, #5790 27, #708 24, #5755 16) dominate the raw
  score through reactions; they are demand signals for *directions*, not reports of a measured pain,
  and are analysed in section 4 (claimed work) rather than as themes.
* Reactions on verl issues are generally sparse (median 0); "+1 / same problem" comments are the
  stronger demand signal and were counted by hand per theme.

### 1.2 Top 60 by score (prose-only scan)

| rank | issue | score | reactions (issue / comments) | comments | created | updated | title | perf tags (title) |
|---|---|---|---|---|---|---|---|---|
| 1 | #3624 | 235.6 | 76 / 46 | 8 | 2025-09-25 | 2025-12-15 | verl next steps | weight-sync,partial-rollout,async/arch |
| 2 | #2388 | 111.2 | 34 / 5 | 3 | 2025-07-07 | 2026-07-10 | [roadmap] verl Q3 development | throughput,weight-sync,partial-rollout |
| 3 | #5365 | 93.8 | 0 / 0 | 10 | 2026-02-22 | 2026-08-05 | Grpo + Lora training in qwen3-vl-8b too slow. | slow,lora,vlm |
| 4 | #5790 | 91.4 | 27 / 2 | 9 | 2026-03-28 | 2026-07-13 | [RFC] Agent Abstractions and Trajectory Gateway for VERL | bottleneck,balance,packing |
| 5 | #708 | 90.3 | 24 / 8 | 14 | 2025-03-21 | 2026-07-03 | [Project] deepseek R1 infrastructure | moe |
| 6 | #6225 | 89.2 | 0 / 0 | 5 | 2026-04-30 | 2026-06-01 | [Feature Proposal] Colocated Checkpoint Engine for engine_workers Hybrid Weight  | weight-sync,async/arch |
| 7 | #5755 | 74.1 | 16 / 8 | 21 | 2026-03-26 | 2026-08-28 | [RFC] Multi-modal Generation RL 2026Q2 Roadmap | vlm |
| 8 | #7232 | 72.8 | 1 / 0 | 2 | 2026-08-03 | 2026-08-05 | [Megatron, perf] FP8 E2E is slower than BF16 on Qwen3-30B-A3B / 8xH200; is this  | slow,perf-generic,megatron,moe |
| 9 | #2618 | 68.3 | 23 / 0 | 1 | 2025-07-18 | 2026-07-12 | [agentic RL] multi-turn rollout and agent loop development tracking | async/arch |
| 10 | #1611 | 66.9 | 4 / 20 | 42 | 2025-05-21 | 2026-07-06 | [SGLang Async Rollout] CUDA error: an illegal memory access was encountered | async/arch,sglang |
| 11 | #3453 | 64.3 | 4 / 22 | 17 | 2025-09-11 | 2026-07-19 | Out Of Memory | memory |
| 12 | #7412 | 64.0 | 0 / 0 | 0 | 2026-08-14 | 2026-08-14 | repsonse_length_cliprate is so height | throughput,weight-sync,rollout-time |
| 13 | #747 | 61.7 | 12 / 8 | 20 | 2025-03-25 | 2026-07-03 | 训练会突然在某个步骤失败，模型某个指标会变成 nan，奖励变成0，然后开始不停重复生成感叹号 !!!!!!!!! | rollout-time,scaling,mfu |
| 14 | #6401 | 61.3 | 8 / 0 | 4 | 2026-05-19 | 2026-10-20 | [RFC] Prefix-Tree Shared Attention for Multi-Turn RL Training | memory,ray-overhead,takes-X-time |
| 15 | #658 | 59.9 | 3 / 9 | 17 | 2025-03-18 | 2026-07-01 | [Feature Request] Load Balancing for rollout phase | balance |
| 16 | #891 | 58.1 | 15 / 7 | 17 | 2025-04-03 | 2026-07-04 | [Potential Bug] KL compute in `low_var_kl (Cause KL NaN and !!!!!!!!!!!!!!! outp | wakeup/offload/startup |
| 17 | #7601 | 56.9 | 0 / 1 | 3 | 2026-08-29 | 2026-09-02 | [Feature Request] Add trainer and rollouter idle-ratio metrics to V1 separate_as | idle,async/arch |
| 18 | #6367 | 56.4 | 0 / 0 | 1 | 2026-05-15 | 2026-05-19 | Large host offloading overheads sometimes seen with Qwen3-30B-A3B on 1x DGXH100 | wakeup/offload/startup,moe |
| 19 | #3114 | 56.3 | 0 / 0 | 14 | 2025-08-18 | 2026-07-16 | [agent_loop, sglang, async] GPU utilization stays at 0% for a long time, trainin | slow,utilization,async/arch,sglang |
| 20 | #6468 | 55.5 | 1 / 0 | 4 | 2026-05-26 | 2026-07-13 | CPU memory keeps increasing during FSDP2 rollout weight sync and eventually trig | weight-sync,memory |
| 21 | #8299 | 53.2 | 1 / 0 | 0 | 2026-10-20 | 2026-10-20 | [RFC] Shared-prefix packing for the FSDP engine (trunk + branches), with a path  | packing,fsdp |
| 22 | #6010 | 52.8 | 0 / 0 | 5 | 2026-04-15 | 2026-05-09 | Extremely slow update_actor in GRPO with Qwen3-4B + FSDP2 on Ascend 910B | slow,npu |
| 23 | #316 | 52.2 | 0 / 0 | 17 | 2025-02-19 | 2026-06-22 | 【bug】Exception raised from c10_cuda_check_implementation at /opt/conda/conda-bld | memory,rollout-time,logprob-time |
| 24 | #384 | 50.4 | 2 / 2 | 7 | 2025-02-25 | 2026-08-13 | Startup times are slow | slow,wakeup/offload/startup |
| 25 | #5375 | 49.7 | 13 / 7 | 6 | 2026-02-23 | 2026-08-05 | [RFC] Support Context Compression into Agentic Training | rollout-time,async/arch,perf-generic |
| 26 | #8380 | 49.0 | 0 / 0 | 0 | 2026-10-26 | 2026-10-27 | [RFC] FT proposal for separate_async: independent trainer-side and rollout-side  | async/arch |
| 27 | #1147 | 48.1 | 13 / 3 | 11 | 2025-04-18 | 2026-08-07 | [feature] There is two types of multi-turn rollout (and training!), maybe verl o | packing,async/arch,perf-generic |
| 28 | #7579 | 45.9 | 0 / 0 | 2 | 2026-08-27 | 2026-08-30 | [RFC] A crash-durable replay buffer for the V1 trainer, behind TransferQueue | partial-rollout |
| 29 | #7904 | 45.7 | 0 / 0 | 1 | 2026-09-17 | 2026-09-29 | [rollout][vllm] Garbled multi-language output after weight sync when `free_cache | weight-sync,wakeup/offload/startup,vllm |
| 30 | #2106 | 44.9 | 1 / 11 | 16 | 2025-06-19 | 2026-07-09 | how to fix qwen3-8b 48k rl sglang backend: Timed out waiting 1800000ms for recv  | timeout,sglang |
| 31 | #8186 | 44.2 | 0 / 2 | 10 | 2026-10-10 | 2026-10-27 | [roadmap] verl 26Q4 roadmap | throughput,weight-sync,rollout-time |
| 32 | #2723 | 44.1 | 16 / 0 | 1 | 2025-07-24 | 2026-07-12 | [Bug] Incompatibility with `datasets` versions leads to `ValueError: Feature typ | serialization |
| 33 | #1524 | 44.0 | 3 / 45 | 11 | 2025-05-14 | 2026-10-13 | Some doubts about batch size | throughput,perf-generic |
| 34 | #3873 | 43.3 | 0 / 1 | 5 | 2025-10-23 | 2026-07-21 | Bug: DAPO training with Megatron+SGLang hangs at second rollout step due to NCCL | rollout-time,hang,timeout,sglang |
| 35 | #2450 | 42.6 | 3 / 6 | 12 | 2025-07-10 | 2026-07-10 | Qwen/Qwen2.5-VL-3B-Instruct on GRPO LORA OOM  千问vl3B 上OOM问题 | memory,lora,vlm |
| 36 | #6266 | 42.3 | 2 / 0 | 2 | 2026-05-07 | 2026-10-15 | [RFC] Support NCCL Suspend/Resume in verl Colocated Mode | async/arch |
| 37 | #5448 | 42.3 | 0 / 0 | 6 | 2026-03-01 | 2026-03-29 | Multi_node training doesn't improve training speed | perf-generic |
| 38 | #2803 | 42.0 | 4 / 31 | 18 | 2025-07-29 | 2026-07-13 | FSDP: CUDA failure 217 'peer access is not supported between these two devices'  | fsdp |
| 39 | #4318 | 41.6 | 5 / 5 | 19 | 2025-11-27 | 2026-07-24 | [RFC] Reward Loop | reward-time,balance,async/arch |
| 40 | #6383 | 40.5 | 0 / 0 | 4 | 2026-05-18 | 2026-07-07 | [RFC] KV-cache-aware request load balancer for rollout servers | balance |
| 41 | #7414 | 40.2 | 0 / 0 | 5 | 2026-08-14 | 2026-08-16 | [RFC] Add Mooncake as an Optional Rollout Data Backend for Verl V1 | throughput,serialization,partial-rollout |
| 42 | #571 | 40.2 | 10 / 8 | 4 | 2025-03-13 | 2026-06-28 | Problems related to the parameter settings of ulysses_sequence_parallel_size | memory,balance,dynamic-batch |
| 43 | #7495 | 40.1 | 0 / 0 | 3 | 2026-08-20 | 2026-08-21 | [BUG][LoRA][Async] Disaggregated weight sync never delivers the LoRA adapter — r | weight-sync,async/arch,lora |
| 44 | #4719 | 39.8 | 3 / 0 | 3 | 2025-12-29 | 2026-07-28 | 同参数下，verl无法复现easyr1的曲线 | weight-sync,scaling,balance |
| 45 | #1803 | 39.8 | 3 / 14 | 27 | 2025-06-02 | 2026-04-07 | Using FP8 inference in long response vllm rollouts  | vllm |
| 46 | #6691 | 39.4 | 0 / 1 | 2 | 2026-06-11 | 2026-06-16 | [Megatron][mbridge][MoE] Fused grouped-expert export (gate_up_proj ~8.6GiB) OOMs | weight-sync,megatron,moe |
| 47 | #3906 | 38.9 | 5 / 9 | 29 | 2025-10-25 | 2026-07-22 | The missing guide for training Qwen3-VL MOE | moe,vlm |
| 48 | #3258 | 38.8 | 7 / 12 | 27 | 2025-08-29 | 2026-02-12 | [BUG] Qwen3 MoE with FSDP2 meets `torch.utils.checkpoint.CheckpointError` when ` | moe |
| 49 | #5964 | 38.6 | 0 / 0 | 5 | 2026-04-10 | 2026-04-29 | Aarch64 GB200 container perf regression from vllm 0.17 to vllm 0.18 | perf-generic,vllm |
| 50 | #4065 | 38.4 | 1 / 10 | 16 | 2025-11-10 | 2026-07-23 | Bug when using `verl` with `sglang + LoRA` | sglang,lora |
| 51 | #340 | 38.2 | 0 / 2 | 12 | 2025-02-21 | 2026-06-22 | [Bug?] vLLMRollout.generate_sequences Randomly Hangs After 1-2 Steps When trying | memory,rollout-time,hang |
| 52 | #3293 | 37.6 | 1 / 0 | 11 | 2025-09-01 | 2026-07-18 | GPU memory usage continuously increased | memory |
| 53 | #3150 | 37.6 | 0 / 7 | 11 | 2025-08-20 | 2026-07-16 | [Error] Task failed: Can't pickle <function xxx>: it's not the same object as cu | serialization |
| 54 | #2325 | 36.8 | 0 / 3 | 15 | 2025-07-02 | 2026-07-09 | Megatron+sglang training occurs nccl timeout during rollout generation | timeout,sglang,megatron |
| 55 | #6537 | 36.7 | 0 / 0 | 0 | 2026-05-29 | 2026-05-29 | [RFC] Generic `RemoteBackend` abstraction for out-of-process RL backends | throughput,tokens/s,weight-sync |
| 56 | #5400 | 36.4 | 7 / 0 | 4 | 2026-02-25 | 2026-03-03 | [RFC] PPOTrainer with TransferQueue Integration | bottleneck,serialization,partial-rollout |
| 57 | #7060 | 36.1 | 0 / 2 | 7 | 2026-07-16 | 2026-10-13 | [Tracking] Sharded delta weight sync (delta_sharded): roadmap & known issues | weight-sync |
| 58 | #6967 | 36.0 | 0 / 0 | 5 | 2026-07-07 | 2026-07-11 | [fully_async] First NCCL checkpoint-engine group init hangs (timing race) — repr | hang |
| 59 | #6706 | 35.9 | 0 / 0 | 0 | 2026-06-12 | 2026-06-12 | qwen3-vl-4b lora 微调超级慢，是本身架构问题？qwen2.5-vl正常，可以用什么办法提速？几个小时跑一个step | lora,vlm |
| 60 | #7787 | 35.8 | 0 / 0 | 1 | 2026-09-08 | 2026-10-17 | [RFC] Delta weight sync on Ascend NPU (HCCL transport) | weight-sync,npu |

Issues outside the top 60 that were pulled in by targeted greps and read in full because they carry
measured numbers: #6946 (2.29× throughput from one constant), #7015 (weight-export profile), #7834
(host-RAM × rank count), #5282 (optimizer residency), #3996 (double log-prob), #336 (adv 10 751 s),
#4743, #5311, #6691, #8160, #7391, #7398, #5646, #5849, #2236, #292, #324, #6575.

Threads read with comments (local dump, all comments present): #5365 #6367 #3114 #6010 #384 #5448
#3413 #496 #4896 #6468 #7015 #658 #2106 #6383 #4264 #2236 #292 #324 #6575 #2551 #3453 #3293 #3902
#5125 #7834 #6366 #456 #5282 #7391 #5183 #6401 #8299 #7561 #7712 #3996 #6280 #336 #1524 #571 #7601
#6946 #7232 #5964 #5750 #6706 #7412 #6691 #5311 #7060 #6225 #8160 #5849 #4318 #5232 #6266 #522
#4743 #5111 #7495 #6726 #7398 #5646 #8186 #3624 #2388, plus closed #7029 #7848 #3160 #1418 #1208
#1857 via `gh issue view`.


## 2. Per-theme analysis

Theme order here is by evidence volume, not final rank (ranking is section 3). Demand counts are
issue reactions + hand-counted "+1 / same problem" comments across the theme's issues.

### Theme A — VLM "vision tax": the vision encoder is re-run on every duplicate image, in every phase

**Issues:** #5365 (10 comments, 3 independent reproductions), #6706, #5111 (10 comments), #4896,
#4743, #3906 (5 reactions, 29 comments), #7391 → PR #7400, closed-by-bot #1418.
**Demand:** ~8 reports across Qwen2.5-VL / Qwen3-VL-2B/4B/8B, GRPO/SAPO/PPO, FSDP + vLLM, LoRA and
full fine-tuning; no issue reactions (VLM users file new issues instead of reacting).

**Representative numbers (all from #5365):**

* Qwen3-VL-8B, GRPO + LoRA, `rollout.n` ≥ 4, vLLM 0.15.1 / torch 2.9, `use_remove_padding=False`:
  `timing_s/step` 7 704 s of which `gen` 21 s, `old_log_prob` 2 534 s, `ref` 2 539 s, `update_actor`
  2 597 s; `perf/mfu/actor` 0.016; `perf/throughput` 29.6 tok/s on the whole node.
* Independent reproduction (paul-org, SAPO + LoRA, n=8, 8 GPUs, transformers 4.57.3): the HF vision
  encoder took ~10 s per image, so `compute_log_prob` ran for hours. Two user-side patches:
  (1) encode only the unique images of a micro-batch and expand: 660 → 82 images per GPU, vision
  time 6 420 s → 154 s (42×); (2) cache the (frozen) vision embeddings from `compute_log_prob` and
  reuse them in `update_policy`.
* jianwenzh: Qwen3-VL-4B LoRA on geo3k is "clearly slower than full training" at commit 5b8986b
  (transformers 5.3.0, vLLM 0.18.0) — i.e. the slowdown is not tied to one old stack.
* #5111: Qwen3-VL-2B actor + critic PPO on 8×H200, 675 s/step with the critic update dominant;
  maintainer answer was configuration advice and a torch 2.9 `conv3d` regression
  (pytorch/pytorch#166122) — the duplicate-encoding cost was not discussed.

**Maintainer response:** none from a human maintainer on #5365; the bot comment (2026-08-05) agrees
that "the vision encoder is executed repeatedly in GRPO/SAPO + LoRA multi-stage" and lists the
dedupe/cache patches as the recommended direction. No PR exists.

**Status on main (V1 path):**

* `verl/models/transformers/qwen3_vl.py:216-218` calls `model.visual(pixel_values, grid_thw=…)`
  on every forward; identical pattern in `qwen2_vl.py:352-354`, `qwen3_5.py:484-486`,
  `glm4v.py:346-348`.
* The FSDP engine builds each micro-batch's vision inputs by concatenating the per-sample tensors
  (`verl/workers/engine/fsdp/transformer_impl.py:1180` → `extract_multi_modal_inputs`,
  `verl/utils/model.py:615-630`, `torch.cat(values, dim=0)` at `:630`), so the `n` GRPO copies of one
  image are encoded `n` times per micro-batch.
* The V1 step runs three actor/ref forwards per sample (`verl/trainer/ppo/v1/trainer_base.py:585-612`:
  old_log_prob → ref → update_actor), so each image is encoded **3·n** times per step
  (more with `ppo_epochs > 1`), and the encoder's deepstack outputs are recomputed each time.
* No dedupe or cache exists for the training path (`grep -rni "dedup\|vision.*cache" verl` only hits
  the rollout-side `qwen2_5_vl_dedup_image_tokens`, `verl/workers/rollout/utils.py:93`).
* `freeze_vision_tower` is declared (`verl/workers/config/actor.py:160`) but nothing consumes it on
  main; PR #8249 (2026-10-15) restores the freeze and skips the *dummy* visual forward
  (`qwen3_vl.py:278-287`) for frozen towers. It does not touch real images.
* **(guess)** #5111's claim that `use_remove_padding` is unimplemented for Qwen3-VL is stale:
  `monkey_patch.py:169-217` now slices `deepstack_visual_embeds` under Ulysses, implying an rmpad
  path exists.

**In-flight PRs:** #8249 (freeze + skip dummy forward), #8414 (pos-embed interpolation tensors
without Python lists, perf), #7323 (align Qwen3.5 vision calls across ranks), #6245 (direct vLLM
`MultiModalInput`), #7400 (drop image bytes after decode, fixes #7391), #5780 (TransferQueue for
image data), merged #7562 (offload multimodal dataset processing), merged #8038 (Megatron THD vision
token embedding). **None dedupes duplicate images or caches encoder outputs across phases.**

### Theme B — Weight sync / resharding: per-parameter FSDP2 export, host-memory growth, MoE export OOM

**Issues:** #7015 (profiled, unclaimed), #6468 (1 reaction, 4 comments, 2 independent measurements),
#4896 (5 comments), #5311 (RFC by a maintainer), #6691, #6225 (5 comments), #7060 (tracking, 7
comments), #7787, #7495 (3 comments), #496 (13 comments, 12× "same problem"), closed #7029, closed
and fixed #7848.
**Demand:** ~20 "+1"s, mostly on #496; the concrete engineering reports (#7015, #6468, #5311, #6691)
have 0–1 reactions but hard numbers.

**Representative numbers:**

* #7015 (ChangyiYang, 2026-07-10): FSDP2 `get_per_tensor_param` does one all-gather per parameter
  (~340 for a 7B model) plus one pageable H2D per parameter when CPU-offloaded. 2-node disaggregated
  Qwen2.5-7B, 8-GPU FSDP2 trainer with param+optimizer offload: full-weight sync 6.8–7.0 s with the
  old staging round trip, 3.4–4.1 s after #7005 skipped staging, of which the export generator is
  2.4–2.7 s — now the largest remaining chunk. Proposed fix: `FSDPModule.unshard()` per fully_shard
  unit (~30 fused collectives instead of ~340). Two people asked to take it (2026-07-19, 2026-08-26);
  no PR.
* #6468 (2026-05-26): host RSS grows every step during FSDP2 rollout weight sync until Ray kills a
  worker: ~0.6 GiB/step (Qwen3.5-2B), ~5.3 GiB/step (Qwen2.5-3B), 6.3 GiB/step (Qwen3-35B, cben484);
  top processes are `WorkerDict.actor_rollout_ref_update_weights`. Last maintainer comment
  (aoshen02, 2026-07-13) asks whether it reproduces on the latest image; unanswered.
* #4896: fully-async Qwen3-VL-8B, `timing_s/param_sync` 2–3 min early, ~25 min later because in-flight
  long-tail generations are not interrupted; maintainer (ArronHZG): "partial rollout has not taken
  full effect, will be resolved in the refactoring".
* #5311 (ISEEKYAN): DeepSeek-V3.1 weight transfer 120 s (128 GPUs, TP8/PP16/EP8) and 80 s (256 GPUs),
  target < 60 s; most time in mbridge one-by-one all_gather/broadcast.
* #6691: Qwen3.5-397B MoE on 128×H20, the fused `experts.gate_up_proj` export is one 8.6 GiB tensor
  that OOMs with 6 GiB free; wuxibin89 redirected to megatron-bridge (export unfused experts).
* #6225: Qwen reports 800+ s/step weight sync for 200B+ MoE colocated before batching; wuxibin89's
  own measurement with CUDA IPC and 2 GB buckets: Qwen3-VL-30B-A3B 5 s (32 GPUs), DeepSeek-V3.1 12 s
  (128 GPUs).
* #7060: `delta_sharded` (merged #6974) measured 7B 3.8 s vs 9.1 s, 32B 12.5 s vs 23.2 s, 72B 12.0 s
  vs 36.9 s against NCCL full broadcast (disaggregated, offload=True); refreshed matrix 2026-07-20.
* closed #7848 (v0.9.0 regression): an `aggressive_empty_cache(force_sync=True)` before resume raised
  median weight sync 2.7 s → 5.2 s (+91 %) and cut throughput 27.5 % on Qwen3-8B/8×H200; fixed by
  #7864 / #7873 (merged 2026-09-15).

**Status on main:**

* `verl/workers/engine/fsdp/transformer_impl.py:981-1057` (`get_per_tensor_param`) still yields
  `param.to(device, non_blocking=True).full_tensor()` per DTensor (`:1036`); the merged-LoRA path does
  the same (`:1086`). `_skip_staging` for FSDP2 (#7005) is present (`:993-996`). The export is
  consumed by `verl/workers/engine_workers.py:785-860` (`update_weights`, `:821/:845/:856`).
* Megatron host offload now reuses one pinned buffer per DDP buffer
  (`verl/utils/megatron_utils.py:460-520`), removing the 2× transient peak; nothing batches mbridge
  exports.
* Whether the #6468 host growth survives #7005 is **unverified**; the reporters were on 0.8.0.dev
  before that change.

**In-flight PRs:** #7552 NCCL M2N reshard backend, #8088 Megatron local shards via M2N, #8087
destination-local M2N weights, #7227 vLLM consumer for delta-sharded, #7324 TorchTitan delta sync,
#8272 delta sync on Ascend (HCCL), #5619 (draft) double-buffered bucket transfer, #7108 Mooncake P2P
checkpoint engine, #7685 dedupe Megatron host-offload replicas, #7496 sync LoRA adapters to
disaggregated engines (fixes #7495), #4876 (draft) TP-aware weight loading, merged #8352 (stage LoRA
buckets in one host block), merged #8315 (rl-insight traces weight-sync stages). **Nothing batches
the FSDP2 per-parameter export (#7015) or investigates the host growth (#6468).**

### Theme C — Rollout long tail, load imbalance, under-utilised generation, poor multi-node scaling

**Issues:** #658 (3 reactions; a comment with 5 reactions; 17 comments), #5448 (6 comments), #4264
(3× "same"), #3114 (14 comments, 6× "same"), #2106 (16 comments, reactions 3+4+2 on comments), #6383
(RFC → draft PR #6940), #6946 (measured), #7601, #4743, #2551, #5646, #1410 (3 reactions), #4791 (4
reactions), #1803 (3 reactions, 27 comments), #5559; closed-by-bot #3160 (3 reactions).
**Demand:** the largest cluster: ~25 "+1"s plus ~20 reactions.

**Representative numbers:**

* #658: with a base model, max-vs-min response tokens per vLLM worker differ ~30 %; per-rank rollout
  time 73 s / 119 s / 167 s (rank 3 2.3× rank 0) because the *max* sequence per worker, not the
  total, bounds the phase. (Same observation drives PR #2629: >1 800 s gap between DP ranks → NCCL
  timeouts.)
* #5448 (H200, vLLM async, FSDP2, prompt 13k/response 2k, n=8): `timing_s/gen` 647 s → 424 s → 324 s →
  314 s and step 984 s → 737 s → 587 s for 1 → 2 → 4 → 8 nodes; generation stops scaling at 4 nodes
  while `update_actor` keeps improving. Maintainer advice was to set `rollout.nnodes`; the reporter's
  follow-up (2026-03-29) is unanswered.
* #3114: `SingleTurnAgentLoop`, 1 024 responses, 8 workers, Qwen2.5-3B/7B + SGLang 0.4.6: 2 min 10 s
  per batch of which ≥ 2 min at 0 % GPU on all devices. Workaround found by users: `enforce_eager=True`
  (CUDA-graph lazy memory mapping stalls the stream). Still default `enforce_eager: False`
  (`verl/trainer/config/rollout/rollout.yaml:55`).
* #6946: fully-async, two replicas, small model: raising the hard-coded in-flight cap from 16 to 32
  samples per replica took end-to-end throughput 3 282 → 7 504 tok/s (2.29×).
* #4264: vLLM async server mode generates "significantly" slower than the removed SPMD sync mode for
  long responses (verl 0.6.1, vLLM 0.11); three users confirm; no maintainer root cause (the sync path
  was deleted in #4411).
* #4743: DAPO on 8×H100, TP=4: 2 000 s per batch, GPU utilisation < 50 % during rollout.
* #5646 (MI308): Ray-actor tool calls add large latency; a local thread pool was faster.

**Status on main:**

* Routing is still sticky + least-in-flight (`verl/workers/rollout/router.py:133-306`,
  `GlobalRequestLoadBalancer`); no token/KV awareness. The agent loop still splits a batch statically
  across workers (`verl/experimental/agent_loop/agent_loop.py:1278`, `prompts.chunk(len(workers))`).
* The fully-async concurrency cap is now configurable:
  `async_training.concurrent_samples_per_replica` (default 16,
  `verl/experimental/fully_async_policy/fully_async_rollouter.py:425`, used at `:503-506`) — #6946 is
  effectively addressed, undocumented in the issue.
* Partial rollout / staleness-aware sampling live in the V1 replay buffer
  (`verl/trainer/ppo/v1/replay_buffer.py:422-600`) and `fully_async_policy`; merged #7373 lends idle
  trainer GPUs to generation in `separate_async`, merged #7999 adds `ray_actor_max_concurrency`,
  merged #8397 fixes the submission gate.
* There is still no per-phase idle/utilisation metric for V1 (#7601; tardis-key points to RL-Insight
  timelines instead).

**In-flight PRs:** #6940 (draft) KV-cache-aware balancer, #5778 batched acquire/release + work-stealing
micro-batch queue (open since 2026-03-27), #6122 group-level balancer, #6533 (draft) un-stick sessions
under imbalance, #2629 request-skewness scheduler (recipe), #3326 (draft), #6342 suffix speculative
decoding, #5925 EAGLE/EAGLE3, #5509 eagle3, #5757 (draft) decoupled speculation, #6990 extra prefix
cache for agent system prompts, #7621/#7277 FlexKV offload, #7783 spec-decode metrics.

### Theme D — Trainer-side forwards are serial and partly redundant; driver round-trips in V1

**Issues:** #324 (5 reactions), #3996, #6280 (1 reaction, 6 comments with measurements), #336 (1
reaction, 3× "same"), #5282 (bot-endorsed, PR #7804), #456 (1 reaction, 9 comments), #571 (10
reactions), #1524 (3 reactions, 45 comment reactions), #7712, #6726, #6810, #5840, #5400 RFC (7
reactions), #7414, #6988.
**Demand:** ~30 reactions once #571/#1524 are counted; those two are *confusion about batch-size
semantics*, i.e. a usability symptom of the same pipeline.

**Representative numbers / facts:**

* #5365 again: `old_log_prob` (2 534 s) ≈ `ref` (2 539 s) ≈ `update_actor` (2 597 s). For a strictly
  on-policy run (`ppo_epochs=1`, one mini-batch) the old-log-prob forward is pure overhead (#3996); the
  `log_prob.detach()` shortcut that existed in `dp_actor.py` was removed with the legacy worker in
  #6067 (confirmed by kaining-never-stop on #6280, 2026-07-31).
* #6280: with `log_prob_micro_batch_size_per_gpu = 2 × ppo_micro_batch_size_per_gpu` (the documented
  recommendation) `old_log_prob ≠ log_prob` even on-policy (`pg_clipfrac` non-zero); GuoCheng24's
  single-GPU harness shows batch-shape bf16 noise pushing 30 %+ of tokens outside a ±10 % ratio band on
  some architectures; MoE (Qwen3.5-35B-A3B) is worse than dense.
* #336: `timing_s/adv` 10 751 s with > 10k-token outputs (2025-02, V0, two "same").
* #456 (vermouth1992): memory spikes come from cross-entropy / entropy in the backward; fix was
  "lower `ppo_max_token_len_per_gpu`".
* #5282: optimizer states are resident for the whole `update_actor` context, not just around
  `optimizer.step()`.

**Status on main (V1):**

* `verl/trainer/ppo/v1/trainer_base.py:563-621` (`_step_once`) is strictly sequential on the driver:
  gen → reward (colocated RM only) → `_balance_batch` → `old_log_prob` → `ref` → `values` → `adv` →
  `update_critic` → `update_actor`. No overlap of ref/critic/reward with anything (#324 still true).
* `old_log_prob` is recomputed unless `algorithm.rollout_correction.bypass_mode` (default false;
  `trainer_base.py:1642-1655`); no on-policy skip (#3996 still true).
* Each of `_compute_old_log_prob` (`:1665-1672`), `_compute_ref_log_prob` (`:1714-1718`),
  `_compute_values` (`:1736-1740`), `_compute_advantage` (`:1747`, 7 fields) and `_compute_metrics`
  (`:1880-1895`, 9 fields incl. prompts/responses) does a full-batch `tq.kv_batch_get` on the driver
  and most write back with `kv_batch_put`. The worker already returns results through `tqbridge`
  (`verl/single_controller/base/decorator.py:425`); the driver pulls token-level `log_probs` +
  `entropy` only to rename them to `old_log_probs`/`ref_log_prob` and to compute one scalar entropy
  metric. **(guess)** For an 8 192-sample × 8k-token batch this is ~2–3 GB of fp32 token fields per step
  crossing the driver; no issue quantifies it yet, but #5400 (7 reactions) and open PR #5958 ("data
  transport time log") show the community suspects it.
* Entropy/logits memory knobs exist and default off: `entropy_checkpointing`,
  `entropy_from_logits_with_chunking` (`verl/trainer/config/_generated_ppo_trainer.yaml:42-45`),
  used at `transformer_impl.py:1416-1420`; the full-vocab `logits_rmpad` is still materialised and
  `full_tensor()`-gathered under TP (`:1403`).
* Optimizer load/offload is still per training context (`verl/workers/engine/base.py:322-342`),
  not per step (#5282 still true).

**In-flight PRs:** #7804 offload optimizer states around each update (for #5282), #8365 chunked
`sum_pi_squared`, #7733 chunked top-k KL (Megatron), #7582 honour entropy chunking without rmpad,
#6519 no grad for entropy/KL when coef = 0, #5838 skip zero-advantage responses (perf), #6025 grouped
mini-batching, #7944 in-place temperature scaling, #8419 opt-in FP32 logits, #7292 (WIP) ref-model
cross-step prompt KV cache, #8300 shared-prefix packing for FSDP (2.7× step on 2k-token prompts,
3.1× actor, 3.9× ref), #6689 (draft) prefix-tree MAGI attention, #5990 standalone log-prob server,
#7281 avoid scalar-tensor dispatch in batch balancing. **Nothing skips the on-policy old-log-prob
forward or moves the V1 rename/metric work off the driver.**

### Theme E — Start-up latency (time to first step)

**Issues:** #384 (label `speed`, 2 reactions + 2 on a comment, 7 comments, maintainer "known
problem" 2025-02), #6967, #7381, #7383, closed-by-bot #3406.
**Numbers:** 3.7 min to step 1 for Qwen2-7B single node, 6.7 min on 4 nodes (2025-02). nataliekung
(2026-08-13, verl 0.8.0.dev0, vLLM 0.11, H200, GRPO + LoRA, `free_cache_engine=true`): "model ready →
first weight sync ready" is **175–184 s for a ~3B LoRA model and 115–133 s for 0.5B**, identical on
cold and warm nodes, i.e. a fixed cost of vLLM engine init + initial weight sync; no sub-milestones
exist to attribute it further.
**Status on main:** V1 `init_workers` is sequential (`trainer_base.py:330-348`: actor `init_model`,
then ref, then reward loop) and the first `update_weights` happens in `on_init_end`
(`trainer_sync.py:31-33`). Rollout defaults `load_format: dummy` (`rollout.yaml:108`), so the block is
engine construction + CUDA-graph capture + first sync, not disk I/O **(guess)**. No stage timers
inside that window.
**In-flight:** #7045 fail fast on hung group init (#6967), #7170/#7382 fail fast on bad topologies,
#8168 replica lifecycle APIs. Nothing targets the time itself.

### Theme F — Memory that forces smaller batches: host RAM at load, NCCL buffers, offload churn, logits

**Issues:** #3453 (4 reactions + a 22-reaction comment, 17 comments, 14× "same"), #7834, #5125 (1
reaction), #6367, #6366, #6266 (2 reactions), #456, #5840, #6810, #3293 (fixed by #6699), #4002,
#5183, #7391, closed #6953, #6698.
**Numbers:** #3453: Ray host OOM at 250 GB / 251.5 GB with 8×A6000 + SGLang, `TaskRunner.run` the
biggest process; "every FSDP process loads the model". #7834: with `tie_word_embeddings: true` every
rank materialises the full checkpoint on host (30B/A3B, 61 GB bf16 → OOM-killed before any GPU use on
2×H100/503 GB). #6367: Megatron Qwen3-30B-A3B on 1×DGX-H100, 3 of 5 runs spend 160–175 s of
`update_actor` in host memory management (`cuMemCreate/Map/SetAccess/Unmap/Release`) = 45–51 % of a
340–360 s step; the other 2 runs 16–17 s. #6266: idle NCCL communicators hold 7–10 GB per rank in
colocated Megatron 3D+EP MoE. #6366: 10–13 GiB resident after `sleep_replicas` with everything
offloaded.
**Status on main:** `transformer_impl.py:271` still passes
`use_meta_tensor=not hf_config.tie_word_embeddings` (→ `fsdp_utils.py:64-70` materialises on every
rank); `rl_dataset.py:340-341` still keeps image bytes next to the decoded PIL image; Megatron offload
keeps one pinned buffer (`megatron_utils.py:460-520`) but the cuMem churn of #6367 has no diagnosis.
**In-flight:** #7685 dedupe Megatron host-offload replicas, #5651 offload fp32 master params, #6408
NCCL communicator suspend/resume (Megatron), #7400 image bytes, #7881 pinned snapshot leak in
decoupled PPO, merged #7770 OOM snapshots, #7961 memray profiler. **Nothing fixes tied-embedding
loading.**

### Theme G — Reward computation serial (largely solved)

**Issues:** #2236 (2 reactions), #292 (3), #324 (5), #4318 RFC (5 reactions, 19 comments), #6575,
#8011, #4417. **Status on main:** Reward Loop (`verl/experimental/reward_loop`, `reward.num_workers`,
`limited` manager) is default; the agent loop streams reward per sample
(`agent_loop.py:1000-1057`, `trainer_base.py:579-581` only runs the colocated RM path when no reward
loop workers exist). Remaining gap: a colocated reward *model* still runs as a serial phase (#6575).
Open PRs #3548 / #2120 / #1871 / #1693 / #1206 are pre-Reward-Loop designs and superseded.

### Theme H — Not actionable or expected

* FP8 E2E slower than BF16 on Qwen3-30B-A3B (#7232, 1 reaction): maintainer (sophiayyya) says
  expected for small-GEMM MoE; quantizer parity is a P0 on #8186.
* vLLM 0.17 → 0.18 GB200 regression (#5964): reporter retracted (re-test showed no regression).
* NCCL timeouts in `compute_log_prob` with dynamic bsz (#5750, #2325, #3873): fixed by #5591 and the
  legacy-worker removal (#6067); Megatron report on 0.7.0 unverified.
* Ascend `update_actor` 800 s/step (#6010, 2× "same"): hardware-specific, no breakdown.

## 3. Ranked themes (demand × likelihood a focused contribution fixes it × measurability)

Scores are 1–5 judgments. "Claim" = whether an open PR already implements the obvious fix.

| rank | theme | demand | fix likelihood | measurability | claim status |
|---|---|---|---|---|---|
| 1 | A. VLM vision-encoder dedupe + cross-phase embedding cache | 4 | 5 | 5 | unclaimed (#8249 only freezes) |
| 2 | D1. Skip the on-policy `old_log_prob` forward in V1 | 3 | 4 | 5 | unclaimed (#3119 shortcut was deleted in #6067) |
| 3 | B1. Batch the FSDP2 weight export (`unshard()` per unit) + verify host growth | 3 | 4 | 5 | unclaimed since 2026-07 (#7015) |
| 4 | C1. Per-phase GPU idle/utilisation metrics for V1 (+ long-tail attribution) | 5 | 4 | 4 (the metric is the deliverable) | unclaimed (#7601, #3114 prototype only) |
| 5 | F1. Rank-0-only loading for tied-embedding models | 3 | 5 | 5 | unclaimed (#7834) |
| 6 | E. Start-up stage timers + overlap engine init with trainer init | 2 | 3 | 5 | unclaimed (#384) |
| 7 | D2. Keep V1 rename/metric work on the workers (no driver token-field round trips) | 1 (no report yet) | 4 | 3 | unclaimed |
| 8 | F2. Megatron host-offload cuMem churn | 1 | 2 | 5 | unclaimed (#6367) |
| — | C2. Token/KV-aware routing, partial rollout, speculative decoding | 5 | 2 | 3 | crowded: #6940, #5778, #6122, #2629, #6342, #5925 |
| — | D3. Optimizer offload around `step()` | 2 | 4 | 5 | claimed: PR #7804 |
| — | Prefix sharing (trunk + branches) | 4 | 4 | 5 | claimed: PR #8300 (FSDP), #6689 (Megatron) |

### 3.1 Theme A — VLM vision-encoder dedupe (rank 1)

**What a contribution looks like.** Two independent, exact optimisations in the FSDP engine's
multimodal path, both default-off behind `actor_rollout_ref.model.vision_dedupe=True` (name
illustrative):

1. *Within a micro-batch*: give every image a stable id (the prompt `uid` plus image index, carried in
   `multi_modal_inputs` as a non-tensor field next to `image_grid_thw`), and in the model wrappers
   (`qwen3_vl.py:216-218`, `qwen2_vl.py:352-354`, `qwen3_5.py:484-486`, `glm4v.py:346-348`) run
   `model.visual` only on the unique images, then `index_select` the embeddings (and deepstack
   embeddings) back to every occurrence before `masked_scatter`. Exact by construction.
2. *Across phases*: when the tower is frozen (PR #8249's `freeze_vision_tower`, or LoRA that excludes
   `visual.*`), keep a bounded per-step GPU/CPU cache keyed by image id in the engine
   (`transformer_impl.py` micro-batch prep around `:1180/:1339`) so old_log_prob → ref → update_actor
   encode each image once per step instead of three times. Must be invalidated when the tower is
   trainable (or cache only in the forward-only phases).

Because `_balance_batch` (`trainer_base.py:1609-1633`) scatters a `uid`'s `n` samples across dp ranks
and micro-batches, (1) alone recovers only the duplicates that land together; (2) recovers the rest.
A follow-up can group same-`uid` samples when packing (`seqlen_balancing.py`), which is also what PR
#8300's prefix packing needs; #8300 is text-only today, so this does not collide with it.

**Benchmark that would show it.** Qwen3-VL-4B (and 8B + LoRA), geo3k, `rollout.n=8`, 8×H100/H200,
FSDP2 + vLLM, `freeze_vision_tower=True`: report `timing_s/old_log_prob`, `timing_s/ref`,
`timing_s/update_actor`, `timing_s/step`, `perf/throughput`, `perf/max_memory_allocated_gb` and the
reward curve for off/on. The user-side prototype on #5365 measured the encoder part 6 420 s → 154 s
with n=8; the honest expectation for the whole step is "the three trainer phases stop being the
bottleneck" (≈ 3·n fewer encoder passes), to be measured rather than promised.

### 3.2 Theme D1 — on-policy old-log-prob skip (rank 2)

**What a contribution looks like.** In V1, when `ppo_epochs == 1` and the actor mini-batch covers the
whole batch (`ppo_mini_batch_size × rollout.n == train_batch_size × rollout.n`) and no rollout
correction is configured, `_compute_old_log_prob` (`trainer_base.py:1635-1694`) is a forward whose
only consumer is `ppo_loss` reading `data["old_log_probs"]` (`verl/workers/utils/losses.py:107`) —
and under those conditions `old_log_prob == log_prob` is the definition of on-policy. The change:
skip the phase, pass an `on_policy=True` flag in `extra_info`, and have `ppo_loss` use
`log_prob.detach()` as `old_log_prob` (ratio = 1, `pg_clipfrac` = 0, which #6280 shows is what users
*expect* and today do not get because batch-shape noise makes the two forwards disagree). Entropy
metrics come from `update_actor`'s forward (`calculate_entropy` already plumbed,
`trainer_base.py:1832-1834`). Keep the recompute whenever `rollout_correction`, `calculate_log_probs`
debug metrics, decoupled PPO (`trainer_separate_async.py:201-222`) or `ppo_epochs > 1` need π_old.

**Benchmark.** Qwen3-4B GRPO on GSM8K/DAPO-17k, 8 GPUs, on-policy config: `timing_s/old_log_prob`
goes to ~0 and `timing_s/step` drops by that amount (in #5365's VLM run it was 33 % of the step; for
text GRPO a forward is roughly ⅓ of `update_actor`, so ~10–15 % of the step **(guess)**); reward and
loss curves overlay the baseline; `actor/pg_clipfrac` becomes exactly 0.

### 3.3 Theme B1 — batched FSDP2 weight export (rank 3)

**What a contribution looks like.** Replace the per-parameter generator in
`transformer_impl.py:1030-1038` with a walk over `fully_shard` units: `unit.unshard()` (one fused
foreach all-gather, FSDP-managed H2D staging under `CPUOffloadPolicy`), yield that unit's parameters
with reconstructed FQNs through `convert_weight_keys`/`unfuse_moe_params`, then `unit.reshard()`,
so at most one layer is materialised at a time (option 1 of #7015). Keep the per-parameter path for
LoRA (`collect_lora_params`) and the QAT quantizer. Pair it with a reproduction of #6468 (host RSS per
step during sync on 2B/3B models) so the PR either shows the growth is gone after #7005 or fixes it
in the same export path.

**Benchmark.** Qwen2.5-7B, FSDP2 trainer with param+optimizer offload → vLLM, colocated and 2-node
disaggregated: `timing_s/update_weights` (reference 3.4–4.1 s with a 2.4–2.7 s export on #7015's
setup) plus a per-phase breakdown (export / gather / pack / broadcast), host RSS before/after each
sync over 50 steps, and a bitwise weight-equality check against the current path.

### 3.4 Theme C1 — idle/utilisation observability (rank 4)

**What a contribution looks like.** An opt-in NVML sampler (`verl/utils/profiler/`) that polls
`gpu_util` / memory per device at ~1 Hz on every worker, aggregated per `marked_timer` phase into
`perf/gpu_util/<phase>/{mean,min_rank}` and, for V1 async trainers, `trainer/idle_ratio` and
`rollouter/idle_ratio` (the contract hsusul and mikequan0425 sketched on #7601: integrate per-replica
in-flight counts over the step window; exclude hybrid-engine rollout time from trainer idle). Backend
agnostic, no engine hooks. This is the thing every thread in theme C lacked: #3114 found its stall by
printing timestamps; #5448 and #4743 could not say which rank or phase was idle.

**Benchmark.** It *is* the measurement; validate on a deliberately imbalanced run (long-tail prompts,
`rollout.n=16`) that the metric shows flat 0 % during a CUDA-graph stall (#3114's signature) versus
partial idleness during long tail, and that overhead is < 1 % of step time.

### 3.5 Theme F1 — tied-embedding loading (rank 5)

**What a contribution looks like.** `transformer_impl.py:270-272` passes
`use_meta_tensor=not tie_word_embeddings`; make meta init unconditional, load the full state dict on
rank 0 only (as the untied path already does), broadcast, then re-tie `lm_head.weight` to
`embed_tokens.weight` after `fsdp2_load_full_state_dict` (the FSDP2 wrap helper at
`fsdp_utils.py:540-561` already refuses to wrap the two separately when tied). #7834 documents the
naive fix that silently unties the model; the test is that `lm_head.weight.data_ptr() ==
embed_tokens.weight.data_ptr()` after load and that logits match the untied path.

**Benchmark.** Host RSS per rank during `init_model` for a tied 30B/A3B (North-Mini-Code) or Qwen3-4B
on 8 GPUs: today N× checkpoint size, after 1×; plus time-to-first-step.

### 3.6 Lower-ranked but noted

* **E (start-up).** First step: stage sub-timers inside "model ready → first weight sync ready"
  (actor `init_model`, vLLM `EngineCore` init, CUDA-graph capture, first `update_weights`) logged as
  `timing_s/startup/*`; second step: launch rollout replicas concurrently with trainer `init_model`
  (`trainer_base.py:330-348` is sequential). Metric: time to step 1 on #384's 3B-LoRA setup
  (175–184 s today).
* **D2 (driver round trips).** Move the `old_log_probs`/`ref_log_prob`/`values` renames and the
  entropy metric into `engine_workers.py:749-771` outputs so the driver never `kv_batch_get`s
  token-level fields except in `_compute_advantage`; measure with PR #5958's transport timers.
  Demand is inferred, not reported — second-tier.
* **F2 (Megatron offload churn).** Needs an nsys reproduction of #6367 first; plausible knobs are the
  allocator (`expandable_segments`) and pinned-buffer reuse for optimizer/grad buffers, but that is a
  guess.

## 4. Already fixed, claimed, or expected — do not duplicate

| pain | status | evidence |
|---|---|---|
| Serial reward computation (#2236, #292, #324 half) | fixed: Reward Loop default, `reward.num_workers`, `limited` manager, remote reward manager | #4318, PR #4752, `agent_loop.py:1000-1057`, `trainer_base.py:579-581` |
| GPU memory grows per micro-batch in actor update (#3293, #6696/#6698) | fixed | PR #6699 merged 2026-06-12 |
| Weight-sync regression from aggressive `empty_cache` (#7848) | fixed | PRs #7864, #7873 merged 2026-09-15 |
| Per-tensor CUDA-IPC overhead in SGLang sync (#7029) | fixed (bucketed IPC) | wuxibin89 on #7029; merged #8352 stages LoRA buckets in one host block |
| Fully-async in-flight cap hard-coded at 16/replica (#6946) | fixed | `async_training.concurrent_samples_per_replica`, `fully_async_rollouter.py:425` |
| NCCL timeout in `compute_log_prob` with dynamic bsz (#5750) | fixed for FSDP | PR #5591; legacy worker removed in #6067 |
| Full-weight sync too slow for disaggregated rollout | shipped: `delta_sharded` 1.9–3.1× | #6974, #7144, #7085, #7181; follow-ups #7552, #8088, #8087, #7227, #7324, #8272 in flight |
| LoRA adapter never synced to disaggregated engines (#7495) | claimed | PR #7496 |
| Image bytes retained after decode (#7391) | claimed | PR #7400 |
| Optimizer states resident during whole update (#5282) | claimed | PR #7804 |
| Shared-prefix compute in GRPO groups (#6401, #7561, #8299) | claimed | PR #8300 (FSDP, 2.7× step on 2k prompts), #6689 draft (Megatron MAGI); PrefixGrouper #4368 merged earlier |
| KV-cache-aware rollout routing (#6383) | claimed (draft) | PR #6940 |
| Agent-loop RPC batching / work stealing | claimed | PR #5778 (open since 2026-03) |
| `freeze_vision_tower` no-op + dummy visual forward (#3496, #7833) | claimed | PR #8249 |
| Idle trainer GPUs during generation in separate_async | fixed | PR #7373 merged 2026-08-21 |
| MoE load-balance metrics (#5232) | fixed (rollout side) | PR #6853 |
| TransferQueue storage units not cluster-size aware (#7398) | claimed | PR #8340 |
| Megatron host-offload replica duplication | claimed | PR #7685; 2× pinned peak already fixed on main (`megatron_utils.py:460-520`) |
| NCCL communicator memory in colocated mode (#6266) | claimed | PR #6408 |
| FP8 E2E not faster than BF16 on small MoE (#7232) | expected per maintainer; parity is 26Q4 P0 | #8287, #8306, #8308 merged |
| vLLM 0.17 → 0.18 GB200 regression (#5964) | withdrawn by reporter | #5964 comment 2026-04-29 |
| Gradient sync during accumulation | fixed | PR #7095 merged |
| Batch balancing scalar dispatch (#7279) | claimed | PR #7281 |
| Partial rollout / streaming for fully-async long tail (#4896, #658) | claimed by maintainers ("will be resolved in the refactoring", ArronHZG 2026-02-02); V1 replay buffer + `fully_async_policy` carry it | `replay_buffer.py:422-600` |

Things the maintainers appear to want handled upstream rather than in verl: fused grouped-expert
export OOM (#6691 → megatron-bridge), torch 2.9 `conv3d` slowdown for Qwen3-VL (#5111 → pytorch),
vLLM sleep/wake accounting on Ascend (#3902 → vllm-ascend).
