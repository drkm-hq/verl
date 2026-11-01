# 07. Open-PR territory and merge dynamics (verl-project/verl)

Agent slice 7. Snapshot: `gh/prs_open.json` (774 open PRs, created 2025-02-11 .. 2026-10-29T01:02Z),
`gh/prs_merged_recent.json` (400 merged PRs, merged 2026-07-16 .. 2026-10-28), `gh/issues_open.json` (641),
clone at `2ab35292` (2026-10-28). Ages are days before 2026-10-29 02:00Z. Scripts: `research/scripts-a7/`
(`cluster.py`, `contested.py`, `perf.py`, `perf_table.py`, `enrich_graphql.py`, `dynamics.py`, `stale.py`, `dupcheck.py`); raw outputs `out_*.json` next to them.

**Caveat on `updatedAt`:** 372 open PRs carry `updatedAt=2026-09-21` and 178 carry `2026-10-24` (bulk relabel/
rebase-bot events), so "last update" from the dump is not a staleness signal. Staleness below uses the last
commit date on the PR head (GraphQL `commits(last:1)`), fetched for all 774 PRs.

## 1. Clusters of the 774 open PRs

Headline: 93 drafts; median age 126 d; 266 PRs (34%) are older than 240 days and are effectively dead
weight; 158 PRs (20%) are younger than 30 days and are the real competition. Size: median 152 lines changed,
p90 1,562 lines; median 3 files. 213 titles do not pass `check_pr_title.py` (no bracket module tag, a
non-allowed tag such as `wip`, `lora`, `npu`, `docs`, or a missing `type:`); those PRs cannot merge as-is.

### 1a. By title module tag (a `[a, b]` title counts in both)

| tag | open | drafts | median age d | newest 3 |
|---|---|---|---|---|
| trainer | 132 | 9 | 82 | #8434 #8424 #8413 |
| rollout | 121 | 21 | 74 | #8435 #8416 #8415 |
| (untagged / invalid) | 119 | 15 | 325 | #8431 #8430 #8429 |
| fsdp | 77 | 13 | 64 | #8419 #8359 #8355 |
| vllm | 62 | 15 | 46 | #8418 #8417 #8416 |
| megatron | 60 | 9 | 61 | #8426 #8425 #8404 |
| algo | 52 | 4 | 109 | #8424 #8422 #8421 |
| ckpt | 50 | 3 | 52 | #8395 #8355 #8316 |
| doc | 39 | 8 | 55 | #8432 #8426 #8422 |
| model | 37 | 6 | 57 | #8417 #8414 #8353 |
| worker | 32 | 2 | 225 | #8201 #8176 #7924 |
| data | 27 | 1 | 93 | #8420 #8221 #8197 |
| perf | 25 | 2 | 83 | #8425 #8365 #8331 |
| cfg | 25 | 4 | 40 | #8340 #8335 #8323 |
| recipe | 23 | 4 | 398 | #8403 #7749 #7720 |
| training_utils | 22 | 1 | 142 | #8378 #8377 #8277 |
| sglang | 20 | 5 | 70 | #8435 #8387 #8363 |
| misc | 20 | 0 | 116 | #7870 #7689 #7661 |
| ci | 19 | 5 | 40 | #8395 #8339 #8328 |
| reward | 19 | 1 | 60 | #8379 #8368 #8305 |
| tool | 17 | 0 | 250 | #8156 #8153 #8084 |
| ray | 16 | 0 | 175 | #8407 #7910 #7600 |
| veomni | 12 | 5 | 24 | #8419 #8418 #8417 |
| fully_async | 11 | 0 | 111 | #8356 #8341 #7496 |
| hardware | 10 | 1 | 56 | #8432 #8404 #8401 |
| single_controller | 9 | 1 | 93 | #8411 #8092 #7840 |
| one_step_off | 0 | - | - | - |
| trtllm | 3 | 0 | 212 | #8323 #5800 #5391 |

Type tags: fix 290, feat 266, refactor 11, chore 9, test 7, none/invalid 191.

Age histogram (created): 0-7 d 67 · 8-30 d 91 · 31-60 d 96 · 61-120 d 124 · 121-240 d 130 · >240 d 266.

Top open authors: tongyx361 16, aoshen02 11, YeonwooSung 8, kahlun 7, nataliekung 7, theely 6,
madScientist10 6, dependabot 5, flyingdoubleG 5, Li-Jicheng 5, ISEEKYAN 5, ewan0x79 5, ETOgaosion 5. 511 distinct
authors: the queue is a long tail of one-PR external contributors.

### 1b. By top-level path touched (a PR counts in every bucket it touches)

| path bucket | open PRs | drafts | median age d |
|---|---|---|---|
| tests/ | 417 | 43 | 69 |
| verl/trainer (other: sft, config yaml, main_*) | 218 | 31 | 128 |
| verl/utils (other) | 186 | 30 | 103 |
| verl/trainer/ppo (ray_trainer, core_algos, ...) | 166 | 17 | 193 |
| docs/ | 142 | 19 | 48 |
| verl/workers/config | 123 | 19 | 89 |
| verl/workers/rollout/vllm_rollout | 98 | 21 | 76 |
| examples/ | 86 | 18 | 217 |
| verl/models | 75 | 12 | 75 |
| verl/experimental/agent_loop | 67 | 7 | 176 |
| verl/workers/engine/fsdp | 66 | 12 | 55 |
| verl/workers/engine (other: engine_workers.py, base.py) | 64 | 14 | 60 |
| .github/ | 63 | 12 | 74 |
| recipe/ | 62 | 10 | 409 |
| verl/trainer/ppo/v1 | 60 | 3 | 50 |
| verl/workers/rollout (other: replica.py, llm_server.py) | 60 | 15 | 76 |
| verl/utils/dataset | 51 | 5 | 335 |
| verl/experimental (other) | 50 | 10 | 84 |
| verl/workers/rollout/sglang_rollout | 45 | 10 | 198 |
| verl/workers/fsdp_workers.py (deleted on main, #6067) | 44 | 5 | 418 |
| verl/workers/engine/megatron | 36 | 7 | 56 |
| verl/single_controller | 33 | 6 | 146 |
| verl/checkpoint_engine | 33 | 3 | 56 |
| verl/experimental/fully_async_policy | 31 | 4 | 149 |
| verl/utils/megatron | 30 | 2 | 85 |
| verl/utils/checkpoint | 26 | 3 | 93 |
| verl/utils/reward_score | 24 | 0 | 305 |
| verl/workers/actor (deleted on main, #6067) | 22 | 3 | 338 |
| verl/workers/engine/veomni | 18 | 6 | 24 |
| verl/utils/torch_functional.py | 18 | 2 | 113 |
| verl/workers/utils | 17 | 2 | 62 |
| verl/workers/reward_manager | 16 | 0 | 507 |
| verl/workers/megatron_workers.py (deleted on main, #6067) | 14 | 2 | 315 |
| verl/protocol.py | 11 | 1 | 356 |
| verl/utils/seqlen_balancing.py | 10 | 1 | 92 |
| verl/utils/kernel | 8 | 2 | 64 |
| verl/workers/sharding_manager (deleted on main, #6067) | 8 | 2 | 476 |
| verl/workers/critic (deleted on main, #6067) | 8 | 0 | 398 |

Reading: the live code paths (`trainer/ppo/v1`, `workers/engine/*`, `checkpoint_engine`, `workers/rollout/vllm_rollout`)
have young medians (50-76 d); the V0 worker files (`fsdp_workers.py`, `workers/actor`, `workers/critic`, `sharding_manager`,
`megatron_workers.py`) no longer exist on main (removed in #6067), so the 69 open PRs against them (medians 315-476 d) are
dead on arrival. `ray_trainer.py` (V0) is still present but slated for removal in the 26Q4 roadmap (#8186).

Full per-cluster PR lists (number, age, draft flag, title) are in `scripts-a7/out_clusters_by_tag.json` and
`out_clusters_by_path.json`.

## 2. Contested files (open PRs touching each hot-path file)

Two facts shape this map (per the lead and verified in the clone):

- **The V0 worker files are gone from main.** `verl/workers/actor/dp_actor.py`, `verl/workers/critic/dp_critic.py`,
  `verl/workers/fsdp_workers.py`, `verl/workers/megatron_workers.py` and `verl/workers/sharding_manager/` were removed in
  #6067; `ls` in the clone confirms they are missing. **69 open PRs still touch them** (median age 380 d, median last commit
  371 d, zero with a commit in the last 30 d). They are dead on arrival, not competition; only their *ideas* matter for a
  duplicate check (see section 5).
- **The V1 trainer is the default** (`trainer.use_v1: true`, `verl/trainer/config/ppo_trainer.yaml:232`), and the 26Q4
  roadmap issue #8186 lists "Remove V0 trainer `verl/trainer/ppo/ray_trainer.py`" and "Deprecate
  `verl/experimental/fully_async_policy|one_step_off_policy` to verl-recipe". PRs are therefore counted separately against
  V1 files, and `ray_trainer.py` traffic is split into V0-only versus V0+V1.

"Active" below = head commit within the last 30 days (GraphQL `commits(last:1)`), the honest contention signal given the
bulk-touched `updatedAt`. `*` = draft. Per-file lists with author and dates: `scripts-a7/out_contested.json`.

### 2a. V0 versus V1 trainer

| group | open PRs | drafts | median age | median last commit | active (<=30 d commit) |
|---|---|---|---|---|---|
| touch any `verl/trainer/ppo/v1/*` | 60 | 3 | 50 d | 35 d | 28 |
| touch `ray_trainer.py` (V0) at all | 92 | 11 | 243 d | 230 d | 15 |
| V0-only (ray_trainer, no v1 file) | 70 | 11 | 327 d | 327 d | 4 (#8204, #7330, #8084, #8071*) |
| V0 + V1 (both) | 22 | 0 | 50 d | 29 d | 11 |
| V1-only | 38 | 3 | 51 d | 37 d | 17 |
| touch deleted V0 worker files | 69 | 9 | 380 d | 371 d | 0 |

So the 92 on `ray_trainer.py` are mostly a graveyard; the live trainer contention is ~28 PRs on V1 files.

| V1 file | open PRs | drafts | active | active PRs |
|---|---|---|---|---|
| verl/trainer/ppo/v1/trainer_base.py | 51 | 3 | 23 | #8379 #8364 #8335* #8327 #8310 #8305* #8303 #8300 #8290 #8249 #8203 #8201 #8118 #8117 #8116 #8114 #8113 #8112 #8111 #8092 #7497 #7240 #6025 |
| verl/trainer/ppo/v1/agent_loop_tq.py | 16 | 3 | 6 | #8335* #8310 #8305* #8203 #8113 #8112 |
| verl/trainer/ppo/v1/utils.py | 9 | 0 | 5 | #8310 #8203 #8116 #8100 #7497 |
| verl/trainer/ppo/v1/trainer_separate_async.py | 7 | 1 | 2 | #8290 #8203 |
| verl/trainer/ppo/v1/trainer_colocate_async.py | 5 | 0 | 3 | #8290 #8203 #8116 |
| verl/trainer/ppo/v1/replay_buffer.py | 5 | 1 | 2 | #8203 #8201 |
| verl/trainer/ppo/v1/trainer_sync.py | 4 | 0 | 1 | #8290 |
| verl/trainer/ppo/v1/{streaming_dataloader,tq_schema}.py | 1 each | 0 | 1 | #8203 |
| verl/trainer/ppo/v1/{trainer_remote_backend,weight_update_hooks,agent_loop_tq_rollout,replay_buffer_session,trainer_fully_async}.py | 1 each | 0 | 0 | quiet |

Note #8203 (16 d, streaming/TQ work) and #8290 (profiling windows) each touch 4-8 V1 files at once; #8201 (`[BREAKING]`
balance V1 mini-batches by UID) touches `trainer_base.py` + `replay_buffer.py` + `engine_workers.py`.

### 2b. Hot-path files (lead list + brief), with activity

| file | open PRs | active (<=30 d commit) | active PRs |
|---|---|---|---|
| verl/workers/rollout/vllm_rollout/vllm_async_server.py | 71 | 24 | #8398 #8366 #8353 #8323 #8290 #8289 #8273 #8272 #8253 #8246* #8228 #8169 #8168 #8158 #8121* #8118 #8117 #8110 #8105* #8046 #8032 #7870 #7484 #7192* |
| verl/workers/engine/fsdp/transformer_impl.py | 62 | 24 | #8359 #8353 #8328* #8309 #8300 #8297* #8272 #8257* #8249 #8246* #8206* #8198 #8194* #8189 #8176 #8121* #8097* #8088 #8087 #8083 #8057 #8026 #7870 #7552 |
| verl/workers/engine_workers.py | 47 | 19 | #8395 #8353 #8327 #8300 #8297* #8257* #8249 #8246* #8201 #8194* #8121* #8116 #8110 #8088 #8087 #8032 #7552 #7484 #6025 |
| verl/workers/rollout/vllm_rollout/utils.py | 30 | 16 | #8395 #8366 #8339* #8328* #8327 #8272 #8257* #8253 #8246* #8228 #8110 #8106* #8105* #8088 #8087 #7484 |
| verl/trainer/ppo/ray_trainer.py (V0, slated for removal) | 92 | 14 | #8327 #8310 #8303 #8300 #8290 #8249 #8204 #8116 #8092 #8084 #8071* #7497 #7330 #7240 |
| verl/workers/rollout/llm_server.py | 25 | 13 | #8356 #8341 #8290 #8289 #8273 #8169 #8168 #8118 #8117 #8112 #8095 #8070* #8046 |
| verl/checkpoint_engine/base.py | 26 | 12 | #8402* #8395 #8355 #8323 #8273 #8168 #8110 #8088 #8087 #7840 #7552 #7045 |
| verl/workers/engine/megatron/transformer_impl.py | 34 | 11 | #8395 #8383 #8327 #8199 #8176 #8088 #8074 #7686 #7685 #7484 #7192* |
| verl/experimental/agent_loop/agent_loop.py | 51 | 8 | #8368 #8335* #8313 #8305* #8101 #8085 #7840 #7343 |
| verl/workers/rollout/vllm_rollout/vllm_rollout.py | 23 | 8 | #8395 #8273 #8272 #8246* #8168 #8110 #8088 #8087 |
| verl/workers/engine/base.py | 15 | 7 | #8297* #8176 #8116 #8088 #8087 #7870 #7552 |
| verl/trainer/ppo/core_algos.py | 45 | 5 | #8386 #8349 #8176 #8054 #7497 |
| verl/workers/rollout/sglang_rollout/async_sglang_server.py | 30 | 5 | #8387* #8363 #8323 #8121* #8032 |
| verl/utils/seqlen_balancing.py | 10 | 4 | #8300 #8257* #8137 #7901 |
| verl/utils/torch_functional.py | 18 | 4 | #8377 #8365 #8121* #8026 |
| verl/experimental/agent_loop/tool_agent_loop.py | 18 | 4 | #8305* #8273 #8169 #8156 |
| verl/workers/utils/padding.py | 10 | 3 | #8300 #8125 #7901 |
| verl/workers/utils/losses.py | 7 | 2 | #8176 #8026 |
| verl/workers/rollout/sglang_rollout/sglang_rollout.py | 24 | 2 | #8327 #8032 (+ #8435, 0 d) |
| verl/checkpoint_engine/nccl_checkpoint_engine.py | 7 | 2 | #7870 #7840 |
| verl/checkpoint_engine/nixl_checkpoint_engine.py | 6 | 2 | #8316 #7840 |
| verl/utils/kernel/kernels.py | 3 | 2 | #8378 #7686 |
| verl/utils/kernel/linear_cross_entropy.py | 3 | 2 | #8378 #8097* |
| verl/workers/rollout/replica.py | 18 | 1 | #8323 |
| verl/workers/engine/fsdp/fsdp_turbo_impl.py | 2 | 1 | #8251 |
| verl/protocol.py | 11 | 1 | #7901 |
| verl/experimental/fully_async_policy/fully_async_rollouter.py | 13 | 1 | #7840 (the 7,043-line Runtime refactor) |
| verl/experimental/fully_async_policy/fully_async_trainer.py | 11 | 1 | #7840 |
| verl/utils/kernel/fp8_kernel.py | 1 | 0 | - |
| verl/workers/engine/fsdp/utils.py, verl/workers/engine/megatron/utils.py | 1 each | 0 | - |
| verl/experimental/fully_async_policy/message_queue.py, verl/experimental/agent_loop/utils.py | 0 | 0 | quiet |

Config churn: `verl/trainer/config/_generated_ppo_trainer.yaml` 90 PRs (28 active), `_generated_ppo_megatron_trainer.yaml`
80, `_generated_ppo_veomni_trainer.yaml` 64, `_generated_ppo_torchtitan_trainer.yaml` 62, `ppo_trainer.yaml` 50,
`rollout/rollout.yaml` 47, `verl/workers/config/rollout.py` 52 (13 active), `verl/workers/config/engine.py` 25 (6 active).
These are regenerated by the `autogen-trainer-cfg` pre-commit hook, so conflicts are mechanical but universal for any PR
that adds a config field.

Interpretation for merge-conflict and reviewer-saturation risk:

- **Saturated (>= 10 active PRs):** `vllm_async_server.py` (24), `engine/fsdp/transformer_impl.py` (24), `engine_workers.py`
  (19), `vllm_rollout/utils.py` (16), `v1/trainer_base.py` (23), `llm_server.py` (13), `checkpoint_engine/base.py` (12),
  `engine/megatron/transformer_impl.py` (11). A PR here rebases weekly and competes for the same CODEOWNERS
  (vllm_rollout: wuxibin89/PeterSH6/chenhaiq/ArronHZG/HollowMan6; engine: eric-haibin-lin/vermouth1992/ZihengJiang/ArronHZG;
  trainer: eric-haibin-lin/vermouth1992/tongyx361/PeterSH6).
- **Busy but structured (5-9 active):** `agent_loop.py` (8), `vllm_rollout.py` (8), `engine/base.py` (7), `core_algos.py` (5,
  all algorithm additions), `async_sglang_server.py` (5).
- **Quiet hot paths (<= 4 active):** `seqlen_balancing.py` (4: #8300 prefix packing, #8257* draft, #8137 KK cache, #7901),
  `torch_functional.py` (4), `workers/utils/padding.py` (3), `workers/utils/losses.py` (2), `utils/kernel/*` (2 each),
  `checkpoint_engine/nccl_*` and `nixl_*` (2 each, one of them the #7840 mega-refactor), `protocol.py` (1), `replica.py` (1),
  `fsdp_turbo_impl.py` (1), `v1/replay_buffer.py` (2), `v1/trainer_separate_async.py` (2), `fully_async_policy/*` (1, and
  slated for deprecation to verl-recipe per #8186, so not a place to invest).

## 3. Open PRs that claim a speedup or memory reduction

Method: `scripts-a7/perf.py` flags a PR when its title carries `perf`, or a perf word in the title
(speed/faster/throughput/overhead/memory/OOM/fuse/overlap/cache/zero-copy/vectorize/latency/idle/bubble/prefetch/offload/
peak), or >= 4 perf words in the body (137 hits); the table keeps the 66 with a real perf/memory intent (full raw list
in `out_perf.json`). Review/CI columns come from the GraphQL enrichment of all 774 open PRs (`out_open_enriched.json`):
"reviews (human)" lists non-bot reviewers other than the author; "CI" is `statusCheckRollup` of the head commit
(note: fork PRs show SUCCESS once the always-on sanity jobs pass; GPU e2e jobs only run after a maintainer approves them).

Headline claims in flight (numbers as stated by the authors, none independently verified):

- **#8300** shared-prefix packing for the FSDP engine (8 d, 2,043 lines): GRPO Qwen3-4B n=8 on 2xH100, training step
  **2.7x faster** with a 2k-token prompt, 1.18x on GSM8K; single-GPU fwd+bwd 1.06x @64-token prompt, 2.18x @2048, 3.81x @8192.
  Off by default (`fsdp_config.use_prefix_packing`). Only bot reviews so far.
- **#8137** cache Karmarkar-Karp `State.spread` in `seqlen_balancing.py` (21 d, 308 lines): CPU balance call 1.60-2.68x faster;
  one non-maintainer APPROVED review (koriyoshi2041), still REVIEW_REQUIRED.
- **#7281** avoid scalar-tensor dispatch in batch balancing (83 d, 89 lines): balance time -65..-70%, 22-27x on the
  partition step; zero comments, zero reviews, UNKNOWN mergeability.
- **#7097** reuse jagged rows across dynamic batches (`rearrange_micro_batches`) (101 d, 217 lines): 2.1-2.9x on the
  micro-batch build; CONFLICTING, fork CI never approved ("41 fork-triggered workflows `action_required`").
- **#7096** defer scalar metric materialization (one stacked D2H transfer instead of `.item()` per metric per micro-batch)
  (101 d, 418 lines): 2.3-4.6x on the metric path; CONFLICTING, no human review.
- **#8365** chunked `calculate_sum_pi_squared_from_logits` (3 d, 43 lines): peak 9.27 GiB -> 2.32 GiB at 8192x152k; MERGEABLE, CI green.
- **#8331** Triton fast path for FP8_BLOCK export in compressed-tensors quantizer (6 d, 222 lines): per-tensor quantize
  0.278 ms -> 0.032 ms; `update_weights` 14.6-15.2 s vs 4.4 s BF16 today on Qwen3-30B-A3B (8xH20).
- **#8435** reduce tensor-lifetime overlap in SGLang bucketed weight sync (0 d, 205 lines): up to 50% less transient host memory.
- **#7804** offload optimizer states around each update (49 d, 439 lines): actor peak -15.8% (14.69 -> 12.37 GiB), +9.6%
  actor time. wuxibin89 replied "use FSDP2 CPUOffloadPolicy"; author argued the distinction; no review since 2026-09-09.
- **#7686** Megatron LM head projected only for response tokens (56 d, 576 lines): actor-update time -7 to -15%, memory -11..-15%.
- **#7685** dedupe host parameter-offload replicas on Megatron (56 d, 595 lines): host memory -50..-75% for offload copies.
- **#7733** chunked top-k KL (54 d, 187 lines): -28.6..-42.8% peak at long N. **#7809** fused linear top-k log-probs (1,958 lines, CI FAILURE).
- **#5838** skip zero-advantage responses in the actor update (210 d, 1,142 lines): 1.7-2.5x `update_actor` at high accuracy;
  CONFLICTING, no human review in 7 months. Touches legacy `dp_actor.py` + `ray_trainer.py`.
- **#5778** agent-loop batch acquire/release RPCs + work-stealing + `empty_cache` dedup (215 d, 1,412 lines): no numbers; CONFLICTING.
- **#6990** extra-prefix-cache for agentic rollouts via LMCache (111 d, 1,308 lines): no numbers; CONFLICTING.
- **#7292** `[wip]` ref-model cross-step prompt KV cache (83 d): 13x on a 16K prompt, 0.9x on GSM8K; invalid title tag.
- **#6832** enable fused kernels by default (127 d, 19 lines): 32x logits-memory reduction claim; pending, Luosuu commented.
- **#7108** P2P checkpoint engine via Mooncake (99 d, 3,418 lines, "78x faster" weight sync); **#5619** double-buffered bucketed
  weight transfer (226 d, draft, 2,012 lines); **#5976** trainer-side FP8 weight quantization (199 d, wuxibin89 reviewed) - all CONFLICTING.
- Rollout load balancing family: **#6940** KV-cache-aware router (draft, 3,581 lines), **#6122** group-level balancer
  (-15..-53% tail), **#6533** sticky-session rebalancing (aoshen02, collaborator, draft), **#2629** request-skewness scheduler
  (466 d, eric-haibin-lin reviewed). All conflicting; the live `GlobalRequestLoadBalancer` superseded parts of them.

Pattern: of the 66, **0 are APPROVED by a maintainer**, 1 has CHANGES_REQUESTED (#1295, 548 d), 13 have had any human
reviewer at all, 40 are CONFLICTING. Perf PRs from external authors sit with bot reviews only; the ones that moved had a
collaborator author (tongyx361, aoshen02, ji-huazhong) or a wuxibin89 reply.

| PR | age / last commit | author (assoc) | size | claim (from body) | reviews (human) | CI | mergeable |
|---|---|---|---|---|---|---|---|
| #8435 [rollout, sglang] fix: reduce tensor lifetime overlap during bucketed weight sync | 0 d / 0 d | Levius-Fubuki (NONE) | 205 l / 3 f, tests y | 128 MiB; 256 MiB; 768 MiB; 640 MiB | no decision; 1 comments | SUCCESS | MERGEABLE |
| #8425 [megatron, perf] fix: fake-quantize all MXFP4 expert weights of a grouped linear in one call | 0 d / 0 d | 0z5a (CONTRIBUTOR) | 216 l / 2 f, tests y | 9×; 2 ×; 4×; 2 × | no decision; 1 comments | SUCCESS | MERGEABLE |
| #8365 [perf, algo] feat: compute sum_pi_squared in row chunks to cut peak memory | 3 d / 2 d | flyingdoubleG (NONE) | 43 l / 2 f, tests y | 24 GB; 4096 ×; 4.64 GiB; 2.32 GiB | no decision; 1 comments | SUCCESS | MERGEABLE |
| #8378 [training_utils] fix: keep confident-label logprobs and gradients in the Triton fused linear cr | 3 d / 1 d | catherinelee274 (NONE) | 298 l / 3 f, tests y | 64×; 64×; 3 × | no decision; 2 comments | SUCCESS | MERGEABLE |
| #8331 [perf, rollout] feat: fused Triton fast path for FP8_BLOCK weight export in the compressed-tens | 6 d / 6 d | BeichenHuang (NONE) | 222 l / 3 f, tests y | 768x; 8x; 64x; 128x | no decision; 2 comments | SUCCESS | MERGEABLE |
| #8338 [megatron] fix memory management in overrided recompute backward | 6 d / 2 d | HydrogenSulfate (NONE) | 241 l / 3 f, tests y | 20%; 40%; 60%; 80% | no decision; 3 comments | SUCCESS | MERGEABLE |
| #8300 [fsdp, perf, trainer] feat: shared-prefix packing for the FSDP engine | 8 d / 6 d | DeeproChoudhury (NONE) | 2043 l / 29 f, tests y | 2x; 2.7x faster; 1.18x faster; 3.8x | no decision; 3 comments | SUCCESS | MERGEABLE |
| #8289 [1/3][perf, reward, rollout] feat: support profiling independent Ray actors | 8 d / 8 d | jiefangxuanyan (CONTRIBUTOR) | 2725 l / 31 f, tests y | none stated | no decision; 2 comments | SUCCESS | MERGEABLE |
| #8290 [2/3][perf, trainer] fix: count and forward worker profiling windows | 8 d / 9 d | jiefangxuanyan (CONTRIBUTOR) | 3237 l / 45 f, tests y | none stated | no decision; 3 comments | SUCCESS | CONFLICTING |
| #8206 (draft) [algo, fsdp, perf] refactor: share support normalization in score centering | 16 d / 16 d | tongyx361 (COLLABORATOR) | 1075 l / 9 f, tests y | 0.350%; 0.373%; 80GB; 0% | ; 1 comments | SUCCESS | MERGEABLE |
| #8137 [training_utils, perf] refactor: cache Karmarkar-Karp state spread | 21 d / 21 d | IsaacLi74 (NONE) | 308 l / 4 f, tests y | 2.13x; 1.33x; 2.68x; 1.34x | no decision, reviewed by koriyoshi2041; 1 comments | SUCCESS | MERGEABLE |
| #8116 [perf, trainer] feat: report opt-in update phase timing and lifecycle metrics | 24 d / 20 d | tongyx361 (COLLABORATOR) | 532 l / 22 f, tests y | none stated | no decision; 0 comments | FAILURE | UNKNOWN |
| #8109 [veomni, perf] fix: stage exported weights one tensor at a time | 24 d / 20 d | tongyx361 (COLLABORATOR) | 135 l / 2 f, tests y | none stated | no decision; 0 comments | FAILURE | UNKNOWN |
| #8097 (draft) [fsdp, model] fix: gather DTensor LM heads for fused kernels | 26 d / 26 d | kolehma8 (CONTRIBUTOR) | 285 l / 9 f, tests y | 2 x | no decision; 0 comments | SUCCESS | UNKNOWN |
| #8096 [fsdp, model] fix: run fused LM heads through FSDP2 hooks | 26 d / 8 d | cyanseek (NONE) | 500 l / 9 f, tests y | 2×; 2×; 2× | no decision; 1 comments | SUCCESS | UNKNOWN |
| #8074 [megatron] feat: adapt 1f1b EP overlap in model engine | 28 d / 28 d | ji-huazhong (COLLABORATOR) | 417 l / 2 f, tests n | none stated | no decision; 1 comments | FAILURE | CONFLICTING |
| #8057 [fsdp] feat: make FSDP2 CPU offload pin_memory configurable | 29 d / 20 d | Longwt123 (NONE) | 47 l / 7 f, tests y | 1 GiB; 800 GiB; 200 GiB; 8× | no decision; 1 comments | FAILURE | UNKNOWN |
| #7910 [ckpt, ray, trainer] feat: add opt-in sub-second fault tolerance with dynamic NCCL re-rendezvou | 41 d / 41 d | sunnylovestiramisu (NONE) | 165 l / 6 f, tests n | 100%; 24x; 80GB; 942.3 MB | no decision; 1 comments | SUCCESS | UNKNOWN |
| #7826 (draft) [megatron] fix: keep fused log-prob kernels available with MTP on bshd | 48 d / 48 d | AliceChenyy (NONE) | 135 l / 4 f, tests n | 30 GB; 96 GB; 0.962 GB; 7.62 GB | no decision; 3 comments | PENDING | CONFLICTING |
| #7832 [perf, tests] feat: add cohere2_moe MFU estimation and Ulysses coverage | 48 d / 48 d | inin-zou (NONE) | 144 l / 3 f, tests y | 8x | no decision; 1 comments | SUCCESS | UNKNOWN |
| #7804 [fsdp, perf] feat: offload optimizer states around each update | 49 d / 50 d | yue-zeng-yue (CONTRIBUTOR) | 439 l / 9 f, tests y | 2×; 80GB; 14.693 GiB; 12.372 GiB | no decision; 2 comments | SUCCESS | CONFLICTING |
| #7809 [megatron] feat: fuse linear top-k log probabilities | 49 d / 47 d | HSYZhang (CONTRIBUTOR) | 1958 l / 10 f, tests y | none stated | no decision; 0 comments | FAILURE | CONFLICTING |
| #7733 [megatron] feat: add memory-efficient chunked top-k KL computation | 54 d / 37 d | HSYZhang (CONTRIBUTOR) | 187 l / 1 f, tests n | 13.28 GiB; 13.29 GiB; 0.01 GiB; 0.1% | no decision; 2 comments | FAILURE | CONFLICTING |
| #7685 [megatron] feat: deduplicate host parameter-offload replicas | 56 d / 16 d | qiangyupei (NONE) | 595 l / 9 f, tests y | 32 GiB; 75.0%; 50.0%; 53.8% | no decision, reviewed by wucong25; 2 comments | PENDING | CONFLICTING |
| #7686 [megatron] feat: project LM head only for response tokens | 56 d / 16 d | qiangyupei (NONE) | 576 l / 9 f, tests y | 100%; 7.94%; 15.96%; 7.07% | no decision, reviewed by tardis-key; 2 comments | PENDING | CONFLICTING |
| #7621 [rollout] FlexKV KV cache offload for dynamic resource | 58 d / 56 d | DylanChen-NV (NONE) | 1211 l / 21 f, tests n | none stated | no decision; 2 comments | PENDING | CONFLICTING |
| #7524 [megatron] fix: balance R2 padding for peak-memory reduction | 66 d / 4 d | EazyReal (CONTRIBUTOR) | 214 l / 4 f, tests y | none stated | no decision; 5 comments | SUCCESS | MERGEABLE |
| #7415 (draft) [rollout, trainer] feat: add pluggable rollout data backends | 75 d / 75 d | zxpdemonio (NONE) | 2680 l / 25 f, tests y | 12.44 MiB; 3.11 MiB; 12.44 MiB; 1.50x | no decision; 1 comments | PENDING | UNKNOWN |
| #7281 [trainer, perf] refactor: avoid scalar tensor dispatch in batch balancing | 83 d / 83 d | ai-yang (NONE) | 89 l / 5 f, tests y | 22.26×; 95.51%; 24.89×; 95.98% | no decision; 0 comments | SUCCESS | UNKNOWN |
| #7292 [wip] ref model cross-step prompt KV cache via native past_key_values | 83 d / 83 d | 0hujun (NONE) | 621 l / 6 f, tests y | 16x; 13x; 0.9x; 8GB | no decision; 1 comments | PENDING | UNKNOWN |
| #7277 [rollout] FlexKV KV cache offload for dynamic resource | 83 d / 89 d | DylanChen-NV (NONE) | 785 l / 12 f, tests n | none stated | no decision; 1 comments | PENDING | CONFLICTING |
| #7131 [gemma4] Support attn_implementation=flash_attention_2 (global layers fall back to SDPA) | 97 d / 98 d | nanastassacos (CONTRIBUTOR) | 638 l / 3 f, tests y | 0.5×; 1150 tok/s; 2131 tok/s; 1.85× | no decision; 3 comments | SUCCESS | UNKNOWN |
| #7105 [perf] feat: mstx profiler support schedule to mini batch | 99 d / 92 d | mengchengTang (CONTRIBUTOR) | 422 l / 15 f, tests y | none stated | no decision; 3 comments | SUCCESS | CONFLICTING |
| #7108 [ckpt, sglang, megatron] feat: P2P checkpoint engine for trainer-to-rollout weight sync via Moo | 99 d / 99 d | AkiRusProd (CONTRIBUTOR) | 3418 l / 31 f, tests y | 78× faster | no decision; 2 comments | SUCCESS | CONFLICTING |
| #7097 [training_utils, perf] feat: reuse jagged rows across dynamic batches | 101 d / 100 d | zhangxin81 (CONTRIBUTOR) | 217 l / 4 f, tests y | 95%; 2.133x; 2.113x; 2.136x | no decision; 1 comments | SUCCESS | CONFLICTING |
| #7096 [worker, perf] feat: defer scalar metric materialization | 101 d / 101 d | zhangxin81 (CONTRIBUTOR) | 418 l / 14 f, tests y | 95%; 2.445x; 2.296x; 2.524x | no decision; 1 comments | FAILURE | CONFLICTING |
| #6990 [vllm, rollout, perf]feat: add extra-prefix-cache for agentic system prefix rollout | 111 d / 111 d | HanlinDu (CONTRIBUTOR) | 1308 l / 17 f, tests y | none stated | no decision; 2 comments | SUCCESS | CONFLICTING |
| #6940 (draft) [rollout, vllm] feat: KV-cache-aware request load balancer | 114 d / 113 d | zhaizhiqiangA (NONE) | 3581 l / 43 f, tests y | none stated | no decision; 2 comments | PENDING | CONFLICTING |
| #6944 [perf] fix: correct Qwen-VL head_dim consistency and merger token count (#6903) | 114 d / 114 d | chethanuk (NONE) | 57 l / 2 f, tests y | none stated | no decision; 0 comments | SUCCESS | UNKNOWN |
| #6832 Enable Fused Kernels by Default for Memory Efficiency | 127 d / 83 d | 0hujun (NONE) | 19 l / 5 f, tests n | 32x reduction; 30 GB; 9.93 GB; 0.31 GB | no decision, reviewed by Luosuu; 2 comments | PENDING | UNKNOWN |
| #6533 (draft) [rollout] feat: load-balance away from sticky sessions under imbalance | 152 d / 145 d | aoshen02 (COLLABORATOR) | 180 l / 9 f, tests y | 1.5x; 2×; 0.05%; 3.7% | no decision, reviewed by wuxibin89; 0 comments | FAILURE | CONFLICTING |
| #6194 (draft) [fsdp, megatron, vllm, trainer, algo, cfg] feat: Nitrobrew on-policy distillation (teacher hidd | 183 d / 183 d | tea-more (NONE) | 1669 l / 16 f, tests y | 2 ×; 2 ×; 14× speedup | no decision; 5 comments | SUCCESS | CONFLICTING |
| #6133 [dataset] feat: estimate vision tokens for faster prompt filtering | 187 d / 187 d | tongxiao2002 (NONE) | 92 l / 1 f, tests n | 90% | no decision; 3 comments | SUCCESS | CONFLICTING |
| #6122 [rollout] feat: global group-level load balancer | 188 d / 182 d | EricWuji (NONE) | 242 l / 16 f, tests y | 15.2%; 53.2% | no decision; 4 comments | SUCCESS | CONFLICTING |
| #6110 [fsdp, training_utils] feat: support per-sample temperature with fused linear cross entropy | 189 d / 140 d | KimmiShi (NONE) | 307 l / 5 f, tests y | none stated | no decision; 8 comments | FAILURE | CONFLICTING |
| #5976 [rollout] feat: trainer-side FP8 weight quantization for colocated and disaggregated modes | 199 d / 176 d | yxs (COLLABORATOR) | 368 l / 14 f, tests y | 8×; 8×; 8×; 8× | no decision, reviewed by wuxibin89; 5 comments | FAILURE | CONFLICTING |
| #5958 [perf,tool] feat: Support data transport time log | 201 d / 196 d | 0oshowero0 (CONTRIBUTOR) | 125 l / 4 f, tests n | none stated | no decision; 0 comments | FAILURE | CONFLICTING |
| #5838 [perf, fsdp, trainer] feat: Skip training for zero-advantage responses to speed up RL. | 210 d / 202 d | sheilaliuxl (CONTRIBUTOR) | 1142 l / 14 f, tests y | 25%; 40.9%; 1.7x; 60.7% | no decision; 0 comments | SUCCESS | CONFLICTING |
| #5778 [perf] feat: batch acquire/release RPCs, work-stealing scheduling, and empty cache dedup | 215 d / 215 d | zzxxxl (NONE) | 1412 l / 7 f, tests y | none stated | no decision; 2 comments | SUCCESS | CONFLICTING |
| #5751 (draft) [megatron, ckpt] fix: reclaim host memory leaked by dist_checkpointin… | 217 d / 217 d | Superjomn (CONTRIBUTOR) | 290 l / 5 f, tests n | 13GB; 13GB | no decision; 1 comments | FAILURE | CONFLICTING |
| #5651 [Megatron] feat: offload optimizer fp32 params to CPU to save GPU memory | 224 d / 223 d | sherry-huang-997 (NONE) | 38 l / 1 f, tests n | 2x; 2x | no decision; 2 comments | SUCCESS | CONFLICTING |
| #5619 (draft) [perf] optimize the bucketed weight transfer using double buffer in vllm rollout weight synchro | 226 d / 226 d | jianjunzhong (CONTRIBUTOR) | 2012 l / 23 f, tests y | 2x; 3.27%; 4.94%; 0.88% | no decision; 0 comments | SUCCESS | CONFLICTING |
| #5549 [perf, training_utils] fix: Run gc regularly in trainer/sft_trainer.py to avoid spikes in step  | 232 d / 216 d | sheilaliuxl (CONTRIBUTOR) | 198 l / 2 f, tests n | 16.2%; 17.0%; 48.0%; 1.92x | no decision, reviewed by vermouth1992; 1 comments | SUCCESS | CONFLICTING |
| #5426 [tool, rollout] refactor: cache tool list initialization with MCP timeout | 243 d / 237 d | denismegerle (CONTRIBUTOR) | 164 l / 3 f, tests y | none stated | no decision; 1 comments | FAILURE | CONFLICTING |
| #5391 [perf, trtllm] feat: Add Nsight support for rollout server mode (trtllm) | 245 d / 243 d | davidmlw (CONTRIBUTOR) | 364 l / 21 f, tests y | none stated | no decision, reviewed by tardis-key; 3 comments | FAILURE | CONFLICTING |
| #5398 [Feature]: enable kv cache sharing using lmcache | 245 d / 245 d | kilig917 (NONE) | 566 l / 11 f, tests n | none stated | no decision; 1 comments | PENDING | CONFLICTING |
| #5338 [fsdp,algo] feat: add use_length_grouped_bsz to reduce padding waste in FSDP training | 253 d / 171 d | roeeh21 (NONE) | 123 l / 12 f, tests n | 1x; 4.7x faster; 10.7% slower; 1x | no decision, reviewed by ai21shakedmeirom,asafkar; 7 comments | PENDING | MERGEABLE |
| #4707 [perf] fix: correct torch profiler logic for distributed trace collection | 304 d / 302 d | AkiRusProd (CONTRIBUTOR) | 71 l / 1 f, tests n | none stated | no decision, reviewed by wuxibin89; 2 comments | SUCCESS | CONFLICTING |
| #4435 [rollout, vllm] feat: support KV cache FP8 | 327 d / 321 d | sharonyu-115 (CONTRIBUTOR) | 445 l / 10 f, tests y | 32x; 3x; 20% speedup; 38% | no decision, reviewed by wuxibin89; 8 comments | SUCCESS | CONFLICTING |
| #4257 [Draft,Don't reivew][fsdp_workers] fix: Skip FSDP loading in async rollout mode to save GPU mem | 338 d / 338 d | JobQiu (CONTRIBUTOR) | 74 l / 1 f, tests n | 50%; 14GB; 22GB; 22GB | no decision; 1 comments | SUCCESS | CONFLICTING |
| #3852 [fsdp] feat: Restructure logprobs_from_logits impl and always compile the naive impl | 372 d / 343 d | StrongerXi (NONE) | 88 l / 3 f, tests y | 8x; 139.55gb; 115.91gb; 129.19gb | no decision, reviewed by wuxibin89; 6 comments | SUCCESS | CONFLICTING |
| #3548 [algo, worker] refactor: async reward computation for 20% verl training speedup | 403 d / 403 d | XuDongYangthu (NONE) | 508 l / 4 f, tests n | 20%; 20%; 20%; 20% | no decision; 2 comments | SUCCESS | UNKNOWN |
| #2629 [rollout, trainer] feat: Enabling Request Skewness Scheduler towards near-equal generated token | 466 d / 442 d | Tyizhanshen (NONE) | 1562 l / 10 f, tests n | 25% | no decision, reviewed by eric-haibin-lin; 12 comments | SUCCESS | CONFLICTING |
| #2605 (draft) [sglang] feat: Support flatten buckets to speed up resharding by 20% | 468 d / 468 d | hebiao064 (CONTRIBUTOR) | 129 l / 1 f, tests n | 20% | no decision; 0 comments | SUCCESS | CONFLICTING |
| #2220 [fsdp] fix: no pad rm_score to avoid oom | 489 d / 489 d | zengwz (NONE) | 9 l / 1 f, tests n | 40GB | no decision; 1 comments | SUCCESS | CONFLICTING |
| #1295 Optimizer-in-backward Fusion Implementation | 548 d / 539 d | tinuademargaret (NONE) | 401 l / 9 f, tests n | none stated | changes req. by eric-haibin-lin; 5 comments | SUCCESS | CONFLICTING |

## 4. Merge dynamics and the anatomy of a merged perf PR

Source: 400 merged PRs (2026-07-16 .. 2026-10-28) enriched via GraphQL for all 400 (`out_merged_enriched.json`:
createdAt, mergedAt, mergedBy, authorAssociation, size, files, reviews, body). Script: `scripts-a7/dynamics.py`.

### 4a. Volume and mix

Merged per month: Jul (from the 16th) 58 · Aug 132 · Sep 112 · Oct (to the 28th) 98, i.e. ~100-130/month. By type:
fix 222 (median 30 lines), feat 92 (median 287 lines), chore 60 (median 23), refactor 18 (median 342), test 4.

Merged per module tag per month (a multi-tag PR counts in each):

| tag | Jul | Aug | Sep | Oct | total |
|---|---|---|---|---|---|
| ci | 7 | 27 | 18 | 26 | 78 |
| doc | 13 | 20 | 13 | 17 | 63 |
| megatron | 4 | 19 | 12 | 17 | 52 |
| rollout | 5 | 11 | 20 | 15 | 51 |
| vllm | 7 | 13 | 15 | 15 | 50 |
| hardware | 3 | 4 | 7 | 20 | 34 |
| trainer | 4 | 10 | 9 | 8 | 31 |
| fsdp | 5 | 8 | 4 | 7 | 24 |
| ckpt | 6 | 10 | 2 | 3 | 21 |
| veomni | 2 | 6 | 4 | 6 | 18 |
| model | 2 | 2 | 8 | 2 | 14 |
| perf | 3 | 1 | 5 | 4 | 13 |
| recipe | 1 | 2 | 5 | 3 | 11 |
| cfg | 0 | 7 | 2 | 2 | 11 |
| sglang | 1 | 2 | 2 | 4 | 9 |
| misc | 0 | 6 | 3 | 0 | 9 |
| algo | 0 | 4 | 0 | 3 | 7 |
| training_utils | 0 | 1 | 3 | 3 | 7 |
| tool / data / tests / worker | 5 / 5 / 4 / 4 | | | | |
| ray / single_controller / fully_async | 3 / 2 / 1 | | | | |

Versus the open queue: `algo` has 52 open but 7 merged in 3.5 months; `trainer` 132 open vs 31 merged; `rollout` 121 vs 51;
`ckpt` 50 vs 21; `perf` 25 vs 13; `ci`/`doc`/`hardware`/`megatron` merge at or above their open volume. Algorithm
additions are the least likely to land; rollout/vllm/ckpt/perf infrastructure fixes are the most likely.

### 4b. Who merges, who reviews, how fast

- **Merged by:** wuxibin89 241 (60%), wucong25 118 (30%; essentially all Ascend/CI/hardware), HollowMan6 9, beirong8kmiles 7,
  Luosuu 6, ETOgaosion 5, tardis-key 4, zw0610 3, tongyx361 2, Begunner 2, aoshen02 2. Per month wuxibin89 merges 63-81.
- **Human reviewers on merged PRs (distinct PRs):** wuxibin89 200, wucong25 120, HollowMan6 11, Luosuu 11, ETOgaosion 8,
  beirong8kmiles 8, jialei777 7, Begunner 7, tardis-key 7, SamitHuang 6, zw0610 4. Approvals: wuxibin89 189, wucong25 134.
  Median 1 review, 1 comment, 1 commit per merged PR. **A merged PR is, in practice, a PR wuxibin89 (GPU) or wucong25
  (Ascend) approved personally.** CODEOWNERS listing (eric-haibin-lin, vermouth1992, PeterSH6, ZihengJiang) does not predict
  who merges; vermouth1992 (31) and eric-haibin-lin (25) do review *open* PRs but did not merge any in the window.
- **Time to merge (createdAt -> mergedAt):** all n=400 median **0.7 d**, p25 0.1 d, p75 2.9 d, mean 3.0 d; 76% within 3 days,
  1% over 30 days. CONTRIBUTOR (n=294, has prior merged commits): median 0.9 d, p75 3.7 d, 71% <= 3 d. COLLABORATOR (n=99):
  median 0.1 d, 90% <= 3 d. NONE (first-time, n=7 of 400): median 0.2 d. Longest in the sample: #7945 Liger TP fused CE
  (25 d), #7917 profiler_markers hook (20 d), #7085 EP-aware delta export (12 d).
- **Implication:** PRs either merge within days or sit forever. 389 of the 774 open PRs are from authors with association
  NONE (no prior merged commit) versus 7 of the 400 merged; 316 open are CONTRIBUTOR vs 294 merged. First-time authors
  merge at ~2% of volume. Of the 774 open PRs only 15 are APPROVED and 16 CHANGES_REQUESTED; 742 are REVIEW_REQUIRED.
  Human review coverage by age: 0-7 d 11/67, 8-30 d 18/91, 31-60 d 22/96, 61-120 d 27/124 (and 92 of those 124 have
  no commit in 60 d). 323 open PRs are CONFLICTING, 481 have a head commit older than 60 d, 379 older than 120 d.

### 4c. Shape of a merged PR, and of a merged perf PR

All merged: median 56 lines in 2 files (p25 12, p75 208 lines; p75 6 files); 44% touch `tests/`, 28% touch `docs/`.

The 13 `[perf]`-tagged merges plus ~20 perf-worded ones (table printed by `dynamics.py`), read in full for ten of them:

| PR | what | size | tests | numbers in body | time to merge | reviewers |
|---|---|---|---|---|---|---|
| #8352 stage LoRA buckets in one host block (aoshen02, collaborator) | one pinned CPU block per bucket instead of per-tensor clone+pin | +22/-5, 1 file | none | 8 (250 MiB, 165 ms -> 35 ms, 4.7x) | 0.2 d | wuxibin89 |
| #8173 don't annotate unit strides as tl.int64 (he-weiwen) | Triton CE kernels 40-50x slower on A100 from one annotation | +73/-70, 1 file | none | 45 | 0.5 d | wuxibin89 |
| #7873 remove full gc during weight refit (wuxibin89) | GC was an accidental sleep hiding a CUDA-IPC race; fixed the race | +22/-21, 7 files | 1 | 19 (433 ms, 2 GiB) | same day | self |
| #7864 avoid full GC before weight resume (lbaolin) | one-line `aggressive_empty_cache` -> `empty_cache`, A/B on 8xH200 | +1/-1 | none | 12 (4.87 s -> 4.21 s, -13.6%) | 0.1 d | wuxibin89 |
| #7095 defer gradient sync during accumulation (zhangxin81) | opt-in `no_sync` across micro-batches, default off | +282/-4, 3 files | 1 CPU test | 51 | 3.1 d | wuxibin89 |
| #7461 Liger fused linear PPO kernel (kolehma8) | optional backend | +168/-16, 9 files | 1 | 15 (-13.5% time, 33.7 GiB) | 6.2 d | wuxibin89 |
| #7945 Liger TP fused CE backend for Megatron (kolehma8) | optional backend | +1102/-54, 15 files | 4 | 120 | 25.4 d | wuxibin89 |
| #7917 profiler_markers() platform hook (kahlun) | device-neutral markers | +380/-140, 15 files | 7 | 3 | 20.1 d | wuxibin89 (3 rounds) |
| #7722 avoid overlapping rollout/training profilers in V1 (mengchengTang) | | +171/-48, 9 files | 4 | 0 | 4.9 d | tardis-key, wuxibin89 |
| #7408 profiler post hook + torch profiler (ETOgaosion, collaborator) | | +3025/-316, 43 files | 8 | 0 | 5.0 d | wuxibin89 |
| #7162 no new thread/event loop per `tqbridge` call (0oshowero0) | | +243/-168, 10 files | 2 | 0 | 2.6 d | wuxibin89 |
| #7085 EP-aware sharded delta export (ChangyiYang) | | +477/-49, 9 files | 4 | 10 (7.9 s -> 32.2 s cases, 4.5x) | 12 d | wuxibin89 |
| #7348 cache Megatron-Bridge HF export plan across weight updates (Begunner) | | +17/-2, 1 file | none | 0 | 0.2 d | HollowMan6 |

Anatomy that lands: (1) **small and surgical** - 9 of 13 are under 300 lines, five are one file; (2) a **mechanism
explanation** of *why* the old code was slow (GC as accidental sleep; int64 stride annotation defeating vectorization;
per-tensor pin_memory), not just a benchmark; (3) **measured before/after on named hardware** (8xH200, A100, H20) with the
verl metric names (`timing_s/update_weights`, `timing_s/reshard`), often with a torch memory snapshot; (4) **opt-in flags
default off** when behaviour could change (#7095, #7461, #7945), or **no behaviour change** stated explicitly (#8352);
(5) the PR body states what was searched for duplicates and names the adjacent PRs; (6) unit tests when the change has a
CPU-testable surface (`tests/**/test_*_on_cpu.py`), otherwise the e2e A/B stands in; (7) it lands in **< 1 day** when
wuxibin89 can read it in one sitting, 1-4 weeks when it is a new backend (Liger) or touches the platform abstraction.
What does *not* land: the 2k-line feature with its own config surface and a benchmark that needs 2 GPUs to reproduce
(#8300, #5838, #5778, #6990 - all unreviewed), or a design wuxibin89 disagrees with (#7804: "use CPUOffloadPolicy").

## 5. Stale open PRs in valuable areas (supersede candidates)

Definition: head commit older than 60 days, touching perf-relevant paths (`scripts-a7/stale.py`; 304 such PRs,
56 with an explicit perf/memory claim, full list in `out_stale.json`). AGENTS.md requires that a fresh PR completing an
abandoned idea "explain the difference in the issue"; the "what a successor must say" column is that difference.

| stale PR | idea | why it stalled | what a successor must say |
|---|---|---|---|
| #7281 (83 d, 89 lines, 0 comments, 0 reviews) avoid scalar-tensor dispatch in batch balancing | convert the workload tensor to a list once before Karmarkar-Karp | never reviewed; UNKNOWN mergeability; V0+V1+SFT trainers | V1-only, cite its 65-70% balance-time numbers, add the CPU test it lacked in CI |
| #8137 (21 d, not stale yet but unreviewed) cache KK `State.spread` | 1.6-2.7x on `get_seqlen_balanced_partitions` | one non-maintainer approval; extended `check_license.py` for a personal header (reviewers dislike that) | if touching `seqlen_balancing.py`, build on both #7281 and #8137 or explicitly not |
| #7097 (101 d) reuse jagged rows across dynamic batches | unbind each NestedTensor field once per `rearrange_micro_batches` | CONFLICTING; fork CI never approved | rebase onto current `tensordict_utils`; same 2.1-2.9x micro-benchmark shape |
| #7096 (101 d) defer scalar metric materialization | one stacked D2H per micro-batch instead of `.item()` per metric | CONFLICTING; 14 files across 4 engines | scope to the FSDP engine + `workers/utils/losses.py` first; note #5600 (state-based metrics, 227 d, also stale) |
| #5838 (210 d) skip zero-advantage responses in `update_actor` | 1.7-2.5x `update_actor` at high accuracy | touches deleted `dp_actor.py`; 1,142 lines; two modes | re-implement as a V1 + engine filter on `advantages == 0` rows before `forward_backward_batch`; one mode, default off |
| #5778 (215 d) agent-loop batched acquire/release + work stealing + `empty_cache` dedup | amortize Ray RPC per request | 1,412 lines, three ideas in one PR, touches deleted workers | split: the `empty_cache` cooldown alone is a 100-line PR against `verl/utils/memory_utils.py` |
| #5619 (226 d, draft) double-buffered bucketed weight transfer | overlap pack and send | 2,012 lines over 5 checkpoint engines | the bucketed path was rewritten (`bucketed_weight_transfer.py`, #8352, #7873); measure first, most of the gain may be gone |
| #5976 (199 d, collaborator yxs) trainer-side FP8 weight quantization | quantize once on the trainer, not per replica | wuxibin89 reviewed, then stalled; roadmap #8186 P0 "train/rollout quantizer parity" landed via #8287/#8306/#8308 | overlaps the roadmap owner's own work; avoid |
| #7108 (99 d) P2P checkpoint engine via Mooncake ("78x") | | 3,418 lines, CONFLICTING | out of scope for a first PR |
| #6990 (111 d) extra-prefix-cache via LMCache / #5398 (245 d) lmcache KV sharing / #7292 (83 d, `[wip]`) ref-model prompt KV cache | reuse prompt KV across the n samples of a group | all conflicting/unreviewed; #8300 attacks the same waste from the training side | if doing prefix sharing in the trainer, name #8300 as the direct competitor (8 d old, CI green, unreviewed) |
| #6122 (188 d) group-level load balancer / #6533 (152 d, draft, aoshen02) sticky-session rebalancing / #2629 (466 d) request-skewness scheduler / #6940 (114 d) KV-cache-aware router | rollout tail-latency | `GlobalRequestLoadBalancer` in `llm_server.py` landed; #8186 lists "session-affinity routing for multi-turn agents" under performance | the live balancer + #7435 (74 d, re-evaluate sticky sessions) already cover most of it |
| #6832 (127 d, 19 lines) enable fused kernels by default | 32x logits-memory claim | default flip is a policy call; Luosuu commented | not a contribution, a maintainer decision |
| #3852 (372 d) restructure `logprobs_from_logits` and always compile the naive impl | | wuxibin89 reviewed 2025; 8x memory numbers | `torch_functional.py` has since changed; measure again |
| #1295 (548 d, CHANGES_REQUESTED by eric-haibin-lin) optimizer-in-backward fusion; #5651 (224 d) offload fp32 master params on Megatron; #7804 (49 d) offload optimizer around each update | memory | eric-haibin-lin asked for changes; wuxibin89 pointed #7804 at `CPUOffloadPolicy` | the maintainers have a stated position: use FSDP2 `offload_policy`; do not re-open |
| #5549 (232 d) periodic `gc` in `sft_trainer.py`; #7873/#7864 (merged) removed GC elsewhere | GC spikes | vermouth1992 reviewed; superseded by the GC-removal direction | opposite direction now; skip |
| #2605 (468 d, draft) flatten buckets to speed resharding 20%; #956 (569 d); #1686 (520 d) | sharding-manager era | files deleted | ideas absorbed by `checkpoint_engine` / bucketed transfer |

Takeaways: the stale queue says the maintainers' answer to "memory" is FSDP2 `offload_policy` and fused kernels, and
their answer to "weight sync" is their own checkpoint-engine line (#7873, #8352, delta/NIXL engines). The uncontested
stale ideas are the **controller-side CPU costs** (#7281, #7097, #7096 - all from one or two authors, all unreviewed,
all small) and **skipping zero-advantage work** (#5838, now orphaned by the V0 deletion).

## 6. Title and template rules (what CI enforces)

Source: `.github/workflows/check-pr-title.yml` runs `tests/special_sanity/check_pr_title.py` and
`check_pr_description.py` on every `opened/edited/synchronize` event.

**Title grammar** (regex in `check_pr_title.py`):

```
[optional "[1/N]" progress prefix] [optional "[BREAKING]"] [module(, module)*] type: description
```

- `allowed_modules` (exact list in the script): `fsdp, megatron, veomni, sglang, vllm, trtllm, rollout, trainer,
  tests, training_utils, recipe, hardware, deployment, ray, worker, single_controller, misc, docker, ci,
  perf, model, algo, env, tool, ckpt, doc, data, cfg, reward, fully_async, one_step_off`.
  Case-insensitive; the bracket may contain commas and spaces; every token must be in the list
  (so `[wip]`, `[lora]`, `[npu]`, `[docs]`, `[utils]`, `[kernel]`, `[sft]` all fail; 213 open PRs fail today).
- `allowed_types`: `feat, fix, refactor, chore, test`, followed by a colon and a space, then free text.
- `[BREAKING]` goes before the module bracket; `[1/3]`-style stacking prefixes go first of all
  (open examples: `[1/3][perf, reward, rollout] feat: ...` #8289, `[2/3][perf, trainer] fix: ...` #8290).
- Compliant examples from the queue: `[rollout, sglang] fix: reduce tensor lifetime overlap during bucketed weight sync`
  (#8435), `[fsdp, perf, trainer] feat: shared-prefix packing for the FSDP engine` (#8300),
  `[training_utils, perf] refactor: cache Karmarkar-Karp state spread` (#8137).

**Description check:** the first 5 lines of the body must differ from the first 5 lines of
`.github/PULL_REQUEST_TEMPLATE.md` (i.e. the `### What does this PR do?` placeholder must be replaced).

**Template sections** (`.github/PULL_REQUEST_TEMPLATE.md`): `What does this PR do?` (concise; reference issues/PRs) ·
`Checklist Before Starting` (search for similar PRs and *paste at least one query link*; title format) · `Test`
(for changes CI cannot cover: experiments, curves, eval results) · `API and Usage Example` (code snippet) ·
`Design & Code Changes` · `Checklist Before Submitting`: read CONTRIBUTING; run
`pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always`; update `docs/`; add unit or
e2e tests to a CI workflow *or explain why not*; once ready for CI, post in the `ci-request` Slack channel (fork
workflows sit at `action_required` until a maintainer approves them; see #7097's comment); if `recipe/` is touched,
bump the submodule.

**AGENTS.md (mandatory, "breaching can result in automatic banning"):** run the duplicate checks
(`gh issue view <n> --comments`, `gh pr list --state open --search "<n> in:body"`, `gh pr list --state open --search
"<keywords>"`); do not open a PR if an open PR addresses the same fix; if the approach is materially different,
explain the difference in the issue; no one-off tiny PRs (typo, single style change, one mutable default) unless
bundled with substantive work; pure code-agent PRs are not allowed, a human must review every line and run the tests;
AI-assisted PR descriptions must state (a) why this is not a duplicate of an existing PR, (b) the test commands run and
their results, (c) that AI assistance was used; commit trailers such as `Co-authored-by: Claude`; verify bot review
suggestions (gemini-code-assist) before applying. `docs/contributing/editing-agent-instructions.md` governs edits to
AGENTS.md itself (keep it under 200 lines, no hardcoded paths, no one-off-incident rules).

**Lint:** pre-commit with `ruff` and `autogen-trainer-cfg` (regenerates `verl/trainer/config/_generated_*.yaml`;
any config-field change must commit the regenerated YAMLs or the hook fails). `check_license.py` wants the header;
#8137 notes it had to extend the check for a 2026 individual-contributor header.

**CODEOWNERS** (reviewers auto-requested): `/verl/trainer` eric-haibin-lin, vermouth1992, tongyx361, PeterSH6 ·
`/verl/workers/engine` (+`/fsdp`) eric-haibin-lin, vermouth1992, ZihengJiang, ArronHZG · `/verl/workers/engine/megatron`
ISEEKYAN, vermouth1992, HollowMan6 · `/verl/workers/engine/veomni` wuxibin89, Luosuu, FoolPlayer ·
`/verl/workers/rollout/vllm_rollout` wuxibin89, PeterSH6, chenhaiq, ArronHZG, HollowMan6 ·
`/verl/workers/rollout/sglang_rollout` zhaochenyang20, SwordFaith, chenhaiq, ArronHZG · `/verl/experimental` wuxibin89,
ArronHZG · `/verl/single_controller` zw0610, wuxibin89, hongpeng-guo · `/verl/models/mcore` ISEEKYAN, vermouth1992,
HollowMan6 · `/verl/models/transformers` vermouth1992, PeterSH6, tardis-key, wucong25, ji-huazhong · `/verl/utils/megatron`
ISEEKYAN, HollowMan6 · `/verl/utils/vllm` wuxibin89, HollowMan6 · `/docs` eric-haibin-lin, zhaochenyang20, hongpeng-guo.
No owner is declared for `verl/utils/*` generally (seqlen_balancing, torch_functional, kernel), `verl/checkpoint_engine`,
`verl/protocol.py`, `verl/workers/utils` or `verl/trainer/ppo/core_algos.py` beyond the `/verl/trainer` owners.

## 7. Quiet, high-value zones

Criteria: on the live (V1/engine) code path, few *active* open PRs (head commit <= 30 d), not slated for removal or
deprecation in the 26Q4 roadmap (#8186), and in the lane the merge data says lands (small, measured, mechanism-explained
infrastructure fixes; see section 4c). Ordered by value-to-contention ratio as this slice sees it.

| zone | open / active PRs | why quiet | why valuable | watch out |
|---|---|---|---|---|
| `verl/utils/seqlen_balancing.py` (Karmarkar-Karp partitioning, `rearrange_micro_batches`) | 10 / 4 (#8300, #8257*, #8137, #7901) | controller-side Python nobody owns in CODEOWNERS | three small unreviewed PRs (#7281, #8137, #7097) each claim 1.6-2.9x on a per-step controller cost; a successor can bundle them | must name #7281/#8137/#7097 as prior art and say what differs; one of them (#8137) touched `check_license.py` |
| `verl/workers/utils/losses.py` + `padding.py` | 7 / 2 and 10 / 3 | V1 engine loss glue, only #8176 and #8026 active | every PPO step passes through it; `.item()`-per-metric (#7096 idea) lives here | #8176 (bypass-mode token-mean normalization) is 19 d old and touches the same lines |
| `verl/utils/torch_functional.py` (logprobs, entropy, sum_pi_squared) | 18 / 4 (#8377, #8365, #8121*, #8026) | small file, no owner | #8365 (43 lines, chunked sum_pi_squared) is the model of a mergeable PR; #3852's compile idea is stale | tests in `tests/utils/test_torch_functional.py` are CPU-runnable, so CI covers you |
| `verl/utils/kernel/{kernels,linear_cross_entropy}.py` | 3 / 2 each (#8378, #8097*, #7686) | Triton, few people touch it | #8173 (int64 stride annotation, 40-50x on A100) merged in 0.5 d: kernel fixes with a mechanism story land fastest | #8378 (confident-label logprobs) is 3 d old; Liger backends (#7461, #7945) merged as alternatives |
| `verl/checkpoint_engine/{nccl,nixl}_checkpoint_engine.py`, `workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | 7 / 2, 6 / 2, 7 / 2 (mostly #7840) | wuxibin89 works here himself (#7873, #8352) | weight refit is a roadmap heading ("Weight refit optimization") and the GC/pinning fixes merged same-day | #7840 (7,043-line backend-neutral Runtime, APPROVED by zw0610) will rewrite call sites; #8395 (sharded NIXL, 2 d) and #8316 are active |
| `verl/trainer/ppo/v1/{replay_buffer,trainer_separate_async,trainer_colocate_async,trainer_sync}.py` | 5 / 2, 7 / 2, 5 / 3, 4 / 1 | V1 async paths are young (median 50 d) | roadmap "Async rollout" section; `trainer_base.py` is saturated (23 active) but the per-mode files are not | #8203 (streaming/TQ) and #8290 (profiling windows) touch all of them; #8201 is `[BREAKING]` on replay_buffer |
| `verl/workers/rollout/replica.py`, `workers/rollout/sglang_rollout/*` | 18 / 1, 30 / 5, 24 / 2 | vLLM gets 24 active PRs, SGLang 5 | SGLang weight sync (#8435, 0 d) shows the lane is open | CODEOWNERS zhaochenyang20/SwordFaith/chenhaiq/ArronHZG; wucong25 merges Ascend-SGLang |
| `verl/utils/memory_utils.py` (`aggressive_empty_cache`), `verl/utils/activation_offload.py` | 1 / 0 (#5778 stale), 1 / 1 (#8277) | nobody files PRs here | #7864/#7873 removed GC on two paths and merged in hours; remaining `aggressive_empty_cache(force_sync=True)` calls are candidates with the same evidence shape (A/B timing + memory snapshot) | the maintainer position is "measure, then remove", not "add a cooldown" (#5778) |
| `verl/workers/engine/fsdp/fsdp_turbo_impl.py` | 2 / 1 (#8251, NPU) | new file | FSDP-Turbo is the Ascend fast path; wucong25 merges Ascend work within a day | NPU-only unless the turbo path is generalized |
| `verl/protocol.py` (`DataProto`) | 11 / 1 (#7901) | core type, people avoid it | one active PR; V1 moves data through TensorDict/TQ (#8203), so DataProto changes may be in the wrong direction | #7000 (111 d, "remove transfer queue and restore RayPPOTrainer") shows the area is contested at the design level |

Explicitly **not** quiet-and-valuable: `verl/experimental/fully_async_policy/*` (0-13 PRs, 1 active, but #8186 deprecates
it to verl-recipe); `ray_trainer.py` (V0, slated for removal: a V0-only PR is wasted); `core_algos.py` (45 open, 5 active,
but `algo` merges at 7 per 3.5 months against 52 open); the generated YAMLs; `vllm_async_server.py`/`engine/fsdp/
transformer_impl.py`/`engine_workers.py`/`v1/trainer_base.py` (19-24 active each: expect weekly rebases and the
wuxibin89 queue).

Maintainer-stated wants that are still open (from #8186 and the issues dump): "Load balance: session-affinity routing
for multi-turn agents" (rollout performance), "Weight refit optimization", low-precision inference-only forwards (ref/teacher/
old_log_prob in FP8/FP4, P1 Common), batch-invariant/deterministic kernels (P2), unified low-precision recipe config (P1),
and #3624's five-service decomposition (rollout, model, weight-transfer, agent loop, data transfer).

## 8. Duplicate-check recipe (for the other jobs)

Run from `<repo>`. Step 1 is free (local dumps); steps 2-3 spend GitHub *search* quota
(30/min, shared across the ten agents: at most a few each); steps 4-6 are core-API reads (5,000/h).

```bash
# 1. Local, no API: keyword OR-search over title+body and path-prefix match over the 774 open PRs,
#    joined with last-commit / mergeable / review state from research/scripts-a7/out_open_enriched.json.
python3 -I research/scripts-a7/dupcheck.py --kw "Karmarkar" --kw "seqlen_balancing" --path verl/utils/seqlen_balancing.py
python3 -I research/scripts-a7/dupcheck.py --kw "zero advantage" --kw "zero_adv" --merged     # also scan merged titles

# 2. AGENTS.md-mandated GitHub searches (paste the query links into the PR body's "Checklist Before Starting").
gh pr list --repo verl-project/verl --state open --search "<area keywords>" --limit 50 \
  --json number,title,isDraft,updatedAt,author --jq '.[] | "\(.number)\t\(.isDraft)\t\(.updatedAt[:10])\t\(.author.login)\t\(.title)"'
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body" --json number,title
gh issue view <issue_number> --repo verl-project/verl --comments

# 3. Has it already landed? (merged search; the local dump covers only 2026-07-16 onward)
gh pr list --repo verl-project/verl --state merged --search "<keywords>" --limit 30 --json number,title,mergedAt

# 4. For each candidate PR found: is it alive, mergeable, reviewed?
gh pr view N --repo verl-project/verl --json state,isDraft,reviewDecision,mergeable,author,commits,files \
  --jq '{state,isDraft,reviewDecision,mergeable,author:.author.login,lastCommit:(.commits[-1].committedDate),files:[.files[].path]}'
gh api repos/verl-project/verl/pulls/N/reviews --jq '.[] | "\(.user.login)\t\(.state)\t\(.submitted_at[:10])"'
gh pr view N --repo verl-project/verl --comments | sed -n '1,120p'      # maintainer verdicts, e.g. #7804's "use CPUOffloadPolicy"
gh pr diff N --repo verl-project/verl | head -300

# 5. Who last touched the file on main, and in which PRs (names the reviewer to expect).
git -C verl log --oneline -15 -- verl/utils/seqlen_balancing.py

# 6. Does the file still exist on main? (69 open PRs touch files deleted in #6067.)
git -C verl ls-files verl/workers/actor/dp_actor.py verl/workers/fsdp_workers.py   # empty output = deleted
```

Decision rule (AGENTS.md "fail closed"): if an open PR with a head commit in the last 60 days addresses the same fix,
stop. If the only matches are stale (section 5) or touch deleted files, proceed, but the PR body must name them and state
the difference (scope, mechanism, measurements, V1-only). If a maintainer has rejected the approach in comments
(#7804, #1295), do not re-open it. Title must pass `tests/special_sanity/check_pr_title.py` (section 6); run
`python3 verl/tests/special_sanity/check_pr_title.py` with `PR_TITLE="[...] feat: ..."` locally before filing.
