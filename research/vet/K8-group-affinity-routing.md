# K8 group-affinity-routing: route the n GRPO siblings of a prompt to one replica (vet)

Vetter: adversarial pass, 2026-10-29. Clone `verl` @ 2ab35292 (read-only). Script and output under
`research/vet/K8/` (`sim_routing.py` imports the real `GlobalRequestLoadBalancer`; `sim_routing.out`).

## 1. Claim

Source: `research/05-rollout-weightsync-audit.md` §5.1, finding F1 (§6), candidate C1 (§7); roadmap #8186
line "Load balance: session-affinity routing for multi-turn agents" (unclaimed). Each of the `n` samples of a
prompt gets its own uuid request id, the global router picks the least-inflight replica per request, so the
siblings scatter over the replicas and the shared prompt is prefilled once per replica touched (audit:
~3.6 of 4 replicas, ~2.0M redundant prefill tokens/step, 2-4% of rollout GPU time at the reference shape;
n copies of the prompt KV). Proposal (~80-90 lines): an opt-in `group_id` (= prompt `uid`) at
`acquire_server`: first sibling least-inflight, later siblings follow the group's replica unless it is more
than `slack` requests above the least-loaded one; the same id as vLLM `session_id` pins the DP rank.

## 2. Mechanism verification (file:line, clone @ 2ab35292)

Request ids and routing:
- `verl/experimental/agent_loop/single_turn_agent_loop.py:69` `request_id = f"det-{priority}" if
  full_determinism else uuid4().hex`; `:70-80` `server_manager.generate(request_id=..., prompt_ids=..., ...,
  priority=priority)`. Multi-turn: `tool_agent_loop.py:127-142` (`_make_request_id`), `:244-250`.
- `verl/trainer/ppo/v1/agent_loop_tq.py:112-134` `_run_prompt`: the `n` siblings are `asyncio` tasks on the
  SAME worker (`:128-134`), `session_id=i` is only the TQ trajectory index; `uid` is read at `:114` and NOT
  popped, so it travels in `**prompt` -> `agent_loop.py:692-726` `_run_agent_loop(..., **kwargs)` ->
  `agent_loop.run(sampling_params, **kwargs)` (`:725`) -> `SingleTurnAgentLoop.run(self, sampling_params,
  priority=0, **kwargs)` (`:38`), which already reads `kwargs["raw_prompt"]` (`:41`). The group key is in hand.
- `verl/workers/rollout/llm_server.py:98-130` `generate()` -> `_acquire_server(request_id, prompt_ids=...,
  **kwargs)`; `:70-81` serialises only the router's `require_acquire_fields()` (lazy, cached); `:150-159`
  forwards `**kwargs` to `server.generate.remote(...)` ("Neither server signature accepts **kwargs, so pop",
  `:141-142`) with `session_id=request_id` (`:158`); the fully-async client mirrors it (`:198-206`, `:228-293`).
- `verl/workers/rollout/router.py:133-204` `GlobalRequestLoadBalancer`: LRU sticky by request id (`:167`,
  `:180-187`), least-inflight with random tie-break (`:199-201`, from merged #7613), `require_acquire_fields()
  -> []` (`:219-221`), `clear_sticky_cache` (`:263-285`, called on phase switches); full-determinism hash
  routing `:193-197`. Construction: `get_router_handle` `:397-433` (`router_config_path` YAML plugin or
  `load_balancer_cls` subclass) from `llm_server.py:525-531`; config `verl/workers/config/rollout.py:314`
  `router_config_path`.
- DP pin: `vllm_async_server.py:845-850` `_dp_rank_for_session = crc32(session_id) % dp_size`, used at `:729`;
  keyed by the per-request id today, so siblings also scatter across DP ranks (merged #8179).

Prefix cache behaviour (the saving exists only if co-located siblings actually hit):
- `verl/workers/config/rollout.py:289` `enable_prefix_caching: bool = True` (`rollout.yaml:98`, passed at
  `vllm_async_server.py:357`); `config/rollout.py:232` `max_num_batched_tokens: int = 8192`, so prompts up to
  8k are scheduled in one chunk (`:1386-1392` raises it to `max_model_len` without chunked prefill).
- vLLM V1 design doc (docs.vllm.ai/en/latest/design/prefix_caching, read 2026-10-29): "If an allocated block
  is already full of tokens, we immediately add it to the cache block, so that the block can be reused by
  other requests in the same batch"; "We only cache full blocks." So concurrently submitted siblings on one
  engine share the prefill except the trailing partial block (<16 tokens) and the >=1 token each must compute.
  The cache lives until the next weight sync (`vllm_rollout.py:247-248` reset), i.e. the whole step.
- Hit counting exists per request but is not aggregated: `vllm_async_server.py:748-749`
  `extra_fields["num_cached_tokens"]` -> `llm_server.py:284-368` -> `single_turn_agent_loop.py:109`
  (`extra_fields=output.extra_fields`); `grep num_cached verl/trainer verl/utils/metric` = 0 hits. The PR must
  add a `rollout/prefix_cached_tokens` sum or the proof has no cache-side metric.

Placement model with the real class (`research/vet/K8/sim_routing.py`, `sim_routing.out`, seed 0; sync
submission = all requests acquired before any release; `stream` = async arrival with releases in between):

| shape | policy | distinct replicas / group | redundant prefill | of all prefill | est. gen-time upper bound | load max/mean |
|---|---|---|---|---|---|---|
| BASELINE-A gsm8k 1.7B, R=8 TP=1, B=128, n=8, P=150 | baseline | 8.00 (stream 7.84) | 134k tok | 87.5% | 3.6% | 1.021 |
| same | affinity | 1.00 | 0 | 0% | 0% | 1.125 (stream 1.49) |
| reference 8B TP=2, R=4, B=512, n=8, P=1.5k, resp 4k | baseline | 4.00 | 2.30M tok | 37.5% | 3.2% | 1.009 |
| same | affinity | 1.00 | 0 | 0% | 0% | 1.006 (stream 1.13) |
| agentic 8B TP=2, R=4, B=256, n=8, P=8k, resp 2k | baseline | 4.00 | 6.14M tok | 37.5% | 18.9% | 1.003 |
| same | affinity | 1.00 | 0 | 0% | 0% | 1.033 (stream 1.08) |

Two corrections to the audit: (i) least-inflight routing does not scatter siblings randomly, it ACTIVELY
anti-colocates them: at the start of a step all counters are equal, every acquire increments the chosen
replica, so the next sibling goes elsewhere; a group touches `min(n, R)` replicas (4.00 of 4, 8.00 of 8),
not 3.6. (ii) The redundant-prefill fraction is therefore `(min(n,R)-1)/n` of all prefill tokens: 37.5% at
R=4, 87.5% at R=8. The gen-time upper bound (redundant prefill time over prefill+decode time at nominal
tok/s) is 3.2-3.6% for short prompts and long responses, ~19% only when the prompt dominates (8k prefix,
2k response). Not modelled: KV headroom (one prompt copy per replica instead of up to `min(n,R)`), which
raises decode concurrency when the replica is KV-bound; for the reference shape prompt KV is ~16% of a
sequence's KV, so the ceiling of that effect is ~10% and only if the batch is KV-bound. Load balance: in
sync mode affinity is harmless at B/R >= 128 groups (1.006 vs 1.009) and slightly worse at 16 groups per
replica (1.125 vs 1.021); under streaming arrival the `slack=8` fallback still leaves 1.08-1.49 max/mean,
so async users need `slack` ~ `n` and a measured check.

## 3. Duplicate / ownership check (research/07 §8 recipe)

Local dump, `dupcheck.py --kw affinity --kw sticky --kw "load balanc" --kw router --kw "prefix cach" --kw
kv-aware ... --path verl/workers/rollout/router.py --path verl/experimental/agent_loop/single_turn_agent_loop.py
--merged` (2026-10-29): 63 open hits (most are MoE "router" noise); the ones that matter, re-read live with
`gh pr view` / `gh api .../comments`:

| PR | what | state (2026-10-29) | relation to K8 |
|---|---|---|---|
| **#6122** `[rollout] feat: global group-level load balancer` (EricWuji) | adds `group_sticky_routing` + `request_group_id` on `acquire_server`, `rollout.load_balance_sticky_mode`; "Task 2 (Phase 2) of the routing roadmap #5442" (number does not resolve), Phase 1 = #5399 | OPEN, CONFLICTING, 16 files, last commit 2026-04-29; targets `verl/experimental/agent_loop/load_balance.py`, deleted by #7115 (2026-09-01); author pinged wuxibin89 2026-04-27, zero maintainer replies in 6 months (only gemini-bot reviews) | **the same idea**. Stale > 60 d so AGENTS.md "fail closed" allows proceeding, but the PR body must name it and state the differences (post-#7115 router, slack-bounded fallback, DP pin, metric, numbers) |
| #6533 `load-balance away from sticky sessions under imbalance` (aoshen02, collaborator) | imbalance gate that re-pins sticky sessions | DRAFT, CONFLICTING, last commit 2026-06-05, 9 files | orthogonal (would also have to move group pins); its author is the collaborator who owns sticky/DP work (#8179) |
| #7435 `re-evaluate sticky sessions against replica load` (ralovets) | `affinity_break_margin` re-pin | OPEN, CONFLICTING STALE, last commit 2026-08-15 | same family as #6533; a 3rd contributor in the same 100 lines |
| #6940 `KV-cache-aware request load balancer` (zhaizhiqiangA) | prefix-hit-rate + load routing from uni-agent, external collectors | DRAFT, 43 files, last commit 2026-07-07; roadmap #6383 "STALLED, on Q3 roadmap" (research/01:192) | the general solution of which group affinity is the degenerate case |
| #2629 skewness scheduler, #5778 batch acquire/release + work stealing, #8095 release after cancelled acquire (llm_server.py, 2026-09-30), #7667 full_determinism cross-process (router.py), #8168 lifecycle APIs (tongyx361, router.py, CONFLICTING) | neighbours | open | merge-conflict surface in `router.py` / `llm_server.py` |

Merged in the area (all by wuxibin89): #7115 pluggable router (touch869, +987/-246, 17 files, 41 days,
2026-09-01); #7613 random tie-break (+2/-1, merged next day, 2026-08-31); #8179 DP-rank session pin
(aoshen02, +18, same day, 2026-10-10); #7565 `num_cached_tokens`. Roadmap #8186 line 118 "Load balance:
session-affinity routing for multi-turn agents": research/01:110 marks it PARTLY DONE via #8179 + #7613 and
"Crowded"; no maintainer comment claims the remainder; mitukx announced a workload-aware (predicted-length)
router on #6985 (2026-10-06, no PR). Owner of `router.py` on main: only #7115's author (hangangqiang/touch869)
since the move; reviewer to expect: wuxibin89.

Reading: G1 is a conditional pass (an open PR implements the same idea; it is dead code against HEAD, so K8
would be a re-implementation, not a duplicate in the AGENTS.md sense, provided the difference is stated).
G2: not formally reserved, but a collaborator (aoshen02) is the de-facto owner of sticky/DP routing, a
maintainer-side router rewrite is announced (mitukx), and the same-idea PR got no engagement for six
months — the file is contested and the idea has not pulled a maintainer.

## 4. Minimum valuable PR

Title: `[rollout] feat: opt-in group-affinity routing for GRPO siblings (shared-prefix cache hits)`

1. `verl/workers/rollout/router.py` (+~35): `GlobalRequestLoadBalancer.__init__(..., group_affinity=False,
   group_slack=8)`, `_group_to_server: LRUCache`; `acquire_server(self, request_id, group_id=None)`: sticky
   request hit as today (`:180-187`); else if affinity and `group_id` known and `inflight[s] - min_inflight
   <= group_slack` -> `s`; else least-inflight (`:199-201`) and record the group; under `full_determinism`
   hash `group_id` instead of `request_id` (`:197`); `require_acquire_fields()` -> `["group_id"]` when
   enabled (`:219-221`); `clear_sticky_cache` also drops the group map and reports `cleared_groups`
   (`:263-285`). `_create_global_sticky_inflight` / `get_router_handle` (`:397-433`) take the two values.
2. `verl/workers/rollout/llm_server.py` (+~8): `generate()` pops `group_id` from `kwargs` before `:150`,
   passes it to `_acquire_server` (`:121-130`), uses `session_id=group_id or request_id` (`:158`) for the DP
   pin; same in the fully-async client (`:228-293`); `:525-531` forwards the config to the router.
3. `verl/experimental/agent_loop/single_turn_agent_loop.py` (+1 at `:70-80`): `group_id=kwargs.get("uid")`.
   `tool_agent_loop.py` (+2, `:244-250`) only if `AgentData` carries `uid`; otherwise leave multi-turn out.
4. Metric (+~6): sum `extra_fields["num_cached_tokens"]` into `rollout/prefix_cached_tokens` next to the
   existing `num_preempted` (`single_turn_agent_loop.py:81-82`), without which the cache effect is invisible.
5. Config: `verl/workers/config/rollout.py` `load_balance_group_affinity: bool = False`,
   `load_balance_group_slack: int = 8`; `rollout.yaml` + four regenerated yamls.
6. Tests in `tests/workers/rollout/test_router_on_cpu.py` (direct-instantiation pattern `:210-214`, client
   serialisation pattern `:216-232`): siblings co-locate; slack breach falls back and re-pins;
   `clear_sticky_cache` clears groups; `require_acquire_fields` toggles; `group_id` crosses the RPC only
   when declared; plus `research/vet/K8/sim_routing.py` as the placement/cache-hit model.
7. Docs: one paragraph in the rollout/router docs. ~90 code + ~80 test lines; default off; no dependency.

## 5. Proof on the Modal harness (bench/README.md)

BASELINE-A as pulled (`bench/results/baseline-a-main-1`, steps 2-6): `prompt_length/mean` 82-87 tokens,
`response_length/mean` ~1200, `timing_s/gen` 12.3-14.3 s (spread +-8% within one run). Redundant prefill
per step = 128 x 84 x 7 = 75k tokens = 2.6e14 FLOP = ~0.06 s over 8 H100 = 0.5% of gen: unmeasurable.
BASELINE-A can only show no regression (3 seeds per arm, ~10 min each, ~1 GPU-hour); it cannot prove K8.

Long-prefix variant (the regime the mechanism is for): build `gsm8k-longprefix/{train,test}.parquet` in the
harness `download` function (prepend a fixed ~4096-token instruction + few-shot block to every prompt;
`data.max_prompt_length=6144`; the default `max_num_batched_tokens=8192` keeps single-chunk prefill); B=128,
n=8, R=8 TP=1, Qwen3-1.7B. Arithmetic: own prefill 128 x 8 x 4.2k = 4.3M tokens -> naive 4.3M/8/150k tok/s
= 3.6 s vs ideal 0.45 s per step = ~3 s of a ~16 s gen (~20%). KV: Qwen3-1.7B 28 layers x 8 KV heads x
128 x 2 x 2 B = 115 KB/token -> a 4.2k prompt = 480 MB per copy; today each replica can hold up to 128
copies (61 GB) against ~40 GB of KV (0.6 x 80 GB - 3.4 GB weights - activations) -> KV-bound with
preemptions (`num_preempted` is already a metric, `single_turn_agent_loop.py:81-82`); with affinity 16
copies (7.7 GB). Expect 20-40% lower `timing_s/gen` and `rollout/prefix_cached_tokens` ~ 7/8 of prompt
tokens vs ~0. A constructed shape: report it as such, next to the BASELINE-A null result.
Arms: main vs branch with `actor_rollout_ref.rollout.load_balance_group_affinity=True`. Metrics:
`timing_s/gen`, `rollout/prefix_cached_tokens` (new), `num_preempted`, per-replica in-flight spread
(`get_status`), `response_length/mean` unchanged (routing must not change outputs).

Honest effect for the reference user (1.5k prompts, 8k responses, 4 replicas): 2-4% of gen (model upper
bound 3.2%) plus an unmodelled KV-headroom effect of at most ~10%, and only if the replica is KV-bound;
GSM8K-class prompts: < 1%; agentic 8k shared prefixes with short turns: 10-20% of gen plus KV headroom.

## 6. Adversarial answers (research/VETTING-RUBRIC.md)

- A1 Strongest reason it will NOT be merged: the same idea has been open as #6122 since 2026-04 and the
  file's owner has not replied in six months while merging two tiny router PRs the same day (#7613,
  #8179); the revealed preference is that group affinity is not a priority. The honest default-shape
  effect (~3% of gen, ~1.5% of step) is inside the noise band of `timing_s/gen`, so the before/after
  numbers wuxibin89 merges on would have to come from a constructed long-prefix benchmark. `router.py` /
  `llm_server.py` are contested: #8168 (collaborator), #7667, #8095, #6940, and mitukx's announced
  rewrite of exactly the selection rule inside `acquire_server` ("route by predicted completion/load
  rather than request count", #6985 2026-10-06).
- A2 What makes the gain disappear: R=1 (one TP=8 replica: siblings already co-located, 0%); short
  prompts (< 1%); `rollout.n=1` / PPO (no groups); external KV caches (LMCache #6990, version-salted caches
  #8323) that make cross-replica hits possible; `full_determinism` users (hash routing; fixable by
  hashing the group id); async streaming modes where the slack fallback re-spreads siblings (model:
  1.08-1.49 max/mean with `slack=8`); mitukx's predicted-work router routing by group itself.
- A3 Correctness risk: routing only, outputs unchanged. Risks: hot-replica avalanche (the #7613 lesson) if
  `group_slack` is large relative to B/R; a group pinned to a replica removed by `remove_servers` (handle
  like `router.py:186-187`); DP pin by group hash ignores load (same property as today's per-request
  hash, coarser); LRU growth (uids are fresh per fetch, bounded by `max_cache_size`). Tests that catch it:
  `tests/workers/rollout/test_router_on_cpu.py` (acquire/release/require-fields/plugin),
  `tests/experimental/agent_loop/test_basic_agent_loop.py`; GPU e2e unchanged.
- A4 Who else: EricWuji (#6122), aoshen02 (collaborator; #6533 draft and #8179 DP pin: the natural
  in-house owner), ralovets (#7435), zhaizhiqiangA (#6940), mitukx (workload-aware router, #6985),
  touch869/hangangqiang (#7115 pluggable router), tongyx361 (#8168, touches `router.py`), cr-gao (#8273,
  the P0 KV-persistence line claimed on #8186 2026-10-11, adjacent).
- A5 Half scope: router-only + the one-line `group_id` in `single_turn_agent_loop.py` + the flag + the
  cached-token metric; no DP pin, no tool loop: ~55 code + 60 test lines. The metric cannot be cut:
  without it the PR has no cache-side number.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | CONDITIONAL | #6122 implements the same thing: open, last commit 182 d ago, conflicting, against a deleted file; allowed only with the stated difference |
| G2 not reserved | AT RISK | no claim on #8186, but aoshen02 owns sticky/DP work, mitukx announced a router rewrite, #6940 stalled on the same roadmap line; "Crowded" (research/01:110) |
| G3 self-contained | PASS | ~90 code lines, 4 files + regenerated yamls, no dependency |
| G4 provable | PASS with caveat | measurable only on a long-prefix benchmark; BASELINE-A effect 0.5% |
| G5 default-path impact | PASS (small) | default router + default prefix caching + GRPO n=8 on > 1 replica is the default user, but the default-shape gain is <= 3% |

Scores: S1 2 (2-4% of gen at the reference shape; 10-20% only for long shared prefixes), S2 3 (roadmap
line; three external PRs show demand, none pulled a maintainer), S3 3 (mechanism verified on the real
class, two audit numbers corrected; magnitude derived, not measured), S4 2 (contested file, same-idea PR
ignored, first-time author; the diff itself is tiny), S5 2 (#6122, Seer's group context, Dynamo/vLLM
KV-aware routing all precede it). Total 12/25.

Verdict: NO-GO as a top-3 pick. Strongest reason: the idea already sits unanswered in #6122 in a contested
file, and its honest default-shape effect (~3% of gen, ~1.5% of step) is below the noise floor of the
metric maintainers merge on; it pays only on long shared prefixes, where the KV-aware router (#6940) and
external KV caches are the maintainers' stated direction. Flip condition: a long-prefix Modal run showing
>= 15% gen-time reduction AND a maintainer reply on #8186 or #6122 wanting group affinity in core; then it
is a cheap 90-line GO-WITH-CHANGES that should reuse #6122's config name (`load_balance_sticky_mode`) and
credit it.
