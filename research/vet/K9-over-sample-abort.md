# K9 — over-sample-abort (adversarial vet)

Vetter pass 2026-10-29. Clone `verl` @ `2ab35292` (read-only); PR/issue dumps in `gh/` (2026-10-28) plus
read-only `gh` reads of #4422, #7994, #8203 (diff), #4349, #8186 and `verl-project/verl-recipe/partial_rollout`.
Model script: `research/vet/K9/tail_model.py` (pure arithmetic, run with `python3 -I`). Measured step anatomy
from `bench/results/baseline-a-main-1` (main@2ab35292, 8×H100, Qwen3-1.7B, GRPO n=8, 128 prompts, 1024/2048).

## 1. Claim

Source: `research/05-rollout-weightsync-audit.md` §1.1 item 5, finding F2 and candidate C2; `research/08-literature.md`
§3 rank 3. `rollout.over_sample_rate` is documented (`rollout.yaml:137-139`) with no consumer
(`verl/workers/config/rollout.py:213-215` only); its implementation was deleted by #4422 (2025-12-05); APRIL exists
only as a pinned `verl-recipe/partial_rollout`; the V1 sync replay buffer waits for every group to be terminal
(`replay_buffer.py:462`), so the long tail (30-50 % of gen per 02 Theme C) is paid in full. Proposal: in V1 sync,
submit ⌈(1+r)·B⌉ prompts, train on the first B complete groups, abort the rest (`abort_only`, merged #7994) and
recycle the partial trajectories (partial-rollout client) into the next step.

## 2. Mechanism verification (file:line @ 2ab35292)

### 2.1 The dead key, and why it died

`grep -rn over_sample_rate verl recipe examples tests docs` → only `rollout.yaml:137-139`,
`workers/config/rollout.py:213-215` (comment says "for multi-turn rollout in sglang") and two Ascend doc tables.
History (`git log -S over_sample_rate`, read-only): added by #2929 (zhaochenyang20, 2025-08-14) inside the SGLang
SPMD class `sglang_rollout.py` — `target_completion = int(total_requests * (1 - over_sample_rate))`, then
`asyncio.as_completed`; cancelled requests were replaced by `_create_padding_request` (request-level abort and
**discard**, GRPO-group-agnostic). Removed by #4422 (PeterSH6, 2025-12-05, "retires sglang spmd mode") together
with the whole class; the PR body never mentions over-sampling. So there is no maintainer verdict on the idea —
the knob is a fossil of a deleted engine path; the V0 agent loop and V1 never consumed it.

### 2.2 The V1 sync step path (what is actually there)

`prepare_step` `verl/trainer/ppo/v1/trainer_base.py:1526-1528` → `_add_batch_to_generate` :1521-1524 →
`_next_train_batch` :1480-1495 (exactly `train_batch_size` prompts in `gen_batch_size` chunks) →
`_submit_batch_to_rollout` :1497-1513 (prompt tags `{"is_prompt", "status": "pending", "global_steps"}`,
`AgentLoopManagerTQ.generate_sequences` fire-and-forget, `agent_loop_tq.py:248-262`). Workers: `_run_prompt`
`agent_loop_tq.py:112-153` spawns n sessions; terminal `finished`/`failure` only after all n settle :136-148;
trajectory tags :210-225. Sampler `ReplayBuffer.sample` `replay_buffer.py:422-511`: snapshot via `kv_list`
:201-235, `has_enough_samples = len(sampleable_keys) >= batch_size` :462, `can_select = has_enough_samples and
(not dapo_enabled or inflight_count == 0)` :488, oldest-first `_select_prompt_uids` :383-392, DAPO surplus
discarded :494-500, `time.sleep(poll_interval)` :408-409. **Correction to the brief:** in non-DAPO sync mode the
sampler already returns as soon as *any* B groups are terminal; nothing waits for every dispatched group. The
"(1+r)" over-dispatch is the only dispatch-side piece missing; "stop waiting at B complete groups" exists.

### 2.3 What happens to the stragglers in sync mode today — the trap

`PPOTrainerSync.on_sample_end` `trainer_sync.py:40-44` calls `sleep_replicas()` only;
`vLLMReplica.sleep` `vllm_async_server.py:1657-1661` first awaits `wait_for_requests_to_drain` → sync mode
**waits for every in-flight request before sleeping**. Over-dispatch without an abort therefore saves nothing.
With `abort_all_requests(abort_only=True)` (`vllm_async_server.py:1111-1166`, merged #7994) the aborted request
returns `TokenOutput(stop_reason="aborted")` with its partial tokens (:734-745, :797-800); the sync client
`LLMServerClient.generate` `llm_server.py:95-162` returns it unchanged; `SingleTurnAgentLoop.run`
`single_turn_agent_loop.py:69-110` never inspects `stop_reason` and post-processes the truncated response as a
normal trajectory; `_agent_loop_postprocess` writes it with `status: "success"` (`agent_loop_tq.py:212`), the
group flips to `finished` (:147-148), and the next step's `sample` selects it **first** (oldest `global_steps`,
`replay_buffer.py:383-392`). A naive sync-mode abort trains on truncated garbage one step later. Any sync
implementation must either tag-and-evict aborted groups or use the retry client (2.4).

### 2.4 The recycle client

`FullyAsyncLLMServerClient` `llm_server.py:165-380`: on `stop_reason in ("aborted", "abort")` it sleeps 1 s and
re-issues `prompt_ids + final_output.token_ids` with the remaining budget (:291-372; `should_retry = True` for V1
because `config.async_training` does not exist, :363-366); `min/max_global_steps` recorded :352-356 so the version
span is visible in the tags. Selected by `PPOTrainerColocateAsync.get_llm_client`
`trainer_colocate_async.py:26-28`; sync uses the base client (`trainer_base.py:397`).

### 2.5 The finding: the proposal already exists as a trainer mode

`trainer.v1.trainer_mode=colocate_async` with `trainer.v1.colocate_async.num_warmup_batches=r` is, step for
step, the K9 proposal on the same GPUs:
- in-flight = (1+r)·B prompt groups: `on_train_begin` :34-35 → `_add_async_warmup_batches` `trainer_base.py:625-650`
  (fractional values supported, rounded down to `gen_batch_size` chunks; `docs/advance/v1_async_trainer.md:196-200`), and the
  inherited `prepare_step` dispatches B more every step;
- train on the first B terminal groups: `ReplayBufferAsync.sample` `replay_buffer.py:591-613` →
  `wait_for_sampleable` :564-589 → `_has_enough_samples` :541-556 = `len(sampleable_keys) >= batch_size`
  (no `inflight == 0` condition), oldest-first selection;
- abort the rest: `on_sample_end` :44-49 → `checkpoint_manager.abort_replicas()`
  (`verl/checkpoint_engine/base.py:477-484`, gate-closing `abort_all_requests`) then `sleep_replicas()`;
- recycle partials into the next step: `on_step_end` :37-42 → `update_weights` + `resume_generation_replicas()`
  (`base.py:487-489`); the parked retries of 2.4 resume with retained tokens under the new weights (partial
  rollout, doc :25 and :51-57), staleness bounded by `trainer.v1.sampler.max_off_policy_threshold: 8`
  (`ppo_trainer.yaml:298`) with `drop`/`wait`.
Trainer and rollout stay serial (same as sync). Reports 05 and 08 read `trainer_mode=sync` literally and
classified "sync straggler abort" as unclaimed; the capability is one flag away on main.

### 2.6 What would still be new, and whether it is wanted

(i) `abort_only=True` is the wrong primitive at a colocated step boundary: the engine sleeps next, so admission
must close and the retries must park until `resume_engine_generation` (:1226) — exactly what the default
`abort_all_requests()` path does (:1168-1207); `abort_only` exists for #7994's standalone-replica lifecycle.
(ii) An on-policy-exact **discard** variant (the 2025 SGLang semantics) does not exist in V1 — but the APRIL
README ("Background", `verl-recipe/partial_rollout/README.md`) states the long-tail samples "are often the
harder, more informative ones — you can't drop them without hurting accuracy", and GRESO/PODS/TailSieve (08 §1B)
argue the same. (iii) Length bias: with recycling there is no drop bias, only a one-version staleness on r/(1+r)
of the groups, which APRIL corrects with token-level TIS (`algorithm.rollout_correction.rollout_is=token`,
threshold 2.0 per the recipe README) — verl already ships that.

## 3. Duplicate / ownership check

Method: `python3 -I research/scripts-a7/dupcheck.py --kw over_sample --kw oversampl --kw "early termination" --kw straggler --kw "partial rollout" --kw "long tail" --kw abort --merged`
(40 open hits), read-only `gh pr list --state merged --search ...`, `gh pr view 4422 / 7994 / 4349`,
`gh issue view 8186 --comments`, `gh pr diff 8203`, `gh api` of `verl-recipe/partial_rollout`.

| item | state | relation to K9 |
|---|---|---|
| #8203 Begunner (COLLABORATOR), "stream V1 rollout dispatch through a dataloader actor with session-watermark refill" | open 2026-10-12, last commit 5 d, MERGEABLE, BREAKING | Replaces `replay_buffer.py` (the file K9 edits) with `StreamingDataloader` (sync) / `StreamingDataloaderAsync`; async refill is session-level (`_refill_to_watermark`: `free_slots // rollout.n` prompts per pass), `num_warmup_batches` → `watermark_batches: 2` + `max_outstanding_batches: 4`; the sync class keeps `has_enough_samples = len(sampleable_keys) >= batch_size` and `inflight_count == 0` only for DAPO — sync stays "bufferless" by the owner's design. Its body cites the DeepSeek-V4.1 report and drops "dispatch only after a whole GRPO group finished" because it "stalled on long-tail samples". The owner's direction is session watermarks in the async modes, not over-sampling in sync. |
| `colocate_async` mode (in tree) | shipped | §2.5: K9's mechanism, via `num_warmup_batches=r`. The project's own table (`docs/advance/v1_async_trainer.md:281-292`, 150 steps, 4 nodes) reports `colocate_async` −35.4 % wall vs `sync`. |
| #4349 "[rollout] partial rollout" HelloWorldBeginner | open since 2025-11, CONFLICTING, stale 323 d; bot reviews only | The APRIL upstream attempt; the work went to `verl-recipe/partial_rollout` pinned to verl `60546ef2` on a fork branch (`REQUIRED_VERL.txt`: `startju/verl-recipe@partial_rollout`) — core grew the V1 async modes with the same abort/resume client instead. |
| #7994 tongyx361 | merged 2026-09-23 | `abort_only` on `abort_all_requests` — the primitive K9 names; built for standalone-replica lifecycles (#8168), not step boundaries. |
| #5430 wuxibin89 (2026-03-04), #5071→#5127 (added then reverted 2026-01), #5253 (vLLM `pause_generation`), #7207 (cap resumes), #6090 Begunner (drain-loop resume) | merged | the resume-on-abort client and its fixes — all on the async path. |
| #6533 aoshen02 (draft, stale), #5778 (stale), #4160 (recipe, stale), #2988 (CHANGES_REQUESTED, stale) | stale | straggler balancing / dynamic sampling; none on the sampler. |
| #8273 cr-gao (open; claimed P0 in #8186) | open | keep KV across weight updates → makes recycling cheaper; complementary. |

Roadmap #8186: the "P1 sample-level dispatch" line has no comment-claim but #8203 implements it; nobody mentions
sync-mode straggler abort or `over_sample_rate`. Issues: zero open issues mention `over_sample_rate` (scan of
`gh/issues_open.json`); long-tail demand is 02 Theme C (#658, #5448, #3114, #4743) — real, and answered in-tree
by `colocate_async`.

Decision rule (07 §8): capability exists on main and the sampler is under an active owner rewrite → stop.

## 4. Minimum valuable PR

Option A — the honest reduction: config/docs hygiene. Remove `over_sample_rate` from `rollout.yaml:137-139`,
`workers/config/rollout.py:213-215`, the four `_generated_*` yamls and the two Ascend doc tables, or keep it and
fail loudly in `RolloutConfig.__post_init__`: "unused since #4422; use `trainer.v1.trainer_mode=colocate_async`
with fractional `num_warmup_batches` (or #8203's `watermark_batches`)". ≤40 lines; tests
`tests/workers/config/*rollout*_on_cpu.py` + `tests/special_sanity/check_pr_title.py`. Not a perf PR; marginal
under AGENTS.md's bar for a first-time author and possibly judged busywork.

Option B — the only genuinely missing behaviour: on-policy-exact early stop in sync (abort and **drop**).
`agent_loop_tq.py:210-225` tags `stop_reason` (surface `TokenOutput.stop_reason` through
`AgentLoopOutput.extra_fields` in `single_turn_agent_loop.py:84-110`); the sync sampler
(`replay_buffer.py:317-334` `_terminal_eviction_reasons`) evicts groups with any aborted session and emits
`training/over_sample/aborted_groups`; `PPOTrainerSync.on_sample_end` (`trainer_sync.py:40-44`) calls
`abort_replicas()` before `sleep_replicas()` when `over_sample_rate > 0`; `prepare_step` dispatches ⌈B(1+r)⌉ on
step 1 and `B + evicted` afterwards (`trainer_base.py:1526-1528`, `_add_prompts_to_generate` :1515-1519).
≈150 lines over four files that #8203 rewrites; CPU test on the real in-process TQ harness
(`tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py:39-80` `tq.init()` fixture, `RolloutProducer` :153-196 writes
trajectories then flips the prompt status). Drops r/(1+r) of each step's groups, biased to the longest/hardest
prompts — APRIL's README and GRESO/PODS say that costs accuracy; demand for strict on-policy early stop is unproven.

## 5. Proof

The decisive experiment needs **no diff**: BASELINE-A on main vs
`--overrides "trainer.v1.trainer_mode=colocate_async trainer.v1.colocate_async.num_warmup_batches=0.25 algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=2.0"`
(also `0.1`, and `0.25` without `rollout_correction`), 20 steps each (`--total-training-steps 20`); metrics
`timing_s/gen` (in `colocate_async` it includes abort+sleep), `timing_s/step`, `training/off_policy/*`,
`response_length/mean` and `critic/score/mean` parity, `actor/rollout_is_*`.

Expected on BASELINE-A as configured: **≈ 0 at any r ≤ 4**. The run's `response_length/clip_ratio` is 0.15-0.21
(`response_length/mean` 1230, max 2048 every step): a lognormal fitted to the clipped mean and that clip
(`research/vet/K9/tail_model.py`, σ ≈ 0.65) puts a capped sequence in 1 − 0.81⁸ ≈ 81 % of GRPO groups, so the
B-th group to finish still runs to the cap for r ∈ {0.1, 0.25, 0.5} (model: 0 % of gen saved); gen is cap-bound,
not tail-bound. The proof therefore needs an uncapped tail: `--max-response-length 8192` on GSM8K (model with a
*guessed* 3 % clip at 8k: 0 % / 2 % / 21 % of gen at r = 0.1 / 0.25 / 0.5) or a DAPO-math-like preset at 16k
(1 % clipped → σ ≈ 0.79: −7 % / −25 % / −37 % of gen; 0.3 % clipped: −24 % / −36 % / −44 %), consistent with
APRIL's +22.5 % rollout throughput and the fully-async doc's 2.35×. Subtract the re-prefill of the r·B resumed
groups after `update_weights` (≈ r·B·n·(P + partial) tokens; ≈ 0.3-0.5 s at r = 0.25 on this model). Model
assumptions: one decode wave (8 × 256 slots ≥ 1024), gen = T0 + k·max_len with T0 = 1.5 s. If the 20-step A/B
confirms the gain on the long-tail config, the deliverable is one doc paragraph in `v1_async_trainer.md`, not code.

## 6. Adversarial answers

- **A1:** it already exists (`colocate_async` + fractional warmup, §2.5), and the sampler it would modify is being
  replaced by the V1 owner (#8203) with a design that rejects group-completion-triggered dispatch in favour of
  session watermarks. A "sync over-sample" PR gets "use colocate_async / wait for #8203" within a day; a "restore
  `over_sample_rate`" PR revives a deleted SGLang-SPMD semantic (abort-and-pad) that breaks GRPO groups.
- **A2:** hard caps: BASELINE-A has 15-21 % of responses at the 2048 cap, so 81 % of groups contain a capped
  sequence and early stop saves nothing for any r ≤ 4 (§5) — the same holds for any run whose tail is the cap
  (DAPO-style 5 % clipped at 8k: 0 %); the re-prefill cost of resumed groups (removed only by #8273); `abort_only` at a step boundary races the engine
  sleep; after #8203 the lever is `watermark_batches` and the in-flight target is counted in sessions.
- **A3:** drop variant → length/difficulty bias (fewer long, hard groups trained); recycle variant → one-version
  staleness on r/(1+r) groups needing TIS, and in *sync* mode `_terminal_eviction_reasons`
  (`replay_buffer.py:317-334`) returns no stale set, so recycled groups are never dropped — unbounded drift unless
  the async sampler is used; the §2.3 trap (truncated trajectories tagged `success`) has no test today. Existing
  tests: `test_replay_buffer_on_cpu.py` (real TQ, tags only), `test_v1_async_alignment_on_cpu.py`,
  `test_agent_loop_tq_on_cpu.py`.
- **A4:** Begunner (#8203, #6090), tongyx361 (#7994, #8117 partial-attempt observability, #8118 token-weighted
  staleness, #8168 lifecycle), cr-gao (#8273), linyueqian (#8341/#8356 resume fixes), wuxibin89 (#5430), the APRIL
  authors (startju fork of verl-recipe).
- **A5:** Option A (dead-key removal/pointer), or a measurement-only contribution: post the zero-diff A/B numbers
  on #8186 under the P1 line.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | **FAIL** | mechanism shipped as `colocate_async` + fractional `num_warmup_batches` (§2.5); dispatch policy owned by #8203 |
| G2 not reserved | **FAIL** | sampler/dispatch under the V1 owner's BREAKING rewrite (#8203) |
| G3 self-contained | PASS | no engine fork, no deps |
| G4 provable | PASS only on a long-tail config (zero-diff A/B); BASELINE-A as configured is cap-bound and shows nothing — and either way there is nothing to merge | §5 |
| G5 default path | PASS | long tail is the largest pain cluster (02 Theme C) |

Scores: S1 **3** (real on long-tail workloads, via an existing mode) · S2 **4** · S3 **3** (model + APRIL numbers,
not measured here) · S4 **1** · S5 **1**.

**Verdict: NO-GO.** Strongest reason: "submit (1+r)·B, train on the first B terminal groups, abort the rest,
recycle the partials next step" is exactly what `trainer.v1.trainer_mode=colocate_async` with
`num_warmup_batches=r` does today on the same GPUs (`trainer_colocate_async.py:26-49`,
`replay_buffer.py:541-613`, `llm_server.py:291-372`); what is missing is a dead config key and a doc sentence,
and the files involved are being replaced by #8203. The useful residue is the zero-diff A/B in §5 and, at most,
a ≤40-line dead-key cleanup.
