# K7 greso-prefilter: pre-rollout prompt skipping with a pass-rate history (vet)

Vetter: adversarial pass, 2026-10-29. Clone `verl` @ 2ab35292 (read-only). Scripts and outputs under
`research/vet/K7/` (`sim_prefilter.py`, `sim_gsm8k.out`, `sim_dapo17k.out`, `sim_1epoch.out`).

## 1. Claim

Source: `research/08-literature.md` §1 (GRESO row), §2 row 30, §3 rank 1. verl filters zero-variance GRPO
groups only AFTER generation (`algorithm.filter_groups`, V1 `ReplayBuffer._dapo_filtered_keys`) and refills,
so every all-correct / all-wrong prompt still costs `n` rollouts per visit. GRESO (NeurIPS 2025, arXiv
2506.02177, built on verl + vLLM) keeps a per-prompt trace of zero-variance outcomes and skips a prompt
before rollout with probability `p_f = 1 - p_e^z` (`z` = consecutive zero-variance visits, `p_e` adapted
per class easy/hard toward a 25% zero-variance share), reporting rollout 2.4x / total 2.0x on
Qwen2.5-Math-7B (8xH100, 1000 steps) at equal accuracy. Proposal: a trainer-side `RolloutPrefilter`
(history keyed by a stable prompt id, fed from the DAPO classification, consulted at the dataloader fetch,
checkpointed), opt-in under `algorithm.filter_groups`, default off.

## 2. Mechanism verification (file:line, clone @ 2ab35292)

What exists today (post-generation only):
- `verl/trainer/config/algorithm.py:43-60` `FilterGroupsConfig {enable=False, metric=None,
  max_num_gen_batches=0, max_inflight_gen_batches=1}`; `:743` `filter_groups: Optional[...] = None`.
- `verl/trainer/ppo/v1/trainer_base.py:141-186` `_build_replay_buffer`: `refill_fn=self._add_prompts_to_generate`
  (`:162`), `filter_groups_metric` (`:167-171`), custom sampler plug-in `trainer.v1.sampler.custom_sampler.{path,name}`
  (`:143-153`, "custom implementations own their filtering semantics" `:164-165`).
- `verl/trainer/ppo/v1/replay_buffer.py:261-315` `_dapo_filtered_keys`: for every NEWLY finished group it reads
  `rm_scores` from TransferQueue (`:283-288`), classifies `classification_cache[uid] = shared value if std==0
  else None` (`:310-312`) and returns the zero-variance uids; `:317-334` eviction reasons; `:345-378`
  `_evict_terminal_groups` (metric `<train|val>/filter_groups/evicted_samples`, `:374`; prefix from
  `_metrics_prefix` `:238-240`); evicted groups are cleared from TQ and the cache (`_clear_groups` `:241-259`).
- `replay_buffer.py:442-500` sync `sample()`: refill credit `2*dapo_count + failed_count` (`:475`), dispatch
  bounded by `max_inflight_gen_batches * train_batch_size` (`:447`, `:478-486`), selection only when
  `inflight_count == 0` (`:488`), surplus effective groups DISCARDED (`:494-500`,
  `filter_groups/discarded_surplus_samples`). So a zero-variance group costs `n` rollouts, triggers a 2x
  over-dispatch, and each refill wave is a full generation latency (drain-before-select).
- No pre-rollout state anywhere: `grep -rni curriculum verl tests` = 0 hits; `create_rl_sampler`
  (`verl/trainer/ppo/utils.py:140-168`) only builds `RandomSampler`/`SequentialSampler`; the only hook is the
  comment at `verl/trainer/ppo/v1/agent_loop_tq.py:118-120` (`__rollout_n__` per prompt). `n=0` is NOT a skip
  path: `_settle_session_tasks([])` publishes `finished` with zero trajectories (`:136-148`), the empty-group
  case #8024 is still fixing. The skip must happen before submission.

Where the prefilter plugs in:
- Fetch: `trainer_base.py:1467-1478` `_fetch_one_gen_batch` (iterator rollover on `StopIteration`
  `:1473-1475`, fresh `uid = uuid4()` per fetch `:1477`); `:1480-1495` `_next_train_batch`; `:1497-1513`
  `_submit_batch_to_rollout`; `:1515-1518` `_add_prompts_to_generate` (the refill_fn). With DAPO on,
  `gen_batch_size` is already forced to 1 (`:728-740`), so a skip loop around `_fetch_one_gen_batch` costs
  nothing structurally.
- Stable prompt key: `verl/utils/dataset/rl_dataset.py:399-408` puts `row_dict["index"] = extra_info.index`
  (default 0 when absent; `examples/data_preprocess/gsm8k.py:80` sets it). Fallback must be a hash of
  `raw_prompt` or the whole key space collapses to 0.
- Reward feedback: the only place that sees rewards of EVICTED groups is `_dapo_filtered_keys` (`:310-312`);
  the trainer never does (they are cleared, `:241-259`). A 6-line optional callback there
  (`on_group_classified(uid, shared_value_or_None)`) covers evicted and selected groups alike.
- Checkpoint: `_save_checkpoint` `:949` saves `train_dataloader.state_dict()` (`:1000`); `_load_checkpoint`
  `:851` restores (`:896-897`); the history dict goes beside it (GRESO does the same with `torch.save`,
  `presampling_selector.py:268-272`).
- Epoch defaults: `verl/trainer/config/ppo_trainer.yaml:144` `total_epochs: 30`, `:147`
  `total_training_steps: null`; shipped scripts (`grep -o 'total_epochs=[0-9]*' examples recipe`): 15 (x12),
  10 (x10), 1 (x7), 2 (x5), 5, 4, 20, 100, 50. Multi-epoch is the norm, but 7 scripts are single-pass.

GRESO facts (paper HTML v1 + repo `Infini-AI-Lab/GRESO`, read 2026-10-29):
- State: per prompt a trace of (epoch, rewards); `z` = consecutive zero-variance visits. Skip prob
  `p_f = 1 - p_e^z` (Eq. 3); `p_easy`, `p_hard` adapted by +-1% per step toward a 25% zero-variance share split
  1:2 (8.3% easy, 16.7% hard), init 0.5, floor 0.05 (App. E); adaptive rollout batch `B_r = min(B_default,
  1.25 * B_delta / (1 - alpha))` (Eq. 6). Skipped prompts stay eligible (no deferral).
- Setup: 1000 steps, train batch 256 (rollout batch 384), n=8, 4k/8k context, DAPO+MATH (~24k prompts, i.e.
  ~10 passes), `trainer.total_epochs=2000` in `train-scripts/math_qwen_7b_dm_greso.sh` (run ends by steps).
- Results (Table 2, hours for 1000 steps): rollout 41.0->25.2 (1.6x), 92.4->62.0 (1.5x), 155.9->65.5 (2.4x);
  total 1.4x / 1.4x / 2.0x; rollouts 7.6M->3.3M, 2.4M->1.6M, 13.1M->6.3M; accuracy within +-0.5 of DS.
  "Prompts previously zero-variance stay so in >90% of cases; ~20% become effective again later; under DS
  the effective-prompt ratio drops to ~20% late in training, GRESO keeps it >70%."
- Code: a fork of V0 `verl/trainer/ppo/ray_trainer.py` (69.8 KB) + `verl/trainer/ttis/presampling_selector.py`
  (`DataProfiler`: trace keyed by `(data_source, index)`, `easy_probabilistic_skip` = `1 - max(p**count, 0.01)`,
  `filter_examples_all_probabilistic`, `save/load`), `rl_dataset.py`; config `+data.dynamic_filtering=True
  dynamic_filtering_strategy=all_probabilistic p_easy=0.5 p_hard=0.5 target_zero_variance=0.25
  sampling_batch_size=384 real_train_batch_size=256 +ttis.type=dynamic_sampling`; vLLM 0.8.2 pin. Not
  upstreamable as-is (V0, deleted at HEAD by #6067).

CPU model (`research/vet/K7/sim_prefilter.py`, pure Python; bimodal latent pass rates + learning drift,
verl refill semantics, GRESO skip + adaptation). Rollouts per effective batch and saving vs DAPO-only:

| population | epoch 1 | epoch 2 | last epoch | total saving |
|---|---|---|---|---|
| GSM8K-like D=7473 B=128 n=8, 6 epochs (`sim_gsm8k.out`) | gen/eff 1.42 vs 1.42, 0% | 1.29 vs 1.48, -21% | 1.21 vs 1.72, -48% | -30.5% |
| DAPO-17k-like D=17k B=512 n=16, 10 epochs (`sim_dapo17k.out`) | 1.21 vs 1.21, 0% | 1.17 vs 1.22, -8% | 1.12 vs 1.39, -32% | -18.1% |
| single pass D=100k B=512 n=8 (`sim_1epoch.out`) | 1.56 vs 1.56, 0% | - | - | 0% |

The population is assumed, not measured: the real saving is a function of the dataset's zero-variance
share, which verl already logs (`filter_groups/evicted_samples`). The model confirms the mechanism and the
two hard limits: zero gain in epoch 1, zero gain on single-pass corpora.

## 3. Duplicate / ownership check (research/07 §8 recipe)

Local dump, `python3 -I research/scripts-a7/dupcheck.py --kw curriculum --kw "pass rate" --kw pass_rate --kw
difficulty --kw greso --kw "skip prompt" --kw "prompt filter" --kw "prompt selection" --kw "dynamic sampling"
--kw prefilter --kw pre-filter --merged` (2026-10-29): 12 open hits, 0 merged-title hits since 2026-07-16.

| PR | what | state (2026-10-29) | relation to K7 |
|---|---|---|---|
| #1020 Integrate AdaRFT (maksimstw, 2025-04-11) | target-difficulty curriculum in V0 `ray_trainer.py` | CONFLICTING, last commit 566 d; author's own comment "perhaps we can merge it to recipe first #1021" | same family (pre-rollout difficulty curriculum); V0 files deleted by #6067; must be named in the PR body |
| #1021 AdaRFT recipe | `recipe/` | STALE 506 d | same |
| #2988 Upstream Dynamic Sampling (Hecate0821) | DAPO DS back-fill into V0, calls itself "pre-filter" | CHANGES_REQUESTED by zhaochenyang20 2025-09-18 ("Add an example to instruct users on adding a custom filter"), last commit 392 d | post-generation; superseded by V1 `filter_groups` (#7792, Begunner) |
| #7497 CoDaPO (X1angyuLu) | confidence/difficulty-adaptive loss weighting (ICML 2026) | OPEN, MERGEABLE, last commit 10 d | loss-side, not sampling; evidence that "difficulty" algorithm PRs are accepted |
| #8024 (dafu-wu) | V1 empty-output group = failed group | OPEN | adjacent only (mentions pass_rate); the group-status machine is still moving |
| #2799 async DAPO recipe (zpqiu) | over-sampling early stop | STALE 426 d | different mechanism |
| #7971 #8221 #7164 #6888 #6133 #1971 | keyword noise (overlong-prompt filter, NVFP4, LoRA) | - | none |

GitHub read-only (2026-10-29): `gh pr list --search "GRESO OR \"selective rollout\" OR \"pass rate\" curriculum"`
= 0 PRs (open or closed); issue search on GRESO / pre-rollout / skip prompts / curriculum sampler / difficulty
= nothing on topic (#689 2025 "How to sample specific kinds of batches", closed; #2791 soft DAPO thresholds,
closed). `verl-recipe` top-level (37 dirs) has no curriculum / GRESO recipe. Roadmap #8186 body: no
curriculum, difficulty or prompt-filtering line (grep matched only the P0 KV line and the LB line).
research/06 §8 reached the same reading: "a verl-internal opportunity more than a competitor gap".

Owners (git log on main): `replay_buffer.py` = Begunner (#7792 2026-09-08 canonical-reward DAPO filter,
#7373 2026-08-21, eviction/refill 2026-07-23, streaming dataloader + async ckpt 2026-07-14) and zhang shuai
(2026-09-18); `trainer_base.py` = tongyx361 (2026-10-12), vOv (#8018 2026-10-01), Begunner. Merger for
trainer/V1 perf work: wuxibin89 (research/07 §4b). Maintainer statements on pre-rollout filtering: none found.
The V1 code already offers `trainer.v1.sampler.custom_sampler.{path,name}` (`trainer_base.py:141-153`) and
notes that custom samplers "own their filtering semantics" (`:164-165`) — the likely redirect ("ship it as
a custom sampler / recipe"), which is what AdaRFT was told in 2025. G1 passes (no live duplicate; the stale
V0 PRs are materially different: V0, deferral/target curriculum, not probabilistic zero-variance skipping),
G2 passes (nobody has claimed it), but the redirect risk is real.

## 4. Minimum valuable PR

Title: `[trainer, algo] feat: skip recently zero-variance prompts before rollout (GRESO) in V1 DAPO`

1. NEW `verl/trainer/ppo/v1/prompt_prefilter.py` (~110 lines): `class RolloutPrefilter` with
   `should_skip(key) -> bool` (`random() < 1 - p[cls]**z`, `z=0` never skips), `observe(key, shared_value |
   None)` (`z += 1` and class easy/hard by whether the shared value is the top reward, else `z = 0`),
   `end_step()` (+-`delta` toward the 1:2 target split, floor `p_min`, cap 0.95), `state_dict()` /
   `load_state_dict()`, `metrics()` -> `prefilter/skipped_prompts`, `prefilter/tracked_prompts`,
   `prefilter/p_easy`, `prefilter/p_hard`, `prefilter/zero_variance_share`.
2. `verl/trainer/config/algorithm.py` (+~14): `PrefilterConfig(enable=False, p_easy=0.5, p_hard=0.5,
   target_zero_variance=0.25, delta=0.01, p_min=0.05, max_consecutive_skips=64)` as
   `FilterGroupsConfig.prefilter`; `ppo_trainer.yaml` (+8) and the four `_generated_*` yamls (regenerated).
3. `verl/trainer/ppo/v1/trainer_base.py` (+~40): build the prefilter only when `filter_groups.enable` and
   `filter_groups.prefilter.enable`; in `_fetch_one_gen_batch` (`:1467-1478`) loop `while
   prefilter.should_skip(key)` with the `max_consecutive_skips` guard (DAPO already forces
   `gen_batch_size=1`, `:728-740`); `uid -> key` map for submitted prompts (key = dataset `index` when the
   dataset provides distinct values, else `hash(raw_prompt)`); pass `on_group_classified` through
   `replay_buffer_kwargs` (`:155-162`); `end_step()` after `sample()`; merge metrics; save/load the state
   next to the dataloader state (`:896-897`, `:1000`).
4. `verl/trainer/ppo/v1/replay_buffer.py` (+~6): optional `on_group_classified(uid, shared_value_or_None)`
   invoked at `:310-312` for every newly finished group (evicted and selected alike).
5. Tests: NEW `tests/trainer/ppo/v1/test_prompt_prefilter_on_cpu.py` (~130 lines): z/reset semantics;
   skip probability `1-p^z` under a seeded RNG; adaptation direction, floor and cap; state round-trip; a
   synthetic-reward run (200 prompts with fixed pass rates, n=8, 3 epochs) asserting `skipped == 0` in epoch
   1, `> 0` after, falling zero-variance share, and no prompt starved below `p_min`; extend
   `test_trainer_base_on_cpu.py` (pattern `:120-128`) for the config gate and the fetch loop; one callback
   test in `test_replay_buffer_on_cpu.py` (`FakeRefiller` pattern `:82`).
6. Docs: a paragraph in `docs/algo/dapo.md`; PR body with the before/after table and the AGENTS.md search links.

Size ~170 code + ~150 test lines, four files in one area (trainer/v1), default off, no dependency, no
engine change. Deliberately out: GRESO's adaptive rollout batch (Eq. 6; verl's refill credit already
adapts), ScaleRL no-positive resampling (10-line follow-up on the same history), `__rollout_n__` reduction,
anything in `ReplayBufferAsync` beyond what the callback gives for free.

## 5. Proof on the Modal harness (bench/README.md)

Prerequisite found in the pulled BASELINE-A (`bench/results/baseline-a-main-1/metrics.jsonl`, steps 1-6):
`critic/score/mean` = 0.00-0.01 at every step (Qwen3-1.7B's thinking output does not emit the `####`
answer the GSM8K scorer parses); `prompt_length/mean` 82-87, `response_length/mean` 1195-1272 (clip ratio
0.15-0.21), `timing_s/gen` 12.3-14.3 s, `timing_s/step` 29.8-32.6 s (steps 2-6). With that reward every
group is all-wrong: DAPO would evict 100% and never fill a batch, and the prefilter would skip everything.
Fix first: `data.apply_chat_template_kwargs.enable_thinking=false` (key exists at HEAD:
`verl/trainer/config/data/legacy_data.yaml:119`, `rl_dataset.py:120,210,250`) and a 20-step smoke asserting
`critic/score/mean` > 0.3; fall back to Qwen2.5-1.5B-Instruct (add it to `download`) if it does not.

Config (both arms, same `data.seed`): preset `baseline-a` + `--overrides "algorithm.filter_groups.enable=True
algorithm.filter_groups.metric=reward data.apply_chat_template_kwargs.enable_thinking=false
trainer.total_epochs=4 trainer.total_training_steps=150 trainer.test_freq=50 trainer.val_before_train=False"`;
arm B adds `algorithm.filter_groups.prefilter.enable=True` with `--src` = the branch.
Metrics: per step `timing_s/gen`, `training/filter_groups/evicted_samples` and
`training/filter_groups/discarded_surplus_samples` (prefix from `replay_buffer.py:238-239`), prompts
dispatched per step = `B + evicted + surplus`, `prefilter/skipped_prompts`, `prefilter/p_easy|p_hard`,
`critic/score/mean`, validation acc at steps 50/100/150; per-pass means via `bench/summarize.py` extended
with these keys.
Expected: pass 1 identical (same prompts and seed, modulo rollout nondeterminism); passes 2-4: dispatched
prompts and `timing_s/gen` down 15-30% on this bench (sim: -21% in pass 2 to -48% in pass 6 for a
saturating population; GSM8K with a 1.7B model saturates fast, so this bench OVERSTATES the typical math
user, whose curated DAPO-17k-like set gives -8% to -32% in the sim); validation accuracy overlapping.
Cost: ~150 steps x 40-50 s (32 s/step + DAPO refill waves) = 1.7-2 h per arm, ~30 GPU-hours (~$150-200)
plus the smoke.

Honest estimate for the typical verl user: multi-epoch math RLVR with `filter_groups` on -> 1.3-2x rollout
time from pass 2 on (paper: 1.4-2.4x rollout, 1.4-2.0x total over 1000 steps = ~10 passes);
`filter_groups` off -> zero step-time gain (sample efficiency only, not provable in this harness);
agentic or single-pass corpora (7 shipped scripts use `total_epochs=1`) -> zero. Only 4 shipped files
mention `filter_groups` (one Ascend script, `docs/algo/dapo.md`, two Ascend tutorials); the DAPO recipe
that relies on it lives in verl-recipe.

## 6. Adversarial answers (research/VETTING-RUBRIC.md)

- A1 Strongest reason it will NOT be merged: it is an algorithmic feature in the V1 trainer core with a
  ready-made escape hatch (`trainer.v1.sampler.custom_sampler`, "custom implementations own their
  filtering semantics", `trainer_base.py:164-165`) and a precedent (AdaRFT -> "recipe first"); it needs an
  accuracy-parity argument from multi-pass runs, so it is not a numbers-in, merged-in-a-day perf PR; no
  issue or RFC asks for it; first-time authors landed 7 of the last 400 merges. The reply that kills it:
  "nice, please ship it in verl-recipe as a custom sampler".
- A2 What makes the gain disappear: `total_epochs=1` or corpus > steps x B (every prompt seen once: 0%,
  `sim_1epoch.out`); curated hard sets with a low zero-variance share (sim DAPO-17k: 8% in pass 2);
  `filter_groups` off (no refill waves to save); async modes (`ReplayBufferAsync` evicts k / refills k
  without drain, `replay_buffer.py:107`, `:577-583` -> rollout-count saving only); #8203 (Begunner,
  BREAKING `StreamingDataloader`) rewriting the fetch path the hook sits on; a dataset without
  `extra_info.index` silently collapsing to one key (guard with a hash of `raw_prompt` + a warning).
- A3 Correctness risk: the training distribution changes (easy and hard prompts under-sampled); the paper
  shows parity at targets 0-100% (App. F.1) because of the 5% floor, so `p_min > 0` must be enforced;
  resume without the history file restarts cold (safe); an all-skip fetch loop must be bounded
  (`max_consecutive_skips`); the `uid -> key` map must be cleared for failed groups. Regressions would be
  caught by `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py` (gen_batch_size semantics),
  `test_replay_buffer_on_cpu.py` (eviction unchanged with the callback) and the DAPO e2e scripts.
- A4 Who else: Begunner (owns V1 DAPO/refill and the streaming dataloader #8203), maksimstw (AdaRFT
  #1020/#1021, touched 2026-10-24 per research/06), X1angyuLu (CoDaPO #7497), dafu-wu (#8024 group
  status), the GRESO authors (no upstream activity found).
- A5 Half scope: fixed `p_easy = p_hard = 0.5` (no adaptation), one class, skip loop + callback +
  checkpoint + `prefilter/skipped_prompts` only: ~90 code + 60 test lines. The replay-buffer callback cannot
  be cut: it is the only place that sees the rewards of evicted groups.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | PASS | no open or merged PR; stale V0 #1020/#1021/#2988 named and materially different (§3) |
| G2 not reserved | PASS (risk) | no claim, no roadmap line; custom-sampler / recipe redirect likely |
| G3 self-contained | PASS | 4 files in one area, ~170 code lines, no dependency, no engine change |
| G4 provable | PASS with caveat | rollouts/step + `timing_s/gen` per pass + CPU test; needs the reward fix and ~30 GPU-hours; accuracy parity needs the full run |
| G5 default-path impact | WEAK PASS | affects the `filter_groups` opt-in (DAPO users), not the default path |

Scores: S1 4 (1.3-2x rollout time in passes >= 2 for multi-epoch DAPO users; 0 for everyone else),
S2 2 (no issue or RFC; paper + ScaleRL/MoPPS precedent only), S3 3 (paper-measured on verl V0; the CPU
model confirms the mechanism; the harness number is pending), S4 2 (algorithmic core change by a
first-time author, touches Begunner's and tongyx361's files, recipe-redirect precedent), S5 4 (first
pre-rollout filter in verl; three papers proved the idea). Total 15/25.

Verdict: GO-WITH-CHANGES. Strongest reason: the gain is real and large but conditional (multi-pass data AND
`filter_groups` on), so the PR must gate itself under `algorithm.filter_groups.prefilter`, ship the CPU
test and a 3-4 pass GSM8K before/after with skip statistics and validation accuracy, cite
#1020/#1021/#2988/#7497 and the paper, and be ready to land as a custom sampler or recipe if asked. It is
not a merged-in-a-day perf PR: rank it below the mechanical candidates on S2/S4, above them on S1/S5.
