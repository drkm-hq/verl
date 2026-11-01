# Adversarial review: contrib/p2-oldlogprob-skip (2026-10-29)

Branch `contrib/p2-oldlogprob-skip`, worktree <repo>/wt/p2, head 1e45a508,
base main @ 2ab35292. Reviewer read the full diff (18 files, +200/-24 product lines, +831 test lines) and
every touched function in context; nothing in the worktree or the clone was modified (`git status` clean
before and after). File:line references are worktree lines at head 1e45a508.

## 1. Verdict: NEEDS-FIXES

The exactness guarantee is enforced only at config time, in prompt units
(`verl/utils/config.py:115-120`: `actor.ppo_mini_batch_size == train_batch_size / parameter_sync_step`), but
the number of optimizer steps the worker actually takes is `data.shape[0] // mini_batch_size_per_gpu`
(`verl/workers/engine_workers.py:320`), where `data.shape[0]` is the row count AFTER `_balance_batch` pads
the batch to a multiple of `lcm(dp, ppo_mini_batch_size * n)` (`verl/trainer/ppo/v1/trainer_base.py:1596-1628`)
and after the replay buffer materialised every `{uid}_{session}_{index}` key of the sampled prompts
(`verl/trainer/ppo/v1/replay_buffer.py:70-73, 393-406`; a session emits one row per agent-loop output,
`verl/trainer/ppo/v1/agent_loop_tq.py:186-198`). One extra trajectory in a multi-turn agent loop pads the
batch to exactly 2 x `ppo_mini_batch_size * n` rows: `make_iterator`'s divisibility assert passes, the
worker runs two optimizer steps, and `ppo_loss` is told `on_policy_skip_old_log_prob=True` on the second
one (`verl/workers/utils/losses.py:100,114`): ratio identically 1 at post-step weights, clipping disabled,
no error anywhere. The worker-side assert only covers `epochs == 1 and not shuffle`
(`engine_workers.py:375-378`). This is the exact failure the design promised could not happen ("no silent
fallback"), it is reachable with shipped features, and the fix is a few lines (defect 1). Everything else
found is medium or lower. The mechanism itself (same weights for every micro-batch of one mini-batch,
row-aligned write-back, detached ratio, no-grad entropy) is correct; see section 6.

## 2. Defects

| # | Sev | Where | Failing input | Proposed fix |
|---|---|---|---|---|
| 1 | HIGH | `engine_workers.py:320,375-378`; `trainer_base.py:1596-1628,1862-1909`; `config.py:115-120` | `train_batch_size=16, n=8, dp=8, ppo_mini_batch_size=16`, flag on, one session returns 2 agent-loop outputs: 129 rows -> padded to 256 -> `256 // 128 = 2` mini-batches; the 2nd trains with `old_log_prob = log_prob.detach()` after an AdamW step (review tests `test_balance_batch_padding_keeps_exactly_one_mini_batch[1,7,128]`, `test_two_mini_batches_under_the_skip_must_be_refused`, both FAIL on the branch) | (a) In `_update_actor` under the skip, raise unless `len(batch) == ppo_mini_batch_size * rollout.n` (message: the sampled batch, including padding rows and multi-trajectory sessions, must be exactly one mini-batch). (b) Defense in depth in `train_mini_batch`: when `on_policy_skip_old_log_prob` is set, assert `data.shape[0] == mini_batch_size_per_gpu` (one optimizer step) in addition to epochs/shuffle; the assert on `return_model_output` alone stays as is. Add a CPU test for both (the review tests above can be adapted). |
| 2 | MEDIUM | `config.py:_validate_skip_old_log_prob_config`; `verl/workers/engine/megatron/transformer_impl.py:1436-1437` | `model_engine=megatron` + `actor.megatron.dynamic_context_parallel=true` + flag: the training pass drops `model_output`, so `train_mini_batch` dies with `IndexError: tuple index out of range` at `model_output.batch_size[0]` (`engine_workers.py:380`) instead of a named validation error (review test `test_megatron_dynamic_context_parallel_must_be_rejected` FAILS) | Add `actor.megatron.dynamic_context_parallel=True` to the violation list. Also state in the docs/PR that Megatron, VeOmni and TorchTitan are untested with the flag (the data path exists: Megatron keeps `model_output` in training, `:1436-1437`, and computes the entropy when `calculate_entropy`, `:1176-1189`, with an extra `logits.clone()`; TorchTitan/VeOmni never drop `model_output`, `torchtitan/transformer_impl.py:427-430`, `veomni/transformer_impl.py:486-492`). |
| 3 | MEDIUM | `trainer_base.py:1642-1646` (property), `config.py:270` | The trainer honours the flag unconditionally; `validate_config` is the only gate and is called from `main_ppo.py:181` / `main_ppo_v0.py:179` / `one_step_off_policy/main_ppo.py:52`; any other entry point that builds `PPOTrainer` (recipes, custom mains, the experimental fully-async stack has its own `_validate_config`) runs the skip unvalidated, and at runtime only epochs/shuffle are asserted | Same runtime checks as defect 1; optionally resolve the gate once in `PPOTrainer.__init__` (the vet's design) so the trainer never depends on an external call for its own exactness invariant. |
| 4 | LOW | `trainer_base.py:615-617,1862-1909` | `trainer.critic_warmup=N>0` with the flag: during warmup `_update_actor` is skipped, so `old_log_probs`/`entropy` are never written and `actor/entropy`, `training/rollout_*`, `rollout_corr/*` are missing for those steps (no crash: `_compute_metrics` and `_log_rollout_data` do not read them, `:1930-1945`, `:1307-1308`) | Document, or log the old-pass metrics only when `critic_warmup == 0`; not a correctness issue. |
| 5 | LOW | `trainer_base.py:1897-1908`, `losses.py:166-172` | Flag on with the default `entropy_coeff=0`: `calculate_entropy=True` is forced, so `actor/entropy_loss` (and the `policy_loss -= 0 * entropy_loss` term) now appears for every run with the skip; today it appears only when `actor.calculate_entropy` or `entropy_coeff != 0` | Either accept (document the new metric) or gate the entropy-loss term/metric on `entropy_coeff != 0` in `ppo_loss`. Cosmetic. |
| 6 | LOW | `trainer_base.py:1897-1909` | `return_model_output`, `entropy_requires_grad`, `on_policy_skip_old_log_prob` stay in `batch.extra_info` after the update (the meta is the carrier for every worker call). Nothing in `_step_once` calls a worker with that meta afterwards, but a subclass or a later stage would inherit `on_policy_skip_old_log_prob` silently | Pop the four keys after `update_actor` returns (or send them on a copy of `extra_info`). |
| 7 | LOW | `engine_workers.py:473` (old pass: `eval_mode`) vs `:315` (update: `train_mode`); `fsdp/transformer_impl.py:300-303` zeroes dropout only for the value head | Any HF config with non-zero dropout (`attention_dropout`, `hidden_dropout`): today's old log-probs are eval-mode values and the ratio is 1 +/- dropout noise (clip bites); under the skip the old log-probs are the train-mode (dropout-sampled) values and the ratio is exactly 1; `training/rollout_probs_diff_*` compares train-mode values to the rollout | Not a defect for dropout-free RL configs; document the semantic change in the docs paragraph and in the PR body; GPU check in section 3. |
| 8 | LOW | `docs/perf/perf_tuning.rst:140-175`; `PR-BODY.md` Benchmark | The doc paragraph and the PR body present the Megatron path as working ("compute the entropy metric with grad as before") without a run; the benchmark table is all `TBD` | Fill the benchmark (two arms, 40-step parity) before posting; soften the Megatron claim to "data path present, untested" or run one Megatron step. |

## 3. Semantic risks needing a GPU check (what to measure)

- S1 Kernel parity between the two passes is now moot, but the skip changes WHICH kernel path produces the
  stored log-probs: update pass = flash-attn CE with `inplace_backward=True` (rmpad, entropy no-grad) vs the
  old pass's `inplace_backward=False` (`torch_functional.py:95-102,129`); the forward is the same kernel.
  Measure in arm B vs arm A (BASELINE-A at `ppo_mini_batch_size=128`): `training/rollout_probs_diff_mean/max`,
  `training/rollout_logprobs_diff_*`, `actor/entropy` within bf16 noise; `actor/pg_clipfrac == 0` and
  `actor/ppo_kl == 0` exactly in arm B; `kl_loss`, `grad_norm` within noise.
- S2 Memory. FSDP rmpad non-fused: `perf/max_memory_allocated_gb` of the update must be unchanged (no-grad
  entropy keeps `inplace_backward=True`). Fused kernels: NO saving, the fused entropy is grad-attached
  regardless (`fsdp/transformer_impl.py:1385-1392`). Megatron: `calculate_entropy=True` in training adds
  `logits_bak = logits.clone()` (`megatron/transformer_impl.py:1177`), a (T, V) copy the update pass did not
  make on main (entropy was computed in the old pass only): peak memory of the training step can go UP on
  memory-bound Megatron runs. Measure `perf/max_memory_allocated_gb` on a Megatron arm before claiming parity.
- S3 Timing: `timing_s/old_log_prob` -> TQ bookkeeping only (~0.05 s); `timing_s/update_actor` + entropy over
  (T, V) logits; net `timing_s/step`. The PR body table is empty (TBD); fill with mean of steps 2-6, two arms,
  raw logs attached (BUILDER-RULES, Evidence).
- S4 TQ write-back of the update pass on a real multi-rank worker group (never exercised on CPU): per-rank
  chunks, `is_mp_src_rank_with_outputs` collect, `extra_info["metrics"]` as a list with rank 0 first. Check:
  (a) `actor/*` metrics equal between arms except clipfrac/ppo_kl; (b) `old_log_probs` in TQ row-aligned:
  run `compute_log_prob` once on the same batch after the update in a debug hook and compare max |diff|
  (expected: within the bf16 shape noise of #6280, zero mismatches of row identity).
- S5 Ulysses SP > 1 and remove-padding: `_gather_and_unpad_packed` of the no-grad entropy (`:1463`);
  `actor/entropy` must match arm A.
- S6 Dropout (defect 7): a model config with `attention_dropout > 0` flips today's eval-mode old log-probs to
  train-mode values; verify the documented semantics or zero dropout for the actor as the value head does.
- S7 LoRA with `ref_in_actor`: adapters enabled during the update pass, so the returned log-probs are the
  adapter policy's (same as the old pass); the ref pass uses `no_lora_adapter`. Check `kl_loss` parity.
- S8 Multi-turn agent loop emitting several outputs per session (defect 1): before the fix, a 2-step run
  shows `actor/pg_clipfrac == 0` while the second mini-batch's rows are trained post-step; after the fix the
  run must fail at startup/first step with the named condition.
- S9 Megatron end-to-end with the flag (defect 2/8): one `tests/special_e2e` GRPO step on Megatron, or
  drop the Megatron claim from the docs.
- S10 `use_fused_kernels=True` with `entropy_coeff != 0`: `actor/entropy_loss` and `grad_norm` equal between
  arms (the entropy gradient must still flow; CPU proof in section 5 is on a stub, not the Triton op).

## 4. Merge-process gaps

- Benchmark: PR-BODY.md "Benchmark" is all `TBD` (timing, entropy, probs-diff, memory, 40-step parity curves,
  raw logs). Blocker for posting; the vet's two arms are specified.
- GPU coverage: no GPU test or e2e script exercises the flag (TQ write-back, fused kernels, SP, Megatron).
  The PR template asks for tests covering the code; add a `tests/special_e2e` GRPO variant with the flag or a
  `tests/models/test_engine.py` case comparing `update_actor`'s returned log-probs with `compute_log_prob`.
- Defects 1-3 need code + CPU tests before review; defect 2 is a one-line validator addition.
- Docs: `docs/perf/perf_tuning.rst:140-175` should state (a) the one-mini-batch condition is in ROWS after
  padding (multi-trajectory sessions), (b) the fused-kernels and Megatron memory caveats (S2), (c) the
  train-mode (dropout) semantics (S6), (d) that `actor/entropy_loss` appears with the flag. "Last updated"
  bump is present (required by `check-docs-time-info`).
- PR body checks: duplicate-search links (4) present; #3119 #6067 #3996 #6280 #6519 #8026 #8364 named with
  the right roles; AI-assistance statement present; 5/5 commits carry `Co-authored-by: Claude Fable 5.1`;
  title `[trainer, perf] feat: ...` passes `check_pr_title.py` (modules `worker`, `fsdp`, `cfg` are also
  touched; maintainers may ask for them). Test commands and counts in the body match what I reproduced.
- Lint: ruff check + `ruff format --check` clean on the 11 changed .py files (ruff 0.16.8 from the
  pre-commit cache; the repo pins v0.12.2, which the author ran via pre-commit). mypy/license/docstring
  hooks not re-run here (read-only review); author reports Passed. Generated yamls: identical to a fresh
  `scripts/print_cfg.py` dump for all four variants.
- Size/scope: +200/-24 product lines in 6 files, +831 test lines; no drive-by refactors (the
  `_store_actor_log_prob` extraction is the old tail verbatim). "ci-request channel" item unchecked (fine).
- Author's reviewer questions (NOTES.md): Q1 `extra_info["metrics"][0]` is acceptable but brittle: it
  relies on `BatchData.concat` appending in DP-rank order and on every rank's dict being DP-aggregated
  already (true today, `engine_workers.py:355-364`); make the `isinstance(list)` branch an assertion with a
  comment instead of silently accepting a non-list. Q2 keep the no-grad entropy scoped to the flag. Q3 keep
  the deferral (same tensors, same step). Q4 keep the assert and extend it per defect 1.

## 5. Adversarial tests and results

Location: <repo>/packages/p2/review-tests/ (outside the worktree). Run:
`cd wt/p2 && PYTHONPATH=$PWD .venv-cpu/bin/python -m pytest -c ../../pytest-cpu.ini --rootdir $PWD
-p no:cacheprovider --asyncio-mode=auto ../../packages/p2/review-tests` -> 29 tests: 24 passed, 5 failed;
every failure is a finding, not a test bug.

| file :: test | result | meaning |
|---|---|---|
| `test_rev_validator_gaps_on_cpu.py::test_parameter_sync_step_is_read_for_any_trainer_mode` | PASS | `colocate_async` (or `sync`) with `parameter_sync_step=2` and `ppo_mini_batch_size == train_batch_size` raises with `... / parameter_sync_step = 8`; the validator reads the key exactly as `trainer_base.py:131` does |
| `::test_megatron_dynamic_context_parallel_must_be_rejected` | FAIL | defect 2: `model_engine=megatron` accepted (fine), `+ actor.megatron.dynamic_context_parallel=true` also accepted although the training pass returns no `model_output` |
| `::test_fused_kernels_with_entropy_bonus_is_accepted_and_flagged_as_grad` | PASS | `use_fused_kernels=True, entropy_coeff=0.001` accepted; exactness of the bonus shown by the fused test below |
| `::test_balance_batch_padding_keeps_exactly_one_mini_batch[0]` | PASS | 128 rows -> multiple 128 -> 1 mini-batch |
| `::test_balance_batch_padding_keeps_exactly_one_mini_batch[1,7,128]` | FAIL x3 | defect 1: 129/135/256 rows pad to 256 -> 2 optimizer steps with the gate satisfied (`_get_required_batch_multiple` = lcm(8, 16*8) = 128) |
| `test_rev_worker_one_mini_batch_guard_on_cpu.py::test_two_mini_batches_under_the_skip_must_be_refused` | FAIL | defect 1/3: real `tu.make_iterator`, 4 rows, `mini_batch_size=2`, flag keys set: `train_mini_batch` runs both mini-batches |
| `::test_two_mini_batches_currently_run_two_optimizer_steps_and_return_aligned_rows` | PASS | documents the silent path: optimizer steps saw rows `[[0,1],[2,3]]`, 4 aligned rows returned (alignment holds, exactness does not) |
| `::test_one_mini_batch_is_accepted` | PASS | the intended shape works |
| `test_rev_dynamic_bsz_alignment_on_cpu.py::test_update_pass_rows_match_old_pass_rows_bitwise_in_original_order` | PASS | REAL `FSDPEngine.forward_backward_batch` -> `prepare_micro_batches` (dynamic bsz, 10 ragged rows, token budget 24 = 3 uneven permuted micro-batches) -> `postprocess_batch_func`: rows bitwise equal to the direct computation in input order; same for a forward-only pass with budget 16 (6+ micro-batches) |
| `::test_training_pass_drops_model_output_without_the_flag` | PASS | flag off: `model_output == {}` as on main |
| `::test_static_micro_batching_keeps_order_too` | PASS | `use_dynamic_bsz=False`, micro_bsz 5: order kept |
| `test_rev_fused_entropy_on_cpu.py::test_fused_rmpad_entropy_and_log_probs_keep_their_graph[True/False]` | PASS x2 | fused rmpad branch: `entropy.requires_grad` and `log_probs.requires_grad` stay True whatever `entropy_requires_grad` says -> no silent zeroing of an entropy bonus (and no memory saving on this path) |
| `test_rev_ppo_loss_modes_on_cpu.py::test_skip_matches_explicit_detached_old_log_probs_bitwise[11 modes]` | PASS x11 | real `ppo_loss` for vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg, clip_cov, kl_cov, geo_mean, dro, cispo: loss and gradient with the flag and NO `old_log_probs` field equal, bitwise, those with `old_log_probs = log_prob.detach()` passed as a field; `pg_clipfrac`/`ppo_kl` exactly 0 where reported |
| `::test_without_the_flag_a_missing_field_still_raises[vanilla,gspo]` | PASS x2 | the explicit-flag claim: a missing `old_log_probs` without the flag is a KeyError |
| `::test_flag_false_in_data_takes_the_normal_path` | PASS | |

Author's suite re-run against the worktree (`scripts/run-cpu-tests-wt.sh`, logs research/logs/rev-*.log):
rev-new 37 passed; tests/trainer 444 passed / 1 skipped; tests/utils 491 / 23 skipped; tests/workers
291 / 40 skipped / 1 deselected: identical to tests.log. Author's tests drive the real `ppo_loss`
(bitwise, with a real tiny Qwen3), the real validator on the composed shipped config, the real
`prepare_model_outputs`, and `train_batch`/`train_mini_batch` with stub engines; the trainer test uses an
in-memory TQ stand-in and a mocked worker group, so the TQ bridge, DP collect and `extra_info` list are
exercised only by reading (S4). The "bitwise" claims are meaningful on CPU fp32 only; the GPU bf16 story is
S1.

## 6. Verified correct (with file:line)

- Same weights for every micro-batch: `engine/base.py:125-128` (zero_grad -> forward_backward_batch ->
  optimizer_step); no step inside the micro-batch loop (`fsdp/transformer_impl.py:755-779`);
  `_gradient_sync_context` only defers reduce-scatter (`:700-726`); GradScaler only scales the loss; Megatron,
  TorchTitan and VeOmni do not override `train_batch` (only `optimizer_step`: `megatron:639`, `torchtitan:465`,
  `veomni:414`). LoRA changes nothing here.
- Row alignment: `postprocess_batch_func` concatenates per-micro-batch nested outputs and applies
  `restore_dynamic_batch(indices)` (`engine/utils.py:158-199`, `seqlen_balancing.py:564-587`), the path
  `compute_log_prob` uses; `make_iterator` yields input order without shuffle; `train_mini_batch`
  concatenates mini-batch outputs in order and asserts the row count (`engine_workers.py:369-383`); the TQ
  bridge writes each collecting rank's rows under that rank's chunk of keys (`transferqueue_utils.py:384-401`,
  `:184-203`), exactly as for `compute_log_prob`; `is_padding` rows are ordinary rows of the chunk
  (`padding_utils.py:153-214`): no double write, no missing rows.
- dtype: both passes call `logprobs_from_logits` on the same autocast logits (`torch_functional.py:73-107`);
  `inplace_backward` changes only the backward; entropy dtype unchanged under no_grad (author's test).
- The detach: `old_log_prob = log_prob.detach()` on the padded tensor (`losses.py:58,114`); the no-grad entropy
  enters only `policy_loss -= 0 * entropy_loss`; TQ never receives a graph (`forward_step` returns
  `detach_tree(model_output)`, `:1613-1615`).
- Nothing before the update reads `old_log_probs` under the gate: `_compute_advantage` uses it only for
  `use_kl_in_reward`, IS/RS rollout correction and the OTB estimators (all rejected); `select_fields`
  tolerates the absent field; `core_algos.py` reads `old_log_probs` only in the OTB estimators
  (`:876-927, :995-1043`) and `compute_rewards` (`:1124-1136`); `metric_utils.py:709-713` only with
  `sum_pi_squared` (rejected); every registered policy loss takes `old_log_prob` as given.
- `rollout_corr/*` deferral: metrics-only mode computes no weights and clones `response_mask` unchanged
  (`rollout_corr_helper.py:849-900, 1041-1066`); `_store_actor_log_prob` runs the same function on the same
  tensors and writes nothing back (`trainer_base.py:1715-1722`); same step's metrics dict.
- Validator: `parameter_sync_step` read as the trainer reads it for any mode; `sample_batch_size` matches
  `step()` (`:542-546`); `bypass_mode`/`rollout_is`/`rollout_rs` are the only estimator switches
  (`algorithm.py:81-134`); `use_v1=False`, R2, distillation, OTB names (`core_algos.py:108-109`), shuffle,
  ppo_epochs, sum_pi_squared, separate_async all rejected by name; one error lists all.
- Metrics plumbing: every collecting rank's metrics are DP-aggregated before return
  (`engine_workers.py:355-364`, `_postprocess_output:242-245`); `BatchData.concat` appends per-rank
  `extra_info` in rank order and clears it before `BatchMeta.concat`, so its conflict check never fires
  (`protocol.py:1303-1317`); `[0]` matches `concat_tensordict_with_none_bsz` (first rank) on the old path.
- FSDP no-grad entropy: computed before `logprobs_from_logits` consumes the logits (`:1420-1434` vs
  `:1452-1456`), so the in-place CE backward cannot clobber it; checkpointing skipped under no_grad; padded
  path identical; default unchanged (author's test + my fused test).
- Megatron is not a silent no-op: `calculate_entropy` honoured in training (`megatron/transformer_impl.py:1176-1189`),
  `model_output` returned in training unless dynamic CP (`:1436-1437`); unverified on GPU (S9).
- Flag off: all changes sit behind `skip_old_log_prob` / `return_model_output` / `on_policy_skip_old_log_prob`
  / `entropy_requires_grad` with the old defaults; `_store_actor_log_prob` is the old tail verbatim; the
  regression suites are identical to main's counts plus the new tests.
- Config surface: dataclass (`workers/config/actor.py:180`), yaml (`actor/actor.yaml:124-132`), four generated
  files (identical to a fresh dump), docs (`config.rst`, `perf_tuning.rst`), valid title, AI trailers.

## Fixes applied (author, 2026-10-29; commits on contrib/p2-oldlogprob-skip, no history rewrite)

| finding | fix | commit |
|---|---|---|
| D1 (HIGH) one extra row pads to a second mini-batch trained post-step | `PPOTrainer._check_single_mini_batch`: `len(batch) == ppo_mini_batch_size * rollout.n` (padding rows included), called before the skipped old-log-prob pass (fail fast, no forward spent) and first thing in `_update_actor`; error names sampled/padding rows and the optimizer-step count. Worker: `train_mini_batch` asserts `batch_size_per_dp == mini_batch_size_per_gpu and epochs == 1` before the iterator whenever `on_policy_skip_old_log_prob` is set. CPU tests: `test_a_batch_that_is_not_exactly_one_mini_batch_is_refused[(4,2),(4,0),(3,1)]`, `test_two_mini_batches_under_the_skip_are_refused_before_any_step`, `test_one_mini_batch_under_the_skip_is_accepted`; the reviewer's `test_balance_batch_padding_keeps_exactly_one_mini_batch[1,7,128]` adapted to assert the refusal (the arithmetic alone cannot become 1) and `test_two_mini_batches_under_the_skip_must_be_refused` passes unchanged (`steps == []`). | 506d2df1 |
| D2 (MEDIUM) Megatron dynamic CP -> IndexError | Validator rejects every `actor.strategy` other than `fsdp`/`fsdp2` (Megatron, VeOmni, TorchTitan, automodel: unverified output path; Megatron entropy (T, V) copy; dynamic CP drops outputs), so dynamic CP is covered by the engine rule. Reviewer's `test_megatron_dynamic_context_parallel_must_be_rejected` adapted: plain Megatron is rejected too. Docs/PR body say FSDP only. | a9fafde8 |
| D3 (MEDIUM) flag honoured without validation | `PPOTrainer.__init__` re-runs `_validate_skip_old_log_prob_config(self.config)` when the flag is on (test `test_trainer_init_validates_the_flag_itself` builds a trainer from the composed config with `ppo_epochs=2` and expects the error); plus the D1 runtime checks. | 506d2df1 |
| D4 (LOW) warmup steps lose the metrics | `_skip_old_log_prob_this_step()` = flag and `critic_warmup <= global_steps`: warmup steps run the old pass as before (test `test_critic_warmup_steps_run_the_old_log_prob_pass`); the rollout-correction deferral follows the same predicate. | 506d2df1 |
| D5 (LOW) `actor/entropy_loss` with coeff 0 | `ppo_loss` skips the entropy term and metric when `on_policy and entropy_coeff == 0` (test `test_entropy_is_a_loss_term_only_when_it_carries_a_coefficient_or_the_flag_is_off`, three cases incl. flag off unchanged). | 4602a7af |
| D6 (LOW) keys linger in `extra_info` | `return_model_output`, `entropy_requires_grad`, `on_policy_skip_old_log_prob` popped in a `finally` after `update_actor` (test `test_skip_keys_do_not_linger_in_extra_info_after_the_update`). `extra_info["metrics"]` must be a non-empty list (assert, per Q1). | 506d2df1 |
| D7 (LOW) train-mode (dropout) semantics | `_hf_config_dropouts` loads the actor's HF config (top level + `text_config`, 10 dropout keys); any value > 0 is a violation; unloadable config -> printed WARNING stating the assumption. Tests with a saved `Qwen3Config(attention_dropout=0.1)`. Docs state the semantics; the FSDP engine zeroes dropout only for the value head. | a9fafde8 |
| D8 (LOW) docs/PR overstate Megatron; benchmark TBD | Docs and PR body: FSDP/FSDP2 only, row-count rule, dropout, fused-kernels memory caveat, warmup, no `actor/entropy_loss`. Benchmark table stays TBD for the benchmark phase. | 5cb89c52 |

Results after the fixes: author suite 55 passed (5 files); review-tests 28 passed, 1 skipped (the
"documents the silent path" test skips by design once the guard exists); regression trainer 450 / workers 297
/ utils 497 passed (main baseline + new tests); ruff, ruff-format, mypy clean on the changed files;
`pre-commit run --all-files` result recorded in CHECKLIST.md. Review tests that were adapted carry a header
note; nothing else in review-tests/ was changed.

## 8. Re-verification (reviewer, 2026-10-29, HEAD 5cb89c52; read-only, worktree and clone clean before/after)

### Verdict: MERGE-READY-PENDING-GPU

Fix commits a9fafde8, 506d2df1, 4602a7af, 5cb89c52 re-read in context (`git diff 1e45a508..HEAD`): D1-D8 are
closed as claimed. The exactness invariant is now enforced in rows where it is decided:
`PPOTrainer._check_single_mini_batch` (`trainer_base.py:1656-1677`, `len(batch) == ppo_mini_batch_size * n`,
padding rows counted, message names sampled/padding rows and the step count) runs before the skipped pass
(`:1685`) and first thing in `_update_actor` (`:1901`); the worker asserts `batch_size_per_dp ==
mini_batch_size_per_gpu and epochs == 1` before the iterator whenever `on_policy_skip_old_log_prob` is set
(`engine_workers.py:305-313`). The validator restricts `actor.strategy` to fsdp/fsdp2 (`config.py:147`; the
actor's `strategy` is what selects the engine backend: `workers/config/actor.py:325` copies it into the engine
config, `engine_workers.py:171` builds from it) and rejects HF-config dropout (`:103-124, 155-166`).
`PPOTrainer.__init__` re-validates (`trainer_base.py:135`); `_skip_old_log_prob_this_step` keeps the old pass on
critic-warmup steps (`:1651-1654`); the skip keys are popped in a `finally` (`:1950-1954`); the metrics list
is asserted (`:1964`); `ppo_loss` adds no entropy term or metric under the skip with `entropy_coeff == 0`
(`losses.py:171`). Docs and PR body now say FSDP/FSDP2 only, row-count rule, dropout, fused-kernels memory,
warmup, no `actor/entropy_loss`.

### Author's adaptations of my tests: legitimate
Only `test_rev_validator_gaps_on_cpu.py` was changed (header note). `test_megatron_dynamic_context_parallel_
must_be_rejected` now requires `ValueError` matching `actor.strategy=megatron` for plain Megatron and still
requires a raise with dynamic CP: stricter, consistent with the engine rule. `test_balance_batch_padding_keeps_
exactly_one_mini_batch[0,1,7,128]` keeps the padding arithmetic and now drives a stub `PPOTrainer`: the exact
case must pass `_compute_old_log_prob` without any forward, every padded case must raise `exactly one
mini-batch` from both `_compute_old_log_prob` and `_update_actor` with the literal row message. Both assert
the refusal; neither weakens a check. The worker-guard test passes unchanged; its "documents the silent path"
twin skips by design.

### Re-runs
- review-tests (5 original files): 28 passed, 1 skipped (was 24/5).
- New `test_rev2_fixes_on_cpu.py` (11 tests): 10 passed, 1 failed (R1 below).
- Author suite 55 passed; regression tests/trainer 450 passed / 1 skipped, tests/utils 497 / 23,
  tests/workers 297 / 40 / 1 deselected: identical to the author's counts (research/logs/rev2-*.log).
- ruff check + format --check clean on the 11 changed .py files (ruff 0.16.8 from the pre-commit cache).
  Generated yamls untouched by the fix commits. Product diff is now +322/-27 (was +200/-24), tests +1050.

### New adversarial cases (targeting the fixes)
| test | result | meaning |
|---|---|---|
| `test_trainer_check_and_worker_assert_count_the_same_thing[128-0]`, `[120-8]` | PASS | exact batch, and padding rows inside ONE mini-batch (one optimizer step, exact): trainer and worker both accept |
| `[128-128]` (sampled rows == `ppo_mini*n`, padding doubles the batch: the critic-lcm shape), `[129-127]`, `[64-0]` | PASS | trainer refuses with `256 rows (128 sampled + 128 padding)` etc.; the worker assert, fed the same rows per rank (dp=2), refuses too: both count total rows, they agree in all five cases |
| `test_strategy_gate[fsdp, fsdp2, megatron]` | PASS | fsdp and fsdp2 accepted, megatron rejected by name |
| `test_dropout_in_top_level_and_text_config_is_detected` | PASS | Qwen3 `attention_dropout=0.1` -> `[("attention_dropout", 0.1)]` and rejected by name; Llava `text_config` dropout detected; clean config -> `[]` |
| `test_unloadable_config_warns_and_accepts` | PASS | missing path -> `WARNING: ... could not load the model's HF config` on stdout, flag accepted |
| `test_vision_tower_dropout_is_detected_too` | FAIL | R1: `LlavaConfig(vision_config=CLIPVisionConfig(attention_dropout=0.1))` -> `[]` |

### Residual findings
| # | Sev | Where | Finding | Proposed fix |
|---|---|---|---|---|
| R1 | LOW | `verl/utils/config.py:120` | `_hf_config_dropouts` scans the top level and `text_config` only; a VLM actor's `vision_config` / `audio_config` dropout (e.g. `CLIPVisionConfig.attention_dropout`) is not detected although the vision tower also runs in train mode during the update, so the skip is accepted for a model whose update-pass log-probs are dropout-sampled | iterate every sub-config: `for name in ("text_config", "vision_config", "audio_config")` or `hf_config.sub_configs` (transformers 5) and scan each; one-line change plus the review test above |
| R2 | LOW | `config.py:147`; `engine/fsdp/fsdp_turbo_impl.py:24` | `fsdp_turbo` is a registered backend allowed by `workers/config/engine.py:299` and `FSDPTurboEngineWithLMHead` subclasses `FSDPEngineWithLMHead` without overriding `forward_backward_batch`/`prepare_model_outputs`/`forward_step`, i.e. it shares the exact code path verified here, yet the whitelist rejects it | conservative, not wrong: whitelist it after the GPU run or name it in the docs as excluded for now |
| R3 | INFO | `config.py:103-118` | the dropout check loads the HF config on the driver at `validate_config` time (`copy_to_local(path)` then `AutoConfig`), before `HFModelConfig.__post_init__` makes the same copy (`trainer_base.py:713`, `workers/config/model.py:156`); same download, just earlier (with `use_shm=True` the later shm copy is additional, as before). The notice is `print`ed, not logged, and the validation runs twice in the standard entry (`main_ppo.py:181` and `trainer_base.py:135`), so it can print twice | use `logger.warning`; cosmetic |
| R4 | INFO | `trainer_base.py:1596-1613,1656-1677` | the config gate cannot predict the padding: with a critic whose `ppo_mini_batch_size * n` does not divide `train_batch_size * n`, `_balance_batch` pads to the lcm and the skip is refused at the first step (correctly, with the row counts), not at startup | optional: pre-check `critic.ppo_mini_batch_size` divides `actor.ppo_mini_batch_size` when `use_critic` |
| R5 | INFO | PR-BODY.md | benchmark table still TBD (benchmark phase); product diff +322 lines, slightly over the ~300 envelope, justified by the runtime checks | state both in the body |

Nothing else found after a genuine attempt: the row-count check and the worker assert agree on every shape
tried; the `finally` pop, the warmup predicate and the loss change behave as described; regression suites are
unchanged. GPU checks S1-S10 of section 3 still apply (S9 is moot: Megatron is rejected).

### Fixes applied, re-verification residuals (author, 2026-10-29)

| finding | fix | commit |
|---|---|---|
| R1 (low) `_hf_config_dropouts` scanned only the top level and `text_config` | Recursive walk over every nested `PretrainedConfig` (vision_config, audio_config, encoder/decoder, ...), cycle-guarded, bare key names, duplicates collapsed. Tests: the reviewer's vision-tower case (`test_vision_tower_dropout_is_detected_too`, now passing unchanged) and `test_hf_config_dropouts_scans_every_nested_sub_config` in the author's file (nested text, vision-only, clean VLM). | 2b259b2d |
| R2 (optional) `fsdp_turbo` excluded by the whitelist | Allowed: it subclasses the FSDP engine and inherits the tested forward / output path (overrides only mesh, parallel state, module build, DP accessors, `is_mp_src_rank_with_outputs`, `optimizer_step`); GPU-untested like the rest, stated in the validator message, the docs paragraph and NOTES.md item 16. Test `test_every_fsdp_family_strategy_is_accepted[fsdp, fsdp2, fsdp_turbo]`. | 2b259b2d |

Results: author suite 59 passed; review-tests rounds 1+2 39 passed, 1 skipped; regression trainer 450 / utils
501 passed; ruff, ruff-format, mypy clean on the changed files. HEAD 2b259b2dd1de70a36c4e83bddf67ec149ff21138.

## 10. Review round 2 (the submitter)

Source: research/review-bundle/REVIEW.md "P2 accepts actors whose training forward differs from the old eval forward"
(submission blocker), the P2 rows of FIX-CHECKLIST.md, and the "P2 pooled gains" section. Fix commits on
contrib/p2-oldlogprob-skip after 2b259b2d, no history rewrite; HEAD ba60cf8e.

| finding | fix commit | test proving it | status |
|---|---|---|---|
| Validator inspected the saved base HF config, not the effective model: `override_config.attention_dropout=0.5` accepted (train/eval gap 0.153) | c6067fec: `_hf_config_dropouts` builds the HF config the way `HFModelConfig.__post_init__` does (override_config, `model_config` sub-key, `update_model_config`) and scans it | `tests/utils/test_skip_old_log_prob_effective_config_on_cpu.py::test_dropout_enabled_through_override_config_is_rejected[plain, model_config sub-key]`; reviewer probe `p2_dropout_override.py` now exits 1 with the ValueError | fixed |
| Loaded LoRA adapter with `lora_dropout=0.3` accepted (gap 0.065) | c6067fec: with `lora_rank > 0` and `lora_adapter_path`, `adapter_config.json`'s `lora_dropout` is read and rejected when > 0; without an adapter path verl's `_build_lora_module` sets no `lora_dropout` (peft default 0.0, `nn.Identity`), documented in the helper | `...::test_loaded_lora_adapter_dropout_is_rejected` (rejected 0.3; accepted 0.0 and verl's own config); module level `tests/workers/test_skip_old_log_prob_module_walk_on_cpu.py::test_loaded_lora_dropout_is_found_through_the_real_loader` (real `_build_lora_module`, lora_B randomised as in the probe, gap > 1e-3) | fixed |
| Mixtral `router_jitter_noise=0.1` accepted (gap 0.00083): training-only behaviour not named dropout | c6067fec: the config scan keys on name fragments (dropout, pdrop, drop_path, drop_prob, drop_rate, layerdrop, jitter) over every numeric key > 0, top level and nested; documented against transformers 5.16 (`router_jitter_noise`, `moe_jitter_eps`, `layerdrop`, `encoder/decoder_layerdrop`, `drop_path_rate` exist as config keys; `jitter_noise`, `stochastic_depth_prob` only as module attributes) | `...::test_saved_router_jitter_is_rejected`, `...::test_layerdrop_and_drop_path_keys_are_detected`; module level `...::test_mixtral_router_jitter_is_found` (`model.layers.0.mlp: jitter_noise=0.1`) | fixed |
| Warning-only behaviour when inspection fails vs the claimed rejection guarantee | 1ef03e93: `verl.utils.model.find_train_mode_dependent_modules` + `ActorRolloutRefWorker._check_actor_for_on_policy_skip`, called in `init_model` right after the actor engine is built (every rank, before any update): `_DropoutNd` with p > 0 (peft `lora_dropout` entries included), DropPath / StochasticDepth with a rate > 0, numeric dropout / jitter / layerdrop / drop-path attributes > 0; ValueError lists up to 5 module paths. c6067fec: the warning now says the check is enforced in the worker | `tests/workers/test_skip_old_log_prob_module_walk_on_cpu.py` (7 tests: Qwen3 functional dropout, Mixtral jitter, real loader, effective-config module, dropout/DropPath/chunks, worker refusal with 5-module cap and flag-off no-op); `tests/utils/...::test_unreadable_adapter_or_config_warns_that_the_worker_enforces_the_rule` | fixed |
| "Adding names to the dropout key list alone does not fix (1) or (2)" | both layers above; the docs paragraph (ba60cf8e) describes the two-layer rule | reviewer probes re-run: all three cases rejected at the validator (tests.log) | fixed |
| RESULTS.md stale "no arm was re-run at the final sha" while the table has final-head arms; "paired same-node" wording | packages/p2/bench/RESULTS.md: sha paragraph corrected (off-3/on-5 at 2b259b2d; later commits: see the GPU note at the end of this section), "same scheduling window (exploratory; similar windows do not prove the same physical node)", pooled-means section added, equality wording softened | n/a (package text) | fixed |
| Pooled claim 5-7 %, subgroup comparisons exploratory, entropy metric / write-back now in the update pass, GPU suite label, AI statement | PR-BODY.md rewritten per REVIEW-ROUND-RULES.md | n/a | fixed |

Not changed: the review-tests' import of `_hf_config_dropouts` is kept (same name and `(key, value)` list, now of the
effective model); `fsdp_turbo` stays allowed (round-1 R2 reasoning unchanged); the round-2 commits change validation and init only (no
measured path); the stock GPU selection later ran on 7f0c53f5, see the GPU note at the end of this section.

### Round-2 re-review (AI reviewer, packages/p2/REVIEW-ROUND2-VERDICT.md): D1-D4 -> fixed; HEAD ba0dccf5

| finding | fix commit | test proving it | status |
|---|---|---|---|
| D1 MEDIUM: a peft LoHa / LoKr adapter at `lora_adapter_path` is loaded by the real `_build_lora_module` (any `peft_type`); its `rank_dropout` is neither read by the validator (`lora_dropout` only) nor seen by the walk (peft keeps it as a dict); gap 1.9e-4 | c7031289: `_lora_adapter_dropout` rejects `peft_type != "LORA"` by name and scans every rate key of adapter_config.json; the walk expands dict / list valued rate attributes (`rank_dropout[default]=0.5`) | `tests/utils/test_skip_old_log_prob_effective_config_on_cpu.py::test_non_lora_peft_adapter_is_rejected_by_type_and_rate` (0.5 and 0.0: the type is the gate), `tests/workers/test_skip_old_log_prob_module_walk_on_cpu.py::test_loha_adapter_dict_valued_rank_dropout_is_found_through_the_real_loader`; the reviewer's `test_loha_adapter_rank_dropout_is_rejected_by_the_validator` and `..._is_found_by_the_module_walk` pass | fixed |
| D2 LOW: "the module walk is the guarantee" overclaims (hard-coded `F.dropout`, config reads in forward, BatchNorm invisible to a structural walk) | ba0dccf5 (worker docstring, docs/perf/perf_tuning.rst) and PR-BODY.md: the walk rejects every train-mode-dependent module or rate that is visible structurally; behaviour hidden in forward code is not detectable, so for custom models the skip remains the user's assertion; the helper docstring and the config warning say the same (c7031289) | the reviewer's `test_documents_that_code_level_train_mode_dependence_is_outside_both_layers` kept; branch copy `test_hard_coded_functional_dropout_is_outside_the_walk` | fixed |
| D3 LOW: `_BatchNorm` / InstanceNorm with `track_running_stats` not flagged | c7031289: `torch.nn.modules.batchnorm._NormBase` with `track_running_stats=True` flagged (`BatchNorm1d(track_running_stats=True)`); without running stats both modes normalise with the batch, not flagged | `test_batchnorm_and_instancenorm_with_running_stats_are_found`; the reviewer's `test_batchnorm_with_running_stats_is_found` passes | fixed |
| D4 LOW: CHECKLIST.md and bench/RESULTS.md still called 2b259b2d the final HEAD | package text: both named HEAD ba0dccf5 and stated the then-missing GPU run; since then the stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`) ran on 7f0c53f5 (bench/results/gpu-tests-p2-r2): 503 passed / 0 failed / 8 skipped + torchrun 2/2, identical to main, and both say so | n/a | fixed |

Noted by the reviewer, out of scope and unchanged: `main_ppo_v0.py` run with the default `trainer.use_v1=true` passes the
validator while the legacy RayPPOTrainer ignores the flag (silent no-op, not inexact); recorded as an open question in NOTES.md.
Re-run after the fixes: author suite 76 passed; review-tests rounds 1+2 plus round2/ 46 passed / 1 skipped; utils 509 / 23 skipped,
workers 306 / 40 skipped / 1 deselected; pre-commit Passed on the changed files.

### Legacy-trainer silent no-op (reviewer's out-of-scope note, closed on the lead's call): HEAD 7f0c53f5

| finding | fix commit | test proving it | status |
|---|---|---|---|
| `main_ppo_v0.py` (or any path building the legacy `RayPPOTrainer`) with the flag set passes the validator (it checks `trainer.use_v1`, whose shipped default is `true`) while the legacy trainer never implements the skip: a silent no-op, the old-log-prob pass kept | 7f0c53f5: `RayPPOTrainer.__init__` raises a ValueError naming the flag and `trainer.use_v1` right after storing the config (the single choke point: `main_ppo_v0.py` and `SeparateRayPPOTrainer` both construct it there); no other legacy change | `tests/trainer/ppo/test_skip_old_log_prob_legacy_trainer_on_cpu.py::test_legacy_trainer_refuses_the_flag_it_does_not_implement` (the composed config passes `_validate_skip_old_log_prob_config`, construction raises; flag off: the constructor proceeds to its usual role-mapping assertion) | fixed |

Re-run: author suite 77 passed (8 files); tests/trainer 451 passed / 1 skipped; pre-commit Passed on the two files.

### GPU note (2026-10-29, bench agent)

On 7f0c53f5 stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`) ran on 7f0c53f5 (bench/results/gpu-tests-p2-r2): 503 passed / 0 failed / 8 skipped + torchrun 2/2, identical to main. It is regression coverage, not an enabled-feature run; the perf numbers were measured at 1e45a508 / 5cb89c52 / 2b259b2d; product code since then (validation, the module walk, the legacy rejection) is not on the measured path.

## 11. Round 3 (the AI follow-up review, research/review-bundle/follow-up-review/REVIEW.md R2-01): HEAD 5532f3a4

| finding | fix commit | test proving it | status |
|---|---|---|---|
| R2-01 MEDIUM: with `actor.fsdp_config.qat.enable=true, mode=w4a4` both the validator and the worker walk accepted the actor, but verl's `QATLinear._fake_quantize_activation` runs its activation observer only when `self.training`, so the eval-mode old pass and the train-mode update pass quantize identical input with different `input_global_scale` values (probe: eval 2688 vs train 1344, observer amax 1 -> 2) | 5532f3a4: (1) `_validate_skip_old_log_prob_config` rejects enabled W4A4 QAT (reads `actor.fsdp_config.qat`, the engine config the FSDP engine gets through `ActorConfig.engine = fsdp_config`), naming the mode and the train-mode activation observers; W4A16 (weight-only, no observer branch) stays accepted and is documented as untested; (2) `find_train_mode_dependent_modules` flags a module with `_fake_quantize_activation` whose `mode` attribute is `w4a4` (`QATLinear(mode=w4a4): the activation observer updates input_global_scale in train mode only`), so `_check_actor_for_on_policy_skip` refuses it; (3) docs/perf/perf_tuning.rst and PR-BODY.md compatibility lists | `tests/utils/test_skip_old_log_prob_config_on_cpu.py`: the qat case in `test_each_inexact_condition_is_rejected_by_name` and `test_w4a4_qat_is_rejected_for_its_train_mode_observers_and_w4a16_is_accepted` (exact message; W4A16 and `enable=False` accepted); `tests/workers/test_skip_old_log_prob_module_walk_on_cpu.py::test_w4a4_qat_linear_is_found_and_the_worker_refuses_it` (verl's unmodified `QATMode` / `QATLinear` from source, CUDA quantizer replaced by a spy as in the probe; walk finds it, worker refuses, eval vs train scale differ, W4A16 not flagged); the submitter's probe `reproduce_qat_skip.py` re-run from a scratch copy exits 1 at the validator naming w4a4 | fixed |

Re-run after the fix: author suite 80 passed (8 files); review-tests rounds 1+2 plus round2/ 46 passed / 1 skipped; utils 511 / 23
skipped, workers 307 / 40 skipped / 1 deselected; pre-commit Passed on the changed files. The round-3 commits are validation-only
(off the measured path); the stock GPU selection ran on 7f0c53f5 and on the final HEAD e19e7949 (next entry).

### Round-3 re-review (packages/p2/REVIEW-ROUND3-VERDICT.md: R2-01 FIXED, MERGE-READY): D1 -> fixed; HEAD e19e7949

| finding | fix commit | test proving it | status |
|---|---|---|---|
| D1 LOW: an Enum-valued `qat.mode` (a programmatic DictConfig carrying `QATMode.W4A4`; YAML / CLI cannot) passed the validator because `str(enum).lower()` is `"qatmode.w4a4"`, while `apply_qat` lowercases the str-enum and applies W4A4 (the worker walk already refused it by `mode.value`) | e19e7949: the validator compares `str(getattr(mode, "value", mode)).lower()`, as the walk does | branch: the enum case in `tests/utils/test_skip_old_log_prob_config_on_cpu.py::test_w4a4_qat_is_rejected_for_its_train_mode_observers_and_w4a16_is_accepted` (a local `str, Enum` mirroring `QATMode`); the reviewer's `round3/test_r3_qat_on_cpu.py::test_enum_valued_mode_is_rejected` passes (11 / 11) | fixed |

Noted by the reviewer, no defect: `actor.qat` (actor.yaml's top-level block) has no reader in the v1 FSDP engine, which takes
`fsdp_config.qat`; the validator accepts the inert key and rejects the live one, consistent with the engine. Re-run: author suite 80
passed; review-tests rounds 1-3 57 passed / 1 skipped; utils 511 / 23 skipped; pre-commit Passed. GPU: the stock unit-test selection (regression coverage: it excludes `*_on_cpu.py` and
`tests/experimental`) ran on the final HEAD e19e7949 (bench/results/gpu-tests-p2-r3-final, 8x H200): 503 passed / 0 failed / 8 skipped (junit 529 / 521 / 0 / 8) + torchrun 2/2, identical to main; the perf numbers stay those measured at 2b259b2d and earlier (the commits since are validation-only).
