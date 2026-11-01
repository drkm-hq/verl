# P2 build notes: skip the old-log-prob pass when strictly on-policy

Branch `contrib/p2-oldlogprob-skip`, worktree <repo>/wt/p2, base main@2ab35292.
Python: <repo>/.venv-cpu with `PYTHONPATH=<worktree>` and cwd=<worktree>
(checked from /private/tmp: `import verl` -> wt/p2/verl/__init__.py; the editable .pth of the clone
sorts after PYTHONPATH, so no second venv was needed).

## Code facts that shaped the design (file:line on main@2ab35292)

- `verl/utils/transferqueue_utils.py:384-401` tqbridge: a TensorDict output whose `batch_size[0]`
  equals the meta size is written to TQ (`_async_update_meta_with_output` :184-203: tensors ->
  TQ fields, NonTensorData -> `meta.extra_info`), and the CALLER THEN RECEIVES A KVBatchMeta, not
  the TensorDict. `compute_log_prob` already works that way (its `metrics` NonTensorData is why
  `perf/mfu/actor_infer` is commented out at trainer_base.py:1683).
- Driver-side collect of per-rank outputs: `verl/protocol.py:1288-1317` `BatchData.concat` ->
  for KVBatchMeta `transfer_queue/metadata.py:1024-1100` `KVBatchMeta.concat`, which merges
  `extra_info` with conflict detection (see `_extra_info_values_equal` :33). For today's batch-less
  metrics TensorDict the collect is `tensordict_utils.py:221-245` `concat_tensordict_with_none_bsz`:
  the FIRST rank's TensorDict wins (metrics are DP-aggregated on the workers first,
  `engine_workers.py:355-364` + `Metric.aggregate_dp`).
- `engine_workers.py:412` `train_batch` pops `model_output` unconditionally;
  `transformer_impl.py:772-775` drops it unless `return_model_output`; `forward_step` :1614 already
  returns `detach_tree(model_output)` (verl#6698), so kept outputs carry no graph.
- `transfer_queue/metadata.py:518-540` `BatchMeta.select_fields` silently drops names that are not
  in the schema -> `_compute_advantage`'s `select_fields=[..., "old_log_probs", ...]` tolerates the
  field being absent (same mechanism as the optional `values` / `ref_log_prob`).
- `verl/utils/config.py:199` is the single `validate_config` entry (called by main_ppo.py and the
  legacy main_ppo_v0.py); `ActorConfig.validate(n_gpus, train_batch_size, model_config)` at
  `verl/workers/config/actor.py:221` sees only the actor section.

## Decisions (and deviations from the brief / vet), with file:line in the worktree

1. Flag name `actor_rollout_ref.actor.skip_old_log_prob_when_on_policy` (the vet's name, used by its
   benchmark plan; the brief's "e.g. skip_old_log_prob_on_policy" was an example). Dataclass
   `verl/workers/config/actor.py` (field after `shuffle`), yaml `verl/trainer/config/actor/actor.yaml`,
   four `_generated_*.yaml` regenerated with `scripts/generate_trainer_config.sh` (it exits 1 on
   `git diff --exit-code` until the files are committed; the regenerated content is what matters).
2. ALL validation lives in one helper, `_validate_skip_old_log_prob_config` in `verl/utils/config.py`,
   called from `validate_config` (the entry point both main_ppo.py and the legacy main_ppo_v0.py go
   through). Deviation from the vet's "put it in ActorConfig.validate": the conditions span
   actor/algorithm/trainer/distillation sections, and the mini-batch condition is against
   `train_batch_size // parameter_sync_step` (trainer.v1.<mode>), which ActorConfig cannot see. One
   ValueError lists EVERY violated condition. The legacy trainer is rejected via `trainer.use_v1=False`
   (no silent no-op), so no change in ray_trainer.py.
3. rollout_correction: the shipped ppo_trainer.yaml always includes the `rollout_correction` group
   (`ppo_trainer.yaml:40`, defaults rollout_is/rollout_rs null, bypass_mode false), and with
   `rollout.calculate_log_probs=True` (default) `_compute_advantage` runs
   `compute_rollout_correction_and_add_to_batch` in a METRICS-ONLY mode (`rollout_corr_helper.py:1013-1068`:
   no IS weights, response_mask cloned unchanged when rollout_rs is None, only `rollout_corr/*`
   diagnostics from old vs rollout log-probs). So the gate rejects only bypass_mode / rollout_is /
   rollout_rs (vet said "absent/bypass"; bypass is rejected because its loss reads old_log_probs as the
   rollout log-probs, a different estimator, and it already skips the pass). In the metrics-only mode
   the trainer defers the block: `_compute_advantage` skips it under the flag and
   `_store_actor_log_prob` computes the same function of the same tensors after the update.
   Also rejected: `actor.engine.router_replay.mode=R2` (records expert routes during the old-log-prob
   forward, `verl/utils/config.py:_validate_router_replay_config`).
4. Metrics return path (the brief's "existing tqbridge put-back"): `update_actor` returns the
   batch-sized TensorDict (nested `log_probs`/`entropy` + `metrics` NonTensorData) exactly like
   `compute_log_prob`; the nd dispatch converts the KVBatchMeta to a BatchMeta before the worker sees
   it (`protocol.py:1279-1285`), so tqbridge returns a BatchMeta per rank and the driver merge is
   `BatchData.concat`'s BatchMeta branch (`protocol.py:1303-1317`): per-rank `extra_info` values are
   appended into lists, no conflict detection. `_update_actor` reads `extra_info["metrics"][0]` (rank 0,
   which is what `concat_tensordict_with_none_bsz` returns for the TensorDict path today). Rejected
   alternative: a second worker call plus a stash (hidden state, a dispatch-order coupling).
5. Worker: `train_batch` keeps `model_output` only when `return_model_output` is set (the engine's own
   key, so Tinker and the skip share one switch); `train_mini_batch` concatenates the per-mini-batch
   outputs (`tu.concat_tensordict`) and asserts `epochs == 1 and not shuffle` (row alignment), which the
   config gate already guarantees. Outputs are already detached by `forward_step` (verl#6698).
6. Engine: `entropy_requires_grad` non-tensor key (default True = unchanged behaviour); the trainer sends
   `entropy_coeff != 0.0` ONLY under the skip. Under no_grad the entropy skips `entropy_checkpointing`
   (nothing to recompute) and `inplace_backward` keys on `calculate_entropy and entropy_requires_grad`.
   The brief's "no extra (T,V) fp32 tensor" is this. Not generalised to every run with
   calculate_entropy=True and entropy_coeff=0 (behaviour-identical but out of scope; #6519's territory).
7. Loss: `ppo_loss` keys on the explicit non-tensor flag `on_policy_skip_old_log_prob`, not on the
   absence of `old_log_probs` (an upstream bug that forgot the field must still KeyError, never silently
   train unclipped). The entropy term is untouched (0 * no-grad tensor is fine).
8. `actor/entropy` and the 8 `training/rollout_*` metrics keep their exact definitions: computed on the
   driver by the same code (`_store_actor_log_prob`, the former tail of `_compute_old_log_prob`) from
   the update pass's log-probs at the same theta.

## Results (2026-10-29)

- Commits on contrib/p2-oldlogprob-skip (base 2ab35292): 3e0f7148 cfg+validation, ae1a3cff fsdp no-grad
  entropy, 5df156ac worker return path + ppo_loss, a1d66592 trainer wiring, 1e45a508 docs.
- Tests: 37 new (5 files) pass; regression groups trainer/workers/utils green (444 / 291 / 491 passed,
  = main baseline + new tests). Full commands in tests.log; logs research/logs/p2-*.log.
- Lint: ruff, ruff-format, mypy and the repo-wide local hooks pass; `scripts/generate_trainer_config.sh`
  prints "All good" on the committed branch. Trap hit once: zsh does not word-split an unquoted
  `$FILES`, so `pre-commit run ruff --files $FILES` saw one bogus path and skipped; pass literal paths.
- Runner for worktrees: scripts/run-cpu-tests-wt.sh (VERL_ROOT=<worktree>; sets PYTHONPATH so `verl`
  resolves to the worktree instead of the clone's editable install).

## Not verified here

- GPU: the FSDP engine path with real kernels (remove-padding + Ulysses + the fused-kernels branch,
  where the entropy comes from the fused op and `entropy_requires_grad` is ignored), the TransferQueue
  write-back of the update pass's outputs on a real multi-rank worker group, the memory claim
  (`perf/max_memory_allocated_gb` unchanged with the no-grad entropy), the timing claim.
- Megatron engine: untouched (the skip is trainer/loss level; its entropy stays with grad).

## Reviewer questions

1. Metrics through `extra_info["metrics"][0]`: acceptable, or should `update_actor` keep returning the
   batch-less metrics TensorDict and the per-token outputs travel through a second worker call?
2. `entropy_requires_grad` is only sent under the skip. Generalising it to every run with
   `calculate_entropy=True, entropy_coeff=0` is behaviour-identical (0 * grad) and saves the extra
   (T, V) tensor for everyone; left out to keep the scope to the flag (#6519's territory).
3. The metrics-only `rollout_correction` default: deferring `rollout_corr/*` to after the update keeps
   the values; is the deferral comment enough, or should the metrics be dropped under the skip?
4. `train_mini_batch` asserts `epochs == 1 and not shuffle` for `return_model_output` (already
   guaranteed by validate_config for the skip); fine as a defensive assert?

## Review round (2026-10-29): fixes for REVIEW.md D1-D8, commits a9fafde8, 506d2df1, 4602a7af, 5cb89c52

9. The exactness invariant is now enforced where it is decided, in ROWS: `_check_single_mini_batch`
   (`trainer_base.py`, called from `_compute_old_log_prob` under the skip and first thing in
   `_update_actor`) requires `len(batch) == ppo_mini_batch_size * rollout.n` after `_balance_batch`'s
   padding; the worker asserts one mini-batch + one epoch per rank before any optimizer step. Padding rows
   count: the worker trains them (zero contribution) and they decide the number of optimizer steps, so the
   rule is on the total row count, with sampled/padding counts in the message. Agent loops returning a
   varying number of trajectories per session are therefore incompatible with the skip (documented).
10. Engines: the validator rejects every strategy other than fsdp/fsdp2 (lead: raise for engines not
    verified). The reviewer's "plain Megatron accepted" expectation was adapted accordingly in
    review-tests (header note in the file). Lifting the restriction is a follow-up with a GPU run.
11. Dropout: `_hf_config_dropouts` in `verl/utils/config.py` loads the HF config with `AutoConfig` +
    `copy_to_local` (the project's own loader path, `HFModelConfig.__post_init__`); failure -> None ->
    printed WARNING (the file's existing style for notices), never a crash at validation.
12. Warmup: the skip applies only to steps that update the actor (`_skip_old_log_prob_this_step`), so
    critic-warmup steps keep today's metrics at today's cost (nothing to save there).
13. `_StubTrainer` tests now set `global_steps` and `trainer.critic_warmup`; the init-time validation test
    builds a real trainer from the composed config with `build_checkpoint_callback`,
    `RolloutMoELoadBalanceMetricsAccumulator` and `_build_replay_buffer` stubbed.
14. Trap: `ActorConfig` fields are frozen (`FrozenInstanceError`), construct with the values; and a test
    that checks what the worker was sent must snapshot `extra_info` at call time now that the keys are
    popped afterwards.

## Re-verification residuals (2026-10-29): commit 2b259b2d

15. R1: `_hf_config_dropouts` now walks every nested `PretrainedConfig` (`vars(cfg)` values, cycle-guarded):
    text_config, vision_config, audio_config, encoder/decoder configs. Keys stay bare (no sub-config
    prefix) so the reviewer's existing assertions hold; duplicates (same key and value in two towers) are
    collapsed. Tests: nested text, vision-only (reviewer's R1 case), clean VLM.
16. R2 (my call): `fsdp_turbo` is ALLOWED. `fsdp_turbo_impl.py` subclasses `FSDPEngineWithLMHead` and
    overrides only `_init_device_mesh`, parallel state, `_build_module`/`_build_fsdp_module`, the Ulysses
    config helpers, the DP rank/size/group accessors, `is_mp_src_rank_with_outputs` and `optimizer_step`;
    `forward_backward_batch`, `forward_step`, `prepare_model_outputs` and the worker return path are the
    tested FSDP code. It is GPU-untested like the whole PR (stated in the docs paragraph and the validator
    message). `is_mp_src_rank_with_outputs` is the same contract `compute_log_prob` already relies on.

## GPU evidence (2026-10-29, benchmark agent; bench/CROSSCHECK.md and packages/p2/bench/RESULTS.md agree)

17. Single-mini-batch preset (8xH100, Qwen3-1.7B, GRPO n=8, 128 prompts, ppo_mini_batch_size=128,
    use_kl_loss), means of steps 2..6, group means: training side main 15.83±0.12 s (n=2), branch OFF
    15.62±1.67 s (n=3), branch ON 14.79±2.01 s (n=5) with old_log_prob 0.00, ref 3.89±0.64, update_actor
    10.90±1.36. The removed forward (3.7 s) is partly paid back in the update pass (+1.0-1.7 s: entropy
    metric + log-prob write-back) and the ref pass read 0.5-1.0 s slower on two slow-node ON runs. Net
    -5 % to -7 % on the training side; the step clock is inside the ±5 s rollout noise. The vet's -11 %
    of step was an estimate that assumed the pass was free to remove; it is not, and the PR body says so.
18. GPU suite: 503 passed / 0 failed / 8 skipped at 1e45a508 and 5cb89c52 (main 503/0/8); the 521 once
    listed in CROSSCHECK.md counted 18 junit subtests and was corrected by the lead. No branch docs quote a gain
    (the perf_tuning.rst paragraph describes the mechanism only), so no docs commit was needed.
19. Follow-up candidate, not in this PR: `actor/entropy` optional under the skip (off by default) so the
    update pass does not inherit the entropy cost; it would change a metric's availability, which this PR
    deliberately avoided.

## Review round 2 (the submitter's independent review, 2026-10-29): commits 1ef03e93, c6067fec, ba60cf8e

20. What the review found (research/review-bundle/REVIEW.md, reproduced here with the two probes): the validator read
    the SAVED base HF config while training uses the effective model. Three accepted configurations differ between
    train and eval mode at identical weights: Qwen3 with `override_config.attention_dropout=0.5` (max |diff| 0.153),
    a dropout-free Qwen3 with a loaded LoRA adapter whose adapter_config.json has `lora_dropout=0.3` (0.065, through
    the real `_build_lora_module`), Mixtral with saved `router_jitter_noise=0.1` (0.00083: the MoE block multiplies
    the hidden states by uniform noise when `self.training`). Plus the warning-vs-guarantee inconsistency when the
    config cannot be loaded, and bench wording (stale final-sha paragraph, "paired same-node").
21. Two layers, both shipped. A (config, c6067fec): `_effective_hf_config` rebuilds the HF config exactly as
    `HFModelConfig.__post_init__` does (AutoConfig on `copy_to_local(path)`, `override_config` with the Megatron-style
    `model_config` sub-key, `update_model_config`); `_hf_config_dropouts` scans it (top level + nested
    `PretrainedConfig`s) for every numeric key > 0 whose name contains dropout / pdrop / drop_path / drop_prob /
    drop_rate / layerdrop / jitter, then `_lora_adapter_dropout` reads `adapter_config.json` when `lora_rank > 0` and
    `lora_adapter_path` is set. Checked against transformers 5.16: `router_jitter_noise` (8 configs: mixtral, phimoe,
    minimax, llama4, deepseek_v4, ...), `moe_jitter_eps` (dbrx), `layerdrop`/`encoder_layerdrop`/`decoder_layerdrop`
    (25-31 configs), `drop_path_rate` (45 configs) exist as config keys; `jitter_noise` and `stochastic_depth_prob`
    do not (module attributes only). verl's `_build_lora_module` never passes `lora_dropout` to `LoraConfig`: peft
    0.19.1 defaults it to 0.0 and builds an `nn.Identity`; > 0 builds an `nn.Dropout` under `lora_dropout.<adapter>`.
    B (worker, 1ef03e93): `verl.utils.model.find_train_mode_dependent_modules(module)` walks `named_modules()` for
    `_DropoutNd` with p > 0, class names containing DropPath / StochasticDepth with `drop_prob`/`p` > 0, and numeric
    attributes > 0 with the same name fragments (Qwen3/Llama/Mixtral attention keep `self.attention_dropout` as a
    float and pass it functionally: `dropout=0.0 if not self.training else self.attention_dropout`). Called from
    `ActorRolloutRefWorker._check_actor_for_on_policy_skip` right after `self.actor.reset()` in `init_model` (the
    earliest point where the built module and `config.actor.skip_old_log_prob_when_on_policy` meet; ONE_TO_ALL, so
    every rank; before any update). A list of modules (pipeline chunks) is walked with an `[i].` prefix.
22. Kept the helper name `_hf_config_dropouts` (the reviewer's review-tests import it and assert `(key, value)`
    lists); its docstring now says what it covers. The warning for an uninspectable config keeps the phrase "could
    not load the model's HF config" (review-tests match it) and adds that the worker enforces the rule.
23. Test trap: peft zero-initialises lora_B, so a freshly saved adapter has NO train/eval gap whatever its dropout
    (the LoRA branch outputs 0); the reviewer's probe randomises lora_B before saving, the fixtures do the same.
    Mixtral's MoE block is `model.layers.N.mlp` in transformers 5.16 (not `block_sparse_moe`).
24. Lint trap: `pre-commit` is not on PATH (it is .venv-cpu/bin/pre-commit), and the autogen-trainer-cfg hook shells
    out to bare `python3`, which needs .venv-cpu/bin first on PATH to find hydra; it passes then.
25. Regression runner: scripts/run-cpu-regression.sh hardcodes the clone (run-cpu-tests.sh) and only takes group
    names, so the worktree run used a scratch copy with RUN=run-cpu-tests-wt.sh and VERL_ROOT=wt/p2 (labels
    p2r2-reg-*); the shared scripts were not modified.
26. Results: author suite 72 passed (59 + 13 new in 2 files); review-tests 39 passed / 1 skipped; both reviewer
    probes now fail at the validator with the ValueError (tests.log). Full CPU regression: small 200 passed; utils 508 passed / 23 skipped; trainer 450 passed / 1 skipped; workers 303 passed / 40 skipped / 1 deselected; experimental 74 passed; single_controller 21 passed: 1556 passed, 64 skipped, 0 failed. The first full pass showed 1 utils failure (tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py, FileNotFoundError test_data/test_kwargs.parquet): the author-suite run finished concurrently in the same worktree and removed test_data/ as the runner does at exit; the utils group re-run alone is 508 passed / 23 skipped. Trap: never run two run-cpu-tests-wt.sh invocations on one worktree at the same time.
27. Duplicate-work check re-run 2026-10-29 (read-only gh search): no open PR implements the skip; new since the last
    check only #8227 (closed 10-14, a per-stage SkipManager that replays cached old log-probs for debugging; the
    conflict assertion it adds against bypass_mode is the same reasoning as this gate's). #8026 still open,
    complementary.
28. Not changed, and why: no rename of `_hf_config_dropouts` (review-tests); `fsdp_turbo` stays allowed; no GPU
    re-run (validation/init-only commits); the pooled 5-7 % claim stands, the pairwise -10.5 % / -12.1 % readings are
    labelled exploratory; the follow-up (entropy metric optional under the skip) stays a follow-up.

## Round-2 re-review (2026-10-29, packages/p2/REVIEW-ROUND2-VERDICT.md): commits c7031289, ba0dccf5; HEAD ba0dccf5

29. D1: `_build_lora_module` loads any peft_type through `PeftModel.from_pretrained`; a LoHa / LoKr adapter keeps
    `rank_dropout` / `module_dropout` as `{adapter: rate}` dicts on the layer (peft `LycorisLayer`), invisible to a
    numeric-only walk and absent from the validator's `lora_dropout` read (measured gap 1.9e-4). Fixed at both layers:
    the validator rejects `peft_type != "LORA"` by name (verl's `lora_rank` / `lora_alpha` config and the rollout LoRA
    sync are LoRA-specific anyway) and scans every rate key of adapter_config.json; the walk expands dict / list rate
    attributes (`rank_dropout[default]=0.5`). D3: `torch.nn.modules.batchnorm._NormBase` (BatchNorm*, SyncBatchNorm,
    InstanceNorm*) with `track_running_stats=True` flagged; without running stats both modes use batch statistics.
30. D2: wording in the PR body, docs/perf/perf_tuning.rst, the worker and helper docstrings and the config warning: the
    walk rejects what is structurally visible (dropout modules and rates, DropPath / StochasticDepth, router jitter,
    layerdrop, BatchNorm); a train-mode branch written into forward code (hard-coded F.dropout(p), a rate read from
    self.config inside forward) is not detectable, so for custom (trust_remote_code) models the skip is the user's
    assertion. The reviewer's documenting test is kept in review-tests/round2 and a branch copy pins the limit.
31. D4: CHECKLIST.md and bench/RESULTS.md no longer call 2b259b2d the final HEAD; they name HEAD ba0dccf5 and say the
    round-2 and re-review commits had no GPU run yet at that point (superseded, see 35).
32. Results: author suite 76 passed (7 files, 4 new tests); review-tests rounds 1+2 plus round2/ 46 passed / 1 skipped
    (the reviewer's 3 failing tests pass); touched regression groups re-run alone: utils 509 / 23 skipped, workers 306 /
    40 skipped / 1 deselected (previous counts + the 4 new tests); pre-commit Passed on the changed files (three E501
    wrapped first). Audit: p2 fresh / ok at ba0dccf5, 15 commits, clean.
33. Open question for the submitter (reviewer's note, out of scope): `main_ppo_v0.py` run with the default `trainer.use_v1=true`
    passes the validator while the legacy RayPPOTrainer ignores the flag (silent no-op, not inexact). Closed on the
    lead's call, see 34.
34. Legacy-trainer rejection (commit 7f0c53f5, HEAD 7f0c53f5): `RayPPOTrainer.__init__` raises a ValueError naming
    the flag and `trainer.use_v1` right after `self.config = config`; that constructor is the single choke point
    (`main_ppo_v0.py` and `verl/experimental/separation/ray_trainer.py`'s `SeparateRayPPOTrainer` both go through
    it), so no change in main_ppo_v0.py. Test `tests/trainer/ppo/test_skip_old_log_prob_legacy_trainer_on_cpu.py`:
    the composed config (shipped `trainer.use_v1=true`) passes `_validate_skip_old_log_prob_config`, construction
    raises; with the flag off the constructor proceeds to its usual role-mapping assertion (the check comes first and
    nothing else changed). Author suite 77 passed (8 files); tests/trainer 451 passed / 1 skipped; pre-commit
    Passed; audit fresh / ok at 7f0c53f5, 16 commits, clean.
35. GPU (2026-10-29, bench agent): the stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`) ran on 7f0c53f5 (bench/results/gpu-tests-p2-r2): 503 passed / 0 failed / 8 skipped + torchrun 2/2, identical to main. Stock selection = regression coverage, not an enabled-feature run; the perf numbers were measured at 1e45a508 / 5cb89c52 / 2b259b2d; product code since then (validation, the module walk, the legacy rejection) is not on the measured path. Package texts (PR-BODY, CHECKLIST, bench/RESULTS, REVIEW section 10) updated; no branch change.

## Round 3 (the AI follow-up review R2-01, 2026-10-29): commit 5532f3a4; HEAD 5532f3a4

36. Finding: with `actor.fsdp_config.qat.enable=true, mode=w4a4` both guards accepted the actor while verl's own
    `QATLinear` (verl/utils/qat/linear.py, `_fake_quantize_activation`) runs `_update_input_global_scale` only when
    `self.training`: identical input is quantized with `input_global_scale` 2688 in eval and 1344 in train (observer
    amax 1 -> 2). Config fix: the validator reads `actor.fsdp_config.qat` (falling back to `actor.engine.qat`; the FSDP
    engine reads `engine_config.qat` and `ActorConfig.__post_init__` aliases `engine = fsdp_config`) and rejects
    `enable=True` with `mode=w4a4`, naming the mode and the observers; W4A16 is weight-only (no activation observer
    branch) and stays accepted, documented as untested with the skip rather than proven. Walk fix: a module with
    `_fake_quantize_activation` whose `mode` attribute (QATMode is a str enum; `.value` or the str itself) is `w4a4`
    is flagged, so the worker check refuses it; the class name is not used for the decision. Docs and PR body updated.
37. Tests: the exact-message config test (W4A16 and disabled QAT accepted) and the walk test, which loads the
    unmodified `QATMode` / `QATLinear` class definitions from verl/utils/qat/linear.py via ast with the CUDA quantizer
    replaced by a spy (triton is not installed in the CPU venv, so the module cannot be imported), finds the W4A4
    module, has the worker refuse it, shows the eval/train scale difference and leaves W4A16 unflagged. The submitter's probe
    re-run from a scratch copy exits 1 at the validator naming w4a4. Author suite 80 passed (8 files); review-tests
    46 passed / 1 skipped; utils 511 / 23 skipped, workers 307 / 40 skipped / 1 deselected; pre-commit Passed.
    Trap: ruff-format rewraps long expressions in the same pre-commit run that reports E501, so a follow-up exact-match
    patch must be written against the formatted text (view first).
38. GPU: 5532f3a4 and e19e7949 change validation, the walk and docs only (off the measured path; perf stays measured at
    2b259b2d and earlier). The stock GPU unit-test selection (regression coverage) ran on the final HEAD e19e7949 (bench/results/gpu-tests-p2-r3-final, 8x H200): 503 passed / 0 failed / 8 skipped (junit 529 / 521 / 0 / 8) + torchrun 2/2, identical to main.

## Round-3 re-review (packages/p2/REVIEW-ROUND3-VERDICT.md, R2-01 FIXED / MERGE-READY, one LOW): commit e19e7949; HEAD e19e7949

39. D1: a programmatic DictConfig can carry `QATMode.W4A4` itself (YAML / CLI cannot); `str(enum).lower()` is
    "qatmode.w4a4", so the validator accepted it while `apply_qat` lowercases the str-enum and applies W4A4 (the
    walk already refused it by `mode.value`). Fixed by comparing `str(getattr(mode, "value", mode)).lower()`;
    the branch test gained the enum case (a local `str, Enum` mirroring QATMode, since tests/utils has no ast
    loader); the reviewer's 11 round-3 tests all pass. Author suite 80 passed; review-tests rounds 1-3 57 passed /
    1 skipped; utils 511 / 23 skipped; pre-commit Passed; audit fresh / ok at e19e7949, 18 commits, clean. The
    reviewer's no-defect note (`actor.qat` is inert in the v1 FSDP engine, `fsdp_config.qat` is the live key) is
    recorded in REVIEW.md; no docs sentence added (keeping the change minimal). GPU: see 38.
