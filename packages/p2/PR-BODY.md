### What does this PR do?

Adds `actor_rollout_ref.actor.skip_old_log_prob_when_on_policy` (default `false`): the V1 trainer skips
the separate old-log-prob forward pass (`timing_s/old_log_prob`) when the actor update is strictly
on-policy, i.e. the whole sampled batch is updated in **one** optimizer step (`ppo_epochs=1`,
`ppo_mini_batch_size == train_batch_size`: the shape of the shipped Qwen3.5 / DeepSeek-V4 / gpt-oss /
Qwen3-235B GRPO examples). Every micro-batch of the update pass then runs at the sampling weights, so
the log-probs it computes *are* the old log-probs:

- `_compute_old_log_prob` runs no forward; `ppo_loss` uses `log_prob.detach()` as `old_log_prob`
  (ratio exactly 1, clipping inert; `actor/pg_clipfrac` and `actor/ppo_kl` are 0 by construction).
- The update pass returns its detached per-token log-probs and entropy through the existing
  TransferQueue bridge (the `compute_log_prob` path), so `old_log_probs`, `actor/entropy`,
  `training/rollout_*` and the metrics-only `rollout_corr/*` keep their definitions; they are now
  computed in the update pass (same weights, same tensors). The entropy is computed under `no_grad`
  when `entropy_coeff == 0`.

Compatibility conditions. The flag is rejected at config validation and again at trainer init, naming
every violated condition; FSDP / FSDP2 / fsdp_turbo engines only; no `auto` mode; nothing changes with
the flag off (the `_generated_*.yaml` only gain the key with its `false` default):

- `ppo_epochs == 1`, `ppo_mini_batch_size == train_batch_size / parameter_sync_step`, no `actor.shuffle`.
  Per step the trained batch must be exactly `ppo_mini_batch_size * rollout.n` rows after `_balance_batch`
  padding: agent loops returning a varying number of trajectories per session pad to a second mini-batch
  and are refused with the row counts (trainer, before any forward; worker, before any optimizer step).
- Nothing reads `old_log_probs` before the update: no `calculate_sum_pi_squared`, `use_kl_in_reward`,
  `rollout_correction` with `bypass_mode` / `rollout_is` / `rollout_rs` (the metrics-only default is
  fine), `optimal_token_baseline` estimators, distillation, router replay `R2`, `separate_async` mode or
  the legacy trainer; the legacy `RayPPOTrainer` (`main_ppo_v0.py`, which runs with the shipped
  `trainer.use_v1=true` default) also refuses the flag at construction, so it cannot be a silent no-op there.
- W4A4 QAT is rejected (`actor.fsdp_config.qat.enable=True` with `mode=w4a4`: verl's `QATLinear` runs its
  activation observer in train mode only, so the update pass would quantize with different scales than
  the eval-mode old-log-prob pass; the module walk also flags a W4A4 `QATLinear`). W4A16 is weight-only
  and accepted, but untested with the skip.
- The actor's forward must not depend on train/eval mode, because the stored old log-probs are the
  update pass's train-mode values. The validator inspects the **effective** model: the HF config after
  `override_config` (every nested sub-config) for dropout, MoE router jitter (Mixtral's
  `router_jitter_noise`, DBRX's `moe_jitter_eps`), layer drop and drop path, and a `lora_adapter_path`
  adapter's train-mode settings (only `peft_type=LORA` adapters are accepted, LoHa / LoKr `rank_dropout` /
  `module_dropout` being train-mode rates; `lora_dropout` must be 0; verl's own LoRA config sets none).
  `ActorRolloutRefWorker.init_model` then walks the built module on every rank before any update and
  rejects every train-mode-dependent module or rate that is visible structurally (dropout modules and
  rates, dict-valued ones included, DropPath / StochasticDepth, router jitter, layerdrop, BatchNorm with
  running statistics), naming the module paths. When the config cannot be loaded at validation time a
  warning says so. Behaviour hidden in forward code (a hard-coded `F.dropout(p)`, a rate read from the
  config inside `forward`) is not detectable, so for custom (`trust_remote_code`) models the skip remains
  the user's assertion.

History: the legacy worker shipped this shortcut in #3119 (merged; removed with the legacy workers in
#6067); #3996 / #6280 ask for it back (#3119 only avoided the mismatch; this PR also removes the
forward). #8026 (batch-invariant actor forward) is complementary: it makes old == new by kernel
determinism but keeps the pass. #8364 adds a consumer of `old_log_probs` (`calculate_sum_pi_squared`),
which the gate rejects. #8227 (closed) is a debug SkipManager that replays cached old log-probs from a
dump directory, a different purpose.

### Checklist Before Starting

- [x] Search for similar PRs (re-run 2026-10-29, read-only):
  https://github.com/verl-project/verl/pulls?q=is%3Apr+%22skip+old+log+prob%22 ,
  https://github.com/verl-project/verl/pulls?q=is%3Apr+old_log_prob ,
  https://github.com/verl-project/verl/pulls?q=is%3Apr+skip_old_log_prob ,
  https://github.com/verl-project/verl/issues?q=is%3Aissue+old_log_prob+on-policy .
  No open PR implements the skip. Nearest: #3119 (merged, legacy), #6519 (stale, entropy no-grad only),
  #8026 (complementary), #8364 (gated consumer), #8227 (closed, debug replay).
- [x] Title `[trainer, perf] feat: skip the old-log-prob pass when strictly on-policy` (not
  `[BREAKING]`: a new key with a `false` default; no signature changes).

### Test

CPU unit tests (`*_on_cpu.py`, collected by `cpu_unit_tests.yml`), 80 tests in 8 files:

- `tests/workers/test_ppo_loss_on_policy_skip_on_cpu.py`: exactness through the real `ppo_loss` on a
  tiny Qwen3 (fp32): the update pass's log-probs (train mode, micro-batches 3/3/2, gradient
  accumulation) equal a separate eval/no-grad pass (2/2/2/2) **bitwise**, as does the gradient with
  `old_log_prob = log_prob.detach()`; `pg_clipfrac == ppo_kl == 0`; negative control: after one AdamW
  step the next mini-batch's log-probs differ (~4e-3), which is why only one mini-batch is exact.
- `tests/utils/test_skip_old_log_prob_config_on_cpu.py`: the shipped config with the flag is accepted
  (incl. `validate_config`); each of the 18 inexact conditions is rejected by name (W4A4 QAT with its
  exact observer message; W4A16 and disabled QAT accepted), several at once listed together.
- `tests/utils/test_skip_old_log_prob_effective_config_on_cpu.py`: effective-model validation: dropout
  enabled through `override_config` (plain key and `model_config` sub-key), a saved Mixtral
  `router_jitter_noise`, and a peft-written adapter with `lora_dropout=0.3` are rejected by name; an
  override that disables a saved dropout, a zero-dropout adapter and verl's own LoRA config are
  accepted; a LoHa adapter is rejected by type (`peft_type=LOHA`) and by its `rank_dropout`; an
  unreadable adapter warns that the worker enforces the rule; layerdrop and drop-path keys are detected.
- `tests/workers/test_skip_old_log_prob_module_walk_on_cpu.py`: the worker guarantee on real tiny
  models: Qwen3 functional attention dropout, Mixtral router jitter and an adapter loaded by the
  engine's real `_build_lora_module` are found (each with a measured train/eval log-prob gap at
  identical weights; the clean model has none); the module built from the effective config is what
  gets walked; a LoHa adapter's dict-valued `rank_dropout` through the real loader; dropout modules,
  DropPath, BatchNorm / InstanceNorm (only with running statistics) and pipeline chunks; the worker
  check refuses, names at most five modules, does nothing with the flag off; a W4A4 `QATLinear` (verl's
  unmodified class loaded from source with the CUDA quantizer replaced by a spy) is found and refused, with
  the eval/train scale difference shown, while W4A16 is not flagged; the documented limit (a hard-coded
  `F.dropout` inside `forward` is outside a structural walk).
- `tests/trainer/ppo/v1/test_skip_old_log_prob_on_cpu.py`: trainer wiring with an in-memory
  TransferQueue stand-in: no forward under the skip, metrics read off the returned KVBatchMeta,
  row-count refusal before any forward, critic-warmup steps run the old pass, init-time validation.
- `tests/trainer/ppo/test_skip_old_log_prob_legacy_trainer_on_cpu.py`: the legacy trainer refuses the flag
  at construction, naming `trainer.use_v1`, and proceeds as before with the flag off.
- `tests/workers/test_engine_workers_return_model_output_on_cpu.py` and
  `tests/workers/test_fsdp_entropy_requires_grad_on_cpu.py`: the worker return path with the
  one-mini-batch guard; the no-grad entropy path (same values, no graph, default unchanged).

```
pytest -c pytest.ini -s --asyncio-mode=auto <the 8 files above>            -> 80 passed
full CPU regression (the cpu_unit_tests.yml selection on macOS; megatron files excluded): 1556 passed, 64 skipped, 0 failed
  (main + the new tests); the utils, workers and trainer groups re-run after the last fixes: 511 / 23 skipped, 307 / 40 skipped, 451 / 1 skipped
pre-commit (ruff, ruff-format, mypy, license, docstrings, autogen-trainer-cfg, ...) on the changed files: Passed
```

GPU: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and
`tests/experimental`, so it does not execute this PR's new tests): 503 passed / 0 failed / 8 skipped at
1e45a508, at 5cb89c52, at 7f0c53f5 and at the final HEAD e19e7949 (8x H200, + torchrun 2/2), equal to the main
reference. Enabled-feature GPU evidence is the timing runs
under Benchmark (6-step runs; no quality-parity run).

### API and Usage Example

```bash
# one optimizer step per sampled batch: ppo_mini_batch_size == train_batch_size, ppo_epochs == 1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true \
    ...
```

Any inexact configuration fails at startup with every violated condition, for example:

```
ValueError: actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True requires a strictly on-policy
update, but: actor.ppo_mini_batch_size=32 != data.train_batch_size / parameter_sync_step = 128 (...);
the effective actor model (HF config after override_config, LoRA adapter) has train-mode-only settings
(attention_dropout=0.5): the update pass runs in train mode, so its log-probs would be dropout- or
noise-sampled instead of the eval-mode old log-probs
```

### Design & Code Changes

- `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml`, `_generated_*.yaml`: the key.
- `verl/utils/config.py`: `_validate_skip_old_log_prob_config` (called from `validate_config` and from
  `PPOTrainer.__init__`; one `ValueError` lists every violated condition); `_hf_config_dropouts` scans
  the effective HF config (built as `HFModelConfig.__post_init__` builds it) and the LoRA adapter config.
- `verl/utils/model.py`: `find_train_mode_dependent_modules(module) -> [(path, reason)]`;
  `verl/workers/engine_workers.py`: `ActorRolloutRefWorker._check_actor_for_on_policy_skip`, run in
  `init_model` right after the actor engine is built; `train_batch` keeps `model_output` when
  `return_model_output`; `train_mini_batch` asserts one mini-batch and one epoch per rank before any
  optimizer step and returns the row-aligned per-token outputs next to the metrics.
- `verl/trainer/ppo/v1/trainer_base.py`: `skip_old_log_prob`, `_check_single_mini_batch` (row-count
  invariant), `_skip_old_log_prob_this_step` (critic warmup runs the old pass), `_store_actor_log_prob`
  (the former tail of `_compute_old_log_prob`, shared with `_update_actor`), deferred metrics-only
  rollout correction.
- `verl/trainer/ppo/ray_trainer.py`: the legacy trainer raises on the flag at construction (it never
  implements the skip).
- `verl/workers/utils/losses.py`: `ppo_loss` keys on the explicit `on_policy_skip_old_log_prob` flag
  (a missing `old_log_probs` still raises); no entropy-loss term or metric without a coefficient.
- `verl/workers/engine/fsdp/transformer_impl.py`: `entropy_requires_grad` (default `True`, unchanged);
  under `no_grad` the entropy skips checkpointing and keeps `inplace_backward`. The fused-kernels branch
  keeps its grad-attached entropy (no memory saving there).
- `docs/perf/perf_tuning.rst`, `docs/examples/config.rst`.

Why not `bypass_mode`: it replaces the actor's old log-probs with the rollout engine's, a different
estimator; this PR keeps PPO's estimator and removes only the redundant recomputation.

### Benchmark

8xH100 80GB (Modal, `verlai/verl:uv.cu130.dev5`), Qwen3-1.7B, GRPO n=8, `train_batch_size=128 =
ppo_mini_batch_size` (one optimizer step per step), prompt 1024 / response up to 2048, `use_kl_loss`,
6 steps; means of steps 2..6 per run, then pooled across runs (seconds; training side =
`old_log_prob + ref + update_actor`). Raw records: `bench/results/*/metrics.jsonl`.

| arm (runs) | old_log_prob | ref | update_actor | training side | step |
|---|---|---|---|---|---|
| main @2ab35292 (2) | 3.68 | 2.89 | 9.26 | 15.83 | 31.78 |
| branch, flag OFF (3) | 3.88 | 2.89 | 8.85 | 15.62 | 32.46 |
| branch, flag ON (5) | **0.00** | 3.89 | 10.90 | **14.79** | 31.94 |

- The old-log-prob forward is removed (3.4-4.7 s -> 0.00 s on every ON run). `update_actor` grows by
  1.0-1.7 s: the entropy metric and the log-prob write-back to TransferQueue now run in the update pass.
  `ref` read 0.5-1.0 s slower on the ON arms that ran later; this code does not touch it.
- Pooled training-side gain: **-5.3 % vs flag OFF, -6.6 % vs main**. Whole-step deltas are inside the
  ±5 s rollout noise of these runs. Pairwise readings in one scheduling window (-10.5 % / -12.1 %) are
  exploratory: similar windows do not prove the same physical node.
- Peak actor memory 14.9-15.1 GB on every arm. `actor/pg_clipfrac` and `actor/ppo_kl` are exactly 0 on
  the ON arms; entropy, `kl_loss` and `rollout_probs_diff` are similar across arms, which does not
  prove unchanged training signals (the bitwise CPU test does, for the single-mini-batch regime).

Arms ran at 1e45a508, 5cb89c52 and 2b259b2d: the perf numbers were measured at 1e45a508 / 5cb89c52 / 2b259b2d; product code since then (validation, the module walk, the legacy rejection) is not on the measured path and was not re-timed. The stock GPU unit-test
selection (regression coverage, see Test) also ran on 7f0c53f5 and on the final HEAD e19e7949: 503 passed / 0 failed /
8 skipped + torchrun 2/2, identical to main.

Limitations: FSDP-family engines only; one mini-batch per step (no partial first-mini-batch skip); the
entropy metric moves into the update pass (making it optional under the skip would recover part of
that cost; left out to keep every metric's definition); the enabled path's GPU evidence is 6-step timing
runs, not a quality-parity run.

### AI assistance

Written with Claude (Anthropic) under the author's direction. Commits carry a Co-authored-by trailer.

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply [pre-commit checks](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md#code-linting-and-formatting) on the changed files; `scripts/generate_trainer_config.sh` regenerated the `_generated_*.yaml`.
- [x] Add / Update [the documentation](https://github.com/verl-project/verl/tree/main/docs): `docs/perf/perf_tuning.rst`, `docs/examples/config.rst`.
- [x] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/verl-project/verl/tree/main/.github/workflows): 8 `*_on_cpu.py` files collected by `cpu_unit_tests.yml`. No new GPU e2e script: `tests/special_e2e` GRPO runs cover the default (flag off) path; the flag's effect is the benchmark above.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [x] Not related to the `recipe` submodule.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
