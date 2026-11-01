### What does this PR do?

**Behaviour change (hence `[BREAKING]`).** Under the default V1 trainer, `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm` (the Dr. GRPO setting in `docs/algo/grpo.md`) with the default `loss_scale_factor=null` normalizes every micro-batch by **its own** max response length, not by the configured `response_length` the yaml promised. The loss and the accumulated gradient of a mini-batch therefore depend on how it was split into micro-batches and drift from step to step, for the actor and the critic, on FSDP and Megatron. This PR makes `validate_config` resolve a null factor to `actor_rollout_ref.rollout.response_length` for the actor and the critic, so every micro-batch is normalized by the same constant.

Mechanism (lines at `2ab35292`): `verl/workers/utils/padding.py:113-119` pads a nested micro-batch to `max_response_len` when present, otherwise to the micro-batch's `response_lens.max()`, and only the legacy trainer's `left_right_2_no_padding` sets it; `verl/workers/utils/losses.py:59,103` (actor) and `:199,222` (critic) pad to that width; `verl/trainer/ppo/core_algos.py:1190-1193` (`agg_loss`) divides by `loss_mask.shape[-1]` when the factor is None. The yaml comment "If null, uses response_length" (#4360) held under the legacy trainer (dense inputs padded to `response_length`, #5366 / #5417) and stopped holding when #6823 made V1 the default.

**What changes for users.** Affected: V1 runs with `seq-mean-token-sum-norm` and a null factor (in tree: `examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh`). Their losses and gradients get **smaller**: divided by `response_length` instead of the micro-batch's max width (never larger), so with an unchanged learning rate the effective step shrinks by `(micro-batch max width) / response_length` (5-10x for 200-400-token responses under a 2048 budget); `actor/pg_loss`, `actor/grad_norm`, `critic/vf_loss` and `actor/entropy` (aggregated with the same mode and factor) shrink accordingly. The old scale varied per micro-batch and per step, so it cannot be restored exactly; to keep a comparable scale set `actor_rollout_ref.actor.loss_scale_factor` explicitly (never overridden). Unchanged: runs that set the factor, every other `loss_agg_mode` (the default `token-mean` included) and the legacy trainer. No CLI / config-key / signature change; the yaml default stays `null` (`--cfg job` prints `null`); the resolved value is logged at INFO. Precedent for `[BREAKING]` on a normalization change: #6823 (merged); #7916 (open) uses the same tag.

**Validation added.** An explicit factor must be a finite positive number that torch takes as a scalar: `0`, a negative value, `nan`, `inf`, a string, a bool or an int beyond int64 (`2**63` and above: torch raises `OverflowError` on `loss /= loss_scale_factor`) raise a `ValueError` naming the key and the value (before, `nan` and `inf` passed a `<= 0` check and a string raised a late `TypeError` inside `agg_loss`). A null factor is only resolved from a positive-integer `response_length`: `0`, a negative value, `2.5` or a bool raise, naming `actor_rollout_ref.rollout.response_length`; an unset `response_length` keeps the warning-only behaviour.

**Scope and limitations.** The fix lives in `validate_config` because that is the hook the in-tree launchers run before the worker dataclasses are built and the only place that sees the actor, the critic and `rollout.response_length` together; `agg_loss`, `padding.py` and `losses.py` are untouched. A custom launcher that composes the config and builds `ActorConfig` without `validate_config` keeps the old per-micro-batch horizon and must set `loss_scale_factor` itself (stated in the yaml comments and `docs/algo/grpo.md`). Verified in tree: `main_ppo.py:181` validates before the V1/V0 split; `main_ppo_v0.py:179` and `experimental/one_step_off_policy/main_ppo.py:52` resolve the config and then validate (both orders tested, same result); `experimental/fully_async_policy` bypasses `validate_config` but updates through `RayPPOTrainer._update_actor` -> `left_right_2_no_padding` (`ray_trainer.py:1327-1335`), which carries `max_response_len`. Not examined: the verl-recipe runners (a submodule; several call `validate_config`, and `partial_rollout` / `gvpo` use this mode with their own configs). Injecting `max_response_len = response_length` into the TensorDict instead was tried and fails (`losses.py:103` / `:222` pad the other fields to the micro-batch max: `RuntimeError: The size of tensor a (16) must match the size of tensor b (12)`). Known non-invariant policy losses left alone: `geo_mean` (#7913 / #7916), `kl_cov` (#7829 / #7830), `clip_cov` (random per-micro-batch selection, unfiled).

Related: #3741 (open issue on the Dr. GRPO divisor), #6157 (closed 2026-04-27, unmerged: `agg_loss`-level tests and docstring for this mode, no V1 horizon change), #2435 (open, stale, legacy-era constant normalizer), #4721 (open, stale, `agg_loss`-level accumulation tests with equal widths), #4294 (open, metric aggregation in the legacy `dp_actor` / `dp_critic`), #6957 (critic normalization; its test docstring notes that differing padded widths need a constant factor), #5417 (the default chosen when the padded width was `response_length`), #6823 (V1 default). Not a collision with #8176 (token-mean in bypass mode) or #6349 (NaN masking in the seq modes): different lines, different defects.

### Checklist Before Starting

- [x] Search for similar PRs (read-only `gh pr list` / `gh issue list`, 2026-10-29): https://github.com/verl-project/verl/pulls?q=is%3Apr+loss_scale_factor , https://github.com/verl-project/verl/pulls?q=is%3Apr+seq-mean-token-sum-norm , https://github.com/verl-project/verl/pulls?q=is%3Apr+max_response_len , https://github.com/verl-project/verl/issues?q=loss_scale_factor , https://github.com/verl-project/verl/issues?q=seq-mean-token-sum-norm . No open or merged PR or issue fixes the V1 horizon; the closest, #6157, is closed and test-only (see "Related").
- [x] Format the PR title as `[{modules}] {type}: {description}` (`tests/special_sanity/check_pr_title.py` accepts it, `[BREAKING]` included).

### Test

Strongest evidence: the new CPU test drives the real `ppo_loss` / `value_loss` with nested (V1) inputs (B=8, response lengths `[12, 1, 3, 7, 12, 2, 9, 5]`, a masked tool span, `response_length=16`) through configs built the way the trainer builds them (compose `ppo_trainer`, `validate_config`, `omega_conf_to_dataclass`), summing per-micro-batch losses and accumulating gradients as the engine does. On `main`, `loss(split) / loss(whole)` is 8.996 (split [1,2,5]) and 48.2 (singles) for the actor and 1.155 / 2.076 for the critic, with the four other modes and the equal-width split exact; on this branch every mode, split and kind is exact and the normalizer equals `response_length` for every micro-batch.

```bash
pytest -s --asyncio-mode=auto tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py
# this branch: 62 passed, 2 xfailed (geo_mean, kl_cov: non-strict, see below); ~9 s, no model snapshot needed
# main @ 2ab35292, same file: 29 failed, 33 passed, 2 xfailed (4 uneven-split invariance cases, 8 horizon cases,
#   4 resolution / warning cases, 13 rejection cases)
pytest -s --asyncio-mode=auto tests/trainer   # 500 passed, 1 skipped, 2 xfailed
pytest -s --asyncio-mode=auto tests/workers   # 278 passed, 40 skipped (megatron / vllm / sglang files skipped on CPU)
pytest -s --asyncio-mode=auto tests/utils     # 473 passed, 23 skipped
pre-commit run --files <the five changed files>   # all 14 hooks passed, nothing regenerated
```

CPU: macOS arm64, Python 3.12, torch 2.13.0 CPU. GPU: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`, so it does not execute this PR's new tests) was run on 8xH200 against the branch head `9c99205c`: 503 passed / 0 failed / 8 skipped plus the torchrun pair 2/2, identical to `main` (an earlier run at `6684917e` gave the same counts). No GPU A/B of a Dr. GRPO run was made; suggested for reviewers: GRPO with the Dr. GRPO overrides, `main` vs this branch, same seed; on `main` `actor/pg_loss` and `actor/grad_norm` are inflated by `response_length / (micro-batch max width)` per micro-batch; timing must be identical.

The two xfails record other open PRs (#7916, #7830) and are non-strict so that their merge cannot fail an unrelated CPU job (`cpu_unit_tests.yml` runs `pytest -x`); drop the marks when they land.

### API and Usage Example

No API change. The Dr. GRPO recipe is unchanged; the factor is resolved and logged at config validation:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=False \
    data.max_response_length=2048 ...
# INFO verl.utils.config: actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm with loss_scale_factor=null:
#   resolved loss_scale_factor to actor_rollout_ref.rollout.response_length=2048 so every micro-batch is normalized by the same constant.
# Another constant: actor_rollout_ref.actor.loss_scale_factor=4096 (kept as given; must be a finite number > 0).
```

### Design & Code Changes

- `verl/utils/config.py`: `_resolve_loss_scale_factor(config, use_critic)` next to the other `_validate_*` helpers, called from `validate_config` before the worker dataclasses are built. For the actor and, when `use_critic`, the critic: if `loss_agg_mode == "seq-mean-token-sum-norm"` and `loss_scale_factor` is None, set it to `rollout.response_length` (which must be a positive integer) and log it; an explicit factor must satisfy `_is_finite_positive_number` (a finite float > 0, or an int in `1..2**63-1`, the range torch accepts as a scalar; not a bool). The critic is resolved on its own because it inherits the actor's keys through `oc.select` but can be set directly. The call is idempotent. This is the first place `validate_config` mutates the config and the first `logging` use in the module (siblings `print`), both deliberate: a resolved default must be visible, and the mutation is one null key.
- `verl/trainer/config/actor/actor.yaml`, `verl/trainer/config/critic/critic.yaml`: the `loss_scale_factor` comments state the resolution, the finite-number requirement and the bypass boundary (`_generated_*.yaml` unchanged: no value changed).
- `docs/algo/grpo.md`: the Dr. GRPO `loss_scale_factor` line.
- `tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py` (new): loss and gradient invariance for 5 modes x 3 splits x actor/critic (30); the normalizer equals `response_length` for every micro-batch (8); config resolution and validation (24: null -> `response_length` for actor and critic, explicit `4096` / `0.5` / `2**63-1` kept, other modes untouched, critic alone, a directly overridden `response_length`, the unset-`response_length` warning, explicit `nan` / `inf` / `-inf` / `0` / `-1` / `"sixteen"` / `True` / `2**63` / `10**400` rejected, inferred `0` / `-16` / `2.5` / `True` rejected); two non-strict xfails.

### AI assistance

Written with Claude (Anthropic) under the author's direction. Commits carry a Co-authored-by trailer.

Suggested reviewers: tongyx361 (divisor semantics, #5366 / #5417) and wuxibin89 (V1 trainer, #6957).

### Checklist Before Submitting

- [x] Read the [Contribute Guide](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md).
- [x] Apply [pre-commit checks](https://github.com/verl-project/verl/blob/main/CONTRIBUTING.md#code-linting-and-formatting).
- [x] Add / Update [the documentation](https://github.com/verl-project/verl/tree/main/docs): `docs/algo/grpo.md`, yaml comments.
- [x] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/verl-project/verl/tree/main/.github/workflows): the new `*_on_cpu.py` file runs under `cpu_unit_tests.yml`.
- [ ] Once your PR is ready for CI, send a message in the `ci-request` channel.
- [ ] Recipe submodule: not touched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
