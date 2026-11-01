# Adversarial review: contrib/p0-loss-horizon (2026-10-29)

Branch `contrib/p0-loss-horizon` @ c57eecaf (base main @ 2ab35292), worktree `<repo>/wt/p0`.
Reviewer ran everything read-only on the worktree and the main clone (`git status --porcelain` empty on both, before
and after). Review tests: `packages/p0/review-tests/` (3 files, 21 tests, all pass on the branch). Logs:
`packages/p0/review-tests/logs/`.

## 1. Verdict: NEEDS-FIXES (small, no blocker in the fix itself)

The fix is correct and proven: the new file fails on main for exactly the right 14 cases and passes on the branch,
the three CPU regression groups are unchanged (483 / 278 / 473), and 21 adversarial tests (bypass paths, explicit
factors, critic-only mode, other modes at the loss level, V0 dense layout, Megatron config) pass. The single most
important reason it is not merge-ready as packaged: the two **strict xfails** in the new test
(`test_known_non_invariant_policy_losses[geo_mean|kl_cov]`, test file :215-243) encode the current state of two
*other* open PRs (#7916, #7830, both OPEN on 2026-10-29). The day either merges, the case XPASSes, `strict=True`
turns it into a failure, and `pytest -s -x` in `.github/workflows/cpu_unit_tests.yml:118` aborts the whole CPU job for
that unrelated PR. Drop the two cases (they exercise nothing this PR changes) or set `strict=False`. Second: the PR
body and the yaml comment must state the coverage boundary (the resolution happens only where `validate_config`
runs; see D2). Both fixes are minutes of work; nothing in `verl/utils/config.py` needs to change.

## 2. Defects

| # | Sev | file:line | Failing input / evidence | Proposed fix |
|---|---|---|---|---|
| D1 | medium, CI | `tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py:215-243` | Merge of #7916 or #7830 -> XPASS -> strict FAIL -> `cpu_unit_tests.yml:118` runs `pytest -s -x tests/` -> the CPU job of the *other* PR aborts at this file. Also fragile on data: `kl_cov` selection is deterministic only for this seed/torch version. | Delete both xfail params (out of scope), or `strict=False`. Keep the issue numbers in the PR body's "Related" instead. |
| D2 | medium, coverage | `verl/utils/config.py:233-234` (placement) | A V1 launcher that composes `ppo_trainer` and builds `ActorConfig` without `validate_config` keeps `loss_scale_factor=None`: review test `test_worker_config_built_without_validate_config_keeps_the_micro_batch_horizon` reproduces the 8.996x split ratio **on the branch**. In-tree every V1 launcher calls it (`main_ppo.py:181`, `main_ppo_v0.py:179`, `experimental/one_step_off_policy/main_ppo.py:52`); `fully_async_policy` (`FullyAsyncTrainer` < `SeparateRayPPOTrainer` < `RayPPOTrainer`, DataProto + `left_right_2_no_padding` at `ray_trainer.py:1255-1381`, `max_response_len` carried at `padding.py:49-51`) is dense-layout and never had the bug; verl-recipe runners that skip it (flowrl, entropy, atropos) are the same V0 lineage; 13 recipe files import `validate_config`. So the gap is custom / library launchers only. | Keep the placement (§3) but say so: PR body "Scope" sentence + yaml comment "resolved when validate_config runs (main_ppo and the other in-tree launchers); a launcher that bypasses it must set loss_scale_factor". |
| D3 | low, docs | `verl/trainer/config/actor/actor.yaml:88-90`, `critic/critic.yaml:66-68`, `docs/algo/grpo.md:60` | `python -m verl.trainer.main_ppo --cfg job` and `_generated_ppo_trainer.yaml:102/:625` still print `null`; the effective value is visible only in the `verl.utils.config` INFO line. The comment reads as if the yaml itself resolves. | Reword to "null = resolved to actor_rollout_ref.rollout.response_length by validate_config (logged at INFO)". |
| D4 | low, PR body | `packages/p0/PR-BODY.md` title / "Behaviour change" paragraph | For V1 Dr. GRPO runs the gradient scale changes by `response_length / micro-batch width` (5-10x on a 2048 budget): an effective learning-rate change for anyone who tuned under the bug. Precedent for `[BREAKING]` on normalization changes: #7916 `[BREAKING][algo] fix: require minibatch normalization for geo_mean`; #6823 `[BREAKING][trainer, cfg]`. | Owner's call. Minimum: move the "Behaviour change, stated plainly" paragraph to the top of "What does this PR do?". Adding `[BREAKING]` is defensible and the checker accepts it. |
| D5 | low, metrics | `verl/trainer/ppo/v1/trainer_base.py:1678-1683`, `verl/trainer/ppo/ray_trainer.py:1591-1594` | `actor/entropy` is aggregated with the actor's mode and factor on `data.to_padded_tensor()`; under the norm mode it is now divided by `response_length` instead of the padded batch width. Consistent with the loss, but a visible metric change absent from the PR body. | One sentence in the behaviour-change paragraph. |
| D6 | low, test robustness | test file `:44,:57`; `verl/workers/config/model.py:156-188` | The snapshot is needed only because `omega_conf_to_dataclass(config.critic)` instantiates `HFModelConfig` (copy_to_local, tokenizer, processor, generation_config, AutoConfig). CI's image has it (`tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py:42` loads it unconditionally and CI is green), so the test does run in CI; locally it skips silently. Runtime: 66 s here on the branch, 37 s on main (author: 20 s). | Optional: `critic.model.load_tokenizer=false` + a tmp dir with a tiny qwen2 `config.json` makes the file snapshot-free and faster. |
| D7 | nit | `verl/utils/config.py:106` | `loss_scale_factor=0` (or negative) is "set" and kept; `agg_loss` then divides by zero (`core_algos.py:1193`). Pre-existing, but this helper is now the one place that inspects the value (probe: "explicit 0 kept -> 0"). | Optional `if factor is not None and factor <= 0: raise ValueError(...)`. |
| D8 | nit, style | `verl/utils/config.py:15-22,109-121,158-291` | First `logging` use in the module while sibling notices use `print` (`:259`, `:267`); and `validate_config` mutates the config for the first time (no other assignment in `:158-291`). | Acceptable; say in the PR body why the resolve step sits inside `validate_config` (the one hook shared by all three launchers). |
| D9 | nit, NOTES | `packages/p0/NOTES.md` "Decisions" bullet 2 | "an explicit assignment makes the critic value a plain int in both cases" is false: when the actor resolves first the critic's `oc.select` interpolation stays (review test `test_critic_keeps_the_interpolation_and_use_critic_false_still_inherits`). Harmless: the typed dataclass merge resolves it and `main_ppo.py:149` `OmegaConf.resolve` materializes it. | Fix the note; no code change. |

No defect was found in `_resolve_loss_scale_factor` itself after a genuine attempt (edge probes: missing
`rollout.response_length` -> warning + untouched; float `16.0` -> `16`; second call idempotent; `use_critic=False`
skips the critic section; critic-only mode resolves the critic alone; explicit actor and critic values kept
independently).

## 3. Placement: validate_config vs the consumer

| Option | Covers | Cost / risk |
|---|---|---|
| **A. `validate_config` (chosen)** | `main_ppo` V1 + V0, `one_step_off_policy`, every verl-recipe runner that imports `validate_config` (dapo, prime, spin, sppo, gvpo, spo, rep_exp, specRL, partial_rollout, async_flow, echo, dance_grpo x2) | The only site that sees `actor`, `critic` and `rollout.response_length` together; 30 lines; no edit to `agg_loss` / `losses.py` / `padding.py`, so no conflict with #8176, #6349, #7916, #7830; idempotent. Risks: "validate" now mutates (D8); bypassing V1 launchers keep the old horizon (D2); invisible in `--cfg job` (D3). |
| B. yaml default `loss_scale_factor: ${oc.select:actor_rollout_ref.rollout.response_length,null}` (critic inherits) | Every hydra-composed launcher, custom ones included; visible in `--cfg job` | Zero Python, but the factor becomes non-null for all five modes: unused by `agg_loss` outside the norm branch (`core_algos.py:1189`, review test bit-equal) and the metric-aggregation flip at `losses.py:85-93/:210-218` is already SUM under both engines (`fsdp/transformer_impl.py:734-739`, `megatron/transformer_impl.py:802-808`), yet it rewrites `_generated_*.yaml`, the documented `null` default, needs `[cfg]`, and removes the (useless) "per-micro-batch width" option. |
| C. consumer side (`ppo_loss` / `value_loss` or `ActorConfig.__post_init__`) | would cover everything | Not constructible: neither the dataclasses nor the loss functions know `response_length`; V1 never carries `max_response_len` (`trainer_base.py:1851-1862` extra_info keys), and injecting it crashes `losses.py:103/:222` (K12 §2.5, reproduced by the author). Needs a new config field, i.e. more surface than A or B. |
| D. hard error in `ActorConfig.__post_init__` when mode is norm and factor is None | everything | Breaks the documented recipe (`examples/grpo_trainer/README.md:35`, `examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh:76`) -> `[BREAKING]`; tongyx361 explicitly chose the "reasonable default" in #5417 (b8d91ef8). |

**Recommendation: keep A.** Defence for maintainers: "the horizon must be `rollout.response_length` (the width V0
always had, `agent_loop.py:793-807`), which only the trainer-level config knows; `validate_config` is the one hook all
three in-tree launchers run before building worker dataclasses, and it leaves the loss code untouched so the
competing fixes in `agg_loss`/`losses.py` stay mergeable. A yaml-level default (B) was considered and rejected
because it changes the recorded config for every mode." Have B ready as the fallback if a reviewer objects to
`validate_config` mutating, and D if tongyx361 prefers explicitness over a resolved default.

## 4. Merge-process gaps

- PR template: all five headers present; AI-assistance statement, non-duplicate rationale (4 query links) and test
  commands/counts present and **verified** (§5). AGENTS.md commands (`gh issue view 3741 --comments`,
  `gh pr list --search "3741 in:body"`) are not quoted verbatim; optional.
- Title `[algo, trainer] fix: ...` passes `tests/special_sanity/check_pr_title.py` (re-run). `[cfg]` could be added
  for the yaml comments; `[BREAKING]` is the D4 decision.
- Commits: two, buildable, trailer `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` on both (AGENTS.md form).
- pre-commit: `ruff@0.12.2 check` + `format --check` clean on both changed .py files (re-run via uvx);
  `check_docs_time_info.py` passes (substring only); mypy not re-run here (author: passed). `_generated_*.yaml`
  unchanged (`:102`, `:625` still `null`; comments are not generated).
- D1 interacts with CI's `-x`: fix before opening.
- Prior-art wording: "#2435 / #4721 stale" is accurate (last commits 2025-07-09 / 2025-12-29, last comments by
  CLAassistant the same day); their `updatedAt` 2026-10-24 is bot/base activity. #6823 merged 2026-06-24 and #4360
  2025-12-01 (added the factor **and** the yaml promise) support the regression story as told.
- Open items for the submitter: ci-request Slack message; reviewer names are plain text (no `@`), keep it that way unless a
  notification is intended; `critic.yaml:63` mode-list comment stays stale (lists 3 of 5 modes) -> leave or one-line
  drive-by, either is fine.
- CHECKLIST.md says REVIEW.md "not produced"; this file closes that item.

## 5. Adversarial tests (packages/p0/review-tests/, run against the branch: 21 passed in 28 s)

```bash
cd <repo>/packages/p0/review-tests
PATH=<repo>/.venv-cpu/bin:$PATH PYTHONPATH=<repo>/wt/p0 \
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_OFFLINE=1 python -m pytest -c ../../../pytest-cpu.ini \
  --rootdir . -p no:cacheprovider --asyncio-mode=auto -q test_review_*_on_cpu.py      # 21 passed
```

| file :: test | what it attacks | result |
|---|---|---|
| placement_gaps :: worker_config_built_without_validate_config_keeps_the_micro_batch_horizon | D2: compose `ppo_trainer`, build `ActorConfig` like `engine_workers.py:633`, no `validate_config` | factor None, split/whole loss ratio 8.996x on the **branch** (gap confirmed) |
| placement_gaps :: dataclass_defaults_are_unchanged | direct `ActorConfig` / `CriticConfig` construction | both None (fix is outside the dataclasses) |
| placement_gaps :: hydra_default_is_still_null_in_the_generated_yaml | D3 | composed config shows null until `validate_config` |
| resolution_semantics :: critic_explicit_factor_differs_from_actor_and_both_are_kept | explicit `critic.loss_scale_factor=7`, actor null | actor 16, critic 7 (config and dataclass) |
| resolution_semantics :: rollout_response_length_wins_over_data_max_response_length | `rollout.response_length=32`, `data.max_response_length=16` | factor 32 (the V0 padded width), critic too |
| resolution_semantics :: critic_keeps_the_interpolation_and_use_critic_false_still_inherits | D9; `use_critic=False` | critic stays `oc.select` (resolves to 16), dataclass 16, plain int after `OmegaConf.resolve`; with `use_critic=False` the critic still inherits 16 |
| resolution_semantics :: validate_config_is_idempotent | double call (main_ppo + one_step_off pattern) | 16 |
| resolution_semantics :: other_modes_ignore_the_factor_at_the_loss_level [8] | token-mean / token-sum / seq-mean-token-sum / seq-mean-token-mean, actor and critic, factor None vs forced 16, whole and uneven splits | `torch.equal` on loss and gradient: other modes exactly as before |
| resolution_semantics :: actor_norm_with_critic_token_mean_inherits_an_unused_factor | actor norm, critic explicitly token-mean | critic factor 16 by inheritance, `critic/vf_loss` aggregation SUM (unchanged under V1) |
| resolution_semantics :: metric_aggregation_flips_only_on_the_test_only_path | `batch_num_tokens=None, global_batch_size=None` | MEAN -> SUM only on inputs no engine produces |
| v0_and_megatron :: v0_dense_inputs_give_the_same_loss_with_factor_none_and_response_length [2] | V0 layout (dense response fields padded to 16, `max_response_len` carried, flat model output) through real `ppo_loss` / `value_loss` | bit-equal loss and gradient for factor None vs 16: "V0 identical" holds |
| v0_and_megatron :: v0_width_narrower_than_response_length_would_change | boundary of that claim (dense width 12) | loss scales by 12/16: precondition is "padded to rollout.response_length", which `agent_loop.py:793-807` guarantees |
| v0_and_megatron :: megatron_trainer_config_is_resolved_too | `ppo_megatron_trainer` composed on CPU | McoreActorConfig and critic both 16 (ran, not skipped) |

Regression and fail-on-main runs (reviewer's own, same harness as research/00-cpu-env.md; logs in `review-tests/logs/`):

| run | result |
|---|---|
| new file, branch (`PYTHONPATH=wt/p0`) | 45 passed, 2 xfailed, 66 s |
| new file copied to a temporary directory, main clone via the venv's editable install, `--import-mode=importlib` | **14 failed**, 31 passed, 2 xfailed, 37 s; the 14 are exactly the PR body's list; main clone clean before/after |
| `tests/trainer` (branch) | 483 passed, 1 skipped, 2 xfailed |
| `tests/workers` (4 megatron files ignored, 1 Apple-BLAS case deselected) | 278 passed, 40 skipped, 1 deselected |
| `tests/utils` (2 megatron files + sandbox ignored) | 473 passed, 23 skipped |

## 6. Verified correct

- `agg_loss` reads `loss_scale_factor` only inside the norm branch (`core_algos.py:1189-1193`); other modes bit-identical (§5).
- Flow: `losses.py:80` puts the factor into `global_batch_info`, so the policy loss, the entropy term (`:164-166`) and the
  KL term (`:176-178`) share the constant; critic via `:236`; `distillation/losses.py:304` likewise.
- Engine call shape matches the test: `fsdp/transformer_impl.py:1597-1599` calls `loss_function(model_output=, data=micro_batch, dp_group=)`
  per micro-batch; `batch_num_tokens`/`dp_size` assigned per mini-batch (`:734-739`; Megatron `:802-808`), `global_batch_size`
  from the trainer (`trainer_base.py:1810,1856`); `no_padding_2_padding` (`padding.py:109-141`) slices a flat leaf by
  prompt+response lengths exactly as the test builds it. Megatron shares `ppo_loss` (`engine_workers.py:687`).
- Horizon choice: `rollout.response_length` is the width V0 always had (`agent_loop.py:793-807`, `padding.py:49-51`) and
  `rollout.response_length = ${oc.select:data.max_response_length,512}` (`rollout.yaml:39`) resolves at validate time; an
  explicit rollout override wins (§5). Critic pads the same response fields (`losses.py:199,221`), so the same constant is right.
- Dynamic bsz / uneven micro-batches: a constant factor is split-independent by construction (branch test, 30 cases).
- The yaml promise "If null, uses response_length" dates from #4360 (6e6ae96c, 2025-12-01); #5417 (b8d91ef8) kept the
  `loss_mask.shape[-1]` default when V0 was the default; #6823 (2026-06-24) made V1 the default (`ppo_trainer.yaml:232`).
- Not touched and not conflicting: `core_algos.py`, `losses.py`, `padding.py` (#8176, #6349, #7916, #7830 all OPEN on 2026-10-29).
- Docs/yaml wording is accurate for the covered launchers; `docs/algo/grpo.md:60` "(i.e. data.max_response_length)" is the default binding.

## 7. Fixes applied (author, 2026-10-29)

Two new commits on contrib/p0-loss-horizon, no history rewrite: `35c3bddd` (config.py, yaml, docs) and `6684917e` (test file).

| # | fix | commit / file |
|---|---|---|
| D1 | both xfails `strict=False` with a comment naming #7916 / #7830 and the `pytest -x` reason; test docstring updated | 6684917e test file |
| D2 | boundary stated: yaml comments ("a launcher that bypasses validate_config must set it"), grpo.md, helper docstring, PR-BODY "The fix and its scope" (in-tree launchers listed; fully_async / V0 lineage never had the bug) | 35c3bddd, PR-BODY.md |
| D3 | comments now read "null = resolved to actor_rollout_ref.rollout.response_length by validate_config when the trainer launches (logged at INFO; `--cfg job` still prints null)" | 35c3bddd actor.yaml / critic.yaml / grpo.md |
| D4 | title prefixed `[BREAKING]` (checker re-run: valid); PR-BODY opens with the behaviour-change paragraph, says the old scale was not a constant and how to keep a comparable one (set the factor explicitly) | PR-TITLE.txt, PR-BODY.md |
| D5 | `actor/entropy` division by response_length named in the behaviour-change paragraph | PR-BODY.md |
| D6 | snapshot-free: stub dir with a tiny qwen2 `config.json` + `+actor_rollout_ref.model.load_tokenizer=false` / `+critic.model.load_tokenizer=false` (the dataclass field is not a yaml key); skipif removed; 18.6 s on the branch | 6684917e test file |
| D7 | `validate_config` raises `ValueError("... loss_scale_factor must be > 0 ...")` for an explicit factor <= 0 under the norm mode; two test cases (0, -1), failing on main | 35c3bddd config.py, 6684917e test file |
| D8 | NOTES.md "D8" bullet: first logging / mutation site in validate_config and why acceptable; PR-BODY "Design" says it too | NOTES.md, PR-BODY.md |
| D9 | NOTES.md decision bullet corrected: the critic's oc.select stays when the actor resolves first | NOTES.md |

Re-verification after the fixes: new file branch 47 passed / 2 xfailed (18.6 s); main 16 failed / 31 passed / 2 xfailed
(the 14 of §5 plus the two D7 cases); review-tests 21 passed (32 s); tests/trainer 485 passed / 1 skipped / 2 xfailed,
tests/workers 278 passed / 40 skipped, tests/utils 473 passed / 23 skipped; `pre-commit run --all-files` exit 0, all 14
hooks passed, no file modified (logs/pre-commit-all-files.log). Main clone untouched.

## 8. Re-verification (reviewer, 2026-10-29, HEAD 6684917e, read-only)

**Updated verdict: MERGE-READY** (pending the submitter's own line-by-line review and the ci-request message). D1-D9 are
addressed; nothing in the two fix commits introduces a defect. Residual findings below are nits, none blocking.

What was re-run (logs in `packages/p0/review-tests/logs/*-6684917e.log`; both checkouts `git status --porcelain` empty
before and after; my `tests/utils` run writes `wt/p0/test_data/` into the cwd, removed again):

| run | result |
|---|---|
| review-tests (21) against the fixed worktree | 21 passed |
| new file on the branch | 47 passed, 2 xfailed (non-strict), 13 s |
| new file copied to a temporary directory, main clone via the editable install | **16 failed**, 31 passed, 2 xfailed: 4 invariance (actor/critic x uneven/singles) + 8 horizon + 2 rejects (0, -1) + 2 config resolution; exactly the PR body's list |
| `tests/trainer` / `tests/workers` / `tests/utils` | 485 passed 1 skipped 2 xfailed / 278 passed 40 skipped 1 deselected / 473 passed 23 skipped |
| `check_pr_title.py` on `[BREAKING][algo, trainer] fix: ...` | valid (BREAKING CHANGE) |
| `ruff@0.12.2 check` + `format --check` on the two changed .py files; `check_docs_time_info.py` | clean |
| author's `logs/pre-commit-all-files.log` tail | all hooks Passed (not re-run by me) |

Fix commit review:
- `35c3bddd` `verl/utils/config.py:108-116`: explicit factor `<= 0` under the norm mode raises a clear `ValueError`
  naming the section; a positive explicit value is kept (probe: `4096` -> actor 4096, critic 4096; `critic.loss_scale_factor=-3`
  -> `critic.loss_scale_factor must be > 0 ...`). Other modes untouched (the check sits after the mode test). Docstring,
  `actor.yaml:88-91`, `critic.yaml:66-68`, `docs/algo/grpo.md:60` now state "> 0", "resolved by validate_config when the
  trainer launches (logged at INFO; `--cfg job` still prints null)" and "a launcher that bypasses validate_config must set it":
  accurate (D2, D3).
- `6684917e` test file: the only changes are the stub model dir (`_stub_model_dir`, tiny qwen2 `config.json`), the two
  `+...model.load_tokenizer=false` overrides, `strict=False` on both xfails with the `pytest -x` comment, and
  `test_validate_config_rejects_a_non_positive_factor[0|-1]`. `_make_batch`, `_accumulate`, the three parametrised
  groups and their assertions are byte-identical to c57eecaf, so coverage is equivalent: it still composes
  `ppo_trainer`, runs `validate_config`, builds the dataclasses and drives the real `ppo_loss` / `value_loss` with nested
  inputs across whole / [1,2,5] / [4,4] / singles for all five modes, actor and critic (30 + 8 + 9 + 2 cases). The
  fail-on-main breakdown is unchanged plus the two new reject cases. No `skipif` remains; `~/models` is not read.
- PR body: opens with the behaviour-change paragraph; "the old scale was not a constant, so it cannot be restored
  exactly; set `loss_scale_factor` explicitly (e.g. the typical max response length of your runs)" is accurate: the
  old per-micro-batch normalizer was `max(response lens in the micro-batch)`, so a constant near that width reproduces
  the old scale on average. `actor/entropy` named (D5). The claim that `fully_async_policy` and the recipe runners that
  skip `validate_config` are legacy dense-layout trainers is verified: `RayFlowRLTrainer`, `RayEntropyTrainer`,
  `RayAtroposTrainer` all subclass `verl.trainer.ppo.ray_trainer.RayPPOTrainer` (verl-recipe, read via gh), which
  calls `left_right_2_no_padding` (`ray_trainer.py:1255-1381`) and so carries `max_response_len`.

Residual findings (optional, file:line):
- R1 nit, `verl/utils/config.py:111`: a non-numeric explicit factor (`loss_scale_factor=abc`) now fails at validate time
  with `TypeError: '<=' not supported between instances of 'str' and 'int'`. On main the same value passes
  `validate_config` (`omega_conf_to_dataclass` does not type-check it: probe shows `actor=abc`) and only explodes inside
  `agg_loss` at train time, so this is earlier, not worse; a clearer message would be
  `if not isinstance(factor, (int, float)) or factor <= 0: raise ValueError("... must be a positive number ...")`.
  A float such as `0.5` is accepted on both (pre-existing; `agg_loss` divides fine).
- R2 nit, test file `_stub_model_dir` (`tempfile.mkdtemp`): the directory is never removed (one tiny dir per session).
  `tmp_path_factory` or an `atexit` cleanup would tidy it; CI does not care.
- R3 nit, test file: `+actor_rollout_ref.model.load_tokenizer=false` is unnecessary (`validate_config` instantiates only
  the critic's `HFModelConfig`; `ActorConfig.validate` ignores `model_config`, `workers/config/actor.py:221-241`). Harmless.
- R4 wording, PR-BODY.md first paragraph: "changes by `response_length / (micro-batch max width)`" does not say the
  direction. Each micro-batch's loss and gradient are multiplied by `width / response_length`, i.e. they get **smaller**
  (5-10x on a 2048 budget); the two ways to keep the old effective step are the explicit factor (as written) or a
  correspondingly larger `lr`. Also "#7916" cited as precedent is an open PR, not a merged one; "#6823 (merged)" carries
  the point.
- R5 process: the first fix commit is titled `[algo, trainer, cfg]` while the PR title carries `[algo, trainer]`;
  irrelevant under squash-merge. Adding `cfg` to the PR title is optional and valid.

## 9. Residual nits closed (author, 2026-10-29)

One final commit `0dab0eb5` `[algo, tests] fix: name the error for a non-numeric loss_scale_factor and tidy the loss-horizon test`
(same trailer, no history rewrite):

| # | fix | where |
|---|---|---|
| R1 | `isinstance(factor, bool) or not isinstance(factor, int \| float) or factor <= 0` -> `ValueError("<section>.loss_scale_factor must be a positive number ... got 'sixteen'")`; test `test_validate_config_rejects_an_invalid_factor[0\|-1\|sixteen]` (fails on main) | config.py, test file |
| R2 | stub model dir from a module-scoped `model_dir(tmp_path_factory)` fixture (pytest cleans it up); the cached helpers take `model_dir` | test file |
| R3 | `+actor_rollout_ref.model.load_tokenizer=false` dropped; only `+critic.model.load_tokenizer=false` remains (with a comment); test passes without it | test file |
| R4 | PR-BODY first paragraph: losses and gradients get smaller (divided by the full `response_length` instead of the micro-batch max, never larger), so an unchanged learning rate means a smaller effective step | PR-BODY.md |
| R5 | PR-BODY: #7916 cited as an open PR (not merged), #6823 as merged; the `[algo, trainer, cfg]` commit title vs `[algo, trainer]` PR title is irrelevant under squash-merge (noted) | PR-BODY.md, NOTES.md |

Re-verification: new file branch 48 passed / 2 xfailed (10.9 s); main 17 failed / 31 passed / 2 xfailed (the 16 of §8
plus `[sixteen]`); review-tests 21 passed; tests/trainer 486 passed / 1 skipped / 2 xfailed, tests/workers 278 passed /
40 skipped, tests/utils 473 passed / 23 skipped; pre-commit ruff / ruff-format / mypy / check-license on the two
changed files: Passed; ruff 0.12.2 check + format --check clean. Main clone untouched.

## 10. Review round 2 (the submitter, 2026-10-29)

Source: research/review-bundle/REVIEW.md ("Additional improvements and scope limits", the two P0 rows; "Contribution
positioning") and FIX-CHECKLIST.md "P0 loss horizon". Fix commit: `e0a7eb22` `[algo, cfg, tests] fix: require a finite
positive loss_scale_factor and a positive integer response_length horizon` (HEAD, on top of 0dab0eb5; base main @
2ab35292). One row per finding: finding -> fix -> test proving it -> status.

| # | finding | fix (e0a7eb22) | test | status |
|---|---|---|---|---|
| N1 | `config.py:112` accepts an explicit NaN (`nan <= 0` is False) and +inf | `_is_finite_positive_number`: int or float, not bool, `> 0`, `math.isfinite` for floats (an int never reaches isfinite, so a huge int cannot raise OverflowError); `ValueError("<key>.loss_scale_factor must be a finite positive number ..., got <value!r>")` | `test_validate_config_rejects_an_invalid_explicit_factor[nan\|inf\|-inf\|0\|-1\|sixteen\|True]` (7; every case fails on main, the message is checked for the key and `got <value!r>`) | closed |
| N2 | inferred branch (`:126`) assigns `int(response_length)` unchecked: 0, negative, 2.5 and a bool are accepted | `response_length` must be an `int` (not bool) `> 0`, else `ValueError` naming `actor_rollout_ref.rollout.response_length` and the value; it is assigned as is (no `int()` cast); an unset (`None`) `response_length` keeps the warning-only path | `test_validate_config_rejects_an_invalid_inferred_horizon[0\|-16\|2.5\|True]` (4), `test_validate_config_warns_when_response_length_is_unset` (caplog), `test_validate_config_resolves_an_overridden_response_length` (32 resolved: the override reaches the hook) | closed |
| N3 | focused tests for NaN, infinity, zero and negative horizons | as N1 / N2, plus explicit `4096` and `0.5` kept (`test_validate_config_keeps_an_explicit_factor[4096\|0.5]`); the 30 invariance and 8 horizon tests are unchanged | file on the branch: 59 passed / 2 xfailed (48 / 2 before); on main: 27 failed / 32 passed / 2 xfailed (17 before) | closed |
| N4 | retain the launcher-scope qualification (a launcher bypassing `validate_config` keeps the old null fallback) | kept: `_resolve_loss_scale_factor` docstring, actor.yaml / critic.yaml comments, docs/algo/grpo.md, PR-BODY "Scope and limitations" | n/a (documentation) | kept |
| N5 | retain the changed-normalization explanation for affected Dr. GRPO jobs; `[BREAKING]` | kept: PR-BODY "What changes for users" (losses and gradients get smaller by `(micro-batch max width) / response_length`; set the factor explicitly to keep the scale); title unchanged | n/a | kept |
| N6 | PR body: shorter, AI statement, GPU-suite wording, fresh duplicate check | PR-BODY.md rewritten per REVIEW-ROUND-RULES; read-only gh searches 2026-10-29 13:08Z (PRs and issues for `loss_scale_factor`, `seq-mean-token-sum-norm`, `max_response_len`): new hits #6157 (closed 2026-04-27, unmerged, `agg_loss`-level tests; touched by a comment today) and #4294 (open, legacy `dp_actor` / `dp_critic` metric aggregation) named as related, not overlapping | n/a | closed |

Not changed, with the reason: a float `response_length` with an integral value (2048.0) is rejected like 2.5, because
`rollout.response_length` is an int everywhere else (sampling params, tensor shapes) and Hydra never turns an int
override into a float; explicit float factors (0.5) stay accepted (round 1 R1; the dataclass field is `Optional[int]`,
actor.py:164 / critic.py:94, but hydra-instantiate does not type-check it and 0.5 is consumed by `ppo_loss` /
`value_loss` as a plain divisor, proven at loss level by the reviewer's round-2 test); the unset-`response_length`
path stays a warning (nothing is assigned there; not raised by the submitter's bundle).

Verification after e0a7eb22: new file 59 passed / 2 xfailed (6.8 s); main 27 failed / 32 passed / 2 xfailed; full CPU
regression (six groups, 13:11-13:16Z) 1543 passed, 64 skipped, 2 xfailed, 1 deselected, 0 failed (small 200, utils 473, trainer 497, workers 278, experimental 74, single_controller 21), exit 0; pre-commit on the five changed files: all 14 hooks passed, `_generated_*.yaml` unchanged
(logs/pre-commit-files-e0a7eb22.log); worktree clean; main clone untouched; nothing pushed.

Round-2 re-review (lead, REVIEW-ROUND2-VERDICT.md: MERGE-READY, four LOW items): D1-D4 -> fixed.

| # | finding | fix | test | status |
|---|---|---|---|---|
| D1 | ints skipped the finiteness check: `loss_scale_factor=10**400` passed validation and `ppo_loss` raised OverflowError on the first micro-batch | commit `9c99205c`: int factors bounded to `1..2**63-1` (`_INT64_MAX`; measured: torch raises OverflowError for any Python int >= 2**63 on `tensor / n`, so the float range would have been the wrong bound), floats finite and > 0 as before | `test_validate_config_rejects_an_invalid_explicit_factor[int64-overflow\|beyond-float]`, `test_validate_config_keeps_an_explicit_factor[2**63-1]`; file 62 passed / 2 xfailed, main 29 failed / 33 passed; reviewer's round2: 40 passed, the 2 D1 encodings now fail by design (strict xfail XPASSes; the "passes validation then overflows" test gets the ValueError), reviewer-owned, left to retire | fixed |
| D2 | package notes said the dataclass field is `Optional[float]` | text corrected in REVIEW.md s10 and NOTES.md: `Optional[int]` (actor.py:164, critic.py:94), not type-checked by hydra-instantiate; 0.5 consumed at loss level | n/a | fixed |
| D3 | "(the submitter: keep it)" attributed the unset-horizon warning decision to the submitter | attribution removed; the reason stated (nothing is assigned on that path) | n/a | fixed |
| D4 | PR body claimed the recipe runners that skip `validate_config` "never had the defect" | hedged to the verified in-tree launchers (`main_ppo.py:181` validates before the V1/V0 split; `main_ppo_v0.py:166/179` and `one_step_off_policy/main_ppo.py:47/52` resolve then validate; `fully_async_policy` has its own `_validate_config` and updates through `RayPPOTrainer._update_actor` -> `left_right_2_no_padding`, `ray_trainer.py:1327-1335`); verl-recipe runners "not examined" | n/a | fixed |

After 9c99205c: pre-commit on the two changed files 14 hooks Passed (logs/pre-commit-files-after-d1.log); round-1 review
tests 21 passed; GPU (lead): verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`, so it does not execute this PR's new tests) on 8xH200 at 9c99205c: 503 passed / 0 failed / 8 skipped + torchrun 2/2, identical to
main (bench/results/gpu-tests-p0-r2); full CPU regression (six groups, 13:39-13:44Z) 1546 passed, 64 skipped, 2 xfailed, 1 deselected, 0 failed (small 200, utils 473, trainer 500, workers 278, experimental 74, single_controller 21), exit 0; worktree clean; main clone untouched; nothing pushed.
