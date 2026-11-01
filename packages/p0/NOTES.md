# P0 author notes (contrib/p0-loss-horizon)

Worktree: <repo>/wt/p0, branch contrib/p0-loss-horizon cut from main @ 2ab35292.
Import check: PYTHONPATH=wt/p0 -> verl.__file__ = wt/p0/verl/__init__.py (no .venv-p0 needed); without PYTHONPATH the
.venv-cpu editable install resolves to the main clone (used for the fail-on-main run).

## Decisions (file:line in the worktree unless noted)

- Fix placement: verl/utils/config.py::validate_config. It runs at main_ppo.py:181 before the use_v1 branch, at
  main_ppo_v0.py:179 and at experimental/one_step_off_policy/main_ppo.py:52, and it is the only place that sees both
  the actor section and actor_rollout_ref.rollout.response_length. Not in padding.py (vet K12 s2.5: injecting
  max_response_len crashes losses.py:103/:222 with a shape mismatch) and not in agg_loss (conflicts with #8176/#6349).
- Critic resolved independently of the actor: critic.yaml:64/:68 inherit through oc.select, but a user can set
  critic.loss_agg_mode directly. When the actor resolves first, the critic's oc.select interpolation STAYS (it now
  resolves to the actor's int; the typed dataclass merge and main_ppo.py:149 OmegaConf.resolve materialize it); the
  helper assigns the critic key only when the critic reads None on its own (critic-only norm mode). Guarded by
  use_critic and config.get("critic"). (D9: the earlier "plain int in both cases" wording was wrong.)
- D8: validate_config now logs (module logger, first logging use in the file; siblings print) and mutates the config
  (first assignment in that function). Acceptable because it is the one hook all in-tree launchers run before the
  worker dataclasses are built (main_ppo.py:181 V1/V0, one_step_off_policy/main_ppo.py:52, 13 verl-recipe runners),
  the mutation is idempotent and limited to a null key, and precedent for config fix-ups exists at
  trainer_base.py:739-740 / :781-786. The yaml default stays null (`--cfg job` prints null; the INFO line shows it).
  Fallbacks if a maintainer objects: a yaml-level `${oc.select:actor_rollout_ref.rollout.response_length,null}`
  default (REVIEW.md s3 option B) or a hard error in ActorConfig.__post_init__ (option D).
- Legacy per-worker actors (dp_actor.py / megatron_actor.py) no longer exist in this tree; every engine (FSDP,
  Megatron) shares verl/workers/utils/losses.py::ppo_loss (engine_workers.py:687) and value_loss, so one resolution
  covers them all. Trainer-side consumers trainer_base.py:1682 / ray_trainer.py:1594 / separation/ray_trainer.py:529
  read actor_config.loss_scale_factor too and now see the same constant.
- V0 (use_v1=false) is identical in output: its padded width is rollout.response_length (agent loop pads to it), so
  loss_mask.shape[-1] == the resolved factor.

## Implementation (worktree paths)

- verl/utils/config.py: `import logging` + module `logger` (config.py had none; `logger.info` is visible under
  hydra.main's default INFO job logging), new `_resolve_loss_scale_factor(config, use_critic)` placed with the other
  `_validate_*` helpers before `_validate_score_centering_config`, called from `validate_config` right before the
  actor dataclass is built (so the dataclass validated there already carries the int). Reads
  `actor_rollout_ref.rollout.get("response_length")`; if that is None (exotic configs) it warns and leaves the
  behaviour as today instead of crashing on int(None). Explicit values untouched; other modes untouched.
- Yaml comments only (actor.yaml:88-90, critic.yaml:66-68); the `_generated_*.yaml` files carry no comments and
  no value changed, so the autogen hook has nothing to regenerate (verified by running it, see tests.log).
- docs/algo/grpo.md:60 reworded; "Last updated" bumped to 10/28/2026 (check-docs-time-info only needs the substring).
- Not touched: core_algos.py agg_loss (docstring "If None, uses loss_mask.shape[-1]" stays true for the function),
  padding.py, losses.py, critic.yaml:63 mode list comment (omits token-sum / -norm; optional drive-by, left out).

## Test design (tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py, 243 lines)

- Configs come from the real pipeline: hydra compose of `ppo_trainer` -> `validate_config` -> `omega_conf_to_dataclass`
  (exactly engine_workers.py:687 / trainer_base.py:324). That is why the file fails on main for the right reason
  (factor stays None) and passes on the branch, with no private helper imported.
- `validate_config` instantiates HFModelConfig, which loads config.json + tokenizer from `model.path` (`~` is not
  expanded by HFModelConfig, so the absolute path is passed); the file uses the CI snapshot
  `~/models/Qwen/Qwen2.5-0.5B` like tests/workers/config/test_model_config_on_cpu.py and skips without it. Both
  `actor_rollout_ref.model.path` and `critic.model.path` must be overridden (the critic defaults to its own path).
- Required overrides for a composable default config: the four `*_micro_batch_size_per_gpu=1` keys
  (ActorConfig.__post_init__ asserts one is set) and `data.max_response_length=16`.
- Inputs are the vet's (B=8, lens [12,1,3,7,12,2,9,5], tool span masked in row 0, response_length 16), splits
  uneven [1,2,5] (widths 12/3/12), even [4,4] (12/12), singles. Loss and gradient on the flat model-output leaf.
- Four test groups: invariance (2 kinds x 5 modes x 3 splits = 30), horizon == response_length (2 x 4 = 8, fails
  on main even for the whole batch because the width there is 12, not 16), config resolution (7), strict xfails for
  geo_mean (#7913/#7916) and kl_cov (#7829/#7830). clip_cov left out: its non-invariance comes from randperm, so a
  strict xfail could XPASS by chance (09 audit C4, no issue filed); noted for the PR body.

## Result (2026-10-28)

- Commits on contrib/p0-loss-horizon (base 2ab35292): 73bab827 fix (verl/utils/config.py +39, actor.yaml, critic.yaml,
  docs/algo/grpo.md), c57eecaf tests (tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py +243).
  Author the submitter Alonso, trailer `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` on both.
- Evidence: main 14 failed / 31 passed / 2 xfailed; branch 45 passed / 2 xfailed (tests.log s1-s2, logs/).
  Regression trainer 483 / workers 278 / utils 473 passed, 0 failed (tests.log s3). Lint clean (tests.log s5).
- Main clone: untouched, on main, `git status --porcelain` empty throughout. Nothing pushed, no remote, no gh writes.

## Deviations from the brief

- The resolution is a private helper `_resolve_loss_scale_factor` called from `validate_config` (same effect as inline;
  matches the file's `_validate_*` helper style). Tests call the public `validate_config`, so they fail on main for the
  right reason (factor None), not with an ImportError.
- Both test groups live in one file under tests/trainer/ppo/ (the brief allowed tests/workers/ or tests/trainer/ppo/),
  next to test_value_loss_normalization_on_cpu.py and test_loss_aggregation_on_cpu.py.
- Strict xfails only for geo_mean and kl_cov; clip_cov omitted (random selection -> a strict xfail could XPASS).
- The dedicated branch log was re-run after the ruff UP033 fix (`functools.cache`) so it matches the committed file.

## Reviewer questions for the submitter

1. Placement: validate_config (chosen; covers main_ppo V1/V0 and one_step_off_policy; `fully_async_policy` has its
   own `_validate_config` and recipes that bypass verl's validate_config keep the old behaviour). A maintainer may
   prefer raising instead of resolving, or putting it in the trainer (trainer_base.py:739 style). Ready to move.
2. Strict xfails for geo_mean / kl_cov will go red for #7916 / #7830 when those land unless they drop the marks;
   keep (the brief), soften to strict=False, or drop?
3. Title modules: `[algo, trainer]` (brief) or `[algo, trainer, cfg]` (vet); both pass the checker.
4. critic.yaml:63 comment lists only three modes (omits token-sum and seq-mean-token-sum-norm): fix as a drive-by?
5. The test needs the `~/models/Qwen/Qwen2.5-0.5B` snapshot (as 24 existing CPU tests do) and skips without it.

## Review fixes (2026-10-29)

REVIEW.md verdict NEEDS-FIXES (small). Applied D1-D9 as two new commits: 35c3bddd (config.py `<= 0` error + yaml/doc
wording D2/D3/D7) and 6684917e (test: non-strict xfails D1, stub model dir D6, factor 0/-1 cases D7). Mapping in
REVIEW.md §7. Title now `[BREAKING][algo, trainer] fix: ...` (D4). D6 detail: `load_tokenizer` is a dataclass-only
field (model.py:99, not in hf_model.yaml) so the override needs the `+` prefix; with it HFModelConfig only reads
config.json (generation_config falls back to None). Re-verified: branch 47/2xfail, main 16 failed, review-tests 21,
regression 485/278/473, `pre-commit run --all-files` clean.

## Residual nits (2026-10-29, REVIEW.md §8 MERGE-READY)

Closed R1-R5 in one commit `0dab0eb5` (REVIEW.md §9): typed `ValueError` for a non-numeric / non-positive explicit factor
(bool excluded, floats such as 0.5 still accepted as before); `model_dir` fixture via `tmp_path_factory` replaces the
never-removed mkdtemp; actor-side `load_tokenizer` override dropped (only the critic's HFModelConfig is instantiated
by validate_config); PR-BODY says gradients get smaller under the fix (smaller effective step at an unchanged lr) and
that #7916 is open. The first fix commit's `[algo, trainer, cfg]` title vs the PR title's `[algo, trainer]` is
irrelevant under squash-merge; adding `cfg` to the PR title stays optional (checker accepts both).

## Review round 2 (the submitter's independent review, 2026-10-29)

What the submitter found (research/review-bundle/REVIEW.md, FIX-CHECKLIST.md "P0 loss horizon"): the explicit-factor check
`isinstance(factor, int | float) and factor <= 0` let NaN and +inf through (`nan <= 0` and `inf <= 0` are both False);
the inferred branch assigned `int(response_length)` without validating it (0, negative, 2.5 and a bool accepted); the
launcher-scope qualification and the Dr. GRPO breaking-behaviour explanation must stay; the PR body was too long, the
AI statement and the GPU-suite wording had to change, the duplicate check had to be refreshed.

What changed: one commit `e0a7eb22` on top of 0dab0eb5 (same trailer, no history rewrite):
- verl/utils/config.py: `_is_finite_positive_number` (int or float, not bool, > 0, `math.isfinite` for floats; an int
  never reaches isfinite); the explicit error reads "must be a finite positive number ..., got <value!r>"; the
  inferred branch requires an `int` (not bool) > 0 and raises naming `actor_rollout_ref.rollout.response_length` and
  the value, else assigns it directly (no `int()` cast); the `None` warning path is untouched.
- actor.yaml / critic.yaml comments and docs/algo/grpo.md: "must be a finite number > 0", response_length "must then
  be a positive integer"; Last updated 10/29/2026. `_generated_*.yaml` unchanged (autogen hook passed).
- test file: `_validated_config(..., response_length=None)` formats a direct `actor_rollout_ref.rollout.response_length`
  override ("null" unsets it); explicit reject cases nan / inf / -inf / 0 / -1 / "sixteen" / True (message checked
  for the key and `got <value!r>`), inferred reject 0 / -16 / 2.5 / True, explicit 4096 / 0.5 kept, an overridden
  response_length (32) resolved, an unset response_length warns (caplog). Invariance and horizon tests untouched.
  File: 59 passed / 2 xfailed (48 / 2 before); on main 27 failed / 32 passed / 2 xfailed (17 before).
- PR-BODY.md rewritten shorter per REVIEW-ROUND-RULES (AI statement, GPU-suite wording, duplicate check 13:08Z).

Decided not to change, and why:
- A float response_length with an integral value (2048.0) is rejected like 2.5: `rollout.response_length` is an int
  everywhere else (sampling params, tensor shapes), Hydra never turns an int override into a float, and accepting it
  would mean a silent cast. One line to relax if a maintainer asks.
- The explicit factor still accepts floats (0.5) as before (round 1 R1): the dataclass field is typed Optional[int]
  (verl/workers/config/actor.py:164, critic.py:94) but hydra-instantiate does not type-check it, and 0.5 is consumed
  by ppo_loss / value_loss as a plain divisor (the reviewer's round-2 loss-level test proves it on an uneven split).
- The warning-only path for an unset response_length stays a warning: nothing is assigned on that path, so there is
  nothing to validate; the submitter's bundle does not mention it. It now has a test.
- The fix stays in validate_config (launcher boundary documented, not widened); `[BREAKING]` stays.
- Duplicate check: #6157 (closed 2026-04-27, unmerged; agg_loss-level tests and docstring for this mode; a comment
  today bumped it) and #4294 (open; legacy dp_actor / dp_critic metric aggregation) are related, not overlapping;
  both named in the PR body. No open PR touches validate_config for this key.
- The shared scripts/run-cpu-regression.sh runs the clone, not a worktree, so the regression was run through
  scripts/run-cpu-tests-wt.sh with VERL_ROOT=wt/p0, the script's own groups / ignores / deselect / 300 s caps and
  p0-prefixed labels (driver script; logs copied to logs/p0-reg-*.log). The shared script was not modified.

Verification: full CPU regression (six groups) 1543 passed, 64 skipped, 2 xfailed, 1 deselected, 0 failed (small 200, utils 473, trainer 497, workers 278, experimental 74, single_controller 21), exit 0; pre-commit on the five changed files: 14 hooks passed
(logs/pre-commit-files-e0a7eb22.log); worktree clean; main clone untouched; nothing pushed, no gh writes.

## Review round 2 re-review (lead verdict, 2026-10-29): D1-D4

Verdict MERGE-READY with four LOW items (REVIEW-ROUND2-VERDICT.md, review-tests/round2/, both reviewer-owned).
- D1 (code) -> commit 9c99205c: `_is_finite_positive_number` skipped the finiteness check for ints, so
  loss_scale_factor=10**400 passed validation and the first micro-batch raised OverflowError in agg_loss. Measured
  before choosing the bound: torch raises OverflowError ("int too big to convert") for ANY Python int >= 2**63 on
  `tensor / n` (it wraps the scalar as int64; 2**63-1 works), so the reviewer's suggested float-range bound would have
  left 2**63..1e308 passing validation and failing in the loss. An int factor must now be in 1..2**63-1
  (`_INT64_MAX`); floats must be finite and > 0 as before. Branch tests: rejects_an_invalid_explicit_factor
  [int64-overflow|beyond-float] (2**63, 10**400), keeps_an_explicit_factor[2**63-1]. File 62 passed / 2 xfailed; on
  main 29 failed / 33 passed / 2 xfailed. The reviewer's two D1 encodings in review-tests/round2 (the strict xfail and
  the "passes validation then overflows" test) now fail by design (40 passed, 2 failed); left untouched for the
  lead to retire.
- D2 (notes): "Optional[float]" corrected to Optional[int] (verl/workers/config/actor.py:164, critic.py:94;
  hydra-instantiate does not type-check it; 0.5 is consumed at loss level) in NOTES.md and REVIEW.md s10.
- D3 (notes): "(the submitter: keep it)" removed; the unset-response_length warning stays because nothing is assigned there.
- D4 (PR body): the recipe-runner claim replaced by what was verified in tree (main_ppo.py:181 validates before the
  V1/V0 split; main_ppo_v0.py:166/179 and experimental/one_step_off_policy/main_ppo.py:47/52 resolve then validate;
  experimental/fully_async_policy has its own _validate_config and updates through RayPPOTrainer._update_actor ->
  left_right_2_no_padding, ray_trainer.py:1327-1335); the verl-recipe runners are stated as not examined.
Verification: pre-commit on the two changed files 14 hooks Passed (logs/pre-commit-files-after-d1.log); round-1 review
tests 21 passed; GPU (lead, after 9c99205c): verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`, so it does not execute this PR's new tests) on 8xH200 at 9c99205c: 503 passed / 0 failed / 8 skipped + torchrun
2/2, identical to main (bench/results/gpu-tests-p0-r2); full CPU regression (six groups, 13:39-13:44Z) 1546 passed, 64 skipped, 2 xfailed, 1 deselected, 0 failed (small 200, utils 473, trainer 500, workers 278, experimental 74, single_controller 21), exit 0; worktree clean; clone untouched; nothing pushed, no gh writes.
