# Review: `[2/2] [algo, trainer] feat: turn_rloo advantage estimator with gated turn-span plumbing`

Branch `contrib/p4b-turn-rloo` (2 commits on `contrib/p4a-turn-signals` @ 71bd8b61: 13c84ffe, d8c16498), worktree
`wt/p4b`; its tree equals the combined `contrib/p4-turn-rewards` (hash `31b7fb6d…`, `git diff` empty). Reviewer:
Claude Fable 5.1 (AI), read-only, 2026-10-29. Full history: `packages/p4/REVIEW.md` (§1-§6 original review, §7 the
author's fixes, §8 re-verification).

## Verdict: MERGE-READY-PENDING-GPU (one documentation sentence recommended)

125 product / 460 test / 26 docs lines over p4a. Title passes `check_pr_title.py`.

## Verified

- Estimator math: matches an independent scalar RFC #7244 reference on 30 random ragged cases (5 seeds x gamma
  {1, .9, .5} x std {off, on}; group sizes 1-4, 1-4 turns, rewards on generated, observation and pad tokens) and on
  16 more with injected past-end spans (D5 fixed, `core_algos.py:692-693`); no spans == `rloo` exactly, singleton
  groups included (R4, R8). The two-turn MDP bias test reproduces in the trainer group.
- D1 (the high finding of the first review): `ray_trainer.py:262-267` forwards `turn_spans` only to estimators whose
  signature declares it. `grpo_vectorized`, a by-name estimator without `**kwargs`, `rloo` and `reinforce_plus_plus`
  run unchanged next to the column, flag off and on, through the real `ToolAgentLoop` -> `_postprocess` ->
  `compute_advantage` chain (R1, R9); `turn_rloo` sees spans only under `rollout.multi_turn.turn_level_rewards`
  and equals RLOO otherwise (R9).
- V1 lift `trainer_base.py:1757-1761`: tolerant of LinkedList / NonTensorStack / non-dict rows (D4), padding rows
  and mixed agent loops (R6).
- D8: `test_core_algos_on_cpu.py:40-51` restores the registry; the trainer group passes in one session (456) with
  the import-time workaround gone.
- Regression (my run): trainer 456 passed / 1 skipped, experimental 84, tools 42, utils 473 / 23 skipped; sanity
  hooks, ruff, mypy (`core_algos.py`, `utils/config.py`) clean; generated yamls regenerate identically. Review
  suite: 96 passed (R1-R10), `packages/p4/review-tests/RESULTS-p4b.txt`.

## Residual (non-blocking)

- Docs: `docs/advance/agent_loop.rst:273-283` and the estimator docstring (`core_algos.py:650-660`) should say that
  without `rollout.multi_turn.turn_level_rewards` no spans reach the trainer, so `turn_rloo` is plain RLOO on
  multi-turn data too (the yaml example sets both, the prose does not say it). Optionally a one-time
  `logger.warning` in `compute_advantage` when `adv_estimator == "turn_rloo"` and the column is absent.
- The signature gate follows `__wrapped__`; a third-party wrapper without `functools.wraps` hides `turn_spans` and
  degrades silently to RLOO.
- GPU cost of the `bs x turns` slice assignments (`core_algos.py:702-707`) is unmeasured; no training curve. Hence
  PENDING-GPU.
- `tests.log` quotes the pre-update reviewer suite ("4 failed, 53 passed"); the current suite passes 96/96.

## Residuals closed (author, 2026-10-29)

- R-1 (docs): commit 97a43c92 on this branch (`contrib/p4b-turn-rloo` HEAD, rebased onto p4a @ f4651857):
  `docs/advance/agent_loop.rst` and the `compute_turn_rloo_advantage` docstring now say that without
  `rollout.multi_turn.turn_level_rewards` no spans reach the trainer, so `turn_rloo` is plain RLOO on multi-turn data
  too, and that both keys must be set. The optional one-time `logger.warning` in `compute_advantage` was not added
  (kept the trainer hunk at its reviewed size).
- The `__wrapped__` note and the GPU cost of the index build remain as documented follow-ups.
- `tests.log` regenerated on 97a43c92 (results below in that file); diff.patch and branch.txt refreshed. The combined
  `contrib/p4-turn-rewards` received both residual commits by cherry-pick (c1733496, 912f79fb) and its tree equals this
  branch's.

## 10. Review round 2 (the submitter)

| Finding | Fix commit | Test proving it | Status |
|---|---|---|---|
| 2. Disabled-path cost: `extra_fields` fetched and unwrapped for every estimator | 28ee7c4a (`adv_estimator_takes_turn_spans`; V1 selects / lifts only for estimators that take spans) | `test_v1_compute_advantage_does_not_fetch_extra_fields_for_estimators_without_turn_spans[rloo,reinforce_plus_plus,grpo]`, `test_adv_estimator_takes_turn_spans_reads_the_signature` | fixed |
| 3. External spans not validated | 28ee7c4a (`_validated_turn_spans`, `ValueError` naming the row; row count checked) | `test_malformed_spans_raise_naming_the_row` (overlap, reversed order, empty, start after end, negative, past the row end, starting past the row end, not a pair; row count) | fixed (replaces the round-1 D5 tolerance; reviewer tests R4(d) / R8 superseded) |
| 5. Unbiasedness conditions; `gamma<1` objective; "exactly RLOO" | 64f8f604 (docstring + docs; wording) | `test_turn_rloo_gives_rloo_policy_advantages_but_keeps_raw_returns`, `test_norm_adv_by_std_is_a_variance_heuristic_not_a_rescaled_gradient`, `test_gamma_below_one_is_the_discounted_return_to_go_heuristic`; the exact-gradient test stays the gamma=1 unnormalised proof | documented + tested |
| 7. 503-pass GPU suite does not exercise the new tests; enabled run required | PR-BODY.md / CHECKLIST.md wording; `packages/p4/bench/ENABLED-RUN.md` (+ the `adv_estimator/rows_with_turn_spans` metric as the lift's proof) | the enabled run's metrics (pending, bench agent) | pending |
| 6. PR body: co-authorship (offered in the text only), AI statement, #8305 coordination, fresh duplicate check | PR-BODY.md | n/a | done |
| round-2 re-review LOW 3: float / bool offsets silently truncated by `int()` | 988a10bc (`_span_offset`: `operator.index`; Python / numpy / torch bools rejected by name) | `test_offsets_must_be_integral_ints_rejected_by_name_and_a_non_sequence_column_is_named` + 4 malformed params; reviewer r2d `test_float_and_bool_spans_are_silently_truncated_to_ints` pins the pre-fix behaviour, fails by design | fixed |
| round-2 re-review LOW 4: a scalar / non-sequence `turn_spans` raised `TypeError` at `len()` | 988a10bc (`ValueError` naming the expected shape and the type given) | same test (`turn_spans=5`); reviewer r2d `test_row_count_mismatch_and_a_non_sized_column` pins the `TypeError`, fails by design | fixed |
| round-2 re-review LOW 5: a `functools.wraps` wrapper with an incompatible own signature passed the gate and crashed at call time | 988a10bc (outermost own signature; `**kwargs` + `__wrapped__` chain declaring `turn_spans` opts in; `**kwargs` alone does not) | `test_wrappers_receive_the_spans_only_when_their_own_signature_can_take_them`; reviewer r2e `test_v0_wraps_wrapper_whose_own_signature_cannot_take_the_kwarg_crashes` pins the crash, fails by design; `..._degrades_to_rloo_silently` still passes; docs paragraph "Custom estimators and the spans" | fixed |
| round-2 re-review LOW (benign): an unknown estimator fails in the gate before the queue fetch | none (same `ValueError`, earlier) | reviewer r2e `test_v1_unknown_estimator_fails_in_the_gate_before_the_queue_fetch` passes | accepted |

## 11. Round 3 (the AI follow-up review): R2-04 / experiment -> fix -> test -> status

| Finding | Fix | Test proving it | Status |
|---|---|---|---|
| R2-04 (the helper lives in [1/2]): flag-off callback contract changed for dapo / gdpo / rate_limited | p4a 0d4cd8ca; this branch rebased onto it (e1052882), no code change here | p4a's probe test (5 managers x 2 flag states) runs on this tree: the P4 files 106 passed | fixed upstream, carried |
| Experiment: per-submission reward farmed; "same curve within noise" too strong; estimator not isolated | PR-BODY.md disclosure (no demonstrated accuracy gain; repeated-submission counts; `data.seed` the only difference); packages/p4/bench/RESULTS.md; `gsm8k_tool_reward.py` v2 + ENABLED-RUN.md "v2 design" (B dense + `reinforce_plus_plus` vs C dense + `turn_rloo` isolates the estimator) | the v2 arms (bench agent) | disclosed; v2 ran on e1052882: no demonstrated accuracy gain for either estimator, farming <= 1.4 %, `done` exercised on GPU |

**Round-3 status: MERGE-READY** (R2-04 fixed and tested; the experiment disclosed and redone: v2 arms A / B / C on e1052882, pooled step-30 accuracy 0.832 / 0.822 / 0.829 within the seed spread, no algorithm-performance claim; `packages/p4/bench/RESULTS.md` "v2").
