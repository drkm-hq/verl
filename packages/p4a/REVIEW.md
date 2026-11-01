# Review: `[1/2] [tool, rollout, cfg] feat: ToolResponse.done, generated turn spans and a dense per-turn reward row`

Branch `contrib/p4a-turn-signals` (3 commits on main @ 2ab35292: 5d4b11f3, 4d21a858, 71bd8b61), worktree `wt/p4a`.
Reviewer: Claude Fable 5.1 (AI), read-only, 2026-10-29. Full history and method: `packages/p4/REVIEW.md` (§1-§6 the
original review of the combined branch, §7 the author's fixes, §8 this re-verification).

## Verdict: MERGE-READY-PENDING-GPU

Stands alone: no `turn_rloo`, no `compute_advantage` / `trainer_base` / `core_algos` change (strict grep; those files
are main's). 193 product / 411 test / 66 docs lines. Title passes `check_pr_title.py`.

## Verified

- Flag off is main: the same scripted 4-turn episode under main's code and under this branch gives bit-identical
  `prompts / responses / response_mask / rm_scores`, `num_turns`, the V0 `_postprocess` batch, `meta_info`,
  `extra_fields` and the non-tensor key set (review-tests R3).
- Span bookkeeping equals the maximal runs of `response_mask == 1` under CT merges that remove 0-2 trailing
  tokens and insert 0-2 boundary tokens, under truncation mid-turn, the context-merge budget hit, tool errors and
  parallel calls with `done` (R2, R5); rewards only ever land on generated tokens; the outcome fallback is the
  last valid token (D3 fixed, `agent_loop.py:101,133-134,1205`).
- Flag on: the three keys are exported (`tool_agent_loop.py:251-265`), the schema columns are added by the worker
  (`agent_loop.py:1239-1241`) so a single-turn chunk concatenates with a tool-agent chunk (R10); `grpo_vectorized`,
  a by-name estimator without `**kwargs`, `rloo`, `reinforce_plus_plus` run on the resulting batch (R9: 8 passed).
- Fail-fast (D2): `verl/utils/config.py:122-139` rejects the flag exactly when
  `RewardLoopManager.reward_loop_worker_handles` would be None (`reward_loop.py:313-320`), checked on the real
  composed `ppo_trainer` config including the legacy `enable_resource_pool: null`; the full `validate_config` reaches
  the hook (R7).
- Warnings instead of silent drops (D6/D7): `agent_loop.py:127-132`, `tool_agent_loop.py:124-129` (R2).
- Regression (my run): trainer 444 passed / 1 skipped, experimental 83, tools 42, utils 473 / 23 skipped; sanity
  hooks, ruff, mypy clean; generated yamls regenerate identically. Review suite: 35 passed, 2 skipped (no estimator
  here), `packages/p4/review-tests/RESULTS-p4a.txt`.

## Residual (non-blocking)

- `agent_loop.py:1239-1241`: a custom worker without `rollout_config` silently omits the schema columns; with the
  flag on, cross-worker `DataProto.concat` then asserts (R10(b)). The shipped worker sets it (`:331`).
- Spans are tracked with the flag off too, so the D7 `RuntimeWarning` can appear in a flag-off run on a pathological
  merge. Harmless.
- `tests.log` quotes the pre-update reviewer suite ("4 failed, 12 passed"); the current suite passes.
- Not run: GPU / end-to-end training with a real Continuous Token author (`tokenization_sanity_check_mode=strict`
  on GLM / Gemma / Kimi) and the reward-loop path on a live job. Hence PENDING-GPU.

## Residuals closed (author, 2026-10-29)

- R-2 (`agent_loop.py` schema columns for a worker without `rollout_config`): commit f4651857 on this branch
  (`contrib/p4a-turn-signals` HEAD). `_postprocess` now calls the module-level
  `_turn_level_rewards_enabled(rollout_config)`, which treats a missing rollout config, a config without a `multi_turn`
  section or one that predates the flag as off (the default batch); the shipped worker's config still switches the
  columns on. CPU test `test_v0_postprocess_without_a_rollout_config_behaves_as_if_the_flag_were_off` (three shapes).
  Cross-worker concat under the flag still needs every worker to carry the config; a worker that cannot see it degrades
  to the default schema instead of a partial one.
- R-3 (D7 warning in a flag-off run): left as is (harmless, documented).
- `tests.log` regenerated on f4651857 (results below in that file); diff.patch and branch.txt refreshed.

## 10. Review round 2 (the submitter)

| Finding | Fix commit | Test proving it | Status |
|---|---|---|---|
| 1. DAPO / GDPO custom reward functions do not receive `generated_turn_spans` / `turn_rewards` / `episode_done` (probe: naive 1.0, dapo 0.0, gdpo 0.0) | 7e4e8e4f (`RewardManagerBase.extra_info_for_reward_fn`, five managers) | `tests/experimental/reward_loop/test_reward_manager_extra_info_on_cpu.py` (6: five managers + the empty case, copy-not-mutation); probe re-run on this tree: naive / dapo / gdpo print the keys, reward 1.0 | fixed |
| 2. Span bookkeeping runs at the merge sites with the flag off | e5111692 (both `_align_generated_turn_spans` calls gated) | `test_flag_off_keeps_the_default_scalar_placement_byte_identical` (unchanged, passes); `test_tool_call_id_on_cpu.py` fake carries the flag | fixed |
| 4. Terminal observation absent when the context merge hits the response budget | e5111692 (docs paragraph + site comment; behaviour unchanged) | `test_step_reward_and_done_are_kept_when_the_final_context_merge_hits_the_response_budget`; real-tokenizer variant in `test_turn_spans_real_tokenizer_on_cpu.py` | documented + tested |
| 8. Real tokenizer boundary / junction test | a600e810 | `test_turn_spans_real_tokenizer_on_cpu.py` (2) on `~/models/Qwen/Qwen2.5-0.5B` through the loop's real CT merges | added |
| 7. 503-pass GPU suite does not exercise the new tests; enabled run required | PR-BODY.md / CHECKLIST.md wording; `packages/p4/bench/ENABLED-RUN.md` + `gsm8k_tool_reward.py` | the enabled run's metrics (pending, bench agent) | pending |
| 6. PR body: co-authorship, AI statement, #8305 coordination, fresh duplicate check | PR-BODY.md (2026-10-29 `gh` searches; #8305 paragraph; AI statement per the rules) | n/a | done |
| round-2 re-review LOW 1: `extra_info_for_reward_fn` requires a mapping (a str `extra_info` column raised under every manager; main's DAPO passed it through) | d8c4b925 (non-mapping values handed over unchanged, one warning per type naming it) | `test_a_non_mapping_extra_info_is_passed_through_unchanged_with_one_warning[DAPO,Naive]`; reviewer r2a 46 passed, its 3 `test_a_string_extra_info_now_raises_under_every_manager` cases pin the pre-fix behaviour and fail by design | fixed |
| round-2 re-review LOW 2: shallow copy, the span / reward lists shared with the loop output | d8c4b925 (list-valued tool fields copied) | `test_list_valued_tool_fields_are_copied_so_a_reward_function_cannot_edit_the_row`; reviewer r2a `test_the_copy_is_shallow_so_nested_lists_stay_shared_with_the_loop_output` pins the pre-fix behaviour, fails by design | fixed |

## 11. Round 3 (the AI follow-up review): R2-04 / experiment -> fix -> test -> status

| Finding | Fix | Test proving it | Status |
|---|---|---|---|
| R2-04: with `turn_level_rewards` off, `extra_info_for_reward_fn` merged the loop metadata and overwrote `num_turns` / `rollout_reward_scores` under dapo / gdpo / rate_limited, where main handed the dataset's `extra_info` over (probe: `num_turns` 7 on main, 3 on p4b; dataset `tool_rewards` replaced) | 0d4cd8ca: `merge_tool_fields` / `rollout_fields` on the helper; naive / remote merge always (their main contract); dapo / gdpo merge only with the flag on, otherwise the dataset's dict untouched; rate_limited merges `tool_extra_fields` as before and adds the rollout fields only with the flag on; the flag is read at init (`turn_level_rewards_enabled`, absent = off); precedence documented (loop fields override dataset keys of the same name; `num_turns` / `rollout_reward_scores` override both) | `test_each_manager_keeps_its_previous_callback_contract_with_the_flag_off` (5 managers x 2 flag states, 7 vs 3), `test_turn_level_rewards_enabled_reads_the_trainer_config_and_defaults_to_false`, the five-manager enabled test kept; the submitter's probe re-run: 7.0 on main and on this branch (tests.log) | fixed |
| PR-body claim "flag off, tensors and metadata are bitwise what main produces" too broad | PR-BODY.md: tensors bitwise; the callback contract of every manager unchanged with the flag off | the probe test above | fixed |
| Experiment: the per-submission reward was farmed (923 / 1085 of 1319 validation episodes repeated an identical answer; 901 / 1049 earned >= 2 step points; controls 2-3 / 0); "same curve within noise" too strong; the comparison could not isolate the estimator | PR-BODY.md and packages/p4/bench/RESULTS.md disclose it ("no demonstrated accuracy gain", `data.seed` the only difference between seeds); `packages/p4/bench/gsm8k_tool_reward.py` v2 (last-submitted-answer outcome, identical for all arms, `done` scored the same way, repeated-submission diagnostics); ENABLED-RUN.md "v2 design" (arms A / B / C, overrides, stage metrics, reporting rule) | the v2 arms (bench agent) | disclosed; v2 ran on e1052882: no demonstrated accuracy gain for either estimator, farming <= 1.4 %, `done` exercised on GPU |

**Round-3 status: MERGE-READY** (R2-04 fixed and tested; the experiment disclosed and redone: v2 arms A / B / C on e1052882, pooled step-30 accuracy 0.832 / 0.822 / 0.829 within the seed spread, no algorithm-performance claim; `packages/p4/bench/RESULTS.md` "v2").
