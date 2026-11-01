# P4a build notes [1/2]: `[tool, rollout, cfg] feat: ToolResponse.done, generated turn spans and a dense per-turn reward row`

Branch `contrib/p4a-turn-signals`, worktree `<repo>/wt/p4a`, base main @ 2ab35292.
Split out of the combined branch `contrib/p4-turn-rewards` after the adversarial review (packages/p4/REVIEW.md); the
design log, deviations and the review-fix mapping live in packages/p4/NOTES.md and packages/p4/REVIEW.md "Fixes applied".

## What is in this half

- `verl/tools/schemas.py`: `ToolResponse.done`.
- `verl/experimental/agent_loop/tool_agent_loop.py`: `AgentData.generated_turn_spans / turn_rewards / episode_done`,
  `_align_generated_turn_spans` (#8305's rule, warns when a merge removes a rewarded turn), `done` -> TERMINATED, the
  three metadata keys exported only under `rollout.multi_turn.turn_level_rewards`.
- `verl/experimental/agent_loop/agent_loop.py`: `write_turn_level_rewards` (last-generated-token rule, `fallback`
  index, warns on an unplaceable reward), `AgentLoopOutput.turn_rewards`, dense rows in `as_dict` / `_postprocess`,
  schema columns only in turn-level mode.
- `verl/utils/config.py`: `_validate_turn_level_rewards_config` (colocated reward model rejected).
- Config: `MultiTurnConfig.turn_level_rewards`, rollout.yaml (+ the reward-loop and `max_assistant_turns` notes), four
  generated yamls (identical to the combined branch's; `scripts/generate_trainer_config.sh` output).
- Tests: `test_turn_level_rewards_on_cpu.py` (without the `turn_rloo` group test, which moves to [2/2]),
  `test_turn_level_rewards_config_on_cpu.py`, `test_agent_loop_tq_on_cpu.py` (+1), `test_tool_call_id_on_cpu.py`.
- Docs: `docs/advance/agent_loop.rst` section without the estimator paragraph; the yaml example uses
  `reinforce_plus_plus`.

## How the branch was built

Files checked out from the fixed combined branch into a fresh worktree on main, then three edits so nothing refers to
the estimator: the group test and its import removed from `test_turn_level_rewards_on_cpu.py`, the estimator paragraph /
title / yaml line removed from `agent_loop.rst`, `turn_rloo` dropped from the rollout.yaml comment ([2/2] adds it back).
Four commits: 5d4b11f3 (writer + config + validation; buildable alone), 4d21a858 (done + spans + loop + its tests),
71bd8b61 (docs), f4651857 (re-verification residual R-2: `_turn_level_rewards_enabled(rollout_config)`, a module-level
helper so fake workers keep working, treats a missing rollout config / `multi_turn` section / flag as off; CPU test with
the three shapes). The branch was rebuilt once after ruff dropped a now-unused numpy import from the loop test; the loop
test file was re-linted and re-run (9 passed) on the final tree. `grep turn_rloo` over verl/ tests/ docs/ on this branch: no hits.

## Verification on this branch

See tests.log (new tests, the reviewer's loop-level tests R2/R3/R5 with the R3 BRANCH path pointed at this worktree,
and the trainer / experimental / tools / utils groups). Reviewer tests R1/R4/R6 need the estimator and only run on [2/2].
Expected-by-wording failures: R2(b)/(c) assert that a dropped turn reward survives (this branch warns and drops, the
reviewer's accepted fix); R3's two metadata assertions encode the pre-fix always-export behaviour (flag-off metadata is
now identical to main, which is what those tests' tensor half and the branch's own test verify).

## Review round 2 (the submitter, 2026-10-29)

What the review found for [1/2] (research/review-bundle/REVIEW.md, FIX-CHECKLIST.md): (1) dapo / gdpo did not merge
`tool_extra_fields` into `extra_info`, so a custom reward function reading `episode_done` scored 1.0 under naive and
0.0 under dapo / gdpo (evidence/reproductions/p4_reward_manager.py, reproduced here); (2) span bookkeeping ran at
both merge sites with the flag off; (4) the response-budget boundary (`tool_agent_loop.py`) terminates before
applying the context merge after the step reward and `done` were recorded, so the terminal observation is absent
from the decoded response; (8) no real tokenizer boundary test; (7) the 503-pass GPU suite excludes the new tests;
(6) PR-body wording (co-authorship, AI statement, #8305 coordination, fresh duplicate check).

What changed (three commits on f4651857, HEAD a600e810):
- 7e4e8e4f `[reward] fix: deliver the agent loop's tool metadata to every reward manager's custom reward function`:
  `RewardManagerBase.extra_info_for_reward_fn(data_item)` (static) returns a copy of the dataset's `extra_info`
  merged with `tool_extra_fields` plus `num_turns` / `rollout_reward_scores`; naive / remote / limited lost their
  inline merge, dapo / gdpo gained the keys (gdpo still adds `experiment_name`, dapo keeps the overlong handling).
  `tests/experimental/reward_loop/test_reward_manager_extra_info_on_cpu.py`: the five managers parametrized (remote
  built without Ray, an in-process stand-in for its worker pool) + the empty case; copy-not-mutation asserted. The
  reviewer's probe, re-run from a scratch copy against this tree: naive / dapo / gdpo all print the keys, reward 1.0.
- e5111692 `[rollout] fix: track turn spans only under turn_level_rewards; document the response-budget boundary`:
  both `_align_generated_turn_spans` calls gated on `self.turn_level_rewards` (the `turn_rewards[-1] +=` add was
  already guarded by the empty list); the `SimpleNamespace` loop in `test_tool_call_id_on_cpu.py` gained the flag
  (the processing-tools site reads it now). Docs paragraph "The response-budget boundary", a comment at the site,
  and `test_step_reward_and_done_are_kept_when_the_final_context_merge_hits_the_response_budget`
  (`make_loop(response_length=14)`: 13 tokens fit, the third observation would make 15).
- a600e810 `[rollout] test: turn spans through the real Qwen Continuous Token merge path`:
  `test_turn_spans_real_tokenizer_on_cpu.py` drives the loop's real `ct_build_initial_tokens` /
  `ct_merge_assistant_token` / `ct_merge_context_msg` on `create_continuous_token_builder(tokenizer,
  hf_model_type="qwen2")` over `~/models/Qwen/Qwen2.5-0.5B` (CI's model-path convention; skipped when absent) with
  the real `HermesToolParser`; the junction is the `<|im_end|>` stop, after which the Qwen author inserts the
  missing newline as a mask-0 token. The trimmed-prefix junction (`removed_prefix_token_count`, GLM / Gemma
  authors) cannot be produced by the Qwen author and stays covered by the scripted `MergeResult`s.

Decided not to change: the behaviour at the response-budget boundary (documented and tested instead; changing it
would alter the pre-existing truncation rule); `done` stays independent of the flag (an episode-ending signal is
useful without dense rewards); the in-place mutation of the dataset's `extra_info` is gone with the copy (nothing
downstream read the mutated dict: `grep rollout_reward_scores` / `["num_turns"]` outside the managers is empty).

Bench deliverables (item 7) in packages/p4/bench: `gsm8k_tool_reward.py` (verl's strict GSM8K scorer + the delivery
diagnostics `turn_count`, `turn_rewards_sum`, `episode_done`, `spans_ok`, `metadata_present`, `num_turns`; the bench
agent ships an identical copy in bench/gsm8k_tool/) and `ENABLED-RUN.md` (arms, knobs, proof metrics; numbers
pending). The gsm8k tool itself is the bench agent's port in bench/gsm8k_tool/ (verl removed its own in #6302); a
duplicate port written here was deleted to keep one source.

Verification: packages/p4a/tests.log (the P4a files, 48 passed @ a600e810); the full CPU regression ran on the
stacked tree (packages/p4b/tests.log); pre-commit 14/14 hooks on each commit's files; worktree clean.

### Re-review (lead, 2026-10-29): LOW 1-2 closed by d8c4b925

- LOW 1: `extra_info_for_reward_fn` now returns a non-mapping `extra_info` unchanged (no merge, no `num_turns`), with one
  `logger.warning` per type (module-level set `_NON_MAPPING_EXTRA_INFO_WARNED`), matching main's DAPO pass-through; the
  return annotation is `Any`. LOW 2: list values of `tool_extra_fields` are copied into the dict handed over.
- Acceptance: reviewer round-2 suite r2a/r2b/r2c on this tree 46 passed; the 4 failures (`test_a_string_extra_info_now_raises...`
  x3, `test_the_copy_is_shallow...`) assert the pre-fix behaviour the lead asked to change and fail by design
  (review-tests not edited). Author suite 51 passed; `tests/experimental` group 98 passed; pre-commit 14/14.

### GPU evidence landed (bench agent, 2026-10-29; package files only, branch unchanged)

The enabled run on the stacked tree p4b@988a10bc (packages/p4/bench/ENABLED-RUN.md, bench/README.md "Review round 2") proves
the data path end to end (metadata_present = spans_ok = 1.0 vs 0.0 on CONTROL, rows_with_turn_spans 512 on 6/6 steps,
returns/mean 0.95-1.13 vs ~0, score 1.16-1.34, acc 0.599/0.633/0.694 vs 0.608/0.627/0.684). The 30-step two-seed curves show
the same accuracy within +-2 points; the dense correctness reward buys more tool calls (turn_count 1.44 -> 2.6-2.7) and 10-28 %
longer steps, not accuracy: PR-BODY.md says so plainly and makes no algorithm-performance claim. Stock suite on 988a10bc:
503 / 0 / 8 + torchrun 2/2. No docs claim in the branch contradicts these facts (grep: no accuracy or speed claim in
docs/advance/agent_loop.rst, the docstrings or rollout.yaml), so nothing changed on the branch.

### Round 3 (the AI follow-up review, 2026-10-29): deliverables for the bench agent, delivered first

- `packages/p4/bench/gsm8k_tool_reward.py` v2: the outcome is the correctness of the LAST submitted answer parsed from the
  trajectory text (last `<tool_call>` JSON `arguments.answer`, judged with the tool's own flexible rule); an episode ended
  by `done` is scored by that same submission; no-submission trajectories fall back to the strict `#### <answer>` rule
  (`reward_kwargs.fallback_to_final_text=false` scores them 0); diagnostics `submissions`, `distinct_submissions`,
  `repeated_submission` (the reviewer's analyze_runs.py rule), `scored_by_submission`, `final_text_correct` + the metadata keys.
- `packages/p4/bench/ENABLED-RUN.md` "v2 design": three arms (A outcome-only GRPO, B dense + reinforce_plus_plus, C dense +
  turn_rloo), overrides per arm, `step_reward: first_correct` + `done_when_correct: true` (bench agent's tool), the stage
  table and the per-arm repeated-submission reporting rule. `packages/p4/bench/RESULTS.md`: the round-2 disclosure.

### Round 3: R2-04 closed by 0d4cd8ca

- What changed: `extra_info_for_reward_fn(data_item, merge_tool_fields=True, rollout_fields=True)`; with
  `merge_tool_fields=False` it returns `non_tensor_batch.get("extra_info", {})` exactly as main's dapo did (the object
  itself, so gdpo's `experiment_name` lands on the dataset dict as on main). dapo / gdpo pass `merge_tool_fields=self.
  merge_tool_fields`, rate_limited passes `rollout_fields=self.rollout_fields`, both read at init from
  `turn_level_rewards_enabled(config)` (`OmegaConf.select` with default False for a DictConfig, a key walk for dicts /
  namespaces); naive / remote unchanged (merge + rollout fields always, their main contract). Docs and the PR body
  state the per-manager flag-off contract and the collision precedence.
- Decided not to change: rate_limited keeps merging `tool_extra_fields` with the flag off (that is its main contract;
  only the standardized `num_turns` / `rollout_reward_scores` are gated), so "rate-limited" sits on both sides of the
  lead's sentence by design. Nothing in the batch is mutated on the merge path (copies, round 2).
- Acceptance: the review's probe prints 7.0 on main and on this branch (tests.log); author file 20 passed (62 across the
  P4a set); `tests/experimental` group 109 passed; pre-commit 14/14. The reviewer's round-2 suite r2a/b/c: 37 passed,
  15 r2a cases fail by design (they construct dapo / gdpo / rate_limited without the flag in the config and assert the
  merged keys, i.e. the pre-R2-04 contract that the submitter's decision reverses); review-tests not edited.

### Round 3 closed: v2 arms ran on e1052882 (bench agent; package files only)

Pooled val acc at steps 0..30: A 0.668 0.731 0.791 0.811 0.815 0.825 0.832, B 0.663 0.752 0.795 0.806 0.812 0.818 0.822,
C 0.662 0.757 0.789 0.802 0.812 0.828 0.829 (step 30 per seed A 0.835/0.830, B 0.823/0.821, C 0.841/0.818); repeated
submissions <= 1.4 % (v1: 70-82 %), no episode with >= 2 step points; episode_done = turn_rewards_sum 0.82-0.84 (done on
GPU); rows_with_turn_spans 512 x 30 on both C seeds; returns/mean 1.39-1.42 (B/C) vs -0.05 (A). Reading written into
PR-BODY.md: no demonstrated accuracy gain for either estimator, turn_rloo indistinguishable from reinforce_plus_plus on
the same dense reward, no algorithm-performance claim (GSM8K is a single-answer task). Status: MERGE-READY.
