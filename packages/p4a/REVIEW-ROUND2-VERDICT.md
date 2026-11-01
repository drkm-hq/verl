# P4a review round 2 — AI re-review verdict

Reviewer: Claude Fable 5.1 (AI), adversarial, read-only on the branch, 2026-10-29. Tree: `contrib/p4a-turn-signals` @ a600e810
(7 commits on main @ 2ab35292), worktree `wt/p4a`; the stacked tree `wt/p4b` @ 64f8f604 was used for the probe and a second
run. Round-2 commits attacked: 7e4e8e4f, e5111692, a600e810. Package audit line:
`p4a   contrib/p4a-turn-signals   a600e810  7   0   fresh  ok   none`.

## the submitter's findings

| Finding | Status | Evidence |
|---|---|---|
| 1. DAPO / GDPO custom reward callbacks do not receive `generated_turn_spans` / `turn_rewards` / `episode_done` | FIXED | The submitter's probe re-run from a scratch copy with `PYTHONPATH=wt/p4b`: naive / dapo / gdpo each print `{'episode_done': True, 'turn_rewards': [2.0], 'generated_turn_spans': [(0, 2)], 'num_turns': None, 'rollout_reward_scores': {}}` (gdpo adds `experiment_name`), reward 1.0 (was 1.0 / 0.0 / 0.0). `num_turns` is `None` when `__num_turns__` is absent; the worker always sets it (`agent_loop.py:1122`). Scope is complete: the reward loop resolves managers only from the experimental registry or a user class (`reward.py:resolve_reward_manager_cls`), so the legacy `verl/workers/reward_manager/*` classes are unreachable from it; `rate_limited` is registered (`limited.py:172`) and uses the helper (`:409`). My r2a (31 cases): the keys arrive under all five managers with the dataset `extra_info` absent / None / `{}` / an OmegaConf mapping, as a fresh plain `dict` each call; multi-row sessions score the last row with its own metadata; a dataset key colliding with a loop key loses to the loop. |
| 2. Span bookkeeping runs at the merge sites with the flag off | FIXED | both `_align_generated_turn_spans` calls gated (`tool_agent_loop.py:334, 462`); r2b flag-off sweep: no new keys, `turn_rewards is None`, main's scalar placement, at every budget 1..23 x 8 junction edits. |
| 4. Terminal observation absent when the context merge hits the response budget | DOCUMENTED, behaviour unchanged | `git diff 2ab35292 a600e810`: the `return AgentState.TERMINATED` at the budget check is main's line, only the comment is new; the docs paragraph matches the code (`>=`, reward and `done` recorded before the check); r2b pins the exact-fill case (budget = 3 turns + 3 observations drops the observation, one more token keeps it); r2c reproduces it with the real tokenizer. |
| 8. Real tokenizer boundary / junction test | ADDED, offline-safe | passes with `HF_HUB_OFFLINE=1` against the local `~/models/Qwen/Qwen2.5-0.5B` (tokenizer files only, no download); 2 skipped when HOME holds no model. |
| 7. 503-pass GPU suite does not exercise the new tests | WORDING DONE, enabled run PENDING | PR-BODY.md carries the rules' sentence verbatim; `packages/p4/bench/gsm8k_tool_reward.py` reads exactly the exported keys and types (`generated_turn_spans` list of `(int, int)`, `turn_rewards` list of float, `episode_done` bool, `num_turns`) and ENABLED-RUN.md's proof metrics name them. |
| 6. PR body wording | DONE | AI statement verbatim; no "already reviewed"; no upstream contact claimed; #8305 facts match read-only `gh pr view`: draft, luzy99, OPEN, updated 2026-10-21T03:43Z, +964/-3, 19 files. |

## The round's key risk, attacked from the producer side (the consumer side is in packages/p4b)

[2/2] now rejects any span outside `prev_end <= start < end <= length`. r2b (scripted loop, every `response_length` 1..23 x 8
junction edits with 0-2 trimmed and 0-2 inserted tokens) and r2c (the real Qwen2.5-0.5B merges and Hermes parser, every
budget 1..92) show the exported spans are always ordered, non-empty, non-overlapping, within `len(response_ids)`, exactly the
runs of `response_mask == 1`, index-aligned with `turn_rewards`, decoding to the generated text or its truncated prefix; the
dense `rm_scores` row lands only on generated tokens and drops no reward. Both budget boundaries (cut mid-generation; `done`
with the terminal merge dropped) and the complete episode occur in every sweep. No HIGH / MEDIUM defect.

## New defects (none HIGH / MEDIUM)

- LOW `extra_info_for_reward_fn` requires a mapping: a `str` `extra_info` column (`dict(str)`) raises
  `ValueError: dictionary update sequence element #0 has length 1` under every manager; main's DAPO passed it through to the
  reward function untouched (probe: main 1.0, branch ValueError; naive / gdpo raised on main too). State the contract or
  fall back to `{}` with a warning.
- LOW the copy is shallow: the dict is new, but `generated_turn_spans` / `turn_rewards` are the loop's own lists, so a reward
  function that mutates them changes what `_postprocess` and the trainer see (r2a); the docstring says "a copy".

## Tests added (packages/p4a/review-tests/round2/; CPU harness from the rules file, `PYTHONPATH=<tree>:<this dir>`)

- `test_r2a_extra_info_contract_on_cpu.py` (31), `test_r2b_budget_sweep_scripted_on_cpu.py` (17, imports
  `packages/p4/review-tests/scripted_loop.py`), `test_r2c_real_tokenizer_budget_sweep_on_cpu.py` (2): 50 passed on
  wt/p4a and 50 passed on wt/p4b. Author's own P4a files re-run by me: 48 passed; regression groups `tests/trainer` 470
  passed / 1 skipped and `tests/experimental` 96 passed on the stacked tree (= tests.log).

## Hygiene

diff.patch fresh, branch.txt names HEAD, worktree clean before and after my runs, tests.log (48 passed @ a600e810) matches
the PR body and CHECKLIST.md, which states the additive keys for dapo / gdpo honestly. Docs name the five managers.

## Merge-readiness: MERGE-READY (pending the enabled GPU run the package already labels pending)

All six round-2 items are fixed or documented with tests that assert the claims; the producer cannot emit a span the
stacked estimator rejects; the two new items are LOW wording / contract notes.
## Lead-requested closure (2026-10-29, p4a @ d8c4b925)

LOW 1 and LOW 2 are closed correctly: a non-mapping `extra_info` is handed over unchanged with one `logger.warning` per type
(main's DAPO behaviour; verified with the warning set cleared and two calls), and list-valued tool fields are copied. The four
tests that pinned the pre-fix behaviour now assert the fixed one (`..._is_passed_through_unchanged_with_one_warning_per_type` x4
managers, `test_list_valued_tool_fields_are_copied_...`). One note, pinned, not a wrong fix: `gdpo` assigns `extra_info["experiment_name"]`
after the helper, so a `str` still raises `TypeError` there, exactly as on main (LOW). Counts: r2a / r2b / r2c 52 passed on
wt/p4a @ d8c4b925 and 52 passed on wt/p4b @ 988a10bc; worktrees clean. Verdict unchanged: MERGE-READY.
