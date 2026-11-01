# P4a review round 3 — AI re-review verdict (the independent follow-up R2-04 + the P4 benchmark section)

Reviewer: Claude Fable 5.1 (AI), adversarial, read-only on the branch, 2026-10-29. Tree: `contrib/p4a-turn-signals` @ 0d4cd8ca
(9 on main @ 2ab35292), worktree `wt/p4a`; stacked `wt/p4b` @ e1052882 (rebase only) also run. Package audit line:
`p4a   contrib/p4a-turn-signals   0d4cd8ca  9   0   fresh  ok   none`.

## the submitter's probe (`flag_off_reward_repro.py`, scratch copy, DAPO with the flag off, callback returns `extra_info["num_turns"]`)

`verl @ 2ab35292: 7.0 [{'num_turns': 7, 'tool_rewards': [123.0]}]` and `wt/p4a @ 0d4cd8ca: 7.0 [{'num_turns': 7, 'tool_rewards': [123.0]}]`
(wt/p4b @ e1052882 identical). Before the fix the branch printed 3.0 with the dataset's `tool_rewards` replaced.

## Findings

| Finding | Status | Evidence |
|---|---|---|
| R2-04: flag-off callback contract changed for dapo / gdpo / rate_limited | FIXED | r3a runs `r3_probe.py` in a subprocess on main and on the branch (flag key absent = off) and compares per manager: the keys and values the callback receives are identical for all five (dapo / gdpo: dataset `num_turns` 7, `tool_rewards` [123.0], `rollout_reward_scores` the dataset's; rate_limited: `tool_extra_fields` merged, dataset `num_turns` 7 kept, no rollout fields; naive / remote: 3, [2.0], `{'acc': 1.0}`); dapo / gdpo hand over the dataset's own object (gdpo's `experiment_name` lands on it, as on main). rate_limited's legacy `__call__` batch path scores through `run_single`, so it follows the same contract (r3b). The legacy `verl/workers/reward_manager` registry holds no class sharing the helper (only `rate_limited` is dual-registered) and is unreachable from the reward loop. |
| Object identity | PRESERVED where main passed the object | dapo / gdpo off: the stored object itself, a mutating callback reaches the dataset dict exactly as on main (r3b). naive / remote / rate_limited: main merged into the dataset dict in place, the branch hands a copy (the round-2 copy the independent follow-up listed under verified closures); unobservable in the shipped flow, since the reward worker scores a pickled `DataProto` (`agent_loop.py:1133 compute_score.remote(data)`), visible in-process only. |
| Config lookup | SOUND | `turn_level_rewards_enabled`: absent key, `multi_turn: null`, `rollout: null`, `actor_rollout_ref: null`, MISSING, struct mode, a reward-only config, plain dict, namespace, `None` all read False; the generated `ppo_trainer` yaml reads its real bool (False, True after `OmegaConf.update`); read once at init (r3b). |
| Flag-on contract | HOLDS | all five managers deliver `generated_turn_spans` / `turn_rewards` / `episode_done` / `tool_rewards` + `num_turns` + `rollout_reward_scores` as a fresh dict (r2a on/off matrix, 82 cases; author's 5 x 2 probe test). |
| PR body / docs / package | DONE | "tensors bitwise; callback contract of every manager unchanged with the flag off" with the per-manager sentence; repeated submissions disclosed (923 / 1085 of 1319 episodes, 901 / 1049 with >= 2 step points, controls 2-3 / 0); "`data.seed` the only difference"; "no demonstrated accuracy gain"; no "same curve within noise" (grep clean); CHECKLIST rows refreshed; tests.log @ 0d4cd8ca: 62 passed, both probes recorded (7.0 / 7.0), `tests/experimental` 109. |

## New defects (none HIGH / MEDIUM)

- LOW `turn_level_rewards_enabled` applies `bool()`: a string `"false"` / `"False"` reads as on. Consistent with the loop's own
  `if self.turn_level_rewards:`, and the composed config holds a real bool, so only a quoted CLI override reaches it (r3b).
- LOW wording: "callback contract unchanged" is exact for a mapping `extra_info`; the two disclosed deviations are the
  non-mapping pass-through under naive / remote (main raised `TypeError`) and the copy noted above.

## Tests (CPU harness from the rules file, `PYTHONPATH=<tree>:<dir>`)

- Round 3, `packages/p4a/review-tests/round3/`: `r3_probe.py` (helper), `test_r3a_flag_off_contract_vs_main_on_cpu.py` (11),
  `test_r3b_flag_matrix_and_reader_on_cpu.py` (19): 30 passed on wt/p4a @ 0d4cd8ca and 30 passed on wt/p4b @ e1052882.
- Round 2 updated to the per-manager flag-off contract (the 15 r2a cases that asserted the reversed contract, now an
  on/off matrix): r2a / r2b / r2c 82 passed on wt/p4a and 82 passed on wt/p4b; worktrees clean before and after.

## Merge-readiness: MERGE-READY (pending the v2 enabled run the package labels pending)

R2-04 is closed by measurement against main, manager by manager, including the batch path and object identity; the
benchmark disclosure is in the PR body; the two remaining items are LOW.

## Lead closure (2026-10-29, p4a 0d4cd8ca / p4b e1052882)
R2-04 FIXED by measurement against main (the submitter's probe: 7.0 on main and on the branch; the five managers' flag-off
contract identical key by key). The two LOW notes are accepted without a further commit: a string "false" for the
flag in a hand-built config reads as on (the composed config carries a real bool; same truthiness as the loop), and
the "contract unchanged" sentence is exact for mapping extra_info with the non-mapping pass-through and the copy as
the two disclosed deviations. Verdict: MERGE-READY pending the v2 enabled runs (p4-v2-{A,B,C}-s{1,2} on e1052882).
