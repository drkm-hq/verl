# P3a [1/2] review round 2 verdict (AI re-review, 2026-10-29)

Branch `contrib/p3a-inflight-ttl` @ 951bc3f4 (2 commits on main 2ab35292), wt/p3a clean. package-audit:
`p3a   contrib/p3a-inflight-ttl         951bc3f4  2    0     fresh  ok      none`.
Author suite re-run in wt/p3a alone (no p3b): 41 passed (test_inflight_recovery_on_cpu 20 items + test_metrics_aggregator_on_cpu 21).

## the submitter's findings
| finding | status | evidence |
|---|---|---|
| 3. no total recovery budget | PARTIAL | The bound exists and the repeated-slow-replacement tests (sync + async, real `_reissue_inflight_prompts`) are real. But `_expired_within_budget` is applied to the whole expired set in `_recover_stuck_inflight` BEFORE the exhausted split and BEFORE the `draining` branch, so an allowance of 0 also disables the two paths that are not regenerations (A1, A2, A5), and the budget is per sampling call, not per step (A4). |
| 3b. "never hang" wording | FIXED | docs/PR body: bounded per prompt and per step, degrades to waiting, "never livelocks"; a never-returning generation still blocks once the budget is spent, stated. "Per step" is really per call (A4). |
| 4. evidence wording | FIXED | PR-BODY / CHECKLIST / bench: main 0/3, TTL-only 0/1 (expected), TTL + relaunch 5/5 with per-run counters, f6846af0 re-issued 8, 951bc3f4 not GPU-run and said so; no pending-GPU phrasing, no "stayed zero on every run". |
| 5. REBASE-PLAN / duplicates | FIXED | refreshed 2026-10-29: #8203 status and hunk map incl. the budget; #6641 named. |

## New defects (tests: `packages/p3a/review-tests/round2/test_r2_step_budget_on_cpu.py` + `_r2_budget_common.py`; in-memory TQ patched into `replay_buffer.tq`, real `ReplayBuffer._recover_stuck_inflight`; all 5 FAIL on 951bc3f4 (`results-p3a-951bc3f4.log`) and identically on p3b 0d28f34f (`results-on-p3b-0d28f34f.log`))
- A1 HIGH: `max_inflight_reissues=0` (the validator accepts it; yaml: "before it is marked failure instead") makes the TTL inert: the default budget is 0 x batch, allowance 0, the expired prompt is neither failed nor re-issued (`metrics={'training/inflight_ttl_suspended': 1}`) and sampling waits as on main. `:27`.
- A2 MEDIUM: once the allowance is 0 an exhausted prompt (attempt > cap) is no longer failed, so the failure / refill path the yaml, the docs and NOTES ("the exhausted-prompt path still fails + refills (not counted against the budget)") promise never runs in that step; under the default budget the uniform-slow case never reaches it. `:42` (`inflight_reissued=1, inflight_reissue_exhausted=0`, prompt still in flight).
- A5 MEDIUM: the sync DAPO drain stops discarding expired in-flight groups once the allowance is 0, so the drain waits for a group it would discard anyway (a hung one blocks the step). `:85`. One fix covers A1/A2/A5: compute `exhausted` and the draining discard from the un-capped expired set and apply `_expired_within_budget` to `to_reissue` only.
- A4 MEDIUM: the budget is per sampling call: `sample()` and `wait_for_sampleable()` each allocate `eviction_metrics` and `reissue_budget`; a step with `parameter_sync_step=k` makes k calls (k+1 in separate_async: `prepare_step`'s switch wait + the mini-batch), each with the full allowance. Default per-step bound up to `max_inflight_reissues x (threshold + train_batch_size)`, explicit `max_inflight_reissues_per_step=N` allows up to (k+1)N; the "suspended for the rest of this step" warning is wrong for k > 1. `:72` (budget 1, two calls, 2 re-issues). Carry the counter across the calls of one step, or name and document the key per call.
- A3 LOW: `inflight_ttl_suspended=1` and the warning fire when the surplus is exhausted prompts that are then failed, with zero re-issues. `:58`.
- LOW: PR-BODY "`# -> 21 + 8 passed`": the two files collect 20 + 21 items (41 passed as run).

## Hygiene
diff.patch fresh, branch.txt ok, worktree clean, no remote, no Ray processes left. AI statement verbatim; GPU suite labelled as the stock selection that excludes the new tests; no "reviewed every line"; no upstream contact claimed ("happy to rebase" is an offer); numbers conservative; tests.log 295 passed matches the body. CHECKLIST truthful except the count line above.

Run: `cd wt/p3a && PYTHONPATH=$PWD .venv-cpu/bin/python -m pytest -c pytest-cpu.ini --rootdir packages/p3a/review-tests/round2 --asyncio-mode=auto -p no:cacheprovider -q packages/p3a/review-tests/round2` (paths relative to verl-contrib).

## Merge-readiness: NEEDS-FIXES
A1/A2/A5 (one reorder in `_recover_stuck_inflight`, plus tests with `max_inflight_reissues=0` and with an exhausted prompt after the budget is spent) and A4 (per-step accounting or honest naming) before filing; A3 and the count line with them. The TTL / fresh-uid / sweep core has no new finding.

## Lead closure (2026-10-29, after p3a 6445f21b / p3b 3f3d2c24)
All findings applied by the author: A1/A2/A5 (budget applied to the re-issuable set only, after the exhausted
split and the drain branch), A4 (budget per step keyed on global_steps; default allowance max_inflight_reissues x
data.train_batch_size), A3 (suspension reported only when a re-issuable prompt is held back), B1 (generations logging
skipped without samples; wandb table tolerates an empty list and a changed sample count), B5 (neutral warning;
validation/prompts_expected + validation/prompts_without_rows returned when nothing else is). The reviewer's round-2
tests pass unedited (p3a 5/5 on both trees; p3b 7/7). Full CPU regression on wt/p3b @ 3f3d2c24: 1533 / 64 / 0.
Verdict: MERGE-READY-PENDING-CHAOS-PROOF (two chaos arms + the stock suite queued on 3f3d2c24).
