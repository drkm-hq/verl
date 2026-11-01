# P3b [2/2] review — `contrib/p3b-worker-relaunch` (reviewer, 2026-10-29)

Scope of this half (on top of p3a): `ping` probe, dead-worker relaunch at dispatch and from the probe,
owner map gated by the flag and pruned per reap, `_orphaned_uids` handover across partitions, validation
groups of a dead worker failed in place, relaunch streak cap, `ActorUnavailableError` handling, original
`RayActorError` re-raised with the flag off, `_launch_worker` inside the TQ manager. Full analysis in
`packages/p3/REVIEW.md` section 9; this file holds the verdict and the residuals for [2/2].

## Verdict: MERGE-READY-PENDING-CHAOS-PROOF (f6846af0)

History: at e43e784c the verdict was NEEDS-FIXES for R-A (the dispatch-path retry on `ActorUnavailableError`
re-sent the same chunk to a live worker whose first dispatch may have executed; N5 reproduced two attempts
of one uid). Commit f6846af0 removes every re-send: an actor-error dispatch parks the chunk's prompts in
`_orphaned_uids` for the buffer's fresh-uid re-issue, a dead worker is re-created and its owned prompts
parked, an unavailable reply gets one bounded second-opinion `ping` under the consecutive-failure rule.
Re-check (`packages/p3/REVIEW.md` section 11): `test_review2_*` 9/9 incl. N5, new N8/N8b/N11/N9 4/4,
`tests/trainer` 470 passed / 1 skipped, `tests/experimental` 74 passed; `git diff
contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` empty. Remaining before posting: docs clause R-H (a
dead worker at a validation dispatch fails that chunk's val groups, counted `acc = 0`), PR-body sentences
R-C and the metric rename (`rollout/prompts_orphaned_at_dispatch`), and the Modal kill run with a healthy
control.

## What I verified (code)

- D4: `_prompt_owner` recorded only with `relaunch_dead_workers`; pruned on every reap to the union of all
  partitions' in-flight uids (the buffer passes the union, so train prompts survive a validation probe);
  orphans parked in `_orphaned_uids` until the owning partition polls.
- D5: `_recover_stuck_inflight` runs the probe for `val` and calls `_fail_groups(fresh_uid=False)` (tag-only
  `failure` on the same uid); `_validate` tolerates failure groups; the worker is relaunched by the reap.
- D6: flag off -> `logger.error(...)` then `raise error` (original `RayActorError`).
- D8: `_relaunch_streak[index]` incremented per relaunch, reset by a served dispatch, `RuntimeError` beyond
  `MAX_WORKER_RELAUNCHES`.
- D9: probe path counts consecutive `ActorUnavailableError` results (`_probe_marks_dead`), `ActorDiedError`
  is final. Since f6846af0 the dispatch path never re-sends a chunk: an unavailable reply gets one bounded
  second-opinion `ping` and the chunk is parked for the buffer's fresh-uid re-issue.
- D2: `training/inflight_dead_worker_reissued` and `validation/inflight_dead_worker_failed` added to the sum
  list. `verl/experimental/agent_loop/agent_loop.py` back to main.

## Residuals for [2/2]

| id | sev | where | finding | fix |
|---|---|---|---|---|
| R-A | closed in f6846af0 | `agent_loop_tq.py` `_recover_dispatch` | was: the unavailable-at-dispatch retry could spawn a second attempt of the same uids on a live worker | now: no re-send on any path; chunk parked for fresh-uid re-issue (N5, N8, N8b, N11 pass) |
| R-B | closed in f6846af0 | same | was: relaunch re-dispatch reused the chunk's uids | now: parked, re-issued under fresh uids |
| R-C | LOW | rotation (inherited) | default-path mapping change not in the PR body | one sentence |
| R-H | LOW | validation dispatch to a dead worker (N9) | that chunk's val groups are failed in place and count as `acc = 0` for the step; not stated in the docs bullet | one clause, or re-dispatch the same chunk only for `ActorDiedError` during validation |

## Results (reviewer runs on e43e784c, `.venv-cpu`, `PYTHONPATH=wt/p3b`)

- `tests/trainer`: 468 passed, 1 skipped (297 s; a first run was killed by the author's process cleanup
  and re-run). `tests/experimental`: 74 passed.
- Original review suite R1-R9, unmodified: 6 passed, 3 failed at characterization lines only: R3:134 (the
  checkpoint garbage is now dropped at restart, D7), R4:47 (two re-issues, the second rotated to worker 1
  and the step completed, D3b), R9:104 (the probe ran during validation and relaunched the worker, D5).
  R1, R2, R5, R8 (the defects they pinned) now pass; R6, R7 still pass. No hidden regression.
- New review suite (`results-p3b.log`, `results-p3b-n4.log`): N1, N1b, N7, N2, N2b, N2c, N3 pass; N4 passes
  (after fixing my own post-`sample` status assertion); **N5 fails**: `a uid was spawned twice on a live
  worker: {uid0: 2, uid1: 1}` (R-A).
- Branch identity at the end of the pass: `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch`
  empty (combined 2f7d9c9b, p3b e43e784c, p3a 7f43e43e). No Ray processes left; worktrees and clone clean.

## Fix applied (author, 2026-10-29): R-A / R-B -> commit f6846af0 on contrib/p3b-worker-relaunch (cherry-picked as f6366f48 on the combined branch)

- `AgentLoopManagerTQ._recover_dispatch` never sends a chunk a second time on any path. The chunk's uids are
  parked in `_orphaned_uids` (metric `rollout/prompts_orphaned_at_dispatch`); `reap_dead_workers` hands them to the
  replay buffer, which re-issues them under fresh uids (old rows cleared) or drops them from the parked set once
  they reached a terminal status on their own. `ActorUnavailableError`: one second-opinion `ping` bounded by
  `DISPATCH_RETRY_INTERVAL_SECONDS` feeds the consecutive-failure rule (`_probe_marks_dead`); a responsive worker
  is only marked suspect, nothing is killed or relaunched. `ActorDiedError` (or the threshold reached): the worker
  is re-created and every prompt it still owned is parked too (`_orphan_worker`, shared with the probe path; this
  also closes a gap where a dead worker's earlier in-flight prompts were not handed over at dispatch time).
  Flag off: unchanged (`logger.error`, then the original error re-raised). `rollout/prompts_redispatched` is gone.
- Tests: `test_unavailable_dispatch_reply_parks_the_chunk_without_resending` (N5 mirror: executed dispatch, lost
  reply -> each uid spawned once, no relaunch, parked set pruned at the next reap),
  `test_a_chunk_is_never_dispatched_twice_under_any_recovery_path` (lost reply + dead worker in one batch, all 6
  groups sampled, every old uid dispatched at most once, fresh uids exactly once), and the two dispatch tests
  updated to the parked semantics.
- Results on f6846af0: reviewer suite `test_review2_*` 9/9 (N5 passes); my 31 new tests pass; `tests/trainer`
  470 passed / 1 skipped; `tests/experimental` 74 passed. On the combined branch f6366f48: 470 / 74, review2 9/9.
  pre-commit ruff / ruff-format / mypy on the changed files: Passed. `git diff contrib/p3-v1-never-hang
  contrib/p3b-worker-relaunch` empty; audit fresh/ok for p3, p3a, p3b.

## 10. Review round 2 (the submitter)

Base moved: rebased onto p3a 951bc3f4 (per-step TTL budget); the first three commits re-apply as 33b8439b, 1f9b6238,
571c4388 (conflicts resolved in `replay_buffer.py`, `utils.py`, the docs); the per-prompt-cap test sets an explicit
per-step budget. Head 0d28f34f.

| finding | fix commit | test proving it | status |
|---|---|---|---|
| 1. Blocker: empty validation batch (every trajectory of a chunk owned by the dead worker) -> `tq.kv_batch_get(keys=[])` raises; colocated reward fails earlier | 0d28f34f: `_validate` skips a batch without rows (warning; no reward pass / replica sleep-wake, fetch, decode, dump, clear); `expected_acc_counts` kept so the sessions score 0 where the source reports `acc`; dump skipped when nothing materialized | `test_validate_recovery_on_cpu.py` (real `_validate`, in-memory TQ with the client's empty-key guard): `test_every_group_of_a_batch_loses_its_worker_with_a_remote_reward_loop`, `test_every_group_lost_with_a_colocated_reward_model_skips_the_reward_pass`, `test_mixed_batch_decodes_the_healthy_row_and_scores_the_failed_session_zero[False|True]` (`val-core/test/acc/mean@1 == 0.5`, aligned log/dump lists), `test_single_dead_worker_owning_every_batch_counts_each_batch_and_drains_the_relaunch`; the review probe passes its former failure point | closed |
| 2. Validation recovery counters discarded; manager counters drained before validation | 0d28f34f: sampling metrics summed over the validation batches and returned; `_consume_recovery_metrics(prefix)` drained by `_validate` as `validation/...` (manager keeps bare names; the step drain namespaces `rollout/`) | the single-dead-worker test above (`validation/inflight_dead_worker_failed == 3` over two batches, `validation/dead_workers_relaunched == 1`, manager counter empty afterwards, `rollout/` drain unchanged) | closed |
| 3 (its [2/2] half): the dead-worker path must stay active under the step budget | 3e63823b: `_ttl_reissues_so_far` excludes `inflight_dead_worker_reissued`; dead-owned prompts are re-issued even when the TTL is suspended | `test_dead_worker_prompts_are_reissued_even_when_the_ttl_budget_is_spent` | closed |
| 4. Evidence wording | PR-BODY.md, bench/RESULTS.md, CHECKLIST.md | n/a | 9/9 record with per-run counters (f6846af0 line 5/5, 0d28f34f 2/2 superseded, final 3f3d2c24 2/2); stock GPU suite green on 3f3d2c24; filled into the PR body |
| 5. REBASE-PLAN vs #8203 + duplicate check | REBASE-PLAN.md | n/a | refreshed 2026-10-29 incl. the `_validate` guard and the drain; #6641 named |

Accounting caveat stated everywhere: a data source whose every session failed reports no accuracy for that validation
(nothing defines `acc`), only the failed count; this is upstream's `process_validation_metrics` policy, unchanged.

### Round-2 re-review (B1, B5, count line) -> fixed in 3f3d2c24 (on the rebase onto p3a 6445f21b)
| finding | fix | test | status |
|---|---|---|---|
| B1 HIGH all-lost validation breaks the next healthy one's wandb generations table (and the reverse) | `_maybe_log_val_generations` logs nothing without samples; `_log_generations_to_wandb` tolerates an empty list and pads / widens on a changed sample count | reviewer `test_r2_b1_*` x2; `test_generations_logger_survives_an_all_lost_validation_before_or_after_a_healthy_one[...]`, `test_generations_logger_pads_a_partially_lost_validation_to_the_table_width` | fixed |
| B5 LOW live worker failing every group blamed as dead; `_validate` returned `{}` | neutral warning; prompt counts returned when nothing else was produced | reviewer `test_r2_b5_*`; `test_live_worker_failing_every_session_without_rows_reports_counts_not_a_dead_worker` | fixed |
| inherited A1-A5 | p3a 6445f21b + 833d0444 (dead-owned prompts excluded at the re-issuable-set stage) | p3a round2 5/5 on this tree; `test_dead_worker_prompts_are_reissued_even_when_the_ttl_budget_is_spent` | fixed |
| LOW count line | PR-BODY: 13 + 9 + 24 items | n/a | fixed |

**Status after the chaos re-run on the final tree (2026-10-29): MERGE-READY** (3f3d2c24 on p3a 6445f21b; reviewer round-2 suites 7/7 + 5/5; chaos 2/2 on 3f3d2c24, cumulative knobs-on 9/9, main 0/3; stock GPU suite 0 failed / 8 skipped + torchrun 2/2 on 3f3d2c24).

## 11. Round 3 (the independent follow-up): R2-02 -> fix -> test -> status
| finding | fix commit | test proving it | status |
|---|---|---|---|
| R2-02 (inherited from [1/2]): zero-cap / spent-budget refill livelock | p3a d834b6b3 (exhaustion failures charged); ec6738dc re-expresses the dead-worker exemption on the charged set | `test_ttl_budget_full_loop_on_cpu.py` 4/4 on this tree; `test_dead_worker_prompts_are_reissued_even_when_the_ttl_budget_is_spent` | fixed |
| round-1 reviewer suites broken by the round-2 counter rename | 45690b0f: manager keeps `rollout/` keys, drain re-keys | `packages/p3/review-tests` 19 pass + R3/R4/R9 characterization lines (recorded state) | fixed |
| stale evidence wording (bench intro, H200, 2.6 s vs recovery) | package files | n/a | fixed |

### Round-3 re-review (B1 / B4 / count) -> fixed in 64212367
| finding | fix | test | status |
|---|---|---|---|
| B1 MEDIUM dead-worker exemption unbounded (die-after-service, unavailable-parked chunk) | per-step dead-worker discard count bounded by `(cap + 1)` x prompts, RuntimeError beyond; per-step relaunch cap regardless of service; the hold-back never applies to dead-owned groups | `test_dead_worker_discards_are_bounded_per_step_and_fail_loudly`, `test_relaunches_within_one_step_are_capped_regardless_of_service` | fixed; reviewer `test_r3_b1_*` x2 expect a batch and fail with the bound's RuntimeError by design |
| B4 LOW row-less counts only when no other metric | emitted whenever row-less prompts exceed the dead-worker failures or nothing else was produced | reviewer `test_r3_b4_*`; `test_live_worker_rowless_batch_is_counted_beside_a_dead_worker_batch` | fixed |
| LOW count line | 14 + 10 + 27 items | n/a | fixed |

### Round-3b closure (D1 / H1) -> fixed in 1b397dfd (on p3a 2bd5c040)
| finding | fix | test | status |
|---|---|---|---|
| D1 MEDIUM one death orphaning more prompts than the per-batch allowance tripped the bound | the bound counts death events per step (probes that reported dead workers), prompts kept for the message | reviewer `test_r3b_d1_*` x2, rewritten `test_r3_b1_*` x2; `test_dead_worker_death_events_are_bounded_per_step_and_fail_loudly` | fixed |
| H1 (inherited) | p3a 2bd5c040 | reviewer `test_r3b_h1_*` x2, `test_r3b_d2_*` | fixed; `test_r3b_h2_*`'s last assertion by design |

| round-3b final closure | n/a | n/a | MERGE-READY (re-review); note recorded: The death-event allowance is loose (one event per liveness period at most), so the manager's per-step relaunch cap (`VERL_MAX_WORKER_RELAUNCHES` re-creations of one worker index within a step) is the effective bound on a worker that keeps dying and the buffer's event bound is a backstop. Chaos arm + stock suite on 1b397dfd running. |

| final closure (chaos proof on the final sha) | n/a | n/a | MERGE-READY: `chaos-p3b-final3-1` on 1b397dfd 6/6 (2.3 s to the next already-generated step, recovery 6.2 + 11.5 s, relaunched 1, orphaned 4 + 4, reissued 7, exhausted 0), cumulative knobs-on 12/12; stock suite on 1b397dfd 0 failed / 8 skipped + torchrun 2/2 |
