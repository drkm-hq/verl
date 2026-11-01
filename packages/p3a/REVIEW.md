# P3a [1/2] review — `contrib/p3a-inflight-ttl` (reviewer, 2026-10-29)

Scope of this half: in-flight prompt TTL, fresh-uid re-issue with an `attempt` tag, the re-issue cap
(`max_inflight_reissues`), dispatch rotation, summed counters, DAPO-drain clear, restart orphan-row drop,
docs. No probe, no relaunch, no owner map (verified by grep and by the `ReplayBuffer.__init__` signature).
Full analysis in `packages/p3/REVIEW.md` section 9; this file holds the verdict and the residuals for [1/2].

## Verdict: MERGE-READY-PENDING-CHAOS-PROOF (7f43e43e)

D1 and D3 are closed (window walk-through and cap path in `packages/p3/REVIEW.md` 9.2-9.3; N1, N1b, N7,
N2, N2b, N2c pass on this branch), D2/D7/D10/D11/D12 verified, `tests/trainer` 455 passed / 1 skipped and
`tests/experimental` 74 passed. Remaining: two PR-body sentences (R-C, R-D), one docs clause, and the
healthy-control chaos run (TTL on, `training/inflight_reissued == 0` on every step).

## What I verified (code)

- D1 window: closed (`replay_buffer.py` re-syncs before acting and retires only `to_reissue -
  registered_prompt_uids` intersected with the uids the trainer returned; walk-through in section 9.2 of the
  package review). Works with a count-returning `reissue_fn` too.
- D3(a) cap: `attempt > max_inflight_reissues` -> `_fail_exhausted` (tag-only failure group under a fresh
  uid, old attempt cleared and retired) -> existing failure paths (async refill, sync refill, sync padding).
  Semantics: `max_inflight_reissues = k` allows k re-issues; the capped prompt is not trained in this epoch.
- D3(b) rotation in `AgentLoopManagerTQ.generate_sequences` (`_dispatch_cursor`), the only manager change.
- D2: `training/inflight_reissued`, `inflight_reissue_exhausted`, `inflight_discarded_while_draining` in the
  `MetricsAggregator` sum list; `inflight_expired_age_max` aggregates as max.
- D7 restart-time orphan-row drop only in the `uids=None` path (async modes); rows without a prompt key can
  never be sampled, so dropping them is safe. D10 bool rejection, D11 fixture `ray.shutdown()`, D12 drain.
- Default path: tag-only sync submit kept; the only default-path changes are the `dispatched_at` /
  `attempt` floats in every prompt tag and the rotated chunk-to-worker mapping after a sub-`num_workers`
  dispatch (equivalent behaviour; say so in the PR body).

## Residuals for [1/2]

| id | sev | where | finding | fix |
|---|---|---|---|---|
| R-C | LOW | `agent_loop_tq.py` rotation | default-path mapping change not mentioned in the PR body | one sentence |
| R-D | LOW | val exclusion | a worker that dies during validation still blocks `_validate` in TTL-only mode | PR-body sentence ("until [2/2]"), or fail-in-place for `val` on TTL expiry |
| docs | NIT | docs section | state explicitly that a capped prompt is not trained in this epoch (padded or replaced) | one clause |

## Results (reviewer runs on 7f43e43e, `.venv-cpu`, `PYTHONPATH=wt/p3a`)

- `tests/trainer`: 455 passed, 1 skipped (235 s). `tests/experimental`: 74 passed.
- New review suite (`packages/p3/review-tests/test_review2_*`, log `results-p3a.log`): 6 passed, 3 skipped
  (the worker tests need [2/2]). N1: a terminal write landing inside the trainer's re-issue window leaves
  exactly one live group (the fresh attempt), old rows swept. N1b: a count-returning `reissue_fn` retires
  nothing when the group turned terminal. N7: two consecutive re-issues leave one group at attempt 3. N2 /
  N2b / N2c: cap reached in sync-with-refill, async, and cap 0: no re-issue, one failure group, evicted,
  refilled exactly once, step completes.
- The original R1-R9 suite is not applicable here by construction (it builds the buffer with `liveness_fn`
  and binds `_reap_dead_workers`); R2 fails only at the [2/2]-only key `training/inflight_dead_worker_reissued`.
- Branch identity at the end of the pass: p3a 7f43e43e unchanged; p3b e43e784c rebuilt on it; combined
  2f7d9c9b; `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` empty. No Ray processes left.

## 10. Review round 2 (the submitter)

| finding | fix commit | test proving it | status |
|---|---|---|---|
| 3. No total recovery budget: the per-prompt cap resets on refills, so re-issues were unbounded when every generation exceeds the TTL (async refill, sync with `sync_refill_failed_groups`) | 951bc3f4: `trainer.v1.sampler.max_inflight_reissues_per_step` (null = `max_inflight_reissues` x batch size), counted over one `sample()` / `wait_for_sampleable()` call; once spent the TTL is suspended for the rest of the call (one warning, `training/inflight_ttl_suspended=1`) and the loop waits as without a TTL | `test_every_replacement_slow_is_bounded_by_the_step_budget_then_the_ttl_waits[sync|separate_async]` (real `_reissue_inflight_prompts`; 3 re-issues == budget, 2 exhaustions + refills, warning once, step completes when the slow generation finishes), `test_step_budget_caps_a_poll_to_the_oldest_expired_prompts`, `test_step_budget_defaults_to_the_per_prompt_cap_times_the_batch`, aggregator test | closed |
| 3b. "Never hang" wording | 951bc3f4 (docs) + PR-BODY.md | n/a | narrowed to: bounded per prompt and per step; a TTL below the honest generation time degrades to waiting; a generation that never returns still blocks once the budget is spent, as on main |
| 4. Evidence wording (pending GPU, obsolete survival counts, "re-issues stayed zero on every completed run") | PR-BODY.md, bench/RESULTS.md, CHECKLIST.md | n/a | fixed from `bench/results/*/summary.json`: main 0/3, TTL-only 0/1 (expected), TTL + relaunch 5/5; pre-final runs `inflight_reissued=0` / `prompts_redispatched=4`; f6846af0 `inflight_reissued=8`; the proof ran on f6846af0 (= d0da57de product code) and was re-run on the final [2/2] sha 3f3d2c24 (contains 6445f21b): 2/2 completed, cumulative knobs-on 9/9, stock GPU suite green; filled into the PR body |
| 5. REBASE-PLAN vs #8203 + duplicate check | REBASE-PLAN.md, PR-BODY.md | n/a | refreshed 2026-10-29: #8203 open / not draft / REVIEW_REQUIRED / updated 10-23 / CI partly red; landing first viable, re-target ~1 day; new adjacent PR #6641 named |
| 1, 2 (validation) | belong to [2/2] | see packages/p3b/REVIEW.md | n/a here |

### Round-2 re-review (A1-A5, count line) -> fixed in 6445f21b
| finding | fix | test | status |
|---|---|---|---|
| A1 HIGH `max_inflight_reissues=0` made the TTL inert | budget applied to the re-issuable set only; exhaustion runs first | reviewer `test_r2_a1_*`; `test_zero_max_inflight_reissues_fails_an_expired_prompt_on_its_first_expiry` | fixed |
| A2 MED exhausted prompts not failed once the allowance is 0 | same | reviewer `test_r2_a2_*`; `test_exhausted_prompts_are_failed_and_the_drain_discards_after_the_budget_is_spent` | fixed |
| A5 MED DAPO drain stopped discarding | draining branch precedes the budget | reviewer `test_r2_a5_*`; same author test | fixed |
| A4 MED budget per call, not per step | counter on the buffer keyed by `global_steps` (`_begin_ttl_budget`), default `max_inflight_reissues` x `train_batch_size` | reviewer `test_r2_a4_*`; `test_step_budget_is_shared_by_the_sampling_calls_of_one_step_and_reset_by_the_next` | fixed |
| A3 LOW false `inflight_ttl_suspended` | reported only when a re-issuable prompt is held back | reviewer `test_r2_a3_*`; the exhausted/drain author test | fixed |
| LOW count line | PR-BODY: 23 + 21 items, 298 passed | n/a | fixed |

**Status after the chaos re-run on the final tree (2026-10-29): MERGE-READY** (6445f21b; reviewer round-2 suite 5/5; chaos 2/2 on 3f3d2c24, cumulative 9/9; stock GPU suite 0 failed / 8 skipped + torchrun 2/2 on 3f3d2c24).

## 11. Round 3 (the independent follow-up): R2-02 -> fix -> test -> status
| finding | fix commit | test proving it | status |
|---|---|---|---|
| R2-02 MEDIUM: `max_inflight_reissues=0` (and any cap once its re-issues are spent) livelocks in refill modes: exhaustion failures bypassed the step budget and the refill restarted at attempt 1 | d834b6b3: exhaustion failures charged against `max_inflight_reissues_per_step` like re-issues; TTL suspended for the step once spent; default `(cap + 1)` x `train_batch_size`; drain exempt; wording narrowed; wait-vs-abort policy listed as a follow-up | `test_ttl_budget_full_loop_on_cpu.py` (the reviewer's test, unchanged body: sync-refill and async, cap 0 and 3, virtual clock) 4/4 on p3a and p3b; `test_exhaustion_failures_are_charged_against_the_budget_and_the_drain_still_discards`; `test_zero_max_inflight_reissues_fails_an_expired_prompt_on_its_first_expiry` (charge asserted) | fixed |
| consequence for round-2 reviewer tests | n/a | `test_r2_a2_*`, `test_r2_a3_*` now fail: they assert exhaustion outside the budget and no suspension for held-back exhausted prompts, the semantics R2-02 replaces; A1, A4, A5 pass | superseded (kept unedited) |

### Round-3 re-review (A1 / A3) -> fixed in e598929b
| finding | fix | test | status |
|---|---|---|---|
| A1 MEDIUM padding mode: the budget failed every slow prompt once, `sample` raised the all-failed RuntimeError | the sync sampler asks the recovery to keep the last finishable groups: held back, TTL suspended for the step, waited for | reviewer `test_r3_a1_*` x2 (cap 0 / 3); `test_padding_mode_never_lets_the_ttl_fail_the_last_finishable_groups` | fixed |
| A3 LOW explicit `max_inflight_reissues_per_step=0` accepted (TTL off); flag summed | positive int or null; flag reduced as a max | `test_step_budget_defaults_to_the_per_prompt_cap_plus_one_times_the_steps_prompt_count` (0 rejected), aggregator test | fixed; reviewer `test_r3_a3_*` asserts acceptance of 0 and fails by design |

### Round-3b closure (H1) -> fixed in 2bd5c040
| finding | fix | test | status |
|---|---|---|---|
| H1 MEDIUM the hold-back spent the step budget, so a dead writer beside a healthy slow group hung the padded step | hold back per poll, charge nothing, warn once per step | reviewer `test_r3b_h1_*` x2; `test_hold_back_is_lifted_once_another_group_finished_so_a_dead_writer_is_padded` | fixed; reviewer `test_r3b_h2_*`'s last assertion (budget marked spent) is the removed behaviour and fails by design |

| round-3b final closure | n/a | n/a | MERGE-READY (re-review); the [2/2] chaos arm and stock suite on 1b397dfd (contains 2bd5c040) are running |

| final closure (chaos proof on the final sha) | n/a | n/a | MERGE-READY: `chaos-p3b-final3-1` on 1b397dfd 6/6 (2.3 s to the next already-generated step, recovery 6.2 + 11.5 s, relaunched 1, orphaned 4 + 4, reissued 7, exhausted 0), cumulative knobs-on 12/12; stock suite on 1b397dfd 0 failed / 8 skipped + torchrun 2/2 |
