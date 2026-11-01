# P3b [2/2] notes — dead agent-loop worker detection and relaunch (V1)

Branch `contrib/p3b-worker-relaunch` (wt/p3b), one commit a2ce4020 on `contrib/p3a-inflight-ttl` @ 7f43e43e.
Its tree equals the combined reference `contrib/p3-v1-never-hang` @ 39f95929 plus p3a's rotation unit test.
Product +253/-25 vs p3a (6 files incl. `AgentLoopConfig`), tests +572, docs +9, generated yaml +8.

Decisions (facts and the original design in packages/p3/NOTES.md; review fixes in packages/p3/REVIEW.md section 8):
- `_launch_worker` / `_init_agent_loop_workers` live in `AgentLoopManagerTQ` (same options block as the base
  manager); `verl/experimental/agent_loop/agent_loop.py` is untouched (CODEOWNERS lane).
- Re-creation instead of `max_restarts` (reasons in packages/p3/NOTES.md decision 4); the old handle is
  `ray.kill`ed best-effort.
- Owner map only with the flag on, pruned on every reap against the in-flight set of every partition
  (the buffer passes the union so train prompts survive a validation probe); orphaned uids accumulate in
  `_orphaned_uids` until the partition that owns them polls (D4).
- Probe: `ray.wait` on `ping` with a 2 s timeout; timeout = alive; `ActorDiedError` = dead;
  `ActorUnavailableError` must repeat `WORKER_UNAVAILABLE_PROBES_BEFORE_DEAD` (3) times; the dispatch path
  retries it with 1 s sleeps (D9). `_relaunch_streak` per index, reset by a served dispatch, cap 5 (D8).
- Validation: the probe runs during `sample("val")`; orphaned val groups are marked `failure` in place
  (`_validate` accounts per uid and tolerates failure groups), the worker is relaunched (D5).
- Flag off: `logger.error(...)` then `raise error` (original `RayActorError`) (D6).
- `liveness_period_s` ctor kwarg falling back to the module constant at call time (review section 7.1).
- Reviewer suite on this branch: 6/9 pass; R3 / R4 / R9 fail at their characterization lines only (tests.log).

## Addendum (2026-10-29): rotation unit test carried on every branch
- `test_dispatch_rotates_the_start_worker_across_calls` (+ `_batch_of`) was committed on p3a and on the combined branch as new commits; p3b was rebuilt on the new p3a HEAD with the combined tree. p3a HEAD 7f43e43e (2 commits), p3b HEAD e43e784c (1 commit on p3a), combined 2f7d9c9b. `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` is empty. diff.patch / branch.txt regenerated; tests/trainer and tests/experimental re-run on both (logs/).

- R-A (2026-10-29): dispatch recovery never re-sends a chunk; parked uids flow through the fresh-uid re-issue; second-opinion ping on unavailable; commit f6846af0 (p3b) / f6366f48 (combined). `rollout/prompts_redispatched` -> `rollout/prompts_orphaned_at_dispatch`.

- R-H docs clause (2026-10-29): commit d0da57de on p3b, cherry-picked as 97f76904 on the combined branch (dead worker at a validation dispatch: chunk never re-sent, val groups failed in place, acc = 0, worker re-created).

## Review round 2 (the AI review, 2026-10-29) - rebased onto p3a 951bc3f4; commits 3e63823b, 0d28f34f
- What the review found: (1) blocker, reproduced here: when the dead worker owned every trajectory of a validation
  chunk, `sample()` returned an empty `KVBatchMeta` and `_validate` called `tq.kv_batch_get(keys=[])` ->
  `ValueError: Received an empty list as keys`; a colocated reward model failed earlier. (2) the validation sampling
  metrics were discarded (`batch, _ = sample(...)`) and the manager's relaunch counters were drained only after a
  training step (lost after the final validation, mis-attributed after `val_before_train`). (4) stale evidence
  wording. (5) #8203 refresh.
- Changes: `_validate` sums the sampling metrics over its batches, skips a batch without rows (warning; no reward
  pass, fetch, decode, dump or clear) and drains the manager's counters itself as `validation/...`;
  `_consume_recovery_metrics(prefix)` namespaces the manager's now bare counter names (`rollout/` per training step).
  The dump is skipped when nothing materialized. `_ttl_reissues_so_far` excludes dead-worker re-issues from [1/2]'s
  budget so the dead-worker path stays active when the TTL is suspended.
- Accounting decision: a failed session counts as `acc = 0` only where its data source reported `acc` on another
  session (upstream's `process_validation_metrics` policy, unchanged); a data source whose every session failed reports
  no accuracy for that validation, only `validation/inflight_dead_worker_failed`. Documented in the docs and PR body.
- Rebase: `git rebase --onto 951bc3f4 7f43e43e` conflicted in `replay_buffer.py` (ctor validation, docstrings,
  `_recover_stuck_inflight` signature / stuck set), `utils.py` (sum list) and the docs bullets (three times); the
  per-prompt-cap test `test_ttl_below_generation_time_...` now sets `max_inflight_reissues_per_step=8` because the
  default budget (2 x 2 = 4) would suspend the TTL before the cap fires (by design).
- Tests on 0d28f34f: `test_validate_recovery_on_cpu.py` 5 (new), dead-worker 13, inflight 22, aggregator / manager /
  trainer / buffer / reissue files 118 together; the review probe gets past the empty-key fetch. Pre-commit 14/14.
- Not changed: the three `validation/<relaunch counters>` are not added to the aggregator sum list (validation metrics
  never pass through `MetricsAggregator`; `validation/inflight_dead_worker_failed` stays listed as before).
- Regression harness: `scripts/run-cpu-regression.sh` targets the clone; the worktree run used
  `scripts/run-cpu-tests-wt.sh` (VERL_ROOT) with the same six groups, cap 600 s (temporary directory wrapper).

## Re-review of round 2 (2026-10-29) - rebased onto p3a 6445f21b; commit 3f3d2c24
- Rebase: p3a's A1-A5 fix moved the budget region; conflicts in `replay_buffer.py` (dead-owned prompts now
  excluded at the re-issuable-set stage and from the step counter, re-expressed in 833d0444), `trainer_base.py`
  (wiring: `train_batch_size` + `liveness_fn`), the docs bullets and the test file tail. The old
  `_ttl_reissues_so_far` is gone; the dead-worker-under-budget test passes unchanged.
- B1 (HIGH): `_maybe_log_val_generations` returns (one warning) when no sample materialized;
  `ValidationGenerationsLogger._log_generations_to_wandb` tolerates an empty list and a changed sample count
  (pads a shorter row with None to the table's width, widens the table and pads the earlier rows for a longer
  one; checked that wandb tables accept None cells). Chosen over "skip when the count changed" so a partially
  lost validation still logs what it has.
- B5 (LOW): the empty-batch warning no longer mentions a dead worker (a live worker that fails every group hits
  the same guard); when a validation produced no metric at all, `_validate` returns
  `validation/prompts_expected` / `validation/prompts_without_rows` so `fit()`'s `assert val_metrics` holds. Only
  in that case: the reviewer's B1 test pins the all-lost dead-worker result to exactly the failed count.
- Tests on 3f3d2c24: validate file 9 items (3 new: logger all-lost x2, partial-loss padding, live worker), the
  reviewer's `review-tests/round2` 7/7 (B1 x2, B5) and p3a's round2 5/5 on this tree; P3 suites 139 + 9.
  Count line fixed (13 + 9 + 24 items).

## Chaos re-run on the final tree (2026-10-29)
- Bench agent: `chaos-p3b-final-1`/`-2` on 3f3d2c24 completed 2/2 (kill -> next step 2.6 s; recovery 6.0 + 10.0 s / 14.8 s; relaunched 1; orphaned 4 + 4 / 12 + 4; `inflight_reissued` 8 / 7; exhausted 0; no `inflight_ttl_suspended` key, i.e. the budget never engaged under the 60 s TTL); the 0d28f34f pair also 2/2 (superseded); stock GPU suite on 3f3d2c24 0 failed / 8 skipped + torchrun 2/2. Cumulative knobs-on 9/9, main 0/3, knobs off 0/2, TTL-only 0/1 (expected). PR-BODY / CHECKLIST / REVIEW updated; package files only, no branch change. Status: MERGE-READY.

## Round 3 (the independent follow-up, R2-02) - rebased onto p3a d834b6b3; commit 45690b0f
- The budget charge for exhaustion failures lives in p3a; on this branch the dead-worker exemption is re-expressed
  on the charged set (ec6738dc after the rebase): candidates = (re-issuable | exhausted) - dead-owned, and the
  step counter never counts a dead worker's prompts. The reviewer's full-loop test passes here too (4/4).
- The round-2 rename of the manager's counters to bare names had broken the reviewer-owned round-1 suites
  (`packages/p3/review-tests` assert `rollout/...`): reverted in 45690b0f, the manager keeps `rollout/` keys and
  `_consume_recovery_metrics` re-keys them under its prefix; the round-1 suites are back at their recorded state
  (19 pass; R3 / R4 / R9 are characterization lines of fixed defects). The docs sentence on dead-worker prompts and
  the budget was restored in the new wording.
- Stale evidence wording fixed per the follow-up review: the bench intro now says the round-2 commits were re-run
  (chaos-p3b-final-1/-2, gpu-tests-p3b-r2-2 on 8x H200), the PR bodies say H100 for the earlier stock-suite runs
  and H200 for 3f3d2c24's, and the 2.6 s "kill to next completed step" is kept distinct from the recovery
  latency (6.0 + 10.0 s / 14.8 s in later steps). The round-3 commits are not GPU-run.

## Round 3b (re-review B1 / B4) - rebased onto p3a e598929b; commit 64212367
- B1: the dead-worker exemption was unbounded (a worker that dies after every served dispatch reset the
  no-service streak; a chunk parked on `ActorUnavailableError` whose pings time out was never relaunched). Now the
  buffer counts the prompts re-issued or failed per `global_steps` because the probe reported their worker dead
  and raises a RuntimeError naming the counts beyond `(cap + 1)` x `train_batch_size` (the TTL's allowance);
  the manager also caps re-creations of one index within a step regardless of service (the trainer hands
  `reap_dead_workers` its `global_steps`; without a step the no-service streak alone applies, which keeps the
  reviewer's round-1 streak test valid). Chosen over "hold back and wait": with the probe saying dead, waiting
  would hang; the reviewer's `test_r3_b1_*` expects a batch from a probe that reports every replacement dead and
  therefore fails with the bound's RuntimeError, by design (flagged for the reviewer).
- Padding hold-back (p3a e598929b) applies to the TTL's own failures only: a dead worker's groups are failed
  regardless; with no live producer left the all-failed RuntimeError stays.
- B4: `validation/prompts_expected` / `prompts_without_rows` are reported whenever row-less prompts exceed the
  dead-worker failures (a live worker failed its groups) or nothing else was produced; the all-dead case keeps
  the exact dict the reviewer's round-2 B1 test pins (the dead-worker counter already carries that count).
- Count line fixed (14 + 10 + 27 items). Reviewer suites on this tree: p3b round2 7/7, round3 3/5 (B1 x2 by
  design), p3a round2 5/5, round3 3/4 (A3 by design), round-1 19 pass + R3/R4/R9.

## Round 3b closure (D1, inherited H1) - rebased onto p3a 2bd5c040; commit 1b397dfd
- D1: the dead-worker allowance was per batch while one death orphans the whole in-flight share of a worker
  (async warm-up, DAPO prefetch, a node loss), so a single recoverable death could trip the bound. The bound now
  counts death events (liveness probes of the step that reported dead workers; one death orphaning many prompts
  is one event) against `(cap + 1)` x `train_batch_size`; the prompt count is kept for the message
  (`_dead_worker_discards_in_step`) and the message carries both ("N liveness probes ... M prompts of step S").
  A worker dying after every served dispatch is still one event per cycle and trips the bound.
- H1 inherited from p3a 2bd5c040 (hold-back charges nothing, re-evaluated every poll); the dead-owned exemption
  from the hold-back is unchanged (D2).
- Reviewer suites on this tree: p3b round3 9/9 (incl. the rewritten B1 and D1-D3), p3b round2 7/7, p3a round2
  5/5, p3a round3 7/8 (H2's last assertion = the removed budget spend, by design), round-1 19 pass + R3/R4/R9.

## Round 3b final closure (2026-10-29): MERGE-READY (re-review); note recorded
- The death-event allowance is loose (one event per liveness period at most), so the manager's per-step relaunch cap (`VERL_MAX_WORKER_RELAUNCHES` re-creations of one worker index within a step) is the effective bound on a worker that keeps dying and the buffer's event bound is a backstop. The chaos arm and the stock suite on 1b397dfd are running; their results go into the PR body's drop-in slot when they land (package files only, no branch commit meanwhile).

## Final closure (2026-10-29): MERGE-READY, chaos proof on the final sha
- `chaos-p3b-final3-1` on 1b397dfd completed 6/6 (kill -> next already-generated step 2.3 s; recovery 6.2 + 11.5 s in steps 5-6; relaunched 1; orphaned 4 + 4; `inflight_reissued` 7 = dead-worker; exhausted 0; no TTL-budget, hold-back or death-event key emitted); 45690b0f and 64212367 also 6/6; cumulative knobs-on 12/12. Stock suite on 1b397dfd (8x H200) 0 failed / 8 skipped + torchrun 2/2. Package files only, no branch change.
