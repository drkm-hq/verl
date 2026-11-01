# P3a [1/2] notes — in-flight prompt TTL with fresh-uid re-issue (V1)

Branch `contrib/p3a-inflight-ttl` (wt/p3a), one commit 7f43e43e on main @ 2ab35292, carved from the combined
reference `contrib/p3-v1-never-hang` after the adversarial review (packages/p3/REVIEW.md sections 2 and 8).
Product +288/-22 (7 files, docstrings included), tests +563, docs +17, generated yaml +8.

Decisions (see packages/p3/NOTES.md for the facts established and the original design):
- Carve rule: everything that works without a worker probe. The manager change is the rotation only
  (`agent_loop_tq.py` `_dispatch_cursor`); no `ping`, no owner map, no relaunch.
- `_reissue_inflight_prompts(uids=...)` returns the retired old uids (list); `uids=None` keeps the int.
  The buffer retires only `to_reissue - registered_prompt_uids` intersected with that list (D1).
- `attempt` tag + `trainer.v1.sampler.max_inflight_reissues` (3): beyond it `_fail_exhausted` writes a fresh
  failure uid (tag-only) and retires the old one; the failure paths finish the step (D3a).
- Sync DAPO drain clears expired prompts instead of regenerating (D12); restart-time re-issue drops orphan
  rows (D7); bool TTL rejected (D10); counters summed (D2); unit fixture shuts Ray down (D11).
- A re-issue to a dead worker raises `RayActorError` (as main); [2/2] adds detection and relaunch.
- Hunks: `replay_buffer.py` `_expired_inflight_keys` / `_sweep_retired_attempts` / `_fail_exhausted` /
  `_recover_stuck_inflight` + hooks in `sample` and `wait_for_sampleable`; `trainer_base.py`
  `_prompt_fields`, `_pending_prompt_tag(global_steps, attempt)`, `_build_replay_buffer` wiring,
  `_submit_batch_to_rollout`, `_reissue_inflight_prompts`, `_validate` stamp; `utils.py` sum list;
  `ppo_trainer.yaml` two keys; docs section "Self-healing: in-flight prompt TTL".

## Addendum (2026-10-29): rotation unit test carried on every branch
- `test_dispatch_rotates_the_start_worker_across_calls` (+ `_batch_of`) was committed on p3a and on the combined branch as new commits; p3b was rebuilt on the new p3a HEAD with the combined tree. p3a HEAD 7f43e43e (2 commits), p3b HEAD e43e784c (1 commit on p3a), combined 2f7d9c9b. `git diff contrib/p3-v1-never-hang contrib/p3b-worker-relaunch` is empty. diff.patch / branch.txt regenerated; tests/trainer and tests/experimental re-run on both (logs/).

## Review round 2 (the AI review, 2026-10-29) - commit 951bc3f4 on 7f43e43e
- Finding 3 (no total recovery budget): added `trainer.v1.sampler.max_inflight_reissues_per_step` (null =
  `max_inflight_reissues` x the batch size of the sampling call), counted over one `sample()` /
  `wait_for_sampleable()` call via the call's own `inflight_reissued` counter; once spent the TTL is suspended
  for the rest of the call (one warning, `training/inflight_ttl_suspended=1`) and the loop waits as without a
  TTL. Expired prompts are re-issued oldest-first up to the remaining allowance. Chosen over a wall-clock
  budget because it is exact, cheap to test and composes with the per-prompt cap (the default is "the whole
  batch re-issued up to the per-prompt cap"). Tests: repeated-slow-replacement in sync and async through the
  real `_reissue_inflight_prompts` (3 re-issues == budget, 2 exhaustions, 2 refills, then the TTL waits and
  the step completes when the slow generation finishes), oldest-first cap, default/validation, summed counter.
- Docs and yaml updated; "never hang" wording narrowed to what is proven (bounded per prompt and per step; a
  TTL below the honest generation time degrades to waiting).
- Regression harness note: `scripts/run-cpu-regression.sh` runs against the clone; the worktree run uses
  `scripts/run-cpu-tests-wt.sh` (VERL_ROOT) with the same groups (wrapper kept in the session temporary directory).
- What the review found for [1/2]: no reproduced blocker; the per-prompt cap does not bound a step (finding 3), the
  evidence wording was stale (finding 4: pending-GPU phrases, "re-issue counts stayed zero on every completed run"
  is false for f6846af0, which re-issued 8), and the #8203 integration plan needed a refresh (finding 5).
- Not changed, and why: the exhausted-prompt path still fails + refills (not counted against the budget): the budget
  bounds regenerations the TTL starts; once it is spent nothing else fires, so the per-prompt cap cannot be hit any
  more in that step (uniform-slow case: 3 rounds of re-issue, then wait). The existing p3b test of the per-prompt cap
  therefore sets an explicit per-step budget. A wall-clock budget was considered and rejected (not exact, hard to test).
- Tests on 951bc3f4: `test_inflight_recovery_on_cpu.py` 21, aggregator 8; tests/trainer/ppo/v1 + metric_utils 295
  passed (tests.log); timing-based budget test run three times, stable. Pre-commit 14/14.
- Rules note: the full regression ran on the stacked p3b tree (contains this branch); counts in PR-BODY / CHECKLIST.

## Re-review of round 2 (2026-10-29) - commit 6445f21b
- A1/A2/A5 (one root cause): the budget filter ran on the whole expired set before the exhausted split and the
  draining branch, so an allowance of 0 made the TTL inert. Now `_recover_stuck_inflight` splits first and applies
  `_within_reissue_budget` to the re-issuable set only; exhausted prompts are failed and the drain discards
  regardless; `max_inflight_reissues=0` fails on the first expiry. A3: the suspension metric / warning fire only
  when a re-issuable prompt is held back. A4: the counter moved from the call's `eviction_metrics` onto the buffer,
  keyed by the `global_steps` of `sample()` / `wait_for_sampleable()` (`_begin_ttl_budget`), so a step's
  `parameter_sync_step` calls share it and the next step resets it; default allowance `max_inflight_reissues` x
  `train_batch_size` (the trainer now wires `train_batch_size` in every mode). Count line fixed (23 + 21 items).
- Consequence for the uniform-slow case: after the budget, the attempt-(cap+1) generations are failed and replaced
  (per-prompt cap), and the replacements' generations are waited for (budget). The slow-replacement test updated.
- Reviewer suite `packages/p3a/review-tests/round2`: 5/5 pass on 6445f21b (and on the p3b tree); author files
  23 + 21 items; tests/trainer/ppo/v1 + metric_utils 298 passed.

## Chaos re-run on the final [2/2] tree (2026-10-29)
- Bench agent: `chaos-p3b-final-1`/`-2` on 3f3d2c24 (contains 6445f21b) completed 2/2 (kill -> next step 2.6 s; recovery 6.0 + 10.0 s / 14.8 s; relaunched 1; orphaned 4 + 4 / 12 + 4; `inflight_reissued` 8 / 7; exhausted 0; no `inflight_ttl_suspended` key, i.e. the budget never engaged under the 60 s TTL); the 0d28f34f pair also 2/2 (superseded); stock GPU suite on 3f3d2c24 0 failed / 8 skipped + torchrun 2/2. Cumulative knobs-on 9/9, main 0/3, TTL-only 0/1 (expected). PR-BODY / CHECKLIST / REVIEW updated; package files only, no branch change. Status: MERGE-READY.

## Round 3 (the independent follow-up, R2-02) - commit d834b6b3
- Livelock with `max_inflight_reissues=0` in refill modes: every first attempt exhausted on its first expiry, the
  fresh-uid failure bypassed the budget, the refill started at attempt 1 and the cycle never ended (34 dispatches /
  31 completions in 20 simulated seconds on the real loops). Fix: every TTL-induced discard, a re-issue or the
  failure that ends an exhausted prompt, is charged against the step budget; once spent the TTL retires nothing
  more in the step (the existing degradation) and the pending healthy work completes. Default allowance now
  `(cap + 1)` x `train_batch_size` so a zero cap still fails an expired prompt once per prompt. The drain's discard
  stays exempt (it never regenerates). Wording: "TTL-induced discards per step are bounded by the budget; after
  that the sampler waits"; the explicit wait-vs-abort policy is a documented follow-up, not implemented.
- The reviewer's full-loop test is on the branch as `test_ttl_budget_full_loop_on_cpu.py` (body unchanged, header
  and docstring added); 4/4 pass on p3a and on p3b. Round-2 reviewer tests A2 and A3 assert the superseded
  semantics (exhaustion outside the budget / no suspension for held-back exhausted prompts) and fail by design;
  A1, A4, A5 pass. Author files 23 + 4 + 21 items; tests/trainer/ppo/v1 + metric_utils in tests.log.

## Round 3b (re-review A1 / A3) - commit e598929b
- A1: in padding mode (sync without `sync_refill_failed_groups`) the charged budget failed every healthy-but-slow
  prompt once and `sample` ended in the all-failed RuntimeError. The sync sampler now passes
  `keep_last_finishable`: an exhaustion failure that would leave no finished group and nothing else in flight is
  held back (`_hold_back_last_groups`: flag, one warning, the step's budget marked spent) and waited for. The
  direct recovery call keeps the raw semantics, so the reviewer's round-2 direct-call tests (single exhausted
  prompt failed) still hold; refill modes are unchanged. Not applied to dead-owned prompts (p3b).
- A3: `max_inflight_reissues_per_step` must be a positive integer or null (0 rejected; `inflight_ttl_s=null`
  disables the TTL); `training/inflight_ttl_suspended` reduced as a max (a per-step 0/1 flag).
- Reviewer suites on e598929b: round2 5/5; round3 A1 x2 + A2 pass, A3 fails by design (it asserts that 0 is
  accepted). Author files 24 + 4 + 21 items; tests/trainer/ppo/v1 + metric_utils 303 passed (tests.log).

## Round 3b closure (H1) - commit 2bd5c040
- H1: the hold-back marked the step budget spent, so a dead writer or poison prompt beside a healthy slow group
  could never be failed once the healthy group finished (padding mode hung). The hold-back now charges nothing
  and is re-evaluated every poll (flag, one warning per step): once another group is finished or in flight the
  exhausted prompt is failed under the normal budget and the step completes padded.
- Reviewer round-3 dir on this head: A1 x2, A2, A3 (rewritten), H1 x2, H3 pass; H2's third assertion
  (`_ttl_discards_in_step == budget`, "the hold-back marks the step budget spent") encodes the spend that H1
  removes and fails by design. Round 2 5/5. Author files 25 + 4 + 21 items.

## Final closure (2026-10-29): MERGE-READY, chaos proof on the final sha
- `chaos-p3b-final3-1` on 1b397dfd completed 6/6 (kill -> next already-generated step 2.3 s; recovery 6.2 + 11.5 s in steps 5-6; relaunched 1; orphaned 4 + 4; `inflight_reissued` 7 = dead-worker; exhausted 0; no TTL-budget, hold-back or death-event key emitted); 45690b0f and 64212367 also 6/6; cumulative knobs-on 12/12. Stock suite on 1b397dfd (8x H200) 0 failed / 8 skipped + torchrun 2/2. Package files only, no branch change.
