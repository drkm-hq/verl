# P4b build notes [2/2]: `[algo, trainer] feat: turn_rloo advantage estimator with gated turn-span plumbing`

Branch `contrib/p4b-turn-rloo`, worktree `<repo>/wt/p4b`, base `contrib/p4a-turn-signals`
@ f4651857 (on main @ 2ab35292). Its tree is identical to the combined reference branch `contrib/p4-turn-rewards`
(`git diff contrib/p4-turn-rewards contrib/p4b-turn-rloo` is empty). Design log: packages/p4/NOTES.md; review-fix
mapping: packages/p4/REVIEW.md "Fixes applied".

## What is in this half

- `verl/trainer/ppo/core_algos.py`: `compute_turn_rloo_advantage` registered as `"turn_rloo"`; spans starting past the
  row end are not turns the sample reached (review D5).
- `verl/trainer/ppo/ray_trainer.py`: `compute_advantage` forwards `turn_spans` only when
  `"turn_spans" in inspect.signature(adv_estimator_fn).parameters` (review D1).
- `verl/trainer/ppo/v1/trainer_base.py`: `_compute_advantage` selects `extra_fields`, normalises it with
  `list(...)` + `tu.unwrap_non_tensor_data`, lifts `generated_turn_spans` into `non_tensor_batch`, tolerates non-dict
  rows (review D4).
- `verl/trainer/config/rollout/rollout.yaml`: the flag's comment names `turn_rloo` among the token-level estimators.
- Tests: `test_turn_rloo_on_cpu.py` (bias study, equalities, D1 gate, D5), `v1/test_compute_advantage_turn_spans_on_cpu.py`
  (lift, no spans, non-dict row), the `turn_rloo` group test added to `test_turn_level_rewards_on_cpu.py`,
  `test_core_algos_on_cpu.py` `TestRegisterAdvEst` save/restore (review D8; the branch's earlier workaround is gone).
- Docs: estimator paragraph + yaml example in `docs/advance/agent_loop.rst` (section title gains `turn_rloo`),
  `docs/algo/ppo.md`, `docs/examples/config.rst`.

## How the branch was built

Worktree on `contrib/p4a-turn-signals`, the remaining files checked out from the fixed combined branch, rebased onto p4a's
R-2 fix: c02a5637 (estimator + gated plumbing + tests + the registry test fix), bd50cf15 (docs), 97a43c92
(re-verification residual R-1: the docs paragraph and the estimator docstring state that without
`rollout.multi_turn.turn_level_rewards` no spans reach the trainer, so `turn_rloo` is plain RLOO on multi-turn data too).

## Verification on this branch

See tests.log: the new tests, all reviewer tests R1-R6 (R3's BRANCH path pointed at this worktree), the trainer /
experimental / tools / utils groups, and one `pre-commit run --all-files` on this tree. Expected-by-wording failures in
the reviewer's suite: R2(b)/(c) (a dropped reward must "survive"; the accepted fix warns) and R3's two metadata
assertions (pre-fix always-export behaviour). Everything else, including R1 (the D1 crash), R4(d) (D5), R5(f) (D3) and
R6(c) (D4), passes.

## Review round 2 (the submitter, 2026-10-29)

What the review found for [2/2] (research/review-bundle/REVIEW.md "P4 mathematical and API qualifications" and the P4
items of FIX-CHECKLIST.md): (2) `_compute_advantage` fetched and unwrapped the whole `extra_fields` column for every
estimator; (3) externally supplied spans were clipped / dropped rather than validated, so malformed spans could
silently distort the grouping; (5) the unbiasedness claim needed its conditions (independent trajectories,
`gamma=1`, `norm_adv_by_std=False`), `gamma<1` needed an explicit objective or a heuristic label, "exactly RLOO"
had to become "the same policy advantages as RLOO" with the `returns` difference documented and tested; (6)
PR-body wording; (7) the 503-pass GPU suite does not exercise the new tests.

Rebase: `git rebase --onto a600e810 f4651857 contrib/p4b-turn-rloo` inside wt/p4b; one conflict in
`docs/advance/agent_loop.rst` (p4a's response-budget paragraph and p4b's estimator paragraph both inserted before
"A minimal device tool"), resolved by keeping both in that order. The three original commits became 3f5cc642,
0a876116, 94480666.

What changed (two commits, HEAD 64f8f604):
- 28ee7c4a `[algo, trainer] fix: validate turn spans in turn_rloo; fetch extra_fields only for estimators that take
  them`: `_validated_turn_spans(row, spans, length)` enforces `prev_end <= start < end <= length` (and int pairs) with
  a `ValueError` naming the row; the row count must match the batch. The round-1 D5 tolerance ("a span starting past
  the row end is not reached") is gone: the loop clips its own spans at the response budget before export, so a span
  past the row end can only come from an external producer, which the review asked to reject. `core_algos.
  adv_estimator_takes_turn_spans(name | fn)` is the one signature check (follows `__wrapped__` as before); V0
  `compute_advantage` uses it, V1 `_compute_advantage` selects `extra_fields` and lifts the spans only when it is
  true and logs `adv_estimator/rows_with_turn_spans` (the enabled GPU run's proof that the lift happened; absent on
  the control arm). Tests: 8 malformed shapes + row count, the helper (names, `**kwargs`, `functools.wraps`), the V1
  no-fetch path for `rloo` / `reinforce_plus_plus` / `grpo` (the fake queue now returns only the selected fields).
- 64f8f604 `[doc, algo] docs: qualify turn_rloo's unbiasedness and its returns tensor`: docstring + docs paragraph
  "Conditions of the unbiasedness argument"; "equals rloo" -> "the same policy advantages as rloo, returns differ".
  Tests: one-token rewards `[1, 3]` (`rloo` returns `[-2, 2]`, `turn_rloo` returns `[1, 3]`, advantages equal);
  the `norm_adv_by_std` enumeration (27 joint outcomes; reproduces the reviewer's `[-0.46,-0.39,0.85]` exactly and
  `[-0.2453,-0.1847,0.4300]` within 2e-4; ratios not a common factor); the `gamma=0.5` two-Bernoulli-turn enumeration
  (`[0.25,0.25]` for both gamma 1 and 0.5, against the objective gradient `[0.25,0.125]`).

Decided not to change: `gamma<1` semantics (kept as the discounted return-to-go without the outer `gamma^t`, the
`reinforce_plus_plus` convention, now labelled a heuristic rather than redefined); `norm_adv_by_std` kept as an
option, labelled a variance-reduction heuristic; the `returns` tensor keeps the raw return-to-go (the reviewer's
"keep raw returns if intended"), documented as a metrics-only difference. The reviewer's R4(d) / R8 tests (past-end
tolerance) fail on this tree by design and are listed as superseded in tests.log.

Verification: packages/p4b/tests.log (the six CPU regression groups via scripts/run-cpu-tests-wt.sh with
VERL_ROOT=wt/p4b, the P4 files, the reviewer's suite 77 passed / 19 superseded); pre-commit 14/14 on each commit's
files; worktree clean.

### Re-review (lead, 2026-10-29): LOW 3-5 closed by 988a10bc (after a clean rebase onto p4a d8c4b925)

- LOW 3: `_span_offset` uses `operator.index` (Python / numpy / 0-d torch integers) and rejects Python, numpy and torch
  bools by name (`operator.index(True)` is 1, so bools need the explicit check); floats, strings, None fail
  `operator.index` and are named by type. LOW 4: `list(turn_spans)` with a `TypeError` -> `ValueError("turn_spans must
  hold one entry per row (N rows), got a <type>")`. LOW 5: the rule chosen is the outermost callable's own signature
  (`inspect.signature(fn, follow_wrapped=False)`): explicit `turn_spans` opts in; `**kwargs` alone does not (so the
  built-in estimators, which all take `**kwargs`, stay off the V1 fetch); a wrapper with `__wrapped__` opts in only
  when it takes `**kwargs` and the followed chain declares `turn_spans`. The lead's alternative ("`**kwargs`
  as accepting") was not taken: it would put every built-in estimator on the `extra_fields` fetch again.
- Acceptance: reviewer round-2 suites on this tree: r2a/b/c 46 passed, r2d/e/f 41 passed; the 7 failures pin the
  pre-fix behaviour of LOW 1-5 and fail by design (review-tests not edited). Author files 95 passed; `tests/trainer`
  and `tests/experimental` groups green; pre-commit 14/14.

### GPU evidence landed (bench agent, 2026-10-29; package files only, branch unchanged)

On this branch @ 988a10bc (packages/p4/bench/ENABLED-RUN.md, bench/README.md "Review round 2"): the 6-step ENABLED vs CONTROL
pair proves every stage (metadata_present = spans_ok = 1.0 vs 0.0, rows_with_turn_spans 512 on 6/6 steps, returns/mean
0.95-1.13 vs ~0, score 1.16-1.34, acc 0.599/0.633/0.694 vs 0.608/0.627/0.684, 17.4 vs 17.8 s per step); the 30-step two-seed
curves give the same accuracy within +-2 points (pooled 0.818 vs 0.828 at step 30), with ENABLED making more tool calls
(turn_count 1.44 -> 2.6-2.7) at 10-28 % longer steps. PR-BODY.md states plainly that there is no accuracy gain over GRPO
outcome on GSM8K with a correctness step reward (expected for a single-answer task) and makes no algorithm-performance claim.
Stock suite on 988a10bc: 503 / 0 / 8 + torchrun 2/2. No docs claim in the branch contradicts these facts, so nothing changed.

### Round 3 (the AI follow-up review, 2026-10-29): bench deliverables delivered first (see packages/p4a/NOTES.md)

`packages/p4/bench/gsm8k_tool_reward.py` v2 (last-submitted-answer outcome, identical for all arms, `done` scored the same
way, repeated-submission diagnostics) and `packages/p4/bench/ENABLED-RUN.md` "v2 design" (arms A / B / C, overrides, stage
metrics, reporting rule); `packages/p4/bench/RESULTS.md` carries the round-2 disclosure. This branch has no code change in
round 3; it is rebased onto p4a's round-3 head.

### Round 3: no code change here; rebased onto p4a 0d4cd8ca (HEAD e1052882)

Acceptance on this tree: the P4 files 106 passed (incl. p4a's probe test), `tests/trainer` 477 / 1 skipped,
`tests/experimental` 110; the reviewer's round-2 suites r2d/e/f 50 passed (the reviewer updated them after LOW 3-5: the
former defect-pinning cases now assert the fixed behaviour), r2a/b/c 37 passed + 15 r2a cases failing by design (the
pre-R2-04 unconditional-merge contract). PR-BODY.md carries the experiment disclosure; ENABLED-RUN.md "v2 design" names
this branch's head for the v2 arms.

### Round 3 closed: v2 arms ran on e1052882 (bench agent; package files only)

Pooled val acc at steps 0..30: A 0.668 0.731 0.791 0.811 0.815 0.825 0.832, B 0.663 0.752 0.795 0.806 0.812 0.818 0.822,
C 0.662 0.757 0.789 0.802 0.812 0.828 0.829 (step 30 per seed A 0.835/0.830, B 0.823/0.821, C 0.841/0.818); repeated
submissions <= 1.4 % (v1: 70-82 %), no episode with >= 2 step points; episode_done = turn_rewards_sum 0.82-0.84 (done on
GPU); rows_with_turn_spans 512 x 30 on both C seeds; returns/mean 1.39-1.42 (B/C) vs -0.05 (A). Reading written into
PR-BODY.md: no demonstrated accuracy gain for either estimator, turn_rloo indistinguishable from reinforce_plus_plus on
the same dense reward, no algorithm-performance claim (GSM8K is a single-answer task). Status: MERGE-READY.
