# P0 review round 2 — AI re-review verdict (2026-10-29)

Branch `contrib/p0-loss-horizon` @ e0a7eb22 (6 commits on main @ 2ab35292), worktree wt/p0, diff base main. Reviewer-owned
files only: `packages/p0/review-tests/round2/` and this file. Branch, clone and the other package files untouched.

## the submitter's findings (research/review-bundle/REVIEW.md "P0 finite normalizer validation", "P0 launcher scope")

| finding | status | evidence |
|---|---|---|
| N1 explicit factor accepts NaN / +inf | FIXED | The submitter's cases re-run with PYTHONPATH at each tree (review-tests/round2/logs/repro_reviewer_p0.out): pre-fix 0dab0eb5 `explicit nan ACCEPTED actor.loss_scale_factor=nan`, `explicit +inf ACCEPTED ... inf`; e0a7eb22 `REJECTED ValueError: actor_rollout_ref.actor.loss_scale_factor must be a finite positive number ... got nan` / `got inf`. Also rejected now: -inf, -0.0, 0.0, 1e400 (=inf), 1e-400 (=0.0), false, [16], '16', an `oc.env` string (`got '2048'`). |
| N2 inferred horizon accepts 0 / negative | FIXED | pre-fix: the actor silently got 0 / -16 and only the critic's inherited copy tripped, naming the wrong key (`critic.loss_scale_factor ... got 0`); e0a7eb22: `actor_rollout_ref.rollout.response_length must be a positive integer ... got 0` / `got -16`, also for 2.5, true, '2048', an env string, `data.max_response_length=0` and 2048.0 (integral float rejected by design, stated in NOTES.md; `${oc.decode:${oc.env:X}}` is accepted as an int). |
| N3 focused NaN / inf / zero / negative tests | FIXED | branch file re-run: 59 passed / 2 xfailed (matches tests.log, CHECKLIST.md); round-1 review tests: 21 passed (matches tests.log s3). |
| N4 keep the launcher-scope qualification | FIXED | kept in the docstring, actor.yaml / critic.yaml, docs/algo/grpo.md and PR-BODY "Scope and limitations". Verified: `main_ppo.main` validates before the V1/V0 split (line 181); `main_ppo_v0.TaskRunner.run` and `one_step_off_policy/main_ppo.py` resolve-then-validate (both orders tested, same result); `fully_async_policy` bypasses it but updates through `RayPPOTrainer._update_actor` -> `left_right_2_no_padding` (`max_response_len` carried), as the PR body says. One wording nit: D4. |
| N5 keep the Dr. GRPO breaking-behaviour explanation | FIXED | PR-BODY "What changes for users" (smaller losses / gradients by `(micro-batch max width) / response_length`, set the factor to keep the scale); `[BREAKING]` title unchanged. |

## New defects

- LOW D1 (code): `_is_finite_positive_number` skips the finiteness check for ints, so `loss_scale_factor=10**400` passes
  `validate_config` and the real `ppo_loss` raises `OverflowError: int too big to convert` on the first micro-batch; the helper
  moved the OverflowError from the validator to the loss rather than removing it. Unreachable by a sane config; a `float(value)`
  in a try/except (or `value <= sys.float_info.max`) closes it. Recorded as a strict xfail plus a test of the actual behaviour.
- LOW D2 (package notes): REVIEW.md §10 and NOTES.md justify keeping float factors with "the dataclass field is typed
  Optional[float]"; it is `Optional[int]` (verl/workers/config/actor.py:164, critic.py:94). The conclusion holds (hydra-instantiate
  does not type-check; 0.5 is consumed by `ppo_loss` / `value_loss`, proven at the loss level in round2), but fix the sentence.
- LOW D3 (package notes): "the unset-response_length path stays a warning (the submitter: keep it)" — the submitter's bundle (REVIEW.md,
  FIX-CHECKLIST.md) says nothing about that path. The decision is defensible (nothing is assigned); do not attribute it to the submitter.
- LOW D4 (PR-BODY wording): "the recipe runners that skip `validate_config` are legacy dense-layout trainers that carry
  `max_response_len` and never had the defect" is unverifiable from the package: the recipe submodule is not checked out, and a
  read-only gh code search finds `validate_config` in 19 verl-recipe files (dapo, prime, spin, sppo, partial_rollout, spo, gvpo,
  ...) and `seq-mean-token-sum-norm` in partial_rollout run scripts and gvpo (own actor config / core_algos). Hedge it to "not
  examined" or verify per runner; this is exactly the over-claim the submitter's N4 warns against.
- No HIGH or MEDIUM. Probed without a defect: critic keyed on its own (error names `critic.loss_scale_factor`; the actor is named
  first when both are invalid; an unused critic is not validated), `ppo_megatron_trainer` (resolves and rejects), `trainer.use_v1=false`,
  a second validation of the resolved int, `${data.max_response_length}` as the factor (accepted as the int), numpy / torch scalars
  (OmegaConf refuses them: unreachable), negative zero, denormal and 1e308 floats (accepted, legal), messages name key and value,
  0.5 and the resolved 16 drive `ppo_loss` / `value_loss` across an uneven split exactly as `seq-mean-token-sum / factor`.

## Tests added (packages/p0/review-tests/round2/; run from wt/p0 with the rules-file CPU harness)

`test_round2_reviewer_cases_on_cpu.py` (29 cases), `test_round2_spellings_and_loss_on_cpu.py` (13 cases, 1 strict xfail = D1),
`round2_common.py`, `conftest.py` (stub config.json, no model snapshot). Result: 41 passed, 1 xfailed in 7.1 s
(logs/round2-e0a7eb22.log). Branch file: 59 passed / 2 xfailed (logs/branch-file-e0a7eb22.log). Round-1 review tests: 21 passed
(logs/round1-review-tests-e0a7eb22.log). Reproduction and probe outputs: logs/repro_reviewer_p0.out, logs/probe2.out.

## Package hygiene

`scripts/package-audit.sh`: `p0 contrib/p0-loss-horizon e0a7eb22 6 0 fresh ok none`; main clone clean, no contrib remote.
tests.log counts (59/2 on the branch, 27 failed on main, 21 review, 1543 passed / 64 skipped / 2 xfailed / 1 deselected / 0 failed)
match the claims; PR-BODY carries the exact AI statement, the stock-GPU-suite wording with the 503 / 0 / 8 label, no "already
reviewed", no claimed upstream contact (suggested reviewers only), conservative numbers; CHECKLIST.md agrees with tests.log;
pre-commit log present (14 hooks passed); worktree clean before and after my runs. Pre-commit not re-run by me (auto-fixing hooks).

## Call

MERGE-READY once D4's sentence is hedged (text only; D2 / D3 are package-note fixes, D1 optional): both round-2 findings are
closed with the right key in every message, the scope qualification and the breaking note are kept, and nothing above LOW was found.

## Lead closure (2026-10-29, after 9c99205c)
D1-D4 applied by the author in 9c99205c (int factors bounded to 1..2**63-1 because torch wraps a Python int as
int64, so 2**63 overflowed too; notes and the PR-body sentence corrected). The two reviewer tests that encoded D1 as
a defect were retired to assert the rejection (review-tests/round2/test_round2_spellings_and_loss_on_cpu.py);
the round-2 review suite is 42 passed against wt/p0 @ 9c99205c. Verdict stands: MERGE-READY.
