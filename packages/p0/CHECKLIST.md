# P0 checklist (verl PR template items with evidence; updated after review round 2 and its re-review D1-D4, 2026-10-29, HEAD 9c99205c)

## Checklist Before Starting
- [x] Similar PRs searched: round-1 vet K12 s3 (dupcheck.py + gh reads) and round-2 read-only gh searches on 2026-10-29
      13:08Z (`gh pr list` / `gh issue list --search` for loss_scale_factor, seq-mean-token-sum-norm, max_response_len):
      no open or merged PR or issue fixes the V1 horizon. New hits named in PR-BODY.md: #6157 (closed 2026-04-27,
      unmerged, agg_loss-level tests for this mode; a comment today bumped it) and #4294 (open, legacy dp_actor /
      dp_critic metric aggregation). Prior art: #2435 (stale), #4721 (stale, tests only), #3741 (open issue), #6957,
      #5417, #6823. Not a duplicate of #8176 / #6349 (different lines in losses.py / agg_loss; neither file touched).
- [x] Title `[BREAKING][algo, trainer] fix: resolve the seq-mean-token-sum-norm horizon to response_length under the
      V1 trainer` passes tests/special_sanity/check_pr_title.py (round 1). `[BREAKING]` kept (the submitter: keep the
      breaking-behaviour explanation for affected Dr. GRPO jobs).

## Test
- [x] Failing-then-passing regression test: tests.log s1-s2 (main: 29 failed; branch: 62 passed, 2 xfailed).
- [x] Round-2 cases: explicit nan / inf / -inf / 0 / -1 / "sixteen" / True rejected naming the key and the value;
      inferred 0 / -16 / 2.5 / True rejected naming actor_rollout_ref.rollout.response_length; 4096 / 0.5 kept;
      a directly overridden response_length resolved; an unset response_length warns (tests.log s1).
- [x] Re-review D1 (9c99205c): explicit 2**63 and 10**400 rejected (torch overflows on any int scalar >= 2**63, so the
      bound is int64, not the float range); 2**63-1 kept (tests.log s1).
- [x] Reviewers' tests on 9c99205c: round 1 21 passed; round 2 40 passed + the reviewer's two D1 encodings now failing
      by design (strict xfail XPASSes; the "passes validation then overflows" test gets the ValueError), reviewer-owned,
      left for the lead to retire (tests.log s3).
- [x] Full CPU regression, six groups (small, utils, trainer, workers, experimental, single_controller): 1546 passed, 64 skipped, 2 xfailed, 1 deselected, 0 failed (small 200, utils 473, trainer 500, workers 278, experimental 74, single_controller 21)
      (tests.log s4) = the 2026-10-28 baseline 1484 passed + 62 new, 0 failed.
- [x] GPU: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`, so it does not execute this PR's new tests) on 8xH200 at the branch head 9c99205c: 503 passed / 0 failed / 8 skipped + torchrun 2/2, identical to
      main (run by the lead: bench/results/gpu-tests-p0-r2/ command.txt, pytest.log/.xml, harness.log, digest
      dfab34c8). The earlier run at 6684917e (bench/RESULTS.md) gave the same counts.
- [ ] GPU e2e A/B (Dr. GRPO, main vs branch): not run; suggested in PR-BODY.md "Test".

## API and Usage Example
- [x] No API change; usage + the INFO log line in PR-BODY.md.

## Design & Code Changes
- [x] 7 commits on contrib/p0-loss-horizon (branch.txt, diff.patch): fix + yaml comments + docs; tests; review
      follow-ups; test follow-ups; residual nits; round 2 e0a7eb22 (finite positive explicit factor, positive-integer
      inferred horizon, 11 new rejection cases + 3 new positive cases); re-review 9c99205c (int factors bounded to
      int64, D1; +2 rejection cases, +1 kept case). Default path (token-mean) untouched; an
      explicit finite positive loss_scale_factor untouched; legacy trainer identical; launcher boundary documented,
      not widened.

## Checklist Before Submitting
- [x] Contribute Guide + AGENTS.md read: AI statement ("Written with Claude (Anthropic) under the author's direction.
      Commits carry a Co-authored-by trailer."), non-duplication rationale and test commands are in PR-BODY.md; all
      seven commits carry `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>`.
- [x] pre-commit: after 9c99205c on its 2 files and after e0a7eb22 on its 5 files: exit 0, all 14 hooks Passed, no file
      modified (logs/pre-commit-files-after-d1.log, logs/pre-commit-files-e0a7eb22.log); `pre-commit run --all-files` run once at 6684917e
      (logs/pre-commit-all-files.log).
- [x] Docs: docs/algo/grpo.md (finite number > 0, positive-integer response_length, Last updated 10/29/2026),
      actor.yaml / critic.yaml comments. `_generated_*.yaml` unchanged (autogen hook passed).
- [x] Tests in CI: the file matches `tests/**/test_*_on_cpu.py` and runs under cpu_unit_tests.yml; no model snapshot
      needed (stub config.json, tokenizer loading off); xfails non-strict so other PRs' merges cannot fail it.
- [ ] ci-request Slack message: the submitter, after review.
- [ ] recipe submodule: not touched.

## Not done / owner decisions
- Nothing pushed, no remote added, no PR or comment created. The main clone is untouched and on main.
- REVIEW.md: s1-s9 round 1 (lead); s10 round 2 (the submitter's findings N1-N6 -> e0a7eb22; re-review D1 -> 9c99205c,
  D2-D4 -> package notes / PR body). REVIEW-ROUND2-VERDICT.md and review-tests/round2/ are reviewer-owned.
- bench/: not applicable (correctness fix, no timing claim).
- Open for the submitter: (1) an integral float response_length (2048.0) is rejected like 2.5 (strict int); one line to relax
  to `float.is_integer()` if preferred. (2) keep `[BREAKING]` (recommended by both reviews) or drop it.
  (3) `critic.yaml:63` mode-list comment left as is (drive-by avoided).
