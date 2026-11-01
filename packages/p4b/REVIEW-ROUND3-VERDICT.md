# P4b review round 3 — AI re-review verdict (the independent follow-up R2-04 + the P4 benchmark section)

Reviewer: Claude Fable 5.1 (AI), adversarial, read-only on the branch, 2026-10-29. Tree: `contrib/p4b-turn-rloo` @ e1052882
(5 on `contrib/p4a-turn-signals` @ 0d4cd8ca; `git rebase --onto 0d4cd8ca d8c4b925`, no p4b code change: the tree over p4a is
the same 11-file / +864 / -16 footprint as before). Package audit line: `p4b   contrib/p4b-turn-rloo   e1052882  15   0   fresh  ok   none`.

## the submitter's probe on this tree

`wt/p4b @ e1052882: 7.0 [{'num_turns': 7, 'tool_rewards': [123.0]}]`, identical to `verl @ 2ab35292` (7.0): the [1/2] fix
0d4cd8ca is carried by the rebase; `turn_rloo`, the span validation and the gated plumbing are untouched by it.

## Findings

| Finding | Status | Evidence |
|---|---|---|
| R2-04 (the helper lives in [1/2]) | FIXED upstream, carried | the p4a round-3 suite (r3a differential vs main, r3b flag matrix / reader / rate-limited batch path) runs on this tree: 30 passed; the p4a round-2 suite with the per-manager contract: 82 passed; the author's P4 files 106 passed (tests.log). |
| Nothing in the estimator path depends on the reward manager's flag read | VERIFIED | the V1 lift and V0 forwarding key on `adv_estimator_takes_turn_spans` and the exported `generated_turn_spans` only (r2e / r2f unchanged: 50 passed on e1052882). |
| Benchmark: repeated-reward farming; "same curve within noise"; estimator not isolated | DISCLOSED; v2 pending | PR-BODY.md: the 30-step curves quoted per step with "no demonstrated accuracy gain", the repeated identical answers (923 / 1085 of 1319), "`data.seed` the only difference between them"; no "same curve" / "within noise" wording left (grep clean); `packages/p4/bench/ENABLED-RUN.md` v2 design adds arm B (dense + `reinforce_plus_plus`) against C (dense + `turn_rloo`) so the estimator is isolated; numbers pending (bench agent). |
| Hygiene | DONE | diff.patch fresh, branch.txt names e1052882 and the rebase, worktree clean; tests.log @ e1052882: `tests/trainer` 477 passed / 1 skipped, `tests/experimental` 110 passed, the P4 files 106 passed; CHECKLIST row for the reviewer suites updated. |

## New defects

None on this branch. The two LOW items of the round (string flag value read with `bool()`; the "contract unchanged"
wording's two disclosed deviations) belong to [1/2] and are recorded in `packages/p4a/REVIEW-ROUND3-VERDICT.md`.

## Tests

- No p4b-specific round-3 tests were needed (no p4b code changed); `packages/p4b/review-tests/round3/` was therefore not
  created. The p4a round-3 suite (30) and both round-2 suites (p4a 82, p4b 50) pass on wt/p4b @ e1052882.

## Merge-readiness: MERGE-READY (pending the v2 enabled run the package labels pending)

The rebase carries the R2-04 fix without touching the estimator or the plumbing; every round-2 and round-3 suite passes on
this tree; the benchmark claim is now a disclosure, not a performance claim.

## Lead closure (2026-10-29, p4a 0d4cd8ca / p4b e1052882)
R2-04 FIXED by measurement against main (the submitter's probe: 7.0 on main and on the branch; the five managers' flag-off
contract identical key by key). The two LOW notes are accepted without a further commit: a string "false" for the
flag in a hand-built config reads as on (the composed config carries a real bool; same truthiness as the loop), and
the "contract unchanged" sentence is exact for mapping extra_info with the non-mapping pass-through and the copy as
the two disclosed deviations. Verdict: MERGE-READY pending the v2 enabled runs (p4-v2-{A,B,C}-s{1,2} on e1052882).
