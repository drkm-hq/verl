# P4b review round 2 — AI re-review verdict

Reviewer: Claude Fable 5.1 (AI), adversarial, read-only on the branch, 2026-10-29. Tree: `contrib/p4b-turn-rloo` @ 64f8f604
(5 commits on `contrib/p4a-turn-signals` @ a600e810, main @ 2ab35292), worktree `wt/p4b`. Round-2 commits attacked:
28ee7c4a, 64f8f604. Package audit line: `p4b   contrib/p4b-turn-rloo   64f8f604  12   0   fresh  ok   none`.

## the submitter's findings

| Finding | Status | Evidence |
|---|---|---|
| 2. Disabled-path cost: `extra_fields` fetched / unwrapped for every estimator | FIXED | `trainer_base.py:1749-1769`: `extra_fields` selected and lifted only when `core_algos.adv_estimator_takes_turn_spans(adv_estimator)`; V0 uses the same helper. My r2e: a registered `**kwargs`-only estimator never selects `extra_fields` and logs no metric; one declaring `turn_spans` gets the column (ragged: 1 of 3 rows) and `adv_estimator/rows_with_turn_spans` = `int` 1; `turn_rloo` with no span rows logs 0 and equals `rloo`; the helper answers correctly for every registry name (only `turn_rloo` opts in), enums, `functools.partial`, callable classes, a ufunc. |
| 3. External spans not validated | FIXED | `_validated_turn_spans` (`core_algos.py:652-676`) + the row-count check. My r2d: 12 malformed row shapes (incl. end past the row, overlap, reversed, empty, negative, triple, None inside, scalar, str / dict rows, non-numeric strings) raise `ValueError` naming the row with the bounds; 11 equivalent spellings (lists, numpy / torch ints, 2-D rows, a 3-D int column, a plain-list column, whole-row span, one-token span at the row end, None / [] / () ragged rows) give the canonical advantages. |
| 3b. The strict change: is dropping the past-end tolerance safe for a legitimately truncated trajectory? | SAFE, by measurement | r2f drives the REAL `ToolAgentLoop` at every `response_length` 1..23 x 5 junction edits (0-2 trimmed, 0-2 inserted) through V0 `_postprocess` -> `compute_advantage("turn_rloo")` (rows padded to `response_length`) and through the V1 `extra_fields` lift -> `turn_rloo` on `as_dict` rows, plus real Qwen2.5-0.5B trajectories at every budget 1..92 in three paddings: no `ValueError`, finite advantages, 0 on `response_mask == 0`, `G[i,0]` = the row's reward sum. The loop clips at `len(response_ids)` (`tool_agent_loop.py:251-257`) and nothing downstream re-truncates (`as_dict`, `_pad_token_ids`, `agent_loop_tq.py:199`). The 19 "superseded" round-1 failures are exactly R4(d) + R8's past-end cases (tests.log list). |
| 5. Unbiasedness conditions, `gamma<1`, "exactly RLOO" | DOCUMENTED + TESTED | Docstring and `agent_loop.rst:299-310` state independent trajectories / `gamma=1` / `norm_adv_by_std=False`, label std-normalisation and discounting heuristics, V1 multi-output correlation; "the same policy advantages as rloo, returns differ" in docs, docstring, PR body; no "exactly RLOO" / "equals rloo" left (grep over docs/ verl/ tests/). The three enumeration tests assert the submitter's numbers: `[-0.46,-0.39,0.85]` (atol 1e-6), `[-0.2453,-0.1847,0.4300]` (atol 2e-4), not a common rescaling (ratio spread > 1.05); `gamma=0.5` -> `[0.25,0.25]` and `!= [0.25,0.125]`; one-token `[1,3]`: advantages equal, returns `[1,3]` vs `[-2,2]`. |
| 7. 503-pass GPU suite does not exercise the new tests | WORDING DONE, enabled run PENDING | PR-BODY.md carries the rules' sentence verbatim; the enabled evidence is listed separately as pending; `ENABLED-RUN.md` names `adv_estimator/rows_with_turn_spans` (= the code's metric name) as the lift's proof. |
| 6. PR body wording | DONE | AI statement verbatim; co-authorship "offered here; no prior contact"; no "already reviewed"; #8305 facts match read-only `gh pr view`: draft, luzy99, OPEN, updated 2026-10-21T03:43Z, +964/-3, 19 files. |

## New defects (none HIGH / MEDIUM)

- LOW `_validated_turn_spans` truncates floats and bools with `int()`: `[(0.9, 2.1)]` -> `(0, 2)`, `(False, True)` -> `(0, 1)`;
  the "(int, int) pair" message never fires for them, so a producer emitting float offsets gets a shifted grouping, not an
  error (r2d `test_float_and_bool_spans_are_silently_truncated_to_ints`). Fix: reject non-integral values.
- LOW a non-sized `turn_spans` (a scalar) fails at `len(rows)` with `TypeError`, not the named `ValueError` (r2d).
- LOW the documented `__wrapped__` limit has a second face: a `functools.wraps` wrapper whose own signature cannot take
  `turn_spans` makes the probe say yes and the call raise `TypeError` (r2e). Pathological; worth one sentence in the docs.
- LOW (benign) V1 now resolves the estimator in the gate before the queue fetch, so an unknown name fails earlier with the
  same `ValueError` (r2e).

## Tests added (packages/p4b/review-tests/round2/; CPU harness from the rules file, `PYTHONPATH=wt/p4b:<this dir>`)

- `test_r2d_span_validation_shapes_on_cpu.py` (24) and `test_r2e_gating_on_cpu.py` (9): 33 passed.
- `test_r2f_strict_validation_vs_real_loop_on_cpu.py` (11, imports `packages/p4/review-tests/scripted_loop.py` and the p4a
  round-2 Qwen harness): 11 passed; the Qwen case skips cleanly without `~/models/Qwen/Qwen2.5-0.5B`.
- Author's own files re-run by me on this tree: 48 passed (p4a set), tokenizer test 2 skipped with HOME redirected; the
  `tests/trainer` and `tests/experimental` regression groups re-run: 470 passed / 1 skipped and 96 passed (= tests.log).

## Hygiene

diff.patch fresh, branch.txt names HEAD, worktree clean before and after my runs, tests.log counts match the PR body (20 + 6
new tests, 85 P4 files, 1538 / 64 / 0 regression), CHECKLIST.md truthful (strict validation and the `[BREAKING]` reasoning
stated). Docs lists in `ppo.md` / `config.rst` carry `turn_rloo`.

## Merge-readiness: MERGE-READY (pending the enabled GPU run the package already labels pending)

Every round-2 finding is fixed or documented with a test that asserts the claim; the strict validation cannot reject a
trajectory the shipped loop produces, at any response budget, through either trainer path; the only new items are LOW.

## Lead-requested closure (2026-10-29, p4b @ 988a10bc on p4a @ d8c4b925)

LOW 3, 4 and 5 are closed correctly: `_span_offset` takes Python / numpy / 0-d torch integers through `operator.index` and
names float and bool offsets of every flavour (Python, numpy, torch); a non-sequence `turn_spans` raises the named
`ValueError` ("must hold one entry per row (N rows), got a int"); the gate reads the outermost callable's own signature, so a
`functools.wraps` wrapper without `**kwargs` gets no spans (rloo behaviour, no TypeError) while a transparent `**kwargs`
wrapper over `turn_rloo` gets them and a transparent wrapper over `rloo` does not; the built-ins stay off the V1 fetch. The
three pinning tests now assert the fixed behaviour (`test_non_integral_offsets_are_rejected_naming_the_row_and_the_type` x6,
`test_integral_offsets_of_every_flavour_are_accepted`, `test_row_count_mismatch_and_a_non_sequence_column_are_named`,
`test_v0_wraps_wrapper_whose_own_signature_cannot_take_the_kwarg_gets_no_spans`). Nothing in the fixes is wrong.
Counts: r2d / r2e / r2f 50 passed on wt/p4b @ 988a10bc; worktrees clean. Verdict unchanged: MERGE-READY.
