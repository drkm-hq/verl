# P3b [2/2] review round 2 verdict (AI re-review, 2026-10-29)

Branch `contrib/p3b-worker-relaunch` @ 0d28f34f (5 commits on p3a 951bc3f4), wt/p3b clean; `git diff contrib/p3a-inflight-ttl...contrib/p3b-worker-relaunch` is the 5 commits only (15 files). package-audit:
`p3b   contrib/p3b-worker-relaunch      0d28f34f  7    0     fresh  ok      none`.
Author suites re-run in wt/p3b: 41 passed (validate 5 + dead-worker 13 + inflight 23 items); p3a's suite passes in wt/p3a without p3b.

The submitter's probe (scratch copy, PYTHONPATH=wt/p3b): past the former `kv_batch_get(keys=[])` failure; it now stops at its own stub:
```
WARNING replay_buffer.py:491 [V1ReplayBuffer] marking 1 validation prompts of dead workers as failure
WARNING trainer_base.py:1170 [PPOTrainer] validation batch of 1 prompts has no rows to score: every group lost its agent loop worker; ...
verl/trainer/ppo/v1/trainer_base.py:1258: in _validate   self._maybe_log_val_generations(inputs=sample_inputs, ...)
E   AttributeError: 'types.SimpleNamespace' object has no attribute '_maybe_log_val_generations'
```

## the submitter's findings
| finding | status | evidence |
|---|---|---|
| 1. empty validation batch crash | FIXED in `_validate` itself; the trainer still dies one call later (B1) | guard before the colocated reward pass / fetch / decode / dump / clear, `expected_acc_counts` kept. Attacks that hold (`test_r2_validate_batches_on_cpu.py`, PASS): all-lost FIRST batch then a healthy one (remote and colocated: one sleep/wake bracket, aligned log and dump lists, `acc/mean@1 = 1/3`), all-lost LAST batch, a dead worker that wrote 1 of 2 sessions (`acc/mean@2 = 0.5`, `acc_success_only/mean@1 = 1.0`, the partial row scored once, nothing left in TQ). |
| 2. validation counters discarded / drained late | FIXED | sampling metrics summed over the batches; `_consume_recovery_metrics("validation")` inside `_validate`; the fit loop drains `rollout/` after `step()` and before that step's validation, so nothing is double-counted or lost and `val_before_train` relaunches are attributed to the validation. Validation metrics bypass `MetricsAggregator` (listing `validation/inflight_dead_worker_failed` in its sum list is dead but harmless). |
| 3 ([2/2] half) dead-worker re-issues outside the budget | FIXED | `_ttl_reissues_so_far` subtracts `inflight_dead_worker_reissued`; test present. p3a's budget defects A1-A5 are inherited unchanged: `packages/p3a/review-tests/round2` fails identically on 0d28f34f (`results-on-p3b-0d28f34f.log`). |
| 4. evidence wording | FIXED | 5/5 with per-run counters, f6846af0 = d0da57de product code, 3e63823b / 0d28f34f not GPU-run and said so. |
| 5. REBASE-PLAN | FIXED | refreshed; `_validate` guard and the drain mapped onto #8203; #6641 named. |

## New defects (tests in `packages/p3b/review-tests/round2/`; results in `results-p3b-0d28f34f.log`: 3 failed, 4 passed)
- B1 HIGH: the all-lost validation survives `_validate` and crashes the generations logger. The real `_maybe_log_val_generations` (stubbed by the author's tests) hands the empty sample list to `ValidationGenerationsLogger._log_generations_to_wandb`, whose columns are derived from `len(samples)`: an all-lost validation makes a 1-column table and the next healthy validation raises inside `wandb.Table` (`ValueError: This table expects 4 columns: ['step', 'input_1', 'output_1', 'score_1'], found 1`); healthy then all-lost raises `expects 1 columns: ['step'], found 4`. Reachable with the stock `trainer.logger: ["console", "wandb"]`, `log_val_generations > 0` (16 example scripts) and `relaunch_dead_workers=True`. `test_r2_validate_generations_logger_on_cpu.py` (2 tests: real method, real logger, real `wandb.Table`, `wandb.run` None) both FAIL at wandb `table.py:518`. Fix: skip the generations log (with a warning) when the materialized sample count no longer matches what was logged before, or pad to `log_val_generations`; a val set smaller than `log_val_generations` with a partial loss hits the same path, so a `len(samples) > 0` check alone is not enough.
- B5 LOW: a LIVE worker that marks every group `failure` without rows (upstream shape: engine or tool down) reaches the same guard: the warning claims "every group lost its agent loop worker" and `_validate` returns `{}` (main crashed at `kv_batch_get(keys=[])`; `fit()` then trips `assert val_metrics` for `val_before_train`). `test_r2_validate_batches_on_cpu.py::test_r2_b5_...:66` FAILS. Condition the message on `validation/inflight_dead_worker_failed` and report the row-less groups as a count.
- LOW: PR-BODY "`# -> 13 + 5 + 22 passed`": the inflight file collects 23 items (41 total as run).

## Hygiene
AI statement verbatim; GPU suite labelled as the stock selection; no "reviewed every line"; no upstream contact claimed; tests.log per-group sum 200 + 473 + 480 + 278 + 74 + 21 = 1526 passed / 23 + 1 + 40 = 64 skipped matches the body; diff.patch fresh; worktree clean; no remote; no Ray processes left after the runs; CHECKLIST truthful except the count line.

Run: `cd wt/p3b && PYTHONPATH=$PWD .venv-cpu/bin/python -m pytest -c pytest-cpu.ini --rootdir packages/p3b/review-tests/round2 --asyncio-mode=auto -p no:cacheprovider -q packages/p3b/review-tests/round2` (paths relative to verl-contrib; `_r2_common.py` loads the author's harness from the worktree on PYTHONPATH).

## Merge-readiness: NEEDS-FIXES
B1 is a crash from a stock logger config on exactly the path this round made survivable, and [2/2] ships on top of p3a's A1/A2/A4/A5. The detection / relaunch / in-place failure / drain logic itself has no new finding; B5 and the count line go with the rest.

## Lead closure (2026-10-29, after p3a 6445f21b / p3b 3f3d2c24)
All findings applied by the author: A1/A2/A5 (budget applied to the re-issuable set only, after the exhausted
split and the drain branch), A4 (budget per step keyed on global_steps; default allowance max_inflight_reissues x
data.train_batch_size), A3 (suspension reported only when a re-issuable prompt is held back), B1 (generations logging
skipped without samples; wandb table tolerates an empty list and a changed sample count), B5 (neutral warning;
validation/prompts_expected + validation/prompts_without_rows returned when nothing else is). The reviewer's round-2
tests pass unedited (p3a 5/5 on both trees; p3b 7/7). Full CPU regression on wt/p3b @ 3f3d2c24: 1533 / 64 / 0.
Verdict: MERGE-READY-PENDING-CHAOS-PROOF (two chaos arms + the stock suite queued on 3f3d2c24).
