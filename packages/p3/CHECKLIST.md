# P3 checklist (verl PR template items, with evidence)

| item | status | evidence |
|---|---|---|
| Similar-PR search | done | PR-BODY.md "Checklist Before Starting": 5 search links; nearest PRs/RFCs #8168, #8380, #8203, #7910, #7045, #6583 read via `gh` (read-only); vet files `research/vet/E4-self-healing-fleet.md` section 3, `S2-agent-loop-worker-liveness.md` section 3 |
| Title format | done | `PR-TITLE.txt`; `check_pr_title.py` -> valid (modules trainer, rollout, cfg; type feat); no `[BREAKING]` (additive kwargs/args only) |
| Default-off config | done | `inflight_ttl_s: null`, `relaunch_dead_workers: false`; yaml + dataclass + 4 regenerated `_generated_*.yaml`; default sync submit path asserted tag-only by a test |
| CPU unit tests for every code path | done | 18 tests in 2 files (`tests.log`); paths: TTL expiry, missing tag, val exclusion, tag survival, submit persistence on/off, re-issue fresh-uid / skip-terminal / sync early return, sync loop, async loop, sweep, TTL-off baseline, liveness hook, stuck worker, kill + relaunch, kill + relaunch off, probe without TTL, dispatch-to-dead on/off |
| Existing CPU suite still passes | done | `tests/trainer` 456 passed / 1 skipped / 0 failed; `tests/experimental` 74 passed (worktree, same harness as research/00-cpu-env.md) |
| GPU / e2e | done (chaos) / GPU suite pre-split | Modal 2xH100 separate_async kill-after-step-2: main 0/3, knobs OFF 0/1, knobs ON 2/2 (5e6f6b74); final p3b tree f6846af0 completed; no-chaos step 3.8-4.1 vs 3.8-4.0 s; GPU suite 503/0/8 on 5e6f6b74 and on p3b e43e784c, final-sha run pending (`bench/CROSSCHECK.md`, `packages/p3/bench/RESULTS.md`) |
| pre-commit on changed files | done | ruff / ruff-format / mypy Passed; check-license, check-docs-time-info, check-docstrings, check-naming-conventions, validate-structure, check-device-api-usage Passed (all-files); `generate_trainer_config.sh` -> All good |
| Docs | done | `docs/advance/v1_async_trainer.md` section "Self-healing: in-flight TTL and worker relaunch" (+17 lines), Last updated bumped |
| Scope ~300 product lines, one area | borderline | 332 insertions / 40 deletions over 7 product files (docstrings included); one area (V1 trainer + TQ agent-loop manager); no drive-by refactors (the only factor-out is `_launch_worker`, needed for relaunch) |
| Small buildable commits with the trailer | done | 11 commits `ec0ebe4a..5e6f6b74`, each with `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` |
| Nothing pushed / no remote / no GitHub writes | done | `git remote -v` in the worktree shows only the clone's origin (never pushed); `gh` used for `pr view` / `pr diff` only |
| Main clone untouched | done | `git -C verl status --porcelain` empty; work only in `wt/p3` |
| Ray hygiene | done | `ray.shutdown()` in the chaos fixture; `pgrep gcs_server/raylet` empty after the final runs |
| REBASE-PLAN onto #8203 | done | `REBASE-PLAN.md`, hunk-by-hunk |
| bench/ (Modal) | n/a for this phase | no Modal runs made; the launch registry (kept privately) not created |

## After the review (2026-10-29)
| item | status | evidence |
|---|---|---|
| D1-D12 fixed as new commits, no history rewrite | done | REVIEW.md section 8 maps each to 898d4317 / 8918ce3c / 1958cf64 / 900883df / 39f95929 |
| Reviewer tests | 6/9 | R1 R2 R5 R8 now pass; R3 R4 R9 fail at characterization lines only (tests.log) |
| Regression after fixes | done | tests/trainer 467 / 1 skipped / 0 failed; tests/experimental 74 |
| Split into p3a / p3b | done | packages/p3a, packages/p3b; both green (455 and 468 in tests/trainer, 74 experimental) |
| `--all-files` pre-commit | done once (p3b) | 14/14 hooks Passed (packages/p3b/logs/pre-commit-all-files.log) |
| Title checks | done | all three titles valid |
