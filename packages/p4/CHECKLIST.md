# P4 checklist (verl PR template, with evidence)

| Item | Status | Evidence |
|---|---|---|
| Similar PRs searched | done | PR-BODY.md "Checklist Before Starting": three search URLs; #8305 (draft, span plumbing only), #8368 (default-path placement), RFC #7244 (the estimator), #8099 (orthogonal), #8203 (rebase after) named with the difference |
| Title format `[{modules}] {type}: {description}` | done | `PR_TITLE="[rollout, algo, tool] feat: ..." python3 tests/special_sanity/check_pr_title.py` -> valid, modules `['rollout', 'algo', 'tool']`, type `feat` |
| `[BREAKING]` needed? | no | no CLI/config/signature change; one new config key with default `False`, one new optional field on `ToolResponse`, `AgentLoopOutput`, `MultiTurnConfig`; `compute_turn_rloo_advantage` is new; `compute_advantage` gains one optional kwarg forward |
| Config: new behaviour behind a key, default off | done | `rollout.multi_turn.turn_level_rewards: False` (rollout.yaml, `MultiTurnConfig`, four `_generated_*.yaml` regenerated with `scripts/generate_trainer_config.sh`); `adv_estimator: turn_rloo` is opt-in by name |
| Default path identical when off | done | `test_flag_off_keeps_the_default_scalar_placement_byte_identical` (as_dict), `test_v0_postprocess_writes_dense_rm_scores_only_for_rows_with_turn_rewards` (V0 row), `test_turn_rloo_without_spans_equals_rloo`; non-tensor metadata gains three keys (`turn_rewards`, `generated_turn_spans`, `episode_done`), tensors unchanged |
| CPU unit test per code path | done | `tests/trainer/ppo/test_turn_rloo_on_cpu.py` (7), `tests/experimental/agent_loop/test_turn_level_rewards_on_cpu.py` (6), `tests/trainer/ppo/v1/test_compute_advantage_turn_spans_on_cpu.py` (2), `test_agent_loop_tq_on_cpu.py` (+1) |
| GPU unit suite | done | `gpu_unit_tests.yml` selection on 8xH100 (Modal): 503 passed / 0 failed / 8 skipped + both torchrun tests rc=0 on this combined branch and on p4b @ 97a43c92, identical to main (`bench/README.md`, `packages/p4b/bench/RESULTS.md`) |
| Real-environment learning curve | not measured | CPU tests + GPU unit suite only; planned grid-world experiment described in PR-BODY.md |
| Existing CPU suite still passes | see tests.log | groups trainer, experimental, tools, utils on the worktree (packages/p4/tests.log) |
| Lint clean | done | `pre-commit run ruff / ruff-format / mypy --files <changed>` all Passed; `check_docs_time_info.py` passed; license headers on new files |
| Docs | done | `docs/advance/agent_loop.rst` new section "Turn-level rewards, episode done and turn_rloo" (+70 lines, dated); `docs/algo/ppo.md` and `docs/examples/config.rst` estimator lists |
| Docstrings on new public functions | done | `write_turn_level_rewards`, `compute_turn_rloo_advantage`, `_align_generated_turn_spans`, `ToolResponse.done`, `AgentLoopOutput.turn_rewards` |
| Scope < ~300 product lines, one area | done | ~210 product lines across 7 files + 4 generated yamls; tests ~420 lines; docs ~75 lines |
| Benchmark on the Modal harness | not done | not requested for P4 (capability unlock, no speed claim); the dense writer is O(turns) CPU work per trajectory |
| AI-assistance statement | done | PR-BODY.md "AI assistance"; commit trailers `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` |
| Nothing pushed / no remote / no GitHub writes | done | branch exists only in the local worktree; `gh` used read-only (`gh pr diff 8305`) |
| ci-request message | pending | The submitter, after review |
