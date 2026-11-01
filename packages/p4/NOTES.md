> 2026-10-29: after the adversarial review (REVIEW.md, verdict NEEDS-FIXES) the fixes landed as four commits on
> `contrib/p4-turn-rewards` (af25a804, 31cda642, 65502e4e, f9d52c4c; mapping in REVIEW.md §7) and the work was split into
> `contrib/p4a-turn-signals` (packages/p4a, [1/2]) and `contrib/p4b-turn-rloo` (packages/p4b, [2/2]). This directory is
> the combined reference; its tree equals p4b's. Test logs per branch: packages/p4a/tests.log, packages/p4b/tests.log.
> Re-verification residuals (REVIEW.md §8-§9) closed by p4a f4651857 (schema columns default off without a rollout
> config) and p4b 97a43c92 (docs: turn_rloo without the flag is RLOO), cherry-picked here as c1733496 / 912f79fb.
> Final HEADs: p4a f4651857, p4b 97a43c92, p4 912f79fb.

# P4 build notes: `[rollout, algo, tool] feat: turn-level rewards, episode done and a turn-LOO advantage for multi-turn agent loops`

Branch `contrib/p4-turn-rewards` in worktree `<repo>/wt/p4`, base main @ 2ab35292.
Author: Claude Fable 5.1 (AI); the submitter reviews every line before anything leaves this machine.
File:line references are the worktree at the base commit unless marked "(new)".

## Session log

- worktree created from main @ 2ab35292; `PYTHONPATH=wt/p4 .venv-cpu/bin/python -c "import verl"` and the same with
  `.venv-vet` (TransferQueue 0.1.11) both print the worktree path: PYTHONPATH wins over the editable install.
- read: BUILDER-RULES, vet/E3, the two CPU studies, 13 §1-2/§5, 09 §3 + bug table, 00-cpu-env, DECISION; the code
  sites listed in the brief; draft #8305's `tool_agent_loop.py` hunk (gh read-only) for the span-alignment rule.

## Design decisions (with evidence)

1. **`ToolResponse.done: bool = False`** (`verl/tools/schemas.py:98-103`). `_call_tool` rebuilds the response
   from `text` + `image`/`video` (`tool_agent_loop.py:508-520`), so `done` is carried through explicitly there.
   No `reward` field: `execute` already returns the step reward (`base_tool.py:61-74`); `FunctionTool` returns
   are normalised by `normalize_function_tool_return` (`function_tool.py:213-241`) and keep the `ToolResponse`
   object, so a function tool can also return `ToolResponse(text=..., done=True)`.
2. **Turn spans** use draft #8305's exact names and alignment rule (`AgentData.generated_turn_spans`,
   `_align_generated_turn_spans(agent_data, merge_result)`: `retained = len(mask) - removed_prefix_token_count`,
   earlier spans clipped to `retained` and dropped when `start >= retained`, an assistant merge appends
   `(retained + len(inserted_token_ids), ... + appended_token_count)`), so the two PRs can stack. Called before
   `agent_data.response_mask` is replaced (`tool_agent_loop.py:274-282` and `:386-393`), and spans are clipped
   again to `response_length` at output (`:197-199`). Difference from #8305: `turn_rewards` is BUILT by the loop
   (tool step rewards summed per assistant turn) and stays index-aligned with the spans (a dropped span drops
   its reward); #8305 has the user supply `turn_rewards` and shapes advantages after the estimator.
3. **Dense writer** behind `rollout.multi_turn.turn_level_rewards` (default False). The flag is read by
   `ToolAgentLoop` only; it sets the new typed field `AgentLoopOutput.turn_rewards` (None otherwise), and
   `as_dict` (`agent_loop.py:143-148`) / `_postprocess` (`:1124-1130`) write the dense row only when that field
   is set. Placement rule = #8368's: a turn's reward lands on the last `response_mask == 1` token of its span,
   the outcome scalar on the last mask-1 token of the response. Rationale: GAE (`core_algos.py:250-256`) and
   REINFORCE++ (`:720-726`) drop a reward that sits on a mask-0 token (bug #7, research/09 §3 row 4), and a
   `done` tool response leaves a mask-0 observation tail. Flag off: `rm_scores[-1] = reward_score`, bitwise as
   before (#8368's default-path fix is its own PR and is not folded in).
4. **Estimator `turn_rloo`** (`core_algos.py`, `register_adv_est("turn_rloo")`): RFC #7244's leave-one-out
   return-to-go baseline per turn index. Conventions (the vetter's `tensor_turn_rloo`, research/vet/E3/turn_rloo_bias.py):
   segment reward of turn t = sum of `token_level_rewards` in `[s_t, s_{t+1})` (`s_m = L`; segment 0 starts at 0
   so a reward placed before the first span is still credited), `G_{i,t} = sum_{t'>=t} gamma^{t'-t} R_{i,t'}`
   (gamma = `config.gamma`, 1.0 by default = the RFC's undiscounted form), baseline = mean of `G_{j,t}` over the
   OTHER group members that reached turn t, 0 when none did (constant => unbiased, high variance; documented).
   Rows without spans (single-turn agent loops, `turn_spans=None`) are one turn `[0, L)`, which makes the
   estimator reduce to `rloo` exactly (tested). Optional `norm_adv_by_std` kwarg, off, not wired to config.
5. **Plumbing**: `compute_advantage` (`ray_trainer.py:251-257`) passes `turn_spans` when
   `generated_turn_spans` is in `non_tensor_batch` (+2; V1 calls this same function through
   `v1/utils.py:169`); `trainer_base._compute_advantage` (`:1746`) selects `extra_fields` too and lifts the spans
   into `non_tensor_batch` (object array built with an explicit loop: `np.array` of equal-length span lists
   would become a 3-D array). V0 `_postprocess` already exports every `extra_fields` key as a column.
6. **GAE / REINFORCE++ need no change**: with rewards on mask-1 tokens both carry the return across observation
   spans (vetter's run A; regression test added). The writer's placement rule is what keeps them correct.

## Session log (continued)

- edits landed in `wt/p4`: schemas.py (+3), tool_agent_loop.py (+62/-1), agent_loop.py (+56), core_algos.py (+104),
  ray_trainer.py (+2), trainer_base.py (+9), workers/config/rollout.py (+1), rollout.yaml (+5), four generated yamls
  (+1 each, `scripts/generate_trainer_config.sh` with the venv's python3 first on PATH).
- smoke checks before the tests: `compute_turn_rloo_advantage` vs the vetter's `tensor_turn_rloo` on ragged random groups:
  max |diff| 1.2e-7; no spans vs `compute_rloo_outcome_advantage`: 2.4e-7; `as_dict` flag off -> `rm_scores[-1]`
  only, flag on -> `[.., 0.2 @2, .., 0.3 @6, .., 1.0 @13, 0 @14]` for the vetter's 3-turn layout with a mask-0 tail.
- tests: new files green (13 + 2), `test_tool_call_id_on_cpu.py` and `test_agent_loop_tq_on_cpu.py` green after the
  fake `MergeResult` / new `AgentData` attributes; regression groups trainer/experimental/tools/utils -> tests.log.
- lint: `pre-commit run ruff|ruff-format|mypy --files ...` Passed (zsh note: `${=FILES}` to word-split, a bare `$FILES`
  is one argument and pre-commit reports "no files to check"); ruff-format wrapped two long lines; the std line was
  rewritten by hand for readability. `check_docs_time_info.py` passed after bumping the three "Last updated" lines.
- title check passed as given (no shortening needed).

## Deviations from the brief / vet, with reasons

- The estimator reads `config.gamma` as a per-turn discount (default 1.0 = the RFC's and the vetter's undiscounted
  form). Reason: research/13 §5.1 lists `algorithm.gamma < 1` for cycle-level discounting; one line, tested.
- `norm_adv_by_std` is a function kwarg (default False), not a config key. Reason: keeps the config surface to the one
  rollout flag; a config key would also need `AlgoConfig` + yaml + four generated yamls for an option nobody asked for.
- `turn_rewards`, `generated_turn_spans`, `episode_done` are exported in `extra_fields` whether or not the flag is on
  (like `tool_rewards` today) so reward functions and #8305-style consumers can read them; the flag only controls the
  dense `rm_scores` writer. The three keys are added to `_postprocess`'s `default_extra_keys` so mixed-agent batches keep
  one schema. This is metadata only; every tensor on the default path is unchanged.
- The V0 `ray_trainer.compute_advantage` hunk is 2 lines and is also the V1 path (`v1/utils.py:169` calls it), so both
  trainers are plumbed; the V1-specific lift in `trainer_base._compute_advantage` is 9 lines (`extra_fields` is already
  selected unconditionally at `trainer_base.py:1116`, so the row contract includes it).
- Episode deadlines: not in this PR (documented as a follow-up in the docs section), as the brief says.
- `_call_tool` reads `done` with `getattr(..., False)` like the existing `hasattr` loop for image/video, so a tool that
  returns a duck-typed response object keeps working.
- Existing test fakes changed: `tests/experimental/agent_loop/test_tool_call_id_on_cpu.py` now passes a real
  `MergeResult` (as #8305 also does) and the three new `AgentData` attributes to the `SimpleNamespace` fake.

## Self-review notes

- `_align_generated_turn_spans` runs before `agent_data.response_mask` is replaced on both merge sites; on the context
  merge it runs after the budget check (`len(response_mask) >= response_length` returns TERMINATED without applying the
  merge, exactly as before, so the spans are not advanced either).
- `done` with `max_parallel_calls > 1`: any `done=True` in the turn ends the episode after all responses are merged.
- A turn with zero generated tokens appends no span and no reward slot (#8305's rule); the tool-reward add-in guards
  `if agent_data.turn_rewards`.
- `write_turn_level_rewards` uses `torch.nonzero` per span (vectorised scan), so the padded V0 path does not loop over
  tokens in Python.

## Regression results and one finding

- Groups on the worktree (tests.log): new tests 24 passed; trainer 448 passed / 1 skipped; experimental 80 passed;
  tools 42 passed; utils 473 passed / 23 skipped. Baseline on main (research/00-cpu-env.md §7): trainer 438,
  experimental 74, tools 42, utils 473 -> the deltas are exactly the new tests (+9 trainer incl. the two v1 tests and
  the TQ +1, +6 experimental).
- Finding (pre-existing, not fixed here): `tests/trainer/ppo/test_core_algos_on_cpu.py::TestRegisterAdvEst` replaces and
  then clears the module-global `ADV_ESTIMATOR_REGISTRY` in `setUp`/`tearDown`, leaving it EMPTY for every later test in
  the same session. The first full `tests/trainer` run therefore failed `test_turn_rloo_is_registered` and the two V1
  lookups that passed in isolation. Fix in commit 5: the registration check is captured at collection time and the V1
  tests pin the entry with `monkeypatch.setitem`. Worth a separate upstream fix (save/restore the registry in that test);
  listed as a reviewer question.
- Commits on `contrib/p4-turn-rewards` (base 2ab35292): d7404832 algo+trainer, 3d0e8f87 writer+cfg, d7f44d95
  tool+rollout (done, spans), 5a72cae3 docs, + the tests fix. `git status` clean in the worktree; the main clone is
  untouched (`git -C verl status --porcelain` empty).

## Review round 2 (2026-10-29)
The combined branch contrib/p4-turn-rewards was re-pointed from 912f79fb to 988a10bc (= contrib/p4b-turn-rloo, which
stacks on contrib/p4a-turn-signals @ d8c4b925). The round-2 findings, fixes and verdicts live in packages/p4a and
packages/p4b (REVIEW.md section 10, REVIEW-ROUND2-VERDICT.md, review-tests/round2/). diff.patch here is the combined
diff against main at 988a10bc. bench/ENABLED-RUN.md and bench/gsm8k_tool_reward.py are the enabled GPU run's inputs;
its results land in packages/p4b/bench/RESULTS.md (and bench/RESULTS.md here when the bench agent writes it).

## Review round 3 (2026-10-29, the independent follow-up)
Combined branch re-pointed to e1052882 (= contrib/p4b-turn-rloo on contrib/p4a-turn-signals @ 0d4cd8ca): the flag-off
callback contract is preserved per manager (R2-04) and the v2 experiment inputs (first-correct reward, done, three
arms) live in bench/gsm8k_tool_reward.py and bench/ENABLED-RUN.md "v2 design". Round-3 records: packages/p4a and
packages/p4b REVIEW.md section 11, REVIEW-ROUND3-VERDICT.md.
