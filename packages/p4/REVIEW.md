# P4 adversarial review: `[rollout, algo, tool] feat: turn-level rewards, episode done and a turn-LOO advantage for multi-turn agent loops`

Reviewer: Claude Fable 5.1 (AI), read-only on `wt/p4` (branch `contrib/p4-turn-rewards`, base main @ 2ab35292) and on the
main clone. Date 2026-10-29. Line numbers are the worktree at HEAD (c2f29703). Adversarial tests and their output:
`packages/p4/review-tests/` (`RESULTS.txt`). Nothing pushed, no GitHub writes, no Modal.

## 1. Verdict: NEEDS-FIXES

The single most important reason: **a default-path crash with the flag OFF**. `ray_trainer.py:260-261` forwards
`turn_spans` to *every* registry estimator whenever `generated_turn_spans` is in `non_tensor_batch`, and the V0
`_postprocess` now adds that column (None-filled) to every agent-loop batch via `default_extra_keys`
(`agent_loop.py:1219-1226`). `compute_grpo_vectorized_outcome_advantage` (`core_algos.py:335-343`) has no `**kwargs`,
so `adv_estimator=grpo_vectorized` with any V0 agent-loop rollout, and any user estimator registered by name without
`**kwargs`, dies at `ray_trainer.py:281` with `TypeError: ... unexpected keyword argument 'turn_spans'`. Reproduced in
`review-tests/test_r1_adv_kwarg_on_cpu.py`. The V1 trainer hits the same line for every tool-agent batch
(`trainer_base.py:1756` lifts the spans whenever one row has them). The fix is a two-line gate.

Everything else is sound: the span bookkeeping mirrors `ContinuousTokenBuilder.align_response_metadata` exactly, the
dense writer only ever touches `response_mask == 1` tokens, the flag-off row is bit-identical to main (verified
against main's code, not against the branch's own flag-off path), and the estimator matches an independent scalar
reference on random ragged groups and reduces to RLOO exactly. The remaining findings are one medium semantic hole
(the flag is a silent no-op whenever the reward comes from a colocated reward model) and a handful of low-severity
robustness gaps, each with a one-to-three line fix.

## 2. Defects

| # | Sev | Where | Failing input | Proposed fix |
|---|---|---|---|---|
| D1 | **High** | `verl/trainer/ppo/ray_trainer.py:260-261` (+ `agent_loop.py:1219-1226`, `trainer_base.py:1756`) | V0: any agent-loop batch (even single-turn, flag off) with `algorithm.adv_estimator=grpo_vectorized`, or any by-name estimator lacking `**kwargs`. V1: any tool-agent batch, same estimators. `TypeError` at `:281`. Test R1 (2 failures). | Gate the kwarg: `if "generated_turn_spans" in data.non_tensor_batch and adv_estimator in ("turn_rloo",)` (or `"turn_spans" in inspect.signature(adv_estimator_fn).parameters or VAR_KEYWORD present`). Without the gate the PR changes the estimator contract and needs `[BREAKING]`. |
| D2 | Medium | V0 `agent_loop.py:1180-1196`, `ray_trainer.py:1565-1567`; V1 `trainer_base.py:578-580`, `:1082-1084`, `:1567-1573`; `reward_loop/reward_manager/base.py:78-82` | `turn_level_rewards=True` with the reward computed by a colocated reward model (`reward_loop_worker_handles is None` / `use_rm`). V0: `_postprocess` writes `rm_scores` only when every `reward_score` is set, so the turn rewards are never written; V1: `as_dict` writes a dense row without the outcome, then `_compute_reward_colocate` *replaces* `rm_scores` (`write_back`, `:1573`) with the RM's scalar row. The flag is a silent no-op; nothing warns. | Make the RM path additive: V0 `_postprocess` writes the dense turn rewards whenever `turn_rewards is not None` (score or not) and `assemble_rm_scores` adds its scalar onto an existing `rm_scores` row; V1 `write_back["rm_scores"] = existing + rm`. Minimum: raise at config validation when `turn_level_rewards` is on and the reward is model-based, and say so in the docs. |
| D3 | Medium/Low | `verl/experimental/agent_loop/agent_loop.py:122-124` used from `:1187-1195` | V0 padded row whose response has no `response_mask == 1` token (e.g. every generated token trimmed, or a zero-length generation) and `turn_rewards is not None`: fallback index `-1` is the last **pad column** (outside `attention_mask`), while the default path (`:1185`) uses the last valid token. REINFORCE++/GAE then lose the outcome. Test R5(f). | Give `write_turn_level_rewards` a `fallback` index; `_postprocess` passes `response_length[i]`, `as_dict` passes `-1`. |
| D4 | Low | `verl/trainer/ppo/v1/trainer_base.py:1755-1759` | A row whose `extra_fields` is not a dict: `any(...)` at `:1756` short-circuits on the first dict row, then `row.get` at `:1759` raises `AttributeError`. `_compute_metrics` (`:1926`) guards `isinstance(..., dict)`; this lift does not. Also `.tolist()` at `:1755` on the popped field, which the same file warns is backend-dependent (`:1545-1549`, `raw_prompt`). Test R6(c). | `rows = list(data.batch.pop("extra_fields"))`; `turn_spans[i] = row.get("generated_turn_spans") if isinstance(row, dict) else None`. |
| D5 | Low | `verl/trainer/ppo/core_algos.py:691-692`, `:702` | A span with `start >= length` is clipped to `(length, length)` but still marked `reached` with return 0, so it enters other members' leave-one-out baselines. Unreachable from `ToolAgentLoop` (spans are clipped at `tool_agent_loop.py:245`), reachable for any other producer of `turn_spans`. Test R4(d). | Drop such spans in the normalisation: `spans = [(s, e) for s, e in spans if s < length]`. |
| D6 | Low | `verl/experimental/agent_loop/agent_loop.py:118-121` | `write_turn_level_rewards` drops a turn reward whose span holds no `response_mask == 1` token, silently. Unreachable from `ToolAgentLoop` (every span is a run of generated tokens), reachable for custom `AgentLoopOutput.turn_rewards` producers. Test R2(c). | `logger.warning` (or raise) when `pos is None` and `reward != 0`. |
| D7 | Low | `verl/experimental/agent_loop/tool_agent_loop.py:430` then `:108-128` | The turn's step reward is added to `turn_rewards[-1]` *before* the context merge; if a author's `removed_prefix_token_count` covers the whole last span, `_align_generated_turn_spans` drops the span **and the reward**. Today's authors strip at most one boundary token (`continuous_token.py:637-645`, `:1029-1037`) or a separator (`:805-809`), so this is theoretical. Test R2(b). | Assert `retained >= spans[-1][0]` in `_align_generated_turn_spans`, or log when a span with a non-zero reward is dropped. |
| D8 | Low (tests) | `tests/trainer/ppo/test_core_algos_on_cpu.py:42-50`; branch workaround in `test_turn_rloo_on_cpu.py:36`, `v1/test_compute_advantage_turn_spans_on_cpu.py:31-35` | `TestRegisterAdvEst.setUp` rebinds the module global to a new dict and `tearDown` clears it, leaving the registry empty for the rest of the session. The branch's workaround (import-time capture + `monkeypatch.setitem`) is order-independent and does not hide a failure, but it leaves the root cause for the next contributor. | Fix `TestRegisterAdvEst` in this PR (save `dict(REGISTRY)` in `setUp`, `REGISTRY.clear(); REGISTRY.update(saved)` in `tearDown`, four lines) and drop the workaround; AGENTS.md allows mechanical fixes bundled with substantive work. |
| D9 | Low (docs) | `verl/experimental/agent_loop/tool_agent_loop.py:339-340`; `docs/advance/agent_loop.rst` item 1 | `max_assistant_turns=N` terminates before tool extraction, so the N-th turn's tool calls never run: `done` cannot fire and the last step reward is never earned (Test R5(a): `max_assistant_turns=1`, tool never called, `episode_done=False`). Pre-existing semantics, but the new feature makes it bite: an env with K steps needs `max_assistant_turns >= K + 1`. | One sentence in the docs and in the flag's yaml comment. |

Not found after a genuine attempt: span drift under assistant/context merges with inserted or removed junction
tokens (R2 parametrised over 0-2 removed and 0-2 inserted), truncation at `response_length` inside a turn or after a
tool call (R5), tool errors / unknown tools / exceptions (R5), parallel tool calls with `done` on one of them (R5),
observation tokens inside a span (R4), ragged groups, singleton groups, all-equal returns, gamma < 1 over turns, the
std option's epsilon (R4), index tensors on CPU vs GPU (every index tensor is built on `token_level_rewards.device`),
the V1 padding rows (R6), mixed single-turn / tool-agent rows (R6), nested tuples through `NonTensorStack` (R6),
`use_kl_in_reward` (per-token subtraction at `ray_trainer.py:95-106` is dense-safe).

## 3. Semantic risks needing a real-environment check

1. **Reward-model paths (D2)**: run one V0 and one V1 job with `turn_level_rewards=True` and a colocated RM and
   assert `rm_scores` carries the turn rewards; today it cannot.
2. **Validation and DAPO metrics change meaning when the flag is on.** `trainer_base.py:1122` / `:1309` and V0
   `ray_trainer.py:529` reduce `rm_scores` by `.sum(-1)`, so `critic/score/*` and `val-core/*/reward` become the
   undiscounted *return* (step rewards + outcome) instead of the outcome; `replay_buffer.py:295` filters DAPO groups
   on the same sum, so groups are rarely all-equal and filtering effectively switches off. The loop applies the flag
   during validation too (`tool_agent_loop.py:143` has no `validate` awareness). Either restrict the dense writer to
   training rows or state the new meaning in the docs.
3. **Real Continuous Token authors.** Alignment is proven only against scripted merges (0-2 removed, 0-2 inserted
   tokens). Run GLM (`continuous_token.py:637-645`, trailing `<|observation|>` removed), Gemma (`:827-834`,
   `<|tool_response>` inserted) and Kimi (`:1029-1037`) with `tokenization_sanity_check_mode=strict` and assert
   `generated_turn_spans == maximal runs of response_mask == 1` on real trajectories.
4. **GPU cost of the estimator.** `core_algos.py:700-706` does `bs x turns` Python slice assignments on device
   tensors (each a kernel launch). Measure at bs=1024, 8 turns; building `segment_id`/`span_id` on CPU and moving
   them once would remove it.
5. **The object column through V0 batch plumbing.** `generated_turn_spans` is a 1-D object array of lists of tuples
   (built with the `np.empty` + slice idiom at `agent_loop.py:1231-1232`, so equal-length rows do not become a 3-D
   array). `DataProto.chunk/concat/repeat/select_idxs/union` are axis-0 operations and fine; the `fully_async` /
   `one_step_off` tensordict conversions and a real TransferQueue round trip (V1) were not exercised here, only
   `list_of_dict_to_tensordict` (R6(d)).
6. **`done` + response budget.** `tool_agent_loop.py:449` returns `TERMINATED` *before* applying the context merge
   when the observation overflows, while `episode_done` (`:431`) and the step reward (`:430`) are already recorded:
   the trajectory then lacks the final observation although `episode_done` is True (R5(c)). Fine for training; a
   reward function that reads the final tool text will not find it.
7. **Group composition in V1.** Multi-output sessions share `uid`, so `turn_rloo`'s baseline mixes outputs of the
   same session; rollout-correction rejection zeroes `response_mask` but rejected rows still enter baselines. Both
   are exactly RLOO's behaviour today, inherited rather than introduced.
8. **Learning effect.** No GPU/e2e evidence in the package (the brief did not ask for Modal). The PR body is honest
   about this; the maintainers will ask for a curve on the grid-world env the body describes.

## 4. Merge-process gaps and split proposal

Compliance verified: title passes `tests/special_sanity/check_pr_title.py`; duplicate-check links and named prior
work (#8305, #8368, #7244, #8099, #8203) with differences; AI statement; `Co-authored-by` trailers on all five
commits; `autogen-trainer-cfg` regenerated bit-identically in a scratch copy (`scripts/generate_trainer_config.sh`,
"All good"); `ruff check` / `ruff format --check` clean (ruff 0.16.8 from a cached hook env; the config pins v0.12.2,
which the author ran); mypy 1.17.0 clean on `core_algos.py`; `check_docstrings`, `check_license`,
`check_docs_time_info`, `validate_structure`, `check_device_api_usage`, `check_dataproto_usage`,
`check_uv_gpu_only` all pass; new files carry the Apache header; docs updated and dated.

Gaps:

- **Test commands.** AGENTS.md asks for "test commands run and results"; the body lists files and counts and
  points at `tests.log`. Add the one-line invocation (`PYTHONPATH=<tree> python -m pytest -c pytest-cpu.ini
  --asyncio-mode=auto tests/trainer tests/experimental tests/tools tests/utils`).
- **Byte-identity claim.** `test_flag_off_keeps_the_default_scalar_placement_byte_identical` compares the branch's
  flag-off path against the branch's own expectation, not against main. R3 does the main comparison (bit-identical
  `prompts/responses/response_mask/rm_scores`, `num_turns`, the whole V0 `_postprocess` batch and `meta_info`; only
  `extra_fields` gains the three keys / three None columns). Say so in the body or port R3.
- **`[BREAKING]`.** With D1 unfixed the kwarg forwarding changes the registry contract (every estimator must accept
  `turn_spans`); with the gate it does not. Fix D1 rather than retitle.
- **Docs omissions**: the RM caveat (D2), the metric meaning (risk 2), `max_assistant_turns >= steps + 1` (D9).
- **Implicit contract tightened**: `_handle_processing_tools_state` now reads `MergeResult` attributes from whatever
  `ct_merge_context_msg` returns (`tool_agent_loop.py:451`), which is why `test_tool_call_id_on_cpu.py`'s
  `SimpleNamespace` fake had to become a real `MergeResult`. Acceptable (the base method returns `MergeResult`), but
  worth one line in the body so the fake change is not read as a drive-by.
- Not run: `compileall`, `check-naming-conventions`, `check-example-naming` (grep-level hooks, no new risk).

**Split proposal (recommended).** 1018 lines = 264 product (incl. 4 generated yamls) + 686 tests + 77 docs, two
reviewer groups (`tool_agent_loop.py`/`agent_loop.py`: Xiaole Guo, tongyx361; `core_algos.py`: vOv, Wyett), and
`core_algos.py` is the slow lane (45 open PRs). The two halves build and test independently:

- **[1/2] `[tool, rollout] feat: ToolResponse.done, generated turn spans and an opt-in dense rm_scores writer for
  ToolAgentLoop`**: `schemas.py`, `tool_agent_loop.py`, `agent_loop.py`, `workers/config/rollout.py`, `rollout.yaml`,
  four generated yamls, docs items 1-3, tests `test_turn_level_rewards_on_cpu.py` (minus the `turn_rloo` group
  test), `test_tool_call_id_on_cpu.py`, `test_agent_loop_tq_on_cpu.py` (+1). About 160 product / 330 test lines.
  Useful alone with `reinforce_plus_plus` / `gae` (the writer's placement rule is what makes them credit turns).
  D1 does not exist in this half (the kwarg hunk is in [2/2]).
- **[2/2] `[algo, trainer] feat: turn_rloo advantage estimator (RFC #7244) with turn-span plumbing`**:
  `core_algos.py`, the gated `ray_trainer.py` hunk, the `trainer_base.py` lift, docs estimator paragraph,
  `ppo.md`, `config.rst`, tests `test_turn_rloo_on_cpu.py`, `v1/test_compute_advantage_turn_spans_on_cpu.py`, the
  group test. About 115 product / 360 test lines. Degrades to RLOO exactly when no spans are present, so it is
  reviewable before [1/2] lands.

If the author prefers one PR, fix D1-D4 and D8-D9 first; the single PR is coherent, just large for the lane.

## 5. Adversarial tests and results

Location `packages/p4/review-tests/` (outside the worktree). Harness `scripted_loop.py`: the real `ToolAgentLoop.run`
with a scripted server/parser and CT merges whose context step can remove trailing prefix tokens and insert boundary
tokens (the branch's own harness never edits the junction). `episode_dump.py` runs one episode under whatever verl is
on `PYTHONPATH`. Run:

```
cd packages/p4/review-tests && PYTHONPATH=<repo>/wt/p4 \
  <repo>/.venv-cpu/bin/python -m pytest -p no:cacheprovider --asyncio-mode=auto \
  -o python_files='*_on_cpu.py' -rA .            # 57 tests: 50 passed, 7 failed (RESULTS.txt)
```

| File | What it tries | Result |
|---|---|---|
| `test_r1_adv_kwarg_on_cpu.py` | `compute_advantage` with the None-filled `generated_turn_spans` column every V0 agent-loop batch now carries: `grpo_vectorized`, a by-name estimator without `**kwargs`, `rloo`, `turn_rloo` with spans | **2 FAIL (D1)**: `TypeError ... 'turn_spans'` at `ray_trainer.py:281` for `grpo_vectorized` and the plain estimator; `rloo` and `turn_rloo` pass |
| `test_r2_ct_clipping_on_cpu.py` | spans vs mask runs and reward placement under (removed, inserted) in {(0,0),(1,0),(0,1),(1,1),(2,2)}; a merge removing more than the whole turn; a writer span over mask-0 tokens | 5 pass (alignment exact, every non-zero `rm_scores` entry sits on a mask-1 token); **2 FAIL by design** (D7: all spans and turn rewards vanish while `tool_rewards` keeps them; D6: the reward is dropped silently) |
| `test_r3_flag_off_vs_main_on_cpu.py` | the same 4-turn episode (one junction token removed per merge, no `done`) under main and under the branch flag-off | 3 pass: `prompts/responses/response_mask/rm_scores` bit-identical and same dtype, `num_turns` and generate calls equal, V0 `_postprocess` batch and `meta_info` identical; `extra_fields` / non-tensor keys differ by exactly `{turn_rewards, generated_turn_spans, episode_done}` |
| `test_r4_turn_rloo_reference_on_cpu.py` | 30 random ragged cases (5 seeds x gamma {1, .9, .5} x std {off, on}; group sizes 1-4, 1-4 turns, rewards on generated, observation and pad tokens) against a scalar RFC #7244 reference; no spans == RLOO with singleton groups; list-of-lists and 2-D numpy spans; a span past the row end | 33 pass; **1 FAIL (D5)**: the clipped-away span counts as reached with return 0 and shifts the other members' baseline (2.0 instead of 0.0) |
| `test_r5_loop_edges_on_cpu.py` | `done` on turn 1 of `max_assistant_turns=1`; truncation mid-turn; the context-merge budget hit; unknown tool / tool exception; two parallel calls with `done` on one; V0 row with no generated token | 5 pass (documenting D9 and risk 6); **1 FAIL (D3)**: dense fallback writes the outcome at the pad column `[5]`, the default path at the last valid token `[4]` |
| `test_r6_v1_lift_on_cpu.py` | V1 `_compute_advantage` with a synthetic padding row (template-copied spans, 1-token response, pad uid), mixed rows with/without spans, a non-dict `extra_fields` row, `list_of_dict_to_tensordict` round trip | 3 pass; **1 FAIL (D4)**: `AttributeError: 'NoneType' object has no attribute 'get'` at `trainer_base.py:1759` |

Minimal fixes for the failures: D1 gate (2 lines, `ray_trainer.py:260-261`); D3 `fallback` argument (3 lines,
`agent_loop.py:94-124`, `:1189-1195`); D4 `list(...)` + `isinstance` (2 lines, `trainer_base.py:1755-1759`); D5 one
filter (`core_algos.py:691`); D6/D7 a warning each. After D1-D5, R1/R4/R5/R6 pass; R2(b)/(c) stay as documentation
of silent drops unless D6/D7 raise.

## 6. What I verified correct

- **Span bookkeeping.** `_align_generated_turn_spans` (`tool_agent_loop.py:108-128`) applies the same three edits in
  the same order as `ContinuousTokenBuilder.align_response_metadata` (`continuous_token.py:386-415`: trim
  `removed_prefix_token_count`, then `[0] * len(inserted_token_ids)`, then the appended tokens), and runs before the
  mask is replaced at both merge sites (`:327-329`, `:451-453`); on the context-merge budget hit (`:449-450`) neither
  the mask nor the spans advance. Spans equal the maximal runs of `response_mask == 1` in every scripted path (R2,
  R5), including truncation at `response_length` inside a turn (`:242-247` clips both) and a turn whose tools ran but
  whose observation overflowed (reward kept, observation dropped). A zero-token generation appends no span and no
  reward slot (`:122-125`); the tool-reward add-in is guarded (`:429-430`).
- **`done`.** Read through `_call_tool`'s rebuilt response (`:568-571`, `getattr` so duck-typed tool results keep
  working, and preserved on the image/video path), any `done` among parallel calls ends the episode after all
  responses are merged (`:426`, `:468-469`), `user_turns` still counts the final observation, `episode_done` is
  exported in all cases.
- **Dense writer.** Only `response_mask == 1` positions receive rewards (R2 checks every non-zero entry); the outcome
  joins the final turn's token (branch tests); `as_dict` and `_postprocess` agree on the placement rule (branch tests
  + R5); with the flag off the writer is `rm_scores[-1] = reward_score` exactly as main (R3 against main's code).
  `AgentLoopOutput.turn_rewards` is None for `SingleTurnAgentLoop` and for the flag-off loop, so the `elif` path is
  taken; `exclude_unset=True` sees the attribute assignment at `tool_agent_loop.py:249` (pydantic v2 tracks
  `__setattr__`; the branch's dense tests prove it).
- **Estimator math.** Segment reward, discounted return-to-go, leave-one-out mean over members that reached the turn
  (baseline 0 when none), std option with ddof=1 and the epsilon in the denominator, observation tokens zeroed
  (`in_span` at `core_algos.py:735-736`) all match an independent scalar implementation on 30 random ragged cases
  (R4). Without spans the row is one segment `[0, L)`, which reproduces `compute_rloo_outcome_advantage`
  (`:612-636`) bit-close including the `n-1` denominator and the singleton-group raw score. The two-turn MDP bias
  test (branch) reproduces: `turn_rloo` within 4 SE of the exact gradient, the per-turn-index normalisation off by
  `-p(1-p)p*mu` at >10 SE. Device handling: every index tensor is created on `token_level_rewards.device`
  (`:698-700`, `:719`).
- **Plumbing.** V0 passes `turn_spans` only in the registry branch (`ray_trainer.py:258-261`; GAE and GRPO are
  special-cased above it); V1 calls the same `compute_advantage` (`v1/utils.py:169`) after lifting the spans from
  `extra_fields` (`trainer_base.py:1747-1760`), which survives `to_padded_tensor()` like `uid` does (branch V1 test
  + R6). Padding rows (`padding_utils.py:178-195`: template `extra_fields`, fresh `pad...` uid, zero mask) are inert.
- **Config.** `MultiTurnConfig.turn_level_rewards = False` (`workers/config/rollout.py:58`), `rollout.yaml:216-219`,
  four `_generated_*.yaml` regenerated bit-identically; the flag is read once in `ToolAgentLoop.__init__`
  (`tool_agent_loop.py:143`), which both V0 and V1 construct. `gamma` defaults to 1.0 (`AlgoConfig`), so
  `turn_rloo` without configuration is the RFC's undiscounted form.
- **Regression on the branch (my runs, `PYTHONPATH=wt/p4`, `.venv-cpu`, `--asyncio-mode=auto`):** `tests/trainer`
  448 passed / 1 skipped, `tests/experimental` 80 passed, `tests/tools` 42 passed, `tests/utils` (minus the three
  macOS exclusions) 473 passed / 23 skipped; identical to the author's `tests.log` and to the main baseline plus the
  new tests. Worktree and main clone `git status --porcelain` empty afterwards.
- **PR body claims** checked against evidence: test descriptions and counts accurate; "V1 calls this same function"
  true; "no overlap in `core_algos.py` / `schemas.py` with #8305" consistent with the vet's read of #8305 (not
  re-fetched here); "flag-off bitwise unchanged" true (R3) but only self-tested in the branch; "trainer-side hunk 9
  lines" true (8 by numstat). The "known limits" paragraph is accurate but incomplete (D2, D9, metric meaning).


## 7. Fixes applied (author, 2026-10-29, on `contrib/p4-turn-rewards` as new commits; then split)

| Defect | Fix | Commit (combined branch) | Where it lives after the split |
|---|---|---|---|
| D1 kwarg half (HIGH) | `compute_advantage` forwards `turn_spans` only when `"turn_spans" in inspect.signature(adv_estimator_fn).parameters`; test `test_compute_advantage_passes_turn_spans_only_to_estimators_that_declare_them` (grpo_vectorized, a plain by-name estimator, rloo, turn_rloo) | f9d52c4c | [2/2] 13c84ffe |
| D1 column half | `ToolAgentLoop` exports `turn_rewards` / `generated_turn_spans` / `episode_done` only under the flag; `_postprocess` adds the three schema columns only in that mode (read from `rollout_config.multi_turn.turn_level_rewards`); flag-off metadata is now identical to main. Tests: flag-off key set, `test_v0_postprocess_adds_the_turn_level_columns_only_when_the_flag_is_on` | af25a804 | [1/2] 4d21a858 / 5d4b11f3 |
| D2 (medium) | fail fast: `verl/utils/config.py::_validate_turn_level_rewards_config` (called from `validate_config`, both trainers) rejects the flag when `reward.reward_model.enable and not enable_resource_pool`; docs + yaml comment state the reward-loop requirement. The additive RM path was not taken: it touches the reward manager, `_compute_reward_colocate` and V0 reward assembly across owner groups and cannot be exercised on CPU here | 31cda642 (+ docs 65502e4e) | [1/2] 5d4b11f3 / 71bd8b61 |
| D3 | `write_turn_level_rewards(..., fallback=-1)`; `_postprocess` passes its last valid token; test `test_v0_row_without_generated_tokens_puts_the_outcome_on_the_last_valid_token` | af25a804 | [1/2] 5d4b11f3 |
| D4 | `list(...)` + `tu.unwrap_non_tensor_data` + `isinstance(row, dict)`; test `..._tolerates_a_row_whose_extra_fields_is_not_a_dict` | f9d52c4c | [2/2] 13c84ffe |
| D5 | spans with `start >= length` filtered out before `reached`; test `test_spans_starting_past_the_row_end_do_not_count_as_reached` | f9d52c4c | [2/2] 13c84ffe |
| D6 | `RuntimeWarning` when a non-zero turn reward has no generated token in its span (`warnings.warn`, once per call site per process); test `test_writer_warns_when_a_turn_has_no_generated_token` | af25a804 | [1/2] 5d4b11f3 |
| D7 | `RuntimeWarning` in `_align_generated_turn_spans` when a merge removes a rewarded turn; test `test_alignment_warns_when_a_merge_removes_a_rewarded_turn` | af25a804 | [1/2] 4d21a858 |
| D8 | `TestRegisterAdvEst.setUp` keeps `dict(REGISTRY)`, `tearDown` clears and restores; the import-time capture and the `monkeypatch.setitem` fixture are gone | f9d52c4c | [2/2] 13c84ffe |
| D9 | docs item 1 and the yaml comment: the last allowed assistant turn never runs its tools, so `K` steps need `max_assistant_turns >= K + 1`; docs item 3 adds the RM caveat and that `rm_scores` sums (`critic/score/*`, `val-core/*/reward`, DAPO's filter) become returns, validation rollouts included | 65502e4e, af25a804 | [1/2] 71bd8b61 / 5d4b11f3 |
| §4 gaps | PR bodies now carry the one-line test command, the main-vs-branch identity statement, and the `MergeResult` fake note | packages | p4a / p4b PR-BODY.md |

Reviewer suite after the fixes (combined tree, same tree as [2/2]): 53 passed, 4 failed by wording: R2(b)/(c) assert the
dropped reward survives (the accepted fix warns and drops), R3's two metadata assertions encode the pre-fix always-export
behaviour (flag-off metadata is now identical to main; R3's tensor identity test passes). On [1/2] only the loop-level
files R2/R3/R5 apply (R1/R4/R6 import the estimator).

Split: `contrib/p4a-turn-signals` (3 commits on main: 5d4b11f3, 4d21a858, 71bd8b61) and `contrib/p4b-turn-rloo`
(2 commits on p4a: 13c84ffe, d8c16498); packages/p4a and packages/p4b. `contrib/p4-turn-rewards` stays the combined
reference (9 commits + 4 fix commits) and its tree equals p4b's.

## 8. Re-verification (reviewer, 2026-10-29, read-only, after af25a804 / 31cda642 / 65502e4e / f9d52c4c and the split)

**Verdicts.** `contrib/p4a-turn-signals` [1/2]: **MERGE-READY-PENDING-GPU**. `contrib/p4b-turn-rloo` [2/2]:
**MERGE-READY-PENDING-GPU** (one documentation sentence recommended, R-1). `contrib/p4-turn-rewards`: same tree as
p4b. Line numbers below are the p4b tree (= the combined tree); p4a shares them for the files it contains.

### 8.1 Tree identity and isolation

`git diff contrib/p4-turn-rewards contrib/p4b-turn-rloo` is empty (both trees hash `31b7fb6d…`); p4b's two commits
(13c84ffe, d8c16498) sit on p4a's HEAD 71bd8b61 (= merge-base). p4a contains no `turn_rloo`, `compute_turn_rloo` or
new `inspect.signature` use (strict grep; `ray_trainer.py`, `core_algos.py`, `trainer_base.py` are untouched vs
main). Sizes: p4a 193 product / 411 test / 66 docs lines over 15 files; p4b 125 / 460 / 26 over 11 files on top.
Both titles pass `check_pr_title.py` (the `[1/2]` / `[2/2]` prefix is handled by its `progress_match`).

### 8.2 Fix verification

| Defect | Where (p4b tree) | Verified by |
|---|---|---|
| D1 kwarg | `ray_trainer.py:262-267`: `turn_spans` only if `"turn_spans" in inspect.signature(fn).parameters` | R1 4/4; R9: `grpo_vectorized`, a plain by-name estimator, `rloo`, `reinforce_plus_plus` through the real loop -> `_postprocess` -> `compute_advantage`, flag off and on (p4b 10/10; p4a 8 passed + 2 skipped, where `compute_advantage` is main's and never forwards the column) |
| D1 column | `tool_agent_loop.py:251-265` exports the three keys only under the flag; `agent_loop.py:1239-1241` adds the schema columns only when `rollout_config.multi_turn.turn_level_rewards` (the shipped worker sets it, `agent_loop.py:331`) | R3 against main's code, flag off: `as_dict` and the V0 `_postprocess` batch bit-identical *including* `extra_fields` and the non-tensor key set; R10: cross-worker `DataProto.concat` with a single-turn chunk first succeeds with the flag on (None columns) and off |
| D2 | `verl/utils/config.py:122-139`, hook at `:222` | R7 on the real composed `ppo_trainer` config: raises exactly when `RewardLoopManager.reward_loop_worker_handles` (`reward_loop.py:313-320`, the property both trainers use; V0 mirrors it at `ray_trainer.py:957`) would be None, for 4 combinations plus the legacy `enable_resource_pool: null`; the full `validate_config` reaches the hook once the earlier actor checks pass. Accepted trade-off: a colocated RM gets a clear error, the additive path stays a follow-up |
| D3 | `agent_loop.py:101`, `:133-134`, `:1205` (`fallback=int(response_length[i])`) | R5(f): outcome on the last valid token, equal to the default path |
| D4 | `trainer_base.py:1757-1761`: `list(...)` + `tu.unwrap_non_tensor_data` + `isinstance(row, dict)` | R6(c) |
| D5 | `core_algos.py:692-693`: spans with `start >= length` dropped before `reached` | R4(d); R8: 16 random ragged cases with injected past-end spans (gamma {1, .9, .5}, std on/off) equal the reference on the clean spans; a straddling span is kept and clipped; an all-past-end row falls back to one segment |
| D6 / D7 | `RuntimeWarning` at `agent_loop.py:127-132` and `tool_agent_loop.py:124-129` | R2(b)/(c) rewritten to `pytest.warns`; a zero reward does not warn |
| D8 | `tests/trainer/ppo/test_core_algos_on_cpu.py:40-51` save/restore | the trainer group runs `test_core_algos` before `test_turn_rloo` and `v1/` in one session: 456 passed with the workaround removed |
| D9 / docs | `rollout.yaml:216-223`, `docs/advance/agent_loop.rst:248-271` | read |

### 8.3 The four pre-fix reviewer failures

R2(b)/(c) asserted that a reward whose tokens a merge removed (or whose span has no generated token) survives. Once the
tokens are gone conservation is impossible; warn-and-drop (not raise) is the right call inside a rollout loop. Legitimate
behaviour change; tests updated. R3's two metadata assertions encoded always-export; flag-off metadata is now exactly
main's, strictly better for the default path, and no consumer loses data it had on main (the keys never existed there).
With the flag on the keys are exported and both trainers consume them (R9, R10). No hidden regression found.

### 8.4 Regression and lint (my runs, `PYTHONPATH=<tree>`, `.venv-cpu`)

p4a: trainer 444 passed / 1 skipped, experimental 83, tools 42, utils 473 / 23 skipped. p4b: 456 / 1, 84, 42, 473 / 23.
All exit 0; worktrees and the main clone clean afterwards. On both: `check_docs_time_info`, `check_docstrings`,
`validate_structure`, `check_license` pass; ruff check and format-check clean; mypy clean on `core_algos.py` and
`utils/config.py`; `scripts/generate_trainer_config.sh` regenerates the four yamls bit-identically (scratch copies).

### 8.5 Reviewer suite after the fixes

p4b: 96 passed (R1-R10, `review-tests/RESULTS-p4b.txt`). p4a: 35 passed, 2 skipped (R2, R3, R5, R7, R9, R10;
`RESULTS-p4a.txt`). New cases since §5: R7 (fail-fast on the real config, equivalence with the handles property,
full `validate_config`), R8 (past-end filter in ragged groups), R9 (the D1 matrix through the real loop, flag off/on,
`turn_rloo` sees spans only under the flag and equals RLOO otherwise), R10 (mixed-worker concat; the hazard of a
worker without `rollout_config`).

### 8.6 Residual findings (none blocking)

- **R-1 Low, docs (p4b)** `docs/advance/agent_loop.rst:273-283`, `core_algos.py:650-660`: nowhere says that without
  `rollout.multi_turn.turn_level_rewards` the trainer receives no spans, so `turn_rloo` is plain RLOO on multi-turn
  data as well (R9 proves it). One sentence; optionally a one-time `logger.warning` in `compute_advantage` when
  `adv_estimator == "turn_rloo"` and the column is absent.
- **R-2 Low, robustness (p4a)** `agent_loop.py:1239-1241`: a worker subclass without `rollout_config` silently omits
  the schema columns and cross-worker concat asserts with the flag on (R10(b)); the shipped worker sets it.
- **R-3 Low (p4a)**: spans are tracked with the flag off too, so the D7 warning can fire in a flag-off run on a
  pathological merge; harmless.
- **R-4 Low (p4b)**: the signature gate follows `__wrapped__`; a third-party wrapper without `functools.wraps` hides
  `turn_spans` and degrades silently to RLOO.
- **R-5 GPU / e2e (both)**: unchanged since §3: the `bs x turns` slice assignments (`core_algos.py:702-707`) are
  unmeasured on GPU and there is no training curve; hence PENDING-GPU.
- **R-6 packages**: p4a/p4b `tests.log` quote the pre-update reviewer suite ("4 failed"); the current suite passes.

## 9. Residuals closed (author, 2026-10-29)

R-2 -> p4a f4651857 (cherry-picked onto the combined branch as c1733496); R-1 -> p4b 97a43c92 (rebased onto the new p4a
HEAD; cherry-picked onto the combined branch as 912f79fb). Final HEADs: p4a f4651857, p4b 97a43c92, p4 912f79fb; the
p4b and p4 trees are identical. Details in packages/p4a/REVIEW.md and packages/p4b/REVIEW.md; fresh test logs in the
two packages' tests.log.
