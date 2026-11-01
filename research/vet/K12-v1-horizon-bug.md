# K12 v1-horizon-bug — vet (2026-10-29, adversarial vetter)

Scripts and raw output: `research/vet/K12/repro_horizon_actor_critic.py`, `.../repro_horizon_actor_critic.out`,
`.../issues_sweep.txt` (run in `.venv-a9`, torch 2.15.0, `PYTHONPATH=verl` @ 2ab35292).

## 1. Claim

Source: `research/09-correctness-audit.md` §1.4 (confirmed C1) and recommendation 1 (§7). On the default V1 trainer,
`actor.loss_agg_mode=seq-mean-token-sum-norm` (the Dr. GRPO setting, `docs/algo/grpo.md:53-62`) with the default
`loss_scale_factor=null` normalizes every micro-batch by **its own** max response length, because only the legacy
trainer's `left_right_2_no_padding` sets `max_response_len` (`padding.py:49-51`); `no_padding_2_padding` falls back
to the micro-batch max (`padding.py:113-119`), `losses.py:103` pads to the micro-batch max, and `agg_loss` divides
by `loss_mask.shape[-1]` (`core_algos.py:1190-1193`). The reproduction `research/repro-a9/repro_v1_horizon.py`
showed loss 3.442× and max relative gradient error 1.88 between split [1,2,5] and the whole mini-batch, identical
with `loss_scale_factor=12`. Unfiled. Proposed fix: make the V1 horizon the configured `rollout.response_length`.

## 2. Mechanism verification

### 2.1 The four code sites, re-read

- `verl/workers/utils/padding.py:49-51` — `left_right_2_no_padding` assigns `max_response_len = response_mask.shape[1]`,
  the padded width, which the agent loop pads to `rollout.response_length` (`agent_loop.py:793-807`). Callers:
  `ray_trainer.py:1255,1270,1295,1335,1381` only (grep over `verl/`), i.e. the V0 driver.
- `padding.py:113-119` — `no_padding_2_padding` reads `max_response_len` with default −1; for nested inputs it uses
  `response_lens.max()` of **the tensordict it is given**, i.e. the micro-batch, and emits `(bsz, max_response_len)`
  (`:132-141`).
- `verl/workers/utils/losses.py:59` pads `log_probs` that way; `:103` `data.select(...).to_padded_tensor()` pads
  `old_log_probs`/`advantages`/`response_mask` to the same micro-batch max; `:80` forwards
  `config.loss_scale_factor` (None: `actor.yaml:90`, `workers/config/actor.py:164`). Critic: `:199` (`vpreds`) and
  `:222-236` (`compute_value_loss(..., loss_scale_factor=config.loss_scale_factor)`), `critic.yaml:64,68` inherit
  the actor's mode and factor through `oc.select`.
- `verl/trainer/ppo/core_algos.py:1189-1193` — `if loss_scale_factor is None: horizon = loss_mask.shape[-1]`.

So on V1 the "horizon" is whatever the largest response in the current micro-batch happens to be. V1 never assigns
`max_response_len`: `trainer_base.py:1851-1862` lists every `extra_info` key (`global_batch_size`,
`mini_batch_size`, `epochs`, `seed`, `temperature`, …), and `extra_info` is how non-tensor data reaches the worker
(`verl/utils/transferqueue_utils.py:163,315,338`).

### 2.2 Which paths hit it

- Engines: `engine_workers.py:60,203-204,401` — `ppo_loss` is the loss function handed to **both** the FSDP and the
  Megatron engine (`megatron/transformer_impl.py:802-808` computes `batch_num_tokens` from nested `loss_mask`, so the
  nested path is the Megatron V1 path too). Actor (`ppo_loss`) and critic (`value_loss`) alike.
- Trainers: V1 sync, colocate-async and separate-async all inherit `_update_actor`/`_update_critic` from
  `trainer_base.py` (the subclasses override only `_compute_old_log_prob` and hooks). V0 (`trainer.use_v1=false`,
  deprecated with a warning at `main_ppo.py:187-195`) is **not** affected: dense inputs plus `max_response_len`.
- Default config **not** affected: `loss_agg_mode: token-mean` (`actor.yaml:86`, critic inherits). Affected only
  when the user selects `seq-mean-token-sum-norm`: documented in `docs/algo/grpo.md:57-62` (Dr. GRPO) and
  `examples/grpo_trainer/README.md:31-38`, and shipped in `examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh:76`
  (max_response_length 8192, no `loss_scale_factor`, no `use_v1` override → runs on V1 + Megatron, affected).

### 2.3 This is a regression of the V1 switch, against the config's own contract

- `actor.yaml:88-89`: "Scale factor for seq-mean-token-sum-norm … **If null, uses response_length.** Set to a constant
  to ensure consistent normalization." `docs/algo/grpo.md:60`: "If not set, uses the current batch's response length."
  On V1 both statements are false (per-micro-batch max).
- History: #4360 (szrlee, 2025-12-01) added `loss_scale_factor`; #5366 and #5417 (tongyx361, 2026-02-24/26) settled
  the divisor and deliberately kept `loss_mask.shape[-1]` as the default ("the horizon … actually a reasonable
  design", commit b8d91ef8) — **when V0 was the default**, where that width equals `rollout.response_length`. #6823
  (Begunner, 2026-06-24) made V1 the default and silently turned the horizon into the micro-batch max. Nobody filed it.

### 2.4 My re-run: actor and critic, five modes, three splits (`repro_horizon_actor_critic.out`)

Nested inputs through the real `ppo_loss`/`value_loss`, B=8, response lengths [12,1,3,7,12,2,9,5] with a tool span
in row 0, configured response_length 16, gradient on the flat model-output leaf:

| loss | mode | split | loss split/whole | max rel grad err |
|---|---|---|---|---|
| actor | token-mean, token-sum, seq-mean-token-sum, seq-mean-token-mean | [1,2,5], [4,4], singles | 1.000 | 0.00 |
| actor | seq-mean-token-sum-norm, factor None | [1,2,5] / singles | **8.996 / 48.2** | **0.26 / 4.87** |
| actor | seq-mean-token-sum-norm, factor None | [4,4] (equal widths 12,12) | 1.000 | 0.00 |
| actor | seq-mean-token-sum-norm, factor 16 | all three | 1.000 | 0.00 |
| critic | the four other modes | all three | 1.000 | 0.00 |
| critic | seq-mean-token-sum-norm, factor None | [1,2,5] / singles | **1.155 / 2.076** | **1.98 / 7.26** |
| critic | seq-mean-token-sum-norm, factor 16 | all three | 1.000 | 0.00 |

Magnitudes depend on the random draw (09 got 3.44× / 1.88 with its seed); the invariance violation is the finding,
and it is exactly one mode, exactly when widths differ, and vanishes with a constant factor. With `use_dynamic_bsz`
or plain micro-batching over nested rows, widths differ in practically every micro-batch.

### 2.5 Fix candidate (i) from report 09 is wrong as stated — tested

Injecting `max_response_len = response_length` into the TensorDict (what V0's padding does) makes **both** losses fail:
`RuntimeError: The size of tensor a (16) must match the size of tensor b (12) at non-singleton dimension 1`, because
`losses.py:103` / `:222` pad the other fields with `to_padded_tensor()` to the micro-batch max while
`no_padding_2_padding` would now emit the full width. The horizon must be fixed in the normalizer, not in padding.

### 2.6 Existing test coverage of the agg modes

`tests/trainer/ppo/test_value_loss_normalization_on_cpu.py` (#6957, abtonmoy, merged by wuxibin89 2026-07-07) checks
micro-batch invariance of `compute_value_loss` for all four modes at equal width, and **excludes**
`seq-mean-token-sum-norm` from the variable-length case (`:102-113`); its docstring (`:54-58`) states that in the
engine "differing padded widths need a constant `loss_scale_factor` (the same requirement the actor has)" — the
limitation is known to that author, treated as a user requirement, and never reconciled with `actor.yaml:88`.
`test_loss_aggregation_on_cpu.py` covers `token-sum` only. No test drives `ppo_loss` or `value_loss` with nested
inputs across splits (`ppo_loss` appears only in score-centering / rollout-correction tests). Report 09's gap is real.

## 3. Duplicate / ownership check (07 §8 recipe: `dupcheck.py` + `gh` reads, 2026-10-29)

`dupcheck.py --kw seq-mean-token-sum-norm --kw max_response_len --kw loss_scale_factor --kw "dr. grpo" --kw horizon
--path losses.py --path padding.py --merged` (49 hits, all read), plus `gh pr list --search seq-mean-token-sum-norm`
(open/merged), `gh pr list --state merged --search loss_scale_factor`, `gh issue list --search seq-mean-token-sum-norm`:

- **No open PR or issue reports the V1 micro-batch horizon.** Issue #3741 (open, `bug`, 2025-10-13, 3 comments + two
  "same question") asks whether the Dr. GRPO divisor is right; Kirrito-k423 (2026-07-21) answered with the current
  formula — nobody noticed the V1 width. #3986 (closed) was the entropy metric for this mode.
- Prior art to cite: **#2435** (johnhaofu, 2025-07-09, CONFLICTING, last commit 2025-07-09): "Update
  seq-mean-token-sum-norm to use constant normalizer (max_response_length)" — the same idea for V0, pre-`loss_scale_factor`,
  bundled with a new `token-mean-unbiased` mode; superseded by #4360. **#4721** (yurekami, 2025-12-29, test-only +449,
  last commit 2025-12-29, "Closes #907"): gradient-accumulation tests for all agg modes at `agg_loss` level with equal
  widths — overlaps the *test idea*, not the fix, and never exercises nested inputs or the loss functions.
- Conflict risk on neighbouring lines: **#8176** (Alexander230, alive, last commit 2026-10-11; `losses.py` +
  `agg_loss` token-mean branch), #6349 (Sun-Conure, stale since 2026-05-18; `agg_loss` seq branches, `torch.where`
  masking), #7916 (geo_mean), #7830 (KL-Cov, `losses.py`). A fix placed in config resolution touches none of them.
- Merged owners: #4360 szrlee (added the factor), **#5366 / #5417 tongyx361** (collaborator; owns the divisor
  semantics, merged by wuxibin89 / self), #6957 abtonmoy (critic normalization + the test file, merged by wuxibin89),
  #6823 Begunner (V1 default). Reviewer to expect: tongyx361 for the semantics, wuxibin89 for V1.

## 4. Minimum valuable PR

- `verl/utils/config.py::validate_config` (`:122`, called at `main_ppo.py:181` **before** the V1/V0 branch and before
  `OmegaConf.resolve` at `:149`): if `actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum-norm"` and
  `loss_scale_factor is None`, set it to `actor_rollout_ref.rollout.response_length` (`rollout.yaml:39`, =
  `data.max_response_length`) with one `logger.info`; same for `critic.*` when `use_critic` (the critic's `oc.select`
  may already be materialized, so set it explicitly). Assigning an existing key needs no `open_dict`; precedent for
  config fix-ups: `trainer_base.py:739-740` (`gen_batch_size`) and `:781-786` (`optim.total_training_steps`). ≈ 15 lines.
  This restores V0's effective behaviour (padded width = `response_length`) and makes `actor.yaml:88` true; it does
  **not** touch `agg_loss`, `padding.py` or `losses.py`.
- Docs: `docs/algo/grpo.md:60` reword to "If not set, `rollout.response_length` is used" (the yaml comment already
  says so); one line in `docs/` release notes if the repo keeps them.
- Tests: new `tests/trainer/ppo/test_loss_horizon_microbatch_invariance_on_cpu.py` (~120 lines) from
  `repro_horizon_actor_critic.py`: (a) nested inputs through `ppo_loss` and `value_loss`, all five modes × splits
  [1,2,5] / [4,4] / singles, asserting loss and gradient invariance, with the Dr. GRPO mode using the resolved factor;
  (b) `validate_config` on a composed `ppo_trainer` config (hydra `initialize_config_dir`/`compose`, precedent
  `tests/experimental/reward_loop/*`; CPU config tests live in `tests/workers/config/`) asserting actor and critic
  `loss_scale_factor == rollout.response_length` and untouched when the mode is `token-mean` or the factor is set.
- ≈ 150 lines, 3-4 files, no new config key, default path untouched. Title:
  `[algo, trainer, cfg] fix: normalize seq-mean-token-sum-norm by rollout.response_length on the V1 trainer`.
  Body: the four sites, the #6823 regression, #2435/#4721/#3741/#6957 named, the before/after table of §2.4.

## 5. Proof

The regression test is the proof: on main, test (a) fails for `seq-mean-token-sum-norm` (loss ratio ≠ 1, grad error
> 0.2) and passes for the other four modes; with the fix all pass; test (b) fails on main (factor stays None).
Optional Modal A/B (`bench/README.md`, preset `baseline-a`: 8×H100, Qwen3-1.7B, bs 128 × n=8, 1024/2048, 6 steps)
with `--overrides "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm actor_rollout_ref.actor.use_kl_loss=False
algorithm.norm_adv_by_std_in_grpo=False"` main vs branch: `actor/pg_loss` and `actor/grad_norm` on main are inflated
by `response_length / (micro-batch max width)` per micro-batch — for GSM8K responses of ~200-400 tokens under a 2048
budget that is ≈ 5-10× and varies per micro-batch and step; on the branch the normalizer is the constant 2048. This is
a correctness A/B (scale of the gradient), not a speed one; `timing_s/*` must be identical.

## 6. Adversarial answers

- **A1** Strongest reason it will not merge: a reviewer says "known and documented — set `loss_scale_factor`" (the
  #6957 test docstring literally says differing widths need a constant factor). Answer: `actor.yaml:88` promises
  `response_length` when null, #5417 chose that default when the width *was* `response_length`, and #6823 broke it
  silently for a documented recipe shipped in-tree; the fix changes nothing for anyone who sets the factor or runs V0.
  Second: the `algo` lane is slow (7 merged vs 52 open) and the author is first-time; mitigated by the ~15-line diff
  and a failing-then-passing test. Third: a maintainer may prefer the fix in the trainer or a hard error instead of
  a resolved default — be ready to move it (`trainer_base.py:739` style) or to raise.
- **A2** Not a speed claim. What makes the *bug* vanish: the user sets `loss_scale_factor`; micro-batches that happen
  to share a max width (never guaranteed with nested rows); V0. Data-plane rewrites (#7000 NeoProto, #7415) keep
  the micro-batch padding and so keep the bug.
- **A3** Risk: for V1 Dr. GRPO users with the factor unset, the effective gradient scale changes (becomes the constant
  documented one) — an intended behaviour change that must be called out in the PR body (not `[BREAKING]`: it restores
  the documented contract). Caught by: the new tests; `test_value_loss_normalization_on_cpu.py::test_loss_scale_factor_is_forwarded`;
  `tests/workers/config/*` for config resolution. Nothing else moves (`token-mean` default untouched).
- **A4** tongyx361 owns the divisor semantics (#5366/#5417) and would be the one to object or to want it his way;
  #8176 (alive) and #6349 (stale) edit `agg_loss`/`losses.py` lines a config-level fix avoids; #4721's test PR
  (stale 10 months) would be partially superseded by test (a). No one has filed the V1 width bug as of 2026-10-29.
- **A5** Half scope: the `validate_config` resolution + test (a) for the one mode on `ppo_loss` only (~70 lines).

## 7. Gates, scores, verdict

| gate | result |
|---|---|
| G1 not a duplicate | pass — no open/merged PR fixes it; #2435 (stale, V0, different mechanism) and #4721 (tests only) stated |
| G2 not reserved | pass — no RFC/roadmap item; owner known, no stated plan |
| G3 self-contained | pass — config resolution + docs + one test file |
| G4 provable | pass — failing-then-passing CPU regression test through the real loss functions |
| G5 default-path impact | partial — default `token-mean` unaffected; hits a documented opt-in (Dr. GRPO, `grpo.md`, grpo README) and a shipped script (`dppo_trainer/run_qwen3_30b_a3b_megatron.sh`) on the **default** trainer |

S1 speed/value **2** (no speed; silent per-micro-batch gradient scaling, actor and critic, for every V1 Dr. GRPO run —
stability) · S2 demand **2** (#3741 + two "same question", docs, one shipped script; no explicit report) · S3
certainty **5** (reproduced through `ppo_loss`/`value_loss` with nested inputs, actor + critic, Megatron path by code)
· S4 merge-ability **4** (≈ 150 lines, no conflicts if placed in config resolution, tests, known reviewers; `algo`
lane slowness is the drag) · S5 novelty **1**.

**Verdict: GO-WITH-CHANGES.** Changes: (1) fix in `validate_config`, not in padding (candidate (i) crashes, §2.5)
and not in `agg_loss` (conflicts with #8176/#6349); (2) frame it as restoring the documented default (`actor.yaml:88`)
broken by #6823, with the before/after table, not as a Dr. GRPO redesign; (3) tests through the real loss functions
with nested inputs for all five modes, actor and critic; (4) cite #2435, #4721, #3741, #6957 and ask tongyx361 for
review; (5) ship it alone — do not bundle with K5 (different area, and K5 is NO-GO). Strongest reason for GO: a
real, reproducible, silent gradient-scaling regression on the default trainer for an in-tree recipe, fixable in
~15 lines with a test that fails on main — exactly the shape that merges. Standing for the top 3: it is a stability
fix with no step-time number, so it ranks below any candidate that moves `timing_s/step`; take it as the campaign's
correctness entry only if a speed slot is not available.
