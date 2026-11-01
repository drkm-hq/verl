# K3 oldlogprob-skip — vetting report (2026-10-29, verl main@2ab35292)

## 1. Claim

Source: research/02-perf-pain-points.md §3.2 (theme D1, rank 2; issues #3996, #6280, #324) and
research/04-engine-gpu-audit.md §2.3 / F3 / C4 ("5F per step where 3F is the floor"). When the
actor performs exactly one optimizer step per sampled batch (`ppo_epochs == 1` and one mini-batch,
i.e. `ppo_mini_batch_size == train_batch_size`), the old-log-prob forward (`timing_s/old_log_prob`)
recomputes, at the same weights, exactly the log-probs the update forward is about to compute. The
V1 trainer never skips it unless `algorithm.rollout_correction.bypass_mode` substitutes the rollout
engine's log-probs (a different estimator). Proposal: an opt-in gate that skips the pass, lets
`ppo_loss` use `log_prob.detach()` as `old_log_prob` (ratio ≡ 1, clipping inert), moves the
`actor/entropy` metric into the update forward (no-grad), and keeps the rollout-vs-actor debug
metrics by returning the update pass's log-probs through the TransferQueue bridge. Expected effect:
`old_log_prob` → 0, i.e. −11 % of step time at BASELINE-A shapes run with one mini-batch (3.49 s
of 31.15 s), 0 in every other regime.

## 2. Mechanism verification (file:line in <repo>/verl)

Where the pass runs and what gates it today
- `verl/trainer/ppo/v1/trainer_base.py:585-587` `_step_once` always calls `_compute_old_log_prob`;
  the only short-circuit is bypass mode (`:1641-1648`, `old_log_probs = rollout_log_probs`).
- `:1635-1694` `_compute_old_log_prob`: `calculate_entropy=True` hard-coded (`:1654`), dispatches
  `actor_rollout_wg.compute_log_prob` (`:1659` → `verl/workers/engine_workers.py:756-759` →
  `infer_batch` `:431-475`, eval mode, `forward_only=True`), writes `old_log_probs`/`entropy` to TQ
  (`:1668-1672`), computes `actor/entropy` (`:1676-1688`) and, when `rollout.calculate_log_probs`
  (default True, `verl/trainer/config/rollout/rollout.yaml:245`), `calculate_debug_metrics`
  (`:1691-1692` → `verl/utils/debug/metrics.py:87-125`, 8 `training/rollout_*` metrics).

Why the update forward sees the same weights (exactness conditions)
- `:1828-1871` `_update_actor`: `mini_batch_size = ppo_mini_batch_size × rollout.n` (`:1830-1831`),
  `epochs = ppo_epochs` (`:1858`), `shuffle` via `dataloader_kwargs` (`:1860`, default False,
  `verl/workers/config/actor.py:177`), `calculate_entropy` only if `actor.calculate_entropy` or
  `entropy_coeff != 0` (`:1832-1834`).
- `engine_workers.py:276-373` `train_mini_batch`: `make_iterator(mini_batch_size_per_gpu, epochs)`
  (`:306-312`; `verl/utils/tensordict_utils.py:559-588`, DataLoader order = input order when
  shuffle is False), one `train_batch` per mini-batch (`:347`) → `verl/workers/engine/base.py:113-131`:
  `optimizer_zero_grad` → `forward_backward_batch` → `optimizer_step` (`:126-128`).
- `verl/workers/engine/fsdp/transformer_impl.py:728-781` `forward_backward_batch`: every
  micro-batch of the mini-batch runs forward+backward (`:755-776`) before the optimizer step; the
  weights are therefore identical across all micro-batch forwards of mini-batch 1 (gradient
  accumulation does not touch parameters). The first optimizer step is `base.py:128`, after the
  last micro-batch. Hence, with one mini-batch per DP rank
  (`data.shape[0] // mini_batch_size_per_gpu == 1`, `engine_workers.py:320`), **every** sample's
  update-pass log-prob is computed at the same θ as the old-log-prob pass. The number of
  mini-batches per rank is `sample_batch_size·n/dp ÷ ppo_mini_batch_size·n/dp`, and
  `sample_batch_size = train_batch_size // parameter_sync_step` (`trainer_base.py:540`), so the
  condition is `ppo_mini_batch_size == train_batch_size / parameter_sync_step` and `ppo_epochs == 1`.
  With k ≥ 2 mini-batches only the first is exact (CPU control below: after one AdamW step the
  ratio on mini-batch 2 already spans [0.9975, 1.0045]). Dynamic-bsz re-chunking, `use_dynamic_bsz`
  token budgets and `log_prob_micro_batch_size != ppo_micro_batch_size` only change kernel shapes
  (the #6280 bf16 noise), never which θ is used.

Every consumer of `old_log_probs` other than the PPO ratio (each one gates the skip)
- `verl/workers/utils/losses.py:98-107` `ppo_loss`: `data.select("old_log_probs", ...)` and
  `old_log_prob = data["old_log_probs"]` → the only place the ratio is formed
  (`verl/trainer/ppo/core_algos.py:1336-1339`, `pg_clipfrac` `:1355`, `ppo_kl` `:1340`). All
  registered policy losses take `old_log_prob` the same way (`core_algos.py:1285-2069`);
  `bypass_mode` loss (`:2420-2544`) is the only one reading it as the rollout log-prob.
- `trainer_base.py:1746` `_compute_advantage` fetches it for: `apply_kl_penalty` when
  `algorithm.use_kl_in_reward` (`:1754-1758` → `core_algos.py:1124-1136`, `kl = old − ref`);
  decoupled rollout correction (`:1766-1773` → `rollout_corr_helper.py:1031-1050`, IS weights from
  `old_log_probs − rollout_log_probs`; needs `rollout_correction` configured, default absent);
  the `OPTIMAL_TOKEN_BASELINE` / `TIR_OPTIMAL_TOKEN_BASELINE` estimators (`core_algos.py:876-927`,
  `:995-1043`, `pi_t = exp(old_log_probs)`; V1 wiring is open PR #8364) and `metric_utils.py:709-713`.
  The same field list already names optional fields (`values`, `ref_log_prob`), so an absent
  `old_log_probs` is tolerated by the fetch.
- `verl/trainer/ppo/v1/trainer_separate_async.py:201-222`: decoupled PPO restores π_old from a CPU
  snapshot before calling `super()._compute_old_log_prob` — by design old ≠ current θ; the gate
  must be off there (`local_trigger_step` path).
- Megatron/other engines: the skip lives in the trainer and `losses.py`, so it is engine-agnostic.

How the update pass can return per-token fields (for the metrics)
- `verl/utils/transferqueue_utils.py:347-410` `tqbridge`: any TensorDict output whose
  `batch_size[0] == meta.size` is written back to TQ (`put_data`, `:385-401`). `update_actor`
  today returns a batch-less metrics TensorDict (`engine_workers.py:365`), `train_batch` pops
  `model_output` (`:412`), and `forward_backward_batch` keeps it only under `return_model_output`
  (`transformer_impl.py:731, 772-775`); outputs are restored to input order by
  `postprocess_batch_func(indices)` (`:781`). So with one mini-batch and `shuffle=False`, the
  update pass can return `log_probs`/`entropy` nested tensors row-aligned with the batch.
- Entropy in the update pass: `transformer_impl.py:1348, 1415-1427` computes it with grad and
  `:1444` sets `inplace_backward = not (calculate_entropy or score_centering)` → a second (T,V)
  tensor in the backward (04 F14, +4.98 GB at 16k×152k). A `no_grad` entropy keeps
  `inplace_backward=True`; the entropy is computed before `logprobs_from_logits` (`:1415` vs
  `:1445`), so the in-place CE backward cannot clobber it.

CPU test (research/vet/K3/cpu_onpolicy_equivalence.py, venv .venv-a9: torch 2.15.0,
transformers 5.21.0, verl 0.10.0.dev from the read-only clone; tiny 2-layer Qwen3, 8 ragged
sequences, verl's `logprobs_from_logits` and `compute_policy_loss_vanilla`):
```
[fp32] |update-pass log_prob - old_log_prob|_max over 114 response tokens = 0.000e+00
[fp32] main pg_clipfrac=0.000e+00 |ppo_kl|=0.000e+00   K3 pg_clipfrac=0.000e+00 |ppo_kl|=0.000e+00
[fp32] |grad(main) - grad(K3)|_max = 0.000e+00  (max |grad| = 9.527e-02, rel 0.000e+00)
[k=2 control] after one AdamW step (lr 1e-4): |log_prob(mb2) - old_log_prob(mb2)|_max = 4.442e-03, ratio range [0.9975, 1.0045]  -> second mini-batch is NOT on-policy
[bf16] micro-batch 2/2/2/2 (old pass) vs 3/3/2 (update pass): |dlogp|_max=0.000e+00, tokens with |ratio-1|>1e-3: 0.0%
RESULT: PASS
```
The old pass used eval/no_grad with 4 micro-batches of 2, the update pass train mode with
micro-batches 3/3/2 and gradient accumulation (no optimizer step): log-probs and the accumulated
gradients are bitwise equal in fp32; `pg_clipfrac`/`ppo_kl` are exactly 0 either way. The bf16
shape-noise of #6280 does not reproduce on CPU (deterministic CPU GEMMs); on GPU it is what makes
today's ratio 1 ± noise instead of exactly 1 (GuoCheng24's single-GPU table on #6280).

## 3. Duplicate / ownership check (research/07 §8 recipe; dupcheck over 774 open PRs)

- `dupcheck.py --kw old_log_prob --kw on-policy --kw on_policy --kw "skip the old" --kw recompute
  --kw log_prob.detach --path verl/workers/utils/losses.py --merged`: 55 hits, none implements an
  on-policy skip. Nearest: #8026 (cr-gao, CONTRIBUTOR, last commit 1 d, CONFLICTING) makes the
  actor forward batch-invariant so old == new by kernel determinism — complementary, opt-in, does
  not remove the pass; #6519 (huaiyizhao, "no grad for entropy and kl if the loss coef is 0",
  last commit 2026-06-05 = 145 d, CONFLICTING, stale) overlaps only K3's no-grad-entropy sub-piece
  and must be named/superseded in the PR body; #8364 (sum_pi_squared through V1, 2 d) adds a
  consumer of `old_log_probs` that the gate must respect; #8201 (BREAKING: V1 mini-batches by UID,
  2 d) and #6025 (grouped mini-batching) change how mini-batches are formed in
  `_update_actor`/`train_mini_batch` — the gate must read the same "one mini-batch" fact they define.
- Merged: #3119 "[trainer, worker] fix: setting old log probs equal to log probs for on policy
  training" (sahilpatelsp, merged 2025-08-19 by vermouth1992) added exactly this shortcut to the
  legacy worker (`dp_actor.py:554` `on_policy = len(mini_batches) == 1 and ppo_epochs == 1`,
  `:604-607` at `044bbba^`); it disappeared as collateral of #6067 (2026-04-20, legacy workers
  deleted). The merged dump (2026-07-16 → 2026-10-28) has nothing on it. Note #3119 only avoided
  the *mismatch*; it never skipped the forward (#3996's complaint).
- Issues: #3996 OPEN (Kirrito-k423, 2026-07-22: "not fixed; bypass_mode is a different path");
  #6280 OPEN (6 comments, 1 reaction; Kirrito-k423 invites a PR; kaining-never-stop 2026-07-31
  confirms the shortcut died with #6067; GuoCheng24 quantifies the bf16 batch-shape noise).
  No maintainer has claimed it; roadmap #8186 has no trainer-throughput line (04 §9).
- Area ownership: `v1/trainer_base.py` is saturated (51 open PRs, 23 with commits ≤ 30 d;
  CODEOWNERS eric-haibin-lin/vermouth1992/tongyx361/PeterSH6); `workers/utils/losses.py` is quiet
  (2 active: #8176, #8026); `engine_workers.py` 19 active; `fsdp/transformer_impl.py` 24 active.

## 4. Minimum valuable PR

Title: `[trainer, worker, fsdp, cfg] feat: opt-in skip of the old-log-prob pass for strictly
on-policy updates`. One PR, no new dependency, no engine fork.

| file | change | est. lines |
|---|---|---|
| `verl/workers/config/actor.py` (+ `verl/trainer/config/actor/actor.yaml`, 4 regenerated `_generated_*.yaml`) | `skip_old_log_prob_when_on_policy: bool = False`; in the existing `validate(train_batch_size)` (`actor.py:224`) raise unless `ppo_epochs == 1`, `ppo_mini_batch_size == train_batch_size`, `shuffle == False`, `calculate_sum_pi_squared == False` | 15 + yaml 10 |
| `verl/trainer/ppo/v1/trainer_base.py` | resolve `self._on_policy_skip` once (flag ∧ `parameter_sync_step == 1` ∧ not `use_kl_in_reward` ∧ `rollout_correction` absent/bypass ∧ estimator ∉ {optimal_token_baseline, tir_...} ∧ no distillation ∧ no score_centering; the async subclass forces False); `_compute_old_log_prob` returns early without a forward; `_update_actor` adds `calculate_entropy=True`, `entropy_requires_grad = entropy_coeff != 0`, `return_log_probs=True`; the 25 lines at `:1662-1692` (entropy metric + `calculate_debug_metrics`) move into a helper called after `update_actor` reading `log_probs`/`entropy` from TQ | ~45 |
| `verl/workers/engine_workers.py` `train_mini_batch` / `train_batch` | when `return_log_probs` and a single mini-batch: keep `model_output` (`:412`), return a TensorDict with `batch_size=[n]` carrying `log_probs`, `entropy` (nested) plus the metrics → `tqbridge` writes them back (`transferqueue_utils.py:385-401`) | ~20 |
| `verl/workers/utils/losses.py` `ppo_loss` | select `old_log_probs` only if present, else `old_log_prob = log_prob.detach()`; when `entropy_coeff == 0` log `actor/entropy` from `entropy.detach()` and skip the loss term | ~10 |
| `verl/workers/engine/fsdp/transformer_impl.py` `prepare_model_outputs` | read `entropy_requires_grad` (default True); compute entropy under `torch.no_grad()` when False and key `inplace_backward` on it (`:1415-1427`, `:1444`; padded branch `:1506-1510`, `:1542`) | ~12 |
| tests (CPU) | `tests/trainer/ppo/v1/test_on_policy_skip_on_cpu.py` (gate matrix, async subclass off); `tests/workers/test_ppo_loss_on_policy_on_cpu.py` (port of research/vet/K3/cpu_onpolicy_equivalence.py: bitwise log-prob/grad equality, clipfrac 0); engine no-grad-entropy test in the monkeypatch style of `tests/workers/test_engine_return_model_output_on_cpu.py` | ~170 |
| docs | `docs/perf/perf_tuning.rst` paragraph; `docs/examples/config.rst` key | ~20 |
Source diff ≈ 110 lines, well under the 300-line merged-perf-PR envelope; default off; the flag
hard-fails on any config where the skip would be inexact (no `auto` mode, so no silent behaviour
change). Metrics keep their definitions: `actor/entropy` and `training/rollout_*` are computed
from the update pass's log-probs at the same θ; `actor/pg_clipfrac` and `actor/ppo_kl` become
exactly 0 (documented); `perf/mfu/actor_infer` disappears.

## 5. Proof on the Modal harness (bench/README.md)

BASELINE-A as shipped has k = 4 mini-batches and gets nothing; both arms must run the on-policy
variant: `--preset baseline-a --ppo-mini-batch-size 128` (one optimizer step per step).
- Arm A: `--src verl` (main@2ab35292), `--run-name k3-base-k1-1`.
- Arm B: `--src <branch>` `--overrides "actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True"`,
  `--run-name k3-skip-k1-1`.
Metrics (mean of steps 2-6 via `bench/summarize.py`): `timing_s/old_log_prob` 3.49 s → ≈ 0.05 s
(TQ bookkeeping only); `timing_s/update_actor` 8.71 s → ≈ 8.8-8.9 s (entropy over the (T,V) logits
≈ 1-2 % of one forward); `timing_s/step` ≈ 30.9 s (arm A, one optimizer step instead of four) →
≈ 27.5 s, i.e. **−11 %**, 2× the ±5 % spread so one run per arm is decisive (run `-2` to tighten).
Equality checks: `actor/entropy` within bf16 noise, `training/rollout_probs_diff_*` within noise,
`actor/pg_clipfrac = 0` and `actor/ppo_kl = 0` exactly in arm B, `kl_loss` equal within noise,
`perf/max_memory_allocated_gb` of the update unchanged (proves the no-grad entropy path).
Quality parity: `--total-training-steps 40` for both arms (reward is ~0 at 6 steps on GSM8K with
the base model); expect overlaid `critic/score/mean` curves — the gradient is identical except
that today's clip occasionally bites tokens whose kernel-noise ratio exceeds 1 ± 0.2.
Arithmetic for the rubric's reference user (8×H100, Qwen3-8B, n=8, 8k): one forward ≈ 49 s/GPU
(04 §2) out of ≈ 250-300 s/step → −16…−20 % of trainer time, −11 % of the step in the k = 1
regime; 0 at k > 1 (a first-mini-batch-only partial skip would save 1/k of the pass but needs the
driver to know each rank's mini-batch partition — not worth it).

## 6. Adversarial answers

- A1 (why not merged): it adds a second code path through three subsystems (trainer, worker,
  engine) for a regime verl's own default config does not use (`ppo_mini_batch_size` 256 vs
  `train_batch_size` 1024), in the most contested trainer file (#8201 BREAKING mini-batch
  grouping is 2 days old and rewrites `_update_actor`'s inputs); maintainers may answer "use
  `bypass_mode`" or "set `ppo_mini_batch_size` smaller" — both wrong answers, but the PR must say
  why (bypass changes the estimator to the rollout engine's log-probs; k = 1 is what the gpt-oss,
  Qwen3.5, DeepSeek-V4 and Qwen3-235B examples ship).
- A2 (gain disappears): any k ≥ 2, `ppo_epochs > 1`, `use_kl_in_reward`, a configured
  `rollout_correction`, `bypass_mode` already on, the optimal-token-baseline estimators, the
  separate-async decoupled trainer, `shuffle=True`. #8026's batch-invariant kernels do not remove
  it (the cost is compute, not noise). A memory-limited user would lose the gain if entropy forced
  `inplace_backward=False` — hence the no-grad flag.
- A3 (correctness risk): a wrong gate silently removes the clip safety net while training
  off-policy — mitigated by hard validation instead of `auto`; row alignment of returned
  log-probs requires one mini-batch and no shuffle (enforced); the async subclass must stay on the
  recompute path. Existing coverage: `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`,
  `tests/workers/test_engine_return_model_output_on_cpu.py`,
  `tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py`, GPU `tests/models/test_engine.py`,
  e2e `tests/special_e2e` GRPO scripts (would catch a KeyError on `old_log_probs` immediately).
- A4 (who else): nobody has a PR. Around it: cr-gao (#8026), kaining-never-stop and GuoCheng24
  (diagnostics on #6280), Kirrito-k423 (triage, asked for a PR), flyingdoubleG (#8364 adds an
  `old_log_probs` consumer), FZYsheep (#8201) and guillemgt (#6025) reshape mini-batching.
- A5 (half scope): drop the engine change and the returned log-probs; gate additionally on
  `rollout.calculate_log_probs=False` (or document that `training/rollout_*` are not logged under
  the skip) and take `actor/entropy` from the update pass with `calculate_entropy=True` (accepting
  the extra (T,V) in the backward). Trainer ~25 + losses ~8 + config ~10 + tests ~80 lines.

## 7. Gates, scores, verdict

G1 PASS (no open or merged PR implements it; stale #6519 overlaps one sub-piece and is named) ·
G2 PASS (no maintainer claim; #3119 is maintainer-merged precedent; no roadmap line) ·
G3 PASS (one PR, 5 source files, no new dependency) · G4 PASS (`timing_s/old_log_prob` → 0 is a
step-wall-time metric, decisive in one BASELINE-A(k=1) run; CPU test is bitwise) · G5 PASS WITH
CAVEAT (opt-in; applies to the single-mini-batch regime: 13 of the 100 shipped scripts that set
both keys resolve to it by default, incl. every Qwen3.5 / DeepSeek-V4 / gpt-oss / Qwen3-235B
GRPO example, and `run_gpt_oss_20b_fsdp.sh:64` recommends it; the shipped default config and
BASELINE-A as configured get nothing).

S1 3 · S2 4 · S3 5 · S4 3 · S5 2 — total 17/25.

**Verdict: GO-WITH-CHANGES.** Changes: opt-in flag with hard validation (no `auto`); entropy via a
no-grad flag so the update pass does not lose in-place CE backward; return the update pass's
log-probs so `actor/entropy` and the rollout-diff metrics keep their exact definitions; prove on
BASELINE-A at `ppo_mini_batch_size=128` with a 40-step parity run; name #3119/#6067/#6519/#8026 in
the PR body. Strongest reason: the mechanism is definitional and bitwise-proven, it removes one of
the actor's five forwards (−11 % of step) in the on-policy GRPO regime verl's largest examples
ship, and a core maintainer already merged the same idea once — the risk is review friction in a
saturated file, not the physics.
