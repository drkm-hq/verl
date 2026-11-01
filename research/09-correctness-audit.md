# 09 — Correctness audit of verl's PPO/GRPO math (agent 9)

Clone audited: `<repo>/verl` @ `2ab35292` (2026-10-28, main).
Reproductions: `<repo>/research/repro-a9/*.py`, run with
`<repo>/.venv-a9/bin/python -I <script>` (CPU, torch 2.15.0, tensordict 0.15.3);
each script's output is saved next to it as `*.out`.

Scope note (course-corrected mid-task): `verl/workers/actor/dp_actor.py`, `dp_critic.py` and `fsdp_workers.py`
no longer exist. The loss path is `verl/workers/utils/losses.py` (policy / value / sft loss functions) called from
the engines (`verl/workers/engine/fsdp/transformer_impl.py`, `verl/workers/engine/megatron/transformer_impl.py`) via
`verl/workers/engine_workers.py`; the default trainer is V1 (`trainer.use_v1: true`,
`verl/trainer/config/ppo_trainer.yaml:232`), so `verl/trainer/ppo/v1/trainer_base.py` is audited first-class and
`ray_trainer.py` (deprecated, `ray_trainer.py:285`) second.

Terminology used below: a **confirmed bug** has a script that ran and printed observed vs expected numbers; everything
else is **suspected**.

---

## 1. Loss aggregation: derivation and reproduction

### 1.1 How the engines accumulate

* The FSDP engine computes the *global* number of loss tokens of the mini-batch once
  (`fsdp/transformer_impl.py:734-739`: `batch_num_tokens = data["loss_mask"].sum()` all-reduced over the DP group,
  `dp_size = world // sp`), splits the local mini-batch into micro-batches (`engine/utils.py:92-130`;
  `rearrange_micro_batches` under `use_dynamic_bsz`), and calls `loss.backward()` on every micro-batch loss with **no
  rescaling** (`fsdp/transformer_impl.py:753-771`). Gradient sync happens once on the last micro-batch
  (`:696-725`) and FSDP/DDP *averages* gradients over the `dp*sp` mesh.
* The Megatron engine does the same but multiplies the per-micro-batch loss by `num_micro_batch`
  (`megatron/transformer_impl.py:1428`) because Megatron's pipeline schedule divides by it; with
  `calculate_per_token_loss` it instead returns `(loss * routed_num_tokens / dp_size, local_num_tokens)`
  (`:1448-1476`). CP>1 handling (issue #5983: `batch_num_tokens` reduced over pure DP undercounts by CP) is not
  CPU-verifiable here and is left as suspected.
* `losses.py:76-80` copies `dp_size`, `batch_num_tokens`, `global_batch_size` (= `ppo_mini_batch_size * rollout.n`,
  `v1/trainer_base.py:1830-1831,1856`; `ray_trainer.py:1352-1362`) and `loss_scale_factor` into
  `config.global_batch_info`, and every registered policy loss forwards `**config.global_batch_info` to `agg_loss`.

### 1.2 What `agg_loss` must compute (`core_algos.py:1140-1206`)

For a mini-batch of sequences `i` with valid-token sets `V_i`, split into micro-batches `m` (any sizes) on `dp` ranks:

| mode | per-micro-batch loss returned | after `sum over m`, then DP mean |
|---|---|---|
| token-mean (`:1170-1175`) | `Σ_{i∈m}Σ_{t∈V_i} l_it / N_global * dp` | `Σ_i Σ_t l_it / N_global` = global token mean |
| token-sum (`:1176-1180`) | `Σ_{i∈m}Σ_t l_it * dp` | global token sum |
| seq-mean-token-sum (`:1181-1188`) | `Σ_{i∈m} Σ_t l_it / B_global * dp` | global per-sequence mean of token sums |
| seq-mean-token-sum-norm (`:1189-1193`) | same `/ loss_scale_factor` where `loss_scale_factor` **defaults to `loss_mask.shape[-1]` of the micro-batch** | correct only if every micro-batch has the same padded width |
| seq-mean-token-mean (`:1194-1202`) | `Σ_{i∈m} (Σ_t l_it / |V_i|) / B_global * dp` | global per-sequence mean of token means |

Fully-masked rows are excluded from the numerator through `seq_mask` (`:1183,1197`) but the denominator is the
configured `global_batch_size`, so rows with zero response tokens (V1 synthetic padding rows, length-0 responses)
shrink the seq-mean losses slightly (suspected, metric-level; see also issue #7771).

### 1.3 Reproduction `repro_loss_agg.py` (gradient w.r.t. log-probs, 8 sequences, lengths `[12,0,3,7,12,1,9,5]`)

Compared for every `loss_agg_mode` x every registered policy loss, plus the entropy term (`losses.py:164-168`),
the `low_var_kl` term (`losses.py:175-180`) and the critic loss (`compute_value_loss`, `losses.py:226-237`):
(a) one micro-batch; (b) three micro-batches of `[1,2,5]` rows with the same global info, gradients accumulated;
(c) two DP ranks with `dp_size=2`, gradients averaged. Two column conventions: *fixed* (all micro-batches keep the
mini-batch padded width, as the legacy trainer guarantees via `max_response_len`) and *mbwidth* (each micro-batch
trimmed to its own max response length, which is what the V1 path does, see 1.4).

Results (`repro_loss_agg.out`):

* `vanilla, gspo, sapo, cispo, gpg, dro, dppo_tv, dppo_kl, clip_cov` and the entropy / KL / value terms are exactly
  micro-batch- and DP-invariant (`rel_grad_err = 0`) in token-mean, token-sum, seq-mean-token-sum and
  seq-mean-token-mean, also with `rollout_is_weights` (TIS path `core_algos.py:1364-1365`).
* **`geo_mean` (GMPO) is not invariant in any mode**: with 3 micro-batches the accumulated loss is **2.214x** the
  whole-batch loss and the gradient differs by **70 %** (`rel_grad_err = 0.698`). Cause: `core_algos.py:2000`
  `pg_loss = torch.mean(pg_losses)` ignores `loss_agg_mode` and `global_batch_info`, so each micro-batch contributes
  a local sequence mean and the sum over `M` micro-batches is `M x` too large (and unequal micro-batches are weighted
  equally regardless of size). Under `use_dynamic_bsz` `M` changes every step, i.e. the effective learning rate of
  GMPO fluctuates with packing. DP averaging happens to be right only because `torch.mean` is also what DDP's mean does.
  Already reported as issue #7913; open PRs #7916 (adds `agg_loss` + guard + tests) and #5614.
* **`kl_cov` is not invariant** (`rel_grad_err = 2.5e-2`, also across DP): the top-`k_percent` covariance selection
  (`core_algos.py:1897-1913`) is done per micro-batch, so the set of penalised tokens depends on the split.
  Issue #7829, open PR #7830 (selects once per optimizer mini-batch with an extra no-grad forward). `clip_cov`
  (`:1818-1828`, `randperm` over candidates, `clip_num = max(int(ratio * tokens), 1)` per micro-batch) has the same
  structure; it came out invariant in this toy only because the chosen tokens had zero effect, but
  `repro_clip_cov_split.py` (64 x 32 tokens, `clip_cov_ratio=0.05`, 6 seeds, split `[8,16,40]`) shows rel grad
  errors of 0.53-1.0, so it is confirmed as well (C4 in section 7).
* **`seq-mean-token-sum-norm` with the default `loss_scale_factor=None` is invariant only under the *fixed* width
  convention.** Under *mbwidth* every loss and every term breaks: vanilla loss ratio 1.916x (grad err 0.58),
  entropy term ratio 1.19x (grad err 3.0), KL term 1.24x, value loss 1.21x. Whether a real run is in the *fixed* or
  *mbwidth* regime is decided by the trainer, see 1.4.

### 1.4 Confirmed: V1 trainer + `seq-mean-token-sum-norm` normalises each micro-batch by its own padded width

Path (all cited at `2ab35292`):

* V1 writes `old_log_probs`, `advantages`, `returns`, `response_mask` back as **nested** tensors
  (`v1/trainer_base.py:1668-1669`, `:1796-1798` via `response_to_nested`).
* In `ppo_loss`, `losses.py:103` `data.select(...).to_padded_tensor()` pads `response_mask` / `old_log_probs` /
  `advantages` to the **micro-batch** maximum, and `losses.py:59` `no_padding_2_padding(model_output["log_probs"],
  data)` pads the model output to `max_response_len`, which `padding.py:113-119` takes from non-tensor data
  `max_response_len` **or, when absent, `response_lens.max()` of the micro-batch**.
* Only the legacy trainer sets `max_response_len` (`padding.py:49-51` inside `left_right_2_no_padding`, called from
  `ray_trainer.py:1335`). Nothing in `verl/trainer/ppo/v1/`, `engine_workers.py` or the engines sets it
  (`grep -rn max_response_len verl` → only `padding.py`).
* `agg_loss` then divides by `loss_mask.shape[-1]` (`core_algos.py:1190-1193`) = the micro-batch max response length.

`repro_v1_horizon.py` drives the real `ppo_loss` with jagged inputs exactly as V1 hands them over
(`dp_size=1`, `batch_num_tokens`, `global_batch_size` assigned as `transformer_impl.py:734-739` /
`trainer_base.py:1856` do, no `max_response_len`), response lengths `[12,1,3,7,12,2,9,5]`:

```
OK  token-mean               split[1,2,5] loss split/whole= 1.000 max rel grad err=0.00e+00
OK  seq-mean-token-mean      split[1,2,5] loss split/whole= 1.000 max rel grad err=0.00e+00
OK  seq-mean-token-sum       split[1,2,5] loss split/whole= 1.000 max rel grad err=0.00e+00
BAD seq-mean-token-sum-norm  loss_scale_factor=None  split[1,2,5] micro-batch widths=[12, 3, 12] whole width=[12]
                                                     loss split/whole= 3.442 max rel grad err=1.88e+00
OK  seq-mean-token-sum-norm  loss_scale_factor=None  split[4,4]   micro-batch widths=[12, 12]  -> 1.000 / 0.00e+00
OK  seq-mean-token-sum-norm  loss_scale_factor=12    split[1,2,5] -> 1.000 / 0.00e+00
OK  seq-mean-token-sum-norm  loss_scale_factor=4096  split[1,2,5] -> 1.000 / 0.00e+00
```

Observed: with the default config the accumulated gradient of a mini-batch depends on how `rearrange_micro_batches`
/ dynamic batch size grouped it (3.44x loss, 188 % gradient error in the toy; in practice the error is the ratio
between the configured `response_length` and the per-micro-batch max, i.e. large for short responses). Even the
whole-mini-batch horizon is the *mini-batch* max (not the configured `response_length` the Dr. GRPO normaliser
`1/L_max` requires), so it also drifts from step to step as response lengths change. Expected: invariant, divisor =
configured `rollout.response_length` (or `loss_scale_factor`).

Severity: silent gradient-scaling error (effective LR varies per micro-batch split and per step). Trigger:
`trainer.use_v1=true` (default) + `actor.loss_agg_mode=seq-mean-token-sum-norm` (the documented Dr. GRPO setting,
`docs/algo/grpo.md`) + `actor.loss_scale_factor` unset (default `null`, `actor.yaml:90`); also the critic in that
mode (`losses.py:231-236`) and the Megatron engine (same `losses.py` path, `megatron/transformer_impl.py` sets no
`max_response_len` either). Not in the issue/PR dumps (issue #3741 only questions the divisor in general; PR #6349
is about NaN masking in the seq modes). Fix candidates: (i) V1 passes `max_response_len=rollout.response_length`
in `extra_info` next to `global_batch_size` (`v1/trainer_base.py:1851-1862`), which `padding.py:113` already
honours; (ii) default `loss_scale_factor` to `rollout.response_length` at config validation. Regression test:
`repro_v1_horizon.py` as a `tests/workers/*_on_cpu.py` case.

### 1.5 Metrics (secondary)

* `actor/pg_loss`, `actor/entropy_loss`, `kl_loss` are tagged `AggregationType.SUM` when global info is present
  (`losses.py:85-93`), summed over micro-batches and *averaged* over DP ranks (`metric/utils.py:139-154`,
  `engine_workers.py:355-367`); consistent with the loss scaling. OK.
* `actor/pg_clipfrac`, `actor/ppo_kl`, `actor/pg_clipfrac_lower` are `masked_mean` over the *micro-batch* and
  then `AggregationType.MEAN` over micro-batches (`losses.py:128-130`), so under dynamic batch size they are
  micro-batch-weighted, not token-weighted (suspected, metrics only).

---

## 3. Advantage estimators (`repro_advantages.py`)

| case | code | observed | expected | status |
|---|---|---|---|---|
| GRPO, 16 identical rewards 0.3 (last token) | `core_algos.py:319-326` (`torch.mean` / `torch.std` in fp32) | adv = **+0.02891** for every response (loop); vectorized (`:349-358`) 0 here but **-0.29372** for 0.7 x 64 and **-0.34818** for 0.9 x 64 | exactly 0 | confirmed; open PR #8349 |
| GRPO singleton group | `:315-317` mean 0 / std 1 | adv = raw score 0.7 | documented convention | OK |
| RLOO singleton | scalar `:629-633` vs vectorized `:864` `* (c > 1)` | 0.7 vs **0.0** | agree | confirmed; issue #7793, PRs #7794 / #7581 / #7150 |
| outcome score on a `response_mask=0` tail token | `agent_loop.py:1127-1129` (score at last attention-mask token) and `:146-147` | GAE returns all 0, REINFORCE++ 0, ReMax 0 (score lost); on the last generated token: GAE returns 1 | reward must reach the policy | confirmed for the estimator side; issue #8367, PR #8368 moves the score to the last generated token. GRPO & co. sum the row so they are unaffected |
| GPG `alpha = bsz / count_nonzero(scores)` | `:810-811` | `{0,1}` encoding: adv 6.0 (alpha 8); `{-1,+1}` encoding of the *same* outcomes: adv 1.5 (alpha 1); paper `B/(B-M)` = 2 → 1.5 / 3.0 | encoding-independent | confirmed; open PR #8422 |
| grpo_passk singleton | `:516-519` | raises ValueError | documented | OK |
| GRPO loop vs vectorized, random groups | | max diff 0 | | OK (`torch.std` and `group_mean_std` both unbiased) |
| REINFORCE++ / GAE whitening with exactly one valid token in the whole batch | `torch_functional.py:322-323` | ValueError crash | degrade gracefully | suspected edge (tiny debug batches) |
| GAE at masked (tool) positions | `:250-263` | `returns` at masked slots = carried value (garbage), whitened advantages not re-masked | harmless: losses mask them (`losses.py:105`, `:224`) | OK |
| GAE skips observation tokens | `:254-256` | tested by `tests/trainer/ppo/test_core_algos_on_cpu.py:144-198` | | OK |
| `compute_rewards` (`:1124-1137`, k1 penalty) | unused anywhere (`grep -rn "compute_rewards("`) | dead code | | note |

V1-specific: `compute_advantage_for_multi_trajectories` (`v1/utils.py:149-218`) only special-cases GRPO (final session
per `{uid}_{session}` key, scalar broadcast to the other sessions with `response_mask`); every other estimator falls
through to `compute_advantage` unchanged, so the GAE/REINFORCE++ multi-trajectory bookkeeping is the same as V0.
`use_pf_ppo` on V1 mis-pairs resampled advantages with rows (issue #7956, PR #7957 rejects the combination).

---

## 4. KL (`repro_kl.py`)

* **k3 cancellation (`core_algos.py:2240-2245`, `exp(d) - d - 1` in fp32):** at `|d| = 1e-4` the per-token value
  is exactly **0** (fp64 truth 5.0e-9); summed over 32768 tokens **0 vs 1.64e-4**; at `d = 1e-3` it is 4.77e-7 vs
  5.00e-7 (5 % low). `torch.expm1(d) - d` gives 4.9986e-9 / 1.638e-4. Confirmed; open PR #8386 (also touches the
  k3-based rejection thresholds in `rollout_corr_helper.py`). Note the KL *loss* uses this estimator by default
  (`actor.yaml:116` `kl_loss_type: low_var_kl`), so small drifts are reported and penalised as zero.
* **`low_var_kl` clamp dead zones (`:2242-2245`):** gradient w.r.t. `logprob` is exactly 0 when
  `d = ref - logprob >= ~2.6` (kld clamped to 10; `d=+3` unclamped 16.1) and when `d <= ~-11`. The penalty therefore
  stops pulling the policy back precisely on tokens where it has drifted far *below* the reference
  (`d=+3` means the reference is e^3 x more likely). In between, the gradient grows as `-(e^d - 1)` (−11.2 at
  `d=2.5`). Suspected design issue (no clamp-aware gradient such as a linear tail); not in the dumps.
* **`k3+` straight-through (`:2200-2213`)** behaves as documented: value == k3, gradient == k2
  (max diff 0). Covered by `tests/trainer/ppo/test_core_algos_on_cpu.py:347-381`.
* **Which estimator has the right gradient (exact 5-way categorical, on-policy):** `|E[∇est] − ∇KL|` = k1 **0.40**,
  k3 **0.087**, k2 1.5e-8, k3+ 1.5e-8. So the default `kl_loss_type=low_var_kl` (k3) optimises a biased gradient;
  `k3+` fixes it and is opt-in. Issue #8423 (today) raises the same point for k3 distillation.
* `apply_kl_penalty` (`ray_trainer.py:78-117`, reused by V1 `trainer_base.py:1756`) uses
  `kld = old_log_probs - ref_log_prob` (k1) masked by `response_mask`, sign correct (penalises tokens where the policy
  is more likely than the reference). In **bypass mode** `old_log_probs` *is* `rollout_log_probs`
  (`rollout_corr_helper.py:1138`), so the in-reward KL compares the vLLM/SGLang (bf16, possibly `logprobs_mode`
  "processed") log-probs with the trainer's reference log-probs: a systematic non-zero KL even at step 0
  (suspected; issue #8362 shows SGLang ignores `logprobs_mode`).
* `use_kl_in_reward` and `use_kl_loss` can both be on (no validation anywhere: `grep use_kl_in_reward verl/trainer`);
  the two penalties then stack (k1 in the reward, k3 in the loss). Documented as separate knobs; not a bug, but
  nothing warns.
* Precision: the FSDP engine computes log-probs from autocast-bf16 logits (`fsdp/transformer_impl.py:1400-1408`,
  `logprobs_from_logits`); the torch fallback `logprobs_from_logits_v2` keeps bf16 (`torch_functional.py:194-206`),
  so confident tokens round to exactly 0 log-prob (issue #8376, PRs #8377/#8378). The reference model goes through
  the same code, so the *difference* is mostly consistent, but `kl_penalty` on two bf16-rounded values is noisy at
  the 1e-3 level — the same regime where k3 underflows (above).

---

## 2. Ulysses sequence-parallel padding (code audit, no CPU repro possible without a process group)

* `prepare_model_inputs` (`fsdp/transformer_impl.py:1194-1264`) flattens the micro-batch to `(1, total_nnz)`,
  rolls the labels by one (`:1213`), optionally right-pads the packed stream to a bucket
  (`_get_packed_pad_size`, `:670-682`) and then pads to a multiple of `sp_size` and slices per rank
  (`ulysses_pad_and_slice_inputs`, `ulysses.py:282-334`; `pad_size = (sp - n % sp) % sp` at `:291`, always at
  the **tail** of the packed stream). `output_args["pad_size"] = static_pad_size + sp_pad_size` (`:1259`).
* Outputs (log-probs, entropy, Σπ², score-centering tensors, critic values) are all-gathered over the SP group and
  the tail `pad_size` is stripped (`_gather_and_unpad_packed`, `:684-693` → `gather_outputs_and_unpad`,
  `ulysses.py:247-279`) **before** being re-nested with the original `cu_seqlens = input_ids.offsets()`
  (`:1460-1468`, `:1640-1643`). The padded positions therefore never belong to any sequence and cannot enter
  `response_mask`, the entropy term, the KL term or any metric: masks are built from the jagged structure, not
  from the padded stream. The rolled labels at the pad positions are garbage but their log-probs are discarded.
* Gradient scaling: `Gather.backward` multiplies by `sp_world_size` (`ulysses.py:228-230`) while `dp_size` passed
  to `agg_loss` is `world // sp` (`:655-656`). Each SP rank holds a slice of the same sequences, FSDP averages
  over `dp*sp` ranks, so the composite factor is `dp * sp / (dp*sp) = 1` per token: consistent with the token-mean
  derivation of section 1.2.
* Static bucket padding (`pad_to_length`) is disallowed with top-k distillation / score centering (`:1158-1166`)
  because those tensors are sliced with the Ulysses rule only — the guard is correct, not a bug.
* Megatron CP: `batch_num_tokens` is reduced over the pure DP group (`megatron/transformer_impl.py:801-808`); with
  static CP>1 each rank holds the full `loss_mask` for its sequences (`loss_mask.values().sum()` is the full count),
  so the count is right unless the THD/CP preprocessing shards `loss_mask` — issue #5983 claims an undercount by CP.
  Not verifiable on CPU; suspected.

Verdict for section 2: no discrepancy found in the FSDP/Ulysses path.

---

## 5. Importance ratios and rollout log-probs

* The PPO ratio is `exp(clamp(log_prob - old_log_prob, -20, 20))` (`core_algos.py:1336-1339`), in whatever dtype the
  engine returns (fp32 `log_probs` from `logprobs_from_logits` on bf16 logits; see 4). All reductions are masked
  (`agg_loss`, `masked_mean`). `pg_clipfrac` / `ppo_kl` are micro-batch means (1.5).
* `old_log_probs` (decoupled mode, default) are recomputed by the trainer in `compute_log_prob` once per batch
  (`v1/trainer_base.py:1659-1669`, sliced from the full-sequence nested output by `response_from_nested`,
  `padding.py:194-210`) with `rollout.temperature` (`:1656`), i.e. the same temperature scaling the engine applies
  (`fsdp/transformer_impl.py:1404-1408`). Issue #6280: `old_log_probs` ≠ training `log_prob` in an on-policy step
  when the inference and training micro-batch sizes differ (batch-variant kernels) — a real-hardware effect, not
  reproducible here.
* `rollout_log_probs` come straight from the engine per generated token (`agent_loop.py:125-127`,
  `continuous_token.py:398-411`), padded with `0.0` on context/tool tokens (`:412-415`) and after the response
  (`agent_loop.py:805-808`); every consumer masks with `response_mask`, so the zeros are never read.
* Rollout correction (`rollout_corr_helper.py:786-901`): `log_ratio = old_log_prob - rollout_log_prob` (`:849`),
  TIS clamps `exp(clamp(log_ratio, ±20))` at `rollout_is_threshold` (`:572-596`), IcePop zeroes outside
  `[lower, upper]` (`:597-604`), sequence-level IS uses the masked sum (`:580-586`); weights are multiplied by
  `response_mask` (`:592`) and detached (`:625`). Rejection sampling edits `response_mask` *before* the update in
  decoupled mode (`:1061`), so `batch_num_tokens` (computed from `loss_mask`, which `left_right_2_no_padding`
  aliases to the *pre-rejection* `response_mask` at `padding.py:71` in the legacy trainer; V1 writes the modified
  mask back at `trainer_base.py:1791-1798`) can disagree with the kept-token count: legacy trainer only (suspected).
  In **bypass mode** the rejection happens inside the loss per micro-batch (`core_algos.py:2505-2523`) while
  `batch_num_tokens` was counted before any rejection, so `token-mean` shrinks the loss by the fraction of tokens
  kept — open PR #8176 fixes this in all engines (confirmed by reading `:1170-1175` with `effective_mask`
  narrower than `loss_mask`; not separately reproduced).
* `geo_mean` with IS weights: `log(w + 1e-10)` averaged over the sequence (`:1992-1998`) — an IcePop-rejected
  token (w=0) contributes `-23/len` to the exponent and the rest of the sequence is down-scaled instead of the token
  being dropped; open PR #8421 (#8406). Not separately reproduced.
* k3-based rejection statistics (`rollout_corr_helper.py:265`) use `exp(x) - 1 - x` and suffer the cancellation of
  section 4 (PR #8386 covers them).
* Score centering (`score_centering.py`, `losses.py:63-74`, `core_algos.py:2396-2397`): adds
  `advantages * sc_correction` after IS weighting in the REINFORCE loss only; aggregated with the same
  `global_batch_info`; the `sc_support_*` metrics use global token normalisation (`losses.py:144-155`). No
  discrepancy found by reading; has its own CPU tests (`tests/trainer/ppo/test_score_centering_*`).

Verdict for section 5: nothing new confirmed; the bypass-mode normalisation (#8176) and geo_mean+IcePop (#8421)
are real and already have PRs.

---

## 6. Multi-turn masks and tokenization

* **What is trained on:** the "continuous token" author (`verl/utils/tokenizer/continuous_token.py`). Assistant
  tokens are the engine's own `token_ids`, appended verbatim (`merge_assistant_tokens`, `:197-204`;
  `tool_agent_loop.py:273-284`), and get `response_mask=1` plus their rollout log-probs
  (`align_response_metadata`, `:398-411`). Tool / user / system appends are rendered as a **suffix diff** of the chat
  template: `render_delta_token_id` renders `prefix` and `prefix+appended` and asserts the prefix tokens are a
  byte-identical prefix (`:243-261`, raises on drift); tool groups use a synthetic
  `[system, user, assistant(tool_calls)]` prefix (`:263-277`, `:343-363`); model-specific subclasses (GPT-OSS, Qwen,
  MiniMax, GLM, Gemma4, DeepSeek, `:466-951`) handle boundary tokens via `inserted_token_ids` /
  `removed_prefix_token_count`, which `align_response_metadata` mirrors in the mask with 0s (`:386-396`).
  So the "two kinds of multi-turn" problem (issues #1147/#3720: training on a re-tokenised chat template that
  differs from the sampled tokens) is solved by construction in the agent loop on main; the remaining trap is a
  template whose rendering of earlier turns is not append-only (`_assert_append_only`, `:327-341`, raises; issue
  #5576 / #6854 are about thinking content being stripped from history, which the author must emulate).
* **Masks:** `response_mask` = 1 on generated tokens (including whatever stop/EOS token the engine returned in
  `token_ids`), 0 on tool responses, inserted boundary tokens and the next generation prompt; the final response is
  cut at `response_length` (`tool_agent_loop.py:195-201`, `single_turn_agent_loop.py:97-98`) and right-padded
  (`agent_loop.py:791-810`, `response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]`).
  No off-by-one found at turn boundaries: masks are extended by exactly `appended_token_count` per merge.
* **Reward placement:** `agent_loop.py:1127-1129` (and `:146-147` for the TensorDict path) writes the trajectory
  score on `attention_mask[:, prompt_len:].sum() - 1`, the last *attention* token of the response. When the
  trajectory ends after a tool response, a template suffix (e.g. the `\n` after `<|im_end|>` in issue #8367) or a
  truncated generation prompt, that token has `response_mask=0` and section 3 shows GAE / REINFORCE++ / ReMax / OTB
  drop it (GRPO/RLOO/OPO/GPG sum the row and are unaffected). Open PR #8368 moves it to the last generated token.
  Rewards from `agent_loop` tool rewards (`tool_agent_loop.py:376`) go into `reward_score` too.
* **GAE and REINFORCE++ across tool spans:** both carry the bootstrap value / running return through
  `response_mask=0` positions (`core_algos.py:254-256`, `:720-726`), tested by
  `tests/trainer/ppo/test_core_algos_on_cpu.py:144-198` and `test_reinforce_pp_multiturn_on_cpu.py`. OK.
* **Legacy SGLang request path** (`verl/workers/rollout/schemas.py:702-713`) keeps a full-sequence `loss_mask`
  and slices the response part; not audited further (the agent loop is the active path).
* **V1 multi-trajectory GRPO** (`v1/utils.py:179-218`): only the final session's advantage is computed and
  broadcast with each row's own `response_mask`; `first_nnz_indices = response_mask.argmax(1)` is 0 for a fully
  masked row, whose advantage is 0 anyway. OK.

Verdict for section 6: no new mask/tokenization bug; #8367 (score on a masked tail) is the one that matters and has
a PR.

---

## 7. Confirmed-bug table

All reproductions: `<repo>/research/repro-a9/` (outputs in `*.out`).

| # | bug | file:line | observed vs expected | severity | trigger config | repro | open PR / issue |
|---|---|---|---|---|---|---|---|
| C1 | **V1 `seq-mean-token-sum-norm` divides every micro-batch by its own max response length** (no `max_response_len` set outside the legacy trainer) | `padding.py:113-119`, `losses.py:59,103`, `core_algos.py:1190-1193`; V1 never sets it (`v1/trainer_base.py:1851-1862`) | split `[1,2,5]` vs whole through real `ppo_loss`: loss 3.442x, max rel grad err 1.88 (expected identical); `loss_scale_factor=12` → identical | silent gradient scaling (per split and per step) | `trainer.use_v1=true` (default) + `loss_agg_mode=seq-mean-token-sum-norm` (Dr. GRPO) + `loss_scale_factor=null` (default); actor, critic, FSDP and Megatron | `repro_v1_horizon.py` | **none** (issue #3741 is only the general divisor question) |
| C2 | `geo_mean` (GMPO) loss is a micro-batch-local `torch.mean`, bypassing `agg_loss`/global info | `core_algos.py:2000` | 3 micro-batches: loss 2.214x, rel grad err 0.698 for every agg mode | silent gradient scaling ∝ number of micro-batches (varies with dynamic bsz) | `policy_loss.loss_mode=geo_mean` with >1 micro-batch per mini-batch | `repro_loss_agg.py` | issue #7913; PRs #7916, #5614 |
| C3 | `kl_cov` selects its top-k tokens per micro-batch | `core_algos.py:1897-1913` | rel grad err 2.5e-2 vs whole, also across DP | update depends on micro-batch size | `loss_mode=kl_cov` | `repro_loss_agg.py` | issue #7829; PR #7830 |
| C4 | `clip_cov` selects `clip_num` random tokens per micro-batch with per-micro-batch means | `core_algos.py:1812-1828` | 6 seeds, split `[8,16,40]`: rel grad err 0.53–1.0 | update depends on micro-batch size | `loss_mode=clip_cov` | `repro_clip_cov_split.py` | none for clip_cov (#7830 covers kl_cov only) |
| C5 | GRPO uniform fractional reward group gets a non-zero advantage (fp32 mean/std residue amplified by `/ (std+eps)`) | `core_algos.py:319-326`, `:352-354` | 0.3 x 16: **+0.0289** every response (loop); 0.7 x 64: **-0.2937** (vectorized) vs 0 | spurious learning signal on no-signal groups (sign depends on the value) | GRPO default, fractional rewards | `repro_advantages.py` A | PR #8349 |
| C6 | RLOO scalar vs vectorized disagree on singleton groups | `core_algos.py:629-633` vs `:864` | 0.7 vs 0.0 | estimator choice silently changes which samples train | `adv_estimator=rloo_vectorized` after filtering | `repro_advantages.py` C | issue #7793; PRs #7794, #7581, #7150 |
| C7 | Outcome score written on the last *attention* token is dropped when that token is masked (tool/template suffix) by GAE / REINFORCE++ / ReMax / OTB | `agent_loop.py:1127-1129`, `:146-147`; `core_algos.py:250-256`, `:720-726`, `:764` | returns all 0 vs 1 | whole trajectory reward lost (silently) | agent loop + multi-turn + any token-level estimator (PPO/GAE, REINFORCE++, ReMax, OTB) | `repro_advantages.py` D | issue #8367; PR #8368 |
| C8 | GPG `alpha = B / count_nonzero(scores)` depends on reward encoding | `core_algos.py:810-811` | same outcomes: adv 6.0 ({0,1}) vs 1.5 ({-1,1}); paper 1.5 / 3.0 | wrong update scale | `adv_estimator=gpg` | `repro_advantages.py` E | PR #8422 |
| C9 | k3 / `low_var_kl` underflows to exactly 0 for small log-ratios in fp32 | `core_algos.py:2240-2245`; `rollout_corr_helper.py:265` | `d=1e-4`: 0 vs 5.0e-9 per token, 0 vs 1.64e-4 over 32768 tokens; `d=1e-3`: 5 % low | KL loss/metric and k3 rejection thresholds blind to small drifts | default `kl_loss_type=low_var_kl` | `repro_kl.py` A | PR #8386 |
| C10 | `actor.clip_ratio` is dead: `clip_ratio_low/high` default to 0.2 and always win | `actor.yaml:36-42`, `core_algos.py:1321-1322` | `clip_ratio=0.3` → effective bounds 0.2/0.2 | config silently ignored | any run that sets only `clip_ratio` | inline check (this session) | PR #8302 |

Not bugs but verified by reproduction: every other registered policy loss and the entropy / KL / value terms are exactly
micro-batch- and DP-invariant in token-mean, token-sum, seq-mean-token-sum, seq-mean-token-mean (`repro_loss_agg.out`);
`k3+` value == k3 and gradient == k2; GRPO loop == vectorized on random groups; GAE/REINFORCE++ bridge tool spans.

## 8. Suspected but unconfirmed

* S1 `low_var_kl` clamp dead zones: the KL-loss gradient is exactly 0 for `ref - logprob >= ~2.6` and `<= ~-11`
  (`core_algos.py:2242-2245`; numbers in `repro_kl.py` B). Behaviour is confirmed, "bug" status is a judgment call.
* S2 Default `kl_loss_type=low_var_kl` (k3) has a biased on-policy gradient (|E∇ − ∇KL| 0.087 on a toy categorical;
  k1 0.40); `k3+`/k2 are unbiased. Design choice, see issue #8423 for the distillation variant.
* S3 Bypass mode `token-mean` divides kept-token loss by the pre-rejection `batch_num_tokens` (`core_algos.py:2505-2523`
  vs `fsdp/transformer_impl.py:734`); PR #8176. Legacy trainer: `loss_mask` aliases the pre-rejection mask
  (`padding.py:71`) while `response_mask` is replaced (`rollout_corr_helper.py:1061`).
* S4 `geo_mean` + IcePop zero weights: `log(w + 1e-10)` down-scales the whole sequence (`core_algos.py:1992-1998`); PR #8421.
* S5 seq-mean modes divide by the configured `global_batch_size` while fully-masked rows (V1 synthetic padding rows,
  empty responses) are excluded from the numerator (`core_algos.py:1183-1188`, `:1197-1202`); issue #7771 is the metric
  side of the same thing.
* S6 `pg_clipfrac` / `ppo_kl` / `pg_clipfrac_lower` are micro-batch means averaged over micro-batches
  (`losses.py:128-130`), not token-weighted, under dynamic batch size.
* S7 Megatron CP>1 `batch_num_tokens` reduced over pure DP (`megatron/transformer_impl.py:801-808`); issue #5983.
* S8 In bypass mode the in-reward KL (`apply_kl_penalty`) compares engine (bf16, `logprobs_mode`) log-probs with the
  trainer's reference log-probs (`rollout_corr_helper.py:1138`); SGLang ignores `logprobs_mode` (issue #8362, PR #8363).
* S9 bf16 `logprobs_from_logits_v2` rounds confident tokens to log-prob 0 (issue #8376, PRs #8377/#8378); interacts with
  C9 (small KL values vanish twice).
* S10 `masked_var` raises when exactly one token is valid in a whole batch (`torch_functional.py:322-323`), crashing GAE
  / REINFORCE++ whitening on degenerate tiny batches.
* S11 `use_kl_in_reward` and `use_kl_loss` can both be on with no warning (k1 in reward + k3 in loss stack).
* S12 `use_pf_ppo` on V1 mis-pairs resampled advantages (issue #7956, PR #7957) — by reading only.
* S13 `compute_rewards` (`core_algos.py:1124-1137`) is dead code.

## 9. Top-3 contribution candidates from this slice

1. **Fix C1 (new, unfiled, default trainer): make `seq-mean-token-sum-norm` use the configured response length.**
   Pass `max_response_len=self.config.actor_rollout_ref.rollout.response_length` in V1's `extra_info` for actor and
   critic updates (`v1/trainer_base.py:1809-1816`, `:1851-1862`) so `padding.py:113` pads the model output to a
   constant width; and in `losses.py` pad the selected fields to that same width (`to_padded_tensor(output_size=...)`),
   or set `loss_scale_factor` to `rollout.response_length` when unset (`agg_loss`, `core_algos.py:1189-1193`). Add
   `tests/workers/test_ppo_loss_microbatch_invariance_on_cpu.py` built from `repro_v1_horizon.py` (nested inputs
   through the real `ppo_loss`, splits `[1,2,5]` and `[4,4]`, all five agg modes, actor and critic). Dr. GRPO users on
   V1 currently get a per-step, per-split effective-LR change; nobody has filed it.
2. **A single micro-batch/DP-invariance regression test over every registered policy loss × agg mode, plus the
   `geo_mean` fix through `agg_loss`.** `tests/trainer/ppo/test_loss_aggregation_on_cpu.py` only covers `token-sum`
   and `test_value_loss_normalization_on_cpu.py` the critic; `repro_loss_agg.py` is already that harness and it caught
   C2, C3, C4. The `geo_mean` fix duplicates PRs #7916/#5614 (open since Sept/March, unmerged), so the differentiator
   is the generic test that pins *all* losses (and marks `clip_cov`/`kl_cov` as known-variant until #7830 lands),
   with a documented quota rule for `clip_cov` (`clip_num` from the global token count × micro-batch share).
3. **Numerically safe KL family in one change:** `expm1(d) - d` for k3 in `kl_penalty_forward` and the three
   rejection statistics (dup of #8386), a clamp-aware `low_var_kl` whose gradient does not vanish outside
   `[-11, 2.6]` (linear tail instead of hard `clamp(kld, -10, 10)`, S1), and docs/tests making `k3+` the recommended
   setting for `use_kl_loss` (S2, unbiased gradient, straight-through already implemented at
   `core_algos.py:2200-2213`). CPU tests from `repro_kl.py` (cancellation sums, gradient tables, exact categorical
   gradient check).

Runner-ups (all already have open PRs, cheap to review/rebase rather than re-implement): C5 (#8349), C6 (#7794),
C7 (#8368), C8 (#8422), C10 (#8302), S3 (#8176), S4 (#8421).
