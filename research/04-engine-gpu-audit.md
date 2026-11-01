# 04 — GPU-side training hot-path audit (FSDP engine)

Clone: `<repo>/verl` @ `2ab35292` (2026-10-27, main). All `file:line`
cites are relative to that clone. No GPU was available; every byte/FLOP figure is arithmetic, every
"would" is a prediction for the H100 benchmark phase. DEFAULT = what a user gets from the shipped
yaml; OPT-IN = behind a flag.

## 0. Scope notes and the defaults that decide the hot path

**The files named in the brief moved.** `verl/workers/actor/dp_actor.py`, `dp_critic.py` and
`fsdp_workers.py` no longer exist on this commit; they were deleted by
`044bbba2 [BREAKING] [misc] refactor: deprecate workers, migrate to engines (#6067)` (git log on
`verl/workers/actor`). The hot path is now:

| Layer | File | Role |
|---|---|---|
| Worker | `verl/workers/engine_workers.py` | `TrainingWorker` (train_mini_batch / train_batch / infer_batch), `ActorRolloutRefWorker` (compute_log_prob, update_actor, update_weights) |
| Engine | `verl/workers/engine/fsdp/transformer_impl.py` | `FSDPEngine` (build, offload, forward_backward_batch, optimizer_step, get_per_tensor_param), `FSDPEngineWithLMHead` (prepare_model_inputs / prepare_model_outputs / forward_step), `FSDPEngineWithValueHead` |
| Loss | `verl/workers/utils/losses.py` + `verl/trainer/ppo/core_algos.py` | `ppo_loss`, `compute_policy_loss_vanilla`, `agg_loss` |
| Micro-batching | `verl/workers/engine/utils.py`, `verl/utils/seqlen_balancing.py` | `prepare_micro_batches` → `rearrange_micro_batches` |
| Log-prob/entropy | `verl/utils/torch_functional.py`, `verl/utils/kernel/{kernels,linear_cross_entropy}.py`, `verl/utils/experimental/torch_functional.py` | flash-attn Triton CE, Triton fused linear-CE, chunked torch fused |
| Trainer | `verl/trainer/ppo/v1/trainer_base.py` (PRIMARY, per the lead) and `verl/trainer/ppo/ray_trainer.py` (V0, slated for removal in roadmap #8186 "Remove V0 trainer") | pass orchestration |

**Defaults that select the code path** (yaml wins over dataclass when the user goes through hydra):

| Knob | Default | Where | Effect on the hot path |
|---|---|---|---|
| `model.use_remove_padding` | **True** | `verl/trainer/config/model/hf_model.yaml:40`; copied into the engine at `engine_workers.py:147` | packed `(1, T)` varlen forward, nested-tensor I/O |
| `model.use_fused_kernels` | **False** | `hf_model.yaml:61`, `actor.yaml:116` | full-vocab logits are materialized (§1) |
| `model.enable_gradient_checkpointing` | **True** | `hf_model.yaml:34`, applied at `transformer_impl.py:339-340` (HF per-decoder-layer, non-reentrant) | +1 forward recompute per update |
| `model.enable_activation_offload` | False | `hf_model.yaml:37` | opt-in (`verl/utils/activation_offload.py:500`) |
| `actor.use_dynamic_bsz` | **False** | `actor.yaml:21` (dataclass `actor.py:154`) | static `ppo_micro_batch_size_per_gpu` chunks; dynamic token-budget packing is opt-in (`engine/utils.py:108-129`) |
| `actor.ppo_max_token_len_per_gpu` / `rollout.log_prob_max_token_len_per_gpu` | 16384 / = actor's | `actor.yaml:25`, `rollout.yaml:89` | micro-batch token budget (T) for update and for the old-log-prob pass |
| `actor.calculate_entropy`, `entropy_coeff` | False, 0 | `actor.yaml:66-68` | no entropy in the update pass; **but** the old-log-prob pass hard-codes `calculate_entropy=True` (`trainer_base.py:1654`, `ray_trainer.py:1300`) |
| `entropy_from_logits_with_chunking`, `entropy_checkpointing` | False, False | `dp_actor.yaml:27-31`, `engine/fsdp.yaml:34-40` | eager (compiled) entropy over the whole micro-batch |
| `use_torch_compile` | True | `engine/fsdp.yaml:38` | only `entropy_from_logits` is compiled (`transformer_impl.py:181-189`); the model is not |
| `strategy` | **fsdp** (FSDP1) | `dp_actor.yaml:19`, `engine/fsdp.yaml:57` | FSDP1 `FullyShardedDataParallel`; `fsdp2` opt-in |
| `model_dtype` | **fp32** | `engine/fsdp.yaml:24` | fp32 master shards for actor, critic **and the forward-only ref** (§5.3) |
| `mixed_precision` | None → param bf16 / reduce fp32 / buffer fp32 | `transformer_impl.py:395-405` | bf16 compute, fp32 grad reduce-scatter |
| `reshard_after_forward` | True → `FULL_SHARD` | `engine/fsdp.yaml:17`, `engine/fsdp/utils.py:72-81` | every micro-batch re-all-gathers every layer |
| `forward_prefetch` | False | `engine/fsdp.yaml:22` | FSDP1 relies on CPU run-ahead for forward all-gather overlap |
| `param_offload`, `optimizer_offload` | False, False | `engine/fsdp.yaml:9-12` | recipes usually turn these on; cost model in §5.2 |
| `use_no_sync_for_gradient_accumulation` | **false** (yaml) vs True (dataclass `config/engine.py:290`) | `engine/fsdp.yaml:43`; read with `getattr(..., True)` at `transformer_impl.py:706-710` | yaml users reduce-scatter every micro-batch (§5.4) |
| `ppo_epochs`, `ppo_mini_batch_size` | 1, 256 (× `rollout.n`) | `actor.yaml:14,84`; `ray_trainer.py:1352-1354`, `trainer_base.py:1830-1831` | number of optimizer steps per batch |
| `pad_to_length` | False | `engine/fsdp.yaml:48` | bucketed static padding is opt-in |
| `rollout.temperature` | 1.0 | `rollout.yaml:12` | unit temperature → logits are not copied (`transformer_impl.py:96-104`) |
| `critic.ppo_max_token_len_per_gpu` | 32768 | `critic.yaml:32` | value head has no vocab tensor, so 2× the actor budget |

Flash-attn is pinned to 2.8.3 and transformers to 5.16.1 (`pyproject.toml:78,100`). The transformers
Qwen3 head does `logits = self.lm_head(hidden_states[:, slice_indices, :])` with
`logits_to_keep: int | torch.Tensor = 0` (fetched `modeling_qwen3.py` v5.16.1 lines 457, 494-495),
i.e. logits come back in the autocast dtype (bf16) with no `.float()`; verl never passes
`logits_to_keep`.

## 1. Actor-update tensor anatomy

Shape used throughout: Qwen3-8B (H = 4096, V = 151 936, L = 36, 8.19 B params with untied
embeddings), one packed micro-batch of **T = 65 536 tokens** (the brief's shape; the DEFAULT budget is
T = 16 384 — scale the T-proportional rows by 1/4), bf16 autocast.

### 1.1 Default path (`use_remove_padding=True`, `use_fused_kernels=False`)

Trace of `FSDPEngineWithLMHead.forward_step` (`transformer_impl.py:1570-1612`) →
`prepare_model_inputs` (`:1155-1342`) → `self.module(**model_inputs)` → `prepare_model_outputs`
(`:1344-1568`) → `ppo_loss` (`losses.py:57-184`):

| # | Tensor | Where | dtype × shape | Bytes @ T=64k | Lives until | Avoidable? |
|---|---|---|---|---|---|---|
| 1 | `input_ids_rmpad`, `position_ids_rmpad`, `input_ids_rmpad_rolled`, `temperature_rmpad` | `transformer_impl.py:1198-1213` | int64 (1,T) ×3, fp32 (1,T) | 1.8 MB | micro-batch | no (tiny) |
| 2 | checkpointed decoder-layer inputs | HF gradient checkpointing (`:340`) | bf16 (T,H) × 36 | **19.3 GB** | backward | only by changing the checkpointing policy (§5.6) |
| 3 | one layer's recompute transients (qkv, attn out, gate/up, act, down) | HF layer forward | bf16 ≈ T × 102 KB | 6.7 GB | one layer | no |
| 4 | final hidden states → `lm_head` input | HF `Qwen3ForCausalLM.forward` | bf16 (1,T,H) | 0.54 GB | backward | no |
| 5 | **`output.logits`** | HF `lm_head`, read at `transformer_impl.py:1400` | **bf16 (T, V)** | **19.9 GB** (4.98 GB @ 16k) | until backward of the CE | **yes**: fused linear-CE (`use_fused_kernels`), or selected-token logits (`logits_to_keep=idx`, §8-C) |
| 6 | temperature scaling | `:1404-1408` via `_scale_logits_by_temperature` (`:96-104`) | — | **0** at `temperature=1.0`; otherwise one more (T,V) bf16 copy = 19.9 GB (`logits / temperature...` is out-of-place) | — | in-place div (open PR #7944) |
| 7 | `logprobs_from_logits` → flash-attn Triton `cross_entropy_loss` | `torch_functional.py:95-101,129` | fp32 (T) losses + fp32 (T) lse + fp32 (T) z_losses | 0.8 MB | — | — |
| 8 | CE autograd save | flash-attn 2.8.3 `cross_entropy.py` (fetched; `lse` buffer at line 192; backward kernel recomputes `probs = exp(logits - lse)` at line 137 and with `inplace_backward` writes `dlogits` into `logits`) | aliases #5 | 0 extra | backward | — |
| 9 | `dlogits` | CE backward | aliases #5 when `inplace_backward=True` (`:1444-1448`: true iff `calculate_entropy` and `score_centering` are both off → DEFAULT) | 0 extra; **+19.9 GB** when entropy is on in the update | lm_head backward | yes (fused kernels never form it) |
| 10 | `lm_head` grads | autograd of `hidden @ W^T` | bf16 (T,H) d_hidden 0.54 GB; bf16 (V,H) dW 1.24 GB; fp32 reduce-scatter staging 2.49 GB | 4.3 GB | reduce-scatter | no |
| 11 | `log_probs` nested → `no_padding_2_padding` (`losses.py:59`, `padding.py:97-141`) | per-sequence `F.pad` + `torch.stack` | fp32 (bsz, max_resp) | ≤ 0.26 MB | loss | no (tiny, but it is a Python loop over sequences, `padding.py:135-138`) |
| 12 | `old_log_probs`, `advantages`, `response_mask` `.to_padded_tensor()` | `losses.py:103` | fp32 (bsz, max_resp) ×3 | ≤ 0.8 MB | loss | no |
| 13 | ratio chain: `negative_approx_kl`, `ratio`, `pg_losses1/2/3`, `clip_pg_losses1/2`, `pg_losses` | `core_algos.py:1336-1361` | fp32 (bsz, max_resp) × ~9 | ≤ 2.4 MB | loss | no (tiny) |

So at the DEFAULT T = 16 384 the live set during the LM-head backward is ≈ 4.98 GB logits/dlogits
(in place) + 4.8 GB checkpointed inputs + 1.7 GB recompute transient + ~2.5 GB head grads +
16.4 GB fp32 param/grad/Adam shards (8.19 B × 16 B / 8 ranks) + ≤ 1.3 GB unsharded bf16 layers ≈
**31 GB**, logits ≈ 16 % of the peak. At T = 64k the same path needs ≈ 19.9 + 19.3 + 6.7 + 4.3 +
16.4 + 1.3 ≈ **68 GB** — the logits are the single largest tensor and the reason nobody runs the
default path at 64k on an 80 GB part. Numbers are arithmetic; fragmentation (verl runs with
expandable segments, `engine_workers.py:886`) is not modelled.

Also note the FLOP side: the vocab projection is `2·T·H·V = 8.2e13` FLOP at T = 64k, i.e. ≈ 6 % of
the micro-batch forward (`2·params·T ≈ 1.07e15` plus ≈ 3.2e14 of causal attention at 8k
sequences). Memory, not FLOPs, is what the logits cost.

### 1.2 Fused path (`use_fused_kernels=True`, OPT-IN; backend `triton` default from `fused_kernel_options`)

`apply_monkey_patch` swaps the model forward for `forward_with_triton_backend`
(`monkey_patch.py:282-284`, `dense_common.py:139-203`), which calls `linear_cross_entropy(hidden,
lm_head.weight, rolled_labels, temperature, "none")` (`dense_common.py:189-195`) →
`kernels.efficient_entropy_forward` (`kernels.py:569`). Live tensors replacing rows 5-9:

| Tensor | Where | Bytes @ T=64k |
|---|---|---|
| `logprobs`, `entropy`, `maximum`, `accumulate_and_entropy_b` | `kernels.py:605-620` fp32 (T) ×5 | 1.3 MB |
| per-split partials `_max`, `_accu`, `_entropy_b` | `kernels.py:623-629` fp32 (T, V/1024=149) ×3 | 117 MB |
| saved for backward: `hidden, weight, labels, _maximum, _accumulate, _entropy_b` | `linear_cross_entropy.py:183` | 0.54 GB + 1.24 GB (aliases) |
| backward `d_hidden` (T,H) + `d_weight` (V,H) in `hidden.dtype` (default `_Split_Dlogits_N`, `kernels.py:168,1562-1567`) | 1.8 GB |

The (T,V) tensor is never formed; the price is that the backward re-does the `hidden @ W^T` tiles
(`kernels.py:1583` comment: "Triton doesn't materialize d_logits at all"), i.e. +1 vocab matmul =
+8.2e13 FLOP ≈ +2 % of a forward+backward micro-batch. The `torch` backend
(`experimental/torch_functional.py:36-58`) is a 512-row chunked reference: each chunk upcasts to
fp32 (`:44`, 512 × V × 4 B = 311 MB), **always** computes softmax + entropy even when nobody asked
(`:46-47`), and the backward builds a one-hot `zeros_like(logits).scatter_` per chunk (`:79`).
Note the engine always reads `output.entropy` only if `calculate_entropy` (`transformer_impl.py:1385`),
so with the Triton backend the entropy epilogue is also computed unconditionally (`kernels.py:613`).

Why it is not the default: PR #6832 ("Enable Fused Kernels by Default", open since 2026-06-24,
updated 2026-10-24) has sat behind a string of correctness fixes still open against the kernel —
#7439 (negative logits), #8378 (confident-label logprob/grad → exactly 0 once the margin passes
≈16.6 in fp32), #8090 (output shape), #8096/#8097/#8312 (FSDP2 DTensor lm_head: the fused forward
reads `self.lm_head.weight` directly, `dense_common.py:125,191`, bypassing the head's FSDP hooks),
#6110 (per-sample temperature; the engine asserts scalar temperature at
`transformer_impl.py:1174-1177`). The `calculate_sum_pi_squared` and `score_centering` features are
explicitly incompatible (`:1356-1365`).

### 1.3 Precision notes (not perf, but they shape what a "fix" may change)

* Logits are bf16 at the CE input; flash-attn's kernel upcasts internally and returns fp32 per-token
  values. There is no fp32 upcast of the (T,V) tensor by default (opt-in in draft #8419).
* `entropy_from_logits` (`torch_functional.py:231-245`) runs on bf16 logits and returns **bf16**
  entropy (softmax/logsumexp keep the input dtype); the trainer `.float()`s it afterwards
  (`ray_trainer.py:1318`). The chunked variant upcasts (`:266`) but is off by default.

## 2. Forward passes per step

Shape: 8×H100, Qwen3-8B, 1024 sequences × 8k tokens = 8.39 M tokens/step → 1.05 M tokens per GPU
→ **64 micro-batches of 16k per GPU per pass** at the default budget. Per-token forward cost ≈
2 × 8.19e9 + attention (2·S·H·L at S = 8k) ≈ 18.8 GFLOP → one forward pass over the step's data ≈
1.97e16 FLOP per GPU ≈ 49 s at 400 TFLOP/s (≈40 % MFU). "1F" below = one such pass.

### 2.1 V1 trainer (`verl/trainer/ppo/v1/trainer_base.py`, PRIMARY)

`_step_once` (`trainer_base.py:563-613`): sample → reward → `_balance_batch` → old_log_prob → [ref]
→ [values] → adv → [update_critic] → update_actor.

| Pass | Condition | Code | Cost | Entropy? |
|---|---|---|---|---|
| old_log_prob (actor, forward-only) | always, unless `algorithm.rollout_correction.bypass_mode=True` (then `old_log_probs = rollout_log_probs`, `trainer_base.py:1643-1649`) | `:1651-1659` → `ActorRolloutRefWorker.compute_log_prob` (`engine_workers.py:756-759`) → `infer_batch` | **1F** | yes, hard-coded `calculate_entropy=True` (`:1654`) for the `actor/entropy` metric (`:1678-1685`) |
| ref log-prob | `use_reference_policy` (KL in loss or in reward) | `:1696-1718`; `ref_in_actor` (LoRA) reuses the actor with adapters disabled | 1F (ref model), forced CPU-offload streaming (§5.3) | no (`:1700`) |
| critic values | `use_critic` (GAE) | `:1722` | 1F (critic) | — |
| critic update | `use_critic` | `:1805-1819` → `train_mini_batch` | `ppo_epochs × (1F + 1F recompute + 2F backward)` = 4F per epoch | — |
| actor update | always after `critic_warmup` | `:1828-1865` → `train_mini_batch` (`engine_workers.py:276-372`) → `train_batch` per mini-batch (`:376-428`) → `forward_backward_batch` (`transformer_impl.py:728-781`) | **4F per epoch** (1 fwd + 1 ckpt recompute + 2 bwd); default `ppo_epochs=1` | only if `calculate_entropy or entropy_coeff != 0` (`:1832-1834`) → DEFAULT no |
| validation | every `test_freq` | `_validate` (`:1032`) → generation only | 0 trainer passes | — |

DEFAULT GRPO-style run (no critic, no KL): **5F per step on the actor** = old_log_prob (1F) +
update (4F). The minimum for PPO-clip semantics with k ≥ 2 mini-batches is 1F + 3F·epochs
(no activation recompute); the floor for a strictly on-policy single-update step is **3F**. So the
default spends 1.67× the floor: 20 % on the old-log-prob recompute, 20 % on checkpoint recompute.
Default PPO (GAE critic enabled by `adv_estimator: gae`, `ppo_trainer.yaml:78`) adds 1F + 4F on an
8B critic. The ref pass (KL users) adds 1F plus the H2D streaming of §5.3.

### 2.2 V0 trainer (`ray_trainer.py`, secondary)

Same structure: `old_log_prob` (`ray_trainer.py:1584-1615`, entropy hard-coded at `:1300`), ref
(`:1618-1622`), values (`:1625-1628`), update_critic (`:1677-1681`), update_actor
(`:1689-1690`); bypass mode at `:1575-1583`. Validation (`:596-640`) sets
`recompute_log_prob: False` and generates only.

### 2.3 Which passes are avoidable

1. **old_log_prob when the step is strictly on-policy.** With `ppo_epochs=1` and exactly one
   mini-batch (`ppo_mini_batch_size·n == train_batch_size·n`), the update pass sees
   `log_prob == old_log_prob` up to kernel noise, so `ratio ≡ 1`, no clipping triggers and the
   gradient equals `-A·∇log π` with `old_log_prob = log_prob.detach()`. Nothing in the trainer
   detects this (`grep -n "num_mini\|on_policy\|skip.*old_log"` over `ray_trainer.py`,
   `trainer_base.py`, `engine_workers.py` finds only the `num_mini_batch` plumbing at
   `engine_workers.py:289-298`). The only existing skip is `bypass_mode`, which substitutes the
   *rollout* log-probs (a different estimator with IS/truncation, `rollout_corr_helper.py:1109-1139`).
   Consumers that need `old_log_probs` before the update and therefore gate such a skip:
   `apply_kl_penalty` (`ray_trainer.py:1639-1643`, `:101`), rollout-correction IS metrics
   (`:1650-1660`, `trainer_base.py:1765-1770`), the path-variance advantage proxy (`:272-273`),
   `calculate_debug_metrics` (`:1611-1615`). Saving: **1F of 5F = 20 % of actor time** in that
   regime (the entropy metric moves into the update pass under `no_grad`). See §8-B.
2. **Gradient-checkpoint recompute (1F of 4F per update).** All-or-nothing per decoder layer
   (`transformer_impl.py:339-340`); no selective / every-k policy in the FSDP engine (torchtitan's
   engine has `activation_checkpoint: "selective"`, `config/engine.py:554`). See §5.6 and §8-D.
3. **Vocab projection on prompt / tool / masked tokens** in every pass (the LM head and CE run over
   all T packed tokens, `transformer_impl.py:1400,1445-1449`; only response rows are consumed,
   `padding.py:135-138`). Not a full pass but 100 % of the (T,V) traffic for rows that are
   discarded; Megatron has an opt-in for this (#7686), FSDP has none. See §8-C.
4. **n-fold prompt recompute in GRPO** (prompt of each group computed `rollout.n` times in every
   pass): covered by open #8300 (shared-prefix packing for the FSDP engine) and #6689 (prefix tree
   via MAGI attention); not pursued here.

## 3. Padding, packing and micro-batching

### 3.1 How a micro-batch is formed (DEFAULT rmpad path)

1. Driver (`trainer_base.py` / `ray_trainer.py`): `left_right_2_no_padding` (`verl/workers/utils/padding.py:23-94`)
   unpads the `(bsz, prompt+resp)` batch into jagged nested tensors (`input_ids`, `position_ids`,
   `loss_mask`); `response_mask`, `old_log_probs`, `advantages` stay padded `(bsz, max_resp)`.
   `_balance_batch` (`ray_trainer.py:1182`, `trainer_base.py:1609`, `trainer.balance_batch` on by
   default) Karmarkar-Karp-balances the sequences across DP ranks before dispatch.
2. Worker `train_mini_batch` (`engine_workers.py:276-372`) slices mini-batches with
   `tu.make_iterator` (`tensordict_utils.py:559-612`); per mini-batch it all-gathers the per-sequence
   token counts (`:326-334`) for the MFU estimate.
3. `forward_backward_batch` (`transformer_impl.py:728-781`): all-reduces `batch_num_tokens`
   (`:734-738`, one collective + one `.item()`), then `prepare_micro_batches` (`engine/utils.py:92-130`):
   * `use_dynamic_bsz=True` (OPT-IN; of the 50 shipped GRPO/GSPO/PPO example scripts 25 set it
     true, 14 set it false, the rest leave it to an env var or the default): `rearrange_micro_batches` (`seqlen_balancing.py:348-504`) — `num_micro_batches =
     min(num_groups, ceil(total_tokens / max_token_len))` (`:398`), **all-reduce MAX across DP** so every
     rank runs the same count (`:402-406`), KK partition on the workload `Σ L_i²`-style
     `calculate_workload` (`:452`), re-partition with one more micro-batch until every micro-batch's
     token sum ≤ `max_token_len` (`:444-462`, with an all-reduce MIN per iteration), then order
     "small at both ends" (`:488-496`). `max_token_len = max_token_len_per_gpu × sp_size` (`utils.py:111`).
   * `use_dynamic_bsz=False` (DEFAULT): `tu.chunk_tensordict(data, N / micro_batch_size_per_gpu)`
     (`utils.py:123-129`): a fixed number of sequences per micro-batch regardless of length, so the
     token count per micro-batch swings with the response-length distribution (8k-max responses: a
     4-sequence micro-batch spans 4×(prompt+~200) … 4×(prompt+8k) tokens). Users must size
     `ppo_micro_batch_size_per_gpu` for the worst case, which is why the recipes that care about
     throughput switch to dynamic.
4. `prepare_model_inputs` (`transformer_impl.py:1194-1284`): `input_ids.values()` → `(1, T)`,
   `position_ids` restart per sequence, `input_ids_rmpad_rolled = roll(-1)` (`:1213`), optional
   static pad to a `pad_to_length_bucket` (OPT-IN, `:1219-1225`, `pad_packed_inputs`
   `utils.py:66-89`), Ulysses pad to a multiple of `sp_size` and slice (`:1230-1256`,
   `ulysses.py:282-334`), `attention_mask=None`. Only `input_ids`/`position_ids` go to the model
   (`:1268-1272`); `cu_seqlens` are passed only when the model forward signature has a `cu_seqlens`
   parameter (`:598-600`, `:1273-1284`), which stock HF models do not.

**Padding waste on the default path is ≈ 0 tokens**: the only pads are ≤ `sp_size − 1` tokens per
micro-batch for Ulysses (`ulysses.py:291`) and the opt-in bucket pad. The waste that remains is
*computational*, not padding: (a) all T rows go through the LM head/CE although only response rows
are read back (`no_padding_2_padding` slices `values[seq_offset − resp_len − 1 : seq_offset − 1]`,
`padding.py:135-138`); (b) the `rollout.n` copies of each prompt in a GRPO group are recomputed n
times (open #8300 / #6689 / RFC #8299, #6401).

### 3.2 Where responses are still padded to a max length

* `use_remove_padding=False` (OPT-IN): `prepare_model_inputs` pads the micro-batch to its own max
  length with `torch.nested.to_padded_tensor` (`transformer_impl.py:1300-1311`) and builds an
  `attention_mask` (`:1313-1315`); HF then un-pads/re-pads **inside every attention layer**
  (`_upad_input` at HF `modeling_flash_attention_utils.py:773,793`), and the LM head runs over
  `(bsz, max_len, V)`. Waste = `Σ(max_len − len_i) / (bsz·max_len)`; for 8k-max responses with a
  ~4k mean that is ≈ 50 % of the vocab projection and of every layer.
* The loss side is always padded: `ppo_loss` converts `log_probs`/`entropy` to `(bsz, max_resp)`
  (`losses.py:59-62`, `:103`) and the ratio chain runs on `(bsz, max_resp)` (`core_algos.py:1336-1361`).
  These are fp32 `(bsz, max_resp)` tensors — ≤ 2.4 MB at 64k tokens — so the waste is time in
  ~12 tiny elementwise kernels and the Python per-sequence loop in `no_padding_2_padding`
  (`padding.py:132-140`: one `F.pad` per sequence + `torch.stack`), not memory.
* Critic: `FSDPEngineWithValueHead` inherits the rmpad `prepare_model_inputs`; its yaml default is
  rmpad (`_generated_ppo_trainer.yaml:693`) although the dataclass default is `False`
  (`workers/config/critic.py:270`).
* Ref: rmpad follows the shared model config (`engine_workers.py:623`).
* Multi-turn: tool-response tokens are masked out of `loss_mask` by the agent loop but still flow
  through every layer and the LM head (no path skips them).

### 3.3 Ulysses SP (`ulysses_sequence_parallel_size > 1`, OPT-IN)

Inputs are padded to a multiple of `sp_size` and each rank takes a contiguous `T/sp` slice
(`ulysses.py:302-334`); per attention layer `_ulysses_flash_attention_forward`
(`monkey_patch.py:119-155`) does two all-to-alls (`gather_seq_scatter_heads`,
`gather_heads_scatter_seq`) **and an all-gather of `position_ids`** (`:139-142`) so that HF can
re-derive `cu_seqlens` from them — the TODO at `:135-137` says so; open #7137 replaces this with
explicit `cu_seq_lens` for the SP path only. Outputs are all-gathered back and unpadded
(`gather_outputs_and_unpad`, `ulysses.py:247-279`; `_gather_and_unpad_packed`,
`transformer_impl.py:684-693`). GQA with `num_kv_heads < sp_size` repeats KV heads
(`monkey_patch.py:126-128`), i.e. SP=8 on an 8-KV-head model is free but SP=16 doubles KV traffic.

### 3.4 Per-layer varlen preparation and its CPU syncs (DEFAULT path, SP=1)

Because verl passes only `position_ids`, HF's `_flash_attention_forward` (fetched transformers
5.16.1 `modeling_flash_attention_utils.py`) takes the `is_fa_with_position_ids` branch (`:766,
:796-800`) and calls `prepare_fa_kwargs_from_position_ids` **in every decoder layer**: a
`(position_ids == min).nonzero()` (`:479`, a D2H sync), `cu_seq_lens_q.diff().max()` (`:492`) and
`max_seqlen_q.item()` in `flash_kwargs` (`:681`, a second sync). For Qwen3-8B that is 2 syncs ×
36 layers = **72 host syncs per micro-batch forward**, and the gradient-checkpoint recompute
repeats them in backward: ≈ 144 per training micro-batch, vs the 5 metric `.item()`s that open
#7096 targets. Each sync stops CPU run-ahead, so the next layer's kernels (and FSDP1's next
all-gather, which overlaps only through run-ahead since `forward_prefetch=False`) are launched
late; the cost is launch-latency-bound — my estimate is 1–3 % of a 16k-token micro-batch on an 8B
model, more for small models or small micro-batches (to be measured with nsys). verl already owns
`packed_cu_seqlens` (`transformer_impl.py:1204`) and could pass `cu_seq_lens_q/k` and
`max_length_q/k` once per micro-batch through `**kwargs` (`FlashAttentionKwargs`, HF `:585-587`),
which skips the per-layer derivation (`:797-804`).

## 4. Entropy

| Pass | `calculate_entropy` | Implementation | Cost |
|---|---|---|---|
| old_log_prob (forward-only) | **always True** (`trainer_base.py:1654`, `ray_trainer.py:1300`), only for the `actor/entropy` metric | `self.compute_entropy_from_logits` = `torch.compile(entropy_from_logits, dynamic=True)` (`transformer_impl.py:181-189`) over the full `(T,V)` bf16 logits (`:1415-1427`) | 3–4 reads of 19.9 GB at T=64k ≈ 60–80 GB of HBM traffic ≈ 20–25 ms vs ≈ 2.6 s of forward: **≈ 1 %** of the pass. With `use_torch_compile=False` (or when compile fails) the eager version (`torch_functional.py:243-244`) materializes `softmax` and `pd·logits`: **+2 × 19.9 GB** at 64k (+10 GB at the 16k default). The chunked variant (`:248-270`, OPT-IN) bounds that at `chunk_size × V × 4 B` = 1.2 GB. |
| actor update | `calculate_entropy or entropy_coeff != 0` (`trainer_base.py:1832-1834`) → DEFAULT **off** | same function, inside autograd | turning it on flips `inplace_backward` off (`:1444`) → the CE backward allocates a second `(T,V)` `dlogits` (+19.9 GB at 64k); `entropy_checkpointing` (OPT-IN) recomputes the entropy in backward instead of saving the compiled graph's intermediates; open #6519 wraps the metric-only case in `no_grad` |
| ref | False (`trainer_base.py:1700`) | — | — |
| fused kernels | — | the Triton epilogue always produces entropy (`kernels.py:613`); the torch backend computes softmax+entropy per chunk unconditionally (`experimental/torch_functional.py:46-47`) | 1 extra softmax pass per chunk even when `calculate_entropy=False` |

Precision: on bf16 logits the compiled eager entropy returns bf16 (`softmax`/`logsumexp` keep dtype,
`torch_functional.py:243-245`); only the chunked path upcasts (`:266`). The `actor/entropy` metric
is therefore quantized to ~3 significant digits before the trainer's `.float()`
(`ray_trainer.py:1318`). Cheap fix: `logits.float()` inside the compiled function (fused by
inductor, no (T,V) fp32 materialization) — a one-liner, flagged in §7.

## 5. FSDP mechanics, offload and host syncs

### 5.1 Wrapping, sharding, precision

* FSDP1 (DEFAULT `strategy: fsdp`): `transformer_auto_wrap_policy` on the model's
  `_no_split_modules` (`fsdp_utils.py:102-160`), `FULL_SHARD` on a 1-D `fsdp` mesh of
  `world_size` when `fsdp_size=-1` (`engine/fsdp/utils.py:35-53,72-81`), `MixedPrecision(param=bf16,
  reduce=fp32, buffer=fp32)` (`transformer_impl.py:395-405`), `sync_module_states=True`,
  `use_orig_params=False`, `forward_prefetch=False` (`:441-453`). The root unit (embeddings, final
  norm, `lm_head`) is one flat param: 1.24 B + 1.24 B params → 2.5 GB bf16 unsharded for the whole
  micro-batch (FSDP1 keeps the root unsharded between forward and backward).
* FSDP2 (OPT-IN `fsdp2`): `fully_shard` per decoder layer plus separately selected targets
  (`fsdp_utils.py:540-604`, `_select_fsdp2_wrap_targets`), `MixedPrecisionPolicy(param=bf16,
  reduce=fp32, cast_forward_inputs=True)` (`:459-461`), `reshard_after_forward` honoured per module
  (`:473`), `forward_prefetch` emulated with depth 1 (`fsdp_utils.py:593-604`). With untied embeddings
  the LM head is its own FSDP2 unit, which is what the fused kernels bypass (issue #8091, PRs
  #8096/#8097/#8312).
* Master weights: `model_dtype: fp32` → fp32 flat shards for actor, critic **and the forward-only
  ref**; per rank for Qwen3-8B/8 GPUs: params 4.1 GB, grads 4.1 GB, Adam 8.2 GB.
* Per micro-batch all-gather volume with `FULL_SHARD` + `reshard_after_forward=True`: the whole model
  in bf16 = 16.4 GB per micro-batch in forward and again in backward (FSDP1 re-gathers for
  backward); 64 micro-batches/pass → ≈ 1 TB per GPU per forward pass, ≈ 2 TB per update. At
  ≈ 300 GB/s NVLink all-gather that is 3.5 s / 7 s per step if fully exposed — normally hidden
  behind compute by stream overlap, but every host sync in §3.4 and §5.7 erodes the run-ahead that
  FSDP1 relies on for forward overlap. Larger micro-batches cut this linearly (another reason the
  logits-bound token budget matters).

### 5.2 Param / optimizer offload cost model (`param_offload=True, optimizer_offload=True`, the recipe setting)

`BaseEngineCtx._context_switch` (`engine/base.py:322-344`) loads on enter and offloads on exit;
`train_mini_batch` sets `disable_auto_offload=True` on each inner `train_batch`
(`engine_workers.py:338-343`), so the train-mode switch happens **once per update**, not per
mini-batch. Per step per rank (Qwen3-8B / 8):

| Event | Code | H2D | D2H | Notes |
|---|---|---|---|---|
| compute_log_prob enter/exit | `EngineEvalModeCtx` → `load_fsdp_model_to_gpu` / `offload_fsdp_model_to_cpu` (`fsdp_utils.py:206-222,169-192`) | 4.1 GB (fp32 shards) | 4.1 GB + `empty_cache` | `flat_param_to(..., non_blocking=True)` on **pageable** host memory → effectively synchronous |
| update_actor enter/exit | `EngineTrainModeCtx` + `load_fsdp_optimizer` (`:244-252`) | 4.1 + 8.2 GB | 4.1 + 8.2 GB + `empty_cache` | optimizer state moved tensor-by-tensor (`.to(device, non_blocking=True)` per state tensor: ~1 k small copies) |
| update_weights (naive) | `get_per_tensor_param` → `load_fsdp_model_to_gpu` (`transformer_impl.py:994-995`); `engine_workers.py:870-874` | 4.1 GB | 4.1 GB + `synchronize` + 2× `empty_cache` (`:828,:873`) | |
| **total** | | ≈ 20.5 GB | ≈ 20.5 GB | ≈ 41 GB / ~12 GB/s pageable PCIe ≈ **3–4 s per step** (≈ 1.5 % of a ~250 s step); pinned staging would halve it, overlap would hide it (draft #5081 non-blocking copies, #7804 optimizer-around-step) |

### 5.3 The reference model streams fp32 shards from the host on every micro-batch

`forward_only=True` (ref, `dp_ref.yaml:14`) forces FSDP1 `CPUOffload(offload_params=True)` /
FSDP2 `CPUOffloadPolicy(pin_memory=True)` and disables the manual offload path
(`transformer_impl.py:433-438, 463-467, 855-857`). Consequences:

* Every FSDP unit's shard is copied H2D before each all-gather, for every micro-batch: 4.1 GB per
  rank per micro-batch (fp32, because `model_dtype: fp32` is the yaml default and the "bf16 when
  forward_only" branch at `transformer_impl.py:264-266` only triggers for `model_dtype=None`).
  64 micro-batches/pass → **262 GB of H2D per rank per step**, ≈ 10–13 s at 20–25 GB/s pinned PCIe
  against ≈ 49 s of ref compute (8B, 1.05 M tokens/GPU @ 400 TFLOP/s): +20–27 % on the ref pass if
  not overlapped; FSDP1 overlaps it only as far as the CPU runs ahead (§3.4 syncs again).
* Host RAM: 8 ranks × 4.1 GB = 33 GB pinned per node for a model that only ever computes in bf16
  (16.4 GB would do); the shards are all-gathered as bf16 anyway (`MixedPrecision`).
* Reach: 57 of the 121 shipped `examples/**/*.sh` + `recipe/**/*.sh` scripts set `use_kl_loss=True`
  (4 more set `use_kl_in_reward=True`), i.e. roughly half of the shipped recipes run this pass.
* Open #7182 makes the forced CPUOffload configurable (same symptom, reported "40–70× slower ref
  forward than actor forward for a 4B model"); it does not touch the fp32 dtype. A bf16
  `model_dtype` for `forward_only` engines (halving the stream and the host RAM) plus per-pass
  residency (load once per `compute_ref_log_prob`, like the actor's `param_offload`) is the full fix.

### 5.4 Gradient communication

`_gradient_sync_context` (`transformer_impl.py:695-726`) defers reduce-scatter to the last
micro-batch only when `use_no_sync_for_gradient_accumulation` is true; the yaml ships **false**
(`engine/fsdp.yaml:43`) while the dataclass says True (`config/engine.py:290`) and the code's
fallback is True (`:706-710`). So hydra users reduce-scatter every micro-batch: fp32
(`reduce_dtype`) 32.8 GB → ≈ 28.7 GB received per rank per micro-batch, ≈ 1.8 TB per step per
GPU at 64 micro-batches, overlapped with backward by FSDP1's post-backward stream except the tail
(the 2.5 GB root flat param's reduce-scatter after the last layer, ≈ 8 ms × 64 ≈ 0.5 s/step).
`no_sync=True` would keep **unsharded fp32 grads (32.8 GB per rank)** alive across the
accumulation — infeasible at 8B/80 GB, which is presumably why the yaml says false; the lever
that is actually available is `mixed_precision.reduce_dtype=bf16` (halves bytes, precision
trade-off), opt-in today. Test: `tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py`.

### 5.5 Clipping and the optimizer step

`optimizer_step` (`transformer_impl.py:792-838`): FSDP1 `clip_grad_norm_` (one all-reduce),
`torch.isfinite(grad_norm)` (implicit sync) and `grad_norm.item()` — 2 host syncs per mini-batch,
fine. `lr_scheduler_step` once per update (`engine_workers.py:406-411`).

### 5.6 Activation checkpointing is all-or-nothing

`module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
(`transformer_impl.py:339-340`) recomputes **every decoder layer in full**: +1F per update = 25 % of
the update FLOPs, ≈ 20 % of the actor step at `ppo_epochs=1`. `HFModelConfig.enable_gradient_checkpointing`
is a bool (`config/model.py:117`); there is no every-k or op-selective policy in the FSDP engine
(grep for `selective|every_n|checkpoint_layers` in `verl/workers/engine/fsdp/` finds nothing),
whereas the torchtitan engine has `activation_checkpoint: "selective"` (`config/engine.py:554`)
and Megatron users reach `recompute_num_layers` through `override_transformer_config`. Memory
arithmetic (Qwen3-8B, T = 16k, baseline peak ≈ 31 GB from §1.1): full per-layer activations ≈
T × 102 KB ≈ 1.7 GB/layer; checkpointing every 2nd layer adds 18 × 1.7 ≈ 31 GB (peak ≈ 62 GB,
fits with the rollout engine asleep) and removes half the recompute (−12.5 % update FLOPs, ≈ −10 %
actor step); op-level SAC that keeps only matmul outputs would add ≈ 46 GB at 16k (too much) but
≈ 23 GB at 8k. Activation offload (`enable_activation_offload`, OPT-IN,
`activation_offload.py:500-558`) is the orthogonal lever (host staging per layer).

### 5.7 `torch.compile`, `empty_cache` and host syncs per step

* `torch.compile` is applied to `entropy_from_logits` only (`transformer_impl.py:181-189`); the
  model is eager (open #3870 compiles the decoder layers, claims 5–10 %).
* `empty_cache` is **not** in the micro-batch loop: `offload_fsdp_model_to_cpu` calls it on each
  eval/train context exit when offloading (`fsdp_utils.py:191-192,201-202`), `update_weights` calls
  it twice plus a `synchronize` (`engine_workers.py:828,872-873`), `aggressive_empty_cache` runs at
  init only (`:744`) → ≈ 5 device-wide syncs per step. Fine.
* Host syncs per training micro-batch on the default path: `loss.detach().item()`
  (`transformer_impl.py:1608`), 3 in `compute_policy_loss_vanilla` (`core_algos.py:1372-1374`),
  `Metric(value=pg_loss)` → `.item()` (`losses.py:133` → `metric/utils.py:110`) = **5** (+2 with
  entropy/KL metrics; open #7096 defers them), plus **≈ 144 from HF's per-layer varlen prep**
  (§3.4). Per mini-batch: `batch_num_tokens.item()` (`:738`), and in `rearrange_micro_batches`
  one `.item()`, two `.cpu().tolist()`, three all-reduces each followed by `.item()`
  (`seqlen_balancing.py:394,406,418-424,430-433,457-459`) — ≈ 6 syncs + 3 collectives, once per
  mini-batch, fine.
* `forward_step` moves each micro-batch to the device with `micro_batch.to(get_device_id())`
  (`transformer_impl.py:1573`) from pageable memory: ≤ 1 MB per micro-batch, negligible.

## 6. Weight-sync preparation on the trainer side

`ActorRolloutRefWorker.update_weights` (`engine_workers.py:785-886`) → `FSDPEngine.get_per_tensor_param`
(`transformer_impl.py:981-1065`):

1. FSDP1 (DEFAULT) or LoRA: `load_fsdp_model_to_gpu` first (`:992-995`; H2D 4.1 GB if offloaded);
   FSDP2 non-LoRA skips staging because `state_dict()` only collects DTensor refs.
2. `params = self.module.state_dict()` (`:1020`) under `SHARDED_STATE_DICT` (`:491-496`) →
   DTensors in the **fp32 master dtype** (FSDP1 returns DTensors here because a `device_mesh` is
   passed at `:449`; inferred from the consumer handling only `DTensor` at `:1036` — torch internals
   not re-verified); `convert_weight_keys` renames (`:1022`).
3. Generator: `param.to(device, non_blocking=True).full_tensor()` per parameter (`:1033-1039`) —
   **one NCCL all-gather per parameter** (≈ 400 for Qwen3-8B: 36 layers × 11 tensors + 3), each
   producing a full **fp32** tensor on every rank: 28.7 GB received per rank per sync (8.19 B × 4 B
   × 7/8), transient peak = the largest tensor, `embed_tokens`/`lm_head` 151 936 × 4096 × 4 B =
   **2.49 GB fp32** (1.24 GB if cast first). `unfuse_moe_params` (`engine/fsdp/utils.py:90-120`) adds
   one `.contiguous()` copy per expert slice for Transformers-5 packed MoE tensors.
4. Consumer boundary: the NCCL checkpoint engine views each tensor as bytes
   (`checkpoint_engine/base.py:565-591`) and copies them into a 2 GB bucket
   (`rollout.yaml:308`, `nccl_checkpoint_engine.py:378`) — **in fp32**, so the wire carries 2× the
   bytes the bf16 rollout needs; dtype conversion happens on the receiver. The LoRA-merged path
   casts to bf16 only *after* the gather (`:1086`), the delta-shard path casts the local shard
   before anything (`:946-948`) — the right pattern, not used by the full-weight path. The QAT
   quantizer path (`:1042-1062`) consumes the fp32 stream and must keep it.
5. After the sync: `to("cpu", model=True)` (D2H 4.1 GB), `synchronize`, `empty_cache`
   (`engine_workers.py:870-874`).

Copies per parameter on the trainer side: all-gather (1) + bucket copy (1); peak extra memory ≈
2.49 GB + 2 × 2 GB buckets on top of the resident shards and the resumed rollout weights. Casting
the DTensor's local shard to bf16 before `full_tensor()` halves the all-gather bytes (28.7 →
14.3 GB/rank), the transient (2.49 → 1.24 GB) and the bucket traffic; it is independent of the
sharded-transport redesign in open #8395 (no trainer-side all-gather at all) and of the veomni
receiver-dtype work in #8110.

## 7. Findings table

Shape for all costs: 8×H100, Qwen3-8B (V = 151 936, H = 4096, 36 layers), 1024 seqs × 8k tokens
per step (1.05 M tokens/GPU, 64 micro-batches of 16k per pass), 1F ≈ 49 s/GPU at 400 TFLOP/s.
"DEFAULT" = shipped yaml.

| # | Finding | Where (file:line) | Quantified cost | Proposed fix | Expected saving | Risk | Tests touching it | Open PRs / dup |
|---|---|---|---|---|---|---|---|---|
| F1 | DEFAULT path materializes full-vocab bf16 logits; `use_fused_kernels` is opt-in and the Triton kernel has open correctness issues | `transformer_impl.py:1400,1445-1449`; defaults `hf_model.yaml:61` | 4.98 GB at T=16k (16 % of a ≈31 GB peak), 19.9 GB at 64k (largest tensor); bounds `ppo_max_token_len_per_gpu` | (a) fix-and-benchmark bundle to make fused the default, or (b) F2 which shrinks the eager tensor | memory only (FLOPs of the head are 6 %); unlocks 2–4× larger micro-batches → fewer all-gathers (§5.1) | fused kernel precision/FSDP2 head hooks | `tests/models/test_model_forward_fused.py`, `tests/utils/test_linear_cross_entropy*.py`, `tests/models/test_engine.py` | #6832 (default), #7439, #8378, #8090, #8096/#8097/#8312, #6110 |
| F2 | LM head + CE run on every packed row although only loss rows are consumed | `transformer_impl.py:1400,1445`; `padding.py:135-138`; HF `modeling_qwen3.py:494-495` supports `logits_to_keep=idx` | prompt fraction of 8.2e13 FLOP and of the (T,V) tensor per micro-batch: 11 % (1k prompt/8k resp), 30–60 % for multi-turn with masked tool output | select loss rows (`response_mask`-rolled) before the head in both eager and fused paths; scatter log-probs back | −11…−60 % of head FLOPs and logits memory; forward-only budget can rise by the same factor | SP slicing bookkeeping; must keep `sum_pi_squared`/score-centering on full rows | `tests/models/test_fsdp_no_padding_on_gpu.py`, `tests/workers/test_packed_pad_to_length_on_cpu.py`, `tests/utils/test_padding_on_cpu.py` | #7686 (Megatron only, opt-in); RFC #7712 asks for *extra* selected-token outputs, different goal |
| F3 | old_log_prob recompute runs even when the step is strictly on-policy (`ppo_epochs=1`, one mini-batch) | `trainer_base.py:585-587,1651-1659`; `ray_trainer.py:1584-1586`; no gate anywhere | 1F of 5F = **20 % of actor time** in that regime (+ the entropy pass) | trainer gate on (`epochs==1`, `num_mini_batches==1`, no `use_kl_in_reward`, no rollout-correction IS, no adv estimator reading `old_log_probs`); `ppo_loss` uses `log_prob.detach()` when `old_log_probs` is absent; entropy metric from the update pass under `no_grad` | −20 % step time for single-update runs (3 of 50 explicit recipes; `fsdp_turbo` recipes), 0 otherwise | behaviour change must be opt-in/auto with logging; ratio ≡ 1 exactly instead of 1±1e-2 | `tests/trainer/*`, `tests/workers/test_engine_return_model_output_on_cpu.py` | none (bypass_mode `rollout_corr_helper.py:1109` substitutes rollout log-probs, a different estimator) |
| F4 | Activation checkpointing is per-layer full recompute, no every-k / selective policy | `transformer_impl.py:339-340`; `config/model.py:117` | +1F per update = 25 % of update FLOPs ≈ **20 % of the actor step**; every-2 needs +31 GB at 16k (fits 80 GB with rollout asleep), SAC +23 GB at 8k | `gradient_checkpointing_mode: full/every_k/selective_op` via HF `context_fn` + `create_selective_checkpoint_contexts` | −10 % (every-2 at 16k) to −20 % (no recompute when memory allows) of actor step | memory; HF `checkpoint` kwargs plumbing; interaction with activation offload (`activation_offload.py:544-551` disables HF ckpt) | `tests/utils/test_activation_offload.py`, `tests/models/test_engine.py` | none for FSDP (torchtitan engine has `activation_checkpoint: selective`) |
| F5 | Ref/forward-only engine streams **fp32** shards from host on every micro-batch (forced CPUOffload) | `transformer_impl.py:433-438,463-467,855-857,264-266`; `engine/fsdp.yaml:24` | 262 GB H2D per rank per step, +20–27 % on the ref pass worst case; 33 GB pinned host RAM/node (2× needed); hits 57/121 shipped recipes | `model_dtype=bf16` when `forward_only`; load once per pass (reuse `param_offload` semantics) instead of per-micro-batch streaming | ref pass −20…−40 %; host RAM −50 % | models that must keep fp32 buffers (#8189) — cast params only | `tests/models/test_engine.py` (ref path), `tests/workers/config/test_engine_config_on_cpu.py` | #7182 (configurable offload, open since 2026-07, no dtype change) |
| F6 | HF recomputes varlen `cu_seqlens` per decoder layer with 2 host syncs each because verl passes only `position_ids` | `transformer_impl.py:1273-1284` (cu_seqlens only if the forward signature has `cu_seqlens`); HF `modeling_flash_attention_utils.py:479,492,681,796-800` | 72 syncs per forward, ≈ 144 per training micro-batch (vs 5 metric syncs); est. 1–3 % at 16k/8B, more for small models | pass `cu_seq_lens_q/k`, `max_length_q/k` once per micro-batch via `FlashAttentionKwargs` (`**kwargs`) | 1–3 % step time, restores FSDP1 forward all-gather run-ahead | models whose forward does not forward `**kwargs` to attention (fallback to current path) | `tests/models/test_transformer.py`, `tests/models/test_transformers_ulysses.py`, `tests/utils/test_attention_utils.py` | #7137 (SP path only, still per layer) |
| F7 | Entropy always computed in the old-log-prob pass, in bf16 | `trainer_base.py:1654`; `torch_functional.py:243-245` | ≈ 1 % of the pass (compiled); +2×(T,V) if `use_torch_compile=False`; metric quantized to bf16 | `logits.float()` inside the compiled fn; keep chunked path for no-compile | precision of `actor/entropy`; negligible time | none | `tests/utils/test_torch_functional*.py` | #7582 (chunking honoured in padded path), #6519 (no-grad metrics) |
| F8 | `use_no_sync_for_gradient_accumulation`: yaml false vs dataclass True; fp32 reduce-scatter per micro-batch | `engine/fsdp.yaml:43`, `config/engine.py:290`, `transformer_impl.py:706-710` | 28.7 GB/rank/micro-batch, ≈ 1.8 TB/step, mostly overlapped; `no_sync=True` would pin 32.8 GB unsharded fp32 grads | align the defaults and document; expose `reduce_dtype=bf16` as a named knob | small (tail reduce-scatter ≈ 0.5 s/step) | precision with bf16 reduce | `tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py` | none |
| F9 | Weight-sync gathers every parameter as **fp32** `full_tensor()` and buckets it in fp32 | `transformer_impl.py:1033-1039`; `checkpoint_engine/base.py:578`; `nccl_checkpoint_engine.py:378` | 28.7 GB received/rank/sync, 2.49 GB transient (embedding), 2× wire bytes | cast the DTensor local shard to bf16 before `full_tensor()` unless a quantizer is attached | −50 % gather bytes/transient/bucket traffic (≈ 0.05–0.1 s/sync, +1.2 GB headroom while rollout weights are resumed) | receivers that expect fp32 (QAT quantizer path keeps fp32) | `tests/utils/test_fsdp2_model_transfer_on_cpu.py`, `tests/checkpoint_engine/*` | #8110 (veomni receiver dtypes), #8395 (sharded transport, no gather) |
| F10 | Manual offload copies are pageable and synchronous | `fsdp_utils.py:187,220,240,252`, `engine/base.py:322-344` | ≈ 41 GB H2D+D2H per step per rank ≈ 3–4 s (≈ 1.5 %) | pinned staging + overlap with the next stage | ≈ 1 % | none | `tests/utils/test_fsdp2_model_transfer_on_cpu.py` | #5081 (draft, non-blocking), #7804 (optimizer around step) |
| F11 | `use_dynamic_bsz` defaults to False → fixed-count micro-batches sized for the worst case | `actor.yaml:21`, `engine/utils.py:123-129` | throughput loss depends on length variance; 23/50 recipes opt in | flip the default or warn when `ppo_micro_batch_size_per_gpu` is set with rmpad | recipe-dependent | config change | `tests/utils/test_seqlen_balancing.py`, `tests/utils/test_prepare_micro_batches_with_group_size.py` | #8201 (V1 mini-batch balancing), #5338 |
| F12 | 5 scalar `.item()` syncs per micro-batch in the loss | `transformer_impl.py:1608`, `core_algos.py:1372-1374`, `losses.py:133` | ≪ F6 | defer to end of mini-batch | < 1 % | none | — | #7096 (covers it) |
| F13 | Torch fused backend always computes softmax+entropy per chunk and builds a one-hot `(chunk,V)` in backward | `experimental/torch_functional.py:46-47,79` | +1 softmax pass per chunk; +311 MB transient per chunk | gate entropy on `needs_entropy`, use `index_add_`/gather for the label term | fused-torch backend only | — | `tests/utils/test_experimental_torch_functional_on_cpu.py` | #7235 (grads gating), #3765 |
| F14 | Update-pass entropy disables in-place CE backward → second (T,V) | `transformer_impl.py:1444` | +4.98 GB at 16k when `entropy_coeff != 0` | fold the entropy gradient into one backward pass (what the fused kernel does) | memory when entropy on | kernel work | `tests/utils/test_linear_cross_entropy.py` | #6519 (partial: no-grad when coef 0) |
| F15 | `unfuse_moe_params` copies every expert slice during sync | `engine/fsdp/utils.py:109-120` | 1 extra copy of each packed expert tensor (MoE only) | yield views when already contiguous | MoE sync time | — | — | none |

## 8. Top-5 contribution candidates from this slice

Ordered by (expected step-time win × breadth of users) with duplicates discounted.

### C1. Selective / every-k activation checkpointing for the FSDP engine (F4)

Design: add `model.gradient_checkpointing_mode: full | every_k | selective_op` (+ `gradient_checkpointing_k`)
to `HFModelConfig` (`config/model.py:117`) and resolve it in `FSDPEngine._build_module`
(`transformer_impl.py:339-340`). `every_k` wraps only every k-th decoder layer by replacing HF's
`gradient_checkpointing_enable` with a per-layer `torch.utils.checkpoint.checkpoint(use_reentrant=False)`
wrapper installed on the selected `_no_split_modules` instances (the same traversal
`activation_offload.py:524-537` uses). `selective_op` keeps HF's per-layer checkpoint but passes
`context_fn=partial(create_selective_checkpoint_contexts, policy)` through
`gradient_checkpointing_kwargs`, with a policy that saves `aten.mm`/flash-attn outputs and recomputes
norms, SiLU, RoPE and residual adds (recompute cost ≈ 0.1–0.2F instead of 1F). Interaction: when
`enable_activation_offload` is on, keep today's behaviour (`activation_offload.py:544-551`).
Benchmark: `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` on 8×H100 at `ppo_max_token_len_per_gpu`
∈ {8k, 16k}, modes {full, every_2, every_4, selective_op}; report `update_actor` time, `perf/mfu/actor`,
`perf/max_memory_allocated_gb` and loss-curve equality (checkpointing is numerically neutral).
Expected: −10 % actor step at 16k (every_2), −15…−20 % at 8k (selective_op or every_4).

### C2. Loss-row-only vocabulary projection, eager and fused, plus a decoupled forward-only budget (F1, F2)

Design: in `FSDPEngineWithLMHead.prepare_model_inputs` (`transformer_impl.py:1194-1284`) build
`keep_idx` = packed positions whose *next* token is a loss token (from `input_ids.offsets()`, the
prompt lengths and `response_mask`, i.e. the rows `no_padding_2_padding` keeps,
`padding.py:127-138`), after the SP slice so it is rank-local. Eager path: pass
`logits_to_keep=keep_idx` to the HF forward (index-tensor form supported in the pinned transformers
5.16.1, `modeling_qwen3.py:457,494-495`; older releases only take an int) so `output.logits` is `(1, K, V)`; gather labels with the same
index; compute CE/entropy on K rows; scatter into a `(T,)` fp32 zeros tensor before
`_gather_and_unpad_packed` (`:1451-1456`). Fused path: slice `hidden_states` with `keep_idx` before
`linear_cross_entropy` / `FusedLinearForPPO` in `dense_common.py:139-203` (add a `keep_idx`
kwarg next to `shift_labels`). Guard: fall back to full rows when `calculate_sum_pi_squared`,
`score_centering` or distillation top-k is requested (they need all rows, `:1412-1439`). Second
half: give `rollout.log_prob_max_token_len_per_gpu` an independent default (e.g. 2–4×
`ppo_max_token_len_per_gpu`) now that forward-only passes are not logits-bound. Benchmark: same
recipe with (a) GSM8K-style 0.3k/1k prompts and (b) an agentic multi-turn set with ≥ 40 % masked
tokens; report old_log_prob/update_actor time, peak memory, and bitwise equality of response
log-probs vs main (row-wise CE is independent of the dropped rows). Expected: −11 % to −60 % of the
head's time and of the logits tensor; 2–4× fewer micro-batches in the forward-only passes
(≈ −3…−8 % of those passes from all-gather and per-micro-batch overhead).

### C3. Forward-only engines in bf16 with per-pass residency (F5)

Design: in `FSDPEngine._build_module` (`transformer_impl.py:262-268`) resolve `model_dtype` to bf16
when `engine_config.forward_only` and the user left the yaml default (introduce `model_dtype: null`
in `engine/fsdp.yaml:24` so the existing `None → bf16 if forward_only` branch at `:264-266`
becomes live, keeping fp32 for the trainable engines); in `_build_fsdp_module` (`:433-438`,
`:463-467`) stop forcing `CPUOffload` for `forward_only` and instead honour `param_offload`
(load before `infer_batch`, offload after, via the existing `EngineEvalModeCtx`), which #7182 also
proposes — coordinate with it rather than duplicate: our increment is the dtype default plus the
pinned, once-per-pass load. Benchmark: Qwen3-8B GRPO with `use_kl_loss=True` on 8×H100; report
`ref` timing, host RSS, and ref log-prob equality (bf16 shards vs fp32 shards cast to bf16 at
all-gather are bitwise identical since the cast is the same rounding). Expected: ref pass
−20…−40 %, host RAM −16 GB/node.

### C4. Skip the old-log-prob pass when the update is strictly on-policy (F3)

Design: `actor.on_policy_skip_old_log_prob: auto|true|false` (default `auto`). In
`trainer_base._step_once` (`:585-587`) and `ray_trainer` (`:1584-1586`), `auto` resolves true iff
`ppo_epochs == 1`, `ppo_mini_batch_size·n == train_batch_size·n` (one optimizer step),
`algorithm.use_kl_in_reward` is false, `rollout_correction` is off, the advantage estimator does
not read `old_log_probs` (`ray_trainer.py:272`), and `calculate_sum_pi_squared` is off. When it
resolves true the trainer skips `compute_log_prob`; `ppo_loss` (`losses.py:98-107`) takes
`old_log_prob = log_prob.detach()` when the key is absent; the update pass is dispatched with
`calculate_entropy=True` and the entropy metric is produced under `no_grad` in
`prepare_model_outputs` (`transformer_impl.py:1415-1427`, with the `inplace_backward` decision at
`:1444` keyed on `entropy_coeff != 0` instead of on `calculate_entropy`). `calculate_debug_metrics`
(rollout vs actor log-prob diff) moves to the update pass. Benchmark: a single-mini-batch GRPO
run (`ppo_mini_batch_size = train_batch_size`), 8×H100 Qwen3-8B; report step time and reward
curves vs main over ≥ 100 steps. Expected: −20 % actor time, curves within seed noise (ratio is
exactly 1 instead of 1 ± kernel noise).

### C5. Two small sync/bandwidth fixes: once-per-micro-batch varlen kwargs, bf16-before-gather (F6, F9)

Design (a): in `prepare_model_inputs` (`transformer_impl.py:1273-1284`) always pass
`cu_seq_lens_q=cu_seq_lens_k=packed_cu_seqlens` and Python-int `max_length_q/k` (known on the host
from `input_ids.offsets()` without a sync) as `**kwargs` whenever the model's attention accepts
`FlashAttentionKwargs` (transformers ≥ 4.53 `flash_attention_forward`), extended with the SP-aware
boundaries that #7137 derives; HF then skips `prepare_fa_kwargs_from_position_ids` per layer
(`:797-804`). Design (b): in `get_per_tensor_param` (`:1033-1039`) and the LoRA-merged generator
(`:1086`) cast the local shard to bf16 before `full_tensor()` when no rollout weight quantizer is
attached. Benchmark: nsys count of `cudaStreamSynchronize`/`cudaMemcpy D2H` per micro-batch before
and after (expect 144 → ~8), step time on Qwen3-8B and on a 1.7B model (where launch latency is a
larger fraction); for (b) `update_weights` wall time and peak memory during the sync. Expected: (a)
1–3 % on 8B, 5–10 % on small models; (b) −50 % sync bytes and −1.2 GB transient.

## 9. Duplicates, roadmap alignment, and what this slice did not cover

* Open PRs that overlap with the findings and would need coordination rather than competition:
  #6832 (fused default), #7686 (Megatron response-only head), #7182 (ref CPUOffload), #7137
  (SP cu_seqlens), #7096 (deferred metrics), #3870 (torch.compile layers), #8300/#6689 (prefix
  sharing), #8395/#8088/#7552 (sharded weight transport), #7804/#5081 (offload timing), #8201
  (V1 mini-batch balancing), #8026/#8419 (precision), #8377/#8378 (confident-label log-probs).
  Searched by changed-file path and by title keyword over `gh/prs_open.json` (774 PRs, 2026-10-28).
* Roadmap #8186 (26Q4) has no Model-Engine throughput item for FSDP beyond low-precision
  training; the "Trainer" section only lists removing V0. C1–C5 therefore do not collide with a
  claimed roadmap line; C3 touches the roadmap's "cheaper inference-only forwards (ref, teacher,
  old_log_prob)" motivation under Low Precision Training without touching quantization.
* Not covered here (rollout agent's slice): TransferQueue sample flow, replay-buffer readiness,
  the hybrid-engine switch cost on the rollout side (`trainer_separate_async.py:417-434`), the
  transport side of weight sync (`checkpoint_engine/*` beyond the trainer-side generator), and the
  roadmap's async-rollout items (token-level interrupt, sample-level dispatch, session affinity,
  KV pooling/offload, fault tolerance).
* Megatron engine, for contrast: `_lm_head_logits_processor` (`megatron/transformer_impl.py:1151-1215`)
  divides logits in place (`:1171`), **clones** the full vocab-parallel logits when entropy is
  requested (`logits_bak = logits.clone()`, `:1177`) because `vocab_parallel_entropy` is
  destructive, and otherwise mirrors the FSDP structure; its fused path requires a uniform
  temperature (`:90-96`) and has the response-only head option in open #7686.
