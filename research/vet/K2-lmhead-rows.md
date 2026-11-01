# K2 "lmhead-rows": loss-row-only vocabulary projection on the DEFAULT FSDP path

Vetted 2026-10-29 (adversarial vetter). Clone: verl-project/verl main@2ab35292 (read-only). Sources read
in situ: transformers 5.16.1 (the bench image's pin, `.venv-cpu`) and 5.21.0 (`.venv-a9`), peft 0.19.1,
flash-attn 2.8.3 `flash_attn_interface.py` (fetched). Scripts: `research/vet/K2/k2_cpu_proof.py`.

## 1. Claim

Source: research/01-roadmap-demand.md §4 item 2 and §4.1 pick 1; research/04-engine-gpu-audit.md §1.1
rows 5-9, findings F1/F2, candidate C2; research/02-perf-pain-points.md themes D and F.

On the default actor/ref/old-log-prob path (`use_remove_padding=True`, `use_fused_kernels=False`,
V1 trainer) the FSDP engine calls the HF CausalLM forward without `logits_to_keep`, so the LM head
projects every packed token (prompt + response + tool observations + static/SP pad) into a bf16
`(T, V)` logits tensor (4.98 GB at the 16k default budget, 7.47 GB at BASELINE-A's 24576), and the
cross-entropy / entropy run on all T rows. Only the rows whose next token is a response token are ever
read (`no_padding_2_padding`, `response_from_nested`). HF supports a tensor-valued `logits_to_keep`
that slices the hidden states before the head; passing the loss-row index projects K rows instead of
T, leaves the consumed log-probs identical (row-wise ops), and removes the fraction `1 - K/T` of the
head FLOPs, the logits tensor and the CE/entropy work: ~6 % on GSM8K-style prompts, 33 % at
1k/2k, 85-90 % for agentic 16k/2k prompts (#8299 reports 3-16x prompt:response ratios).

## 2. Mechanism verification (file:line, clone main@2ab35292)

Call path, all in `verl/workers/engine/fsdp/transformer_impl.py` unless noted:

| step | evidence |
|---|---|
| forward | `forward_step` :1570 -> `prepare_model_inputs` :1155 -> `raw_output = self.module(**model_inputs, use_cache=False)` :1587-1590 -> `prepare_model_outputs` :1344 -> loss function |
| model kwargs | rmpad branch builds `model_inputs = {"input_ids", "attention_mask": None, "position_ids"}` :1268-1272; the only extras are `cu_seqlens`/`cu_seqlens_cpu` when the module's forward signature has `cu_seqlens` (probe :598, use :1273-1284) and fused-kernel args :1322-1339. `grep -rn logits_to_keep verl/` hits only `verl/models/transformers/dense_common.py:84,152` (fused forwards accept it and never use it). **Nothing passes `logits_to_keep` today.** |
| per-row labels / temperature | `input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, -1)` :1213; stored :1263-1264 as `(T,)` tensors, so any row subset can be gathered with one index |
| logits | `logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)` :1400; TP DTensor `full_tensor()` :1402-1403; temperature :1404-1408 |
| per-row consumers of the full tensor | `calculate_sum_pi_squared_from_logits(logits_rmpad)` :1413; entropy :1415-1427 (`compute_entropy_from_logits(logits_rmpad)` :1423, `torch_functional.py:231-245`, row-wise); `logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled, inplace_backward=not (calculate_entropy or score_centering))` :1444-1449 (`torch_functional.py:95-129`: flash-attn Triton CE or `logprobs_from_logits_v2`, both row-wise) |
| full-row consumers (must fall back) | distillation top-k / score centering: `logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), ...)` with `assert v.shape == (logits_rmpad.shape[0],)` :1430-1439 |
| layout restore | `_gather_and_unpad_packed` :1454 (:684-693: Ulysses all-gather on dim 0, else strip the static pad) -> `nested_tensor_from_jagged(log_probs, cu_seqlens)` :1464 |
| who reads the rows | update: `ppo_loss` -> `no_padding_2_padding(model_output["log_probs"], data)` (`verl/workers/utils/losses.py:57,59` -> `padding.py:97-141`, slice `values[seq_offset - resp_len - 1 : seq_offset - 1]` :138, i.e. rows `[prompt_len-1, seq_len-1)`); forward-only passes: `response_from_nested` (`padding.py:194-210`, same slice :209) at `verl/trainer/ppo/v1/trainer_base.py:1668-1669` (old_log_probs, entropy), :1717 (ref), :1739 (values); the entropy metric is masked by `response_mask` :1678-1685; `sum_pi_squared` consumers read response-shaped tensors (`verl/trainer/ppo/metric_utils.py:709-714`, `ray_trainer.py:264-271`). **Every PPO consumer reads exactly the rows `[e_i - r_i - 1, e_i - 1)` per sequence.** SFT (`losses.py:28-44`) reads rolled `loss_mask` rows instead: K2 must stay off when `responses` is absent. |
| HF contract (5.16.1) | `modeling_qwen3.py:457` `logits_to_keep: int | torch.Tensor = 0`; :494-495 `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep; logits = self.lm_head(hidden_states[:, slice_indices, :])`. Identical in `modeling_llama.py:447,479-480`, `modeling_qwen2.py:432,464-465`, `modeling_qwen3_moe.py:614,658-659`, `modeling_qwen2_5_vl.py:1323,1386-1387`, `modeling_qwen3_vl.py:1313,1371-1372`. A 1-D LongTensor on the packed `(1, T, H)` input yields `(1, K, H)` -> `(1, K, V)`. In 5.21.0 the same contract is `slice_logits_to_keep` (`modeling_utils.py:5008-5022`), which also documents a 2-D bool `(batch, seq)` form "e.g. packed training": stable upstream, not a 5.16-only accident. |
| wrappers | PEFT 0.19.1 `PeftModelForCausalLM.forward(**kwargs)` -> `self.base_model(..., **kwargs)` (`peft_model.py:1958-2001`): LoRA passes it through. FSDP1/FSDP2 wrap `forward` transparently. VeOmni overrides `prepare_model_inputs` (`verl/workers/engine/veomni/transformer_impl.py:882`): untouched. |
| VLM caveat | under rmpad verl swaps the HF forward for `forward_with_normal_backend` (`monkey_patch.py:387-392, 436-444`), which does `logits = self.lm_head(hidden_states)` on all rows (`verl/models/transformers/qwen2_vl.py:473-482`, `qwen3_vl.py:327-336`). Qwen2.5-VL / Qwen3-VL need the same two-line slice there, or the build-time signature probe leaves them on the full path. |
| critic | `FSDPEngineWithValueHead` :1616 reads `output.logits` of a `num_labels=1` head (`(total_nnz, 1)`, :1629-1636): no vocab dimension, nothing to gain, nothing to break. |
| SP | `ulysses_pad_and_slice_inputs` (`verl/utils/ulysses.py:302`, slice :121-135) gives rank r the contiguous rows `[r*T/sp, (r+1)*T/sp)` of the padded global sequence; the head runs per rank, log-probs are all-gathered afterwards (:684-693). A rank-local keep index (global index minus the rank offset, scattered into the local `(T/sp,)` before the gather) keeps the gather order unchanged. A rank with zero loss rows must still project >= 1 row (empty `(1,0,H)` matmul and a 0-row CE are untested) - #7686 solves it with a dummy row. |

Row fractions `K/T` (what the flag can save): BASELINE-A (GSM8K prompts ~83 tokens per #8299's own
measurement, mean response 1230 from `bench/results/baseline-a-main-1`): 0.937 -> 6.3 % of the head;
1k prompt / 2k response: 0.67 -> 33 %; agentic 16k / 2k: 0.11 -> 89 %; multi-turn with masked tool
observations inside `responses` (`verl/experimental/agent_loop/agent_loop.py:95,123`): more, in a
phase-2 "mask-aware" variant (v1 keeps all response rows, so nothing masked-but-read changes).

### CPU proof (`research/vet/K2/k2_cpu_proof.py`, run 2026-10-29 in `.venv-cpu`)

The script runs the *real* `FSDPEngineWithLMHead.prepare_model_inputs` / `prepare_model_outputs` on a
bare engine instance (the pattern of `tests/workers/test_packed_pad_to_length_on_cpu.py`) with a random
2-layer Qwen3 (V=512), four packed sequences (prompt/response 37/9, 5/14, 20/3, 11/6), fp32, then the
same model with `logits_to_keep=<loss rows>` plus the proposed ~15-line output delta (index labels and
temperature by the kept rows, `index_copy` the K results into a zero `(T,)` tensor, then the unchanged
gather / re-jag), and compares what both consumers read, plus the gradients through the head and the
embedding:

```
torch 2.13.0  transformers 5.16.1
=== case: unit temperature, no static pad (temperature=1.0, pad_to_length=False) ===
packed tokens T=105 (static pad 0), loss rows K=32, kept fraction 0.305
response log-probs  max|diff| = 0.000e+00   (allclose atol=1e-6: True)
nested consumer     max|diff| = 0.000e+00
response entropy    max|diff| = 0.000e+00
lm_head.weight.grad max|diff| = 0.000e+00  (|grad| max 3.591e+00)
embed.weight.grad   max|diff| = 0.000e+00  (|grad| max 4.521e+00)
forward FLOPs total  full=2.801e+07  k2=2.323e+07  (17.1% fewer)
lm_head FLOPs        full=6.881e+06  k2=2.097e+06  (69.5% fewer; expected 69.5%)
logits tensor bytes  full=215040  k2=65536  (69.5% smaller)
PASS
=== case: per-sample temperature, pad_to_length bucket 64 (temperature=tensor([1.0, 0.7, 1.3, 1.0]), pad_to_length=True) ===
packed tokens T=128 (static pad 23), loss rows K=32, kept fraction 0.250
response log-probs  max|diff| = 0.000e+00   (allclose atol=1e-6: True)
nested consumer     max|diff| = 0.000e+00
response entropy    max|diff| = 0.000e+00
lm_head.weight.grad max|diff| = 0.000e+00  (|grad| max 6.974e+00)
embed.weight.grad   max|diff| = 0.000e+00  (|grad| max 6.615e+00)
forward FLOPs total  full=3.565e+07  k2=2.936e+07  (17.6% fewer)
lm_head FLOPs        full=8.389e+06  k2=2.097e+06  (75.0% fewer; expected 75.0%)
logits tensor bytes  full=262144  k2=65536  (75.0% smaller)
PASS
ALL CASES PASS
```

Bitwise identical on CPU fp32 (log-probs, entropy, head and embedding gradients), head FLOPs and
logits bytes down by exactly the dropped-row fraction, with per-sample temperature and the static pad
exercised. On GPU bf16 the identity is "up to cuBLAS kernel selection for a different M": the same row
can round differently when the GEMM tile changes, so the GPU acceptance metric is max |dlogp| <= bf16
noise (~1e-3), not bitwise equality.

## 3. Duplicate / ownership check

### 3.1 Local dump (`research/scripts-a7/dupcheck.py`, 774 open PRs as of 2026-10-28)

`python3 -I research/scripts-a7/dupcheck.py --kw logits_to_keep --kw "response-only" --kw "response only"
--kw "active rows" --kw "loss rows" --kw "selected token" --kw lm_head --kw "lm head" --kw "vocab projection"
--merged` -> 20 open hits, 0 merged-title hits. The ones that matter:

| PR | state (gh, 2026-10-29) | what it does | overlap with K2 |
|---|---|---|---|
| #7686 qiangyupei `[megatron] feat: project LM head only for response tokens` | OPEN, created 2026-09-02, rebased 2026-10-12, CONFLICTING, REVIEW_REQUIRED, +533/-43, 9 files, all Megatron (`verl/models/mcore/*`, `verl/workers/engine/megatron/transformer_impl.py`, `engine/megatron.yaml`, `workers/config/engine.py`); reviews: one tardis-key nit (".to(bool) ?", 2026-09-14), nothing since | key `megatron.response_only_lm_head` (default false); unfused + Triton fused heads, TP/SP/CP/PP; A100 Qwen3-8B TP2/PP2/CP2: 16K->4K step -7.9 % unfused / -16.0 % fused, actor update -7.1 % / -15.0 %, memory -5.1 GiB (-15 %); 8K->8K -1.3 % / -5.7 % | Megatron only. Its body hands FSDP to #7370 ("complements the FSDP/FSDP2 row-selection work in #7370"). K2 should reuse the key name under `fsdp_config` and cite it as the twin and as the measured precedent. |
| #7370 Zhaoyi-Tian `[fsdp] feat: project only active rows in the Torch fused LM head` | CLOSED by the author 2026-09-20 (opened 2026-08-12), +159/-36, files `verl/utils/experimental/torch_functional.py`, `transformer_impl.py`, 3 tests; **zero reviews**; the only comment is the author asking wuxibin89 for a reviewer (2026-08-13), unanswered | `use_fused_kernels=True` with the torch backend only: `-100` labels make `FusedLinearForPPO` skip rows | the eager default path was never touched. Not rejected, starved. K2 is materially different (default path, HF `logits_to_keep`, config flag) and the PR body must say so and credit it. |
| #7712 eteluna `[RFC] Selected-token log-probabilities from the actor engine, including fused kernels` | OPEN issue; narrowed 2026-10-03 to one default-off output `model_output["selected_token_logprobs"]` (log-softmax at fixed token ids for verl-genrank); cc vermouth1992 / zhtmike / wuxibin89 on 2026-09-03, **no maintainer reply in 8 weeks**; WenzheWang offered a kernel slice | more outputs, not fewer rows | orthogonal: their projection would run on K2's kept rows for free. Not competing; say "fewer rows, same outputs" in the body so a reviewer does not conflate them. |
| #8300 DeeproChoudhury `[fsdp, perf, trainer] feat: shared-prefix packing for the FSDP engine` | OPEN since 2026-10-20, MERGEABLE, +1968/-75, only coderabbit + self reviews | computes each GRPO group's shared prompt once (trunk + branches); edits `prepare_model_inputs` (`pack_index` gather of `input_ids_rmpad` / `temperature_rmpad`) and `prepare_model_outputs` (`index_select(0, unpack_index)` on log_probs / entropy / sum_pi_squared) | same function, different mechanism; composes (after packing, prompt rows shrink n-fold but still go through the head: at 16k/2k, n=8, 50 % of rows are still prompt). K2 must gate off or derive its index from the packed layout when `use_prefix_packing` is on; conflicts are mechanical. Competes for the same reviewer bandwidth. |
| #7944 (in-place temperature, `transformer_impl.py:1404-1408`), #8419 draft (FP32 logits, same lines, updated 2026-10-28), #8096 / #8097 / #8312 (fused LM head through FSDP2 hooks) | OPEN | adjacent lines of `prepare_model_outputs` | trivial conflicts; none selects rows. |

### 3.2 AGENTS.md-mandated GitHub searches (run 2026-10-29, read-only)

- `gh pr list --state open --search "logits_to_keep"` -> none; `--state merged --search "logits_to_keep"` -> none;
  open `"lm_head response tokens"` -> #8419, #7483 (unrelated).
- `--search "7370 in:body"` -> #7686 only; `"7686 in:body"` -> #7826 (Megatron MTP).
- `git log -S logits_to_keep -- verl/workers/engine/fsdp/ verl/models/transformers/` -> only #462 (2025) and
  #1212: nothing of this kind has landed on main; merged dump 2026-07-16..10-28: no title match.

### 3.3 Ownership and maintainer statements

- `transformer_impl.py` last 10 commits: Joel = wuxibin89 (`wuxibin@bytedance.com`: #8180, #8171, #7873,
  #7272, #7161), cr-gao (#8010 score centering), Lei Ding (#7874), MinjieZhang (#7428 unit temperature,
  #7458). `padding.py`: wuxibin89 (#7407, #5401). Expect wuxibin89 to merge; MinjieZhang / cr-gao know
  the exact lines; tardis-key is the only maintainer who has reviewed a response-only head (Megatron).
- No 26Q4 roadmap line (#8186) claims FSDP head work (01 §2.1); no comment on #7370 / #7686 / #7712
  reserves it. The maintainers' stated memory answer is fused-by-default (#6832, open since June with
  correctness fixes pending); it does not change the row count (`dense_common.py:84,152` ignore
  `logits_to_keep`), and #7686 measured fused + response-only as the best combination.

G1 PASS (no open FSDP PR; closed #7370 and Megatron #7686 must be named with the difference). G2 PASS.

## 4. Minimum valuable PR

Title (passes `tests/special_sanity/check_pr_title.py`): `[fsdp, perf] feat: project the LM head only for loss rows`.

| file | change | ~lines |
|---|---|---|
| `verl/workers/config/engine.py` `FSDPEngineConfig` (:233) | `response_only_lm_head: bool = False` + docstring (same name as #7686's Megatron key) | 8 |
| `verl/trainer/config/engine/fsdp.yaml` (next to `entropy_from_logits_with_chunking` :47-51) and the regenerated `_generated_ppo_trainer.yaml` (actor :27 and ref :235 `fsdp_config` blocks, via the `autogen-trainer-cfg` hook) | key + comment | 6 |
| `verl/workers/utils/padding.py` | `loss_row_index(micro_batch, total_len) -> LongTensor`: rows `[e_i - r_i - 1, e_i - 1)` from `input_ids.offsets()` and `responses.offsets().diff()` (the proof script's helper, vectorised, no host sync) | 20 |
| `transformer_impl.py` `_build_model_optimizer` (:596-600) | probe `"logits_to_keep" in signature(module.forward).parameters` once, like `pass_packed_cu_seqlens` | 4 |
| `transformer_impl.py` `prepare_model_inputs` (:1268-1272) | when flag, probe, rmpad, `responses` in the batch, and none of {fused, `distillation_use_topk`, `score_centering`, prefix packing, TP}: build the index after the static pad (:1221-1227) and the SP slice (:1240-1246, rank-local offset, >= 1-row guard); `model_inputs["logits_to_keep"] = idx`; `output_args["loss_row_index"] = idx`; otherwise a no-op | 25 |
| `transformer_impl.py` `prepare_model_outputs` (:1400-1449) | if `loss_row_index` in `output_args`: index `temperature_rmpad` / `input_ids_rmpad_rolled` by it; compute sum_pi_squared / entropy / log-probs on K rows; `index_copy` each into a zero `(T_local,)` before :1454 | 20 |
| `verl/models/transformers/qwen2_vl.py:473-482`, `qwen3_vl.py:327-336` | optional: honour `logits_to_keep` in `forward_with_normal_backend` (2 lines each) | 6 |
| `tests/workers/test_response_only_lm_head_on_cpu.py` | the proof script as pytest: identity against full logits for both consumers, per-sample temperature, `pad_to_length`, an SFT-shaped batch (no `responses`) -> flag no-op, each fallback guard | 130 |
| `tests/models/test_fsdp_no_padding_on_gpu.py` | one case asserting `model_inputs["logits_to_keep"]` and the `(1, K, V)` output (pattern of `test_prepare_model_inputs_passes_declared_packed_sequence_boundaries_on_gpu`, :141-166) | 30 |
| `docs/perf/perf_tuning.rst` (section 9, :358-364) | when to turn it on, what it skips, fused-kernel / SP / prefix-packing interaction | 12 |

About 260 lines, ~95 of them engine code. Default **off** in v1 (the convention of `use_fused_kernels`,
`pad_to_length`, #7686); the PR body proposes default-on once the GPU A/B shows max |dlogp| within bf16
noise. Scope: v1 = eager rmpad path, text CausalLM via the probe, SP=1 (SP support is ~30 more lines
plus a 2-rank test; include it if the owner can run `tests/special_distributed` on the harness, because
long-prompt users are the SP users). Out of v1: the fused path (#7370's territory, three backends),
mask-aware dropping inside responses, and C2's forward-only budget decoupling.

## 5. Proof on the Modal harness (`bench/README.md`)

BASELINE-A as shipped cannot show K2: GSM8K prompts are ~83 tokens (#8299's own measurement) and the
mean response is 1230 (`bench/results/baseline-a-main-1`), so K/T = 0.94. Arithmetic for Qwen3-1.7B
(H=2048, V=151936, 28 layers, 1.41e9 non-embedding params): head 2HV = 6.2e8 FLOP/token; body 2.82e9 +
attention scores ~0.15e9 at 1.3k-token sequences -> the head is 17 % of forward FLOPs; dropping 6.3 % of
it is -1.1 % of each FSDP pass (-0.04 s of old_log_prob 3.49 s, -0.1 s of update_actor 8.71 s), inside
the +-5 % spread. Run it once anyway as the no-regression pair (flat timings, identical `actor/entropy`,
`pg_clipfrac`, `kl_loss`).

Isolating config "BASELINE-A-LP": same preset and model, a GSM8K variant whose question is prefixed by a
fixed ~6000-token reference block (a 30-line sibling of `examples/data_preprocess/gsm8k.py`, kept outside
the clone, written to the hf-cache volume by the harness `download` step as `/hf-cache/data/gsm8k-lp6k/`),
`--max-prompt-length 8192 --max-response-length 1024 --ppo-max-token-len-per-gpu 24576` (3 sequences per
micro-batch). Expected K/T ~ 0.12-0.15.

```
PY=.venv-bench/bin/python
$PY bench/verl_bench.py train --preset baseline-a --run-name k2lp-main-1 --src verl \
  --max-prompt-length 8192 --max-response-length 1024 --ppo-max-token-len-per-gpu 24576 \
  --overrides "data.train_files=/hf-cache/data/gsm8k-lp6k/train.parquet data.val_files=/hf-cache/data/gsm8k-lp6k/test.parquet"
$PY bench/verl_bench.py train --preset baseline-a --run-name k2lp-k2-1 --src /path/to/k2-worktree \
  --max-prompt-length 8192 --max-response-length 1024 --ppo-max-token-len-per-gpu 24576 \
  --overrides "data.train_files=... data.val_files=... actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True actor_rollout_ref.ref.fsdp_config.response_only_lm_head=True"
python3 bench/summarize.py bench/results/k2lp-main-1; python3 bench/summarize.py bench/results/k2lp-k2-1
```

Expected effect (1.7B at ~7k-token sequences: attention scores ~0.8e9/token, head share 14.6 %; K/T =
0.13 -> head -87 % -> **-12.7 % of the FLOPs of every FSDP pass**): `old_log_prob` and `ref` -10…-13 %
each, `update_actor` -9…-13 % (forward, recompute and backward all carry the head; optimizer step and
all-gathers do not), `gen` unchanged, `step` -4…-6 % because generation dominates;
`actor/perf/max_memory_allocated_gb` down by the logits (24576 x 151936 x 2 B = 7.47 GB -> ~1.0 GB) plus
the old-log-prob entropy intermediates: -5…-8 GB per rank. Correctness columns side by side:
`actor/entropy`, `actor/ppo_kl`, `actor/pg_clipfrac`, `kl_loss` (equal to bf16 noise), reward-curve
overlap, and a 1xH100 forward of the same weights with the flag on/off reporting max |dlogp| on response
rows (expect <= 1e-3). Metrics maintainers accept: phase wall time and peak memory from `metrics.jsonl`
(#7686 reported exactly these) plus the identity check. Repeat any arm whose delta is under 5 % (`-2`).

Rubric reference user (8xH100, Qwen3-8B, 8k responses): head 2HV = 1.24e9 vs body 13.9e9 + attention
~2.4e9 -> 7 % of FLOPs; with 1-2k prompts K/T ~ 0.85 -> -1 % FLOPs and -0.7 GB of the 4.98 GB logits at
16k tokens. The value sits in prompt-heavy and agentic runs (16k/2k: -6 % FLOPs, -4.4 GB), not in the
plain long-response user.

## 6. Adversarial answers

**A1 Strongest reason it will NOT be merged.** The FSDP engine has no responsive owner for this kind of
change. #7370 (same idea, fused path) got zero reviews in five weeks and died after the author asked
wuxibin89 by name; #7686 (Megatron) has one nit in eight weeks and is CONFLICTING; #7712 has no
maintainer reply in eight weeks. A first-time author (7 of the last 400 merges) with an opt-in flag on
the eager path is the profile that stalls. Mitigation: a ~100-line engine diff, the CPU identity test,
harness numbers, #7686's A100 table and key name, reviewers tardis-key + wuxibin89 asked by name.

**A2 What makes the gain disappear.** Short prompts (GSM8K: 6 %); `use_fused_kernels=True` (flag is a
no-op in v1; the fused forwards project all rows, `dense_common.py:84,152`) which is the maintainers'
direction (#6832); #8300 prefix packing enabled (prompt rows shrink n-fold; at 16k/2k, n=8 they are still
50 % of rows, so the gain halves); small models / small budgets where logits are not the peak (only the
FLOP win remains); a transformers release dropping the tensor form (5.21.0 kept and extended it).

**A3 Correctness risk, and which tests catch it.** Off-by-one in the row index: the CPU identity test
against `no_padding_2_padding` and `response_from_nested`, plus
`tests/models/test_fsdp_no_padding_on_gpu.py::test_prepare_model_outputs_can_be_sliced_back_to_response_shape_on_gpu`.
A consumer reading a non-response row and now seeing 0: none exists on main (every PPO reader slices
`[e_i - r_i - 1, e_i - 1)`); the SFT loss reads other rows, hence the `responses`-present guard (tested).
SP rank offsets and an empty rank: 2-rank test or exclusion in v1. `pad_to_length`: tested.
Distillation / score centering / TP DTensor / prefix packing: guards, with a test that each holds.
GPU GEMM rounding for a different M: reported as max |dlogp|, never hidden. The scatter is `index_copy`
with unique indices: deterministic. `tests/models/test_engine.py` and the `special_e2e` GRPO scripts cover
the flag only if a CI script flips it; leave CI as is and let the reviewer decide.

**A4 Who else is working on it.** qiangyupei (#7686, Megatron; his body hands FSDP to #7370 and he has
not touched FSDP); Zhaoyi-Tian (#7370, closed; could resurrect the fused-torch half, so credit him and
leave the fused path alone); DeeproChoudhury (#8300, same function, different goal); eteluna (#7712,
more outputs from `prepare_model_outputs`, orthogonal). Nobody is building the eager-path version.

**A5 Half scope.** Drop SP, VLM and sum_pi_squared handling (full rows for all three), keep the flag,
`loss_row_index`, the two engine hunks, the CPU test and the docs paragraph: ~150 lines, still the
whole win for the default single-rank text path.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | PASS | no open PR touches the FSDP eager head rows; #7370 closed / unreviewed (fused torch only), #7686 Megatron only; both named with the difference |
| G2 not reserved | PASS | no roadmap line, no maintainer claim; #8300 is opt-in and composes |
| G3 self-contained | PASS | one engine file + config + helper + tests; no engine fork, no dependency |
| G4 provable | PASS | CPU identity test (bitwise, fp32, real engine code) + BASELINE-A-LP phase times and peak memory on 8xH100 |
| G5 default-path impact | PASS with caveat | the default path's largest tensor and 7-17 % of its FLOPs, but behind an opt-in flag until the GPU A/B justifies default-on |

Scores: **S1 3** (reference user ~1 %; long-prompt / agentic users 10-15 % of every FSDP pass and
5-8 GB per rank; #7686 measured -7…-16 % step on Megatron at 16K->4K). **S2 3** (two independent
implementations on other paths, #8299 / #6401 / #7561 document the prompt-heavy workloads, HF / TRL
precedent; no maintainer ask). **S3 4** (mechanism verified line by line, identity measured on the
real engine code, A100 twin numbers; GPU wall time derived, not yet measured). **S4 3** (small,
conventional, tested; FSDP review starvation and three open PRs on adjacent lines). **S5 2** (first on
verl's FSDP eager path; the idea is HF / TRL standard).

**Verdict: GO-WITH-CHANGES.** Strongest reason: a ~100-line opt-in change on the default path whose
output identity is proven on the real engine code, whose win scales with exactly the prompt-heavy
workloads the tracker complains about, and which has no living competitor on FSDP. Required changes:
(1) prove it on BASELINE-A-LP, not BASELINE-A; (2) default off in v1 with the default-on proposal in the
body; (3) `responses`-present / fused / distillation / score-centering / TP / prefix-packing guards, each
with a test; (4) decide SP in or out before filing (in if a 2-rank test can run); (5) PR body names
#7370, #7686 (same key name), #7712 and #8300 with the differences, and asks tardis-key + wuxibin89.
