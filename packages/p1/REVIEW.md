# Adversarial review: `contrib/p1-lmhead-rows` (`[fsdp, perf] feat: project the LM head only for loss rows`)

Reviewer: Claude Fable 5.1, 2026-10-29, read-only on `wt/p1` @ 4bcbbeb9 (base main @ 2ab35292). Everything
below was read in the code, not taken from NOTES.md / PR-BODY.md. File:line references are the worktree.
Adversarial tests: `packages/p1/review-tests/` (run log in section 5). Nothing was edited in the worktree.

## 1. Verdict: NEEDS-FIXES

The product code does what the PR says: the row index equals what both consumers slice, the scatter puts zeros
only on rows no PPO consumer reads, the guards hold, the config is consistent, lint and the CPU regression set
are green. I could not make a consumed number differ from the full projection on CPU fp32 in any layout
(degenerate lengths, static pad, LoRA, tied embeddings, entropy checkpointing/chunking, `forward_step`).

The single most important reason it is not merge-ready: **the shipped CPU test asserts `torch.equal` on
`lm_head.weight.grad` and `embed_tokens.weight.grad`** (`tests/workers/test_response_only_lm_head_on_cpu.py:207`).
Gradient GEMMs run with a different M (K rows vs T rows) and bitwise equality is not a BLAS guarantee: on this
Mac a 1-token prompt / 1-token response layout already gives `head.grad` max|diff| 4.66e-09 and `embed.grad`
1.67e-06 (section 5, test `test_single_sequence_single_response_token`). verl's CPU CI runs
`pytest -s -x ... tests/` on x86 (`.github/workflows/cpu_unit_tests.yml:116-118`): a different blocking over M
on MKL/OpenBLAS would turn the PR's own test red and abort the whole CPU job. The author's NOTES Q1 planned
`assert_close(atol=1e-6)` for the gradients; the test does not do it. One-line fix. Secondary: the docs and PR
body promise a start-up warning for `use_fused_kernels=True`; the code never emits one (the fused forward's
signature lists `logits_to_keep`, so the flag resolves True and every micro-batch no-ops silently).
Benchmark section is TBD and no GPU test exists, so the PR cannot be filed as-is anyway.

## 2. Defects

| # | severity | where | failing input / evidence | proposed fix |
|---|---|---|---|---|
| D1 | **major** (CI flake, `-x` aborts the job) | `tests/workers/test_response_only_lm_head_on_cpu.py:206-207` (`torch.equal` over all of `full`, including the two `.grad` entries) | `_make_micro_batch` with prompt/response `[1]/[1]`: forward quantities equal, `lm_head.weight.grad` differs by 4.66e-09, `embed_tokens.weight.grad` by 1.67e-06 (Apple BLAS, M=1 vs M=2). 0 of 22 gradient tensors differed over 11 random larger layouts here, so it passes locally by luck of the kernel path; x86 CI uses a different BLAS. It is rounding, not a bug: relative 1.3e-09 / 5.6e-07 in fp32, and 2.8e-18 / 0.0 when the same layout runs in float64 | gradients: `torch.testing.assert_close(value, kept[name], rtol=1e-5, atol=1e-6)` (rtol=0, atol=1e-6 is not enough for the M=1 case) or run the model in float64; keep `torch.equal` for log-probs / entropy / Σπ² (row-wise, inner dim H or V, equal in every layout I tried) |
| D2 | minor (doc contradicts code) | `docs/perf/perf_tuning.rst:393-397`, `PR-BODY.md` "API and Usage Example" paragraph, vs `verl/trainer/config/engine/fsdp.yaml:78-80` and `verl/workers/config/engine.py:276-278` ("silently keeps") | flag on + `use_fused_kernels=True`: `_resolve_response_only_lm_head` returns True with no warning (the probe sees `logits_to_keep` on `dense_common.py:84,152`), then `_response_only_lm_head_index` returns None on every batch, silently (review test `test_docs_claim_a_start_up_warning_for_fused_kernels`, FAILS on the branch) | either warn at build time (`self.model_config.use_fused_kernels` is known in `_build_module` :355) for fused / top-K / score-centering, or reword rst + body to "per-batch conditions apply silently" so all four places agree |
| D3 | minor (wrong claim) | `PR-BODY.md` "the tensor form transformers has supported since 4.45" | transformers v4.45.0 `modeling_llama.py:1150` has only `num_logits_to_keep: int`; the tensor form `logits_to_keep: Union[int, Tensor]` appears in v4.49.0 (:800) with the deprecation of the old name | say "since 4.49" or just cite verl's pin (`requirements.txt:20` `>=5.5.3`) |
| D4 | minor (dead alias) | `transformer_impl.py:106` `_LOGITS_TO_KEEP_KWARGS = ("logits_to_keep", "slice_logits_to_keep")`, docs / body / yaml "`slice_logits_to_keep` is also accepted" | 5.21.0 still names the kwarg `logits_to_keep`; `slice_logits_to_keep` is the helper in `modeling_utils` (verified in `.venv-a9`); no release names a kwarg that way | drop the alias (and the three wording sites) or mark it explicitly speculative; the `RuntimeError` at :1495 already covers a renamed kwarg |
| D5 | minor (perf) | `verl/workers/utils/padding.py:161,166` | the default rmpad path without `pad_to_length` had no device->host sync in `prepare_model_inputs`; the branch adds two per micro-batch (`assert bool(...)` and `int(...sum().item())`) on old-log-prob / ref / update passes | `total = data["responses"].values().shape[0]` (static, no sync); drop the assert (`no_padding_2_padding` :126 already asserts the same invariant on the consumer side) or keep it under a debug flag |
| D6 | minor (readability) | `transformer_impl.py:1349`, `:1497` | local `loss_row_index = ...` shadows the imported `loss_row_index` function inside both methods; ruff is silent | rename the locals (`keep_index`) |
| D7 | minor (dead code) | `transformer_impl.py:1555` `if not distillation_only:` inside the scatter block | `_response_only_lm_head_index` returns None whenever `distillation_only` is set (:1428-1430), so the branch cannot be False | drop the condition (scatter `log_probs` unconditionally) |
| D8 | nit | commit 4bcbbeb9 `[doc] docs: ...` | `docs` is not one of feat/fix/refactor/chore/test; only the PR title is CI-checked (squash merge) | `[doc] chore:` if commits are kept |
| D9 | nit | `PR-BODY.md` / rst "~6 % for GSM8K-style 80-token prompts with 1k responses" | 1 - 1000/1080 = 7.4 %; the vet's 6.3 % was 83/1230 | "6-7 %" |
| D10 | design note | `transformer_impl.py:623-648` | `use_remove_padding=False` + flag is a contradiction handled by one warning per rank (8 lines in an 8-GPU log); SP>1 (the long-prompt users the flag targets) likewise | raise for `use_remove_padding=False` (the file already raises for pad_to_length + top-K at :1223); keep the warning for SP / VLM / no-kwarg; warn on rank 0 only |

No defect was found in: the index rule, the scatter, the per-batch guards, the config wiring, the probe order
(raw HF module before PEFT/FSDP), or any consumer of the engine outputs (section 6 lists what was checked).

## 3. Semantic risks that need a GPU check (and what to measure)

None of these can be settled on this Mac; each is a place where the CPU identity proof does not transfer.

| risk | why it is open | measure |
|---|---|---|
| bf16 GEMM with a different M rounds differently | cuBLAS picks kernels by (M, N, K); the kept-row GEMM (K x H x V) is a different problem than the T-row one. The PR body already says so; nobody has a number | 1xH100, same weights, flag on/off, one old-log-prob pass over a real batch: `max |dlogp|` and `max |dentropy|` on response rows (bf16 autocast). Accept <= ~1e-3; report the number in the body |
| flash-attn Triton cross-entropy on K rows with `inplace_backward=True` | `logprobs_from_logits_flash_attn` (`torch_functional.py:110-133`) writes `dlogits` into the logits buffer; the K-row logits are a fresh `(K, V)` tensor from `lm_head(hidden[:, idx, :])`, so this should be fine, but it is the one kernel the CPU test never runs | run the identity pass twice, once with `VERL_DISABLE_FLASH_ATTN_CE=1`; compare `lm_head.weight.grad` on a tiny model, flag on vs off, in fp32 on GPU (`assert_close` 1e-5) |
| kwarg pass-through under FSDP1 / FSDP2 + PEFT + activation checkpointing / offload | `forward_step` calls `self.module(**model_inputs, use_cache=False)` (:1699); PEFT forwards `**kwargs` (`peft_model.py`, checked 0.19.1 signature), `fully_shard` keeps the class forward; the HF slice sits outside the checkpointed decoder layers. Unexercised end to end | the vet's missing GPU case: `tests/models/test_fsdp_no_padding_on_gpu.py` with a forward hook on `lm_head` asserting `(1, K, V)` for `strategy in {fsdp, fsdp2}` x `lora_rank in {0, 8}` x `enable_gradient_checkpointing`; and the `RuntimeError` at :1493 never firing in the GRPO e2e script |
| `torch.compile(entropy_from_logits, dynamic=True)` (:211) now sees K-row inputs | K varies per micro-batch like T did, but K also differs from the padded T when `pad_to_length` is on, so a run that used to be shape-static becomes dynamic in the entropy kernel | count recompiles (`TORCH_LOGS=recompiles`) with `pad_to_length=True` and the flag on; if it recompiles per step, document or mark dynamic shapes explicitly |
| one host sync per micro-batch (D5) | `padding.py:161,166` | nsys on an old-log-prob pass: look for a `cudaStreamSynchronize` before each forward; if visible, apply D5 |
| timing / memory, the whole point | nothing measured (`### Benchmark` is TBD) | BASELINE-A no-regression pair + BASELINE-A-LP (8192/1024) per `research/vet/K2-lmhead-rows.md` §5: `timing_s/old_log_prob`, `ref`, `update_actor`, `step`, `actor/perf/max_memory_allocated_gb`, mean of steps 2..N; correctness columns `actor/entropy`, `actor/ppo_kl`, `actor/pg_clipfrac`, `kl_loss` side by side |
| MoE / other HF text models | only Qwen3 was run; `Qwen3MoeForCausalLM`, Llama, Qwen2 carry the same `logits_to_keep` (vet §2) but a router aux-loss path or a remote-code model (`trust_remote_code`) may differ | one forward each on Qwen3-30B-A3B (or a tiny MoE config) and a `trust_remote_code` model: assert the hook row count or the warning, never the `RuntimeError` |
| NPU | `index_copy` / advanced indexing on the Ascend path (`logprobs_from_logits_torch_npu`) untested | not needed for merge; note in the body that NPU is unverified |

## 4. Merge-process gaps

1. **`### Benchmark` is a TBD table** (`PR-BODY.md`). Cannot be filed. The vet (§5) requires BASELINE-A-LP, not
   BASELINE-A; the body's own "which the benchmark section reports" promise for `max |dlogp|` is unmet.
2. **No GPU test** although the vet's minimum PR (§4) lists a `tests/models/test_fsdp_no_padding_on_gpu.py` case
   and the PR template asks for tests "to cover all the code" or a reason. The body's checklist ticks the box with
   "GPU: benchmark section", which is not a test. Either add the case or write the explicit "why not" line.
3. **Checklist item "Apply pre-commit checks" is ticked, but `pre-commit run --all-files` was not run end to end**
   (NOTES.md "Not run"; only the sanity hooks and ruff on changed files). I re-ran ruff 0.12.2 check + format on the
   four Python files (clean) and regenerated all four `_generated_*.yaml` (identical), so the risk is low, but the
   tick should be honest or the full run done on Linux before filing.
4. **Doc/body vs yaml/docstring contradiction on warnings** (D2) and the wrong transformers version (D3): a
   maintainer who checks either will stop trusting the rest of the body.
5. **`ci-request` message** pending (expected; submitter's step).
6. **AGENTS.md compliance**: duplicate-check links present (three GitHub queries + `git log -S`), differences to
   #7370 / #7686 / #7712 / #8300 stated, AI-assistance statement present, test commands + results present,
   `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` on all four commits. OK.
7. **Title** `[fsdp, perf] feat: project the LM head only for loss rows`: `check_pr_title.py` valid (re-run). OK.
8. **Scope**: 170 product lines, one engine file + helper + config + docs; no drive-by changes (diff read in full).
   OK. The `[doc] docs:` commit type (D8) is cosmetic.
9. **Docs**: rst heading underline length matches (46/46), `Last updated` bumped, item 11 added to the list, the
   section uses only plain rst. OK. The paragraph "with a warning at start-up where the layout is fixed, when it
   cannot apply: use_fused_kernels=True, ..." is the D2 overclaim.
10. **Reviewer ask**: the vet asked that the body name tardis-key (reviewed #7686) and wuxibin89 (owner of
    `transformer_impl.py`, `padding.py`). Not in the body; the template does not require it, but it is the only
    mitigation the vet had for A1 (FSDP review starvation).

## 5. Adversarial tests (`packages/p1/review-tests/`, run against `wt/p1` @ 4bcbbeb9)

Run (from the worktree, `.venv-cpu`, torch 2.13.0 CPU, transformers 5.16.1, peft 0.19.1):

```
export PYTHONPATH=<repo>/wt/p1:<repo>/packages/p1/review-tests
.venv-cpu/bin/python -m pytest -c pytest-cpu.ini --rootdir packages/p1/review-tests -p no:cacheprovider -q -s \
    --asyncio-mode=auto packages/p1/review-tests
# 18 passed, 2 failed (both failures are the intended demonstrations of D1 and D2)
```

All tests drive the real `FSDPEngineWithLMHead.prepare_model_inputs` / `prepare_model_outputs` (and once
`forward_step`) around a 2-layer random Qwen3 (V=512, H=64), flag on vs off, comparing what
`no_padding_2_padding` and `response_from_nested` read plus gradients (`_helpers.py`).

| test | targets | result |
|---|---|---|
| `index_edges::TestDegenerateLengths::test_response_length_zero_and_one_rows_are_identical_to_the_full_projection` | prompts `[3,7,1,12,4]`, responses `[1,0,5,2,0]` (response_length 1, two prompt-only rows, a 1-token prompt), per-sample temperature | PASS: head projects 8 of 35 rows; log-probs, entropy, both grads bitwise equal; prompt-only rows read as zeros in both paths |
| `...::test_a_micro_batch_of_only_prompt_rows_keeps_the_full_projection_and_the_same_outputs` | all responses empty -> empty index -> fallback | PASS: no `logits_to_keep`, no `loss_row_index`, 16-row output, empty nested response |
| `...::test_single_sequence_single_response_token` | `[1]/[1]`, scalar temperature 0.8: the smallest M (gemv vs gemm) | **FAIL (D1, intended)**: forward quantities 0.0 diff; `head.grad` 4.66e-09 (rel 1.3e-09), `embed.grad` 1.67e-06 (rel 5.6e-07); the same layout in float64 gives 1.0e-17 / 0.0, so it is GEMM rounding for a different M. Even `assert_close(rtol=0, atol=1e-6)` fails on `embed.grad`, so the fix needs `rtol=1e-5` (or float64) |
| `...::test_gradient_bitwise_identity_over_random_layouts` | 11 random layouts (bsz 1-4, p 1-12, r 0-8) | PASS: 0 of 22 gradient tensors differ bitwise on Apple BLAS; `assert_close(atol=1e-6)` holds; forward bitwise equal. The shipped test passes here by the kernel path, not by guarantee |
| `index_edges::TestStaticPadInteraction::test_pad_to_length_variants_are_identical[32-...]` | T=96 exact bucket multiple (pad 0) + Σπ² + per-sample temperature | PASS |
| `...[64-[10,20,30]-[6,20,10]]` | pad 32 | PASS |
| `...[64-[40,20,60]-[6,1,0]]` | pad 1, last sequence prompt-only (no loss rows at the tail), response_length 1 with its own temperature | PASS; index max < T, pad stripped |
| `...[128-[2]-[3]]` | T=5 with 123 pad rows | PASS |
| `index_edges::TestFuzzIndexAgainstConsumers::test_random_layouts_match_both_consumers` | 60 random layouts (bsz 1-6, p 1-9, r in {0,1,2,3,5,7}): index vs brute force `[e-r-1, e-1)`, vs `response_from_nested`, and `no_padding_2_padding` on a zeroed-outside-index stream vs the full stream | PASS |
| `...::test_prompt_length_zero_is_rejected_not_misread` | p=0 row | PASS: `loss_row_index` and `no_padding_2_padding` both assert |
| `wrappers::TestLoRA::test_peft_wrapped_model_receives_the_index_and_is_identical` | `get_peft_model` (r=4, q/v), non-zero `lora_B`, per-sample temperature | PASS: `_probe_logits_to_keep_kwarg(peft_model) is None` (so the pre-wrap probe order is load-bearing), base module probes `logits_to_keep`; K=32 of 105 rows through PEFT; log-probs, entropy and `lora_A/lora_B` grads bitwise equal |
| `wrappers::TestTiedEmbeddings::test_tied_head_and_embedding_gradients_are_identical` | `tie_word_embeddings=True` | PASS (bitwise, this layout) |
| `wrappers::TestEntropyVariants::...[True-False]`, `[False-True]`, `[True-True]` | `entropy_checkpointing`, `entropy_from_logits_with_chunking` (chunk 8), both | PASS (bitwise) |
| `wrappers::TestForwardStepEndToEnd::test_forward_step_with_the_real_module_call` | the real `forward_step` (`micro_batch.to(device)`, `self.module(**inputs, use_cache=False)`, `detach_tree`) with the device helpers monkeypatched to CPU | PASS: flag on/off response log-probs bitwise equal; packed output length T |
| `wrappers::TestBF16Characterization::test_cpu_bf16_max_abs_dlogp` | bf16 model on CPU, K vs T rows | PASS, max abs dlogp = 0.0 and entropy 0.0 on this CPU (characterisation only; says nothing about cuBLAS) |
| `wrappers::TestBuildTimeClaims::test_docs_claim_a_start_up_warning_for_fused_kernels` | the documented "warning at start-up for use_fused_kernels=True" | **FAIL (D2, intended)**: resolves True, no warning |
| `...::test_probe_sees_the_fused_forward_as_accepting_the_kwarg` | why D2 happens | PASS: `dense_common.forward_with_{torch,triton}_backend` list `logits_to_keep` |

Not written (cannot run here): SP>1 with the flag beyond the resolver (`use_ulysses_sp=True` returns False
with the warning, covered by the shipped test), anything FSDP/CUDA.

Regression (my re-run of the groups the branch touches, same harness as `tests.log`): new file 23 passed;
`tests/workers` 301 passed / 40 skipped / 1 deselected; `tests/utils` 473 passed / 23 skipped. Author's
`packages/p1/logs/`: small 200, trainer 438 / 1 skipped (not re-run by me; untouched areas).

## 6. What was verified to be correct

- **Index rule.** `loss_row_index` (`padding.py:144-172`) returns `[e_i - r_i - 1, e_i - 1)` per sequence in packed
  order; equals `no_padding_2_padding`'s slice (`padding.py:138`) and `response_from_nested`'s (`padding.py:209`) in
  60 fuzzed layouts including r = 0, r = 1, p = 1; strictly increasing, int64, on the offsets' device.
- **Shift / labels / temperature.** `input_ids_rmpad_rolled[idx]` and `temperature_rmpad[idx]` (:1501-1502) are the
  same per-row label and per-sample temperature the full path uses at those rows (`expand_as_nested` :1258-1260);
  `temperature_is_one` fast path unchanged.
- **Static pad.** Index built on the unpadded layout, the pad is a suffix (:1226-1232 roll before pad); verified for
  pad 0 / 1 / 32 / 123; `total_rows` for the scatter is the padded local length (:1552) and the existing
  `_gather_and_unpad_packed` strips it (:1560-1565).
- **Scatter.** `_scatter_rows` (:126-132): `new_zeros` keeps dtype/device of the K-row result; `index_copy` with
  unique indices; gradient = gather (LoRA / tied / full grads match within rounding; bitwise in all but the M=1 case).
- **Every consumer reads only indexed rows.** `ppo_loss` (`losses.py:59-64`), `score_centering_ppo_loss` /
  `distillation_ppo_loss` (guarded off), `_compute_old_log_prob` (`trainer_base.py:1668-1669`), ref (:1717),
  `actor/entropy` (:1678-1681, response-shaped after `response_from_nested`), `sum_pi_squared`
  (`ray_trainer.py:1313-1316`), `calculate_debug_metrics` (reads padded `old_log_probs`, `metrics.py:87-96`).
  `sft_loss` reads rolled `loss_mask` rows (:33-44) only from the SFT trainers, whose batches carry no `responses`
  (`sft_trainer.py:161-163`, `sft_trainer_ray.py:110-112`); the actor's `loss_mask` is `response_mask`
  (`padding.py:71`). No other reader of `output.logits` on this path: `automodel` (:608), `prefix_grouper_utils`
  (:124, calls the model itself) and `naive_rollout` are separate paths.
- **Guards are consistent with the trainer.** `distillation_use_topk` / `distillation_only` / `score_centering` are set
  at `trainer_base.py:1853-1855` and the loss is picked from the same config at `engine_workers.py:681-687`;
  `use_fused_kernels` on the batch is `engine_config.use_fused_kernels` (`engine_workers.py:148,390,444`).
- **Probe order and wrappers.** Probe on the raw HF module (:659) after `apply_monkey_patch` (:355) and before PEFT
  (:663) / FSDP (:678); PEFT 0.19.1 and 0.21.2 forward `**kwargs`; transformers 5.16.1 and 5.21.0 both expose
  `logits_to_keep: int | Tensor` on Qwen3/Qwen2/Llama (5.21 via `slice_logits_to_keep`, 1-D long form kept).
- **Other engines unaffected.** `VeOmniEngineWithLMHead` has its own `_build_model_optimizer` (veomni :358) and
  `VeOmniEngineConfig` has no such key, so `getattr(self, "response_only_lm_head", False)` is False;
  `FSDPTurboEngineWithLMHead` inherits the build path and is refused by the `strategy == "fsdp_turbo"` guard;
  the critic (`FSDPEngineWithValueHead`, `_supports_response_only_lm_head = False`) warns and projects every row.
- **Config.** Dataclass default False, `engine/fsdp.yaml` key + comment, `ref/dp_ref.yaml` follows the actor via
  `oc.select`; all four `_generated_*` yamls regenerated into the temporary directory are identical to the committed ones.
- **Lint / title / docs.** ruff 0.12.2 `check` and `format --check` clean on the four Python files;
  `check_pr_title.py` valid; rst underline 46/46; `Last updated` bumped; commit trailers as required.
- **PR-body numbers.** #7686's "-8..-16 % step time at 16K->4K on A100" matches its body (7.94 % / 15.96 %);
  7.47 GB logits at 24576 x 151936 bf16 correct; K/T fractions correct except D9.

## 7. Fixes applied (author, 2026-10-29, new commits on `contrib/p1-lmhead-rows`, history kept)

| item | resolution | commit |
|---|---|---|
| D1 | gradients compared with `torch.testing.assert_close(rtol=1e-5, atol=1e-6)`; log-probs / entropy / Σπ² stay `torch.equal`. Checked on the reviewer's `[1]/[1]` layout: `lm_head.weight.grad` max abs 4.66e-9 (rel 6.6e-7), `embed_tokens.weight.grad` max abs 1.67e-6 (rel 4.5e-5 at a near-zero element): both within the tolerance; `[3,7,1,12,4]/[1,0,5,2,0]` and `[2,1]/[1,1]`: 0.0 | `6f6fc85a` |
| D2 | `_resolve_response_only_lm_head` reads `self.model_config.use_fused_kernels` and logs one rank-0 warning ("use_fused_kernels=True: the fused forwards project every row"), resolving False; the per-batch guard stays as a backstop. Docs, yaml comment, config docstring and PR body now say the same thing (start-up warning for the fixed layouts, silent per-batch fallback for top-K distillation / score centering / SFT batches) | `bc0e081f` (code), `4d36e7d2` (wording) |
| D3 | PR body: "since 4.49; verl pins `>=5.5.3`" | PR-BODY.md (outside the repo) |
| D4 | `slice_logits_to_keep` alias removed from the probe, the resolver message, docs, yaml, docstring, PR body and the tests (the probe test now checks that a renamed kwarg is *not* guessed) | `bc0e081f`, `4d36e7d2` |
| D5 | `loss_row_index`: `total = data["responses"].values().shape[0]`, assert dropped (consumer-side `no_padding_2_padding` keeps it); docstring states the prompt_len > 0 assumption and "no device-to-host sync"; the author's `test_rejects_an_empty_prompt` removed accordingly | `80aaef48` |
| D6 | locals renamed `keep_index` in `prepare_model_inputs` and `prepare_model_outputs` | `bc0e081f` |
| D7 | `if not distillation_only:` inside the scatter block removed | `bc0e081f` |
| D8 | commit `4bcbbeb9 [doc] docs:` kept (no history rewrite); noted in NOTES.md; the PR title (`feat`) is what CI checks on squash-merge; all new commits use fix/chore | NOTES.md |
| D9 | "7.4 % for GSM8K-style 80-token prompts with 1k responses" in docs and PR body | `4d36e7d2`, PR-BODY.md |
| D10 | `use_remove_padding=False` + flag raises `ValueError` at build time (after the value-head check, which keeps ignoring the flag with a warning); all warnings gated on `getattr(self, "rank", 0) == 0`; tests for the raise and for rank 1 staying silent | `bc0e081f` |
| lint | two review-fix lines exceeded 120 columns; wrapped by the all-files `ruff-format` hook | `b32c777e` |

Re-runs after the fixes (`packages/p1/logs/*-r2.log`, `*-r3.log`, `pre-commit-all-files*.log`): new test file 23 passed;
`tests/workers` 301 passed / 40 skipped / 1 deselected; `tests/utils` 473 passed / 23 skipped;
`pre-commit run --all-files`: first run ruff + ruff-format Failed on the two long lines (only file touched: the new
test), second run 14 hooks Passed, 0 Failed.

Reviewer tests (`packages/p1/review-tests/`, unchanged): 18 passed, 2 failed, both by construction of the test rather
than the branch: `test_single_sequence_single_response_token` asserts the gradients with `rtol=0, atol=1e-6`
(the review's own D1 text says that is not enough for M=1; the shipped test uses `rtol=1e-5`) and
`test_prompt_length_zero_is_rejected_not_misread` expects the assert that D5 asked to drop.
`test_docs_claim_a_start_up_warning_for_fused_kernels` (the D2 demonstration) now passes.

Section 3 (GPU checks) and section 4 items 1-2 (benchmark table, GPU test) remain for the benchmark phase.

## 8. Re-verification (reviewer, 2026-10-29, `contrib/p1-lmhead-rows` @ b32c777e, read-only)

### 8.1 Fix commits read in context

| commit | what it changes (verified at the lines) |
|---|---|
| 6f6fc85a | `tests/workers/test_response_only_lm_head_on_cpu.py:201-211`: forward quantities stay `torch.equal`, the two weight gradients use `assert_close(rtol=1e-5, atol=1e-6)` (D1); `test_rejects_an_empty_prompt` and the `slice_logits_to_keep` probe test removed; `_bare_engine` gains `model_config` |
| bc0e081f | `transformer_impl.py:106-117` single-name probe (D4); `:629-633` `ValueError` for `use_remove_padding=False` after the value-head check (D10); `:634-635` `self.model_config.use_fused_kernels` -> warning reason (D2); `:646-648` warning gated on `getattr(self, "rank", 0) == 0` (`self.rank` is set at `:160`); locals `keep_index` at `:1355-1358` and `:1498-1507` (D6); dead `if not distillation_only` removed, `:1555-1558` (D7); tests for the raise and for rank 1 silence (`:335-345`) |
| 80aaef48 | `padding.py:163` `total = data["responses"].values().shape[0]`, assert dropped, docstring states the prompt_len > 0 assumption (D5) |
| 4d36e7d2 | rst `:384-397`, `engine/fsdp.yaml:75-80`, `engine.py:272-279`: one wording (raise for padded inputs, rank-0 start-up warning for the fixed layouts, silent per-batch fallback for top-K / score centering / SFT); 7.4 % (D9) |
| b32c777e | ruff-format of the test file only |

PR-BODY.md: "since 4.49; verl pins `>=5.5.3`" (line 11, D3), "7.4 %" (line 13), the warning paragraph (line 64) agrees
with the code. D8 (`[doc] docs:` commit type) kept, as agreed: only the PR title is CI-checked.

### 8.2 Re-runs

- My 20 review tests: 18 passed, 2 failed, both by construction of my tests, not the branch:
  `test_single_sequence_single_response_token` asserts the gradients with `rtol=0, atol=1e-6` (my own D1 text says
  that is not enough for the M=1 layout; the shipped test uses `rtol=1e-5`, which passes it), and
  `test_prompt_length_zero_is_rejected_not_misread` expects the assert D5 asked to drop. The D2 demonstration
  `test_docs_claim_a_start_up_warning_for_fused_kernels` now passes (resolves False, one warning naming
  `use_fused_kernels=True`). Confirms the author's statement.
- Regression (my run, same harness): `tests/workers` 301 passed / 40 skipped / 1 deselected; `tests/utils` 473 passed /
  23 skipped. ruff 0.12.2 `check` + `format --check` clean on the four Python files; `check_pr_title.py` valid; rst
  underline 46/46; `Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>` on all five new commits; the four
  `_generated_*.yaml` regenerate identically (the fix commits changed comments only).

### 8.3 New adversarial cases (`packages/p1/review-tests/test_review_fixes_on_cpu.py`, 13 tests)

| test | targets | result |
|---|---|---|
| `TestSyncFreeSizing::test_all_prompt_micro_batch_with_pad_to_length_falls_back_identically` | D5 sizing with `responses.values().shape[0] == 0`, `pad_to_length` bucket 32 (pad 16) | PASS: empty long index, no `logits_to_keep`, 32-row padded forward, outputs equal to flag-off, pad stripped |
| `...::test_sized_from_the_shape_matches_the_offsets_on_sliced_micro_batches` | micro-batches cut with verl's own `tu.index_select_tensor_dict` (list, tensor, contiguous) and `tu.chunk_tensordict`: could a child keep the parent's values buffer? | PASS: both helpers rebuild nested tensors from unbound rows (`tensordict_utils.py:496-500`), so `values().shape[0] == sum(lengths)`, offsets start at 0, index equals brute force. (Raw `TensorDict[1:3]` on a jagged tensor raises `slice(): not supported` on torch 2.13, so that path cannot occur) |
| `...::test_no_device_to_host_sync_in_the_index_path` | `torch.Tensor.item/tolist/__bool__/__int__/__index__/__float__/numpy` monkeypatched to raise during `_response_only_lm_head_index` | PASS: no sync-shaped call in the index path (`numel()` is shape-based) |
| `TestBuildTimeGuardsAfterTheFixes::test_padded_inputs_raise_with_both_knobs_named` | D10 error path | PASS: `ValueError` names `use_remove_padding=True` and `response_only_lm_head=False` |
| `...::test_padded_inputs_do_not_raise_when_the_flag_is_off_or_on_the_critic` | ordering of the guards | PASS: flag off -> False silently; critic + padded -> one "vocabulary projection" warning, no raise |
| `...::test_raise_precedes_the_fused_kernels_warning` | both contradictions at once | PASS: raises |
| `...::test_warning_fires_once_on_rank_zero_only[rank0/rank7/no-rank-attr]` | rank gating | PASS: rank 0 -> one warning per call (two calls -> two lines), rank 7 -> none with the same False, missing attribute -> treated as rank 0 |
| `...::test_alias_is_gone_and_the_literal_key_is_used` | D4 | PASS: `slice_logits_to_keep` no longer probed; `model_inputs["logits_to_keep"]` is the literal key regardless of the attribute |
| `TestDroppedAssertCharacterization::test_empty_prompt_in_the_first_row` | what an empty prompt does after D5 | PASS (characterisation): index `[-1, 0, 4, 5]`; the HF slice wraps `-1` silently; `_scatter_rows` raises `IndexError: index_copy_(): index -1 is out of bounds` (`transformer_impl.py:126`) |
| `...::test_empty_prompt_in_a_middle_row_reads_the_same_wrong_row_as_the_full_path` | same, row i > 0 | PASS: both paths read the previous sequence's last row, exactly as `response_from_nested` does on main |
| `test_identity_still_holds_after_the_fixes` | 4-sequence layout, per-sample temperature, `pad_to_length`, Σπ² | PASS: 128 -> 32 rows; forward bitwise, gradients within 1e-6 |

Whole set after the fixes: 33 tests, 31 passed, 2 intended failures (the two listed in 8.2).

### 8.4 Residual findings

| # | severity | where | note |
|---|---|---|---|
| R1 | nit | `padding.py:163-170`, `transformer_impl.py:126` | With the assert gone, an empty prompt in row 0 fails as `IndexError` from `index_copy` on CPU and as a CUDA device-side assert on GPU, instead of a message naming the cause; rows i > 0 silently read the previous sequence's last row, which is what `response_from_nested` (`padding.py:209`) already does on main for the forward-only passes, while `no_padding_2_padding` (`padding.py:126`) still asserts on the update pass. Invalid input either way; no cheap sync-free check exists. Acceptable; mention "prompts must be non-empty" in the yaml comment if wanted |
| R2 | nit | `transformer_impl.py:620-627` docstring | says `use_remove_padding=False` "raises"; for the value-head engine it warns instead (the `_supports` check comes first, by design). One clause would make it exact |
| R3 | unchanged | section 3 | every GPU item stands: bf16 `max abs dlogp` flag on/off, flash-attn Triton CE on K rows with `inplace_backward`, FSDP1/FSDP2 + PEFT + checkpointing kwarg pass-through (hook on `lm_head`), `torch.compile` entropy recompiles with `pad_to_length`, BASELINE-A / BASELINE-A-LP timing and peak memory, MoE / remote-code model |
| R4 | process | `PR-BODY.md` | `### Benchmark` still TBD; no GPU test; `ci-request` pending (benchmark phase, as planned) |

### 8.5 Updated verdict: MERGE-READY-PENDING-GPU

D1-D7, D9 and D10 are fixed as described, D3 in the body, D8 accepted. No new defect was found in the fixes: the
sync-free sizing is correct on every way verl cuts a micro-batch, the raise and the rank-0 warning behave as
documented, and the identity (forward bitwise, gradients within tolerance) holds on the fixed HEAD. What remains
is the GPU evidence the PR body promises (section 3 / R3) and the benchmark table; file after those are in.

## 9. Fixes applied, re-verification nits (author, 2026-10-29, one commit, history kept)

| item | resolution | commit |
|---|---|---|
| R1 | `loss_row_index` raises `ValueError` naming the sequence index and its prompt / response lengths (plus the full length lists) when a prompt is empty; costs one device-to-host transfer per micro-batch with the flag on (`int(prompt_lens.min())`, lists only on failure), stated in the docstring. CPU test `test_an_empty_prompt_is_rejected_by_name` added | `a11ca00b` |
| R2 | resolver docstring: the value-head engine ignores the flag with a warning before the `use_remove_padding` check | `a11ca00b` |

Re-runs @ `a11ca00b` (`logs/p1-new-r4.log`, `review-tests-r3.log`, `reg-workers-r3.log`, `reg-utils-r3.log`): new test
file 24 passed; `tests/workers` 302 passed / 40 skipped / 1 deselected; `tests/utils` 473 passed / 23 skipped; ruff,
ruff-format and mypy Passed on the three changed files.

Reviewer tests (33, unchanged): 28 passed, 5 failed by construction of the tests: `test_no_device_to_host_sync_in_the_
index_path` (the named check is the one transfer R1 said cannot be avoided), `TestDroppedAssertCharacterization` (2:
characterise the pre-nit IndexError / silent wrong-row read; both inputs now raise the ValueError),
`test_prompt_length_zero_is_rejected_not_misread` (expects `AssertionError`; the input is rejected with `ValueError`),
`test_single_sequence_single_response_token` (rtol=0 gradient assertion, as in §7). R3 / R4 remain for the benchmark phase.

## 10. Review round 2 (the submitter)

Source: research/review-bundle/REVIEW.md "P1 performance is promising but numerical equivalence still needs a direct
check" and FIX-CHECKLIST.md "P1 loss row LM head" (2026-10-29). Author's round: NOTES.md "Review round 2".

| finding | fix commit | test proving it | status |
|---|---|---|---|
| Frozen-batch, same-weights bf16 GPU flag-on/off comparison: response log-probs, entropy, Σπ², gradients, one optimizer update; entropy-loss, PEFT and checkpointing cases; justified tolerances | `tests/workers/test_response_only_lm_head_on_gpu.py`: 2634019d (the test), d7d820aa (fsdp + fsdp2): {plain, gradient-checkpointing, lora rank 8} x {fsdp, fsdp2}, flag-off twice bitwise, deltas printed, tolerances justified in the docstring | the test itself (6 cases) | **GPU identity test: pending the bench run, file tests/workers/test_response_only_lm_head_on_gpu.py** (6 skipped on this Mac: no CUDA) |
| PEFT: does the probe see the raw module or the wrapper; does the kwarg reach the base model | The probe runs before `_build_lora_module` (no silent off) and PEFT's `**kwargs` passes the index on; 724d3d5e unwraps a `PeftModel` (`get_base_model()`) so the answer is order-independent | `test_probe_and_resolver_see_the_base_model_through_a_peft_wrapper`, `test_the_index_passes_through_a_peft_wrapper_to_the_base_model` (CPU, pass) | done |
| PR body: "no device-to-host sync" is wrong (`int(prompt_lens.min())` per micro-batch) | PR-BODY.md: the index is sync-free, the empty-prompt guard is one `.item()` per micro-batch; code unchanged (round-1 R1 decision) | n/a (wording) | done |
| PR body: measured response length / K:T are ~657 / 0.099, not 636 / 0.096 | PR-BODY.md, CHECKLIST.md; recomputed from the four table runs' metrics.jsonl (NOTES.md: 657.24 / 0.0987) | n/a | done |
| Training-phase time and memory separate from whole-step deltas | PR-BODY.md tables: phases + peak memory claimed; `gen` and step "not claimed" | n/a | done |
| Keep opt-in | PR-BODY.md says so and why | n/a | done |
| PR-body rules: shorter, GPU unit suite label, AI statement, refreshed duplicate check | PR-BODY.md rewritten; read-only gh searches 2026-10-29 13:19Z, five adjacent LM-head PRs named, none overlapping | n/a | done |

Re-runs (tests.log §8): new file 26 passed; full CPU regression 1510 passed / 64 skipped / 1 deselected / 0 failed;
`pre-commit run --all-files` 14 hooks Passed at 2634019d; d7d820aa touches only the GPU test file.

### 10.1 Round-2 re-review (REVIEW-ROUND2-VERDICT.md, NEEDS-FIXES on the GPU test design only)

| finding | fix commit | test proving it | status |
|---|---|---|---|
| D1 MEDIUM: `param/` after AdamW compared at 1e-2 of max\|ref\|; `lora_B` starts at zero so one noise-floor sign flip is a 2·lr delta (rel-Frobenius 2.4e-2 .. 1.25e-1 on a non-defect); same for small FSDP2 weights | 407082ec: plain SGD step (`optimizer="SGD"`, no decay, clipping disabled) so the update is exactly -lr·grad; `update/` under the gradient rule + 99 % sign agreement; `param/` bounded by 2·lr·1.01 + 1e-2·\|ref\| | review-tests/round2 `test_check_tolerates_adamw_sign_noise_on_lora_B`, `test_check_tolerates_one_sign_flip_in_a_trained_weight[2]`, `test_ulp_noise_on_the_head_backward_flips_lora_B_signs` (pass); CPU smoke: 2 % update sign flips and a 3·lr param delta fail | fixed |
| D2 MEDIUM: entropy / Σπ² tolerance (rtol=atol=1e-2) below one bf16 ulp of a top logit (measured dH 0.029, dΣπ² 0.0196) | 407082ec: per row rtol 2e-2 / atol 5e-2 on log-probs, entropy, Σπ²; batch means within that over sqrt(rows); loss rtol 1e-2 / atol 5e-3; documented in the docstring | `test_entropy_tolerance_covers_one_bf16_ulp_of_a_top_logit` (pass); CPU smoke: a +0.03 shift on 128 of 498 rows fails the mean rule, two one-ulp rows pass | fixed |
| D3 LOW: `gpu_unit_tests.yml` runs `pytest -s -x`; a silent 6-skip if the model is not linked | 407082ec: conservative tolerances + printed deltas (docstring, PR body, NOTES); model resolved from `~/models`, `/root/models` or the env var, skip reason names the candidates, the test prints the resolved path | 6 skipped here with the reason shown under `-rs`; `MODEL_PATH` resolves on this Mac | fixed |

Reviewer's round2 CPU tests after the fix: 13 passed / 0 failed (were 2 / 4 by design + 7). GPU identity test: still
pending the bench run (the queued d7d820aa run carries the AdamW rule; judge the 407082ec run).

### 10.2 GPU run 1 (lead, 2026-10-29) and the fp32-reference redesign

| finding | fix commit | test proving it | status |
|---|---|---|---|
| `gpu-identity-p1-2` (d7d820aa) and `-p1-3` (407082ec), 1x H100: 6/6 failed at the first gradient key: layer-0 rel-Frobenius 1.7e-2 (fsdp flat param) / 2.2e-2 (fsdp2 input_layernorm) / 1.7e-2 (lora down_proj.lora_B) vs 1e-2; forward log-probs and entropy bitwise, Σπ² 1.1e-7; later layers and `train/`, `loss/`, `update/`, `param/` unreported | 363470c1: fp32 reference arm (mixed_precision fp32, sdpa math kernel, TF32 off; must agree to 1e-4 and match the bf16 forward within bf16 noise); bf16 forward asserted as measured (exact; Σπ² 1e-6); bf16 grad / update 1e-1 per tensor (~4x measured) + element-wise 1e-1 of scale; `before/` bitwise; every tensor printed in a per-layer table before asserting; LoRA coverage asserted | CPU smoke of `_compare` (NOTES.md); reviewer round2 12/13 (the ulp-rule test fails by construction) | **pending the bench run `gpu-identity-p1-4`** |

### 10.3 GPU run 2 (gpu-identity-p1-4 @ 363470c1) and the final criterion

| finding | fix commit | test proving it | status |
|---|---|---|---|
| p1-4: 2 passed (lora x2) / 4 failed; forward bitwise in bf16 and fp32; fp32 gradients the same on every tensor (1.1e-5 .. 1.4e-5 per layer, worst 5.7e-5); bf16 arm-vs-arm 1.7e-2 (layer 0) -> 2.2e-3 (layer 23); the 4 failures were rule artefacts: fp32 across-arm UPDATE at 1.0e-4 .. 1.3e-4 (catastrophic cancellation of `param_after - param_before`) and the bf16 relative update rule on q/k biases (1.39e-1, ulp noise on small-norm tensors) | cf539906: bf16 gradients judged by distance to the fp32 truth (`D_on <= 2 D_off + 1e-6 |g32|` per tensor, both distances printed per layer); across-arm update check dropped for a within-arm SGD consistency check (4 fp32 ulp); bitwise forward and fp32 1e-4 proof kept; sign agreement printed only; docstring states the measured profile | CPU smoke (NOTES.md: equal-error arms pass, 3x-worse fails, fp32 5e-4 fails, 9-ulp residual fails); reviewer round2 12/13 (ulp-rule test by construction) | **MERGE-READY**: `gpu-identity-p1-5` @ cf539906 6 passed / 0 failed / 0 skipped (fp32 agreement <= 5.7e-5 per tensor; D_on/D_off <= 1.22; SGD residual <= 0.29 ulp; forward bitwise); stock suite `gpu-tests-p1-r2-final` 509 / 0 / 8 incl. the 6 identity cases, torchrun 2/2 |
| RESULTS.md stale 636 / 0.096 / 9.6 % (the submitter) | packages/p1/bench/RESULTS.md headline and memory line: 657 / 0.0987 / 9.9 % | n/a | done |

## 11. Round 3 (the independent follow-up): R2-03 -> fix -> test -> status

| finding | fix commit | test proving it | status |
|---|---|---|---|
| R2-03: `_compare` passed NaN (every check `delta > tol`); a skipped optimizer step (NaN grads, unchanged params, zero update) passed | e7c0d616: finiteness required on every inspected tensor of every arm and the truth before any check; step grad norm finite and > 0; update non-zero wherever the reference gradient is | `tests/workers/test_response_only_lm_head_oracle_on_cpu.py` (19 passed: 12 NaN injections, 2 reference/truth NaN, 2 skipped-step cases, bound/guard/floor/forward cases); reviewer round3 oracle 11/12 (the 12th characterizes the superseded guard-only rule) | fixed; GPU run pending (`gpu-identity-p1-6`) |
| Gradient vs update conflated (0.139 is an update error; worst bf16 gradient 0.0385 / 0.0317 LoRA); no direct bf16 gradient bound; guard described as identity | e7c0d616: direct bound 0.1 per tensor + 1e-6 layer floor retained next to the guard; docstring / PR body / NOTES / verdict addendum corrected; guard called a relative-error guard | oracle `test_the_direct_gradient_bound_rejects_what_the_guard_alone_accepts`, `..._absolute_floor...` | fixed |
| Bitwise forward asserted as portable | e7c0d616: ulp-justified row tolerance (rtol 2e-2 / atol 5e-2, mean tighter), maxima printed, bitwise reported as an observation | reviewer round2 one-ulp test passes again (13/13); oracle forward test | fixed |
| Scope: one rank, SGD, fresh zero-B LoRA, separate variants | e7c0d616: scope stated; `lora_B` seeded; `lora-ckpt` variant; two-rank plain cases (skip without two devices) | GPU run pending | fixed, run pending |
| Round-3 re-review D1-D5 (REVIEW-ROUND3-VERDICT.md): D1 HIGH two-rank rank 0 crashed in `ppo_loss` (log_prob padded to 256, the other fields to the local 40); D2 a key missing from the truth silently exempt; D3 the layer floor could hide a negated tensor; D4 empty shards crashed `_compare`; D5 all-zero rows counted as padding | 1dbc0c07: rank-local `max_response_len`, comparison view padded to the global width, `mask/response` per arm; missing truth keys reported; direct bound `0.1 |g_off| + 1e-12` with near-zero tensors listed; empty shards skipped with a note; row checks on the response mask, all-zero rows reported | reviewer round3 tests 12/12 (were 7/12), round2 13/13, branch oracle 22/22, the submitter's oracle 11/12 (by construction) | fixed; `gpu-identity-p1-6` (H100:2) pending |
| GPU runs 6-7: p1-6 @ e7c0d616 8 passed / 2 failed (the predicted two-rank D1 crash); p1-7 @ 1dbc0c07 10 failed: `_worker` line 537 subtracted the new boolean `mask/response` in the determinism-baseline loop (`RuntimeError: Subtraction ... two bool tensors`) | 8c340e47: `_assert_bitwise` (torch.equal on every key, max_abs only for floating tensors); oracle covers bool / empty / float baselines and `_response_mask` on rank-local batches | branch oracle 26/26; reviewer round3 12/12; round2 13/13; the submitter's oracle 11/12 (by construction) | **MERGE-READY**: `gpu-identity-p1-8` @ 8c340e47 10 passed / 0 failed / 0 skipped (forward exact; fp32 within 1e-4; bf16 gradients <= 3.59e-2 under the 0.1 bound; D_on/D_off <= 1.22); stock suite `gpu-tests-p1-r3-final` 513 / 0 / 8 incl. the 10 identity cases, torchrun 2/2 |
