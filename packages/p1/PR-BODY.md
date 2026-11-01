### What does this PR do?

On the default FSDP path (`use_remove_padding=True`, `use_fused_kernels=False`) the actor, old-log-prob and
reference forwards project the LM head, and run the log-prob / entropy math, for **every packed token**, while
every PPO consumer reads only the row before each response token (`no_padding_2_padding`, `response_from_nested`
in `verl/workers/utils/padding.py`). The `(tokens, vocab)` logits tensor is usually the largest activation of these
passes (24576 x 151936 x bf16 = 7.5 GB) and the head is 7-17 % of the forward FLOPs for 1.7B-8B models.

This PR adds `actor_rollout_ref.actor.fsdp_config.response_only_lm_head` (opt-in, default off). The engine hands
the HF model the loss-row index as `logits_to_keep` (the tensor form transformers has had since 4.49; verl pins
`>=5.5.3`), scores the K kept rows with their own labels and temperature, and `index_copy`s log-probs, entropy and
`sum_pi_squared` back into the packed layout, so nothing downstream changes. Head FLOPs, the logits tensor and the
CE / entropy work shrink by the prompt fraction of the micro-batch: 7.4 % for GSM8K-style 80-token prompts with 1k
responses, 33 % at 1k/2k, 86 % at 6k/1k, more for agentic prompts (#8299 reports 3-16x prompt:response ratios). On
CPU fp32 the kept rows are bitwise identical to the full projection; on GPU a GEMM over a different row count may
round differently at the bf16 noise level, which the GPU identity test below measures.

Related: #7686 (same idea and key name on Megatron; open, conflicting), #7370 (closed; the `use_fused_kernels=True`
torch backend only: this PR covers the eager default path and leaves the fused path alone), #7712 (RFC for *more*
outputs from `prepare_model_outputs`: orthogonal), #8300 (shared-prefix packing: composes; the index must be derived
from the packed layout once it lands).

### Checklist Before Starting

- [x] Similar PRs (read-only searches, 2026-10-29): [`logits_to_keep`](https://github.com/verl-project/verl/pulls?q=is%3Apr+logits_to_keep)
      (#7370 only, closed), [`lm head`](https://github.com/verl-project/verl/pulls?q=is%3Apr+%22lm+head%22) (#7686;
      the others, #8441 / #8312 / #8096 / #8097 / #8004, are fused-kernel, FSDP2-hook or FP32-head fixes on other
      paths), [`response_only_lm_head`](https://github.com/verl-project/verl/pulls?q=is%3Apr+response_only_lm_head)
      (#7686 only). Nothing of this kind has landed on the FSDP eager path (`git log -S logits_to_keep -- verl/workers/engine/fsdp/`).
- [x] Title `[fsdp, perf] feat: project the LM head only for loss rows` (`tests/special_sanity/check_pr_title.py`).
      Not `[BREAKING]`: a new optional key, default off.

### Test

CPU (collected by `cpu_unit_tests.yml`): `tests/workers/test_response_only_lm_head_on_cpu.py`, 26 tests, runs the
real `FSDPEngineWithLMHead.prepare_model_inputs` / `prepare_model_outputs` on a bare engine around a tiny random
Qwen3, flag on vs off: `torch.equal` on response log-probs (both consumers), entropy and `sum_pi_squared`;
`assert_close(rtol=1e-5, atol=1e-6)` on the head and embedding gradients (their GEMMs reduce over K instead of T
rows; observed 0.0 on these layouts, 5.6e-7 relative on a 1-token layout); the head projects 32 of 105 rows (32 of
128 under `pad_to_length`); zeros on the unread rows; `loss_row_index` against the consumers' slicing rule; every
per-batch and build-time guard; the kwarg probe on the HF model and through a PEFT wrapper, and the index reaching
the base model through PEFT's `**kwargs`; the error raised when a forward ignores the index.

GPU identity, **ten cases, passing**: `tests/workers/test_response_only_lm_head_on_gpu.py` (one CUDA
device, two for the two-rank cases; Qwen2.5-0.5B under `~/models` or `/root/models`, or `VERL_RESPONSE_ONLY_LM_HEAD_MODEL`;
skips with the paths named otherwise; verl's stock GPU unit-test selection collects it). Cases: plain, gradient
checkpointing, LoRA rank 8 (`lora_B` seeded with small non-zero values so `lora_A` gets a first-backward gradient) and LoRA
with checkpointing, each on FSDP1 (`fsdp`, the default) and FSDP2 on one rank, plus the plain case on two ranks (FULL_SHARD;
rank r keeps sequences r::2 with its own rank-local `max_response_len`, every rank compares its own shard). One frozen packed micro-batch (six sequences, 1- to 520-token prompts, 1- to 256-token
responses), same checkpoint and seed; five engines per case: bf16 flag off, bf16 flag off again (bitwise baseline), bf16
flag on, and an fp32 reference (`mixed_precision` fp32, sdpa on torch's math kernel, TF32 off) with the flag off and on.
Scope: one model and shape, plain SGD rather than AdamW, a fresh adapter: a numerical comparison on this workload, not a
proof for every GPU or GEMM implementation.

Measured on 1x H100 (torch 2.13 cu130, transformers 5.16.1, flash-attn 2.8.3; `bench/results/gpu-identity-p1-5`, the six
one-rank cases before the seeded-LoRA and two-rank additions): response log-probs, entropy and the loss were **bitwise
identical with the flag on and off, in bf16 and in fp32**, an observation on this device and shape rather than a
requirement (asserted per row at rtol 2e-2 / atol 5e-2, one bf16 ulp of a top logit, the batch mean tighter), and
`sum_pi_squared` within 1.1e-7; **fp32 gradients agree on every tensor, 1.1e-5 .. 1.4e-5 relative Frobenius pooled per
layer (worst tensor 5.7e-5)**: the K-row and T-row GEMMs are the same math, so the gradient is the same; bf16 gradients
differ by 1.7e-2 pooled at layer 0 falling to 2.2e-3 at layer 23, worst single tensor 3.9e-2 (3.2e-2 for LoRA): bf16
re-rounding of the head's backward through 24 layers, while the bf16 arm's own distance from fp32 is 2.9e-2 nat on the
log-probs. Asserted on the gradients: the direct bound `|g_on - g_off| <= 0.1 |g_off| + 1e-12` per tensor (~2.5x the worst
measured; a negated tensor never passes; tensors below 1e-6 of their layer's largest gradient norm are listed) and the
relative-error guard `|g_on - g_fp32| <= 2 |g_off - g_fp32| +
1e-6 |g_fp32|` (it admits on/off differences up to the baseline's own bf16 error, measured ratios 0.97 .. 1.22, so it is not
an identity statement by itself); fp32 gradients within 1e-4 per tensor; inside every arm `param_after = param_before -
lr * grad` within 4 fp32 ulp, a finite positive step gradient norm and a non-zero update wherever the gradient is; weights
before the step bitwise; every inspected tensor finite before any comparison (a key missing from the fp32 reference, an
all-zero set of response rows and a differing response mask are failures; an empty FSDP2 shard is skipped with a note; the
row checks run on the response mask); the fp32 reference within bf16 noise of the bf16 forward. Every tensor is printed in
a per-layer table. A CPU oracle, `tests/workers/test_response_only_lm_head_oracle_on_cpu.py` (26 tests), injects NaN,
skipped optimizer steps, out-of-bound and negated gradients, missing references, empty shards and all-zero rows.

Result (`gpu-identity-p1-8` @ 8c340e47, H100:2, 321 s): **10 passed / 0 failed / 0 skipped**; the forward rows agree exactly
in every case (e.g. two-rank `fsdp` rank 0: `forward/log_probs max_abs=0 rows=50`), the fp32 arms agree per tensor within
the 1e-4 bound, and the flag-off-twice baseline is bitwise on every case (110 tensors per rank on two-rank `fsdp`). The
bf16 on-vs-off gradient difference is at most 1.68e-2 (`fsdp`), 3.59e-2 (`fsdp2`) and 1.44e-2 (two-rank) relative
Frobenius per tensor, under the 0.1 direct bound, and the distance-to-fp32 ratio D_on/D_off at most 1.02 (`fsdp`), 1.22
(`fsdp2`), 1.11 (`lora`, `lora-ckpt`) and 1.01 / 1.13 (two-rank `fsdp` / `fsdp2`), e.g. two-rank `fsdp` layer 0 D_off 1.450e-1 vs
D_on 1.448e-1. The SGD residual is 0.217 ulp in both two-rank arms and the fp32-vs-bf16 flag-off log-probs differ by 4.7e-2
nat mean (2.3e-1 max) there, i.e. the two-rank sharded comparison (FULL_SHARD, shard-local) reads like the one-rank ones.

```
pytest -o python_files=*_on_cpu.py --asyncio-mode=auto tests/workers/test_response_only_lm_head_on_cpu.py  # 26 passed
pytest -s tests/workers/test_response_only_lm_head_on_gpu.py
```

CPU regression on the branch (macOS harness, megatron-importing files excluded): tests/workers 304 passed / 40 skipped
(302 + the 2 PEFT tests), tests/utils 473 / 23, tests/trainer 438 / 1, models + protocol + checkpoint + tools 200,
tests/experimental 74, tests/single_controller 21: unchanged vs main apart from the new tests.

### API and Usage Example

```bash
python3 -m verl.trainer.main_ppo ... \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True   # the ref follows the actor
    # actor_rollout_ref.ref.fsdp_config.response_only_lm_head=False  # to override it for the ref
```

`use_remove_padding=False` with the flag on is rejected at start-up. Where the layout rules the flag out, the engine
logs one warning on rank 0 at start-up and keeps the full projection: `use_fused_kernels=True` (the fused forwards
project every row), `ulysses_sequence_parallel_size > 1`, `fsdp_turbo`, vision-language models, the critic (value
head), and models whose `forward` does not accept `logits_to_keep`. Per micro-batch it silently keeps the full
projection for top-K distillation, score centering and batches without `responses` (SFT). Engine-based FSDP workers only.

### Design & Code Changes

- `verl/workers/config/engine.py`, `verl/trainer/config/engine/fsdp.yaml`, `ref/dp_ref.yaml` (`oc.select` from
  the actor), regenerated `_generated_ppo_trainer.yaml`: the key (same name as #7686).
- `verl/workers/utils/padding.py`: `loss_row_index(data)`: the rows `[e_i - r_i - 1, e_i - 1)` per sequence, i.e.
  exactly what `no_padding_2_padding` / `response_from_nested` read, vectorised. The index itself is built without
  host syncs; the empty-prompt guard costs one device-to-host transfer per micro-batch (`int(prompt_lens.min())`)
  to raise a named `ValueError`, negligible next to the forward (the engine already syncs here under `pad_to_length`).
- `verl/workers/engine/fsdp/transformer_impl.py`:
  - build time, next to the `cu_seqlens` probe: `_probe_logits_to_keep_kwarg(module)` on the HF module before the
    PEFT / FSDP wrap (a PEFT wrapper is read through `get_base_model()`; a bare `**kwargs` does not count) and
    `_resolve_response_only_lm_head(module)`, which raises for `use_remove_padding=False` and otherwise turns the
    flag off with one rank-0 warning for the value head, fused kernels, Ulysses SP, `fsdp_turbo`, VLMs and a
    `forward` without `logits_to_keep`;
  - `prepare_model_inputs` (rmpad branch, after the static pad): `_response_only_lm_head_index` returns the index
    unless fused kernels / distillation / score centering / no `responses` / empty, then `model_inputs["logits_to_keep"]
    = index` and `output_args["loss_row_index"] = index`;
  - `prepare_model_outputs` (eager rmpad branch): index `input_ids_rmpad_rolled` / `temperature_rmpad` by the kept
    rows, raise if the model returned a different row count, compute Σπ² / entropy / log-probs on K rows,
    `_scatter_rows` each into zeros of the local packed length before the existing gather / unpad / re-jag.
- `docs/perf/perf_tuning.rst`: item 11, when to use it and what it skips.

Out of scope (follow-ups): Ulysses SP (rank-local index + empty-rank guard), the fused path (#7370), mask-aware
dropping of tool-observation rows inside `responses`, prefix packing (#8300).

### Benchmark

Modal harness (`bench/README.md`): 8x H100 80GB, image `verlai/verl:uv.cu130.dev5` (torch 2.13.0+cu130, vllm 0.32.0,
transformers 5.16.1, flash-attn 2.8.3), Qwen3-1.7B, GRPO n=8, `use_kl_loss=True`, vLLM async rollout TP=1, dynamic
bsz, gradient checkpointing, lr 1e-6, 6 steps. Pooled means over steps 2..6 of all runs of an arm; ± = half the
max-min spread across runs. Flag on the actor (the ref follows through `oc.select`); the OFF arm sets it to `False`
on the same tree. Raw records `bench/results/<run>/{summary.json,metrics.jsonl,train.log}`, cross-checked by two
independent readings (`bench/CROSSCHECK.md`, `packages/p1/bench/RESULTS.md`).

The claim is the **training phases** this PR touches (old_log_prob + ref + update_actor) and **peak memory**. `gen`
and the whole step are listed separately and not claimed: generation varies ~10 % between identical runs on this
harness, so whole-step deltas mix the feature's effect with that noise.

**Long-prompt preset** (synthetic GSM8K with prompts padded to ~6000 tokens, `bench/build_gsm8k_long.py`; 64 prompts
x n=8, `ppo_mini_batch_size=16`, prompt 6144 / response 1024, `ppo_max_token_len_per_gpu=32768`). Measured over
steps 2..6, pooled over the main and ON runs: prompt 5999 and response 657 tokens on average, K/T = 0.099.

| arm | runs | old_log_prob (s) | ref (s) | update_actor (s) | training phases (s) | peak mem (GB/rank) | gen (s) | step (s) |
|---|---|---|---|---|---|---|---|---|
| main @ 2ab35292 | 2 | 8.06 ± 0.03 | 6.59 ± 0.00 | 25.86 ± 0.06 | 40.52 ± 0.10 | 16.56 | 17.02 ± 0.04 | 62.16 ± 0.07 |
| branch, flag OFF (b32c777e) | 1 | 8.06 | 6.55 | 26.00 | 40.61 | 16.57 | 16.61 | 61.87 |
| branch, flag ON (b32c777e, a11ca00b) | 2 | 6.19 ± 0.03 | 5.84 ± 0.02 | 23.65 ± 0.08 | 35.68 ± 0.10 | 11.91 | 15.77 ± 0.20 | 54.08 ± 0.12 |
| ON vs main | | -23 % | -11 % | -9 % | **-12 %** | **-4.65 GB (-28 %)** | not claimed | not claimed |

Flag OFF equals main (40.61 vs 40.52 s, 16.57 vs 16.56 GB): a no-op when disabled. Peak memory is
`actor/perf/max_memory_allocated_gb`; the bf16 logits of a 32768-token micro-batch are 9.96 GB and shrink to 9.9 %
of that. Training metrics ON vs OFF agree within run noise (`actor/entropy` 0.134 / 0.134, `pg_clipfrac` 0.002 /
0.002, `ppo_kl` 0.000, `kl_loss` 0.002, `rollout_probs_diff_mean` 0.005): consistent with the identity, not a proof
of it; the GPU test above is.

**BASELINE-A** (GSM8K, ~83-token prompts, 128 prompts x n=8, `ppo_mini_batch_size=32`, prompt 1024 / response 2048,
`ppo_max_token_len_per_gpu=24576`; K/T = 0.94): the no-regression reference. Exploratory: two runs per arm and only
6 % of the head work to skip.

| arm | runs | old_log_prob (s) | ref (s) | update_actor (s) | training phases (s) | peak mem (GB/rank) | gen (s) | step (s) |
|---|---|---|---|---|---|---|---|---|
| main @ 2ab35292 | 2 | 3.52 ± 0.03 | 3.41 ± 0.02 | 8.82 ± 0.11 | 15.74 ± 0.11 | 14.66 | 12.58 ± 0.19 | 31.25 ± 0.10 |
| branch, flag ON (4bcbbeb9, b32c777e) | 2 | 3.43 ± 0.05 | 3.05 ± 0.01 | 8.08 ± 0.01 | 14.56 ± 0.06 | 14.54 | 10.49 ± 0.02 | 28.89 ± 0.35 |
| ON vs main | | -3 % | -11 % | -8 % | -7.5 % | -0.12 GB | not claimed | not claimed |

No training phase is slower with the flag on. The `gen` difference (12.58 vs 10.49 s) is harness noise: this PR does
not touch generation. Correctness columns are identical across the four runs.

**GPU unit tests**: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and
`tests/experimental`, so it does not execute this PR's CPU tests; it does collect the GPU identity test), at the final sha
8c340e47 (`gpu-tests-p1-r3-final`, H200:8): **513 passed / 0 failed / 8 skipped** (the 503 stock cases, identical to main's
503 / 0 / 8, plus the 10 identity cases; junit 539 cases incl. subtests: 531 passed / 0 failed / 8 skipped) and the two
torchrun TP scripts 2/2. Earlier: 503 / 0 / 8 at 4bcbbeb9 and b32c777e, 509 / 0 / 8 at cf539906 (six identity cases).
Enabled-feature GPU evidence: the identity test above and the A/B runs.

**Not measured**: models larger than 1.7B, MoE or `trust_remote_code` models, Ulysses SP (guarded off),
`torch.compile` entropy recompiles under `pad_to_length`, NPU.

The flag stays **opt-in** in this PR: the identity is proven on one model family and size (Qwen2.5-0.5B; Qwen3-1.7B in
the A/B runs), and the gain scales with the prompt fraction (7 % of the head work on short-prompt GSM8K, 86 % at 6k/1k).
A follow-up can flip the default after exposure on more models.

Suggested reviewers (no mention until the submitter decides): tardis-key (reviewed the Megatron twin #7686) and
wuxibin89 (owner of `transformer_impl.py` and `padding.py`). Co-authorship is offered to the author of #7686 for
the shared key and design.

### AI assistance

Written with Claude (Anthropic) under the author's direction. Commits carry a Co-authored-by trailer.

### Checklist Before Submitting

- [x] Read the Contribute Guide.
- [x] Apply pre-commit checks: `pre-commit run --all-files --show-diff-on-failure` passes (14 hooks).
- [x] Add / update the documentation (`docs/perf/perf_tuning.rst`).
- [x] Add unit test(s) to the CI workflow (`tests/workers/*_on_cpu.py`, CPU job; `*_on_gpu.py`, GPU job).
- [ ] CI request in `ci-request` once the PR is ready.
- [x] Not a `recipe` change.
