# P1 loss-row-only LM head: Modal GPU evidence (bench/README.md harness)

Flag: `actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True` (the ref engine follows the actor through
`oc.select`; the OFF arm sets it to `False` explicitly on the same tree). Image, lock file and data identical in all arms
(`pip-freeze.txt`, `manifest.txt`, `digest_match=True`); full commands in `bench/results/<run>/command.txt`.
Reading rules (bench/README.md "A/B results"): the training-side phases (`timing_s/old_log_prob`, `ref`,
`update_actor`, their sum) are the metrics; `timing_s/step` and `gen` are reported separately because `gen` varies ~10 %
between identical runs. Means over steps 2..6; peak mem = `actor/perf/max_memory_allocated_gb` per rank.

## Headline: `long-prompt` preset (8x H100, Qwen3-1.7B, GRPO n=8, 64 prompts x 8, prompt 6144 / response 1024, dynamic bsz 32768)

Synthetic GSM8K with ~6000-token prompts (`bench/build_gsm8k_long.py`, Project Gutenberg #1342 filler, question last);
measured over steps 2..6, pooled over the main and ON runs: K/T = response / (prompt + response) = **0.0987** (prompt 5999, response 657 tokens on average: GSM8K answers under a
1024-token cap), i.e. the LM head is projected for 9.9 % of the rows when the flag is on.

| run | tree (sha) | old_log_prob | ref | update_actor | train sum | step | gen (noise) | peak mem GB |
|-----|------------|--------------|-----|--------------|-----------|------|-------------|-------------|
| long-prompt-main-1 | main@2ab35292 | 8.098 | 6.595 | 25.928 | 40.62 | 62.24 | 16.99 | 16.56 |
| long-prompt-p1-off-1 | contrib/p1-lmhead-rows@b32c777e (9 commits), flag OFF | 8.055 | 6.552 | 26.000 | 40.61 | 61.87 | 16.61 | 16.57 |
| long-prompt-p1-on-2 | contrib/p1-lmhead-rows@b32c777e, flag ON | **6.158** | **5.857** | **23.577** | **35.59** | 54.20 | 15.96 | **11.90** |
| ON vs OFF (same tree) | | **-23.6 %** | **-10.6 %** | **-9.3 %** | **-12.4 %** | -12.4 % | -4 % | **-4.67 GB (-28 %)** |
| long-prompt-main-2 | main@2ab35292 | 8.030 | 6.593 | 25.801 | 40.42 | 62.09 | 17.06 | 16.56 |
| long-prompt-p1-on-3 | p1@a11ca00b (10 commits, final HEAD), flag ON | **6.224** | **5.825** | **23.730** | **35.78** | 53.96 | 15.57 | **11.92** |
| main mean +- spread (main-1, main-2) | | 8.06 +- 0.03 | 6.59 +- 0.00 | 25.86 +- 0.06 | 40.52 +- 0.10 | 62.16 +- 0.07 | 17.02 +- 0.04 | 16.56 |
| p1 ON mean +- spread (on-2, on-3) | | 6.19 +- 0.03 | 5.84 +- 0.02 | 23.65 +- 0.08 | 35.68 +- 0.10 | 54.08 +- 0.12 | 15.77 +- 0.20 | 11.91 |
| **ON vs main (2 runs each)** | | **-23.2 %** | **-11.4 %** | **-8.5 %** | **-11.9 %** | -13.0 % | -7 % | **-4.65 GB (-28 %)** |

Two runs per arm agree within +-0.4 % on every training-side phase (main-1/main-2 and on-2/on-3 were launched 30 min
apart on different nodes), so the -11.9 % / -4.65 GB headline is well outside the noise; the OFF arm sits on the main
numbers (40.61 vs 40.52 s), i.e. the branch is a no-op when disabled.

Shas: baseline-a-p1-on-1 and gpu-tests-p1-1 ran at 4bcbbeb9 (4 commits); long-prompt-p1-on-2, long-prompt-p1-off-1,
baseline-a-p1-on-2 and gpu-tests-p1-2 at b32c777e (9 commits); the branch then gained one small final commit
(a11ca00b, 10 commits) from which long-prompt-p1-on-3 was launched. Every row's sha is in its `meta.json`
(`src_git`, `src_head`, `src_commits_ahead_of_main`).

- OFF equals main on every column (8.06 / 6.55 / 26.00 vs 8.10 / 6.60 / 25.93 s; 16.57 vs 16.56 GB): the branch is a
  no-op when disabled.
- ON: `old_log_prob` -23.6 % (a pure forward: the head is 2HV = 6.2e8 of ~3.8e9 FLOP/token at 6.6k-token sequences, and
  the (T,V) log-softmax/entropy over 32768 x 151936 is gone with it), `ref` -10.6 %, `update_actor` -9.3 % (forward +
  recompute + backward carry the head; optimizer step and all-gathers do not), training-side sum **-12.4 %**, in line
  with the vet's -10..-13 % expectation; peak memory per rank **16.57 -> 11.90 GB (-4.67 GB)**: the bf16 logits for a
  32768-token micro-batch are 9.96 GB and shrink to 9.9 % of that.
- Correctness ON vs OFF: `actor/entropy` 0.134 / 0.134, `pg_clipfrac` 0.002 / 0.002, `ppo_kl` 0.000, `kl_loss` 0.002,
  `training/rollout_probs_diff_mean` 0.005, reward 0.130 / 0.127 (within the run-to-run noise of 64-prompt batches).
- `step` -12.4 % (62 -> 54 s) and `gen` -4 % are reported outside the claim (rollout noise ~10 %).

Warm-up step 1 (not in the means, but the same direction): main 14.85 / 15.29 / 24.07 s and 16.02 GB; OFF 14.96 / 15.19 /
23.87 s and 16.27 GB; ON 11.68 / 11.25 / 22.27 s and 11.72 GB.

## No-regression: BASELINE-A (K/T = 0.94, prompt 1024 / response 2048, 128 x 8; the head is ~17 % of forward FLOPs, expected -1 %)

| run | tree (sha) | old_log_prob | ref | update_actor | train sum | step | gen (noise) | peak mem GB |
|-----|------------|--------------|-----|--------------|-----------|------|-------------|-------------|
| baseline-a-main-1 | main@2ab35292 | 3.491 | 3.430 | 8.709 | 15.63 | 31.15 | 12.77 | 14.70 |
| baseline-a-main-2 | main@2ab35292 | 3.547 | 3.384 | 8.922 | 15.85 | 31.35 | 12.39 | 14.62 |
| baseline-a-p1-on-1 | p1@4bcbbeb9 (4 commits), ON | 3.484 | 3.059 | 8.069 | 14.61 | 29.23 | 10.51 | 14.51 |
| baseline-a-p1-on-2 | p1@b32c777e (9 commits), ON | 3.379 | 3.036 | 8.086 | 14.50 | 28.54 | 10.47 | 14.56 |
| main mean +- spread | | 3.52 +- 0.03 | 3.41 +- 0.02 | 8.82 +- 0.11 | 15.74 +- 0.11 | 31.25 +- 0.10 | 12.58 +- 0.19 | 14.66 |
| p1 ON mean +- spread | | 3.43 +- 0.05 | 3.05 +- 0.01 | 8.08 +- 0.01 | 14.56 +- 0.05 | 28.89 +- 0.35 | 10.49 +- 0.02 | 14.54 |

No regression: every training-side phase is equal or faster with the flag on (ref -10 %, update_actor -8 %,
old_log_prob -3 %, train sum -7.5 %); two runs per arm agree within +-1 % on the branch and +-1.5 % on main. The
branch arms also show a lower `gen` (10.5 vs 12.6 s, both runs): the rollout is not touched by this change, so the
`gen` column stays outside the claim (host / vLLM scheduling variance on this harness), and the -2.4 s step-time delta
is reported as "at least the -1.2 s of the training phases". Correctness columns are identical across the four runs:
`actor/entropy` 0.203 / 0.202 / 0.204 / 0.206, `pg_clipfrac` 0.000, `ppo_kl` 0.000, `kl_loss` 0.001-0.002,
`training/rollout_probs_diff_mean` 0.006, reward ~0 (6 steps of a base model on GSM8K: timing runs).

## GPU unit tests

`uv run --frozen --all-packages --extra sglang --extra megatron --extra ci pytest -s <gpu_unit_tests.yml ignores> tests/`
+ torchrun `test_special_linear_cross_entropy_tp` / `test_special_megatron_kl_loss_tp`, 8x H100; main reference
`gpu-tests-main-2`: 503 / 0 / 8, torchrun 2/2.

| run | tree | pytest | torchrun | new failures |
|-----|------|--------|----------|--------------|
| gpu-tests-p1-1 | p1@4bcbbeb9 (4 commits) | **503 passed / 0 failed / 8 skipped**, 26 min 41 s, rc=0 | 2/2 rc=0 | none |
| gpu-tests-p1-2 | p1@b32c777e (9 commits; final HEAD a11ca00b adds one docs/test commit) | **503 / 0 / 8**, 33 min 6 s, rc=0 | 2/2 rc=0 | none |

Apps (stopped by id, the launch registry (kept privately)): baseline-a-main-1 <modal-app>, baseline-a-main-2
<modal-app>, baseline-a-p1-on-1 <modal-app>, baseline-a-p1-on-2 <modal-app>,
gpu-tests-p1-1 <modal-app>, long-prompt-main-1 <modal-app>, long-prompt-p1-on-2
<modal-app>, long-prompt-p1-off-1 <modal-app> (the H200 launches of the same three arms,
<modal-app> / <modal-app> / <modal-app>, never got a GPU and were stopped in the
queue). Results: `bench/results/<run>/{summary.json,metrics.jsonl,train.log,command.txt,meta.json}`.

## Review round 2: GPU identity test (bench/README.md harness, 2026-10-29) -- run on d7d820aa, superseded by 407082ec (redesigned tolerances; re-run `gpu-identity-p1-3`)

`tests/workers/test_response_only_lm_head_on_gpu.py` (frozen packed micro-batch, flag on vs flag off on the real
`FSDPEngineWithLMHead`, bf16 mixed precision, {plain, gradient-checkpointing, lora r8} x {fsdp, fsdp2}), first ever GPU run,
on `contrib/p1-lmhead-rows@d7d820aa` (detached worktree, `meta.json` `digest_match=True`), 1x H100 80GB (the test is a
single `mp.spawn(nprocs=1)` process; the H100:8 copy `gpu-identity-p1-1` never got a node and was withdrawn):
run `gpu-identity-p1-2`, app <modal-app>, `pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`,
`~/models/Qwen/Qwen2.5-0.5B` linked from the hf-cache. Log: `bench/results/gpu-identity-p1-2/pytest.log` (not patched).

Result: **6 failed / 0 passed in 235.8 s** (rc=1). What the log measures before each assertion fires:

| variant | flag-off twice | forward log_probs | forward entropy | forward sum_pi_squared | first gradient key over `REL_FRO_TOL = 1e-2` |
|---------|----------------|-------------------|-----------------|------------------------|-----------------------------------------------|
| fsdp/plain | bitwise identical (82 tensors) | max_abs 0 | max_abs 0 | max_rel 3.804e-06 | `grad/_fsdp_wrapped_module.model.layers.0._fsdp_wrapped_module._flat_param`: relative Frobenius difference **1.694e-02** |
| fsdp2/plain | bitwise identical (877 tensors) | max_abs 0 | max_abs 0 | max_rel 3.804e-06 | `grad/model.layers.0.input_layernorm.weight`: **2.201e-02** |
| fsdp/gradient-checkpointing | bitwise identical (82) | 0 | 0 | 3.804e-06 | layer-0 `_flat_param`: **1.694e-02** |
| fsdp2/gradient-checkpointing | bitwise identical (877) | 0 | 0 | 3.804e-06 | `layers.0.input_layernorm.weight`: **2.201e-02** |
| fsdp/lora | bitwise identical (1015) | 0 | 0 | 3.804e-06 | `...layers.0...mlp.down_proj.lora_B.default..._flat_param`: **1.676e-02** |
| fsdp2/lora | bitwise identical (1015) | 0 | 0 | 3.804e-06 | `grad/base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight`: **1.676e-02** |

Reading: the forward-only quantities are identical to the bit (the kept-row LM head reproduces log-probs and entropy
exactly on this batch); the determinism baseline holds (flag off twice = bitwise); the assertion that fails is the
gradient check at layer 0, the first key the test visits, at 1.7-2.2 % relative Frobenius difference against a 1 % bound.
The test stops at the first key, so the per-layer profile and the optimizer-step comparison were not printed. The P1
author is redesigning the tolerances (lead, 14:05Z); the test is re-run on the next p1 sha as `gpu-identity-p1-3`.

## Review round 2: GPU identity test on the FINAL tree (p1@407082ec, "linear optimizer step and ulp-justified tolerances")

Run `gpu-identity-p1-3` (app <modal-app>, 1x H100 80GB, detached worktree `wt/bench-p1b`, `digest_match=True`),
`pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`: **6 failed / 0 passed / 0 skipped in 246.9 s** (rc=1).
Log: `bench/results/gpu-identity-p1-3/pytest.log` (not patched). The measured deltas before each assertion:

| variant | flag-off twice | forward log_probs / entropy (498 rows) | forward sum_pi_squared | first gradient key over the tolerance |
|---------|----------------|----------------------------------------|------------------------|---------------------------------------|
| fsdp/plain | bitwise identical (82 tensors) | max_abs 0, mean_delta 0 | max_abs 1.080e-07, mean_delta 2.826e-10 | layer-0 `_flat_param`: relative Frobenius difference **1.694e-02** |
| fsdp2/plain | bitwise identical (877) | 0 / 0 | 1.080e-07 / 2.826e-10 | `model.layers.0.input_layernorm.weight`: **2.201e-02** |
| fsdp/gradient-checkpointing | bitwise identical (82) | 0 / 0 | 1.080e-07 / 2.826e-10 | layer-0 `_flat_param`: **1.694e-02** |
| fsdp2/gradient-checkpointing | bitwise identical (877) | 0 / 0 | 1.080e-07 / 2.826e-10 | `layers.0.input_layernorm.weight`: **2.201e-02** |
| fsdp/lora | bitwise identical (1015) | 0 / 0 | 1.080e-07 / 2.826e-10 | `...layers.0...down_proj.lora_B.default..._flat_param`: **1.676e-02** |
| fsdp2/lora | bitwise identical (1015) | 0 / 0 | 1.080e-07 / 2.826e-10 | `...layers.0.mlp.down_proj.lora_B.default.weight`: **1.676e-02** |

Same numbers as on d7d820aa (the gradient values are deterministic across the two runs): the failing rule is still the
gradient relative-Frobenius check at the first key the test visits (layer 0), 1.7-2.2 % against the bound; the per-layer
profile and the optimizer-step comparison are not reached. Routed to the P1 author by the lead.

## Review round 2: GPU identity test on p1@363470c1 ("fp32 reference arm and per-layer report"), FINAL

Run `gpu-identity-p1-4` (app <modal-app>, 1x H100 80GB, detached worktree `wt/bench-p1c`, `digest_match=True`),
`pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`: **2 passed (lora-fsdp, lora-fsdp2), 4 failed (plain-fsdp,
plain-fsdp2, gradient-checkpointing-fsdp, gradient-checkpointing-fsdp2), 0 skipped, 215.7 s** (rc=1). Log:
`bench/results/gpu-identity-p1-4/pytest.log` (not patched).

What holds in every case, bf16 and fp32 arms alike: flag-off twice bitwise identical (107 / 902 / 1040 tensors); forward and
training-forward `log_probs`, `entropy` and `loss/ppo` max_abs 0; `sum_pi_squared` max_abs 1.08e-07 (bf16) / 9.31e-08 (fp32);
and the reference line, verbatim for fsdp/plain (identical in all six cases):

    [response_only_lm_head][fsdp/plain] fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01

i.e. the bf16 arm differs from the fp32 arm of the SAME flag-off model by 2.9e-2 on average, three orders of magnitude more
than any flag-on/off delta below. Per-layer gradient deltas, fsdp/plain (`grad layer NN: max rel_fro / pooled`):

| layer | bf16 flag on vs off | fp32 flag on vs off | layer | bf16 | fp32 |
|---|---|---|---|---|---|
| 0 | 1.694e-02 | 1.236e-05 | 12 | 1.171e-02 | 1.415e-05 |
| 1 | 1.411e-02 | 1.361e-05 | 13 | 1.031e-02 | 1.385e-05 |
| 2 | 1.326e-02 | 1.075e-05 | 14 | 1.047e-02 | 1.565e-05 |
| 3 | 1.338e-02 | 1.139e-05 | 15 | 1.048e-02 | 1.646e-05 |
| 4 | 1.393e-02 | 1.347e-05 | 16 | 8.993e-03 | 1.792e-05 |
| 5 | 1.306e-02 | 1.341e-05 | 17 | 7.776e-03 | 1.788e-05 |
| 6 | 1.278e-02 | 1.316e-05 | 18 | 8.355e-03 | 1.377e-05 |
| 7 | 1.464e-02 | 1.468e-05 | 19 | 7.569e-03 | 1.432e-05 |
| 8 | 1.276e-02 | 1.524e-05 | 20 | 7.144e-03 | 1.493e-05 |
| 9 | 1.208e-02 | 1.495e-05 | 21 | 8.776e-03 | 1.016e-05 |
| 10 | 1.144e-02 | 1.384e-05 | 22 | 5.184e-03 | 1.459e-05 |
| 11 | 1.399e-02 | 1.334e-05 | 23 | 2.199e-03 | 1.059e-05 |

fsdp/plain summaries: bf16 `grad root` 9.579e-03, `grad` pooled 1.190e-02 (worst layer-0 flat_param 1.694e-02), `param` pooled
2.890e-07, `update` pooled 1.191e-02 (worst 1.695e-02, min sign agreement printed); fp32 `grad` pooled 1.214e-05 (worst
layer 16 1.792e-05), `param` pooled 1.563e-09, `update` pooled 6.422e-05 (worst layer 18 1.290e-04, sign agreement 1.0000).

What fails, by strategy: the two `fsdp` cases (flat params) trip the fp32 arm's optimizer-update rule at 1e-4 by 0.2-29 %
("11 check(s) failed": `update/...layers.{6,7,10,12,13,14,18,19}._flat_param: relative Frobenius difference 1.002e-04 ..
1.290e-04 > 0.0001`, plus "1 elements beyond 0.0001 of the tensor's scale" on layers 13, 14, 17); the two `fsdp2` cases
(per-parameter tensors, "230 check(s) failed") trip the bf16 arm's update rule at 0.1 on small bias tensors
(`update/model.layers.1.self_attn.k_proj.bias: relative Frobenius difference 1.055e-01 > 0.1`, `layers.2.self_attn.k_proj.bias`
1.390e-01, "n elements beyond 0.1 of the tensor's scale" on q/k_proj biases; fsdp2/plain bf16 `update` pooled 1.191e-02, worst
1.390e-01, min sign agreement 0.8901) and the fp32 1e-4 rule on others. The LoRA cases pass (bf16 `grad` / `update` pooled
1.308e-02, worst `layers.1.self_attn.q_proj.lora_B` 3.171e-02, sign agreement 0.9854). Routed to the P1 author.

## Review round 2: GPU identity test on p1@cf539906 ("judge the bf16 gradients by their distance to the fp32 truth"), FINAL

Run `gpu-identity-p1-5` (app <modal-app>, 1x H100 80GB, detached worktree `wt/bench-p1d`, `digest_match=True`),
`pytest -s -rs tests/workers/test_response_only_lm_head_on_gpu.py`: **6 passed / 0 failed / 0 skipped in 196.6 s** (rc=0).
Log: `bench/results/gpu-identity-p1-5/pytest.log`. Per case (bf16 arm: `D_off` / `D_on` = distance of the flag-off / flag-on
bf16 gradient to the fp32 truth, `ratio` = D_on / D_off; fp32 arm: flag-on vs flag-off relative Frobenius difference):

| case | outcome | max `grad layer NN` ratio D_on/D_off (groups) | fp32 agreement max (on vs off rel_fro) | `SGD residual` bf16 / fp32 |
|---|---|---|---|---|
| fsdp/plain | PASSED | 1.01 at layer 12 (25 groups: 24 layers + root) | 1.792e-05 (layer 16; pooled 1.214e-05) | off=0.213 on=0.213 ulp / off=0.213 on=0.213 ulp |
| fsdp2/plain | PASSED | 1.22 at layer 15 (26 groups) | 5.663e-05 (layer 20, `input_layernorm.weight`; pooled 1.214e-05) | off=0.292 on=0.245 ulp / off=0.261 on=0.261 ulp |
| fsdp/gradient-checkpointing | PASSED | 1.01 at layer 12 | 1.792e-05 (layer 16) | 0.213 / 0.213 ulp, both arms |
| fsdp2/gradient-checkpointing | PASSED | 1.22 at layer 15 | 5.663e-05 (layer 20) | off=0.292 on=0.245 / off=0.261 on=0.261 ulp |
| fsdp/lora | PASSED | 1.10 at layer 15 (24 groups) | 3.834e-05 (layer 20; pooled 1.425e-05) | off=0.000 on=0.000 ulp, both arms |
| fsdp2/lora | PASSED | 1.10 at layer 15 | 3.834e-05 (layer 20, `k_proj` lora; pooled 1.425e-05) | 0.000 / 0.000 ulp, both arms |

Unchanged from the earlier trees: flag-off twice bitwise identical (108 / 903 / 1041 tensors); forward and training-forward
`log_probs`, `entropy`, `loss/ppo` max_abs 0 in both arms; `sum_pi_squared` max_abs 1.08e-07 (bf16) / 9.31e-08 (fp32);
`[response_only_lm_head][fsdp/plain] fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01`. Verbatim, fsdp/plain:

    [response_only_lm_head][fsdp/plain/bf16] SGD residual: off=0.213 on=0.213 ulp
    [response_only_lm_head][fsdp/plain/fp32] SGD residual: off=0.213 on=0.213 ulp
    [response_only_lm_head][fsdp/plain/bf16] grad layer 00: n=1 max on-vs-off rel_fro=1.694e-02 D_off=1.024e-01 D_on=9.974e-02 max ratio=0.97
    [response_only_lm_head][fsdp/plain/bf16] grad layer 01: n=1 max on-vs-off rel_fro=1.411e-02 D_off=1.224e-01 D_on=1.232e-01 max ratio=1.01
    [response_only_lm_head][fsdp/plain/bf16] grad layer 07: n=1 max on-vs-off rel_fro=1.464e-02 D_off=1.312e-01 D_on=1.288e-01 max ratio=0.98
    [response_only_lm_head][fsdp/plain/bf16] grad layer 12: n=1 max on-vs-off rel_fro=1.171e-02 D_off=7.757e-02 D_on=7.804e-02 max ratio=1.01
    [response_only_lm_head][fsdp/plain/bf16] grad layer 21: n=1 max on-vs-off rel_fro=8.776e-03 D_off=3.364e-01 D_on=3.371e-01 max ratio=1.00
    [response_only_lm_head][fsdp/plain/bf16] grad layer 23: n=1 max on-vs-off rel_fro=2.199e-03 D_off=6.202e-02 D_on=6.198e-02 max ratio=1.00
    [response_only_lm_head][fsdp/plain/bf16] grad root: n=1 max on-vs-off rel_fro=9.579e-03 D_off=7.780e-02 D_on=7.801e-02 max ratio=1.00
    [response_only_lm_head][fsdp/plain/fp32] grad: pooled rel_fro=1.214e-05 max_abs=1.317e-05 worst grad/_fsdp_wrapped_module.model.layers.16._fsdp_wrapped_module._flat_param rel_fro=1.792e-05

Reading: the flag-on bf16 gradient is as far from the fp32 truth as the flag-off one (ratios 0.97-1.01 on fsdp, up to 1.22 on
a single fsdp2 layer-15 tensor, 1.10 on LoRA), while in fp32 the two flags agree to 1.8e-05 .. 5.7e-05; the on/off bf16
difference of 1-2 % measured earlier is bf16 rounding of the kept-row GEMM, not a change of the gradient. Stock suite on this
tree: `gpu-tests-p1-r2-final`.

## Review round 2: stock GPU unit-test selection on p1@cf539906 (regression coverage; the identity test is now collected)

`gpu-tests-p1-r2-final` (app <modal-app>, 8x H100 after the H200:8 copy never got a node; `digest_match=True`),
exactly the `gpu_unit_tests.yml` selection: console **509 passed, 8 skipped, 0 failed in 27 min 19 s** (rc=0; 503 stock + the 6
identity cases); junit (`summary.json`, counts subtests separately) tests 535 / passed 527 / failures 0 / errors 0 / skipped 8;
torchrun `test_special_linear_cross_entropy_tp` rc=0 (32.6 s) and `test_special_megatron_kl_loss_tp` rc=0 (28.1 s). Earlier tree
407082ec (`gpu-tests-p1-r2`): 503 passed / 6 failed (the identity cases of that test version) / 8 skipped, same torchrun 2/2.

## Review round 3: GPU identity test on p1@e7c0d616 ("finite checks, direct bf16 gradient bound and a CPU oracle"), ten cases

Run `gpu-identity-p1-6` (app <modal-app>, 2x H100 80GB, detached worktree `wt/bench-p1e`, `digest_match=True`):
**8 passed / 2 failed / 0 skipped in 323.6 s** (rc=1). Log: `bench/results/gpu-identity-p1-6/pytest.log` (not patched).

| case | outcome | bf16 max on-vs-off rel_fro (direct gradient difference) | max distance ratio D_on/D_off | sgd residual (ulps) off / on |
|---|---|---|---|---|
| plain-fsdp (1 rank) | PASSED | 1.694e-02 (layer 0) | 1.01 | 0.213 / 0.213 |
| plain-fsdp2 (1 rank) | PASSED | 3.586e-02 (layer 0, 12 tensors) | 1.22 | 0.292 / 0.245 |
| gradient-checkpointing-fsdp | PASSED | 1.694e-02 | 1.01 | 0.213 / 0.213 |
| gradient-checkpointing-fsdp2 | PASSED | 3.586e-02 | 1.22 | 0.292 / 0.245 |
| lora-fsdp / lora-fsdp2 | PASSED / PASSED | (lora_B tensors) | 1.10 / 1.10 | 0.000 / 0.000 |
| lora-ckpt-fsdp / lora-ckpt-fsdp2 (new) | PASSED / PASSED | | 1.10 / 1.10 | 0.000 / 0.000 |
| plain-fsdp-2rank (new, two devices) | **FAILED** - predicted by the round-3 re-review, fixed in the next commit | crashed before any measurement | - | - |
| plain-fsdp2-2rank (new) | **FAILED** - same | crashed before any measurement | - | - |

Unchanged one-rank facts: flag-off twice bitwise identical, forward/train `log_probs`, `entropy`, `loss/ppo` max_abs 0,
`fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01`, fp32 arm on-vs-off 1.8e-05 .. 5.7e-05; the log prints no
separate "bound" line, so the direct bf16 gradient bound is reported here through the quantity it bounds (the per-layer
`max on-vs-off rel_fro`, worst 1.694e-02 for fsdp flat params and 3.586e-02 for fsdp2's layer-0 tensors) and the per-layer
`max ratio` (worst 1.22, fsdp2 layer 15). No two-rank measurement exists: both two-rank cases crash in the bf16 flag-off
reference arm, before the first comparison. First failing traceback, verbatim (plain-fsdp-2rank):

    E       torch.multiprocessing.spawn.ProcessRaisedException:
    E       -- Process 0 terminated with the following error:
    E         File "/workspace/verl/tests/workers/test_response_only_lm_head_on_gpu.py", line 498, in _worker
    E           reference = run(precision="bf16", freeze=True, label=f"{label}/bf16", flag=False)
    E         File "/workspace/verl/tests/workers/test_response_only_lm_head_on_gpu.py", line 305, in _run_arm
    E           train = engine.forward_backward_batch(batch, loss_function=partial(ppo_loss, config=actor_config))
    E         File "/workspace/verl/verl/workers/utils/losses.py", line 117, in ppo_loss
    E         File "/workspace/verl/verl/trainer/ppo/core_algos.py", line 1336, in compute_policy_loss_vanilla
    E           negative_approx_kl = log_prob - old_log_prob
    E       RuntimeError: The size of tensor a (256) must match the size of tensor b (40) at non-singleton dimension 1

The stock suite on e7c0d616 (`gpu-tests-p1-r3`) was withdrawn unrun on the lead's hold; identity-7 and the suite run on
the next p1 commit.

## Review round 3: GPU identity test on p1@1dbc0c07 ("rank-local padding, masked row checks and a tensor-relative bound"), FINAL for round 3

Run `gpu-identity-p1-7` (app <modal-app>, 2x H100 80GB, detached worktree `wt/bench-p1f`, `digest_match=True`):
**0 passed / 10 failed / 0 skipped in 255.0 s** (rc=1). Log: `bench/results/gpu-identity-p1-7/pytest.log` (not patched).

All ten cases (plain / gradient-checkpointing / lora / lora-ckpt x fsdp / fsdp2, 1 rank; plain x fsdp / fsdp2, 2 ranks) fail at
the same place, inside the "flag-off twice" bitwise comparison, before any flag-on measurement is printed (the only lines per
case are `flag=off: gradients on 25 tensors`, twice). First failing traceback, verbatim (plain-fsdp-1rank; identical in the
other nine):

    E       torch.multiprocessing.spawn.ProcessRaisedException:
    E       -- Process 0 terminated with the following error:
    E       Traceback (most recent call last):
    E         File "/workspace/verl/.venv/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 87, in _wrap
    E           fn(i, *args)
    E         File "/workspace/verl/tests/workers/test_response_only_lm_head_on_gpu.py", line 537, in _worker
    E           max_abs = (baseline[key] - reference[key]).abs().max().item()
    E                      ~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
    E       RuntimeError: Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.

So no direct-bound ratio, distance ratio or two-rank line exists for this tree: one of the compared entries is a bool tensor
(a mask added by the "masked row checks" change) and the new line 537 subtracts it. The stock suite `gpu-tests-p1-r3` on the
same tree (queued on H200:8) collects this file and will show these ten failures next to the 503 stock cases.

## Review round 3: GPU identity test on p1@8c340e47 ("compare the baseline without subtracting boolean tensors"), FINAL for round 3

Run `gpu-identity-p1-8` (app <modal-app>, 2x H100 80GB, detached worktree `wt/bench-p1g`, `digest_match=True`):
**10 passed / 0 failed / 0 skipped in 320.5 s** (rc=0). Log: `bench/results/gpu-identity-p1-8/pytest.log`.

| case | outcome | max distance ratio D_on/D_off (bf16 grad vs fp32 truth) | bf16 max on-vs-off rel_fro (the directly bounded quantity; the log prints no separate "bound ratio" field) | sgd residual ulps off / on |
|---|---|---|---|---|
| plain-fsdp-1rank | PASSED | 1.02 | 1.680e-02 (layer 0) | 0.213 / 0.213 |
| plain-fsdp2-1rank | PASSED | 1.22 | 3.586e-02 (layer 0 tensors) | 0.292 / 0.245 |
| gradient-checkpointing-fsdp-1rank / -fsdp2-1rank | PASSED / PASSED | 1.02 / 1.22 | as plain | as plain |
| lora-fsdp-1rank / lora-fsdp2-1rank | PASSED / PASSED | 1.11 / 1.11 | lora_B tensors | 0.000 / 0.000 |
| lora-ckpt-fsdp-1rank / lora-ckpt-fsdp2-1rank | PASSED / PASSED | 1.11 / 1.11 | | 0.000 / 0.000 |
| plain-fsdp-2rank (two devices) | PASSED | 1.01 | 1.441e-02 (layer 7 flat param; pooled 1.026e-02) | 0.217 / 0.217 |
| plain-fsdp2-2rank (two devices) | PASSED | 1.13 | | |

Two-rank lines, verbatim (plain-fsdp-2rank; 50 rows per rank after rank-local padding):

    [response_only_lm_head][fsdp/plain/2rank/bf16] flag-off twice: bitwise identical (110 tensors)
    [response_only_lm_head][fsdp/plain/2rank] fp32 vs bf16 flag-off log-probs: mean=4.694e-02 max=2.309e-01
    [response_only_lm_head][fsdp/plain/2rank/bf16] forward/log_probs: max_abs=0.000e+00 mean_delta=0.000e+00 rows=50
    [response_only_lm_head][fsdp/plain/2rank/bf16] forward/entropy: max_abs=0.000e+00 mean_delta=0.000e+00 rows=50
    [response_only_lm_head][fsdp/plain/2rank/bf16] loss/ppo: max_abs=0.000e+00 mean_delta=0.000e+00 rows=1
    [response_only_lm_head][fsdp/plain/2rank/bf16] sgd/grad_norm: off=21.033 on=21.028
    [response_only_lm_head][fsdp/plain/2rank/bf16] sgd/residual_ulps: off=0.217 on=0.217
    [response_only_lm_head][fsdp/plain/2rank/bf16] grad layer 00: n=1 max on-vs-off rel_fro=1.117e-02 D_off=1.450e-01 D_on=1.448e-01 max ratio=1.00
    [response_only_lm_head][fsdp/plain/2rank/bf16] grad root: n=1 max on-vs-off rel_fro=7.365e-03 D_off=6.465e-02 D_on=6.467e-02 max ratio=1.00
    [response_only_lm_head][fsdp/plain/2rank/bf16] grad: pooled rel_fro=1.026e-02 max_abs=9.766e-03 worst grad/_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module._flat_param rel_fro=1.441e-02
    [response_only_lm_head][fsdp/plain/2rank/bf16] update: pooled rel_fro=1.028e-02 max_abs=9.766e-06 worst update/_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module._flat_param rel_fro=1.446e-02
    [response_only_lm_head][fsdp/plain/2rank/fp32] forward/sum_pi_squared: max_abs=3.958e-09 mean_delta=6.985e-12 rows=50

Stock suite on this tree: `gpu-tests-p1-r3-final` (H200:8); the 1dbc0c07 suite (`gpu-tests-p1-r3`) is superseded.

Stock suite on 8c340e47, `gpu-tests-p1-r3-final` (app <modal-app>, 8x H200): console **513 passed, 8 skipped, 0 failed
in 30 min 08 s** (= 503 stock + the 10 identity cases, now collected by the stock glob); junit 539 tests / 531 passed / 0 failures /
8 skipped; torchrun `linear_cross_entropy_tp` rc=0 (31.6 s), `megatron_kl_loss_tp` rc=0 (26.6 s). The 1dbc0c07 suite
(`gpu-tests-p1-r3`, superseded) read 502 passed / 11 failed / 8 skipped: the 10 identity cases of that tree plus one environmental
port collision (`test_sharded_delta::test_gather_slot_entries_sub_rounds_world1`, `EADDRINUSE` 29512).
