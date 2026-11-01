P1 round-two independent review
===============================
Reviewed HEAD cf5399069c7dcfb40acf280a318ea838d7273bb2 against a11ca00b.
Source worktree: <repo>/wt/p1 (unchanged and clean).

Verdict
-------
No new production-code defect found. The only production delta is the PEFT-aware
logits_to_keep signature probe; it is reasonable and its focused tests pass.
Earlier documentation corrections are closed, and the requested feature-enabled
GPU experiment now exists and ran. However, fix the GPU test's non-finite-value
hole and correct the gradient/update interpretation before calling its current
acceptance rule an identity proof. The evidence is substantially better, but its
scope and assertions need to be described accurately.

Confirmed finding 1 — P2: the GPU comparison silently accepts NaNs
----------------------------------------------------------------
File: wt/p1/tests/workers/test_response_only_lm_head_on_gpu.py
Relevant lines: 323-325, 331-352, 363-378 (especially 334, 343, 351, 363, 375).

_compare has no torch.isfinite check. All decisive checks for forward quantities,
losses, gradients and the SGD residual are comparisons of the form NaN > bound,
which are false. The flag-off-twice baseline catches a broken baseline but does
not catch this failure introduced only by the enabled arm.

CPU reproduction using the actual imported helper: injecting NaN in any of
forward/log_probs, train/entropy, loss/ppo, grad/model.layers.0.weight, or
sgd/residual_ulps yields no reported failures, for bf16 AND fp32. A fuller case
with NaN gradient, unchanged enabled-arm parameters, zero update, and NaN SGD
residual also passes. This corresponds to the real engine's non-finite-gradient
behavior: transformer_impl.py:895-899 skips optimizer.step and clears gradients.
The loose after-parameter bound does not catch the skipped small update.

Fix: explicitly require every inspected arm/reference/truth tensor, loss and
residual to be finite before numerical comparisons; assert optimizer_step's
returned grad norm is finite. Keep the injected-NaN cases as regression tests.

Confirmed finding 2 — P2: raw gradient and subtracted-update errors are conflated
-------------------------------------------------------------------------------
Files: tests/workers/test_response_only_lm_head_on_gpu.py:40-50, 363-378;
packages/p1/PR-BODY.md:54-58; REVIEW-ROUND2-VERDICT.md Addendum 2.

The narrative says bf16 gradients reach 0.14 relative error on q/k biases. In
gpu-identity-p1-4 and -5, 0.1390 is the SUBTRACTED UPDATE error. Actual worst
fsdp2 gradient relative Frobenius error is 0.03851 (layer-4 input_layernorm),
and LoRA's is 0.03171. The run-4 failure lists explicitly name update/... keys;
the earlier 0.1 direct bf16 gradient checks were passing. Cancellation explains
the noisy parameter-after minus parameter-before measurement; it does not
justify removing the direct gradient check.

Current code only applies a direct on/off gradient bound in fp32 (line 363).
In bf16 it allows error relative to the fp32 reference to double, and reports
on/off differences without bounding them. A finite scalar characterization
passes truth=1, off=1.1, on=0.8: a 27.27% on/off gradient difference. This is not
evidence that the actual implementation has such an error; it demonstrates the
acceptance region. The rule is a relative-error regression guard, not identity
or "no worse than baseline" (which would be factor 1).

Fix: distinguish gradients from subtracted updates in every report; retain a
justified direct bf16 gradient bound alongside the fp32-reference check, with
appropriate absolute floors for near-zero tensors. Keeping the SGD within-arm
residual instead of a cancellation-sensitive update-relative bound is sensible.
Use wording such as "numerical comparison on this workload" instead of claiming
the observed agreement proves all bf16 differences harmless. The proposed
split-K/rounding-amplification mechanism is plausible, not isolated by this test.

Coverage limitations / improvements (not production bugs)
---------------------------------------------------------
1. Actual FSDP1/FSDP2 engine APIs are exercised, but only world_size=1:
   _build_engine:203 fsdp_size=1 and test:452 nprocs=1. Raw log line 33 explicitly
   says FSDP1 switches FULL_SHARD to NO_SHARD. Existing 8-H100 training runs add
   distributed smoke/performance evidence, not a distributed identity comparison.
2. LoRA and checkpointing are separate variants (test:111-115), not their
   combination. Fresh default LoRA has zero B and zero A gradients at this first
   backward. Confirmed with the current CPU helper (lora-initialization.log).
   Seed nonzero B values or test a trained adapter, and add LoRA+checkpointing.
3. The optimizer is SGD, not the usual AdamW. Within-arm SGD consistency is
   valuable but does not establish AdamW trajectory equivalence. The source and
   main PR description now say SGD; preserve this limitation in summaries.
4. Exact bf16 forward equality (test:121, 338-349) is an observation on one H100
   and shape. It is stricter than the source's own explanation that different
   GEMM shapes may round differently. This is a portability/false-failure risk,
   not a demonstrated failure on another device. Upstream CI may use L20x8
   (gpu_unit_tests.yml:106), so prefer justified tolerances rather than assuming
   the one-H100 bitwise observation is universal.
5. The fp32 reference also changes attention backend to math SDPA and uses the
   bf16 baseline old_log_probs. Its agreement supports the mathematical indexing
   argument but does not by itself isolate the bf16 backward GEMM as the sole
   error source. A frozen-hidden-state/head-backward high-precision comparison
   would more directly test that mechanism. The <=0.5-nat mean forward sanity
   check alone is not a proof of packed-attention-mask equivalence.

Closed prior items and verification
----------------------------------
- Sync documentation: PR-BODY.md:95-98 now acknowledges one host transfer per
  microbatch for the empty-prompt guard.
- Long-prompt quantities: PR-BODY.md:129-131 now says 657 response tokens and
  K/T=0.099; training phases/memory are separated from rollout/whole-step noise.
- Feature stays opt-in; noisy training metrics are no longer the sole evidence.
- Shipped CPU tests + new PEFT reviewer tests: 33 passed (26 + 7), independently
  rerun here. Log: cpu-tests.log.
- New oracle tests: 11 failed as expected from missing NaN rejection, 1 passed
  finite-rule characterization. Files: test_gpu_oracle_on_cpu.py, oracle-tests.log.
- Raw gpu-identity-p1-5: 6 passed, 0 failed, 0 skipped, cf539906, digest_match=true.
  Forward/training log-probs, entropy and loss agree exactly on the tested inputs;
  sum_pi_squared max delta 1.08e-7; fp32 worst gradient delta 5.663e-5 relative.
  Maximum reported bf16 distance ratios: fsdp 1.01, fsdp2 1.22, LoRA 1.10.
- Raw gpu-tests-p1-r2-final: console 509 passed / 8 skipped; JUnit counts subtests
  separately (527 passed / 8 skipped); both torchrun checks return 0. Same digest
  as the identity run. These are inspected existing runs, not GPU reruns by me.
- No GPU spending, source edits, commits, or external messages performed.

Reproduce local checks
----------------------
Run from <workdir>

env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=<repo>/wt/p1 \
  TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
  <repo>/.venv-cpu/bin/python -B -m pytest \
  -q -s -p no:cacheprovider work/round2/p1/test_gpu_oracle_on_cpu.py

The nonzero exit is intentional: tests require the helper to reject NaNs.

Shipped CPU + PEFT reviewer checks (33 passed):

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=<repo>/wt/p1 \
  TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  <repo>/.venv-cpu/bin/python -B -m pytest \
  -c <repo>/pytest-cpu.ini \
  --rootdir <repo>/wt/p1 \
  -p no:cacheprovider --asyncio-mode=auto \
  <repo>/wt/p1/tests/workers/test_response_only_lm_head_on_cpu.py \
  <repo>/packages/p1/review-tests/round2/test_r2_peft_probe_on_cpu.py -q
```

Precise raw evidence for the gradient/update distinction:

- <repo>/bench/results/gpu-identity-p1-5/pytest.log:270
  reports actual fsdp2/plain bf16 grad pooled=1.190e-2 and worst=3.851e-2.
- The same file:272 reports UPDATE worst=1.390e-1 on layer-2 k_proj.bias.
- <repo>/bench/results/gpu-identity-p1-4/pytest.log:1210
  explicitly identifies the >0.1 failure as update/...k_proj.bias, not grad/....
- Run -5 log:33 states that one-rank FSDP1 changes FULL_SHARD to NO_SHARD.

All wt/p1 and packages/p1 relative paths above are rooted at
<repo>. Source HEAD was verified clean after review.
