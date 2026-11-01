# P1 round-3 re-review verdict: contrib/p1-lmhead-rows @ e7c0d616 (17 commits on main @ 2ab35292)

Reviewer: Claude Fable 5.1, 2026-10-29, read-only on wt/p1. Tests added (never into the branch): `packages/p1/review-tests/
round3/test_r3_two_rank_split_on_cpu.py` (3 pass, 1 fails = D1), `.../test_r3_finite_gate_and_bounds_on_cpu.py` (4 pass, 4 fail =
D2-D5). The submitter's oracle from a scratch copy against wt/p1: **11 passed, 1 failed** (`test_distance_rule_is_not_identity`: the direct
bound rejects the 27 % case, "2.727e-01 relative Frobenius apart (bound 0.1 + floor)"), as the author read it. Shipped counts at
e7c0d616 reproduced: oracle 19 + CPU file 26 + round-2 reviewer tests 13 = 58 passed; the GPU file skips 10/10 on this Mac.

## the submitter's R2-03 findings
| finding | status | evidence |
|---|---|---|
| NaN passed every `delta > tol`; a skipped step passed | FIXED | `_finite` on every key of on / off / truth first; grad norm finite and > 0; non-zero update where the gradient is. My attacks (0-d NaN grad norm, inf residual, -inf rows, one NaN element) are all reported. |
| gradient / update conflated; no direct bf16 bound; guard called identity | FIXED | docstring :60-82 and PR-BODY.md:55-70 separate them (0.0385 / 0.0317 gradient, 0.139 update); direct bound `0.1 |g_off| + 1e-6 G_layer` per tensor (:432) beside the guard, now "a relative-error guard". |
| bitwise forward as a portable requirement | FIXED | per row rtol 2e-2 / atol 5e-2 + mean rule, maxima printed, "observation" in docstring and PR body. |
| scope: one rank NO_SHARD, zero-B LoRA, SGD, no LoRA+ckpt, no multi-rank | **PARTIAL** | SGD / scope stated; `lora_B` seeded identically in every arm (sorted names, fresh generator per engine, `to_local()` / flat-param write); `lora-ckpt` enables both (`_build_module`:366 + `_build_lora_module`); the two-rank cases crash on rank 0 (D1), so multi-rank is still unproven. |

## New defects
- **D1 (HIGH, crash in a supported config)**: rank 0 of the two-rank cases keeps sequences 0,2,4 (responses 40/9/1) while
  `max_response_len` stays 256: `ppo_loss` pads `old_log_probs` / `advantages` / `response_mask` to the local max 40
  (`to_padded_tensor()`, losses.py:103) and `log_prob` to 256 (:59) -> `RuntimeError: The size of tensor a (256) must match the
  size of tensor b (40)` (core_algos.py:1336) in the first arm's `forward_backward_batch`. Reproduced on CPU with the real
  `ppo_loss` on `_make_batch(vocab, 0, 2)`; rank 1 and every one-rank case pass only because their local max is 256. Any host
  with >= 2 devices and the model (the planned H100:2 run; verl's L20x8 CI under `pytest -x`) fails both `plain-*-2rank` cases
  and aborts the job. The normalisation is right (dp_size 2 / 498 tokens / global batch 6: the half losses average exactly to
  the whole-batch loss, 0.252489). Fix: per-rank local `max_response_len` (or none), so both paths pad alike.
- **D2 (LOW)**: a key absent from `truth` is silently exempt from the guard and the finiteness check (`key in truth`, :437).
- **D3 (LOW)**: the floor is 1e-6 of the layer's largest gradient norm: a tensor below ~5e-6 of it can be negated and pass (|big| 1e3 / |tiny| 5e-4); unreachable for this model's norm / bias ratios, worth stating.
- **D4 / D5 (LOW)**: an empty FSDP2 shard (dim 0 < world size; not this model) crashes `_compare` at `delta.max()` (:403);
  rows zero in both arms count as padding (:363), caught in `_worker` only through a NaN cross-precision mean.
- Hygiene: audit `p1 contrib/p1-lmhead-rows e7c0d616 17 0 fresh ok none`; worktree clean; PR body "observation" / "run pending on this
  sha"; CHECKLIST run 4 "[ ] pending"; tests.log §13 matches (only `tests/workers` re-run: 323 = 304 + 19; the commit touches nothing
  else); the GPU module imports safely on a CUDA host (no `device_count()` at import; `pytest.skip` below 2 devices, so `-x` sees a skip).
## GPU test: predicted first-run failures
1. `[plain-fsdp-2rank]`, `[plain-fsdp2-2rank]`: rank 0 raises in `ppo_loss` (D1) -> `ProcessRaisedException`; certain on
   >= 2 devices, skipped on 1.
2. Eight one-rank cases: no new failure mode found (the flow that passed at cf539906; `lora-ckpt` is verl's stock PEFT +
   non-reentrant pairing); margins on another GPU: fp32 1e-4 vs 5.7e-5 measured (1.8x), bf16 direct 0.1 vs 0.0385 (2.6x).

## Call: NEEDS-FIXES
R2-03 is fixed and the bounds are justified, but the two-rank cases crash on rank 0 before any comparison and would abort upstream's multi-GPU CI; one padding fix plus the pending `gpu-identity-p1-6` run remain.

## Lead closure (2026-10-29, after 1dbc0c07)
D1 HIGH and D2-D5 LOW applied in 1dbc0c07: rank-local `max_response_len` in `_make_batch` (consumer views padded to
the width over all ranks, response mask carried and compared bitwise), missing reference keys fail, direct bound
`0.1 |g_off| + 1e-12` with tiny tensors listed, empty shards skipped with a note, row checks on the response mask.
Reviewer round-3 tests 12/12, round-2 13/13, the submitter's oracle 11/12 (the 27 % case rejected by the direct bound, as
intended). Verdict: MERGE-READY pending gpu-identity-p1-7 (H100:2, ten cases) and the stock suite on 1dbc0c07.

Addendum (author, after the fixes): D1-D5 fixed in 1dbc0c07 (reviewer round3 tests 12/12), the boolean-mask baseline crash
of gpu-identity-p1-7 fixed in 8c340e47. `gpu-identity-p1-8` @ 8c340e47 (H100:2, 320.5 s): 10 passed / 0 failed / 0 skipped: forward
rows exact in every case, fp32 arms within 1e-4 per tensor, bf16 on-vs-off gradients at most 1.68e-2 (fsdp) / 3.59e-2 (fsdp2) /
1.44e-2 (two-rank) under the 0.1 direct bound, D_on/D_off at most 1.02 / 1.22 / 1.11 / 1.01 / 1.13 (fsdp / fsdp2 / lora + lora-ckpt /
two-rank fsdp / fsdp2), two-rank plain-fsdp: flag-off twice bitwise (110 tensors), `forward/log_probs max_abs=0 rows=50`,
SGD residual 0.217 ulp, fp32 vs bf16 flag-off log-probs mean 4.694e-2 / max 2.309e-1. Stock suite `gpu-tests-p1-r3-final` @ 8c340e47
(H200:8): 513 passed / 0 failed / 8 skipped (503 stock + 10 identity; junit 539 / 531 / 0 / 8), torchrun 2/2.

## Lead closure (2026-10-29, after gpu-identity-p1-8 on 8c340e47)
8c340e47 fixed the bool-tensor subtraction in the bitwise-baseline loop that failed all ten cases of 1dbc0c07
(identity-7). gpu-identity-p1-8 (H100:2): 10 passed / 0 failed / 0 skipped: forward identical in every case, fp32
agreement <= 5.7e-5, bf16 on-vs-off gradient rel_fro at most 1.68e-2 (fsdp) / 3.59e-2 (fsdp2) / 1.44e-2 (two-rank)
under the 0.1 direct bound, D_on/D_off at most 1.22, SGD residual 0.217 ulp; stock suite 513 / 0 / 8 (+ the ten cases).
Verdict: MERGE-READY.
