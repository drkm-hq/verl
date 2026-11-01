# P1 round-2 re-review verdict: contrib/p1-lmhead-rows @ d7d820aa (13 commits on main @ 2ab35292)

Reviewer: Claude Fable 5.1, 2026-10-29, read-only on wt/p1. Tests added (never into the branch):
`packages/p1/review-tests/round2/test_r2_peft_probe_on_cpu.py` (7 tests, 7 pass) and
`packages/p1/review-tests/round2/test_r2_gpu_check_mechanics_on_cpu.py` (6 tests: 2 pass, 4 fail by design = findings).
The submitter's bundle has no P1 reproduction script (evidence/reproductions/ holds p2/p3/p4 only): nothing to re-run.

## the submitter's findings
| finding | status | evidence |
|---|---|---|
| Frozen-batch bf16 GPU on/off comparison (log-probs, entropy, Σπ², grads, one AdamW step; entropy-loss, PEFT, checkpointing; justified tolerances) | **PARTIAL** | `tests/workers/test_response_only_lm_head_on_gpu.py` covers every item and desk-checks clean against the engine API (below), but it has not run and two tolerances are not justified (D1, D2): as written it is predicted to go red on the LoRA cases for a non-defect. |
| "no device-to-host sync" claim | FIXED | PR-BODY.md:83-85 matches `padding.py:165`: one `int(prompt_lens.min())` per micro-batch, the index itself sync-free. |
| 657 / 0.099 instead of 636 / 0.096 | FIXED | Recomputed from `bench/results/long-prompt-{main-1,main-2,p1-on-2,p1-on-3}/metrics.jsonl`, steps 2..6: response 657.24, prompt 5998.78, K/T 0.0987 (per run 660.10 / 652.88 / 651.94 / 664.02; OFF run 656.12 / 0.0986; all five 657.01 / 0.0987). PR-BODY.md:118 says 657 / 0.099. |
| phases and memory separate from whole-step deltas; keep opt-in | FIXED | PR-BODY.md:112-131, 154-156 (`gen` and step "not claimed"; opt-in with the reason). |
| PEFT probe (author's 724d3d5e) | FIXED | PEFT 0.19.1 `PeftModelForCausalLM.forward` does not name `logits_to_keep`, so the `get_base_model()` unwrap is load-bearing. Round-2 tests: wrapper -> on; bare `**kwargs` -> off with the named warning, also through a PEFT wrapper; VLM guard seen through the wrapper; rank != 0 silent; resolver run after the wrap and the index reaching the base model with `lora_dropout` 0.0 / 0.3 in eval (log-probs bitwise, LoRA grads <= 1e-6). |

## New defects
- **D1 (MEDIUM, test design; likely red first run)**: `_check` (GPU test :237-241) asserts rel-Frobenius < 1e-2 and element-wise
  `atol = 1e-2 * max|ref|` on `param/` after the AdamW step. `lora_B` starts at zero, so its post-step tensor *is* the update
  (~ -lr*sign(g), max 1e-3 -> atol 1e-5), while one gradient sign flip at the noise floor is 2*lr = 2e-3. The docstring says
  exactly this ("not an identity failure") but exempts only `update/`. Measured: one flip in a 896x8 `lora_B` -> rel_fro 2.36e-2;
  one bf16 ulp on 1 % of the lm-head input gradient `dh` of the tiny LoRA Qwen3 -> 2 of 8192 `lora_B` gradient signs flip ->
  rel_fro 1.25e-1 on `k_proj.lora_B`. On the GPU the arms differ in `dh = dlogits @ W` (M = 498 vs 1589 rows, reduction over
  V = 151936; cuBLAS picks split-K per shape), so ulp-level `dh` differences on the loss rows are expected, and any of them
  flips signs somewhere in 2.4M `lora_B` elements. Same mechanism for FSDP2 per-parameter weights with max|w| < ~0.2 (2e-3 vs
  1.5e-3 allowed); FSDP1 flat params are shielded by the norm weights in the same flat param (that variant passes).
- **D2 (MEDIUM)**: `rtol=atol=1e-2` on entropy / Σπ² is below the docstring's own premise (one bf16 ulp of a logit). A 151936-vocab
  row, top logit 20 (ulp 0.125, p_top 0.88) one ulp higher: entropy 0.3992 -> delta 0.0291 vs 0.0140 allowed, Σπ² 0.7871 -> 0.0196
  vs 0.0179, log-probs 0.1107 vs 0.2535 (passes). Needs the K-row vs T-row head GEMM (same K = 896 reduction) to round a top logit
  differently: less likely than D1, not excluded.
- **D3 (LOW)**: `gpu_unit_tests.yml:139` runs the file under `pytest -s -x`: a D1/D2 failure aborts verl's whole GPU job.
- Suggested fix (one commit): assert `param/` through `|delta| <= 2*lr + 1e-2*|ref|` plus a sign-agreement rate (>= 1 - 1e-3) on
  the updates; derive the entropy / Σπ² tolerance from the bf16 ulp of the row's max logit (or compare against an fp32
  recomputation from the same bf16 hidden states); keep the rel_fro gradient check. Run once before tightening anything.

## GPU test: predicted first-run failures
1. `test_flag_on_matches_flag_off[lora-fsdp]`, `[lora-fsdp2]` at `_check` :239/:241, key `param/...lora_B.default.weight` (D1):
   likely (one sign flip suffices; expected whenever `dh` differs by >= 1 bf16 ulp on any loss row).
2. `[plain-fsdp2]`, `[gradient-checkpointing-fsdp2]` at :241, `param/<weight with max|w| < ~0.2>` (D1): medium (depends on
   Qwen2.5-0.5B's per-tensor maxima; no weights on this Mac to check).
3. Any case at :229 on `forward/entropy` / `forward/sum_pi_squared` (and `train/`) if the forward GEMM rounds a top logit
   differently (D2): low-medium.
4. Trap, not a failure: `/root/models/Qwen/Qwen2.5-0.5B` must be linked by `bench/modal_verl.py::_link_models` (it is in the
   download list); otherwise all 6 cases skip, not fail (`-rs` shows it).
No API or runtime error found: every constructor kwarg exists; `forward_backward_batch` returns `{"model_output": nested per
key, "loss": [float], "metrics"}` (so `train["loss"]` is a shape-(1,) tensor); `ppo_loss` gets `dp_size` / `batch_num_tokens` from
the engine and `global_batch_size` from the batch; `load_tokenizer=False` -> `processing_class=None` accepted; FSDP1 + LoRA wraps
each trainable leaf; `train_mode(zero_grad_on_exit=False)` keeps grads; T 1589 / K 498, sequence 0's rows [0, 40);
`FLASH_ATTENTION_DETERMINISTIC` is read at call time; `lora_dropout` and `attention_dropout` are 0.0 (bitwise baseline holds).

## Hygiene
`scripts/package-audit.sh`: `p1 contrib/p1-lmhead-rows d7d820aa 13 0 fresh ok none`; diff.patch == `git diff main...HEAD` (9 files,
+893/-3); worktree clean; all 13 commits carry the trailer. PR-BODY.md: AI statement verbatim, GPU suite labelled per the rules, no
"already reviewed" / upstream-contact claims, pooled numbers, GPU test marked "not yet run". CHECKLIST.md truthful ("[ ] GPU
identity test run: pending"). tests.log: 26 passed on the new file, 1510 / 64 / 1 regression, matching CHECKLIST, NOTES, REVIEW §10.

## Call: NEEDS-FIXES
Product code and documentation corrections are sound; the GPU identity test is complete in scope but its `param/` (D1) and
entropy / Σπ² (D2) tolerances contradict its own docstring and are predicted to fail the LoRA cases on a non-defect. Fix the two
tolerance rules before the Modal run; the run itself is then the remaining gate.

## Lead closure (2026-10-29, after 407082ec)
D1-D3 applied by the author in 407082ec: the one-optimizer-step check now uses plain SGD (update = -lr*grad, so the
gradient tolerance carries over; >= 99 % sign agreement; param bounded by 2*lr*1.01 + 1e-2*|ref|), per-row
log-prob / entropy / sum-pi-squared tolerances justified by one bf16 ulp of a top logit (rtol 2e-2, atol 5e-2) with
tighter batch means, `-x` noted, loud skip naming the model candidates. The reviewer's 13 round-2 CPU tests pass
against wt/p1 @ 407082ec. Product code, PEFT probe and doc corrections were FIXED in the verdict above. Remaining
evidence item: the GPU run of the identity test (gpu-identity-p1-2) and the tolerance-tightening follow-up on its
measured deltas. Verdict: MERGE-READY-PENDING-GPU-IDENTITY.
Addendum (after the first GPU runs): gpu-identity-p1-2 (d7d820aa) and -p1-3 (407082ec) both 6 failed / 0 passed /
0 skipped: flag-off twice bitwise; flag on vs off forward log-probs and entropy BITWISE identical in all six cases,
sum-pi-squared max 1.08e-7; the gradient check failed at its first key (layer 0) with 1.69e-2 (fsdp flat param) /
2.20e-2 (fsdp2 input_layernorm) / 1.68e-2 (lora down_proj.lora_B) relative Frobenius vs the 1e-2 bound; later keys
unreported. Author's reading: the head's backward GEMMs split the vocabulary reduction differently for K vs T rows
(ulp-level dH differences) re-rounded through 24 bf16 backward layers. 363470c1 adds an fp32 reference arm (must
agree to 1e-4 per gradient tensor), reports every tensor per layer, and sets the bf16 gradient/update tolerance at
1e-1 (~4x the largest measured) with the fp32 agreement as the justification. The reviewer's
`test_entropy_tolerance_covers_one_bf16_ulp_of_a_top_logit` now fails by construction (the forward is asserted exact
as measured) and is retired by this note. Decision pending gpu-identity-p1-4.
Addendum 2 (after gpu-identity-p1-4 on 363470c1: 2 passed / 4 failed): the fp32 arms agree to 1.0-1.8e-5 per gradient
tensor (worst 5.7e-5), which settles the question: the bf16 deltas are rounding amplification. The four failures were
rule artefacts (fp32 "update" check limited by subtraction cancellation at 1.0-1.3e-4 > 1e-4; bf16 bias tensors up to
1.39e-1 relative where ulp noise dominates). cf539906: bf16 gradients judged per tensor by distance to the fp32 truth
(D_on <= 2 D_off + 1e-6 |g_fp32|), fp32 agreement 1e-4 kept as the proof, within-arm SGD residual asserted, bitwise
forward kept. Decision pending gpu-identity-p1-5; the stock p1 suite re-runs on cf539906 (the stock glob collects the
file).

## Lead closure (2026-10-29, after gpu-identity-p1-5 on cf539906)
6 passed / 0 failed / 0 skipped: forward and train log-probs, entropy and loss bitwise identical in both arms; fp32
on-vs-off gradient agreement max 1.8e-5 (fsdp) / 5.7e-5 (fsdp2) / 3.8e-5 (lora); bf16 distance-to-truth ratio
D_on/D_off at most 1.01 / 1.22 / 1.10; SGD residual 0.21-0.29 ulp (0 for lora). Stock GPU suite on cf539906: 509 / 0 / 8
+ torchrun 2/2 (the six identity cases included). Verdict: MERGE-READY.

Addendum 3 (author, round 3, after the AI follow-up review): Addendum 2's "1.39e-1 relative where ulp noise dominates" is
the SUBTRACTED-UPDATE error (`update/model.layers.2.self_attn.k_proj.bias`, gpu-identity-p1-5/pytest.log:272; cancellation
of `param_after - param_before`), not a gradient error. The worst actual bf16 on/off gradient relative Frobenius difference
at cf539906 is 0.03851 (fsdp2, `layers.4.input_layernorm.weight`, log:270) and 0.03171 for LoRA; the run-4 failures named
`update/` keys. The direct bf16 gradient bound (0.1 per tensor, 1e-6 layer floor) is back in e7c0d616 next to the
distance-to-fp32 guard, which is a relative-error guard, not an identity statement.
