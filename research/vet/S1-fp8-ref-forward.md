# S1 — FP8 inference-only forward for the frozen reference model (`torch._scaled_mm`, opt-in; K10's follow-up)

Vetter: sweeper. Clone `2ab35292`. Not on the shortlist. Source: report 01 §2.1 (row "Low-precision
inference-only forwards (ref, teacher, old_log_prob) P1 — OPEN, unclaimed") and §4 item 1; report 04 §5.3, §8 C3.

## 1. Claim
The 26Q4 roadmap (#8186, "LP Common", wuxibin89's own column) lists "cheaper inference-only forwards (ref,
teacher, old_log_prob): less memory, less memory traffic, faster GEMM" as P1; nobody has claimed it and the FSDP
engine's forward-only path has no low-precision compute at all. The candidate: for `forward_only` engines (the
reference policy, `dp_ref.yaml:20`), an opt-in `inference_dtype: fp8` that, after the weights are loaded, swaps
the decoder `nn.Linear`s for an `FP8InferenceLinear` (e4m3 weights + per-row fp32 scales, per-token dynamic
activation scaling, `torch._scaled_mm` with bf16 output) and logs a mismatch KL against a bf16 canary forward.
Expected on H100: ref GEMMs 1.5-1.8× faster (vendor peak 2×), ref weight bytes halved (8.2 GB for Qwen3-8B), so
FSDP all-gather bytes and host staging halve too; stacked on K10 (bf16 + per-pass residency) the ref phase drops
≈30-40 % (est.). Reach: 57 of 121 shipped recipes set `use_kl_loss=True` (report 04 §5.3), i.e. run this pass.

## 2. Mechanism verification (file:line)
- Reference engine = `forward_only: True` (`verl/trainer/config/ref/dp_ref.yaml:20`); worker calls it with
  `forward_only=True` (`verl/workers/engine_workers.py:468`) → `forward_backward_batch(..., forward_only=True)`
  (`verl/workers/engine/fsdp/transformer_impl.py:728-781`, `torch.no_grad()` at `:747`) → `forward_step` (`:783`)
  → `self.module(**inputs)` on stock HF `nn.Linear`s.
- Dtype today: `_build_module` keeps the yaml default `model_dtype: fp32` (`engine/fsdp.yaml:33`) even for
  forward-only engines; the `None → bf16 if forward_only` branch (`:264-266`) is dead unless the user sets
  `model_dtype: null` (this is K10's lever).
- Offload today: FSDP1 forces `CPUOffload(offload_params=True)` for forward-only (`:436-438`); FSDP2 forces
  `CPUOffloadPolicy(pin_memory=True)` (`:463-467`) → every micro-batch streams shards H2D (report 04 §5.3:
  262 GB/rank/step at 64 micro-batches for 8B). K10/#7182 address the streaming; this candidate addresses the GEMMs.
- Module-swap precedent in this engine: the QAT swap is gated on `self._qat_enabled and not
  self.engine_config.forward_only` (`transformer_impl.py:606`), i.e. QAT deliberately skips forward-only engines —
  the exact hole this fills; `verl/utils/qat/linear.py:195 class QATLinear(nn.Linear)` (Triton fake-quant,
  "FSDP compatibility" in its docstring) shows swap-before-wrap is accepted here: `verl/utils/qat/core.py:87 apply_qat` replaces `nn.Linear`
  with `QATLinear.from_linear` (`:116`) through `_set_module` (`:131`), called at `transformer_impl.py:605-607`
  ("Apply QAT before FSDP wrapping (training only)"). `QATLinear` is fake-quant (bf16 weights); nothing
  in-tree runs a real low-precision GEMM on the trainer side.
- Primitives: the pinned torch 2.13.0 exposes `torch._scaled_mm` and `torch.float8_e4m3fn` (checked in
  `.venv-cpu`); rowwise-scaled `_scaled_mm` is native on sm_90+. torchao is NOT a dependency
  (`pyproject.toml`/`uv.lock`: no pin) and is not needed.
- Cost split (report 04 §2): 1F ≈ 49 s/GPU for Qwen3-8B at 1.05 M tokens/GPU; attention is ≈ 2·S·H·L = 2.4 GFLOP
  of ≈ 18.8 GFLOP per token at S=8k, so linear layers are ≈ 87 % of ref FLOPs — the part fp8 accelerates.
- FSDP constraint: FSDP1 flat params need one dtype per unit, so fp8 weights require FSDP2 (per-param DTensors);
  FSDP2's `MixedPrecisionPolicy(param_dtype=bf16)` (`:459-461`) would cast fp8 shards to bf16 before all-gather,
  so fp8 Linears must be wrapped with their own `mp_policy(param_dtype=None)` (nested `fully_shard` via
  `apply_fsdp2`, `verl/utils/fsdp_utils.py:567-604`). FSDP2 shards dim 0 (rows of `[out, in]`), so shard-local
  rowwise quantization equals global rowwise quantization — each rank quantizes its own rows after
  `fsdp2_load_full_state_dict` (`:479-535`).

## 3. Duplicate / ownership check (dumps 2026-10-28; GitHub read-only)
- `dupcheck.py --kw float8 --kw "fp8 ref" --kw inference-only --kw "inference only forward" --kw torchao
  --kw "ref model fp8" --merged`: 7 open hits, none a trainer-side low-precision forward: #8432 (ROCm docker),
  #8402 (AutoModel async draft), #8327 (wengeezhang, MXFP8 rollout refit checks), #8331 (FP8_BLOCK export Triton),
  #5925 (EAGLE), #4325 / #1490 (FSDP2 FP8 *training* WIPs, 321/534 d stale, both mention torchao).
- Hand-checked: #8246 (Step-5 FP8 GRPO+LoRA, draft) and #8257 (Muse Spark NVFP4, draft) are FP8/NVFP4
  *checkpoint* recipes that dequantize to bf16 for training; #7182 (ref CPUOffload configurable) = K10's territory;
  #8419 (fp32 logits) orthogonal. Merged since 2026-07-16: #8287/#8306/#8308 (rollout-side quantizer parity),
  #8127 (Megatron MXFP8 training), #8195 (chunked vocab-parallel entropy) — none touches the ref forward.
- Roadmap: #8186 LP Common P1, listed under wuxibin89 (+FoolPlayer/amenomori); 4 claim comments on the thread,
  none for this bullet (VikingDeng withdrew the config item). The roadmap text says low-precision compute "is to
  come from torchao" under VeOmni — verl's FSDP engine is not mentioned → open, but ask-first (needs the submitter).
- Owners: FSDP engine CODEOWNERS eric-haibin-lin/vermouth1992/ZihengJiang/ArronHZG; in practice wuxibin89 merges.

## 4. Minimum valuable PR (FSDP2-only, ref-only, dense Linear only; ≈ 250-300 code + 150 tests)
- `verl/utils/fp8_inference.py` (new, ~120): `FP8InferenceLinear(nn.Module)` with buffers `weight_fp8`
  (e4m3, `[out, in]`), `weight_scale` (fp32, `[out, 1]`), bf16 bias; `forward`: per-token rowwise quantization of
  `x` (amax/448 → e4m3), `torch._scaled_mm(x_fp8, weight_fp8.t(), scale_a, scale_b.t(), out_dtype=bf16,
  use_fast_accum=True)`; `quantize_rowwise(w)`, `from_linear(linear)`, a CPU reference `dequant_matmul` for tests,
  and `swap_linear_for_fp8_inference(module, skip=("lm_head",), min_dim=1024)`.
- `transformer_impl.py` FSDP2 branch (`:455-477`, ~+60): when `engine_config.forward_only and
  engine_config.inference_dtype == "fp8"`: after `fsdp2_load_full_state_dict`, quantize each decoder Linear's local
  shard rowwise, replace the module, `fully_shard` it with `MixedPrecisionPolicy(param_dtype=None)`; asserts:
  `strategy == "fsdp2"`, not QAT, not `ref_in_actor`/LoRA, dense (MoE expert modules skipped), device sm_90+.
- Config: `FSDPEngineConfig.inference_dtype: str | None = None` (`verl/workers/config/engine.py`, +5),
  `engine/fsdp.yaml` (+3), regenerated `_generated_*.yaml`.
- Metric (+30): on the ref worker, every `inference_fp8_canary_every` steps run one micro-batch with the swap
  bypassed (bf16 dequant path) and log `ref/fp8_mismatch_kl` (k3) and `ref/fp8_mismatch_max_abs`.
- Tests: `tests/utils/test_fp8_inference_on_cpu.py` (rowwise quant/dequant error bound, swap selection, CPU
  reference vs `F.linear` within e4m3 tolerance — `torch.float8_e4m3fn` casts work on CPU in torch 2.13);
  `tests/special_e2e` GPU case for the FSDP2 swap (shape/dtype of all-gathered params, log-prob mismatch < 1e-2).
- Docs: `docs/low_precision/inference_only_forward.md` (+40). Default off.

## 5. Proof (Modal, 8×H100, bench/README.md harness)
Qwen3-8B GRPO, `use_kl_loss=True`, FSDP2, 1024 seq × 8k, three arms: A = main, B = K10 (bf16 + resident ref),
C = K10 + `inference_dtype=fp8`. Report `timing_s/ref` (expect B ≈ 50-55 s → C ≈ 32-38 s: GEMMs ≈ 87 % of ref
FLOPs × 1.6-1.8× fp8 speed-up at M=16k, K=4096; arithmetic, not measured), `timing_s/step`,
`perf/max_memory_allocated_gb`, `ref/fp8_mismatch_kl` (expect 1e-3-level, the order #8025's rollout-vs-trainer
metric reports), and a 100-step reward-curve overlay of C vs B. Negative control: Qwen3-30B-A3B (experts skipped
→ expect no gain, documented). Small-model control: Qwen3-1.7B (launch-bound → expect < 10 %).

## 6. Adversarial answers
- A1 (why not merged): the bullet sits in wuxibin89's column and the roadmap says low-precision compute "is to
  come from torchao" (VeOmni); a hand-rolled `_scaled_mm` module next to the compressed-tensors quantizer (#8287)
  may read as a second quant stack; it is 300 lines, Hopper-only, FSDP2-only, from a first-time author in the
  most contested file (24 active PRs). Mitigation: claim the bullet on #8186 first, ship ref-only, name #8287's
  quantizer as the export side it does not touch.
- A2 (gain disappears): if the ref pass is offload/PCIe-bound (today's default) K10 captures most of the win and
  fp8 adds little until residency lands; models ≤ 1.7B are launch-latency-bound; MoE models get nothing
  (experts skipped); on Blackwell the maintainers want MXFP8/NVFP4, not e4m3 per-row.
- A3 (correctness): ref log-prob drift changes the KL penalty/loss → the canary metric and the curve overlay are
  mandatory; the swap must never reach the actor (`forward_only` gate, asserted); `score_centering`/
  `sum_pi_squared` read the actor, not the ref. Existing coverage: `tests/models/test_engine.py`,
  `tests/workers/test_engine_return_model_output_on_cpu.py`, `tests/workers/config/test_engine_config_on_cpu.py`.
- A4 (who else): wuxibin89/FoolPlayer (VeOmni low precision), amenomori (low-precision comm), wengeezhang/
  zhangyimi/sophiayyya (NVIDIA Megatron MXFP8/NVFP4), cr-gao (batch invariance). Nobody on the FSDP ref forward.
- A5 (half scope): per-tensor-scaled fp8 ref on FSDP2, lm_head excluded, no canary (~150 lines); or smaller still, a
  benchmark-only PR/issue comment with the A/B/C table to claim the bullet.

## 7. Gates, scores, verdict
G1 pass · G2 ask-first (maintainer-listed, unclaimed) · G3 pass with caveat (no new dependency; FSDP2-only; ~300
lines, above the 300-line merged-perf-PR pattern) · G4 pass (Modal H100; `timing_s/ref`) · G5: opt-in, but
`use_kl_loss` is a very common opt-in (57/121 recipes).
S1 4 (the ref pass is ≈ 1 of 6 forward-equivalents of a KL run; -30…-40 % of it ≈ -6…-8 % step, plus memory) ·
S2 5 (wuxibin89's own P1 bullet) · S3 2 (arithmetic only; fp8 GEMM gain unmeasured at RL shapes) · S4 2 (size,
Hopper/FSDP2-only, contested file, first-time author) · S5 4 (first trainer-side low-precision forward in verl;
prime-rl/NeMo-RL run FP8 inference but not a trainer-side ref). Total 17 vs shortlist median ≈ 14-15.
Verdict: GO-WITH-CHANGES — only after K10 lands; claim the bullet on #8186 before coding; ref-only, FSDP2-only;
it is the one roadmap-P1 speed item the shortlist does not contain. Strongest reason: maintainer demand is
explicit and unclaimed; strongest risk: the owner of that column may want to do it his way.
