# K10 ref-bf16-residency — vetting report (2026-10-29, verl main@2ab35292)

## 1. Claim

Source: research/04-engine-gpu-audit.md §5.3 / finding F5 / candidate C3. Forward-only FSDP
engines (the reference policy; also reward models) are forced onto FSDP CPU offload and load
their weights in fp32, so every micro-batch of the ref pass streams the fp32 local shard
host→device before the bf16 all-gather: 4.1 GB per rank per micro-batch for Qwen3-8B, 262 GB per
rank per step at the 8×H100 / 8k-response reference shape, 33 GB of pinned host RAM per node for
a model that only ever computes in bf16. Proposal: (a) default `model_dtype` to bf16 for
forward-only engines (halves the stream and the pinned RAM, bitwise-identical compute for bf16
checkpoints), and (b) keep the ref resident for the duration of one `compute_ref_log_prob` call
(load once per pass, offload after, like the actor's `param_offload`), removing the per-micro-batch
stream. The ref pass exists for 57 of 121 shipped recipes (`use_kl_loss=True`; 4 more via
`use_kl_in_reward`). Partial overlap with open #7182 (configurable CPUOffload, no dtype change).

## 2. Mechanism verification (file:line in <repo>/verl)

Dtype defaults (verified)
- `verl/trainer/config/engine/fsdp.yaml:24` `model_dtype: fp32`; dataclass
  `verl/workers/config/engine.py:282` `model_dtype: str = "fp32"`. The ref inherits the engine
  yaml unchanged (`verl/trainer/config/ref/dp_ref.yaml` defaults `../engine@fsdp_config: fsdp`,
  sets only `forward_only: True`); generated `verl/trainer/config/_generated_ppo_trainer.yaml:245,257`
  → ref `model_dtype: fp32`, `forward_only: true`.
- `verl/workers/engine/fsdp/transformer_impl.py:262-266`: `None → bf16 if forward_only` exists but
  is dead for yaml users because the yaml value is `"fp32"`, not null. This is NOT a regression of
  the engine refactor: the legacy worker had the same branch (`fsdp_workers.py:391-395` at
  `044bbba^`) and the legacy generated yaml also pinned the ref to `model_dtype: fp32` (line 211 at
  `044bbba^`). The fp32-shard ref has been the shipped default since the legacy worker.
- `:337` `module.to(torch_dtype)` casts registered buffers too, so loading in bf16 rounds
  `RotaryEmbedding.inv_freq` irreversibly (issue #8154; open PR #8189 replaces it with a
  params-only cast). Any bf16 default therefore depends on #8189 (or an equivalent helper).

Offload defaults (verified)
- `strategy: fsdp` (FSDP1) is the yaml default (`fsdp.yaml:67`). `transformer_impl.py:433-438`:
  `forward_only` → `CPUOffload(offload_params=True)` and `_is_offload_param = False`; FSDP2
  `:463-467` → `CPUOffloadPolicy(pin_memory=True)`. `is_param_offload_enabled` returns
  `_is_offload_param` (`:193-195`), so `BaseEngineCtx._context_switch`
  (`verl/workers/engine/base.py:322-336`) returns before any manual load, and `FSDPEngine.to`
  returns early for `forward_only` (`:855-857`). Consequence: the user's
  `ref.fsdp_config.param_offload=True` (BASELINE-A sets it, `bench/modal_verl.py:318`) is silently
  ignored; the only residency mode available for a forward-only FSDP engine is "stream every
  micro-batch".
- torch 2.15 FSDP1 (`.venv-a9/.../torch/distributed/fsdp/_flat_param.py`): with `offload_params`
  the local shard is pinned at init (`:1258-1265`), `pre_unshard` copies it H2D per unshard
  (`:1335-1337`, `non_blocking=True`), `_use_low_precision_shard` (`:1342-1353`) then allocates a
  bf16 copy on device before the all-gather. `forward_prefetch: False` (`fsdp.yaml:30`) means no
  forward run-ahead, so each layer's H2D + cast + all-gather is exposed before its compute.

Bytes and seconds (derived; parameter counts from the Qwen3 architecture tables)
| shape | model | micro-batches / ref pass | H2D per rank per step, fp32 | bf16 | one-shot residency (bf16 shard) | pinned host RAM / node fp32 → bf16 |
|---|---|---|---|---|---|---|
| BASELINE-A (1.35 M tok/step measured, ref budget 24576) | Qwen3-1.7B (1.721 B, tied) | 7 | 6.0 GB (0.30 s @ 20 GB/s) | 3.0 GB (0.15 s) | 0.43 GB (0.02 s) | 6.9 → 3.4 GB |
| BASELINE-A shape | Qwen3-8B (8.191 B) | 7 | 28.7 GB (1.43 s) | 14.3 GB (0.72 s) | 2.05 GB (0.10 s) | 32.8 → 16.4 GB |
| reference user 8k (1.05 M tok/GPU, budget 16384) | Qwen3-8B | 65 | 266 GB (10.7 s @ 25 GB/s) | 133 GB (5.3 s) | 2.05 GB (0.08 s) | 32.8 → 16.4 GB |
Against the measured BASELINE-A ref pass of 3.43 s (bench/README.md, mean of steps 2-6; spread
2.65-4.26 s), the dtype half is worth ≤ 0.15 s/step (0.5 % of the 31.15 s step) and the residency
half ≤ 0.3 s (1 %) — both inside the ±5 % run-to-run spread. At the 8B/8k reference shape the
ref pass is ≈ 49 s of compute + ≤ 10.7 s of exposed streaming: dtype −5 s, residency −10 s, i.e.
2 % / 4 % of a ≈ 250-300 s step (the "+20-27 % on the ref pass" of 04 §5.3 is this worst case
with zero overlap; FSDP1 does overlap the copy as far as the CPU runs ahead, which §3.4 says is
little). For a 1.7B model the stream is latency-bound (29 FSDP units × 7 micro-batches = 203
copies of ~30 MB), so even the bytes halving under-delivers.

Numerics of a bf16 `model_dtype` (reasoned, not measured)
- Today: fp32 shard → `MixedPrecision(param_dtype=bf16)` casts to bf16 at every all-gather
  (`:397-405`). bf16 checkpoints (Qwen3, Llama-3, DeepSeek) round-trip bf16→fp32→bf16 exactly, and
  fp16/fp32 checkpoints are rounded once with the same round-to-nearest-even either way, so the
  gathered bf16 weights — and the ref log-probs — are bitwise identical to today **provided buffers
  stay fp32** (#8189) and `_keep_in_fp32_modules` get the same treatment as today (`:337` already
  flattens them to `model_dtype`, then MixedPrecision casts at gather). Without #8189 the rounded
  `inv_freq` shifts the ref log-probs with position → a silent `kl_loss` bias for 57 recipes.
- Other frameworks: TRL loads no ref at `beta=0`; prime-rl's ref is optional; verl's own Megatron
  ref runs in bf16 (`megatron_ref.yaml` params dtype) — research/06 §13 already rates "CPU-offloaded
  / sharded / absent reference model" as "verl: HAS, no gap". The fp32-host-shard ref is a verl
  FSDP-engine idiosyncrasy, not a precision requirement.

## 3. Duplicate / ownership check (research/07 §8 recipe; dupcheck over 774 open PRs)

- `dupcheck.py --kw forward_only --kw CPUOffload --kw "cpu offload" --kw cpu_offload --kw
  model_dtype --kw "reference model" --kw "ref model" ... --path verl/trainer/config/engine/fsdp.yaml
  --path verl/trainer/config/ref/ --merged`: 49 hits. Directly overlapping:
  - **#7182** "[fsdp] fix: make ref model CPUOffload configurable instead of hardcoded"
    (Shadowking912, NONE association; created 2026-07-28, last commit 2026-09-02 = 56 d;
    MERGEABLE, REVIEW_REQUIRED; files `config/engine.py`, `fsdp/transformer_impl.py`,
    `tests/workers/config/test_engine_config_on_cpu.py`). Diff read via `gh pr diff`: makes
    `param_offload: Optional[bool] = None` (→ True for `forward_only`), gates the forced
    `CPUOffload`/`CPUOffloadPolicy` on it (same lines `:433-438`, `:463-467`). `param_offload=False`
    = permanently resident ref; it does NOT add per-pass load/offload and does not touch dtype.
    CODEOWNERS auto-requested eric-haibin-lin and vermouth1992; no maintainer review in 3 months,
    only a non-maintainer (ErenAta16) reviewing; the author asks "how to get someone to test/merge".
  - **#8189** "[fsdp] fix: preserve FP32 buffers when casting the model to model_dtype" (li-lizhe,
    NONE; last commit 2026-10-10 = 18 d; MERGEABLE; CI never approved to run, author pinged
    2026-10-17). Adds `verl.utils.torch_dtypes.cast_module_params_to_dtype` at `:337`. Prerequisite
    for any bf16 forward-only default.
  - Adjacent on the same lines: #8057 (FSDP2 offload `pin_memory` configurable, 20 d, touches
    `:463-467` and `fsdp.yaml`), #8083 (engine `dtype` → mixed-precision defaults, 28 d, touches
    `:396-407`), #7165 (preserve HF fp32-keep modules in FSDP2, 36 d), #7804 (optimizer offload
    around each update, `fsdp.yaml`), #8004 (fp32 lm_head, draft).
- Merged in the window: #8187 "prevent parameter corruption during FSDP2 CPU offload" (2026-10-16),
  #7103 "skip manual model.to(device) under FSDP2 CPU offload" (2026-07-21), #7077 checkpoint-save
  with CPU offload — the offload path is being patched by several outside authors; none changed
  the forward-only dtype or residency.
- Issues: #8154 (buffer rounding, fixed by #8189); #7182's own report ("40-70× slower ref forward
  than actor forward for a 4B model", unverified — the arithmetic above gives 2-3× at most for a
  bandwidth-bound stream, so that number likely reflects tiny micro-batches or unpinned memory).
  Roadmap #8186 lists "cheaper inference-only forwards (ref, teacher, old_log_prob)" under Low
  Precision Training (04 §9): a maintainer-owned direction (FP8/NVFP4), adjacent, not this.
- Ownership: `fsdp/transformer_impl.py` is the second most contested hot-path file (62 open PRs,
  24 active; CODEOWNERS eric-haibin-lin/vermouth1992/ZihengJiang/ArronHZG); `config/engine.py`
  25 open PRs (6 active); `fsdp.yaml` is regenerated into four `_generated_*.yaml` files.

## 4. Minimum valuable PR (if pursued)

(a) dtype default — `[fsdp, cfg] feat: load forward-only engines in bf16 by default`
| file | change | est. lines |
|---|---|---|
| `verl/trainer/config/engine/fsdp.yaml:24` + 4 `_generated_*.yaml` | `model_dtype: null` with the comment "null = fp32 for trainable engines, bf16 when `forward_only`" | 3 + 4 |
| `verl/workers/config/engine.py:282` | `model_dtype: Optional[str] = None`; docstring | 4 |
| `verl/workers/engine/fsdp/transformer_impl.py:262-266` | unchanged (the `None` branch becomes live); `:337` must become a params-only cast = #8189's `cast_module_params_to_dtype` (rebase on it, or 6 lines if it is closed) | 0-6 |
| tests | `tests/workers/config/test_engine_config_on_cpu.py` (+8: None→bf16 iff forward_only); a CPU `_build_module` test asserting bf16 params and fp32 `inv_freq` | ~40 |
| docs | `docs/workers/model_engine.rst`, `docs/perf/perf_tuning.rst` | ~10 |
Behaviour change for every KL user (57/121 recipes) but no numerical change given #8189.

(b) residency — `[fsdp, perf] feat: forward-only engines use the actor's manual offload path`:
drop the forced `CPUOffload`/`CPUOffloadPolicy` at `:433-438`/`:463-467` (keep FSDP2
`offload_policy` when the user sets it), set `_is_offload_param = param_offload`, delete the early
return at `:855-857` so `EngineEvalModeCtx` loads the bf16 shard once per `compute_ref_log_prob`
and offloads after; `param_offload=False` = resident (what #7182 offers). ~15 source lines +
`tests/special_distributed` coverage (GPU) and the CPU config test. This supersedes #7182 and
therefore cannot be filed while #7182 is alive under the 60-day rule (2026-11-01).

## 5. Proof on the Modal harness (bench/README.md)

The dtype half is measurable TODAY on main with no code: `ref.fsdp_config.model_dtype` is an
existing key. Arm A: BASELINE-A (`results/baseline-a-main-1`, ref 3.43 s, step 31.15 s). Arm B:
`--preset baseline-a --overrides "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16"`
(`--run-name k10-refbf16-1`). Expected: `timing_s/ref` 3.43 → ≈ 3.28 s (−0.15 s), `timing_s/step`
−0.5 %, `perf/cpu_memory_used_gb` −3.4 GB per node; `kl_loss` equal within bf16 noise but **not
bitwise** until #8189 (the `inv_freq` rounding) — a useful diagnostic of exactly that bug. Both
timing deltas sit inside the ±5 % spread (ref alone ranges 2.65-4.26 s on main), so ≥ 3 runs per
arm would be needed and would still not separate 0.15 s. The residency half needs the branch:
expected ref 3.43 → ≈ 3.1 s (−1 % of step), equally unresolvable at this shape. A shape where the
effect is provable: Qwen3-8B, 8k responses (`--model Qwen/Qwen3-8B --max-response-length 8192
--ppo-max-token-len-per-gpu 16384`, H100:8, `gpu_memory_utilization` lowered for the 8B rollout):
ref ≈ 55-60 s with streaming → −5 s (dtype) / −10 s (residency) per step of ≈ 250-300 s, i.e. 2 % /
4 %, needing 2-3 runs per arm to clear noise. Host-RAM halving (33 → 16 GB/node pinned for 8B) is
the one unambiguous number and is reported by `perf/cpu_memory_used_gb`.

## 6. Adversarial answers

- A1 (why not merged): both mechanisms already have open PRs in the same lines (#7182 residency,
  56 d; #8189 params-only cast, 18 d), and neither has drawn a maintainer review in months — the
  area is low-priority for the owners (eric-haibin-lin/vermouth1992/ZihengJiang), who are busy
  with 24 active PRs in `transformer_impl.py`. The residual contribution, a default flip, is a
  behaviour change for half the recipes that buys < 1 % at the benchmark shape; a reviewer will
  ask for numbers the harness cannot produce above its noise floor.
- A2 (gain disappears): once #7182 merges, `param_offload=False` removes the stream entirely and
  the dtype flip's time saving becomes ≈ 0 (host RAM remains); any forward prefetch/overlap hides
  the exposed H2D; for ≤ 4B models the stream is latency-bound; FSDP2 users with
  `offload_policy=False` are unaffected either way; models shipped in fp32 see identical compute.
- A3 (correctness risk): the buffer rounding (#8154) — a bf16 default without #8189 shifts ref
  log-probs with position and biases `kl_loss` silently for 57 recipes; `_keep_in_fp32_modules`
  models (Gemma-style norms) are already flattened by `:337` today; FSDP1 `buffer_dtype=fp32`
  upcasts already-rounded buffers. Tests that would catch a regression:
  `tests/workers/config/test_engine_config_on_cpu.py`, GPU `tests/models/test_engine.py` (ref
  path), e2e `tests/special_e2e` GRPO with `use_kl_loss=True` (a `kl_loss` drift at step 1).
- A4 (who else): Shadowking912 (#7182), li-lizhe (#8189), Longwt123 (#8057), zupengwang (#8083),
  chengcuiping (#7165), SelfDLer (#8187 merged), cben484 (#7103 merged); the roadmap's
  low-precision owners for "cheaper inference-only forwards".
- A5 (half scope): a 7-line default flip (`fsdp.yaml` + dataclass) rebased on #8189, or just a
  docs line telling KL users to set `ref.fsdp_config.model_dtype=bf16` and, after #7182,
  `ref.fsdp_config.param_offload=False` when the shard fits.

## 7. Gates, scores, verdict

G1 FAIL for the residency half (#7182 head commit 56 d < 60; same lines, same fix) and
CONDITIONAL for the dtype half (correct only on top of #8189, 18 d, alive) · G2 PASS (no
maintainer claim; roadmap adjacency only) · G3 PASS (small, self-contained) · G4 FAIL at the
benchmark shape (≤ 0.15 s / 0.3 s against a ±1.5 s spread; provable only at 8B/8k with repeats) ·
G5 PASS (default-path change for the 57 KL recipes).

S1 2 · S2 2 · S3 3 · S4 2 · S5 1 — total 10/25.

**Verdict: NO-GO** as a top-3 contribution. Strongest reason: both halves of the mechanism are
claimed by open PRs with recent commits (#7182, #8189), and what is left — flipping the
forward-only dtype default — is worth < 1 % of step time at BASELINE-A and ≈ 2 % at the reference
shape, below what the harness can prove, while carrying a silent-KL-bias risk until #8189 lands.
What is worth doing anyway, at zero PR cost: run `ref.fsdp_config.model_dtype=bf16` on main as a
free data point (and as a reproduction of #8154), and revisit a superseding "forward-only engines
use the manual offload path" PR with 8B/8k measurements only if #7182 is still untouched after
2026-11-01 — as a secondary candidate, not one of the three.
