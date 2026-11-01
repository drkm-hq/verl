# K11 selective-ckpt (vet, 2026-10-29)

Vetter: adversarial pass over the clone at verl-project/verl@2ab35292 (read-only), gh dumps in gh/, scripts and
outputs under research/vet/K11/. GitHub was only read; the clone was not modified.

## 1. Claim

Source: research/04-engine-gpu-audit.md §2.3 item 2, §5.6, finding F4 and candidate C1. Activation checkpointing in
the FSDP engine is all-or-nothing per decoder layer (`verl/workers/engine/fsdp/transformer_impl.py:339-340`;
`HFModelConfig.enable_gradient_checkpointing` at `verl/workers/config/model.py:117`, default True in
`verl/trainer/config/model/hf_model.yaml:34`), so every actor update pays one extra forward over all layers (the "1F
recompute" of the 5F-per-step accounting). Proposal: every-k / N-layer and op-selective (SAC: keep matmul outputs,
recompute norms, SiLU, RoPE, attention) checkpointing for the FSDP engine, with Megatron-style naming, for
-10..-20 % of the actor update when memory allows; the torchtitan engine already exposes
`activation_checkpoint: selective|full|none`.

## 2. Mechanism verification

### 2.1 How checkpointing is applied today, and what HF already offers

- `transformer_impl.py:339-340`: `module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":
  False})` on the HF module inside `_build_module` (:258-341), before FSDP wrapping (`_build_fsdp_module` :389-497).
  No `apply_activation_checkpointing`, no `checkpoint_wrapper`, no `context_fn`, no SAC anywhere:
  `grep -rn "create_selective_checkpoint_contexts\|CheckpointPolicy\|checkpoint_wrapper\|apply_activation_checkpointing\|context_fn" verl/`
  -> 0 hits; `grep -rn "every_n_layers\|gradient_checkpointing_kwargs"` -> only :340.
- HF side, transformers 5.16.1 (verl pin, `verl/uv.lock:5325`; 5.21.0 in the venv):
  `PreTrainedModel.gradient_checkpointing_enable(gradient_checkpointing_kwargs=None, every_n_layers: int = 1)`
  (v5.16.1 `modeling_utils.py:3186`, read via the GitHub contents API; 5.21 adds `offload: bool`). It builds
  `functools.partial(checkpoint, **gradient_checkpointing_kwargs)` and `_set_gradient_checkpointing(...,
  every_n_layers)` sets `module.gradient_checkpointing = layer_index % every_n_layers == 0` on each
  `GradientCheckpointingLayer` (:3240-3261). `GradientCheckpointingLayer.__call__` (`modeling_layers.py:80-113`)
  runs `self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)` when its flag is set, so any
  `torch.utils.checkpoint.checkpoint` kwarg, `context_fn` included, flows through. huggingface/transformers#48200
  (qgallouedec, merged 2026-08-25) added `every_n_layers`; it is inside verl's pin. Consequences: (a) the audit's
  "HF checkpoint kwargs plumbing" risk is moot; (b) HF's knob checkpoints ONE layer in k, i.e. it can only keep
  >= half of the layers un-checkpointed, which is the wrong direction for a memory-bound 8B run (2.4), so verl
  should set the per-layer flags itself ("checkpoint N of L layers", Megatron `block` semantics, a 4-line loop).
- Knobs today: `model.enable_gradient_checkpointing` (bool); `model.enable_activation_offload`
  (`config/model.py:118`, `hf_model.yaml:37`, off): when on, `enable_activation_offloading(module, strategy,
  enable_ckpt)` (`transformer_impl.py:481-483` -> `verl/utils/activation_offload.py:500-551`) DISABLES HF
  checkpointing (`gradient_checkpointing_disable` :550-551) and re-implements per-FSDP-layer checkpointing with
  `use_reentrant=True` (`ActivationHandler` :416-425). `fsdp_config` has no checkpoint field (`engine/fsdp.yaml`).
  `use_torch_compile` compiles only `entropy_from_logits` (:184-185), never the model.
- Naming elsewhere. Megatron (`verl/trainer/config/engine/megatron.yaml:70-87`, under `override_transformer_config`):
  `recompute_granularity: full|selective`, `recompute_modules: [core_attn, moe_act, layernorm, mla_up_proj, mlp,
  moe]`, `recompute_method: uniform|block`, `recompute_num_layers`. torchtitan engine: `activation_checkpoint:
  selective|full|none` (`verl/workers/config/engine.py:514-516,554,571`, `engine/torchtitan.yaml:68-71`) mapped to
  torchtitan `SelectiveAC.Config` / `FullAC.Config` (`verl/workers/engine/torchtitan/transformer_impl.py:36,206-210`);
  upstream torchtitan has since moved to region-based `ActivationRematConfig(save_regions, recompute_regions)`
  (`torchtitan/distributed/activation_remat.py`, fetched read-only). Automodel: `activation_checkpointing: bool`
  (`engine/automodel.yaml:26`, `engine.py:671`). Docs: `docs/examples/config.rst:242-247` ("FSDP only ... Megatron
  uses recompute options in override_transformer_config"), `docs/perf/perf_tuning.rst:113`.

### 2.2 CPU test: numerical equivalence (`research/vet/K11/test_selective_ckpt_cpu.py`, output `test_output.txt`)

Tiny Qwen3 (4 layers, H=64, fp32, eager attention), identical weights and inputs, four modes: none / full (HF
default, `use_reentrant=False`) / every2 (`every_n_layers=2`) / SAC (`context_fn=partial(
create_selective_checkpoint_contexts, policy)` with MUST_SAVE for `aten.mm|addmm|bmm`, PREFER_RECOMPUTE otherwise):

```
mode     layer ckpt flags               loss   max|dgrad|  saved bytes  vs none
none     [0, 0, 0, 0]             5.55481005    0.000e+00      3428612    1.000  OK
full     [1, 1, 1, 1]             5.55481005    0.000e+00       329476    0.096  OK
every2   [1, 0, 1, 0]             5.55481005    0.000e+00      1879044    0.548  OK
sac      [1, 1, 1, 1]             5.55481005    0.000e+00       329476    0.096  OK
ALL GRADIENTS MATCH
```

Bit-identical loss and gradients in fp32 (torch warns that SAC-saved tensors bypass `saved_tensors_hooks` unless
`respect_saved_tensors_hooks=True`, hence the SAC byte count equals full here; 2.3 sets that flag). Checkpointing
is numerically neutral; the only risk is in what HF/torch recompute, not in the loss.

### 2.3 Measured bytes saved per token per layer (`research/vet/K11/measure_saved_bytes_scaled.py`, `measure_output.txt`)

bf16, HF Qwen3 modeling (sdpa, which on CPU saves O + lse like flash-attn: 0 T x T bytes), parameters excluded
(FSDP re-gathers weights), measured at 1/8 (8B) and 1/4 (1.7B) of the real H/I/head counts and scaled back
(every per-token tensor is linear in H, I, nh*d), per layer = (saved(L=4) - saved(L=2)) / 2 / T:

| mode | Qwen3-8B KB/token/layer | Qwen3-1.7B KB/token/layer | what is kept |
|---|---|---|---|
| none | 302.6 | 158.3 | 4 x I (gate, up, silu, act*up) + ln in/out + qkv pre/post RoPE + O + fp32 RMSNorm upcasts (104 KB of the 8B figure: (T,H) fp32 x2, q/k-norm fp32) |
| full (today) | 8.0 | 4.0 | layer input only |
| every2 (HF `every_n_layers=2`) | 155.3 | 81.2 | mean of one kept, one checkpointed layer |
| sac (mm outputs) | 84.0 | 44.0 | q,k,v,o,gate,up,down outputs + input = exactly 76+8 / 40+4 KB |

The audit's "T x 102 KB" per 8B layer (04 §1.1 row 3, §5.6) is 3x too low: it missed the fp32 norm upcasts and the
second q/k/v copy. That flips its headroom claim (next section). `use_liger=True` (opt-in) would shrink the norm
share; `use_fused_kernels` does not touch the layers.

### 2.4 Memory and time arithmetic

Measured BASELINE-A (`bench/results/baseline-a-main-1`, main@2ab35292, 8xH100, Qwen3-1.7B, GRPO n=8, bs 128,
1024/2048, `ppo_max_token_len_per_gpu=24576` = the harness default, not 16k): step 31.2 s = gen 12.8 +
old_log_prob 3.5 + ref 3.4 + update_actor 8.7 + update_weights 2.0 + adv 0.6; `actor/perf/max_memory_allocated_gb`
15.0 (reserved 18.1); 1.35 M tokens/step = 168 k per GPU = ~7 micro-batches of <= 24 k; `perf/mfu/actor` 0.26.

Activation memory ABOVE today's full checkpointing (layers x tokens x KB), Qwen3-1.7B, 28 layers:

| mode | @24k tokens | @16k | total @24k with the 15 GB in use |
|---|---|---|---|
| none (`enable_gradient_checkpointing=False`, existing knob) | +106 GB | +71 GB | 121 GB: does not fit at either budget |
| every2 (keep 14) | +53 GB | +35 GB | 68-71 GB: fits, little slack |
| sac | +27 GB | +18 GB | 42 GB: fits |
| checkpoint 21 of 28 (keep 7) | +26 GB | +18 GB | 41 GB: fits |

So at the owner's own benchmark the existing all-or-nothing knob cannot be turned off even at 16k (the audit said
it fits), while SAC or keep-7 fit with 35+ GB to spare: the new knob is the only way to buy back recompute here.

Qwen3-8B, 36 layers, static ~18.4 GB/GPU (bf16 params + fp32 master, grads, Adam over 8 ranks) + logits/CE
10-15 GB at 16k: none +178 GB (no); every2 +92 GB (no); keep 6 of 36 +30 GB (~63 GB, fits); sac +50 GB (~83 GB,
no at 16k; +25 GB = ~50 GB at 8k, fits). For the rubric's 8B reference user the mode that fits at 16k is "checkpoint
30 of 36" (which buys back only 1/6 of the recompute), and SAC only at a halved micro-batch budget.

Time: the update is 1F + 1F recompute + 2F backward (04 §2.1). The LM head is never checkpointed and is 18 % of
1.7B's forward FLOPs (8 % of 8B's), so the recompute is 0.82F / 0.92F. SAC recompute re-runs the norms, RoPE, SiLU,
mul (memory-bound, ~0.1F) plus attention unless the flash-attn op is also saved (at BASELINE-A's ~1.35 k-token
sequences attention is ~11 % of layer FLOPs; ~30 % at 8 k), so ~0.15-0.25F remains. Expected at BASELINE-A
(update_actor 8.7 s, MFU 0.26 so part of it is not FLOP-bound): every2 -0.41F of 4F = -10 % of update_actor =
-0.9 s = -2.8 % of the step; sac -0.6F = -15 % = -1.3 s = -4 %; keep-7 -0.2F = -5 % = -0.4 s. For 8B/8k-response
runs the update share is larger (~35 %) but the fitting modes are weaker: keep-6 -4 % of update_actor at 16k; sac
at 8k -15..-20 % of update_actor minus the cost of twice as many micro-batches. These are single-digit step gains:
the claim's -10..-20 % is of the actor UPDATE, and only for modes that fit.

### 2.5 Interactions

- Ulysses SP (`ulysses_sequence_parallel_size>1`, :248-252; attention monkey-patched by `apply_monkey_patch`
  :328-334): the all-to-all already sits inside the layer HF checkpoints, so N-layer modes change nothing; SAC's
  dispatch mode recomputes unlisted ops (the functional all-to-all included), the same symmetric recompute as today.
- LoRA (`_build_lora_module` :343-387): `enable_input_require_grads` :344 exists for reentrant checkpointing;
  non-reentrant modes are unaffected and PEFT layers are plain `aten.mm` under the policy.
- Activation offload: mutually exclusive by construction (:550-551 disables HF ckpt; its own uses
  `use_reentrant=True` :425); the new knobs must raise when `enable_activation_offload=True`.
- FSDP1 (default `strategy: fsdp`, `engine/fsdp.yaml:79`) and FSDP2: both wrap the decoder layer outside
  `GradientCheckpointingLayer.__call__`; unshard happens in pre-forward/pre-backward hooks, so the checkpoint region
  sees plain tensors (torchtitan runs SAC + FSDP2 exactly so). torch.compile: not applied to the model.
- VLMs: `_set_gradient_checkpointing` counts vision and text `GradientCheckpointingLayer`s in one index; an N-layer
  loop in verl must index per stack (`model.language_model.layers`) or document the interleave. Critic shares
  `_build_module` (`FSDPEngineWithValueHead` :1616).
- SAC + custom ops: flash-attn 2.8.3 (`uv.lock:1161`) registers its varlen forward as a `torch.library` custom op, so
  the policy can MUST_SAVE it by name (to confirm on GPU; CPU has no flash-attn); Liger kernels (`use_liger`) and
  fused kernels are opt-in and would simply be recomputed. A misclassified op costs recompute time, not correctness.

## 3. Duplicate / ownership check (research/07 §8, `research/scripts-a7/dupcheck.py`)

`python3 -I research/scripts-a7/dupcheck.py --kw gradient_checkpointing_kwargs --kw "selective activation" --kw
"selective checkpoint" --kw "checkpoint every" --kw every_n_layer --kw recompute_num_layers --kw "activation
checkpointing" --kw "gradient checkpointing" --merged`: 6 open hits, none implements it, merged titles: none.

| PR | what it is | overlap |
|---|---|---|
| #8402 (zpltys, draft, 1 d) NeMo AutoModel engine for fully-async RL | mentions "dsv4 whole-layer activation checkpointing" inside the Automodel engine | different engine |
| #8338 (HydrogenSulfate, 2 d) Megatron recompute backward memory | Megatron `recompute_num_layers` path | different engine |
| #8300 (DeeproChoudhury, 6 d) shared-prefix packing for FSDP | mentions gradient checkpointing in its benchmark table | touches `_build_module`'s file, not the ckpt lines |
| #8096 (cyanseek, 8 d) fused LM heads through FSDP2 hooks | ran its matrix "including activation checkpointing" | none |
| #7688, #5855 | recipe / Ray error handling | none |

`--path verl/workers/engine/fsdp/transformer_impl.py` returns 30+ open PRs touching the file (#8419, #8056, #7866,
#8272, #8026, #8300, #8353, #8309, #8194 ...): the file is hot, but the change is a 2-line hunk at :339-340 plus a
helper, trivially rebased. Issues: #7146 "[Bug][FSDP] Nemotron-H actors cannot enable gradient checkpointing"
(adjacent, hybrid model); nothing asks for selective or every-k. Roadmap #8186: no line. Maintainer statements: none.
Upstream precedent: HF shipped `every_n_layers` (#48200) and `offload`; torchtitan and Megatron expose selective
modes; verl's torchtitan engine already defaults to `selective`. Recent committers on the file: Joel (#8180, #8171,
#7873, #7272, #7161), MinjieZhang (#7428, #7458), vOv (#8010), Lei Ding (#7874); expect wuxibin89/vermouth1992 as
the merging reviewer (07 §4b). Verdict on G1: nobody is doing this in verl.

## 4. Minimum valuable PR

`[fsdp, cfg] feat: selective and partial activation checkpointing for the FSDP engine`

- `verl/workers/config/model.py` (+~12 after :117): `gradient_checkpointing_num_layers: Optional[int] = None`
  (number of decoder layers to checkpoint, Megatron `block` semantics, None = all; HF's `every_n_layers` is not
  used because it can only keep >= half the layers) and `gradient_checkpointing_mode: str = "full"`
  (`full | selective_mm`), `__post_init__`: `num_layers >= 0`, `selective_mm` needs `enable_gradient_checkpointing`,
  both rejected with `enable_activation_offload`.
- `verl/trainer/config/model/hf_model.yaml` (+6 commented lines after :34), regenerate the four
  `verl/trainer/config/_generated_*_trainer.yaml` (`tests/special_sanity/test_config_docs.py` checks parity),
  `docs/examples/config.rst:242-247` (+6), `docs/perf/perf_tuning.rst:113` (+4).
- `verl/workers/engine/fsdp/transformer_impl.py` :339-340 -> ~20 lines: build `gradient_checkpointing_kwargs`
  (`use_reentrant=False`, plus `context_fn=partial(create_selective_checkpoint_contexts, _mm_policy)` for
  `selective_mm`); after `gradient_checkpointing_enable`, if `num_layers` is set, flag the first N
  `GradientCheckpointingLayer`s per stack and clear the rest. `_mm_policy` (~12 lines, module level): MUST_SAVE for
  `aten.mm | addmm | bmm | _scaled_mm` and the flash-attn varlen forward op when importable; PREFER_RECOMPUTE else.
- Tests: `tests/workers/test_fsdp_selective_ckpt_on_cpu.py` (the vet script: loss/grad equality for none / full /
  N-layer / selective_mm on a tiny Qwen3, flags as expected, offload combination rejected) and a parametrised case
  in `tests/models/test_engine.py` (GPU: FSDP1/FSDP2 x SP 1/2, `selective_mm` with flash-attn).
- ~70 source lines + ~90 test lines + docs; defaults (`None`, `full`) reproduce today's behaviour exactly; opt-in.
- Naming: `model.*` is the HF-model group and the docs already say "Megatron uses override_transformer_config", so
  `gradient_checkpointing_num_layers` (maps to Megatron `recompute_num_layers` + `block`) and
  `gradient_checkpointing_mode` (maps to `recompute_granularity`) keep both vocabularies recognisable.

## 5. Proof on the Modal harness (bench/README.md)

- BASELINE-A as pulled (24 k budget) and a 16 k variant (`--ppo-max-token-len-per-gpu 16384`), 8 steps each,
  mean of steps 2..8: A main; B `+actor_rollout_ref.model.gradient_checkpointing_num_layers=21`; C
  `+actor_rollout_ref.model.gradient_checkpointing_num_layers=14`; D
  `+actor_rollout_ref.model.gradient_checkpointing_mode=selective_mm`; E (expected OOM, documents the headroom claim)
  `actor_rollout_ref.model.enable_gradient_checkpointing=False`.
- Metrics: `timing_s/update_actor` (8.7 s -> expected ~8.3 B, ~7.8 C, ~7.4 D), `actor/perf/max_memory_allocated_gb`
  (15.0 -> ~41 B, ~68 C, ~42 D at 24 k), `timing_s/step` (31.2 -> ~30.8 / ~30.3 / ~29.9), and `actor/pg_loss`,
  `actor/grad_norm`, `actor/entropy` curves equal within bf16 noise (checkpointing is neutral: the CPU test is
  bit-exact in fp32).
- Qwen3-8B pair for the reference user: main at 16 k vs `selective_mm` at 8 k (equal-memory comparison) and
  `num_layers=30` at 16 k; expect -15..-20 % and -4 % of `timing_s/update_actor` respectively.
- Noise: steps 2..6 of the pulled run vary 29.8-32.6 s (`update_actor` 8.3-9.3 s, +-6 %), so a -2.8 % step claim
  needs 10+ steps or must be reported on `update_actor`; the memory column is deterministic and is the honest half.

## 6. Adversarial answers (research/VETTING-RUBRIC.md)

- A1 Strongest reason it will not be merged: the step-level gain is single-digit (-2.8..-4 % at BASELINE-A) and the
  biggest lever (no checkpointing) already exists; a maintainer may answer "pass HF's `every_n_layers` through a
  generic kwargs field" or defer to the torchtitan/Automodel engines the roadmap (#8186) is pushing. A perf PR that
  cannot show >= 5 % on named hardware is below what wuxibin89 has merged (07 §4c); the 8B equal-memory pair is the
  number that decides it.
- A2 What makes the gain disappear: memory-bound runs (8B at 16 k, long responses, large token budgets) where only
  "checkpoint 30 of 36" fits (-4 % of update_actor); `enable_activation_offload=True` (modes rejected); H200/B200
  users (141/192 GB) who turn checkpointing off outright; SAC without the flash-attn op saved (attention is 11-30 %
  of layer FLOPs, so the recompute saving shrinks by that much); `use_liger` norms (less fp32 upcast, so the memory
  side of the story shrinks, not the time side); a very small model (0.6B) that fits with ckpt off already.
- A3 Correctness risk: numerically neutral (bit-exact on CPU). Residual risks: SAC's dispatch mode with custom ops
  (flash-attn varlen, Liger, fused kernels) recomputes rather than corrupts, but an op with hidden RNG or state
  would diverge (none in the default path: dropout 0, no MoE routing in dense Qwen3; MoE routers under SAC must save
  their topk, as torchtitan does); any later `gradient_checkpointing_enable/disable` resets the per-layer flags
  (activation offload :550-551 does; PEFT helpers could); VLM stacks interleave unless indexed per stack. Tests that
  would catch a regression: `tests/models/test_engine.py`, `tests/models/test_transformers_ulysses.py`,
  `tests/utils/test_activation_offload.py` (GPU), plus the new CPU equality test.
- A4 Who else: nobody in verl (0 PRs, 0 issues). qgallouedec (HF/TRL) owns the HF knob and may surface it in TRL; the
  Automodel (#8402) and torchtitan engines carry their own modes; Megatron users already have `recompute_*`.
- A5 Half scope: `gradient_checkpointing_num_layers` only (block semantics; ~25 source lines, config, docs, CPU test),
  no SAC policy: zero custom-op risk, measurable at BASELINE-A (keep-7 and keep-14 both fit), and it is the only
  mode that fits the 8B/16k reference user at all. `selective_mm` follows once the flash-attn op is saved and the
  GPU test exists; it is the larger win (-15 % of update_actor at BASELINE-A).

## 7. Gates, scores, verdict

- G1 Not a duplicate: PASS (no open or merged PR, no issue; the HF upstream feature is unsurfaced in verl).
- G2 Not reserved: PASS (no maintainer/RFC claim; hot file, 2-line hunk).
- G3 Self-contained: PASS (~70 source lines, one area, no new dependency, no engine fork).
- G4 Provable: PASS (`timing_s/update_actor` and `actor/perf/max_memory_allocated_gb` on Modal; the CPU equality
  test already passes bit-exact; the memory column is deterministic).
- G5 Default-path impact: PASS, weakly. Opt-in knob, but `enable_gradient_checkpointing: True` is the default FSDP
  population, and at the owner's own BASELINE-A the existing off-switch does not fit while the new modes do.
- S1 Speed/value 3 (-10..-15 % of update_actor, -2.8..-4 % of step at BASELINE-A; -4 % of update_actor for the
  8B/16k reference user, -15..-20 % only at a halved budget). S2 Demand 2 (no issue; HF, torchtitan, Megatron
  precedent). S3 Certainty 4 (mechanism bit-exact on CPU; memory measured per layer; wall time derived from the
  measured step split, not yet measured). S4 Merge-ability 4 (small opt-in diff, CPU test, docs, trivial
  conflicts; reviewer appetite for a single-digit perf PR is the open question). S5 Novelty 2 (HF shipped
  every_n_layers; the mm-policy SAC is the only new piece in verl).
- Verdict: GO-WITH-CHANGES. Changes: (1) do not use HF's `every_n_layers` as the knob; expose "checkpoint N of L
  layers" so memory-bound runs can keep a few layers; (2) ship `selective_mm` with the flash-attn forward op saved
  and a GPU test, since it is the mode that pays at BASELINE-A; (3) lead the PR body with the measured memory
  table (the audit's headroom numbers were 3x off) and the 8B equal-memory `update_actor` pair, not a step-time
  claim; (4) reject the modes with `enable_activation_offload`. Strongest reason for: it is numerically neutral,
  opt-in, ~70 lines, nobody else is doing it, and at the owner's own benchmark it is the only way to spend the
  60 GB of idle HBM on fewer recomputes. Strongest reason against: the step-level number will be single-digit.
