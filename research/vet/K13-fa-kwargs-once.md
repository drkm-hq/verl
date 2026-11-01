# K13 "fa-kwargs-once": pass packed varlen kwargs to flash attention once per micro-batch

Vetted 2026-10-29 (adversarial vetter). Clone: verl-project/verl main@2ab35292 (read-only). Sources read
in situ: transformers 5.16.1 (`.venv-cpu`, the bench image's pin) and 5.21.0 (`.venv-a9`), flash-attn
2.8.3 `flash_attn_interface.py` (fetched from the v2.8.3 tag). Script: `research/vet/K2/k13_sync_count.py`.

## 1. Claim

Source: research/04-engine-gpu-audit.md §3.4, finding F6, candidate C5(a).

On the default path verl hands the HF forward only `input_ids` / `position_ids`
(`verl/workers/engine/fsdp/transformer_impl.py:1268-1272`), so HF's `_flash_attention_forward`
re-derives the varlen metadata (`cu_seqlens`, `max_seqlen`) from `position_ids` in every decoder layer
with host syncs (the audit counted 2 per layer, ~144 per training micro-batch on Qwen3-8B). Passing
`cu_seq_lens_q/k` and Python-int `max_length_q/k` (HF's `FlashAttentionKwargs`) once per micro-batch
makes HF skip the derivation; claimed 1-3 % of step time, more for small models.

## 2. Mechanism verification (file:line)

transformers 5.16.1 `modeling_flash_attention_utils.py`:

| item | evidence |
|---|---|
| per-layer entry | `_flash_attention_forward` :694; `is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))` :766; `is_fa_with_varlen_kwargs = all(kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k))` :767-769; `elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:` :796; `if cu_seq_lens_q is None or cu_seq_lens_k is None: ... _prepare_from_posids(...)` :797-800, else reshape only :801-804. Both branches end in the same `flash_varlen_fn(q, k, v, cu_seqlens_q=..., cu_seqlens_k=..., **flash_kwargs(max_seqlen_q=..., max_seqlen_k=...))`: same kernel, same arguments, bitwise-identical attention. |
| sync 1 | `_is_packed_sequence` :536-549 returns a device tensor `(...).abs().sum().bool()` :549; at :796 `False or <tensor>` makes `elif` call `Tensor.__bool__`: one D2H. With the kwargs present `True or ...` short-circuits (the four tiny kernels still launch, no sync). |
| sync 2 | `prepare_fa_kwargs_from_position_ids` :458-495: `indices_q = (position_ids == position_ids.min()).nonzero()` :479 (`nonzero` must learn its output size: D2H); `max_length_q = cu_seq_lens_q.diff().max()` :492 stays a tensor. |
| sync 3 (+1 cheap) | `_process_flash_attention_kwargs` :591-691 passes the tensor max through in eager (`is_tracing` :680/:687 is only true under dynamo / jit / stream capture); its own comment :672-678: "`flash_attn_varlen_func` may require `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`" and `same_max_seqlen = ... # to avoid 2x device syncs`. flash-attn 2.8.3's custom op `flash_attn::_flash_attn_varlen_forward` declares `max_seqlen_q: int` (`flash_attn_interface.py:142-149`; `flash_attn_varlen_func` :1370-1376, docstring :1418 "max_seqlen_q: int"), so each tensor-to-int conversion at the op boundary is another `.item()`; the second finds an already drained queue and is cheap. |
| plumbing | HF `integrations/flash_attention.py:38` `**kwargs` -> :76-96 `_flash_attention_forward(..., **kwargs)`. Qwen3 `modeling_qwen3.py:247,275` (attention `**kwargs: Unpack[FlashAttentionKwargs]` -> `attention_interface(..., **kwargs)`), :302-314 decoder layer, :375-420 model, :458-489 CausalLM; Llama :199-448 likewise; Qwen3-VL text stack :469-502, :537-557, :769-835, :1149-1239, :1300-1365; critic `GenericForTokenClassification.forward(**kwargs: Unpack[TransformersKwargs])` `modeling_layers.py:280-298`. |
| verl's own wrapper | `_ulysses_flash_attention_forward` (`verl/models/transformers/monkey_patch.py:87-156`) is installed whenever `use_remove_padding or ulysses_sp_size > 1` (:544-552), i.e. on the DEFAULT path; it forwards `**kwargs` to `_flash_attention_forward` :146-148 and carries the TODO at :128-130: "we can eliminate this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly". HF gradient checkpointing is non-reentrant (`transformer_impl.py:340`), so int-tensor kwargs pass through the recompute. |
| data verl already has | `packed_cu_seqlens = input_ids.offsets()` :1203 (int64, global); the static-pad segment is appended at :1273-1284 for the custom `cu_seqlens` forward path (today used only when the module's signature has `cu_seqlens`, probe :598); `max_length` is one `.item()` per micro-batch (:1276 already does `int(model_cu_seqlens[-1].item())` on that path). FA wants int32 `cu_seqlens`. |
| upstream still per-layer | 5.21.0: `_is_packed_sequence` docstring "4. ... reads the values and forces a device sync in every layer" (:560-575) and `if _is_packed_sequence(position_ids, batch_size):  # this check is expensive so not precomputed` (:851); the varlen-kwargs path is HF's recommended bypass. The gain survives transformer upgrades; HF leaves the fix to callers. |
| out of scope | Qwen2 / 2.5-VL under rmpad use verl's `_custom_flash_attention_forward` (`verl/models/transformers/qwen2_vl.py:182-232`), which derives cu_seqlens itself from `position_ids` with its own `.all()` sync (:228-231). Ulysses SP>1: the wrapper all-gathers `position_ids` and re-derives (:128-140), #7137's territory. |

### CPU proof (`research/vet/K2/k13_sync_count.py`, run 2026-10-29 in `.venv-cpu`)

Replays lines 766-800 with the real 5.16.1 helpers under a `TorchFunctionMode` that counts
`item / __bool__ / __int__ / __index__ / nonzero / tolist / numpy / cpu`, modelling the op-boundary
conversion as `int(tensor)`:

```
torch 2.13.0  transformers 5.16.1
(a) position_ids only  -> sync-inducing calls per layer: {'__bool__': 1, 'nonzero': 1, '__int__': 2}  total 4
(b) explicit varlen kwargs -> sync-inducing calls per layer: {}  total 0
cu_seqlens / max_seqlen identical between (a) and (b): True
Qwen3-1.7B  28 layers: forward-only micro-batch 112 -> 0 syncs; training micro-batch (forward + gradient-checkpoint recompute) 224 -> 0
Qwen3-8B    36 layers: forward-only micro-batch 144 -> 0 syncs; training micro-batch (forward + gradient-checkpoint recompute) 288 -> 0
Qwen3-32B   64 layers: forward-only micro-batch 256 -> 0 syncs; training micro-batch (forward + gradient-checkpoint recompute) 512 -> 0
```

The audit under-counted: 3 stalling round-trips plus 1 cheap one per layer, not 2; the kwargs remove all
of them, and the FA backward has none (the ints are saved in `ctx`). The mechanism is real.

## 3. Duplicate / ownership check

- `python3 -I research/scripts-a7/dupcheck.py --kw cu_seqlens --kw cu_seq_lens --kw max_length_q --kw FlashAttentionKwargs
  --kw position_ids --kw "host sync" --kw nonzero --merged` -> 49 open hits, relevant:
  **#7137** Zhanli-Li `[model] Pass explicit cu_seqlens through the Ulysses FA-varlen path, eliminating
  position_ids dependency` (OPEN since 2026-07-24, last update 2026-09-21, +90/-10, `monkey_patch.py`
  only, MERGEABLE, **0 reviews**): computes global cu_seqlens inside `_ulysses_flash_attention_forward`
  for SP>1 and stops passing `position_ids`; motivated by the 3-D mrope crash (#6284), the sync removal
  is a side effect; SP=1 untouched. **#8300** prefix packing: its `_prefix_packed_flash_attention_forward`
  pops `cu_seq_lens_q/k`, `max_length_q/k`, `position_ids` from kwargs and passes its own (diff hunk in
  `monkey_patch.py`), so K13's kwargs are ignored under prefix packing: composes, no conflict. **#7096**
  (deferred metric `.item()`s, 5 per micro-batch) is the only other sync PR; different syncs.
- GitHub (2026-10-29): open `cu_seq_lens` -> #7137 only; merged `cu_seqlens flash` -> #6846, #5115, #2182,
  #91 (none pass kwargs); `"7137 in:body"` -> #7932, #7586, #8249 (unrelated); open "device sync item()" -> none.
- `git log -S cu_seq_lens_q -- verl/` -> #5900 (veomni), #3496, #3192, #735: nothing on the FSDP engine path.
- Owners: `monkey_patch.py` recent authors 鐘天楽 (#7703), Weiwei Sun (#6660 Qwen3.5 SP), Qing An
  (#6512), SaltFish11 (#6482); the engine file is wuxibin89's. No maintainer statement either way.

G1 PASS (engine-side kwargs on SP=1 is not what #7137 does; cite #7137 and #8300). G2 PASS.

## 4. Minimum valuable PR

Title (passes the checker; `perf` is a module tag, `fix` the type): `[fsdp, perf] fix: pass packed varlen
kwargs to flash attention once per micro-batch`.

- `transformer_impl.py` `_build_model_optimizer` (:596-600): probe once `accepts_var_kwargs = any(p.kind is
  VAR_KEYWORD for p in signature(module.forward).parameters.values())` and
  `str(getattr(module.config, "_attn_implementation", "")).startswith("flash_attention")` (4 lines).
- `prepare_model_inputs` after :1272: `if self._pass_fa_varlen_kwargs and not self.use_ulysses_sp and
  position_ids_rmpad.dim() == 2:` build `cu` = `packed_cu_seqlens` plus the pad segment exactly as
  :1273-1284, cast to int32, `max_len = int(cu.diff().max().item())`,
  `model_inputs.update(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max_len, max_length_k=max_len)`
  (12 lines; one `.item()` per micro-batch replaces 112-512).
- `tests/models/test_fsdp_no_padding_on_gpu.py`: assert the kwargs, dtypes and the pad segment (pattern
  :141-166), ~20 lines. `tests/models/test_transformer.py` (GPU, rmpad vs padded logits) already guards
  boundary correctness end to end.
- No config key: always on where the probe passes. About 40 lines, under AGENTS.md's "tiny PR" line
  unless it ships with a number.
- SP>1: leave to #7137, or pass the global cu_seqlens through the same kwargs (the wrapper forwards them
  untouched :146-148, which makes #7137's all_gather redundant) - coordinate with its author, do not race.

## 5. Proof on the Modal harness (`bench/README.md`)

The harness records phase wall times, not sync counts, so the first question is whether BASELINE-A can
see this at all. Per GPU per pass: 1.35 M tokens/step / 8 = 168k tokens = ~7 micro-batches of 24576.
Stalls per step: old_log_prob 7 x 112, ref 7 x 112, update_actor 7 x 224 (forward + recompute; the FA
backward has none). A stall costs the GPU the time the CPU needs to refill the queue after the drain:
30-50 us on a GPU-bound layer (a Qwen3-1.7B layer forward at 24576 tokens is 2 x 50.3e6 x 24576 /
450e12 = 5.5 ms, so 3 stalls are ~2 % of a layer). Per step at 40 us: old_log_prob -31 ms of 3.49 s
(0.9 %), ref -31 ms of 3.43 s (0.9 %), update_actor -63 ms of 8.71 s (0.7 %), gen 0 -> **step -0.13 s of
31.15 s = 0.4 %** (1 % even at 100 us per stall). The run-to-run spread is +-5 %: BASELINE-A cannot
prove it, and neither can any 8B run (a Qwen3-8B layer at 16k tokens is 16.5 ms: 0.7 %).

Where it is measurable: small micro-batches on small models, because layer time shrinks and the stall
does not. Qwen3-1.7B at 4096-token micro-batches: layer 0.9 ms vs 0.12 ms of stalls -> 10-13 % of the
forward; at the 16384 default budget: 3.6 ms -> 3 %. So F6's "1-3 %" holds for 1.7B at the default
budget and over-states 8B; C5's "5-10 % for small models" holds only at <= 4k-token micro-batches.

Proposed proof, in order:
1. 1xH100 micro-benchmark (script outside the clone, `tests/models/test_transformer.py` pattern):
   Qwen3-1.7B bf16 FA2, packed micro-batches of T = 4096 and 24576 drawn from the BASELINE-A length
   distribution, 50 forwards each, kwargs on/off; report mean forward ms and `torch.profiler`
   `key_averages()` counts of `cudaStreamSynchronize` / `Memcpy DtoH` per forward (expect 112 + k -> k)
   and max |dlogits| (expect 0: same kernel, same arguments).
2. BASELINE-A with a reduced budget so the effect clears the noise floor:
   `--ppo-max-token-len-per-gpu 4096 --overrides "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
   actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096"`, main vs branch: expect old_log_prob /
   ref -5…-10 %, update_actor -3…-6 %.
3. The stock BASELINE-A pair as the no-regression run (flat).
Metric maintainers accept: forward ms and phase times; the sync count is the mechanism evidence, not
the headline. If (2) shows under 5 %, the PR has no number to stand on.

## 6. Adversarial answers

**A1 Strongest reason it will NOT be merged.** The value is too small to show on any default-sized run:
0.4 % of a BASELINE-A step, under 1 % on 8B at the default budget. A 40-line always-on change that
cannot carry a step-time delta reads as the "tiny PR" AGENTS.md discourages, and it sits next to #7137
(same kwargs, SP path, 0 reviews in three months): a reviewer will say "fold it into #7137" or "show me
the number". The number only exists at a 4k-token budget nobody runs by default.

**A2 What makes the gain disappear.** Large models or large micro-batches (GPU-bound, deep queues: 0.4-
0.8 %); launch-bound regimes (the CPU is the bottleneck anyway, a drained queue costs nothing extra);
HF precomputing the metadata once per forward (not in 5.21.0, explicitly left to callers, but one
upstream PR away); `forward_prefetch` / explicit FSDP prefetch hiding the all-gather anyway (not default:
`engine/fsdp.yaml:30 forward_prefetch: False`); a future verl `torch.compile` of the model (not an option
today).

**A3 Correctness risk, and which tests catch it.** Wrong boundaries silently change attention
(cross-sequence leakage or truncation) with no error. Sources: the `pad_to_length` pad segment (must
append `[padded_total]` as :1273-1284 does, and include it in `max_len`); the SP pad (excluded by the
guard); 3-D mrope position ids (excluded by `dim() == 2`; VLMs use their own attention anyway); int64 ->
int32 cast of verl's offsets; models whose attention drops kwargs (probe + HF interface contract).
Tests: `tests/models/test_transformer.py` compares rmpad vs padded outputs on GPU and would catch
leakage; the new assertion on `model_inputs`; `tests/models/test_engine.py` end to end.

**A4 Who else is working on it.** Zhanli-Li (#7137, SP path, same kwargs, dormant since 2026-09-21);
DeeproChoudhury (#8300 passes explicit cu_seqlens inside its wrapper); the HF maintainers
(`_is_packed_sequence`'s docstring shows they know and chose not to fix it there); nobody on the SP=1
engine path.

**A5 Half scope.** The 12-line kwargs hunk with the `flash_attention` guard and the GPU assertion, no
probe; or the opposite cut: no PR at all, a comment on #7137 proposing to extend it to SP=1 (the TODO
it cites is in the same function).

Bundle into K2? **No.** Different mechanism (attention metadata vs LM-head rows), different default
(always-on vs opt-in), and K2's A/B must stay attributable to K2; a reviewer evaluates one concern per
PR, and a silent attention-metadata change inside an LM-head feature PR widens K2's blast radius for a
0.4 % payoff. AGENTS.md's bundling clause is for mechanical cleanups, which this is not.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | PASS | #7137 is the SP>1 wrapper path, 0 reviews, 3 months; K13 is the SP=1 engine path; both it and #8300 must be named |
| G2 not reserved | PASS | no roadmap line, no maintainer statement |
| G3 self-contained | PASS | ~40 lines in one engine function + a test |
| G4 provable | **FAIL on the reference bench** | 0.4 % of a BASELINE-A step, derived; provable only with a profiler count or a 4k-token-budget run, which is not a default configuration |
| G5 default-path impact | PASS | default path, every flash-attention user |

Scores: **S1 1** (reference user < 1 %; 3 % at 1.7B / 16k; 10 % only at <= 4k-token micro-batches).
**S2 2** (verl's own TODO `monkey_patch.py:128-130` and HF's docstring ask for it; no issue, no
reactions). **S3 3** (sync count measured, 4 -> 0 per layer; time derived). **S4 3** (tiny and
conventional; busywork risk; two open PRs in the neighbourhood). **S5 1** (HF's documented fast path).

**Verdict: NO-GO** as a standalone top-3 candidate, and **do not bundle into K2**. Strongest reason: the
effect on the reference workload is ~0.4 % of a step, below the harness noise floor, so it cannot earn
the before/after number the maintainers merge perf PRs on. Right home: a follow-up `[fsdp, perf] fix:`
after K2 lands, proven with the 4k-budget run and a per-micro-batch sync count, or a comment offering
the SP=1 hunk to #7137's author.