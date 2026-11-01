# K1 vlm-dedupe — adversarial vet (2026-10-29)

Clone: verl-project/verl @ 2ab35292 (read-only). Scripts and outputs: `research/vet/K1/`
(`poc_dedupe_cpu.py` + `.out`, `sim_colocation.py` + `.out`). Venv: `.venv-a9` (torch 2.15 CPU,
transformers 5.21.0, no peft). GitHub read-only via `gh`.

**Verdict: NO-GO as a top-3 speed contribution; GO-WITH-CHANGES only as a small, exact, default-off
hook if the owner wants a VLM-lane PR.** Strongest reason: under V1's unconditional per-sample
seqlen balancing only ~6 % of duplicate images ever share a micro-batch, the vision tower is 2–12 %
of forward FLOPs for the common VLM configs on a healthy stack, and the cross-phase cache that would
recover more is not exact under the default LoRA config (peft `all-linear` adapts the tower) and
cannot skip the tower in `update_actor` without PR #8249's freeze. The 42× headline (#5365) measured
a ~10 s/image ViT on torch 2.9 (the conv3d regression the maintainers cited in #5111), not verl's
current stack.

## 1. Claim

Source: `research/02-perf-pain-points.md` §Theme A and §3.1 (rank 1); `research/13-edge-embodied.md`
gap (g) and candidate C4. For Qwen2.5-VL / Qwen3-VL GRPO the vision encoder runs on every sample's
`pixel_values` in every training-side pass (old_log_prob, ref, update_actor), so an image shared by
the `n` GRPO siblings is encoded `3·n` times per step; a user's manual dedupe cut vision time
6 420 s → 154 s (#5365). Proposal: (1) encode each unique image once per micro-batch (exact
`index_select` expansion); (2) a per-step cache keyed by image id reused across the three passes
when the tower's weights are identical between passes; (3) rollout side: stop re-preprocessing /
re-sending frames every turn (13 §C4), possibly by enabling vLLM's multimodal processor cache.

## 2. Mechanism verification

### 2.1 Where `pixel_values` flow (default config, V1 trainer, FSDP engine)

* Engine build → `apply_monkey_patch(model=module, use_remove_padding=…, ulysses_sp_size=…,
  use_fused_kernels=…)` at `verl/workers/engine/fsdp/transformer_impl.py:328-334`.
* `verl/models/transformers/monkey_patch.py:387-391` installs `qwen2_vl_base_forward` as
  `Qwen2_5_VLModel.forward` / `Qwen2VLModel.forward` and `forward_with_normal_backend` on the
  CausalLM class; `:438-443` does the same for Qwen3-VL / Qwen3-VL-MoE (`qwen3_vl_base_forward`).
  This is unconditional for these `model_type`s (no flag), so HF's own `get_image_features`
  (`transformers 5.21: modeling_qwen2_5_vl.py:1076-1087`) is bypassed. Note: there is **no**
  `qwen2_5_vl.py` in the clone (the brief's filename is wrong); Qwen2.5-VL goes through `qwen2_vl.py`.
* `qwen2_vl_base_forward` (`qwen2_vl.py:421-436`) → `_get_input_embeds` (`qwen2_vl.py:342-402`):
  `model.visual(pixel_values, grid_thw=image_grid_thw)` once per forward at `:352-354` on the whole
  micro-batch's concatenated `pixel_values`, then `masked_scatter` at `:368`. Qwen3-VL:
  `qwen3_vl_base_forward` (`qwen3_vl.py:307-324`) → `_get_input_embeds` (`:206-298`), visual call
  `:216-218`, deepstack embeds returned at `:293-298`. Text-only micro-batches run a dummy 16-patch
  visual forward (`qwen2_vl.py:388-394`, `qwen3_vl.py:278-287`) because FSDP needs every parameter
  touched — the same constraint applies to any cached path that would skip the tower (see 2.3).
* Micro-batch assembly: `transformer_impl.py:1180`
  `extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))` → `verl/utils/model.py:583-642`:
  one dict per sample, `torch.cat(values, dim=0)` at `:629-630` for `pixel_values` and `image_grid_thw`;
  `model_inputs.update(multi_modal_inputs)` at `:1339`. The per-sample dicts are produced once per
  trajectory by the HF processor in `_compute_multi_modal_inputs`
  (`verl/experimental/agent_loop/agent_loop.py:899-949`, called from
  `verl/trainer/ppo/v1/agent_loop_tq.py:193`, stored as field `multi_modal_inputs` at `:207`), so the
  `n` siblings of a prompt carry `n` bit-identical copies of the same `pixel_values` rows (processor
  determinism verified bitwise, PoC below).
* Result: the tower is called **once per micro-batch**, but on **every image occurrence**; nothing
  detects duplicates (`grep -rn dedup verl/` → only `qwen2_5_vl_dedup_image_tokens`,
  `verl/workers/rollout/utils.py:93`, a vLLM placeholder-token collapse, unrelated).
* Phases: `verl/trainer/ppo/v1/trainer_base.py:585-612`: old_log_prob (`:586`), ref (`:591`),
  values (`:596`, PPO only), update_actor (`:611`). Under LoRA the ref pass is the same engine inside
  `disable_adapter()` (`verl/workers/engine_workers.py:458`, `transformer_impl.py:1104`). The critic
  (`transformer_impl.py:1570-1590`) reuses `prepare_model_inputs` (`:1574`) so PPO adds a 4th encode;
  `ppo_epochs > 1` multiplies update_actor's. **The `3·n` encodes per unique image per step claim
  holds.** Megatron takes `pixel_values` into the mcore model (`verl/models/mcore/model_forward.py:147-152`,
  #8038 merged by wuxibin89) and is out of K1's scope — #3906's Megatron users would not benefit.

### 2.2 Do the `n` siblings land in the same micro-batch? Mostly no.

* `_balance_batch` is **unconditional** in V1 (`trainer_base.py:583`; legacy honoured
  `trainer.balance_batch`) and partitions per sample with Karmarkar-Karp, `equal_size=True`,
  no uid awareness (`:1623-1628`). The uid-aware `get_group_balanced_partitions`
  (`verl/utils/seqlen_balancing.py:590-660`) is wired only in legacy `ray_trainer.py:1198-1221`
  under `use_prefix_grouper`. Per rank, `prepare_micro_batches` (`verl/workers/engine/utils.py:92-130`)
  → `rearrange_micro_batches` (`seqlen_balancing.py:348-506`): KK over per-sample workloads with
  `equal_size=False` (`:449`) then the interleave sort (`:485-506`). `force_group_size` exists but is
  RM-only (`engine/utils.py:106`). TQ keys are `{uid}_{session_id}_{i}` (`agent_loop_tq.py:198`), so a
  V1 group-aware option is wirable.
* `research/vet/K1/sim_colocation.py` runs verl's real partitioners on the reference shape
  (128 prompts × n=8, dp=8, `ppo_max_token_len_per_gpu=24576`, 1 image/prompt), output
  `sim_colocation.out`:

```
lengths        samples/rank  mb/rank |  A: mb-dedupe  B: rank cache  C: grouped+mb  (encoder work saved per phase)
geo3k-like              128      6.0 |          6.2%          34.3%          42.3%
short-uniform           128      3.0 |         12.6%          34.7%          63.8%
long-tail               128     13.0 |          2.3%          33.7%          20.3%
```

  A = within-micro-batch dedupe only (the cheap exact part): **2–13 % of encodes, not 87.5 %**.
  B = exact per-rank cache across the 3 passes (only valid for an adapter-free, frozen tower):
  encodes/step = 0.66 × samples instead of 3 × samples (4.5× fewer). C = uid-grouped dp split
  (`get_group_balanced_partitions`) + A. The user's "660 → 82 images per GPU" (8× exactly) means all
  siblings were co-resident — that is not what V1 does by default.

### 2.3 Is the vision tower trainable? By default, yes — which breaks the cross-pass cache.

* `freeze_vision_tower` is declared (`verl/workers/config/actor.py:160`, `actor.yaml:52`, the four
  generated YAMLs) and consumed **nowhere** (`grep -rn freeze_vision_tower verl/`). PR #8249 restores it
  (open, unreviewed by a human, see §3).
* LoRA: `target_modules: "all-linear"` and `exclude_modules: None` by default
  (`verl/workers/config/model.py:126,129`) → `LoraConfig` at `transformer_impl.py:364-372`.
  peft 0.19.1 (`verl/uv.lock:3549-3550`) `_maybe_include_all_linear_layers`
  (`src/peft/tuners/tuners_utils.py:1891-1949`) excludes only the output embedding, so the ViT's
  `attn.qkv/proj`, `mlp.*` and the merger linears get adapters; even an explicit LLM-style list
  (`gate_proj/up_proj/down_proj`) matches Qwen2.5-VL's ViT MLP names. Consequences: actor tower ≠ ref
  tower (adapters disabled), and the actor tower changes after every optimizer step. The cross-pass
  cache is exact only for old_log_prob → the first mini-batch of update_actor; ref needs its own
  encode; full fine-tuning trains the tower too. Exactness across all three passes needs a tower
  with no trainable/adapted parameters, which today requires an explicit `exclude_modules` regex
  (and #8249 for full FT).
* Skipping the tower inside `update_actor` while it holds trainable params silently stops training
  it under FSDP2 (unit never forwarded, no grad) and errors under FSDP1 ("parameters not used") —
  the reason the dummy forward exists. So a cache may only serve forward-only phases, or a tower
  excluded from wrapping (#8249's mechanism). Nothing in the proposal addresses this.

### 2.4 What to cache and what it costs

Post-merger image embeddings (`tokens × out_hidden`, bf16) and, for Qwen3-VL, the deepstack list
(3 × the same; `qwen3_vl.py:218, :293-298`). From the HF configs (fetched 2026-10-29):
Qwen2.5-VL-3B 154-token image 0.6 MiB, 1280-token 5 MiB; Qwen3-VL-2B 2.4 / 20 MiB; Qwen3-VL-8B
4.8 / 40 MiB. 128 unique images/step on 8 dp ranks: 10–640 MiB per rank. Memory is not the problem.

### 2.5 How big is the prize on a healthy stack? Small. (derived)

FLOP share of the tower per sample, from the HF `config.json` sizes (vision block params:
Qwen2.5-VL ~0.49 B, Qwen3-VL-2B 0.30 B, Qwen3-VL-8B 0.41 B; full attention in 4 of 32 ViT layers
for Qwen2.5-VL, all layers for Qwen3-VL; LM over prompt + 2048-token response):

| model | tokens/image | V/(V+L) forward | ideal n=8 saving, fwd phases | V/(V+3L) train, frozen tower | ideal saving, update_actor |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 154 / 400 / 1280 | 4.1 / 9.2 / 19.8 % | 3.6 / 8.1 / 17.4 % | 1.4 / 3.3 / 7.6 % | 1.2 / 2.9 / 6.7 % |
| Qwen2.5-VL-7B | 154 / 400 / 1280 | 1.9 / 4.3 / 10.1 % | 1.6 / 3.8 / 8.8 % | 0.6 / 1.5 / 3.6 % | 0.6 / 1.3 / 3.2 % |
| Qwen3-VL-2B | 154 / 400 / 1280 | 4.8 / 11.8 / 30.4 % | 4.2 / 10.3 / 26.6 % | 1.7 / 4.3 / 12.7 % | 1.5 / 3.7 / 11.1 % |
| Qwen3-VL-8B | 154 / 400 / 1280 | 1.5 / 3.9 / 11.8 % | 1.3 / 3.4 / 10.3 % | 0.5 / 1.3 / 4.3 % | 0.4 / 1.2 / 3.7 % |

"Ideal" = every duplicate removed (needs B or C above). Multiply by the co-location fractions of
2.2 for the cheap variant: geo3k-like (~250 tokens/image, Qwen2.5-VL-3B) ≈ 6 % × 6 % ≈ 0.4 % of
old_log_prob/ref, ~0.1 % of update_actor. The step-level effect is inside run-to-run noise (±3 %
on the harness). #5365's numbers are a different regime: `old_log_prob` 2 534 s for 21 s of `gen`
means ~10 s per image in the ViT; paul-org's reproduction was torch 2.9.0 (comment 2026-03-28), and
the maintainer answer on #5111 attributes the slow VLM step to the torch 2.9 `conv3d` regression
(pytorch/pytorch#166122) — the ViT patch embed is a Conv3d. The bench image runs torch 2.13
(`bench/README.md`), so that regime is not reproducible there **(guess for #5365; stated for #5111)**.

### 2.6 CPU proof of exactness (`research/vet/K1/poc_dedupe_cpu.py`, output `poc_dedupe_cpu.out`)

Tiny random-weight Qwen2.5-VL and Qwen3-VL (transformers 5.21), 8 rmpad-concatenated samples from
2 unique images (4 siblings each, shuffled, distinct responses), naive vs unique-encode + `index_select`,
compared against verl's own `_get_input_embeds` output:

```
qwen2.5-vl attn=eager images=8 unique=2 patches=160 tokens=40 | image_embeds bitwise=True max|d|=0.000e+00 deepstack bitwise=None | inputs_embeds bitwise=True verl-wrapper==dedupe bitwise=True | logits bitwise=True max|d|=0.000e+00 | visual naive 9.9 ms, dedupe 1.9 ms
qwen2.5-vl attn=sdpa  images=8 unique=2 patches=160 tokens=40 | image_embeds bitwise=True ... logits bitwise=True max|d|=0.000e+00 | visual naive 11.0 ms, dedupe 1.6 ms
qwen3-vl   attn=eager images=8 unique=2 patches=160 tokens=40 | image_embeds bitwise=True max|d|=0.000e+00 deepstack bitwise=True | inputs_embeds bitwise=True verl-wrapper==dedupe bitwise=True | logits bitwise=True max|d|=0.000e+00 | visual naive 5.1 ms, dedupe 1.9 ms
qwen3-vl   attn=sdpa  images=8 unique=2 patches=160 tokens=40 | image_embeds bitwise=True ... logits bitwise=True max|d|=0.000e+00 | visual naive 3.2 ms, dedupe 2.4 ms
processor   grid=[[1, 6, 8]] rows=48 same-image-twice rows==2x: True copy1 bitwise=True copy2 bitwise=True
RESULT PASS
```

Bitwise-identical embeddings, deepstack features and logits on both models and both attention
backends: the within-micro-batch dedupe is exact by construction (the ViT is block-diagonal per image
via `cu_seqlens`; window index and rotary positions are per image). The processor is deterministic,
so a `(uid, image_idx)` key is an exact identity without hashing pixels. The mechanism is real; the
dispute is only about how much it is worth and where it can be applied.

### 2.7 Rollout-side half (13 §C4)

* `_compute_multi_modal_inputs` (`agent_loop.py:899-949`) runs the HF processor once per trajectory
  (n× per prompt) on CPU inside the agent-loop workers; siblings may be on different workers, so a
  per-step memo is awkward and it is not GPU time.
* `tool_agent_loop.py:244-253` re-sends all images each turn; `vllm_async_server.py:689-701` builds
  `TokensPrompt(multi_modal_data=…)`. verl sets `mm_processor_cache_gb=0` **only for vLLM < 0.22**
  (`vllm_async_server.py:1416-1418`, the pause/resume desync workaround). `pyproject.toml:95` pins
  `vllm==0.32.0`, `docker/Dockerfile.stable.vllm:13` 0.24.0, so on every current image vLLM's
  content-hash processor cache is already on, and `enable_prefix_caching: True`
  (`verl/workers/config/rollout.py:289`) lets siblings share the image prefix KV (vLLM V1 hashes mm
  items into block hashes), i.e. the ViT runs once per unique (image, prefix) inside vLLM. The
  "one-line win" is already the default; the 55-preprocessings figure describes cache lookups, not
  encodes. Nothing to contribute here beyond a doc note.

## 3. Duplicate / ownership check

* `python3 -I research/scripts-a7/dupcheck.py --kw pixel_values --kw "vision encoder" --kw "vision tower"
  --kw "visual encoder" --kw dedup --kw "image cache" --kw "multimodal cache" --kw mm_processor
  --kw freeze_vision --kw "vit cache" --path verl/models/transformers/{qwen2_5_vl,qwen3_vl,qwen2_vl}.py --merged`:
  37 open hits. Relevant ones (number, state, last commit, files overlap):
  * **#8249** richardChenzhihui (NONE) "[model, fsdp, cfg] feat: restore freeze_vision_tower and skip the
    dummy visual forward for frozen vision towers" — OPEN, MERGEABLE, REVIEW_REQUIRED, 17 commits,
    29 files (~+1 900), last commit 2026-10-19; reviews only from bots and the author as of today.
    Overlaps K1 in `qwen2_vl.py`, `qwen3_vl.py`, `glm4v.py`, `qwen3_5.py`, `monkey_patch.py`,
    `transformer_impl.py`, `utils/model.py`, `config/actor.py`. It freezes/unwraps the tower and skips
    only the **dummy** forward; it does not dedupe real images. Not a duplicate, but a direct
    conflict and the prerequisite for any cache that skips the tower in `update_actor`.
  * #8414 Turoad "[model] perf: build Qwen3-VL/Qwen3.5 pos-embed interpolation tensors without Python
    lists" — `qwen3_vl.py`, 0 d old; small conflict risk.
  * #7996 Tohrusky "[data, rollout] fix: avoid duplicate Qwen VL media resize" — rollout preprocessing
    (`do_resize=False`), different mechanism; shows the area gets unreviewed first-time PRs (35 d, no review).
  * #6245 aoshen02 (stale, CONFLICTING) direct vLLM `MultiModalInput`; #5780 RobotGF TQ for image data;
    #7323 (stale) Qwen3.5 vision call alignment; #6320 (stale) `freeze_module_pattern`; #6670 (stale)
    tower-connector LoRA; #8096/#8097/#8090 fused-head fixes touching the same wrappers.
  * Merged dump (2026-07-16 → 10-28): #8038 wuxibin89 Megatron THD vision token embedding (09-28);
    #8035 chenyingshu unpack visual output in dummy paths (09-29); #7562 YZJF offload multimodal dataset
    processing (08-27); #7067 disable MM processor cache for Qwen3.5 Ascend (07-16). **None dedupes
    encodes or caches embeddings.**
* `gh pr list --state all --search "vision encoder cache OR dedup pixel_values OR unique images"` and the
  issue search for "vision encoder duplicate OR dedup image GRPO": no match.
* Maintainer statements: #5365 (label bug, 0 reactions, no assignee) has two comments by
  Kirrito-k423 (COLLABORATOR, 2026-03-17 and 2026-08-05) listing dedupe/cache as "可尝试的规避/优化"
  (workarounds to try) — a suggestion to the reporter, not a claim; everyone else on the thread is
  NONE. No RFC or roadmap item (01-roadmap-demand.md:40,145 moved "SOTA multimodal VLM RL recipe"
  to verl-recipe / verl-omni; the training engine stays in verl). verl-omni is a separate repo for
  generation RL (#5755, #5916), not a dedupe owner.
* Recent authors of the touched files (`git log`): `qwen2_vl.py`/`qwen3_vl.py` — fixes #8035
  (chenyingshu), #7703, #6659, #6268; `transformer_impl.py` — #8180, #8171 (ckpt), #8010, #7874;
  `utils/model.py` — #8043, #6715 (Gemma4 multimodal). Expected reviewer: wuxibin89 / the engine
  owners **(guess from history)**. G1 passes; G2 passes with the #8249 caveat.

## 4. Minimum valuable PR (the only shape that is exact today)

Title: `[model, fsdp] feat: encode each unique image once per micro-batch for Qwen2.5-VL/Qwen3-VL (opt-in)`
(tags `model`, `fsdp` pass `tests/special_sanity/check_pr_title.py:22-25`). FSDP engine only, default off.

| file | change | est. lines |
|---|---|---|
| `verl/experimental/agent_loop/agent_loop.py` `_compute_multi_modal_inputs` (`:899-949`) | add `image_ids: LongTensor[num_images]` = 64-bit blake2b of `f"{uid}:{image_idx}"` (images are per-prompt and bit-identical across siblings, PoC); tool-returned images get a per-trajectory id. An int64 tensor rides the existing path unchanged (`utils/model.py:629-630` concatenates it; `agent_loop_tq.py:207` stores it). | +15 |
| `verl/models/transformers/vision_dedupe.py` (new) | `dedupe_visual(visual, pixel_values, grid_thw, image_ids) -> (embeds, deepstack)`: unique ids in first-occurrence order, slice patch rows by `grid_thw.prod(-1)`, one `visual()` call, `index_select` embeds and each deepstack tensor back (the PoC's `dedupe_image_embeds`, 40 lines). Falls through to the plain call when `image_ids is None` or all unique. | +60 |
| `verl/models/transformers/qwen2_vl.py` (`_get_input_embeds :352-354`, `qwen2_vl_base_forward :421-436`), `qwen3_vl.py` (`:216-218`, `:307-324`) | accept `image_ids=None`, route through `dedupe_visual`; video path untouched | +20 |
| `verl/workers/engine/fsdp/transformer_impl.py` `prepare_model_inputs` (`:1180`, `:1339`) | pop `image_ids` from `multi_modal_inputs`; pass it only when `model_config.vision_dedupe` (otherwise drop so the kwarg never reaches HF) | +10 |
| `verl/workers/config/model.py` (`:124-138`), `verl/trainer/config/model/hf_model.yaml`, generated YAMLs | `vision_dedupe: bool = False` | +8 |
| `verl/workers/engine_workers.py` `_postprocess_output` (`:214`) | metric `perf/vision_images_encoded` vs `perf/vision_images_total` so the effect is visible per phase | +15 |
| `tests/models/test_vision_dedupe_on_cpu.py` | tiny Qwen2.5-VL + Qwen3-VL (pattern `tests/models/test_liger_vl_compat.py:34-60`), bitwise equality vs the plain wrapper, mixed text/image micro-batch, multi-image prompt order, `extract_multi_modal_inputs` round-trip of `image_ids` | +120 |
| `docs/examples/multi_modal_example.rst` | one paragraph + the flag | +10 |

≈ 260 lines with tests; one area; no new dependency; no engine fork. Under the V1 default this PR
saves 2–13 % of encoder work in the forward-only phases (2.2) — honestly, a sub-1 % step effect.

Not in the minimum PR (each a separate, later PR, in this order):
1. **uid-grouped `_balance_batch` for V1** (`trainer.balance_batch_group_by_uid: false`): parse uid from
   the TQ key (`agent_loop_tq.py:198`), sort uid-contiguous, call the existing
   `get_group_balanced_partitions` (`seqlen_balancing.py:590`); mirrors legacy `ray_trainer.py:1198-1221`.
   Raises capture to 20–64 % (sim C) and is also what PR #8300-style prefix sharing needs in V1.
2. **Cross-phase per-rank cache**, forward-only phases first (old_log_prob → ref share a cache only
   when the tower has no adapters: detect by walking `module.visual` for `BaseTunerLayer`/`requires_grad`),
   `update_actor` only after #8249 lands (tower excluded from wrapping). Needs an invalidation rule per
   optimizer step and bounded GPU/CPU memory; the research doc's "keyed by image hash" cache without
   these rules is not exact in the default LoRA config (2.3).

## 5. Proof on the Modal harness (`bench/README.md`)

The harness has no VLM model or dataset yet: extend `bench/modal_verl.py::download` with
`Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct` and the geo3k parquet from
`examples/data_preprocess/geo3k.py` (`hiyouga/geometry3k`, `data.image_key=images`) at
`/hf-cache/data/geo3k/{train,test}.parquet`. Then A/B (`--src` main vs branch), 6 steps, mean of 2..6:

```
python3 bench/verl_bench.py train --run-name k1-{main,dedupe}-1 --gpu 'H100!:8' \
  --model Qwen/Qwen2.5-VL-3B-Instruct --n-gpus 8 --rollout-n 8 --train-batch-size 128 \
  --max-prompt-length 1024 --max-response-length 2048 --total-training-steps 6 --tp 1 \
  --overrides "data.image_key=images data.train_files=/hf-cache/data/geo3k/train.parquet \
    data.val_files=/hf-cache/data/geo3k/test.parquet \
    actor_rollout_ref.model.vision_dedupe=True"         # branch only; add trainer.balance_batch_group_by_uid=True for step 1
```

Metrics: `timing_s/old_log_prob`, `timing_s/ref`, `timing_s/update_actor`, `timing_s/step`,
`perf/vision_images_encoded` per phase, `perf/max_memory_allocated_gb`, `critic/score/mean` overlay.
Before any A/B: a 1×H100 micro-benchmark of `model.visual` per image on the bench image (torch 2.13,
flash-attn 2.8.3, transformers 5.16.1) to establish ms/image — if it is ~10 ms (healthy), the table in
2.5 bounds everything below.

Expected effect, with arithmetic (geo3k ≈ 250 tokens/image, Qwen2.5-VL-3B, vision ≈ 6 % of forward
FLOPs): minimum PR 6 % × 6 % ≈ 0.4 % of old_log_prob/ref, ≈ 0.1 % of update_actor → **step −0.3 %,
not resolvable against ±3 % noise**. With the grouped balance: 42 % × 6 % ≈ 2.5 % / 0.9 % → step ≈ −1 %.
With an exact cross-phase cache for a frozen tower: ≤ 6 % of the forward phases → step ≈ −3 %. A
double-digit result needs a large-image, small-LM config (Qwen3-VL-2B, 1280-token images: vision 30 %
of forward; old_log_prob/ref −25 %, update_actor −10 %, step ≈ −10 %), i.e. a config chosen to show
the effect, which a reviewer will notice. The 42× of #5365 is not reproducible on this stack unless the
ViT is pathologically slow (2.5).

## 6. Adversarial answers

**A1 — strongest reason it will NOT be merged.** The reviewer asks for before/after on named
hardware and gets noise: under V1's unconditional per-sample balancing (`trainer_base.py:583,
:1623-1628`) only ~6 % of duplicate images share a micro-batch (sim), and the tower is ≤ 10 % of
forward FLOPs for the common configs (2.5), so the exact, mergeable part of K1 is worth ≈ 0.3 % of
a step. The part that would be worth more (cross-phase cache) is not exact under the default LoRA
config (peft `all-linear` adapts the ViT, 2.3), cannot skip the tower in `update_actor` without the
unmerged #8249, and touches the same wrappers/engine lines as #8249 (13 d old, ~1 900 lines, no
human review yet) — a maintainer will sequence it behind that PR or ask the two authors to
coordinate. A first-time author's VLM-engine PR in a queue that already holds #8249, #7996 (35 d, no
review) and #8414 is not "merged by wuxibin89 within a day" territory.

**A2 — what makes the measured gain disappear.** (i) torch ≥ 2.10 without the `conv3d` regression:
the ViT returns to ~10 ms/image and the prize collapses to the FLOP table; (ii) bigger LMs (7B/8B)
halve the share again; (iii) images ≤ 300 tokens (geo3k, most math/chart sets) keep it under 6 %;
(iv) vLLM ≥ 0.22 already caches on the rollout side, so there is nothing to gain there
(`vllm_async_server.py:1416-1418`); (v) #8249's freeze removes the tower backward, shrinking K1's
share in `update_actor`; (vi) `ppo_mini_batch_size` small / `ppo_epochs` > 1 raise the number of
optimizer steps and shorten any cache's valid window; (vii) a user who groups siblings per rank
(legacy `use_prefix_grouper`, or `balance_batch=False` in legacy) already gets most of A for free.

**A3 — correctness risk and which tests catch it.** Risks: wrong `index_select` order for
multi-image prompts (ids must follow `image_grid_thw` row order); id collisions (64-bit hash — use
`(uid, image_idx)` identity, not pixel hashes, so collisions are impossible by construction);
video (`pixel_values_videos`) silently not deduped (fine, but must not break); Ulysses SP (the
deepstack slicing in `monkey_patch.py:158-230` happens after `_get_input_embeds`, so unaffected, but
untested); text-only micro-batches keep the dummy forward; a cache that skips the tower while it is
trainable silently freezes it under FSDP2 (no error) — the most dangerous failure, and nothing in the
proposal guards it; MFU over-counts after dedupe (`flops_counter.py:159-162` uses `images_seqlens`).
Existing tests: none exercise the VL forward on CPU (`tests/models/test_liger_vl_compat.py` is
accelerator-gated); no VL e2e under `tests/special_e2e/`. Only the new CPU test would catch a
regression; the GPU proof is the geo3k A/B reward curve.

**A4 — who else is on it.** richardChenzhihui (#8249, freeze/unwrap/skip-dummy, same files);
Turoad (#8414, Qwen3-VL pos-embed perf); Tohrusky (#7996, rollout resize); aoshen02 (COLLABORATOR,
#6245 direct vLLM mm inputs, stale); Kirrito-k423 (COLLABORATOR) suggested dedupe/cache on #5365
but filed nothing; wuxibin89 owns the Megatron THD vision path (#8038). Nobody is implementing the
dedupe itself; the risk is sequencing and conflicts, not duplication.

**A5 — half scope.** The §4 minimum PR without the metric and without Qwen3-VL: `image_ids` in
`_compute_multi_modal_inputs`, `dedupe_visual`, Qwen2.5-VL wrapper only, flag, CPU test (~150 lines).
Honest expected gain on defaults < 1 %; its value is the exact hook that the uid-grouped balance
(the better-placed follow-up: it exists in legacy and is missing in V1) and a guarded cache build on.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | **pass** | no open/merged PR dedupes encodes or caches embeddings (§3); adjacent: #8249, #7996, #8414 |
| G2 not reserved | **pass with caveat** | no maintainer claim; #8249 (outside author) is pending in the same lines and is a prerequisite for the cache variant |
| G3 self-contained | **pass** | FSDP engine + two wrappers + agent loop + config; no engine fork, no dependency |
| G4 provable | **weak pass** | exactness proven bitwise on CPU (2.6); the speed effect on the reference VLM config is below harness noise (§5); only a chosen large-image/small-LM config shows double digits |
| G5 default-path impact | **fail** for the exact PR, **conditional** for the full one | opt-in flag; V1's balancing scatters siblings; VLM users are a minority of the reference user base; the demanding issues come from a torch-2.9 regime |

| score | value | note |
|---|---|---|
| S1 speed/value | **2** | ≈ 0.3 % step on defaults; ≈ 3 % with a frozen-tower cache; 10 % only on large-image small-LM configs; 0 for the 8×H100 / Qwen3-8B text reference user |
| S2 demand | **3** | 8 issue reports (#5365 ×3 reproductions, #6706, #5111, #4896, #4743, #3906), 0 reactions, one collaborator nudge, no RFC |
| S3 certainty | **4 / 2** | exactness measured (bitwise); speed derived, and the headline numbers come from a pathological ViT |
| S4 merge-ability | **2** | four wrappers + engine + loop + config; conflicts with #8249; unreviewed-queue precedent; first-time author |
| S5 novelty | **2** | user patches exist on the issue; vLLM does it on the rollout side; EasyR1/others not checked **(gap)** |

**Verdict: NO-GO as a top-3 pick** (GO-WITH-CHANGES only if the owner wants a VLM-lane PR, and then
as §4's minimum PR plus the uid-grouped balance, measured first with the ms/image micro-benchmark).
Strongest reason: the cheap exact dedupe recovers ~6 % of duplicates under V1's default balancing
and the tower is ≤ 10 % of forward FLOPs on a healthy stack, so the mergeable PR gains ≈ 0.3 % of a
step; the variant that gains more is inexact under default LoRA, blocked on #8249 for `update_actor`,
and would have to be benchmarked on a config picked to flatter it. I could not find a reason the
mechanism does not work — it is exact — only that it is not worth what the research doc claims.
