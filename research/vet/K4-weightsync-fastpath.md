# K4 — weightsync-fastpath: bf16 at the sender, double-buffered IPC buckets, batched gathers (+ C4 LoRA export speed)

Vetter verdict file (adversarial). Clone `<repo>/verl` @ `2ab35292`,
read-only; every `file:line` is into that tree. Scripts + outputs in `research/vet/K4/`. GitHub
read-only; dumps of 2026-10-28 in `gh/`. The LoRA-disaggregated bug (folded into this brief) has its
own verdict in `research/vet/K4b-lora-disaggregated-bug.md`.

## 1. Claim

Sources: `research/05-rollout-weightsync-audit.md` §3.1 fixes (a)/(b) and candidate C3;
`research/04-engine-gpu-audit.md` §6 finding 5; `research/02-perf-pain-points.md` §3.3 (issue
#7015); `research/11-edge-weight-updates.md` C4. Claim: the DEFAULT colocated sync (V1 forces
`backend="naive"`, `verl/trainer/ppo/v1/trainer_base.py:383`) exports FSDP's fp32 master weights with
one all-gather per parameter and ships them uncast through a single 2 GiB CUDA-IPC bucket in
lock-step with the receiver's load, so (a) casting each shard to the rollout dtype before
`full_tensor()` halves gather and bucket bytes and bucket count, (b) a second bucket overlaps
gather+copy of bucket k+1 with the engine's load of bucket k, (c) batching small parameters cuts
~400 collectives to ~40; together "−50-60 % of `update_weights`" (report 05). C4: skip the whole-model
staging for non-merged LoRA, cast the adapter to bf16, default `layered_summon=True`.

## 2. Mechanism verification

### 2.1 The default path (naive, FSDP → vLLM), with file:line
1. `trainer_sync.py:36-38` (`marked_timer("update_weights")`) → `CheckpointEngineManager.update_weights`
   `verl/checkpoint_engine/base.py:518-520` → `actor_wg.update_weights(mode="naive")` →
   `verl/workers/engine_workers.py:785-886` (`:827-828` expandable segments off + `empty_cache`,
   `:840-841` `rollout.resume(["weights"])`, `:845-847` export, `:863-865` `rollout.update_weights`,
   `:870-874` offload + `synchronize` + `empty_cache`, `:881-882` barrier + `resume(["kv_cache"])`).
2. Export `verl/workers/engine/fsdp/transformer_impl.py:981-1065`: training dtype is forced to fp32
   (`:264-266`, `torch_dtype=None and not forward_only`); `:1020` `self.module.state_dict()` (sharded
   DTensors); `:1033-1039` per parameter `param.to(device, non_blocking=True).full_tensor()` — one
   all-gather per parameter, **no cast**; `:1040` `unfuse_moe_params`. The delta-shard export casts the
   local shard first (`:946-948`, `p.to(torch.bfloat16)` on the DTensor, then `to_local()`); the
   merged-LoRA export casts **after** the gather (`:1086`). So the pattern exists in-tree, unused here.
3. Sender `verl/workers/rollout/vllm_rollout/vllm_rollout.py:229-241`: fires the receiver RPC
   `update_weights_from_ipc` non-blocking (`:229-233`), then `BucketedWeightSender(bucket_size_mb=
   config.checkpoint_engine.update_weights_bucket_megabytes)` (`:235-240`; default 2048,
   `verl/trainer/config/rollout/rollout.yaml:308`, `verl/workers/config/rollout.py:136`).
   `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:102-166`: one device buffer of
   `bucket_size` uint8 (`:183`); the cast is explicitly declined (`:118-125`, vermouth1992: "some
   parameters (such as moe gate) have to keep fp32 precision ... the rollout should automatically
   cast on demand. However, this would incur a higher weight transfer volume"); per-tensor alignment
   (`:129-130`); when the next tensor does not fit: `synchronize()` → `send_pyobj(meta)` → **`recv()`**
   (`:133-136`); tensors larger than the bucket bypass it through a direct IPC handle (`:140-146`,
   `:223-237`); the last bucket likewise (`:160-164`).
4. Receiver `:269-307`: rebuild views on the shared buffer, `on_bucket_received(weights, is_last)`,
   `synchronize()`, ack `send(b"")` (`:300-304`); for the last bucket the ack is deferred to `_cleanup`
   after the buffer is released (`:306`, `:346-353`). `on_bucket_received` is
   `verl/workers/rollout/vllm_rollout/utils.py:302-330` → `_update_weights` → `:395-403`
   `model.load_weights(...)` per bucket; `process_weights_after_loading` once at the end (`:340-346`).
   Hence the sender's `recv()` returns only after vLLM has finished loading bucket k: zero overlap,
   as claimed. The ack protocol and the single buffer are exactly as reports 04/05 describe.
5. **Where the dtype conversion happens**: inside vLLM's per-parameter `weight_loader`
   (`param.data.copy_(loaded_weight)` and its sharded variants), into parameters created in the
   rollout `dtype` (`rollout.yaml:43` `bfloat16`). The receiver never casts explicitly; the fp32→bf16
   rounding is the `copy_` kernel's round-to-nearest-even. The LoRA receiver is the exception: it packs
   each bucket into one host block of vLLM's `lora_dtype` (`utils.py:306-313`, merged #8352).
6. `nccl` / `nixl`: both constructors take a `rollout_dtype` (nccl `:129-147`, nixl `:249-261`) that is
   **stored and never used** on the send path (`grep -n rollout_dtype`: only the assignments);
   `split_weight_chunks` (`base.py:565-591`) views bytes as-is (`weight.view(-1).view(torch.uint8)`),
   so fp32 goes on the NCCL/RDMA wire too. `kimi` (`kimi_checkpoint_engine.py:42-50`) and
   `delta_sharded` (`delta_checkpoint_engine.py:528-529`) do cast to `rollout_dtype`. Megatron exports
   bf16 through mbridge (`verl/workers/engine/megatron/transformer_impl.py:1004-1013`), so the 2× is
   FSDP-only (and VeOmni, which #8110 addresses).

### 2.2 Where a sender-side cast is NOT identical (the guard the flag needs)
- Receiver parameters vLLM keeps in fp32: DeepSeek-family `e_score_correction_bias`, Mamba/hybrid
  `A_log` / `D` / `dt_bias` (Qwen3-Next class). A blanket pre-cast would round them to bf16 before
  vLLM upcasts them again — lossy. All of them are 1-D and tiny; a "floating-point and `ndim >= 2`"
  rule covers >99.9 % of the bytes and none of these; #8110 instead fetches the receiver's real
  `{name: dtype}` map in a preflight RPC (the correct general answer, and the expensive one).
- Quantized receivers: `is_quantized_model` → `load_quanted_weights` (`utils.py:382-386`,
  `verl/utils/vllm/vllm_quant_utils.py:235`) quantizes whatever dtype arrives; fp32→bf16→fp8 double
  rounding differs from fp32→fp8, so the cast must be off whenever `rollout.quantization` is set
  (`rollout.yaml:486`). Trainer-side quantizers: FSDP consumes no `_rollout_weight_quantizer` (only
  Megatron does, `megatron/transformer_impl.py:1017`); the FSDP QAT path (`transformer_impl.py:1042-1062`)
  feeds `QATQuantizer.quantize_with_fusion`, which casts to `param_dtype` itself
  (`verl/utils/qat/quantizer.py:212` `tensor.to(device=..., dtype=self.param_dtype)`), so a bf16
  pre-cast is identical there when `param_dtype` is bf16 (the default, `:1048-1050`) and must be
  skipped otherwise. Net: the flag is provably identical only for an unquantized bf16/fp16 receiver
  with no fp32-kept parameters → it cannot be default-on; it is an opt-in like #8110's.
- MXFP4 / #8287: the unified quantizer path is Megatron/VeOmni-side on this clone; the FSDP branch
  above is the only FSDP consumer of the fp32 stream.

### 2.3 C4 (trainer-side LoRA export speed), verified against the code
- `_skip_staging = fsdp_version == 2 and not _is_peft` (`transformer_impl.py:992-995`, offload back at
  `:1025-1026`): the staging round trip only moves bytes when `actor.fsdp_config.param_offload=True`;
  BASELINE-A runs the actor with no offload (`bench/README.md:151`), so for the reference config the
  cost is zero. With offload it is 2 × the per-rank fp32 shard (4.1 GB per rank for 8B on 8 GPUs,
  ~0.2-0.4 s each way at 12-25 GB/s), not 2 × 32.8 GB as report 11 states for a single-GPU trainer.
- `layered_summon` defaults to False (`verl/workers/config/rollout.py:293`, `rollout.yaml:111`) and
  hard-requires the base to be preloaded (`fsdp_utils.py:716-721`); flipping the default is a
  behaviour change for `load_format=dummy` users, not a perf fix.
- Adapter dtype: the `.cpu()` copies (`fsdp_utils.py:745-751`) carry the fp32 master; the receiver
  already packs them into vLLM's `lora_dtype` block (#8352). For 8B r=8 that is 44 MB → 22 MB on a
  PCIe-class copy: ~1-2 ms. Not a contribution.

### 2.4 CPU test: cast-before-gather is bitwise identical, and the bucket arithmetic
Script `research/vet/K4/k4_cast_order_cpu.py` (output `k4_cast_order_cpu.out`; run with
`../../../.venv-k4/bin/python k4_cast_order_cpu.py`, no GPU). It execs the REAL
`bucketed_weight_transfer.py` source (sha256 `b835c242…`, two import lines substituted by
`k4_shim.py`: the device module and the trivial `ensure_async_iterator`), runs `BucketedWeightSender`
and `BucketedWeightReceiver` over zmq ipc + POSIX shm, and compares the receiver's bf16 parameters
(`torch.empty(bf16).copy_(t)`, what vLLM's weight loader does) for the two orders: today's
(all-gather fp32 → send fp32 → cast on copy) and K4(a) (cast each shard → all-gather bf16 → send bf16
→ copy). Seven tensors seeded with every value class (NaN, ±Inf, ±0, subnormals, bf16 rounding ties,
±3e38), 2,654,464 elements:

```
transfer(current) done in 0.19s ; transfer(k4_cast_first) done in 0.18s
model.embed_tokens.weight (4096, 256) ... lm_head.weight (4096, 256): bitwise_equal=True for all 7
all 2654464 elements bitwise identical (int16 view, NaN payloads included)
buckets: current=2 (bytes 10.1 MiB)  cast-first=1 (bytes 5.1 MiB)
offline check: cat(shards).to(bf16) == cat(shard.to(bf16)) bitwise for every tensor
```
Why it must hold: `full_tensor()` is a byte copy (all-gather + concatenation along the shard dim),
and `.to(bf16)` is elementwise with the same RNE conversion (`c10::BFloat16(float)`, also used by
`copy_`), so `cast ∘ cat == cat ∘ cast` exactly; a DTensor `.to(bf16)` is likewise elementwise on the
local shard (the delta export already relies on it, `transformer_impl.py:946-948`). The identity is
for bf16/fp16 receivers only (§2.2).

Side-finding from the first run: with `use_shm=True` (NPU / no CUDA IPC) a tensor larger than the
bucket is a hard `AssertionError` (`bucketed_weight_transfer.py:140-144`); the CUDA-IPC path falls
back to a direct handle (`:145`, `:223-237`). Qwen3-8B's fp32 `embed_tokens` / `lm_head` are 2.49 GB >
the 2 GiB default, which is issue #5952 ("Weight too large to fit in bucket", Qwen3-VL-8B, 2026-04).
A sender-side bf16 cast would make them 1.24 GB and fit — a robustness argument for (a) on the
shm path only.

Exact replay of the sender's packing rule (`:127-158`, alignment + flush + direct send) for the Qwen3
shapes (36×11+3 = 399 tensors for 8B, 28×11+2 = 310 for 1.7B; 8.191 B / 1.721 B params),
`update_weights_bucket_megabytes=2048`, 8 trainer ranks, each rank streaming the whole model to its
co-located vLLM worker:

| model | wire dtype | all-gathers | bytes/rank | buckets | direct sends | all-gather recv/rank | NVLink @300 GB/s | PCIe-only @20 GB/s | bucket copy @1.3 TB/s | lock-step RT @7 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | fp32 (today) | 310 | 6.88 GB | 4 | 0 | 6.02 GB | 20 ms | 0.30 s | 5 ms | 28 ms |
| Qwen3-1.7B | bf16 (K4a) | 310 | 3.44 GB | 2 | 0 | 3.01 GB | 10 ms | 0.15 s | 3 ms | 14 ms |
| Qwen3-8B | fp32 (today) | 399 | 32.76 GB | 15 | 2 | 28.67 GB | 96 ms | 1.43 s | 25 ms | 119 ms |
| Qwen3-8B | bf16 (K4a) | 399 | 16.38 GB | 8 | 0 | 14.33 GB | 48 ms | 0.72 s | 13 ms | 56 ms |

Assumptions stated in the table header (per-rank all-gather receive rate on an NVSwitch H100 node;
HBM read+write copy rate; a fixed 7 ms per lock-step round trip = `synchronize` + ZMQ pyobj + the
receiver's per-tensor view/`load_weights` Python for ~25-50 tensors + `synchronize` + ack). Python
overhead of the cast itself: 0.9 µs per parameter (0.36 ms per 8B sync) — irrelevant.

### 2.5 Measured anchors and the expected effect
- **BASELINE-A on main@2ab35292** (`bench/results/baseline-a-main-1/metrics.jsonl`, key
  `data/timing_s/update_weights`, Qwen3-1.7B, 8×H100, vLLM TP=1, naive): 2.151 (warm-up), 2.008,
  2.150, 2.171, 1.910, 2.020 s → **mean 2.05 s over steps 2-6, spread ±6 %**, of a 31.15 s step.
- **#7864 (merged, 8×H200, verl v0.9.0, after its GC fix)**: naive `update_weights` Qwen3-8B dense
  **4.207 s**, 8B with parameter offload 3.031 s, Qwen3-30B-A3B 7.086 s.
- **Issue #7015** (ChangyiYang, FSDP2 export): per-parameter gather loop profiled at **0.1-0.25 s**
  when shards are GPU-resident; 2.4-2.7 s per sync only in the offloaded no-staging regime, and the
  issue's own caveat attributes it to per-parameter *pageable H2D*, not to collective count.
- The byte-proportional part of today's sync, from the table: 1.7B ≈ 20 + 5 + 28 + ~5 (receiver
  load) ≈ **60 ms of 2,050 ms (3 %)**; 8B ≈ 96 + 25 + 119 + ~15 ≈ **255 ms of 4,207 ms (6 %)**.
  Everything else (≥ 94 %) is the envelope: `wake_up(["weights"])`, ~310-400 `full_tensor` launches,
  vLLM `load_weights` Python, `process_weights_after_loading`, two `empty_cache`, the barrier,
  `wake_up(["kv_cache"])`, five Ray RPC round trips, and the trainer offload when enabled. Report 05's
  "5-15 s observed, −50-60 %" was labelled (guess) there; the measurements above replace it.
- **Expected effect** of (a)+(b): 1.7B ≤ 30 ms (a) + ≤ 25 ms (b: overlap of 4 receiver loads) ≈
  **≤ 55 ms = 2.7 % of the sync = 0.18 % of the step**, inside the sync's own ±6 % spread; 8B ≤ 130 ms
  (a) + ≤ 120 ms (b: 17 × ~7 ms) ≈ **≤ 0.25 s = 6 % of the sync ≈ 0.1-0.25 % of a 100-250 s step**.
  Independent confirmation: #8110's author measured **3.29-4.48 %** export-time reduction (256 → 128
  MiB payload) for the same cast on VeOmni; #5619's author measured **0.88-4.94 %** for the double
  buffer on the IPC path (13-27 % on the shm path, 910C). Three sources, same answer: ≤ 5 %.
- (c) batched gathers: 310-400 launches × ~0.3-0.6 ms of DTensor Python per `full_tensor` is
  0.1-0.25 s (matches #7015's profile) → at most 5-10 % of the sync if replaced by ~30 unit-level
  `unshard()` calls, and only measurable without param offload; the offloaded case needs pinned bulk
  staging (#7015's option 1), a different change.
- The only regime where (a) is a visible win is a PCIe-only box (table: 1.43 → 0.72 s at 8B), which
  is not the maintainers' reference hardware (H20/H100/H200 NVSwitch) and rarely has 8 GPUs.

## 3. Duplicate / ownership check (research/07 §8 recipe; `dupcheck.py` + `gh` read-only)

| PR / issue | state | overlap with K4 |
|---|---|---|
| **#8110** "[veomni, vllm, rollout] feat: export weights using verified receiver dtypes" (tongyx361, COLLABORATOR) | open since 2026-10-04, last commit 2026-10-08, CONFLICTING, 0 reviews, 0 comments; depends on #8109 (CI FAILURE) | **the same mechanism as (a)**: casts temporary shards to the receiver's verified dtype *before* gathering, opt-in `checkpoint_engine.export_receiver_dtype` (default false), receiver `{name: dtype}` metadata RPC in `vllm_rollout/utils.py` + `vllm_async_server.py`, all-rank preflight in `checkpoint_engine/base.py`, `verl/utils/rollout_weight_dtype.py`, config + docs + 4 CPU tests; scoped to VeOmni (FSDP is "unsupported layout" today). Its measured gain: 3.29-4.48 % export time. An FSDP version either builds on it (blocked, collaborator-owned) or re-implements its infrastructure (duplicate). |
| **#5619** "[perf] optimize the bucketed weight transfer using double buffer" (jianjunzhong, draft) | 2026-03-16, last commit 226 d ago, CONFLICTING, 2,012 lines over 5 engines, "hold until #5378" | **the same mechanism as (b)**; measured 0.88-4.94 % on IPC (910C), 13-27 % on shm. A successor must state the difference; the numbers argue against writing one. |
| **#7015** issue (ChangyiYang, 2026-07-10) | open, no PR (`gh pr list --search "7015 in:body"`: none) | (c) batched gathers via `FSDPModule.unshard()`; owned by the delta-sync author as an idea; its own profile says the big cost is pageable H2D under offload, not collectives. |
| #8395 sharded_nixl (rafiw, NVIDIA) | 2026-10-26, MERGEABLE | no trainer-side all-gather at all, disaggregated only; touches `bucketed_weight_transfer.py` + its test file → rebase churn for (b). |
| #7108 Mooncake P2P (3,418 lines), #8088/#8087/#7552/#7432 NCCL M2N, #8272 HCCL delta, #7060 delta tracking | open | disaggregated transports; orthogonal to the naive path. |
| #8435 (SGLang tensor lifetime, 0 d), #8352 merged (aoshen02, LoRA host block), #7864/#7873 merged (GC removal; #7873 by wuxibin89) | — | the maintainers' own refit line: GC/pinning/lifetime fixes with measured A/B; none is dtype or double-buffer. |
| #7137 | open | **unrelated** (Ulysses `cu_seqlens`, `monkey_patch.py`); the brief's "SP path" label is a mislabel. |
| #8246 (aoshen02 draft, FP8 GRPO + LoRA on FSDP2), #7366 (receiver failure surfacing), #7920 (oversized CPU weights), #7866 (Kimi K3 NPU), #5378 (stale) | open | all touch `bucketed_weight_transfer.py` (8 open PRs on the file). |
| Roadmap #8186 "Weight refit optimization" | — | items: delta shard for quantized `(weight, scale)`, NCCL M2N #8088. No bf16-on-sender or double-buffer line. |

Ownership: `verl/checkpoint_engine` has no CODEOWNER; `verl/workers/rollout/vllm_rollout` →
wuxibin89, PeterSH6, chenhaiq, ArronHZG, HollowMan6; `verl/workers/engine/fsdp` → eric-haibin-lin,
vermouth1992, ZihengJiang, ArronHZG. By git: wuxibin89 ("Joel") authored the weight-sync clean-up
(#7179), the GC fix (#7873) and the unified quantizer (#8287) and merges the line; aoshen02 has the
latest commit on `vllm_rollout/utils.py` (#8352) plus drafts #8105/#8106/#8246; ChangyiYang owns delta;
tongyx361 owns receiver-dtype (#8110) and standalone lifecycle (#8168); Parinay Chauhan the NCCL
multi-sender. The standing design note against unconditional sender casts is vermouth1992's
(`bucketed_weight_transfer.py:120-124`). Contention: `engine_workers.py` 19 active PRs,
`transformer_impl.py` 24, `checkpoint_engine/base.py` 12 (research/07 §2b) → weekly rebases.

## 4. Minimum valuable PR (what it would be; not recommended)

- (a) as a follow-up to #8110 once it merges: `transformer_impl.py:1033-1039` cast the local DTensor
  shard to the receiver dtype before `.full_tensor()` when the preflight map says bf16/fp16
  (`p.to(dtype)` on the DTensor, as `:946-948`), skip under `_qat_enabled`/`rollout.quantization`;
  ~40 lines + `tests/utils/test_fsdp_export_dtype_on_cpu.py` (~80 lines, the identity check above)
  + a line in `docs/advance/rollout_weight_dtype.md`. Config key: #8110's
  `checkpoint_engine.export_receiver_dtype` (default off — it cannot be default-on, §2.2).
- (a) standalone (if #8110 dies): `checkpoint_engine.sender_cast: null|bf16|fp16` casting
  floating tensors with `ndim >= 2` only, refused when `rollout.quantization` is set or `rollout.dtype`
  differs; ~60 lines in `transformer_impl.py` + 20 in `config/rollout.py`/`rollout.yaml` (+ regenerated
  YAMLs) + the CPU test. Must cite #8110 and explain the narrower guard.
- (b): two buffers in `BucketedWeightSender` (`_init_buffer` allocates 2, flush sends bucket k and
  waits for the ack of k−1 instead of k; receiver unchanged except the deferred-ack rule for both
  buffers), flag `checkpoint_engine.double_buffer` default off, parametrize
  `tests/utils/test_bucketed_weight_transfer.py`; ~120 lines. Must cite #5619 and the #7873 IPC race.
- Total ≈ 250-300 lines, 2 flags, 2 CPU tests, one GPU A/B. Title `[rollout, vllm, perf] feat: ...`.

## 5. Proof on the Modal harness

- Metric: `data/timing_s/update_weights` in `metrics.jsonl` (V1 sync timer, `trainer_sync.py:36-38`;
  it wraps the whole `engine_workers.update_weights` envelope). BASELINE-A gives 2.05 ± 0.12 s at 1.7B;
  the expected (a)+(b) delta there is ≤ 55 ms — **unresolvable** (the sync's own spread is ±6 %).
- 8B run (proposed): `--preset baseline-a` with `Qwen/Qwen3-8B`, `rollout.tensor_model_parallel_size=2`,
  `train_batch_size=128`, `n=8`, 6 steps, A/B flags off/on, **≥ 3 repeats each** because the expected
  ≤ 0.25 s is ≈ the sync's noise; report mean(2..6) of `update_weights` and `step`. Expected: 4.2 →
  ≥ 3.95 s (≤ 6 %), step −0.1-0.25 %.
- Go/no-go gate before any of that: a per-phase breakdown of the 2.05 s at 1.7B — RLInsight
  (`trainer.logger=[console,file,rl_insight]`, `engine_workers.py:781-784` `update_weight` trace state)
  or five `marked_timer`s in a local branch around `resume(weights)`, export+send, offload, barrier,
  `resume(kv_cache)`. Proceed only if the gather+bucket phases exceed 30 % (the arithmetic says ~3 %).
- Correctness: greedy outputs of 32 fixed prompts with the flag on/off must be byte-identical for a
  bf16 receiver (the CPU proof predicts it); `tests/checkpoint_engine/test_correctness_on_gpu.py`
  covers nccl/nixl/kimi only, so a naive-path correctness test (the shape of #5378/#5619's
  `test_naive_correctness.py`) would have to be added.

## 6. Adversarial answers

- **A1** Strongest reason it will not merge: it re-does #8110 (collaborator, open) and #5619 (stale,
  measured ≤ 5 %) for a gain the arithmetic bounds at ≤ 6 % of a 2-4 s sync — a PR body cannot show
  "before/after on named hardware" that beats the run-to-run spread without many repeats, and
  vermouth1992's in-file note argues against the cast by default.
- **A2** What makes the gain disappear: #8110 merging (an FSDP port is then a 40-line follow-up, not
  a contribution); quantized rollouts (fp8/MXFP4) and models with fp32-kept params (DeepSeek bias,
  Mamba `A_log`/`D`), where the flag must stay off; Megatron trainers (already bf16); NVSwitch
  hardware (bytes are ~3-6 % of the sync); larger `update_weights_bucket_megabytes` (fewer round trips).
- **A3** Correctness risk: lossy casts for fp32-kept receiver params and double rounding on quantized
  receivers (§2.2); for (b) a second IPC buffer doubles the exposure to the CUDA-IPC ref-count race
  #7873 fixed (`:346-353`). Tests that would catch it: none on the naive path today (GPU correctness
  tests cover nccl/nixl/kimi); `test_bucketed_weight_transfer.py` catches transport corruption only.
- **A4** Who else: tongyx361 (#8110/#8109), jianjunzhong (#5619/#5378), ChangyiYang (#7015, delta),
  rafiw (#8395), dongha-yoon/ss16118 (#8088/#7552), aoshen02 (#8246), Levius-Fubuki (#8435),
  wuxibin89 (the refit line itself).
- **A5** Half scope: nothing of K4 survives a halving with value — the honest minimum is the §5
  breakdown measurement, which is not a PR; or the 40-line FSDP follow-up inside #8110 after it lands.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | **FAIL** for (a) | #8110 open 24 d, same mechanism + infrastructure, collaborator; (b) only stale #5619 (would pass with an explanation) |
| G2 not reserved | **FAIL** (soft) | tongyx361 is actively doing receiver-dtype export; wuxibin89 works in the file himself (#7179, #7873) |
| G3 self-contained | pass | 3-4 files, no engine fork, no dependency |
| G4 provable | **FAIL** | expected ≤ 2.7 % (1.7B) / ≤ 6 % (8B) of a 2-4 s sync = ≤ 0.25 % of a step; BASELINE-A resolves ~5 % on the step and ~6 % on the sync |
| G5 default-path impact | pass | FSDP + vLLM naive is the default |

Scores: S1 **1** (≤ 0.25 % of the reference step; PCIe-only boxes are the only real win), S2 **2**
(#6225 and #7015 want faster colocated sync, roadmap heading exists, but no issue asks for bytes),
S3 **4** (the smallness is certain: arithmetic anchored on two measured syncs, confirmed by #8110's
and #5619's own numbers; the CPU identity proof is exact), S4 **2** (small diff possible, but collides
with #8110, the standing in-file note, and 19-24 active PRs per file), S5 **1**.

**Verdict: NO-GO.** Strongest reason: ≥ 94 % of the measured naive sync (2.05 s at 1.7B on BASELINE-A,
4.2 s at 8B per #7864) is not byte-proportional, so (a)+(b)+(c) are bounded at ≤ 6 % of the sync
(≤ 0.25 % of a step) — three independent measurements agree — and the one mechanism with a general
guard is already an open collaborator PR (#8110). C4 adds nothing for the reference config (no param
offload; adapter bytes ~22 MB) and `layered_summon=True` by default is a behaviour change.

What would flip it: a per-phase trace showing > 30 % of the sync in gather/bucket lock-step (contradicting
the arithmetic), or a maintainer target on PCIe-only hardware. What the data point at instead
(not vetted here, worth a candidate id): the sync is **6.6 % of the BASELINE-A step (2.05 of 31.15 s)
while its bytes cost ~60 ms** — the ~1.9 s envelope (two `wake_up`s, ~310 `full_tensor` launches, vLLM
`load_weights` Python, `process_weights_after_loading`, 2 × `empty_cache`, barrier, 5 Ray RPCs) is the
actual target, and a measurement-first PR of that shape (#7864/#7873 style: find the accidental sleep,
A/B it) is the kind wuxibin89 merges in a day.
