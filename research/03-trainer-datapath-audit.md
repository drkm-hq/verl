# 03 — Trainer data-path audit: driver-side / CPU hot paths of the PPO/GRPO loop

A3 slice. Clone: `<repo>/verl` @ `2ab35292` (main, 2026-10-27).
Machine for all measurements: Apple M5 Max, 18 cores, 128 GB, macOS 25.5, CPU torch (venv `.venv-a3`, see section 3).
Every number in this file comes from a run on this machine with the stated shape; anything else is labelled *guess*.

Line numbers are for the clone commit above.

## 1. Step anatomy — what the driver does per training step

There are two trainers in the tree and they have very different driver data paths:

| | Legacy `RayPPOTrainer` (`verl/trainer/ppo/ray_trainer.py`) | V1 `PPOTrainer` (`verl/trainer/ppo/v1/trainer_base.py`) |
|---|---|---|
| Status | `@deprecated(... removed in v0.9.0, use trainer.use_v1=True)` at `ray_trainer.py:285` | **default**: `trainer.use_v1: true` (`verl/trainer/config/ppo_trainer.yaml:232`); `main_ppo.py:187` |
| Batch container on the driver | `DataProto` holding the whole padded batch in driver RAM | `KVBatchMeta` (keys + tags only); tensors live in TransferQueue (TQ) storage; driver pulls fields with `tq.kv_batch_get` when it needs them |
| How the batch reaches workers | driver chunks the TensorDict per DP rank, `consolidate()`s, `ray.put`s, workers `ray.get` (`decorator.py:208-246`) | driver sends metadata; each worker pulls its rows from TQ inside `tqbridge` (`transferqueue_utils.py:364-376`) |
| Rollout output | agent-loop workers return padded `DataProto`s, driver `DataProto.concat`s them (`agent_loop.py:1278-1286`) | agent-loop workers `tq.async_kv_batch_put` unpadded rows straight into TQ (`v1/agent_loop_tq.py:228-233`); driver never sees rollout tensors until it asks |

Shapes used below: `N = B×n` sequences (1024 or 4096), prompt width `P = 2048`, response width `L = 8192` or `32768`, `S = P+L`. The agent loop pads every sample to `P` and `L` (`agent_loop.py:779-795`), so legacy-path tensors are dense `N×P`, `N×L`, `N×S` int64/fp32 regardless of real lengths. Dense payload per sequence of the rollout output alone (prompts, responses, response_mask, input_ids, attention_mask, position_ids, rollout_log_probs, rm_scores): **448 KB at L=8k, 1.6 MB at L=32k** → **1.8 GB / 6.6 GB for N=4096**. After old_log_probs, ref_log_prob, token_level_scores/rewards, advantages, returns are unioned in (6 × N×L fp32) the batch shipped to `update_actor` is **~2.6 GB (8k) / ~9.8 GB (32k)**.

### 1a. Legacy `RayPPOTrainer.fit()` — per step (`ray_trainer.py:1405-1823`), driver-serialized unless marked *worker*

| # | Op | Where | Data touched | Driver cost class |
|---|---|---|---|---|
| 1 | `collate_fn` (dataloader) | `rl_dataset.py:41-69` | B items; tensors are just `dummy_tensor` (the dataset no longer tokenizes, `rl_dataset.py:386-412`) | trivial |
| 2 | `DataProto.from_single_dict`, `uuid4()`×B, `_get_gen_batch` (pops non-tensor keys, shares arrays) | `ray_trainer.py:1466-1476, 572-586` | B object arrays | trivial |
| 3 | `gen_batch.repeat(n, interleave=True)` | `1480` → `protocol.py:968-1010` | `np.repeat` on B object arrays (pointer copies) | trivial in agent-loop mode (no prompt tensors) |
| 4 | *worker* `AgentLoopManager.generate_sequences`: `prompts.chunk(W)`, one `generate_sequences.remote(chunk)` per agent-loop worker; each worker pads every sample with `tokenizer.pad` (3 calls/sample) and `torch.cat`s into dense tensors | `agent_loop.py:1263-1292, 728-757, 1088-1180` | W pickled chunks of object arrays out; W dense `DataProto`s back (each pickled via `DataProto.__getstate__` = `consolidate()` + `torch.save` to BytesIO + `getvalue()` = 3 copies, `protocol.py:377-399`) | pickle-bound |
| 5 | `DataProto.concat(outputs)` | `agent_loop.py:1286` → `protocol.py:914-958` | `torch.cat` of W dense chunks: **1 full copy of the rollout payload** (1.8/6.6 GB) | memcpy |
| 6 | `combined_gen_output.slice(0,N)`; `batch.repeat(n)`; `batch.union(gen_batch_output)` | `1512, 1532-1533` | `union_numpy_dict` (`protocol.py:188-200`) runs `_deep_equal` **element-by-element in Python** over every non-tensor key present on both sides (uid, data_source, reward_model, extra_info — the agent loop echoes the input non-tensor batch back, `agent_loop.py:1134-1135`) | Python O(N×dict) |
| 7 | `compute_response_mask` | `120-135` | view | trivial |
| 8 | `_balance_batch`: `attention_mask.sum(-1)` (N×S), `calculate_workload`, `get_seqlen_balanced_partitions(workload_tensor, dp, equal_size=True)` (Karmarkar-Karp over N items **fed a torch tensor, so every sort/heap comparison is a 0-d tensor op**), per-partition `sort(key=lambda x:(workload_lst[x],x))` (tensor indexing per item), `batch.reorder(global_idx)` | `1182-1250`, `seqlen_balancing.py:49-172, 213-254`, `protocol.py:960-966` | KK: Python O(N log N) with tensor scalars; **reorder = `self.batch[indices]` = 1 full copy of the batch** (1.8/6.6 GB) | Python + memcpy |
| 9 | `global_token_num = attention_mask.sum(-1).tolist()`; loop over `multi_modal_inputs` | `1545-1552` | second N×S reduction; N-iteration Python loop | small |
| 10 | `extract_reward` | `reward.py:160-167` | dict lookups | trivial |
| 11 | `_compute_old_log_prob`: (a) `batch.to_tensordict()` (b) `left_right_2_no_padding` (c) dispatch (d) collect (e) `no_padding_2_padding`×2 (f) `agg_loss`, `union` | `1290-1325` | (a) `NonTensorStack.from_list([NonTensorData(x) for x in val])` **per non-tensor key over N items** (`protocol.py:1119-1121`; ~15-20 keys in agent-loop mode) (b) `unpad_input` over N×S + **Python loop over N rows** building `position_ids_list` (`padding.py:52-62`) (c) `chunk_tensordict` + `contiguous().consolidate()` per DP chunk (**1 full copy**) + `parallel_put` (**Ray pickle = 1 more copy into plasma**) (`protocol.py:1268-1275`, `decorator.py:208-246`) (d) `concat_tensordict` of per-rank nested outputs (`tensordict_utils.py:247-300`, copy of N×S_valid fp32 ×2) (e) **Python loop over N rows** with `F.pad` + `torch.stack` (`padding.py:120-133`) producing dense N×L fp32, twice | Python + 2-3 full-batch copies |
| 12 | `_compute_ref_log_prob` | `1266-1288` | **repeats 11(a)(b)(c)** on the same batch (now + old_log_probs) → second full `to_tensordict`, second `unpad`, second consolidate+pickle | same again |
| 13 | `_compute_values` (critic) | `1252-1264` | third repeat of 11(a)(b)(c) | same again |
| 14 | `apply_kl_penalty` or alias; rollout-correction; `compute_advantage` | `78-118, 187-283`; `core_algos.py:268-331` (GRPO) | GRPO: **two Python loops over N** with 0-d tensor arithmetic per sample plus `torch.stack/mean/std` per group (B groups) | Python O(N) |
| 15 | `_update_critic` | `1378-1403` | fourth repeat of 11(a)(b)(c) on the full batch (+advantages, returns, values) | same again |
| 16 | `_update_actor` | `1327-1376` | **fifth** repeat of 11(a)(b)(c) on the full batch (~2.6/9.8 GB) | same again |
| 17 | `compute_data_metrics`, `compute_timing_metrics`, `compute_throughout_metrics`, `compute_variance_proxy_metrics` | `metric_utils.py:429-610, 611-653, 655-689, 691-803` | `_compute_response_info` computed twice (N×S sums), 3 `masked_select`s over N×L, ~25 `.item()`s | tens of ms |
| 18 | validation (`test_freq`): `_validate` | `596-720` | `[item... for item in test_batch]` iterates `DataProto.__getitem__(int)` → a TensorDict int-index per sample; **`tokenizer.decode` per sample for responses AND prompts** (`654-665`), serial on the driver | seconds (decode-bound) |

### 1b. V1 `PPOTrainer._step_once()` — per step (`v1/trainer_base.py:563-615`)

| # | Op | Where | Data moved through the driver | Driver cost class |
|---|---|---|---|---|
| 1 | `prepare_step` → `_next_train_batch` → `_fetch_one_gen_batch`: dataloader, uuid, `tu.get_tensordict(batch_dict)` | `1467-1495, 1526-1528` | B prompts; `NonTensorStack.from_list` per non-tensor key over B | small (B, not N) |
| 2 | `_submit_batch_to_rollout`: `tq.kv_batch_put(keys, tags)`; `AgentLoopManagerTQ.generate_sequences` (`prompts.chunk(W)` + `.remote` per worker, `ray.get`) | `1497-1513`, `agent_loop_tq.py:248-262` | W pickled TensorDict chunks of B/W NonTensorStacks | small |
| 3 | *worker* agent loops write unpadded rows + tags to TQ, one `kv_batch_put` per prompt group | `agent_loop_tq.py:155-233` | — | — |
| 4 | `replay_buffer.sample`: loop { `tq.kv_list()` (ALL tags of ALL keys in TQ), rebuild 5 sets, evict, select } with `time.sleep(poll_interval)` between polls | `replay_buffer.py:422-515, 201-237` | O(total keys in TQ) Python per poll | ms per poll, but adds up to `poll_interval` latency to every step |
| 5 | `_balance_batch`: `upsample_batch_to_divisible_size`, `torch.tensor(seq_len tags)`, `calculate_workload` → **tensor** → `get_seqlen_balanced_partitions` (same 0-d-tensor KK as legacy), `batch.reorder(list)` (metadata only) | `1609-1633`, `padding_utils.py:153-230` | tags only | Python O(N log N) with tensor scalars |
| 6 | `_compute_old_log_prob`: `compute_log_prob(KVBatchMeta)` (dispatch = `kv_batch_meta2batch_meta` RPC + chunk of metadata; workers pull rows from TQ in `tqbridge`, push `log_probs`/`entropy` for the **full sequence** as nested back to TQ); driver `tq.kv_batch_get(entropy, log_probs, response_mask[, responses, rollout_log_probs])`; `response_from_nested`×2 (**Python loop over N**, `padding.py:166-181`); `tq.kv_batch_put(old_log_probs, entropy)`; `to_padded_tensor()`; `agg_loss`; `calculate_debug_metrics` | `1635-1694` | **down**: 2×N×S_valid fp32 + N×L_valid int64 (+int64+fp32); **up**: 2×N×L_valid fp32; then dense N×L fp32 ×2 (+2) built on the driver | TQ transfer + Python O(N) + densify |
| 7 | `_compute_ref_log_prob` | `1696-1720` | down: N×S_valid fp32 + N×L_valid int64; `response_from_nested` loop; up: N×L_valid fp32 | same pattern |
| 8 | `_compute_values` | `1722-1742` | same pattern | same |
| 9 | `_compute_advantage`: `tq.kv_batch_get(uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values)`; `to_padded_tensor()` (dense N×L × up to 6); `np.array(uid.tolist())`; KL/alias; `compute_advantage_for_multi_trajectories` (Python loop over N keys with `rsplit`, `data.select_idxs(final_indices)` = copy of all dense tensors, GRPO loop); `response_to_nested` per output field (**Python loop over N**, `padding.py:184-199`); `tq.kv_batch_put` | `1744-1803`, `v1/utils.py:149-230` | down: N×L_valid×(8+4×4) B; dense ×6; up: 2-4 × N×L_valid fp32 | TQ + densify + Python O(N) ×3 |
| 10 | `_update_critic`, `_update_actor` | `1805-1871` | metadata only; workers pull from TQ | small |
| 11 | `_compute_metrics`: `tq.kv_batch_get` of 9 fields incl. **prompts and responses (int64)**; `to_padded_tensor()`; `DataProto.select_idxs(non_padding_mask)` (**copy of every dense tensor**); `compute_data_metrics` etc. | `1873-2002` | down: N×(P_valid+L_valid) int64 + N×L_valid×(8+5×4) B; dense ×9 then copied once more | TQ + densify + copy |
| 12 | `tq.kv_clear` | `fit`, `510` | — | — |

So per V1 step the driver performs **~12 serialized TQ round trips** (5 `kv_batch_get`s that each pull N rows, 4 `kv_batch_put`s, plus meta RPCs in every dispatch), densifies the response-length fields **three separate times** (steps 6, 9, 11), and runs **~7 Python loops over N rows** (`response_from_nested`×4, `response_to_nested`×2-4, the GRPO loops, the `rsplit` loop). Rough byte count at N=4096, mean valid P/L = 1k/3k (*guess for the mean lengths; the formula is exact*): ~1.2-1.4 GB pulled to the driver and ~250 MB pushed per step at L=8k; ×3-4 at L=32k. None of this is overlapped with GPU work: the whole chain is serialized between `compute_log_prob` and `update_actor`.

The agent-loop side (not on the driver, but on the per-node agent-loop workers) still pays `tokenizer.pad` ×3 per sample and per-sample `torch.tensor` construction (`agent_loop.py:779-795`, `agent_loop_tq.py:188-199`).

Roadmap note: issue #8186 ("[roadmap] verl 26Q4 roadmap", section "Trainer") lists "Remove V0 trainer `verl/trainer/ppo/ray_trainer.py`". Findings below are therefore ranked by whether they live in V1 or in shared code (`core_algos.py`, `protocol.py`, `seqlen_balancing.py`, `metric_utils.py`, `workers/utils/padding.py`, `utils/tensordict_utils.py`, `utils/transferqueue_utils.py`); V0-only items are listed but marked secondary.

## 2. Findings table

Costs are per training step on this Mac at the stated shape; "per step" extrapolations multiply by the number of calls per step in the current code. Risk column names the CPU tests that pin the semantics.

| # | Path | file:line | What / why slow | Measured | Fix | Saving per step | Risk / tests | Open PRs |
|---|---|---|---|---|---|---|---|---|
| F1 | **shared** (`core_algos`) | `core_algos.py:216-262` `compute_gae_advantage_return`; `:694-732` `compute_reinforce_plus_plus_outcome_advantage` | O(L) Python loop over timesteps, ~6 width-N tensor ops per step; serialized on the driver in both trainers (`ray_trainer.py:235`, `v1/trainer_base.py:1780` via `compute_advantage`) | GAE: 1.30 s (N=4096, L=8k), **5.22 s (L=32k)**; RF++: 0.81 s / 3.46 s | Exact vectorized scan: for γλ=1 (verl default `gamma=1.0, lam=1.0`) the masked recurrence is a reverse `cumsum` over the response axis; for general γλ a blockwise closed-form scan (block of 512-1024 steps, powers bounded so fp32 is safe); keep the mask semantics (values/TD-error skipped on observation tokens) | 1.3-5.2 s for GAE users (every PPO-with-critic run); 0.8-3.5 s for REINFORCE++ | Must match the loop to fp32 rounding; `tests/trainer/ppo/test_core_algos_on_cpu.py::test_multi_turn_compute_gae_advantage_return`, `tests/trainer/ppo/test_reinforce_pp_multiturn_on_cpu.py`; add a randomized loop-vs-scan parity test incl. observation masks | #2132 (open since 2025-06, conv1d reformulation, no mask handling, no tests, stale) |
| F2 | **V1 driver ↔ TQ** | `v1/trainer_base.py:1641-1669, 1713-1718, 1734-1740, 1746-1801, 1876-1894` | Driver pulls full-sequence `log_probs`/`entropy` (prompt+response) from TQ only to slice the response part and push it back; `_compute_advantage` and `_compute_metrics` each re-pull the same response-length fields; prompts/responses (int64) are pulled just to compute lengths/metrics; `select_idxs` copy; 3 separate densifications | **0.8 s (N=1024/8k), 3.5 s (N=4096/8k), 10.1 s (N=4096/32k)** of serialized driver time per step; 5.1 GB pulled to the driver per step at 32k (3.8) | (a) have the worker return response-sliced `log_probs`/`entropy`/`values` (it already has `response_mask`; `response_from_nested` runs there instead) so the driver never pulls full-seq tensors; (b) merge the advantage GET with the metrics GET (one pull of response-length fields, keep the dense tensors for `_compute_metrics`) and derive prompt/response lengths from tags (`tags[i]["prompt_len"/"response_len"]` already exist, `agent_loop_tq.py:200-203`) instead of pulling `prompts`/`responses`; (c) stop pulling `entropy` to the driver: compute the `agg_loss` entropy metric on the worker | ~50-60 % of the TQ bytes per step (the full-seq GETs ≈ 110 MB of 334 MB; the metrics GET ≈ 120 MB) ⇒ ~0.4-0.5 s at N=1024, ~2 s at N=4096 (see 3.8) | Semantics unchanged (same tensors, different place); tests: `tests/trainer/ppo/v1/*` (trainer_base CPU tests), `tests/utils/test_padding_on_cpu.py::test_response_from_nested`; needs an e2e V1 run on GPU for proof | #7415 (DRAFT pluggable rollout data backends), #8201 (V1 mini-batch by UID; touches `_compute_advantage`/replay), #7000 (remove TQ, NeoProto) — data plane is in flux |
| F3 | **V1 sampler** | `v1/replay_buffer.py:144, 408-409, 422-515` | `time.sleep(poll_interval=2.0)` between `kv_list()` polls; not configurable from `ppo_trainer.yaml` (`_build_replay_buffer` never passes it, `trainer_base.py:155-162`); in sync mode every step waits a mean ~1 s (max 2 s) after the last trajectory lands | poll body ≤ 2 ms at 12k keys, 17.5 ms at 87k keys (3.7) → the sleep is 99 % of the loop | Expose `poll_interval` (default 0.2 s) or, better, make the controller push a "sampleable" notification (TQ already has production-status checks: `client.py:743-823`) | 0.5-1.8 s per sync step, free | Zero numerical risk; `tests/trainer/ppo/v1/` replay-buffer tests | none found |
| F4 | shared (`seqlen_balancing`) + both trainers | `seqlen_balancing.py:49-172, 213-254`; callers `ray_trainer.py:1227-1237`, `v1/trainer_base.py:1623-1625` | KK partitioner and per-partition sort are fed a torch tensor; every comparison is a 0-d tensor op | 90 ms + 22 ms at N=4096 (vs 5 + 0.4 ms with lists) | convert once with `.tolist()` | ~0.1 s | partitions identical (asserted) | **#7281 does exactly this (open, 2026-08); #8137 caches `State.spread`** → duplicate, skip |
| F5 | shared (`core_algos`) | `core_algos.py:268-331` GRPO loop, `:588-636` RLOO, `:640-690` OPO, `:771-830` GPG | O(N) Python loops with 0-d tensor math | GRPO 19-27 ms, RLOO 30-38 ms, OPO 21-29 ms at N=4096 | route `GRPO` → the existing vectorized kernel (`:335`) once #8349's uniform-group fix lands (vectorized version uses `group_mean_std` with `std` floor `eps=0`, so bit-identical here) | 15-25 ms | `test_grpo_and_vectorized_equivalence`, `test_rloo_and_vectorized_equivalence` exist | #8349, #4598 (numerics), #7150 (dense ids) — crowded, low value |
| F6 | shared (`padding.py`) | `padding.py:52-62` `left_right_2_no_padding` position_ids loop | Python loop over N rows with a boolean-mask gather per row, then `as_nested_tensor` copy | 213 ms (N=4096, L=8k), 922 ms (L=32k) of the 245 / 835 ms call; vectorized 22 / 75 ms (identical, asserted) | `position_ids[attention_mask.bool()]` + `nested_tensor_from_jagged(..., offsets=cu_seqlens)`; 3-D (mrope) position ids: `position_ids.transpose(0,1)[:, mask]` then `_ragged_idx=2` | V0 driver: ×3-5 calls ⇒ 0.7-1.2 s (8k) / 2.5-4 s (32k) per step; V1: 0 (rows are already nested) | `tests/utils/test_padding_on_cpu.py::test_padding_conversion_*`, `test_padding_roundtrip`; `tests/models/test_fsdp_no_padding_on_gpu.py` | #8413 (position-id packing fix), #7901, #5886 touch this function — coordinate |
| F7 | V0 driver | `protocol.py:1099-1123` `to_tensordict()`; `tensordict_utils.py:377-455` `get_tensordict()` | `NonTensorStack.from_list([NonTensorData(x) ...])` per non-tensor key per call, ~5 µs/item; called 3-5× per step on the same batch | 0.52 s per call at N=4096 with 17 keys (independent of L) ⇒ 1.5-2.6 s per step | convert once per step and cache the TensorDict (or keep non-tensor columns as a single `NonTensorData(np.ndarray)`), or drop the non-tensor columns the workers never read before dispatch | 1-2 s per V0 step | `tests/test_protocol_on_cpu.py::test_to_tensordict*`, `test_to_tensordict_agent_loop_scenario` | none; but V0 is scheduled for removal (#8186) |
| F8 | V0 driver dispatch | `protocol.py:1268-1275` + `decorator.py:208-246`; callers `ray_trainer.py:1290-1376` | The same ~2-10 GB padded batch is chunked, consolidated (1 copy) and pickled into plasma (1 copy) 3-5 times per step (old_log_prob, ref, values, update_critic, update_actor) | consolidate 0.39-0.44 s + pickle 0.26-0.73 s per dispatch at N=4096 ⇒ 2-6 s per step (+ plasma copies, not measured) | ship once per step (`ray.put` the consolidated chunks once; later calls pass the ObjectRefs and only the new small columns) | 1.5-4.5 s per V0 step | dispatch semantics; `tests/single_controller/*` | #3168 (DistDataProto, WIP), #5958 (transport timing) |
| F9 | V0 agent loop (worker CPU) | `agent_loop.py:728-757, 779-795` | `tokenizer.pad(...)` ×3 per sample to build fixed-width tensors; runs inside the worker's asyncio loop | 2.9 ms/sample (L=8k), 10.1 ms/sample (L=32k) vs 0.01 ms with `torch.full` + slice ⇒ 12-41 s of CPU per 4096 samples, spread over W agent-loop workers | replace with `torch.full` + slice assignment (identical tensors incl. pad id and attention mask) | 1.5-5 s of per-worker CPU per step (partly overlapped with generation; the tail is exposed) | `tests/experimental/agent_loop/*` | #5506 touches `AgentLoopWorker` tensor sizes |
| F10 | shared (`metric_utils`) | `metric_utils.py:429-610` | 3 `masked_select`s and ~25 `.item()`s over N×L; `_compute_response_info` computed twice (also in `compute_timing_metrics`) | 74 ms (8k) / 317 ms (32k) with critic at N=4096 | fuse the masked reductions (one `masked_select` of advantages/returns/values, or sum/max/min via masked ops) | 40-150 ms | `tests/trainer/ppo/test_metric_utils_on_cpu.py` (38 tests) | #8303, #7906 add metrics here |
| F11 | both validation paths | `ray_trainer.py:654-665`; `v1/trainer_base.py:1105-1108` | serial `tokenizer.decode` per row of padded ids (prompts and responses) on the driver | 0.9 ms/row (3.6k tokens) → 0.9 s per 1024 rows; 3.6 ms/row (14k tokens) → 3.7 s per 1024 rows; `batch_decode` gives no speed-up | decode in a thread pool (fast tokenizers release the GIL) or on the agent-loop workers (they already have the text when `validate=True`); V1 already decodes only `final_indices` for metrics but decodes **all** rows for the dump | 1-4 s per validation batch | text equality; `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py` | #8303, #8104, #7240 touch `_validate`/validation metrics |
| F12 | V0 driver | `protocol.py:188-200` `union_numpy_dict` deep-equal | Python element-wise compare of overlapping object columns on every `batch.union(gen_batch_output)` | 9 ms at N=4096 with small dicts (grows with `extra_info` size) | skip the check when `a is b` / identical buffers; or compare `id()`s first | ~10 ms | `tests/test_protocol_on_cpu.py::test_union_numpy_dict` | — |

Not findings (checked, cheap): `DataProto.concat` of agent-loop chunks (57-68 ms), `DataProto.__setstate__` (90 ms for 2 GB), `select_idxs` (25-81 ms), replay-buffer dict rebuild (≤ 18 ms), `response_from_nested`/`response_to_nested` loops (≤ 25 ms each; vectorizing does not help), `to_padded_tensor` (≤ 11 ms per field), `compute_timing_metrics` (≤ 7 ms), `MetricsAggregator` (dict ops over ~100 keys).

## 3. Benchmark scripts and raw numbers

Scripts: `<repo>/research/bench-a3/` (`bench_common.py` builds the synthetic batches; one `bench_*.py` per area; every `*.out` file is the verbatim run log). Venv: `<repo>/.venv-a3` (python 3.12.14, torch 2.15.0 CPU, **tensordict 0.10.0 = the project lock**, numpy 2.5.4, TransferQueue 0.1.11, ray 2.55). verl is imported from the clone through `PYTHONPATH` (nothing installed into or written to the clone). Machine: Apple M5 Max, 18 cores (torch uses 6 threads), 128 GB. All timings are medians of 2-3 runs after a warm-up, wall clock.

Synthetic batch (`bench_common.make_padded_batch`): per-row prompt length ~ N(P/2, P/4) clipped to [16, P] (mean ≈ 1.0k for P=2048); response length lognormal around L/3 with 5 % of rows truncated at L (mean ≈ 3.5k for L=8k, ≈ 14k for L=32k); left-padded prompts, right-padded responses, 8 rollouts per uid, binary outcome reward on the last token. The non-tensor batch mirrors the 17 keys the agent loop echoes (uid, data_source, reward_model, extra_info, raw_prompt, index, tools_kwargs, interaction_kwargs, agent_name, priority, `__num_turns__`, acc, turn_scores, tool_rewards, min/max_global_steps, extras).

### 3.1 Advantage estimators (`bench_adv.out`) — shared `core_algos.py`

| function | N=1024, L=8k | N=4096, L=8k | N=4096, L=32k |
|---|---|---|---|
| `compute_grpo_outcome_advantage` loop (:268) | 5.1 ms | 19.0 ms | 27.2 ms |
| `compute_grpo_vectorized_outcome_advantage` (:335) | 1.5 ms | 4.8 ms | 13.2 ms |
| `compute_rloo_outcome_advantage` loop (:588) | 7.8 ms | 29.7 ms | 37.5 ms |
| `compute_rloo_vectorized_outcome_advantage` (:834) | 1.3 ms | 4.7 ms | 13.0 ms |
| `compute_opo_outcome_advantage` loop (:640) | 4.8 ms | 21.5 ms | 29.4 ms |
| **`compute_gae_advantage_return` loop over L (:216)** | **387 ms** | **1302 ms** | **5218 ms** |
| **`compute_reinforce_plus_plus_outcome_advantage` loop over L (:694)** | **250 ms** | **815 ms** | **3456 ms** |

Loop vs vectorized GRPO: max abs diff 0.0 on these inputs; RLOO 6e-8. The group loops are O(N) Python with 0-d tensor arithmetic (~5 µs per sample) — cheap. The two **time loops** are O(L) Python with ~6 tensor ops of width N per step (~25-40 µs per step): 32k steps ⇒ 3.5-5.2 s on the driver, serialized, every step, for every PPO-with-critic (GAE) or REINFORCE++ run.

**Prototype of the fix (`bench_gae_vec.py`, `bench_gae_vec.out`)** — an exact blockwise linear-recurrence scan (`_linear_scan_reverse`) with the same observation-token carry semantics, float64 math, unoptimized:

| | loop (same run) | scan, γλ = 1 (verl default `gamma: 1.0, lam: 1.0`, `ppo_trainer.yaml:72-75`) | scan, γλ = 0.95 |
|---|---|---|---|
| GAE N=4096, L=8k | 3538 ms (1302 ms in `bench_adv`, different masks) | 132 ms | 515 ms |
| GAE N=4096, L=32k | 5218 ms (`bench_adv`) | 1052 ms | 2123 ms |
| REINFORCE++ N=4096, L=8k | 2092 ms (815 ms in `bench_adv`) | 116 ms | 658 ms (γ=0.99) |
| REINFORCE++ N=4096, L=32k | 3456 ms (`bench_adv`) | 377 ms | 1902 ms (γ=0.99) |

Parity against the loops on random inputs with multi-turn observation spans, N up to 512 and L up to 8192, γ,λ ∈ {(1,1),(1,.95),(.99,.95),(.9,.9),(.5,.5)}: max |Δadvantage| on valid tokens 3.6e-7 … 1.1e-5, max |Δreturns| ≤ 5.7e-4 at |returns| ≈ 254 (≈ 2e-6 relative) — fp32 rounding level, not bitwise. The masks used here have ~37 % masked tokens; an fp32 in-place version with fewer passes should be ~2× faster again (*guess*).

### 3.2 Batch balancing (`bench_balance.out`) — shared `seqlen_balancing.py`, callers in both trainers

| op | N=1024, dp=8 | N=4096, dp=32 | N=4096, dp=128 | N=4096, L=32k, dp=32 |
|---|---|---|---|---|
| `attention_mask.sum(-1)` N×S | 0.5 ms | 1.7 ms | 1.8 ms | 5.6 ms |
| `get_seqlen_balanced_partitions(workload **tensor**)` — as both trainers call it (`ray_trainer.py:1227`, `v1/trainer_base.py:1625`) | 20.2 ms | 90.1 ms | 78.1 ms | 80.7 ms |
| same with a Python list | 1.4 ms | 5.0 ms | 4.2 ms | 5.8 ms |
| per-partition `sort(key=lambda x:(workload_lst[x],x))` with tensor indexing (`ray_trainer.py:1234-1237`) | 6.5 ms | 22.1 ms | 16.8 ms | 23.2 ms |
| same with list indexing | 0.1 ms | 0.4 ms | 0.4 ms | 0.5 ms |
| `log_seqlen_unbalance` | 0.1 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| `DataProto.reorder(global_idx)` full copy (V0 only) | 8-22 ms | 32-61 ms | 27 ms | 105-163 ms |

The partitions are identical for tensor and list input (asserted). The 16-18× slowdown is purely 0-d-tensor comparisons inside `sorted()` / `heapq` / `State.__lt__` — **open PR #7281 already fixes exactly this** (converts the workload to a list once), and #8137 caches `State.spread`. ~100 ms per step; not a contribution target.

### 3.3 Padding ⇄ nested conversions (`bench_padding.out`) — shared `workers/utils/padding.py`

| op | N=1024, L=8k | N=4096, L=8k | N=4096, L=32k |
|---|---|---|---|
| **`left_right_2_no_padding` full batch (:23)** (V0: 3-5× per step; also any TensorDict-dispatch caller) | 60.7 ms | 245 ms | 835 ms |
| .. of which `unpad_input` | 10.0 ms | 38.7 ms | 124 ms |
| .. of which the `position_ids` Python loop over N rows (:52-62) | 51.5 ms | 213 ms | 922 ms* |
| .. vectorized position_ids (`position_ids[attention_mask.bool()]` + `nested_tensor_from_jagged(cu_seqlens)`), identical values/offsets (asserted) | 5.9 ms | 22.2 ms | 75.0 ms |
| `no_padding_2_padding` loop (:103) | 7.8 ms | 31.0 ms | 46.7 ms |
| .. vectorized gather (identical output) | 4.4 ms | 16.0 ms | 60.1 ms |
| `response_from_nested` loop (:166) (V1: 4× per step) | 5.5 ms | 22.2 ms | 25.1 ms |
| .. vectorized | 5.3 ms | 18.1 ms | 64.4 ms |
| `NestedTensor.to_padded_tensor` N×L fp32 (V1: 3× per step, several fields) | 1.6 ms | 5.9 ms | 10.5 ms |
| `response_to_nested` loop (:184) (V1: 2-4× per step) | 2.8 ms | 10.7 ms | 13.8 ms |
| .. vectorized | 5.0 ms | 19.6 ms | 78.2 ms |

(*measured in isolation; inside the full call the loop and unpad overlap in the 835 ms total.) Conclusion: the V1 per-row loops (`response_from_nested`, `response_to_nested`) cost ≤ 25 ms each and my vectorized versions are **not** faster at long L (the loop body is a cheap view slice; the cost is the final copy) — not worth touching. The one real win in this file is the `position_ids` loop in `left_right_2_no_padding`: 10× faster vectorized, 0.2-0.9 s saved per call, but it is only on the V0 driver path and on callers that dispatch padded TensorDicts.

### 3.4 DataProto / TensorDict (`bench_dataproto.out`) — shared `protocol.py`, `tensordict_utils.py`

| op | N=1024, L=8k (0.54 GB) | N=4096, L=8k (2.15 GB) | N=4096, L=32k (7.78 GB) |
|---|---|---|---|
| `DataProto.to_tensordict()` with 17 non-tensor keys (:1099) — V0: 3-5× per step | 87.7 ms | 522 ms | 538 ms |
| same with 1 non-tensor key | 5.0 ms | 19.5 ms | 20.1 ms |
| `NonTensorStack.from_list([NonTensorData(x) for x in val])`, one key | 5.0 ms | 21.0 ms | 22.2 ms |
| `tu.get_tensordict(dict with N-length object arrays)` (:377) | 90.3 ms | 585 ms | 535 ms |
| `union_numpy_dict` deep-equal, 4 overlapping object keys (:188) | 1.0 ms | 9.1 ms | 4.6 ms |
| `DataProto.concat` of 8 agent-loop chunks (:914) | 6.6 ms | 56.5 ms | 68.4 ms |
| `BatchData(td).chunk(dp)` = `chunk_tensordict` + `contiguous().consolidate()` per DP chunk (:1268) | 78.7 ms (dp=8) | 388 ms (dp=32) | 440 ms (dp=32) |
| `pickle.dumps` one consolidated chunk (ray.put side) | 5.4 ms (0.07 GB) | 7.8 ms (0.07 GB) | 9.3 ms (0.24 GB) |
| `pickle.loads` one consolidated chunk (worker side) | 15.8 ms | 19.2 ms | 22.6 ms |
| `pickle.dumps` all dp chunks, serial | 44 ms | 262 ms | 726 ms |
| `pickle.dumps` a **non**-consolidated chunk view (pickles the whole parent storage) | 55 ms | 268 ms | 1169 ms |
| `pickle.dumps(DataProto)` via `__getstate__` (consolidate + `torch.save` + `getvalue`) (:377) | 243 ms | 800-1158 ms | (skipped, >5 GB) |
| `pickle.loads(DataProto)` via `__setstate__` (`torch.load`) | 18.3 ms | 89.6 ms | — |
| `DataProto.select_idxs(bool mask)` full copy (:632) — V1 `_compute_metrics` | 5.5 ms | 25.3 ms | 81.1 ms |
| `pickle.dumps(NonTensorStack of N raw_prompt dicts)` | 1.6 ms | 9.0 ms | 8.4 ms |
| `pickle.dumps(np object array of the same dicts)` | 0.3 ms | 1.0 ms | 1.1 ms |

Reading: NonTensorData construction is ~5 µs per item per key, so `to_tensordict()`/`get_tensordict()` cost scales with (#non-tensor keys × N): 0.5 s per call at N=4096 with the agent loop's 17 keys, **independent of L**. The consolidate+pickle dispatch path moves bytes at 5-17 GB/s on this machine (one full copy for consolidate, one for pickle); it is bandwidth-bound, not Python-bound.

### 3.5 Metrics (`bench_metrics.out`) — shared `metric_utils.py`

| op | N=1024, L=8k | N=4096, L=8k | N=4096, L=32k |
|---|---|---|---|
| `compute_data_metrics(use_critic=True)` (:429) | 18.4 ms | 73.8 ms | 317 ms |
| `compute_data_metrics(use_critic=False)` | 12.1 ms | 48.8 ms | 187 ms |
| `compute_timing_metrics` (:611) (recomputes `_compute_response_info`) | 0.9 ms | 2.5 ms | 6.6 ms |
| `agg_loss(entropys, response_mask)` driver entropy metric | 2.6 ms | 9.8 ms | 32.9 ms |
| `[item ... for item in DataProto]` over 1024 items (V0 `_validate`, `ray_trainer.py:619`) | 9.7 ms | 9.3 ms | 9.5 ms |
| same via `non_tensor_batch["reward_model"]` directly | 0.1 ms | 0.1 ms | 0.1 ms |

### 3.6 Tokenizer (`bench_tokenizer.out`) — Qwen2.5 tokenizer (vocab 151,665), N=1024 rows

| op | mean response 3.6k tokens (L=8k) | mean response 14.3k tokens (L=32k) |
|---|---|---|
| validation `tokenizer.decode(padded row, skip_special_tokens=True)`, responses (`ray_trainer.py:655`, `v1/trainer_base.py:1107`) | 0.91 ms/row → **0.9 s per 1024 rows** | 3.64 ms/row → **3.7 s per 1024 rows** |
| same for prompts | 0.24 ms/row → 0.2 s | 0.25 ms/row → 0.3 s |
| decode valid tokens only (list) | 0.80 ms/row | 3.12 ms/row |
| `batch_decode` (Rust) on valid tokens | 0.76 ms/row (no parallel speed-up) | 2.92 ms/row |
| **V0 agent loop `tokenizer.pad` ×3 per sample** (`agent_loop.py:779-795`) | **2.92 ms/sample → 3.0 s of CPU per 1024 samples** | **10.1 ms/sample → 10.3 s per 1024 samples** |
| same padding with `torch.full` + slice assignment | 0.01 ms/sample | 0.01 ms/sample |

### 3.7 V1 replay buffer bookkeeping, simulated (`bench_replay.out`)

| TQ contents | `_sync_metadata_from_transfer_queue` rebuild (`replay_buffer.py:201`) | `_materialize_batch` (:394) | `kv_list` payload (pickled) |
|---|---|---|---|
| 512 prompts × 8 (2,976 keys) | 0.5 ms | 0.4 ms | 0.2 MB |
| 2048 prompts × 8 (11,880 keys) | 1.9 ms | 1.0 ms | 0.9 MB |
| 8192 prompts × 16 (86,848 keys) | 17.5 ms | 7.3 ms | 6.6 MB |

Negligible per poll. The cost of the sampler is its **`time.sleep(poll_interval)`** between polls (`replay_buffer.py:409`; default `poll_interval=2.0` s at :144, not exposed in `ppo_trainer.yaml`): in sync mode every step ends with up to 2 s (mean ~1 s) of idle wait after the last trajectory lands.

### 3.8 Live TransferQueue on this Mac (`bench_tq_*.out`) — the V1 driver's per-step traffic

Local Ray + `tq.init()` with the default SimpleStorage backend (2 storage units, ZMQ over loopback, msgpack zero-copy frames). Rows are unpadded nested tensors exactly as `agent_loop_tq.py:188-233` writes them; field sets are the ones `trainer_base.py` requests. Single node = lower bound for a multi-node deployment.

N=1024 (mean prompt 1.0k, mean response 3.5k; rollout payload 0.23 GB):

| op (driver unless noted) | time | bytes | rate |
|---|---|---|---|
| agent loop: `list_of_dict_to_tensordict` ×128 groups of 8 (worker side) | 266 ms | — | — |
| agent loop: `kv_batch_put` ×128 groups (worker side) | 782 ms, 6.1 ms/group | 230 MB | 297 MB/s |
| `kv_list()` 1,152 keys | 2 ms | — | — |
| `kv_retrieve_meta(1024 keys)` (every dispatch) | 8 ms | — | — |
| worker-style PUT log_probs+entropy full-seq (all DP ranks together) | 65 ms | 37 MB | 567 MB/s |
| GET old_log_prob fields [entropy, log_probs, response_mask] | 51 ms | 65 MB | 1268 MB/s |
| PUT old_log_probs+entropy | 62 ms | 28 MB | 460 MB/s |
| GET ref fields [log_probs, response_mask] | 47 ms | 47 MB | 988 MB/s |
| PUT ref_log_prob | 46 ms | 14 MB | 308 MB/s |
| PUT values | 120 ms | 14 MB | 119 MB/s |
| GET advantage fields (7) | 109 ms | 100 MB | 915 MB/s |
| `to_padded_tensor` of those 7 fields | 38 ms | — | — |
| PUT advantages+returns | 69 ms | 28 MB | 415 MB/s |
| GET metrics fields (8, incl. prompts/responses int64 + `num_turns` NonTensorStack) | 269 ms | 122 MB | 454 MB/s |
| `kv_clear(1024 keys)` | 15 ms | — | — |

Per-step driver total at N=1024/L=8k: **≈ 0.48 s of GETs (334 MB) + ≈ 0.30 s of PUTs (84 MB) + ~10 ms of meta RPCs ≈ 0.8 s**, serialized on the driver between `compute_log_prob` and `update_actor`.

N=4096 (`bench_tq_4096_8192_rerun.out`, `bench_tq_4096_32768.out`; the first 4096/8k run, `bench_tq_4096_8192.out`, was contended by the previous Ray instance shutting down — `tq.init` took 59 s and every rate was 3-5× lower — and is discarded):

| op | L=8k (mean resp 3.5k) | L=32k (mean resp 14.0k) |
|---|---|---|
| agent loop (worker side): `list_of_dict_to_tensordict` ×512 groups | 1.11 s | 7.05 s |
| agent loop (worker side): `kv_batch_put` ×512 groups of 8 | 5.18 s (10.1 ms/group, 0.93 GB, 181 MB/s) | 11.66 s (22.8 ms/group, 3.34 GB, 286 MB/s) |
| `kv_list()` 4,608 keys | 9 ms | 30 ms |
| `kv_retrieve_meta(4096)` per dispatch | 21 ms | 23 ms |
| worker PUT log_probs+entropy full-seq (all ranks) | 336 ms (148 MB) | 842 ms (491 MB) |
| **GET old_log_prob fields** | **363 ms (262 MB)** | **921 ms (950 MB)** |
| PUT old_log_probs+entropy | 416 ms (115 MB) | 899 ms (458 MB) |
| `to_padded_tensor` full-seq log_probs | 18 ms | 47 ms |
| **GET ref fields** | **183 ms (188 MB)** | **464 ms (704 MB)** |
| PUT ref_log_prob | 323 ms (57 MB) | 1008 ms (229 MB) |
| PUT values | 221 ms (57 MB) | 452 ms (229 MB) |
| **GET advantage fields (7)** | **628 ms (401 MB)** | **2298 ms (1604 MB)** |
| `to_padded_tensor` of the 7 fields | 146 ms | 581 ms |
| PUT advantages+returns | 398 ms (115 MB) | 939 ms (458 MB) |
| **GET metrics fields (8)** | **646 ms (491 MB)** | **2334 ms (1866 MB)** |
| `kv_clear(4096)` | 114 ms | 120 ms |
| **driver total per step (GET+PUT+densify+meta+clear)** | **≈ 3.5 s (1.34 GB in / 0.34 GB out)** | **≈ 10.1 s (5.1 GB in / 1.4 GB out)** |

Rates sit at 0.2-1.5 GB/s with a large per-row fixed cost: a 2-field GET of 4096 rows costs ~180 ms at 188 MB and ~460 ms at 704 MB, i.e. ≈ 25-40 µs per row per field of fixed overhead (unbind + per-frame msgpack + `_pack_field_values` → `as_nested_tensor` on 4096 views; the packing alone is 10-24 ms per field, measured in `bench_tq_4096_8192_rerun.out` tail). The driver pulls **5.1 GB per step at 32k responses**, 73 % of it in two GETs that exist only to slice/aggregate (old_log_prob + metrics) and 31 % of it bytes the driver already had or never uses (full-sequence prompt positions of `log_probs`/`entropy`; `prompts`/`responses` token ids pulled only to compute lengths that are already in the tags).

## 4. Top 5 contribution candidates from this slice

Ranked by (measured saving per step) × (lives in V1 or shared code) × (low duplicate risk).

### C1. Exact vectorized GAE and REINFORCE++ time scans (shared `core_algos.py`) — 1.2-5 s per step, every PPO-with-critic / RF++ run

Replace the `for t in reversed(range(gen_len))` loops in `compute_gae_advantage_return` (`core_algos.py:216-262`) and `compute_reinforce_plus_plus_outcome_advantage` (`:694-732`) with the blockwise linear-recurrence scan prototyped in `bench_gae_vec.py`: `nextvalues` via a reverse `cummin` of valid indices + `gather`, `delta` in one shot, then `g_t = q^{m_t} g_{t+1} + b_t` solved per 256-position block in closed form (`q^{-S_t} · revcumsum(b_j q^{S_j})`, S = in-block exclusive prefix count of valid tokens) with an L/256-step cross-block carry; `q = γλ = 1` (the default) collapses to one reverse `cumsum`. Same semantics as the loops, including the carry through observation tokens and the whitening; measured max |Δ| ≤ 1e-5 (fp32 rounding), 7-27× faster at γλ=1, 2.5-7× at γλ<1, in an unoptimized float64 prototype. Production version: fp32 path when `q ≥ 0.9` (exponents ≤ 256 keep `q^{±S}` inside fp32 range), float64 otherwise; keep the loop as `compute_gae_advantage_return_reference` for the parity test; add a randomized parity test with observation masks (extends `test_multi_turn_compute_gae_advantage_return` and `test_reinforce_pp_multiturn_on_cpu.py`). Open PR #2132 (conv1d GAE, June 2025) is stale, O(L²)-ish, ignores masks and has no tests — ours supersedes it; cite it. The driver-side `compute_advantage` call sites in both trainers need no change.

### C2. Stop pulling the batch to the V1 driver three times per step (`v1/trainer_base.py` + `engine_workers.py`) — 2-6 s per step at N=4096

Today the V1 driver GETs 1.3 GB (8k) / 5.1 GB (32k) per step from TransferQueue and PUTs 0.3 / 1.4 GB, 3.5 s / 10.1 s serialized on a single node (3.8). Three changes, each independently testable: (a) **slice on the worker**: `compute_log_prob` / `compute_ref_log_prob` / `infer_batch` return `log_probs`/`entropy`/`values` already restricted to the response window (`response_from_nested` moved into `engine_workers.py` next to `tu.get_tensordict(model_output)` at `:271`; the worker has `response_mask`), writing `old_log_probs` / `ref_log_prob` / `values` directly under the final field names via the `tqbridge` put — this deletes the driver's full-sequence GET + re-PUT in `_compute_old_log_prob` (`:1654-1665`), `_compute_ref_log_prob` (`:1713-1718`) and `_compute_values` (`:1734-1740`), i.e. 3 GETs + 3 PUTs per step (≈ 1.9 s of the 3.5 s at 8k; ≈ 4.2 s of 10.1 s at 32k); the entropy metric becomes a worker-side `agg_loss` returned in `metrics`. (b) **one driver pull per step**: `_compute_advantage` already fetches `response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values`; keep the dense tensors it builds and hand them to `_compute_metrics`, which then needs only `advantages/returns` (local) and the lengths — and lengths are already in the tags (`prompt_len`, `response_len`, `agent_loop_tq.py:200-203`), so the `prompts`/`responses` int64 pull (the largest single GET, 0.65 s / 2.3 s) disappears. (c) **poll interval**: expose `trainer.v1.sampler.poll_interval` (default 0.2 s) through `_build_replay_buffer` (`:155-162`) — up to 1.8 s per sync step for free. Risk: pure data-movement refactor, semantics unchanged; CPU tests in `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`, `test_replay_buffer_on_cpu.py`, `tests/utils/test_padding_on_cpu.py`; needs one e2e V1 GRPO run on GPU with metric parity (entropy, advantages) as proof. Coordinate with #8201 (mini-batch by UID; touches `_compute_advantage`) and the DRAFT #7415 (pluggable data backends); #7000/NeoProto would change the substrate but not the "pull once, slice at the source" principle.

### C3. Agent-loop padding without `tokenizer.pad` (`agent_loop.py:728-795`) — 12-41 s of worker CPU per 4096 samples (V0 agent loop; also any recipe using `AgentLoopWorker._agent_loop_postprocess`)

`_pad_token_ids` calls `self.tokenizer.pad(...)` three times per sample (prompt left-pad, response right-pad, response_mask right-pad) at 2.9 ms (8k) / 10.1 ms (32k) per sample versus 0.01 ms for `torch.full((1, max_len), pad_id)` + a slice assignment producing identical `input_ids` and `attention_mask`. It also flips `tokenizer.padding_side` as a global side effect inside an asyncio worker. The replacement is ~15 lines, keeps the `(1, max_length)` contract and the empty-input branch, and removes the tokenizer from the hot path entirely. Tests: `tests/experimental/agent_loop/*` plus a direct equality test against `tokenizer.pad` for left/right padding and empty tokens. Secondary because V1's `AgentLoopWorkerTQ` overrides the postprocess, but V0 recipes and `trainer.use_v1=false` users still run it on every sample.

### C4. Vectorize the `position_ids` loop in `left_right_2_no_padding` (shared `workers/utils/padding.py:52-62`) — 0.2-0.9 s per call

`position_ids[attention_mask.bool()]` + `torch.nested.nested_tensor_from_jagged(..., offsets=cu_seqlens)` gives identical values/offsets (asserted) in 22 ms vs 213 ms (N=4096, L=8k) and 75 ms vs 922 ms (32k); the 3-D mROPE case is `position_ids.transpose(0,1)[:, mask]` reshaped with `_ragged_idx=2`. Only on the V0 driver path today (×3-5 per step ⇒ 0.7-4 s), and on any caller that dispatches padded TensorDicts (SFT trainers via `tu.get_tensordict`); small, self-contained, covered by `tests/utils/test_padding_on_cpu.py::test_padding_conversion_*`/`test_padding_roundtrip`. Coordinate with #8413/#7901 which touch the same lines.

### C5. Validation decode off the driver's critical path (`ray_trainer.py:654-665`, `v1/trainer_base.py:1105-1108`) — 1-4 s per validation batch

Both validators decode every prompt and every response row serially with `tokenizer.decode` on padded ids (0.9 ms/row at 3.6k tokens, 3.6 ms/row at 14k; `batch_decode` does not help). Options in increasing ambition: a `ThreadPoolExecutor` over rows (the Rust fast tokenizer releases the GIL — to be measured), decoding only `final_indices` rows in V1 unless `validation_data_dir` is set (the dump is the only consumer of the others, `trainer_base.py:1098-1110`), or returning the decoded text from the agent loop when `validate=True` (it already has the strings). Zero numerical risk; `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py`. Overlaps with #8303/#8104/#7240 only in surrounding lines.

Deliberately **not** proposed: F4 (KK with lists) — PR #7281 already does it; F5 (GRPO loop → vectorized) — ≤ 25 ms and crowded by #8349/#4598/#7150; F7/F8 (V0 `to_tensordict` ×5 and 5× consolidate+pickle per step, 3-8 s per step) — large but V0 is scheduled for removal (#8186) and #3168 covers the design space; F12 (`union_numpy_dict`) — 9 ms.
