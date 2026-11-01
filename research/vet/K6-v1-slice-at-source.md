# K6 — v1-slice-at-source (adversarial vet)

Vetter pass 2026-10-29. Clone `verl` @ `2ab35292` (read-only); PR/issue dumps in `gh/` (2026-10-28) plus
read-only `gh` reads; TransferQueue 0.1.11 from `.venv-a3`. Derivation script:
`research/vet/K6/bytes_baseline_a.py` (pure arithmetic, run with `python3 -I`). Measured step anatomy from
`bench/results/baseline-a-main-1` (main@2ab35292, 8×H100, Qwen3-1.7B, GRPO n=8, 128 prompts, 1024/2048).

## 1. Claim

Source: `research/03-trainer-datapath-audit.md` findings F2 + F3 (§2) and candidate C2 (§4). Per V1 sync step
the driver `kv_batch_get`s full-sequence `log_probs`/`entropy` (prompt+response positions) from TransferQueue
only to slice the response window with `response_from_nested` and `kv_batch_put` it back under a new name
(`old_log_probs`, `ref_log_prob`, `values`); `_compute_advantage` and `_compute_metrics` then re-pull the same
response-length fields, and `_compute_metrics` pulls `prompts`/`responses` (int64) only to compute lengths that
are already in the row tags. Measured on a live local TQ: 0.8 s (N=1024, 8k), 3.5 s (N=4096, 8k), 10.1 s
(N=4096, 32k; 5.1 GB in / 1.4 GB out), serialized between `compute_log_prob` and `update_actor`. Separately the
replay buffer's `poll_interval=2.0` is not exposed, idling a mean ~1 s per sync step. Proposal: (a) workers
return response-sliced tensors through `tqbridge`, (b) one driver pull per step with lengths from tags,
(c) expose `poll_interval`.

## 2. Mechanism verification (file:line @ 2ab35292)

### 2.1 The driver sites exist exactly as described

- `_compute_old_log_prob` `verl/trainer/ppo/v1/trainer_base.py:1635-1694`: bypass branch :1643-1649 (renames
  `rollout_log_probs` without a forward); otherwise `compute_log_prob(batch)` :1659, GET fields
  `["entropy", "log_probs", "response_mask"]` + `["responses", "rollout_log_probs"]` when
  `rollout.calculate_log_probs` (default `True`, `rollout.yaml:245`) :1662-1665; `response_from_nested` ×2
  :1668-1669; PUT `old_log_probs`+`entropy` :1670-1672; densify :1674; `actor/entropy` via `agg_loss`
  :1678-1688 (needs `entropy` on the driver); `calculate_debug_metrics(data)` :1691-1692 (reads `rollout_log_probs`,
  `old_log_probs`, `response_mask` — `verl/utils/debug/metrics.py:25-31`; `responses` is pulled but unused there — so for
  the default user this GET cannot vanish, only shrink from 5 to 3 fields).
- `_compute_ref_log_prob` :1696-1720: GET `["log_probs", "response_mask"]` :1714-1716, slice :1717, PUT :1718.
- `_compute_values` :1722-1742: GET `["values", "response_mask"]` :1736-1738, slice :1739, PUT :1740.
- `_compute_advantage` :1744-1803: GET 7 fields :1746-1747, densify :1750, PUT advantages/returns :1801.
- `_compute_metrics` :1873-: GET 9 fields incl. `prompts`/`responses` :1876-1896; lengths from
  `data["prompts"].offsets().diff()` :1899-1900 and `global_token_num` :1901; `_compute_response_info`
  (`verl/trainer/ppo/metric_utils.py:60`) short-circuits when `prompt_length`/`response_length` are in the batch
  (:75), which `_compute_metrics` sets at :1928-1929 → the int64 token ids are pulled for lengths only. The tags
  already carry `prompt_len`, `response_len`, `seq_len` (`verl/trainer/ppo/v1/agent_loop_tq.py:210-215`).
  Confirmed.

### 2.2 Worker side: slicing at the source is mechanically possible

`compute_log_prob` `verl/workers/engine_workers.py:753-759` → `infer_batch` :430-470 →
`_postprocess_output` :214-273, which builds `final_output = tu.get_tensordict(tensor_dict=model_output, ...)`
at :272. The `tqbridge` decorator (`verl/utils/transferqueue_utils.py:347-418`) writes any TensorDict output
with a batch dimension back into TQ (`put_data` :401-407 → `_update_meta_with_output` :413), under the keys the
TensorDict carries. The worker's input `data` is materialised by `_meta_to_realdata` :160-180 from
`async_kv_retrieve_meta` :302-316 (all row fields unless `meta.fields` is set), so `response_mask` is in reach
inside `infer_batch`. Mechanism holds; the caller can pass the output field name in `batch.extra_info`.

### 2.3 Bytes re-derived at the BASELINE-A shape (`research/vet/K6/bytes_baseline_a.py`)

Dtypes from the writer: `log_probs`/`entropy`/`values` fp32 full-sequence; `prompts`/`responses`/`response_mask`
int64 (`AgentLoopOutput.response_mask: list[int]`, `verl/experimental/agent_loop/agent_loop.py:95`, tensorised by
`list_of_dict_to_tensordict`); `rollout_log_probs`, `rm_scores`, `old_log_probs`, `ref_log_prob`, `advantages`,
`returns` fp32 response-length. At N=1024 with the run's means (`prompt_length/mean` 82-87, `response_length/mean`
1195-1272, `response_length/clip_ratio` 0.15-0.21; script run with P=84, L=1230):

| site | GET | PUT | time model (03 §3.8: 25-40 µs/row/field fixed + ~1 GB/s) |
|---|---|---|---|
| old_log_prob | 36.0 MB (5 fields) | 10.4 MB (2) | 0.32 s |
| ref | 15.5 MB (2) | 5.0 MB (1) | 0.14 s |
| adv | 30.3 MB (6; `values` absent in GRPO) | 10.1 MB (2) | 0.35 s |
| metrics | 36.0 MB (7) | — | 0.25 s |
| total | 118 MB | 26 MB | **≈1.06 s/step** |

The brief's "~60 % of bytes" reproduces: the old_log_prob + ref GETs plus the prompts/responses part of the
metrics GET are ≈63 % of GET bytes, and ≈64 % of the chain time in the fixed-cost model (1.06 → 0.38 s if the
old_log_prob GET vanished entirely; → ≈0.55 s if the 3-field GET for `calculate_debug_metrics` stays, which is
the default-user case). What does not reproduce is the *importance*: the full-sequence `log_probs`/`entropy`
are 14 % of GET bytes and only 6 % of them are prompt positions on GSM8K (22 % at 03's 1k/3.5k shape). Time is
fixed-cost-bound (23.5k row-fields per step), not byte-bound. Cross-check: at 03's own shape (P=1000, L=3500)
the model gives 1.35 s vs 0.8 s measured — same order, model pessimistic.

### 2.4 Is it on the critical path, and what fraction of a step?

Yes: `_step_once` :563-621 is strictly sequential on the driver; every `tq.kv_batch_get`/`put` is synchronous
(`_run_async_in_temp_loop`) and `compute_log_prob` is a blocking dispatch; nothing overlaps GPU work. But the
BASELINE-A step is 31.15 s (mean of steps 2-6: gen 12.77, old_log_prob 3.49, ref 3.43, adv 0.61, update_actor
8.71, ≈2.1 s unaccounted incl. update_weights/reward/balance/metrics/kv_clear). The whole TQ chain is ≈3.4 % of
the step; K6 (a)+(b) removes ≈0.5-0.7 s ≈ **1.6-2.2 %**, inside the run's own step noise (steps 2-6 span
29.8-32.6 s, σ≈1.2 s). At 05's reference (Qwen3-8B, N=4096, 8k) the 3.5 s chain sits in a step of several
hundred seconds: <1 %. Honest answer to the brief: this is a 1-2 % item.

### 2.5 `poll_interval`

Default 2.0 at `verl/trainer/ppo/v1/replay_buffer.py:144`, slept at :409 (`_wait_for_next_poll` :408);
`_build_replay_buffer` `trainer_base.py:140-175` never passes it (kwargs :155-162); only
`trainer_separate_async.py:313` reads it via `getattr`. Mean idle = half the interval ≈ 1.0 s ≈ **3.2 %** of the
BASELINE-A step, counted inside `timing_s/gen` (:566-575). This is the largest and cheapest part of K6 — and it is
already in the V1 owner's open PR (§3).

### 2.6 Event-driven wait in TransferQueue 0.1.11: none

`.venv-a3/lib/python3.12/site-packages/transfer_queue/client.py`: `async_get_meta(mode="fetch"|"force_fetch")`
:237-306 is a non-blocking readiness query returning `BatchMeta.is_ready`; `async_get_production_status` :743 and
`async_check_production_status` :823 return status tensors; `kv_list` :1718 is a snapshot. The only "notify" is
producer→controller (`controller.py:1994 _handle_notify_data_update_request`). No subscribe/wait-on-key API, so
an event-driven sampler needs a TQ change (separate repo) or a Ray-side signal from `AgentLoopWorkerTQ` into the
sampler — possible only once the sampler is an actor, which is what #8203 makes it.

## 3. Duplicate / ownership check

Method: `python3 -I research/scripts-a7/dupcheck.py --kw kv_batch_get --kw slice --kw poll_interval --kw transferqueue --path verl/trainer/ppo/v1/ --merged`
(107 open PRs touch the directory), read-only `gh pr diff` of #8201 and #8203, bodies of #7415/#7000 from
`gh/prs_open.json`, `gh issue view 8186 --comments`.

| PR | author / state | overlap with K6 |
|---|---|---|
| #8203 "[BREAKING][trainer, cfg] feat: stream V1 rollout dispatch through a dataloader actor with session-watermark refill" | Begunner (COLLABORATOR, V1 owner); open 2026-10-12, last commit 5 d, MERGEABLE; 7,313-line diff over `trainer_base.py`, `agent_loop_tq.py`, `replay_buffer.py` (deleted → `streaming_dataloader.py`), `tq_schema.py`, config, 12 CPU tests | **Adds `trainer.v1.sampler.poll_interval: 2.0`** to `ppo_trainer.yaml` and `self.replay_poll_interval = self.config.trainer.v1.sampler.poll_interval` in `trainer_base.py`; the actor's `_coordinator_loop` waits `asyncio.wait_for(self._wakeup.wait(), timeout=self.poll_interval)`. K6 (c) duplicates a collaborator PR; K6 (a)/(b) edit a file this PR rewrites. |
| #8201 "[BREAKING][trainer, worker] feat: balance V1 PPO mini-batches and group outputs by UID" | FZYsheep; open 2026-10-12, last commit 2 d, MERGEABLE | Edits the PUT in `_compute_old_log_prob` (hunk `@@ -1666,9 +1759,12` = HEAD :1667-1672) and the PUT in `_compute_advantage` (`@@ -1797,20` = :1801), plus `_balance_batch`, `replay_buffer.py`, `engine_workers.py:242-330` (`train_mini_batch`, not `_postprocess_output`). K6 (a) deletes the exact lines #8201 edits → textual conflict, no functional overlap. |
| #7415 "[rollout, trainer] feat: add pluggable rollout data backends" | zxpdemonio; DRAFT since 2026-08-14, updated 2026-10-24, blocked on a Mooncake PR | Wraps every `tq.kv_batch_get/put` call in `trainer_base.py`, `agent_loop_tq.py`, `replay_buffer.py`, `transferqueue_utils.py` behind a `RolloutDataRef` facade → conflicts at every K6 site. |
| #7000 "[single_controller] fix: remove transfer queue and restore RayPPOTrainer" | zw0610 (ByteDance); open 2026-07-09, updated 2026-09-21 | Removes TQ, `use_v1=false`; NeoProto (#6988) "to land on the `neo` branch first". Not merged in 3.5 months and main moved on, but the substrate K6 optimises is formally contested by a core author. |
| #8424, #8364, #8379, #8310, #8300, #7497, #8303 | various, Oct 2026 | all touch `trainer_base.py` in the adv/metrics region — crowded file, rebase churn. |

Merged (`gh/prs_merged_recent.json`, 2026-07-16 →): nothing slices on the worker or reduces driver GETs; #8290 (open,
worker profiling windows) and #5958 (open, "data transport time log") measure the traffic rather than remove it.
Maintainer statements: roadmap #8186 lists "Remove V0 trainer" and no driver-byte item; RFC #5400 is the TQ design
home. No open issue asks for this (scan of `gh/issues_open.json` for TransferQueue/kv_batch_get bytes: none).

Decision rule (`research/07 §8`, fail closed): an open PR with a head commit in the last 60 days addresses (c) →
stop on (c). (a)/(b) are unaddressed, but sit inside two active rewrites of the same functions.

## 4. Minimum valuable PR

Scope that survives §3: (a)+(b), default off, filed only after #8201 and #8203 land (both MERGEABLE and recent).

- `verl/workers/engine_workers.py` `infer_batch` :431-470: `response_only = tu.pop(data, "response_only_outputs", False)`,
  `output_names = tu.pop(data, "output_field_names", None)`; after `self.engine.infer_batch(...)`, slice
  `model_output["log_probs"]`/`["entropy"]`/`["values"]` with `response_from_nested(..., data["response_mask"])`
  (`verl/workers/utils/padding.py:166-181`), rename per `output_names`, and for the actor compute
  `agg_loss(entropy, response_mask, loss_agg_mode)` into `metrics["actor/entropy"]` so the driver never needs
  `entropy`. ≈45 lines.
- `trainer_base.py`: `_compute_old_log_prob` :1651-1672 sets the two extra_info keys and drops the GET/PUT of
  `entropy`/`log_probs`; keeps a 3-field GET (`rollout_log_probs`, `old_log_probs`, `response_mask`) for
  `calculate_debug_metrics` when `rollout.calculate_log_probs` (default True) — or moves that metric to the
  worker too (+20 lines; all three fields are in the worker's row); `_compute_ref_log_prob` :1713-1718 and
  `_compute_values` :1735-1740 lose their GET/PUT. `_compute_metrics` :1876-1901 drops `prompts`/`responses` and
  takes lengths from `batch.tags[i]["prompt_len"/"response_len"]`; `_log_rollout_data` (:1301-1302) keeps its own
  GET. ≈50 lines.
- Config: `trainer.v1.response_only_outputs: false` (`ppo_trainer.yaml` + the four `_generated_*` yamls).
- Tests (CPU): `tests/trainer/ppo/v1/test_trainer_base_on_cpu.py` (workers are `MagicMock`, :177-200 pattern) —
  assert the extra_info keys and that no `kv_batch_get` of `log_probs` happens under the flag;
  `tests/workers/test_engine_workers_mini_batch_trace_on_cpu.py` pattern for the worker slice;
  `tests/utils/test_padding_on_cpu.py::test_response_from_nested` already pins the slice semantics. GPU: one V1
  GRPO e2e (BASELINE-A) with `actor/entropy` and `advantages` parity.
- Size ≈150-180 lines in two source files + two tests + yaml; no new dependency; one doc line in
  `docs/advance/v1_async_trainer.md`.

## 5. Proof on the Modal harness (`bench/README.md`)

Reference on main@2ab35292 (`bench/results/baseline-a-main-1`, steps 2-6): step 31.15 s = gen 12.77 +
old_log_prob 3.49 + ref 3.43 + adv 0.61 + update_actor 8.71 + update_weights 2.05 + ≈0.1 other (metrics.jsonl,
nested under `data`); 1.35 M tokens/step; `prompt_length/mean` 84, `response_length/mean` 1230,
`response_length/clip_ratio` 0.15-0.21 (Qwen3 thinking responses hit the 2048 cap often).

Run: `python3 bench/verl_bench.py train --preset baseline-a --run-name baseline-a-k6-<n> --src <branch> --total-training-steps 20 --overrides "trainer.v1.response_only_outputs=True"`
against the same 20-step run of `--src verl` (main); compare `timing_s/old_log_prob`, `timing_s/ref`,
`timing_s/adv`, `timing_s/step` means over steps 2..20 (`bench/summarize.py`). 20 steps because the expected
delta is below the 6-step σ.

Expected (script): chain 1.06 s → 0.55 s with the residual 3-field GET (−0.5 s, 1.6 % of step), → 0.38 s if the
debug metric also moves to the worker (−0.7 s, 2.2 %). By phase: old_log_prob −0.15/−0.3 s, ref −0.14 s,
adv+metrics −0.2 s. At 05's reference (Qwen3-8B, N=4096, 8k) 03 measured a 3.5 s chain → ≈−2 s on a step of
several hundred seconds (<1 %). The honest metric is a driver-side timer/bytes counter, which maintainers accept
less readily than step time. The 3.2 % poll-interval item belongs to #8203: lowering its new key's default from
2.0 to 0.2 s is a review comment there, not a PR.

## 6. Adversarial answers

- **A1 (strongest reason it will not be merged):** a ≤2 % data-movement refactor, invisible in step time at the
  reference scales, landing in the exact functions the V1 owner (#8203, BREAKING, 7k lines) and #8201 are rewriting
  this week, adding a flag that doubles the driver code paths (sliced/unsliced × bypass/ref/critic/debug-metric)
  — while its one cheap, visible part (poll interval) is already in the owner's PR. A first-time author would be
  told to wait for #8203 and then to justify a flag for a sub-noise gain.
- **A2 (what makes the gain disappear):** short prompts (prompt positions are 6 % of the full-seq bytes on
  GSM8K); `calculate_log_probs=True` keeps a 3-field GET; `rollout_correction.bypass_mode=True` already skips the
  old_log_prob GET and forward (:1643-1649); #8203's actor may batch GETs; NeoProto (#7000/#6988) or Mooncake
  (#7415) replace the substrate; and the bottleneck is TQ's per-row-per-field fixed cost (25-40 µs), so a TQ-side
  batching fix removes most of the second without touching verl.
- **A3 (correctness risk / tests):** the worker writes `old_log_probs` under a name the driver's bypass branch also
  writes (ordering); `actor/entropy` becomes a worker DP-all-gathered scalar that must match `agg_loss(loss_agg_mode,
  loss_scale_factor)` :1678-1683; `calculate_debug_metrics` parity; `_log_rollout_data` still needs token ids.
  Existing tests do not exercise the TQ data path (`test_trainer_base_on_cpu.py` mocks the workers;
  `test_replay_buffer_on_cpu.py` runs a real in-process TQ but only tags); a regression surfaces only in the V1
  e2e GPU runs (`tests/special_e2e/run_v1_*`), which the Modal `gpu-tests` preset can carry.
- **A4 (who else):** Begunner (#8203), FZYsheep (#8201), zxpdemonio (#7415), zw0610 (#7000/#6988); measurement
  side jiefangxuanyan (#8290) and #5958.
- **A5 (half scope):** (b) alone — lengths from tags, drop the int64 `prompts`/`responses` GET and merge it with
  the adv GET: ~30 lines, −0.1 s at BASELINE-A, −0.65 s at N=4096/8k (03 §3.8 metrics GET 646 ms). Below
  AGENTS.md's bar on its own.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | (a)(b) PASS, (c) **FAIL** | #8203 adds `trainer.v1.sampler.poll_interval` |
| G2 not reserved | **FAIL** | `trainer_base.py`/`replay_buffer.py` under the V1 owner's BREAKING rewrite (#8203) and #8201 at the same lines; substrate contested (#7000, #7415) |
| G3 self-contained | PASS | two source files, no engine fork, no deps |
| G4 provable | **FAIL** for step time (≈2 % < σ); PASS only with a driver bytes/time counter | §5 |
| G5 default path | PASS (V1 sync is the default) but 1-2 % | §2.4 |

Scores: S1 speed **1** · S2 demand **1** · S3 certainty **3** (derived from 03's measured per-row costs, not re-measured
on GPU) · S4 merge-ability **2** · S5 novelty **1**.

**Verdict: NO-GO.** Strongest reason: the measurable saving is ≈0.5-0.7 s of a 31 s BASELINE-A step (<1 % at the
8B/8k reference) and it lands in the lines two active PRs (#8201, #8203) are rewriting, while the one cheap,
visible piece (`poll_interval`) is already in the V1 owner's PR. Re-entry condition: after #8203 merges, propose
on that PR — not as a new PR — a wake-up signal from `AgentLoopWorkerTQ` into the `StreamingDataloader` actor's
`_wakeup` event plus a 0.2 s default (the 3 % item), and fold (b) in if the owner wants it.
