# K14 fsdp-async-ckpt-save (vet, 2026-10-29)

Vetter: adversarial pass over the clone at verl-project/verl@2ab35292 (read-only), gh dumps in gh/, scripts and
outputs under research/vet/K14/. GitHub was only read.

## 1. Claim

Source: research/06-competitor-gaps.md §2.3 item 18 ("Checkpointing: async_save 'Only implemented for Megatron'")
and §3 gap #4 (size S-M, files named). Claim: `checkpoint.async_save` exists in `CheckpointConfig`
(`verl/trainer/config/config.py:35,47`, default False) but only the Megatron manager honours it, so every FSDP/FSDP2
save blocks all trainer ranks for the whole serialize-and-write; an asynchronous save for the FSDP engine (stage the
sharded state dicts to host memory, write them from a background thread, publish `latest_checkpointed_iteration.txt`
once every rank has finished) cuts the blocking time to the staging copy, which makes frequent recovery points
(self-healing) affordable. It would also close the FSDP half of issue #7952.

## 2. Mechanism verification

### 2.1 What an FSDP save does today (`verl/utils/checkpoint/fsdp_checkpoint_manager.py`)

- `save_checkpoint` :360-523. Rank 0 prunes capacity first (:385 -> `checkpoint_manager.py:166-179`, `shutil.rmtree`
  at :164), barrier :388.
- Sharded state dicts with `offload_to_cpu=is_device_available()` :399-400. That context only exists for FSDP1:
  `get_fsdp_state_ctx` returns `nullcontext()` for FSDP2 (`verl/utils/fsdp_utils.py:436-440`), so under `fsdp2`
  `self.model.state_dict()` :409 yields live DTensors on the GPU and `torch.save` :435 serialises straight from
  device memory (an async writer would read parameters the next optimizer step is mutating: staging is mandatory).
- Per-rank `torch.save` of the model :435, optimizer :439-440 and extra state (lr_scheduler + RNG) :449 as
  `model|optim|extra_state_world_size_{W}_rank_{R}.pt`. Plain pickles, not DCP.
- Rank 0 writes `huggingface/` config + tokenizer, `fsdp_config.json`, LoRA meta (:451-513); barrier :515; optional
  `hf_model` export :518 (`_save_hf_weights_from_per_tensor_param` :138-203: EVERY rank must drain the
  `per_tensor_param_fn` generator because it issues collectives, rank 0 writes 5 GB shards) + barrier :520; rank 0
  `register_checkpoint` :523 (`checkpoint_manager.py:181-188`, prunes beyond `max_ckpt_to_keep`).
- Engine wrapper `verl/workers/engine/fsdp/transformer_impl.py:878-896`: loads offloaded params to the GPU before the
  save, barrier, re-offloads after (:895). Worker RPCs `verl/workers/engine_workers.py:477-478` and :776-778
  (ONE_TO_ALL, blocking). Trainer: `verl/trainer/ppo/v1/trainer_base.py:485-486` wraps `_save_checkpoint` (:949-1030)
  in `marked_timer("save_checkpoint")`: actor :979, critic :993, dataloader `torch.save` :1000, TransferQueue :1006,
  then the tracker block :1012-1030.
- So the trainer blocks for: D2H copy (FSDP1) + pickle and write of model+optimizer+extra on every rank, three
  barriers, and on rank 0 the HF export.

### 2.2 Bytes and time for the reference models

- FSDP mixed precision keeps fp32 master params (`transformer_impl.py:396-405`: `param_dtype` bf16, flat params
  fp32) and fp32 AdamW moments: shard per rank = 12 B/param / 8. Qwen3-8B (8.2 B): 12.3 GB per rank, 98 GB per
  8-GPU node per save; Qwen3-1.7B (2.0 B): 3.0 GB per rank, 24 GB per node. `hf_model` adds 16.4 GB (8B) / 4.0 GB
  (1.7B) bf16 written by rank 0 alone.
- `torch.save` of host tensors runs about 1-2 GB/s per process on local NVMe (pickle + write); eight writers share
  one device or one NFS mount. Expected blocking per save: 8B 15-60 s (+10-16 s if `hf_model`), 1.7B 5-15 s. The CPU
  test below measured 287 ms sync vs 29 ms async-call for a 35 MB shard (10x).
- Cost per run = blocking x steps / save_freq. `trainer.save_freq: -1` by default (`ppo_trainer.yaml:174`): the
  DEFAULT user never saves and gains nothing. A save_freq=20 user on 8B with 5-min steps loses 0.3-0.6 %;
  save_freq=5 -> 1.2-2.4 %; save_freq=1 (what self-healing wants) -> 6-12 %. Async makes save_freq~1 cost about the
  staging copy: 12 GB D2H per rank (1-2 s on H100) plus 12 GB host RAM per rank (98 GB per node, transient).

### 2.3 Megatron's async path (the in-tree pattern)

- `verl/utils/checkpoint/megatron_checkpoint_manager.py:1045-1061`: `save_dist_checkpointing(async_save=...)`
  (`verl/utils/megatron/dist_checkpointing.py:32,47` -> MCore `async_sharded_save`); the finalize fn is attached to
  the LAST request (:1142-1147, `AsyncCallsQueue.schedule_async_request`) and writes
  `latest_checkpointed_iteration.txt` on rank 0 at `dirname(dirname(local_path))` (:1114-1123).
- Finalization has to be driven at step boundaries. V0 `verl/trainer/ppo/ray_trainer.py:1467-1468` (non-blocking
  every step) and :1809-1810 (blocking at the end), `verl/experimental/separation/ray_trainer.py:383-384,767-768`.
  The V1 trainer has NO such call (`grep -rn async_calls_finalize_fn_exec verl/trainer/ppo/v1/` -> 0), which is
  what PR #8056 adds.

### 2.4 Issue #7952 (YeonwooSung, 2026-09-20, open)

Both trainers skip the tracker whenever `async_save` is truthy, on the assumption the manager writes it later
(`trainer_base.py:1012-1020`, `ray_trainer.py:1037-1045`); the FSDP, veomni and torchtitan managers never do, so
`resume_mode=auto` silently restarts from step 0. Verified in the clone. The reporter opened #7955 (reject the
combination in config) and wrote "or would rather wait on #4311".

### 2.5 DCP `async_save` (torch 2.13 pin at uv.lock:5041/5067; 2.15 in the venv)

`torch.distributed.checkpoint.async_save(state_dict, *, checkpoint_id, storage_writer, planner, process_group,
async_checkpointer_type=THREAD, async_stager, no_dist, use_collectives)`: "de-stages the state_dict on to the staging
storage (defaults to CPU memory), and then calls the save in a separate thread"; marked experimental. It needs a CPU
backend in the group (`torch/distributed/checkpoint/state_dict_saver.py:304`: "A CPU backend must be enabled for
async save; try initializing process group with 'cpu:gloo,cuda:nccl'"), which verl already does
(`verl/utils/distributed.py:63,89`). ShardedTensor (FSDP1) and DTensor (FSDP2) are DCP-native. The catch is the
on-disk layout: DCP writes `__N_0.distcp` + `.metadata`, so `load_checkpoint` :268-358,
`verl/model_merger/fsdp_model_merger.py` and every reader of `model_world_size_N_rank_R.pt` would need a second
format. That is exactly the 512-line shape of stale PR #4311. Races: the staging copy is what protects the write
from the next optimizer step and from `offload_fsdp_model_to_cpu` (:895); weight sync reads live params through
`get_per_tensor_param`, never the staged copies, so it is unaffected.

### 2.6 CPU test (gloo, 2 ranks, the real `FSDPCheckpointManager`, clone untouched)

`research/vet/K14/test_async_save_cpu.py`: a prototype subclass stages model/optimizer/extra to host synchronously
(`_stage` deep-copies every tensor), one worker thread writes `.tmp` files and `os.replace`s them, and
`finalize(blocking)` all-reduces MIN(done) on the MAIN thread (gloo) before rank 0 publishes the tracker at
`dirname(dirname(local_path))`. Output (`research/vet/K14/test_output.txt`):

```
[rank 0] shard 34.6 MB | sync save 287 ms | async call 29 ms, finalize 143 ms | sync_bitexact=True live_changed_after_call=True async_equals_pre_race_snapshot=True tracker_only_at_finalize=True
[rank 1] shard 34.6 MB | sync save 288 ms | async call 25 ms, finalize 139 ms | sync_bitexact=True live_changed_after_call=True async_equals_pre_race_snapshot=True tracker_only_at_finalize=True
K14 CPU TEST PASS
```

The live model was mutated by an optimizer step after the async call returned; the files on disk still equal the
pre-mutation snapshot bit for bit (`torch.equal` on every model tensor and AdamW moment), the sync path reloads
bit-exact, and the tracker appears only at finalize with the right step.

## 3. Duplicate / ownership check (research/07 §8, `research/scripts-a7/dupcheck.py`)

`python3 -I research/scripts-a7/dupcheck.py --kw async_save --kw "async save" --kw "asynchronous checkpoint"
--kw "async checkpoint" --kw "checkpoint save" --kw 7952 --kw dcp --path verl/utils/checkpoint/ --merged`:
42 open hits, no merged-title hit. The ones that matter:

| PR | author (assoc.) | state | last commit | overlap |
|---|---|---|---|---|
| #4311 "[fsdp, doc] feat: Enhance FSDP checkpointing with DCP support" | dkremez (NONE) | open, REVIEW_REQUIRED, 0 reviews, +512/-94, 1 commit | 2026-01-05 (296 d, STALE; author pinged vermouth1992 for CI, no reply) | SAME GOAL: DCP format + `async_save` for FSDP; touches fsdp_checkpoint_manager.py, fsdp_model_merger.py, config.py, tests/special_distributed/test_fsdp_ckpt.py |
| #8056 "[ckpt, trainer, megatron] fix: finalize async checkpoints in RL trainers" | ewan0x79 (NONE) | open, MERGEABLE, REVIEW_REQUIRED (coderabbit only, 10-28), 27 files, +2207 | 0 d | the trainer-side plumbing any FSDP async save needs: new `verl/trainer/ppo/async_checkpoint.py` (`AsyncCheckpointCoordinator`), 3 `finalize_async_checkpoint` calls in `trainer_base.fit`, `BaseEngine.finalize_async_checkpointing` / `prune_checkpoints`, `save_checkpoint(update_tracker=, defer_retention=)`, FSDP engine `prune_checkpoints` (read with `gh pr diff 8056`) |
| #8003 "drain async checkpoints before SFT exits" | ewan0x79 | open, CHANGES_REQUESTED | 0 d | Megatron SFT, same author and subsystem |
| #8264 "synchronize overlapping parameters before checkpoint save" | ewan0x79 | open | 0 d | Megatron engine, same author |
| #7955 "[cfg, ckpt] fix: reject checkpoint.async_save on non-Megatron strategies" | YeonwooSung (CONTRIBUTOR) | open, REVIEW_REQUIRED (one approval, non-member) | - | turns `async_save` + fsdp into a config error; an implementation must relax it |
| #8355 "[ckpt, fsdp] feat: add filesystem checkpoint engine" | wdl339 (CONTRIBUTOR) | open | 5 d | weight sync trainer -> rollout over a shared FS ("synchronizes trainer weights with rollout workers"), NOT training checkpoints; touches fsdp_checkpoint_manager.py only to factor the HF export |

Issues: #7952 (open, 1 comment, no reactions). No RFC; roadmap #8186 has no FSDP-checkpoint line. File owners by
`git log` on fsdp_checkpoint_manager.py: Joel (#8171 #8180 #7283), kahlun (#8232), Zhiliang, Changyi Yang - HF
export and LoRA, nobody on async. Decision rule (07 §8): #4311 is stale (>60 d) so one may proceed, naming it and
the difference (format-preserving staged write vs DCP). #8056 is not stale and owns the trainer-side half; a K14 PR
either rebases on its API or re-implements 12 of its lines and collides.

## 4. Minimum valuable PR

Option A (recommended, format-preserving): `[fsdp, ckpt] feat: asynchronous checkpoint save for the FSDP engine`.

- `verl/utils/checkpoint/fsdp_checkpoint_manager.py` (+~110): `_stage_to_host(state_dict)` (deep copy to pinned
  host memory; FSDP1 `offload_to_cpu` shards are already on the host but alias nothing, FSDP2 DTensors must be
  copied), `save_checkpoint` writes model/optim/extra from one worker thread (`.tmp` + `os.replace` per file),
  `finalize_async_checkpointing(blocking) -> bool` (future done -> gloo all-reduce MIN on the main thread -> rank 0
  writes the tracker -> `register_checkpoint` and retention). `huggingface/` config, `fsdp_config.json` and the
  `hf_model` export stay synchronous (the export issues collectives on every rank).
- `verl/workers/engine/fsdp/transformer_impl.py` (+~15): engine `finalize_async_checkpointing` delegating to the
  manager; the post-save re-offload at :895 stays, the writer holds host copies.
- `verl/workers/engine_workers.py` (+~6): the `finalize_async_checkpointing` RPC (ONE_TO_ALL).
- `verl/trainer/ppo/v1/trainer_base.py` (+~12): non-blocking finalize at the top of each step, blocking before
  return/exit, and the tracker written only when finalize reports complete (replaces the skip at :1012-1020). This
  is the overlap with #8056; if #8056 lands first, use its `AsyncCheckpointCoordinator` and leave this file alone.
- `verl/trainer/config/config.py:35` docstring; `docs/advance/checkpoint.rst`. Config key: the existing
  `actor_rollout_ref.actor.checkpoint.async_save` (default False, opt-in, default path untouched); `critic.checkpoint.async_save` likewise.
- Tests: `tests/utils/ckpt/test_fsdp_async_checkpoint_on_cpu.py` (gloo x2, the prototype's four checks plus a
  blocking drain at exit) and an FSDP1/FSDP2 case in `tests/special_distributed/test_fsdp_ckpt.py` (GPU).
- About 200 source lines + 120 test lines. Not under the 300-line perf-PR shape unless the trainer part is dropped.

Option B (`torch.distributed.checkpoint.async_save`): fewer lines for the write itself, but a new on-disk format,
loader and model-merger changes, and it is what stale #4311 already did. Not minimum.

## 5. Proof on the Modal harness (bench/README.md)

- BASELINE-A as pulled (`bench/results/baseline-a-main-1`, main@2ab35292, 8xH100, Qwen3-1.7B, GRPO n=8, bs 128,
  1024/2048; measured: step 31.2 s mean, update_actor 8.7 s, gen 12.8 s, `actor/perf/cpu_memory_used_gb` 117) with
  `trainer.save_freq=1 trainer.max_actor_ckpt_to_keep=2 trainer.default_local_dir=/results/<run>/ckpt
  trainer.total_training_steps=8`; A: `actor_rollout_ref.actor.checkpoint.async_save=false`, B: `=true`.
- Metric: `timing_s/save_checkpoint` (the `marked_timer` at `trainer_base.py:485`) and `timing_s/step`, mean of
  steps 2..8; `actor/perf/cpu_memory_used_gb` for the staging cost; then a resume run (`trainer.resume_mode=auto`,
  +2 steps) proving the tracker and files are complete, and `torch.equal` of the two checkpoints' tensors.
- Arithmetic: 1.7B = 3.0 GB per rank, 24 GB per node per save; local NVMe at ~1.5 GB/s per process with 8 writers
  -> sync `save_checkpoint` about 6-12 s, async about 0.5-1.5 s (stage + barriers): -5 to -10 s per step at
  save_freq=1 (-15 to -30 % of a 31-s step), 0 at save_freq=-1. Second pair on Qwen3-8B (`--model Qwen/Qwen3-8B`):
  12.3 GB per rank -> 20-50 s sync vs 2-3 s async.
- Isolation: same seed and data, only `async_save` differs; the whole A/B difference must sit in
  `timing_s/save_checkpoint`, with `actor/pg_loss` and `actor/grad_norm` identical.

## 6. Adversarial answers (research/VETTING-RUBRIC.md)

- A1 Strongest reason it will not be merged: the trainer-side half is already in flight in #8056 (27 files, no
  maintainer review in 29 days) and the config guard in #7955; a reviewer will say "rebase on #8056" or "wait for
  the DCP direction (#4311)", and the ckpt area has shown no appetite for outside design PRs (#4311 unreviewed for
  11 months). The default run (`save_freq=-1`) gets nothing, so there is no before/after number for a default user.
- A2 What makes the gain disappear: saving rarely (save_freq >= 20 -> < 1 %); host RAM (BASELINE-A already shows
  117 GB used; staging adds 24-98 GB per node transiently, which with `param_offload`/`optimizer_offload` recipes on
  512 GB nodes can swap); storage slower than the step (NFS at 200 MB/s keeps the writer busy into the next save, so
  `finalize(blocking)` stalls: at most one checkpoint in flight); `hf_model` in `save_contents` (stays synchronous,
  16 GB on rank 0); a future DCP migration that makes the staged-pickle path dead code.
- A3 Correctness risk: publishing before every rank finished (mitigated: all-reduce MIN + tracker at finalize,
  tested); a crash mid-write (mitigated: `.tmp` + `os.replace`, the tracker still names the previous complete
  step); retention deleting a directory still being written (must move `register_checkpoint` to finalize; #8056's
  `defer_retention`); FSDP2 DTensor staging (`DTensor.to("cpu")` keeps a DTensor, the CPU test used plain tensors,
  so the GPU test is mandatory); HDFS upload and `del_local_after_load` unchanged. Existing tests that catch
  regressions: `tests/special_distributed/test_fsdp_ckpt.py` (GPU), `tests/utils/ckpt/test_checkpoint_cleanup_on_cpu.py`;
  the `tests/trainer/ppo/test_async_checkpoint_*` files exist only inside #8056.
- A4 Who else is working on it: ewan0x79 (#8056, #8003, #8264; commits every day), YeonwooSung (#7955), dkremez
  (#4311, silent since 2026-01-05), wdl339 (#8355, adjacent file). No maintainer statement either way.
- A5 Half scope: manager-only `async_save` that stages, writes in the background and finalizes at the START of the
  next `save_checkpoint` call plus a blocking drain in `__del__`/`atexit`; the trainer writes the tracker only when
  the manager reports complete. About 120 lines, no new trainer hooks, no collision with #8056 except the 8-line
  tracker block; cost: the last checkpoint of a run is published only at exit, and after a crash `resume_mode=auto`
  resumes from the previous save.

## 7. Gates, scores, verdict

- G1 Not a duplicate: PASS with a caveat. #4311 has the same goal (DCP + async for FSDP) and is stale at 296 d; it
  must be named with the difference stated. #8056 is adjacent (Megatron finalize + trainer plumbing), not the same
  fix.
- G2 Not reserved: PASS. No maintainer or RFC claim; but the trainer API is being shaped by an outside PR (#8056).
- G3 Self-contained: PASS. Manager + engine + one RPC + <= 12 trainer lines; no new dependency; no engine fork.
- G4 Provable: PASS. `timing_s/save_checkpoint` A/B at save_freq=1 on Modal; the CPU gloo test (bit-exact reload,
  race isolation, tracker at finalize) already passes.
- G5 Default-path impact: FAIL. `save_freq=-1` and `async_save=false` by default; value accrues only to users who
  save often (the self-healing case the owner wants, but not a default run), and #7952 asks for the bug to be
  fixed, not for the feature.
- S1 Speed/value 2 (0 % default; -15 to -30 % of step at save_freq=1; stability value real but unquantified).
  S2 Demand 2 (one issue, no reactions; Megatron precedent in-tree; torchtitan/NeMo-RL precedent out of tree).
  S3 Certainty 4 (mechanism measured on CPU, bytes derived, GPU numbers pending). S4 Merge-ability 2 (three open
  PRs in the same files, two from one active outsider; a reviewer will ask to rebase on #8056; ckpt outside-PR
  throughput is slow). S5 Novelty 2 (Megatron has it; DCP async_save is standard).
- Verdict: NO-GO for the top-3 now. GO-WITH-CHANGES only as a follow-up once #8056 (or a maintainer ruling on
  #7955/#4311) lands, implemented as Option A on #8056's `finalize_async_checkpointing` API. Strongest reason: the
  merge path runs through two unreviewed outside PRs (#8056 plumbing, #7955 guard) and a stale same-goal PR
  (#4311), and the default user gets nothing, so the "before/after on named hardware, merged within a day" bar is
  not reachable without inheriting someone else's unmerged design.
