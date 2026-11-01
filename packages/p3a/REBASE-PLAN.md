# P3a [1/2] rebase plan onto #8203 (`StreamingDataloader` replaces `replay_buffer.py`)

Refreshed 2026-10-29 against `gh pr view 8203` / `gh pr diff 8203` (read-only) for the final code (2bd5c040).

## #8203 status (2026-10-29)

Open, not a draft, `REVIEW_REQUIRED`, mergeable, last updated 2026-10-23 (no author activity for six days);
one review (coderabbitai, COMMENTED, 2026-10-23) and two comments (coderabbitai, cla-assistant, 2026-10-12);
90 checks, mostly green, six failing (`sgl`, `vllm_ascend`, `e2e_ppo_trainer_fsdp_vllm_rocm`,
`model_rmpad_fsdp2_unstable_ascend`, `e2e_ppo_trainer_megatron-moe-expert-parallel_rocm`,
`e2e_ppo_trainer_megatron-qwen3`); 39 files, +3311 / -2152. It is `[BREAKING]`: `replay_buffer.py` becomes
`streaming_dataloader.py` (a Ray actor that owns rollout dispatch, sampling, refill and the restart-time re-issue),
`trainer_base.py` samples through the actor, and `test_replay_buffer_on_cpu.py` moves to
`test_streaming_dataloader_sampling_on_cpu.py`.

## Is landing before #8203 viable?

Yes, mechanically: every hunk of this branch is additive on `main` and does not touch the files #8203 deletes
except `replay_buffer.py`, where #8203 would then have to carry five small methods into the actor. Whether it is
sensible depends on #8203's timeline, which this review cannot read from CI counts: it has sat for six days with a
bot review only and a partly red CI, so there is no sign it lands in the next days, and this branch is small enough
to re-target in about a day if it does. Recommendation: propose this PR now and say in its text that it re-bases
onto #8203 on request (the mapping below is the plan).

## Hunk mapping (final code)

| hunk | lands on #8203 as |
|---|---|
| `ppo_trainer.yaml` `inflight_ttl_s`, `max_inflight_reissues`, `max_inflight_reissues_per_step`; generated yamls | unchanged; regenerate after the merge (#8203 adds `sampler.poll_interval` in the same block) |
| `_pending_prompt_tag(global_steps, attempt)` + stamp in `_submit_batch_to_rollout` / `_validate` | `tq_schema.prompt_tag(status, global_steps)` gains `dispatched_at` and `attempt`; `StreamingDataloader._register_prompts` already has `_persist_prompt_fields` (set it when the TTL is on in sync mode) |
| `_reissue_inflight_prompts(partition_id, uids=None)` fresh-uid path + orphan sweep | the actor's `_reissue_inflight_prompts` (restart-time version exists there, pops `__rollout_n__`); the fresh-uid branch is textual |
| `ReplayBuffer` kwargs `inflight_ttl_s`, `max_inflight_reissues`, `max_inflight_reissues_per_step`, `reissue_fn`; `prompt_dispatched_at`, `prompt_attempt`, `_retired_uids` | `StreamingDataloader.__init__` from `config.trainer.v1.sampler`; `reissue_fn` disappears (the actor owns `_reissue_inflight_prompts`) |
| `_expired_inflight_keys`, `_sweep_retired_attempts`, `_fail_exhausted`, `_hold_back_last_groups`, `_recover_stuck_inflight(..., reissue_budget, keep_last_finishable)` (budget charged for re-issues and exhaustion failures, after the draining branch; the padding-mode sampler passes `keep_last_finishable`) | same methods on `StreamingDataloader`; `request.metrics` replaces `eviction_metrics` |
| `_reissue_budget(batch_size)` (default `(cap + 1)` x prompts), `_begin_ttl_budget(partition_id, global_steps)`, `_within_reissue_budget` over the re-issuable and exhausted sets (per-step counter `_ttl_discards_in_step` keyed by `global_steps` on the buffer; `train_batch_size` wired in every mode) | the same three methods and the keyed counter on `StreamingDataloader`; `_begin_ttl_budget` is called with `request.global_steps` when a request is taken (the actor already serves the requests of one step in order), so "per step" stays per step |
| hook after `_sync_metadata_from_transfer_queue` in sync `sample` (with `draining`) and `wait_for_sampleable` | sync: top of `_advance_request`, return `None` + `self._wakeup.set()` after a re-issue; async: inside `poll_once` |
| `generate_sequences` rotation | #8203 already rotates (`_next_worker`); drop this hunk |
| `MetricsAggregator` sum list (incl. `inflight_ttl_suspended`) | unchanged |
| tests (`test_inflight_recovery_on_cpu.py` 23 items, `test_ttl_budget_full_loop_on_cpu.py` 4 items: the latter patches `replay_buffer.tq` / `time` and would patch the actor module instead) | re-target to the actor handle (`sample` / `wait_sampleable` RPC with a timeout) using #8203's `test_streaming_dataloader_actor_on_cpu.py` fixtures; the repeated-slow-replacement test keeps its shape (the real re-issue method is the actor's) |

Estimated re-targeting cost: about one day (the buffer methods and the per-step budget state move to the actor,
the trainer wiring disappears, 23 test items re-targeted); no semantic change.
