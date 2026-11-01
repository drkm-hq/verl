# P3 self-review findings and how each was resolved

1. **Stub-bound tests broke on a new instance method.** The first `_pending_prompt_tag` was a `PPOTrainer`
   method; `test_reissue_inflight_on_cpu.py` binds `_submit_batch_to_rollout` / `_reissue_inflight_prompts`
   to bare stubs, so 5 tests failed with `AttributeError`. Resolved: module-level `_pending_prompt_tag(global_steps)`
   and `_prompt_fields(batch)`; the two methods only read `trainer_mode`, `global_steps`, `agent_loop_manager`
   and (via `getattr`) `inflight_recovery_enabled`, exactly what the existing stubs expose.
2. **Partial configs in `test_trainer_base_on_cpu.py`.** `_build_replay_buffer` read
   `config.actor_rollout_ref.rollout.agent.relaunch_dead_workers` directly; 6 tests build a config without
   `actor_rollout_ref`. Resolved: `OmegaConf.select(..., default=False)`.
3. **Two attempts of one uid.** Re-issuing a slow-not-dead prompt under the same uid would let the old attempt's
   trajectory writes and terminal status merge into the new group (`_materialize_batch` gathers by uid prefix).
   Resolved by design: fresh uid per mid-run re-issue plus a per-poll sweep of anything the retired uid still
   writes (a tag-only status put recreates the key as a non-prompt row; verified on TQ 0.1.11 and covered by
   `test_retired_attempt_late_writes_are_swept`).
4. **Sync mode had no prompt rows to re-issue from.** `kv_batch_get` on a tag-only key raises. Resolved: persist
   rows in sync mode only when recovery is enabled; default path unchanged (test asserts tag-only put).
5. **Liveness probe vs. busy workers.** A `ping` on an actor whose event loop is saturated can time out.
   Resolved: only `RayActorError` marks a worker dead; a `ray.wait` timeout is "alive".
6. **Ownership map growth.** `_prompt_owner` gets one entry per dispatched prompt. Resolved: pruned to the
   in-flight set on every reap (`test_dispatch_to_dead_worker_relaunches_and_redispatches` asserts it empties).
7. **Validation partition.** Re-issuing val prompts under fresh uids would break `_validate`'s per-uid
   accounting and their rows are not persisted. Resolved: TTL/re-issue are train-only, documented; dead workers
   are still relaunched at the next dispatch. Open question for the reviewer below.
8. **Metric prefix in tests.** Test partitions are not the literal `train`, so keys are `validation/...`;
   tests use `ReplayBuffer._metrics_prefix` instead of hard-coding.
9. **Lint.** One E501 (ruff-format rewrapped it), two files reformatted; all hooks pass.

## Reviewer questions
- Keep the probe period/timeout as env-tunable constants, or add `trainer.v1.sampler.worker_liveness_period_s`?
- Should the `val` partition get the TTL too (would need persisted val rows and same-uid re-issue there)?
- Land before #8203 (additive, 332 lines) or hold and re-target to `StreamingDataloader` (REBASE-PLAN.md)?
- Is a `RuntimeError` with the worker index + uids the right flag-off behaviour, or should it stay a bare
  `RayActorError` for compatibility with anything catching it?
