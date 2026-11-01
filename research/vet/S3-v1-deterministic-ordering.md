# S3 — Deterministic V1 trajectory ordering so `full_determinism` survives the V0 removal

Vetter: sweeper. Clone `2ab35292`. Not on the shortlist. Source: report 06 §3 gap 5; report 08 §2 row 12;
`docs/advance/determinism.md:45-48, 82, 104-106, 125`.

## 1. Claim
verl's documented reproducibility mode (`rollout.full_determinism`, vLLM batch invariance, deterministic
request ids and reward routing) requires `trainer.use_v1=false` because the V1 replay buffer hands the trainer
its batch in TransferQueue completion order, which varies run to run. The 26Q4 roadmap (#8186) removes the V0
trainer, which would delete the only reproducible path. The candidate stamps a submission sequence number in
each prompt's TQ tag and orders the sampled batch by it (prompts) and by `(seq, session_id, index)`
(trajectories), so two runs with identical seeds feed `_balance_batch` the same rows in the same order.

## 2. Mechanism verification (file:line)
- Prompt tags at submit carry only `is_prompt / status / global_steps` (`verl/trainer/ppo/v1/trainer_base.py:1499`,
  `_submit_batch_to_rollout` `:1497-1512`); the dataset `index` reaches the worker in the batch
  (`agent_loop_tq.py:88`, `get_trajectory_info(batch["global_steps"], batch["index"], validate)`) but is not in
  the tag the sampler reads.
- `_select_prompt_uids` sorts a *set* of uids by `global_steps` only (`verl/trainer/ppo/v1/replay_buffer.py:388-391`);
  all prompts of one step tie, so the order is set-iteration order of uuid4 strings (hash-seed dependent across
  processes).
- `_materialize_batch` builds `keys` by iterating `partition_snapshot` (`:401-405`), a dict filled in
  `tq.kv_list()` order by `_sync_metadata_from_transfer_queue` (`:201-235`) = storage/completion order.
- `_balance_batch` (`trainer_base.py:1609-1633`) runs Karmarkar-Karp on that order and `batch.reorder`s it, so the
  micro-batch composition (and therefore numerics even with batch-invariant kernels) follows completion order.
- Existing determinism plumbing that this completes: `rollout.full_determinism` (`verl/workers/config/rollout.py:207`),
  `det-{priority}` request ids (`single_turn_agent_loop.py:69`, `tool_agent_loop.py:127-142`), the reward router's
  body-keyed worker choice (`reward_loop/router/naive_router.py:98-147`), `VLLM_BATCH_INVARIANT` in `main_ppo.py`.
- The doc states the requirement and its cause (`determinism.md:104`: "collected in completion order ... varies
  across runs"; V0 "uses submission order"), i.e. the fix is exactly "restore submission order".

## 3. Duplicate / ownership check
- `dupcheck.py --kw full_determinism --kw "completion order" --kw bitwise --kw "deterministic order" --kw reproducib
  --merged`: 35 hits; relevant: #8026 (cr-gao, batch-invariant actor forward + sampler log-prob formula; kernels,
  not ordering; CONFLICTING, commit 1 d), #8366 (flyingdoubleG, keep vLLM compile under `full_determinism`, 3 d),
  #8403 (ISEEKYAN, DSV4.1 LoRA recipe with bitwise log-probs). None touches `replay_buffer.py` ordering.
- Files contested: `replay_buffer.py` is touched by #8203 (Begunner, [BREAKING] streaming dispatch) and #8201
  (FZYsheep, [BREAKING] mini-batch by UID); `_select_prompt_uids`/`_materialize_batch` may move → rebase risk.
- Roadmap: #8186 "batch-invariant / deterministic train+rollout kernels" (P1/P2) CLAIMED by cr-gao for kernels;
  the trainer-ordering piece is unclaimed; "Remove V0 trainer" is the maintainers' own item this unblocks.
- Merged since 2026-07-16: none on ordering (#8025 raw log-prob mismatch metrics is the measurement side).
- Ask-first on #8026 or the roadmap thread (not posted; needs the submitter).

## 4. Minimum valuable PR (~50 code + 120 tests + 20 docs)
- `trainer_base.py:_submit_batch_to_rollout` (`:1497-1512`): add `"seq": self._submit_seq + i` to every prompt
  tag from a monotonic counter saved/restored with the trainer checkpoint state (+10); `_reissue_inflight_prompts`
  (`:911-947`) re-puts tags → keep the original `seq` (+3).
- `replay_buffer.py`: record `prompt_seq[partition][key] = tag.get("seq", -1)` in `_sync_metadata_from_transfer_queue`
  (+4); sort `_select_prompt_uids` by `(global_steps, seq, key)` (+2); order `_materialize_batch` keys by
  `(seq[uid], session_id, index)` parsed from `{uid}_{session}_{index}` (+8). Unconditional: it replaces an
  arbitrary order with a stable one at zero cost; KK balancing re-sorts by workload anyway.
- Docs: `determinism.md` drops the `use_v1=false` requirement (`:45-48`, `:82`, `:104-106`, `:125`) once the GPU
  run below holds (+20).
- Tests: in `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py` (real TQ on Ray local): submit 8 prompts with
  `seq` tags, write their trajectories in several shuffled completion orders, assert identical `keys`; async
  variant with mixed `global_steps` (older first, then seq); `test_reissue_inflight_on_cpu.py`: seq preserved on
  re-issue.

## 5. Proof
- CPU: ordering invariance under shuffled completion (above).
- Modal, 1×H100 (2× for the doc's TP case): the doc's own protocol — two runs of the determinism example with
  `full_determinism=true`, `trainer.use_v1=true`, identical seeds; compare the per-step series
  `rollout_probs_diff_mean`, `actor/pg_loss`, `critic/rewards/mean`. Expected: before the change the V1 series
  diverge at step 1 (different batch order → different micro-batches); after it they are bitwise identical on the
  rollout side and, with #8026's training-side batch invariance, on the training side; without #8026 assert the
  batch key order and rollout outputs are identical (vLLM batch invariance already holds).

## 6. Adversarial answers
- A1: maintainers may treat determinism as a V0-era niche and let the V0 removal retire it, or want it inside
  cr-gao's batch-invariance line; #8203 rewrites selection, so the diff may have to move into the
  `StreamingDataloader` actor. Mitigation: 50-line additive change, ask on #8026/#8186, offer to rebase.
- A2: nothing cancels it; a future TQ that lists keys in insertion order would still reflect completion order.
- A3: ordering-only; no numerics change. The 41 sampler tests pin oldest-first, DAPO and failure-refill semantics.
- A4: cr-gao (#8026/#8025), flyingdoubleG (#8366), Begunner (#8203), FZYsheep (#8201).
- A5: seq tag + prompt ordering in sync mode only (~25 lines + 1 test).

## 7. Gates, scores, verdict
G1 pass · G2 pass-with-ask (kernels claimed, ordering unclaimed) · G3 pass · G4 pass (CPU ordering test; GPU series
per the doc's protocol) · G5: `full_determinism` is opt-in and documented; the ordering change itself is on the
default path at zero cost.
S1 2 (capability/debuggability, no speed) · S2 3 (documented requirement, roadmap P2, V0 removal) · S3 4 ·
S4 3 (tiny but in contested files) · S5 3. Total 15, at/just above the median; its justification is the V0-removal
dependency more than the score. Verdict: GO-WITH-CHANGES (ask first; coordinate with #8203; land before V0 is removed).
