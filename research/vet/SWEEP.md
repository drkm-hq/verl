# SWEEP — missed candidates, combinations, balance critique (sweeper, 2026-10-28/29)

Method: all thirteen reports read end to end (notes: `research/vet/SWEEP-notes.md`), 45 non-shortlist items
pooled (C-01..C-45), the leaders verified in the clone (`file:line` below) and against GitHub read-only
(`gh pr/issue view`, `gh api` GET, `research/scripts-a7/dupcheck.py`). Shortlist median estimated at 14-15/25
(K13/K9/K10-class items). Scores are mine, adversarial, and derived (no GPU run).

## 1. Ranked missed candidates (not on the shortlist)
| # | candidate | S1-S5 (total) | gates | why the shortlist missed it |
|---|---|---|---|---|
| 1 | **C-02 FP8 inference-only forward for the frozen ref** (`torch._scaled_mm`, FSDP2, opt-in) — roadmap #8186 LP Common P1, unclaimed; `transformer_impl.py:606` gates QAT off for forward-only engines, `:264-266`/`:436-438` keep the ref fp32 + CPU-offloaded → **S1 write-up** | 4/5/2/2/4 (17) | G2 ask-first; G3 size | the shortlist took K10 (bf16 + residency) and stopped one step short of the maintainer's own bullet |
| 2 | **C-36 agent-loop worker liveness: a dead `AgentLoopWorkerTQ` fails its groups instead of hanging the step** (`replay_buffer.py:449-503` loops forever; `agent_loop_tq.py:115,148,153` only the worker writes terminal status; `trainer_base.py:913` no re-issue in sync; 0 `max_restarts` anywhere) → **S2 write-up** | 3/3/4/3/3 (16) | G2 ask on #8380 | E4 covers GPU replicas; report 10's G3/G4 (CPU workers) were listed but not shortlisted |
| 3 | C-34 micro-batch/DP-invariance regression suite over every policy loss × agg mode + the unfiled `clip_cov` defect (report 09 C4; `core_algos.py:1812-1823` per-micro-batch random selection and local centering; no PR; #7830 covers kl_cov only, #7916/#5614 geo_mean) | 3/3/5/3/3 (17 nominal; test-heavy: `type: test` merged 4× in 3.5 months) | G5 n/a for a test | K12 was shortlisted as one bug; the harness that found C1-C4 was not |
| 4 | **C-28 deterministic V1 trajectory ordering under `full_determinism`** (`replay_buffer.py:388-391` sorts a set by `global_steps` only; `:401-405` walks `kv_list` order; `determinism.md:45-48,82,104` demands `use_v1=false`; V0 removal on the roadmap) → **S3 write-up** | 2/3/4/3/3 (15) | G2 ask on #8026 | report 06 gap 5 listed it; no vetter took "reproducibility" |
| 5 | C-10 per-phase GPU idle/utilisation + V1 idle-ratio metrics (#7601 contract; #3114/#5448/#4743 could not attribute idle) | 2/4/5/3/3 (17 nominal) | **G1/G2 risk**: #8116 (tongyx361, maintainer, "opt-in update phase timing and lifecycle metrics", 24 d) and #8289/#8290 (profiling windows) overlap | observability is not "speed" so no slice ranked it |
| 6 | C-27 length-aware dispatch: per-uid response-length EMA + longest-first submission (`agent_loop_tq.py:248-262` submits in dataset order; Seer/HistoPipe/RollPacker) | 3/3/2/3/4 (15) | G4 GPU-only; router half contested | K8 took affinity; the dispatch-order lever was left; pairs with K7 (§2) |
| 7 | C-09 FSDP2 export via `FSDPModule.unshard()` per unit instead of ~340 per-parameter `full_tensor()` (#7015 profile: export 2.4-2.7 s of a 3.4-4.1 s sync; `transformer_impl.py:1030-1038`) | 3/3/3/3/3 (15) | FSDP2 opt-in | same function as K4(a); fold into K4 (§2) |
| 8 | C-11 rank-0-only loading for tied-embedding models on the DEFAULT FSDP1 strategy (`transformer_impl.py:271`, `fsdp_utils.py:64-74`; #7834) | 3/2/4/2/2 (13-14) | **#7835 exists** (FSDP2-only, CPU test, one non-maintainer approval; wuxibin89 inline 2026-09-15: "tie_word_embeddings is usually used in small size model, so it's not a problem") | report 02 rank 5; the vetting pass skipped it; the maintainer headwind keeps it at the median — support #7835 rather than open a third PR |
| 9 | C-01 NCCL communicator suspend/resume rebase of #6408 (7-10 GB/rank in colocated Megatron MoE; ETOgaosion "tremendous release of memory") | 4/4/4/1/3 (16 nominal) | **G1/G2**: living PR (last human activity 2026-06-05), Megatron engine reserved (#8324), NCCL ≥ 2.29.7 docker | Megatron + rebase-of-someone-else's-PR = outside the campaign's lane |
| 10 | C-03 unified low-precision recipe block / fail-loud on train-vs-rollout format mismatch (#8186 P1; VikingDeng's 0.0058 → 0.031 silent mismatch) | 2/5/5/3/2 (17 nominal, low value) | #8287 merged the quantizer boundary; the cross-backend block is still open | plumbing; no speed; the withdrawn volunteer scared everyone off |
| 11 | C-21 validation with a colocated reward model does sleep → RM → full `update_weights` per validation batch (`trainer_base.py:1082-1085`): 30-60 s per validation | 2/1/4/3/2 (12) | pass | small and niche (model-based RM users only) |
| 12 | C-23 `separate_async` decoupled-PPO π_old shuffle: `save_model_to_cpu` / `restore_model_from_cpu` N + 2(N−1) times per step (`trainer_separate_async.py:201-225`) | 2/2/3/2/3 (12) | pass | **corrected**: the helpers are SHARDED (`experimental/separation/engine_workers.py:139-190`), 4.1 GB/rank for 8B → ≈ 3-7 s/step at N=4, not 16 s; the fix (a second forward-only engine for π_old) is M-size |
| 13 | C-05 fail-fast degenerate-run detector (RFC #7405, 0 comments; zero advantage variance / zero grad_norm) | 2/2/5/3/2 (14) | tiny (AGENTS.md no-tiny-PR rule) | too small to notice; nataliekung may do it |
| 14 | C-44 frame-hash dedupe in the agent loop (`agent_loop.py:899-951`; report 13 C4) | 3/2/4/4/3 (16) | pass | it is K1's missing source-side key → combination (§2) |

Dead after verification (do not spend time): C-32 SP no-op on Qwen3.5 — **retracted by its reporter on #7476**
(the memory was FSDP2 unsharded fp32 accumulated grads under `no_sync`); C-30 `aggressive_empty_cache` — only at
init (`engine_workers.py:744`) and in SFT; C-14/C-15 position-id loop and `tokenizer.pad` — V0-only callers
(`ray_trainer.py:1255-1381`; V1's `agent_loop_tq.py:155` postprocess never pads); C-33 `use_prefix_grouper` is a
dead flag on V1 (no engine reference) — a one-line fail-fast, bundle it with anything touching `actor.py`;
C-22 colocated delta IPC — G3 (L-size, vLLM patch API, sleep-level change); C-12 start-up overlap — cluster-only proof.

## 2. Combinations and conflicts

### Pairs that make one stronger PR (same file/area, same reviewer, shared tests)
- **K12 + C-34** → "[trainer, algo] fix: micro-batch-invariant loss aggregation on V1": K12's fix (pass
  `max_response_len`/`loss_scale_factor`, `padding.py:113-119`, `core_algos.py:1189-1193`) with the generic
  invariance harness (`research/repro-a9/repro_loss_agg.py` → `tests/trainer/ppo/test_loss_microbatch_invariance_on_cpu.py`)
  as its proof, plus `xfail(strict=True)` markers for `geo_mean` (#7916/#5614), `kl_cov` (#7830), `clip_cov` (no PR).
  One reviewer (losses/padding are a quiet zone: 2-3 active PRs), one test file, and the harness is what makes a
  first-time correctness PR credible.
- **K1 + C-44 (13-C4)** → stacked `[1/2]` frame-hash at the agent loop (`agent_loop.py:899-951`, `image_hashes` in
  `multi_modal_inputs`), `[2/2]` encoder dedupe keyed on it (`utils/model.py:615-630`, model wrappers). K1 alone
  recovers only duplicates that land in one micro-batch; the hash is the key for the cross-phase cache.
- **K7 + C-27** → one `PromptHistory` store keyed by uid (pass-rate EMA for GRESO skipping, response-length EMA for
  longest-first submission), one checkpoint hook, one config block, both consumed in
  `agent_loop_tq.py`/`trainer_base.py`. Two PRs on the same store would collide; one PR with two opt-in consumers
  shares the harness (DAPO-17k, epoch ≥ 2).
- **K10 → S1 (C-02)**: sequential. K10's `model_dtype: null` + per-pass residency is the prerequisite; the fp8
  swap is its second commit or a follow-up PR; same benchmark arms (A/B/C).
- **K4 split + C-09**: `get_per_tensor_param` (`transformer_impl.py:981-1065`) is one function; bf16-before-gather
  and `unshard()`-per-unit are the same ~30 lines. But K4 as listed bundles four ideas (bf16 gather, double-buffered
  IPC, LoRA export staging, disaggregated-LoRA bug) — that is the 2k-line shape that does not merge. Split:
  (a) trainer-side export: bf16 cast + `unshard()` per unit (`transformer_impl.py`), (b) double-buffered IPC
  (`bucketed_weight_transfer.py`, cite stale #5619), (c) LoRA export: skip base staging (`:992-995`) + bf16 cast
  (`fsdp_utils.py:745-751`), (d) disaggregated-LoRA fail-fast (`engine_workers.py:815-825`) — but check #7496
  (fully-async LoRA sync, fixes #7495) first; it may already be the fix. Each < 150 lines with its own A/B.
- **K6 + K3 (+ S2 hook)**: both live in `_compute_old_log_prob`/`trainer_base.py`. Order: K6 first (pure data
  movement, semantics unchanged), K3 on top (removes the phase in the on-policy case). K6's poll-interval change,
  05-C5 and S2's `liveness_fn` all land in the same 10 lines of `replay_buffer._wait_for_next_poll` — one PR.
- **E1 + 11-C3 + E2** → `[1/3]` versioned two-slot adapter swap with verify/rollback (E1 + C3 are one mechanism),
  `[2/3]` disaggregated adapter export, `[3/3]` N adapters (`max_loras: 2N`). E2 without E1's second slot is
  stop-the-world for N adapters.
- **E6 + E5 (12-C1)**: the ingest's `build_tq_row` and the device HTTP schema are the two ends of one contract;
  land the schema/row-author refactor first, keep the ingest actor in a recipe until #8203 settles (report 13 C3).
- **K13 + C-18 + C-19**: the entropy-in-fp32 one-liner (`torch_functional.py:243-245`) and the
  `use_no_sync_for_gradient_accumulation` dataclass/yaml alignment (`config/engine.py:290` vs `engine/fsdp.yaml:43`)
  are too small alone (AGENTS.md); C-18 rides with K3 (it touches the entropy metric path), C-19 with K11 (engine config).

### Pairs that conflict
- **K2 / K10 / K11 / K13** all edit `transformer_impl.py` (24 active PRs): four PRs from one author in weeks =
  weekly rebases of each other. Sequence by size: K13 (inside `prepare_model_inputs`) → K10 → K2 → K11, or fold
  K13 into K2's `prepare_model_inputs` change.
- **K3 / K6 / K12 / S3** all edit `trainer_base.py` (23 active) or `replay_buffer.py`; and **#8203 (Begunner,
  [BREAKING])** rewrites dispatch/selection: K6, K9, S2, S3 must either land before it or be written against it.
- **K9 vs #8203 and #8112**: sync straggler abort targets the sync sampler #8203 replaces with a dataloader actor;
  #8112 (tongyx361, prefix checkpointing) touches the abort/resume path.
- **K8 vs #6122/#6533/#7435/#6940**: the router is the most-contested rollout file after `vllm_async_server.py`;
  K8 passes G1 only because those are stale/draft — say so in the PR body.
- **E4 vs #8168/#8380/#7910**: highest ownership risk on the list; only the client-event-path + LB-fence minimum
  slice has a merge path (report 10 §7 G1 "A5").
- **K5 vs E3's turn-RLOO**: both add to `core_algos.py` — textual only; but `algo` merges 7 per 3.5 months.

## 3. Balance critique (speed · stability/self-healing · edge · merge-ability)
- **Speed-heavy, stability-light.** 14 of 18 shortlist items are speed; stability is represented only by K12 (one
  bug), K14 (recovery points) and E4 (pending). The sweep's verified stability gaps are generic, not edge-specific:
  any agent-loop worker death hangs the DEFAULT sync trainer (S2); no actor sets `max_restarts`; `generate` RPCs
  have no timeout (report 10 A6); the LB routes to dead servers (E4's minimum slice); reproducibility needs the
  trainer that is being deleted (S3); silent gradient-scaling classes (C-34) and silent no-op flags
  (`use_prefix_grouper`, `over_sample_rate` until K9). The top-3 should contain at least one of S2 / K12+C-34.
- **Edge is coherent but dependent.** E1-E6 hang together, but E3/E6 depend on #8203 and on uni-agent's direction,
  E2 is L-size, and E4 collides with three named FT owners. The edge item with the clearest merge path is E1's
  two-slot swap + verify/rollback (11-C3) — it also fixes a datacenter silent failure (base policy served after a
  failed `add_lora`, `vllm_rollout/utils.py:266-269`, `vllm_async_server.py:705-713`).
- **Merge-ability is the binding constraint.** 7 of 400 merged PRs came from first-time authors; a merged perf PR
  is < 300 lines, one area, numbers on named hardware, opt-in, and read by wuxibin89 in one sitting. The K items
  concentrate in the three most contested files (`transformer_impl.py` 24, `trainer_base.py` 23,
  `engine_workers.py` 19 active PRs). K4 is over-bundled (split above). Recommend leading with two small, verified,
  CPU-provable fixes (K12+C-34, S2's half slice or K6's slice-at-source) to become a CONTRIBUTOR, then K2/K1.
- **Under-weighted gates on the shortlist.** K3 only pays for single-mini-batch runs (3 of 50 explicit recipes;
  G5 weak). K8's 2-4 % is an estimate with no replay harness. K9 carries a short-response bias that must be
  measured. K7 only acts with `filter_groups` on and from epoch 2. K13 is 1-3 % on 8B (GPU-only proof). K5 is
  CPU-measured but only PPO-with-critic/REINFORCE++ users feel it.
- **Maintainer demand is absent.** None of #8186's P1 items is on the shortlist; S1 (FP8 inference-only forward)
  is the one that fits the FSDP lane. "Session-affinity routing" (K8) is the only roadmap-adjacent item.
- **Corrections to report numbers the shortlist may be using:** report 05's 16 s/step for C-23 is ≈ 5-8× high
  (sharded helpers); report 08's SP no-op (#7476) is retracted; report 02's tied-embedding item already has a PR
  (#7835) and a maintainer "not a problem"; `over_sample_rate` is confirmed dead config (K9's G5 story holds);
  #7496 may already fix K4's "suspected disaggregated-LoRA bug" — K4's vetter must read its diff.

## 4. Pointers
- `research/vet/S1-fp8-ref-forward.md` (17/25, GO-WITH-CHANGES after K10; claim on #8186 first).
- `research/vet/S2-agent-loop-worker-liveness.md` (16/25, GO-WITH-CHANGES; ask on #8380; half slice ≈ 70 lines).
- `research/vet/S3-v1-deterministic-ordering.md` (15/25, GO-WITH-CHANGES; ask on #8026; land before V0 removal).
- `research/vet/SWEEP-notes.md`: per-report notes, the 45-item pool, verification log with every `file:line`.
Nothing was posted to GitHub; the clone is untouched (`git -C verl status --porcelain` empty).
