# K4b — LoRA adapter is never shipped on the disaggregated (non-naive) weight-sync path

Vetter verdict file (adversarial). Clone `<repo>/verl` @ `2ab35292`,
read-only; all `file:line` cites are into that tree. Scripts and outputs: `research/vet/K4/`.
GitHub was used read-only (`gh issue view`, `gh pr view/diff/list`), dumps of 2026-10-28 in `gh/`.

## 1. Claim

Source: `research/11-edge-weight-updates.md` §3.6 gap 1, folded into the K4 brief. Claim: every
non-naive checkpoint-engine backend (`nccl`, `nixl`, `mooncake`, `kimi`, `hccl`) calls
`self.actor.engine.get_per_tensor_param()` with the default `base_sync_done=False` and discards the
returned `peft_config` (`verl/workers/engine_workers.py:815-825`); for a PEFT model with
`model.lora.merge=False` (the default) that branch exports the frozen base weights only
(`verl/utils/fsdp_utils.py:753-765`), and the rollout-side `CheckpointEngineWorker.update_weights`
forwards no `peft_config` (`verl/checkpoint_engine/base.py:354-360`), so a disaggregated LoRA RL run
(V1 `separate_async`, V0 `fully_async_policy`, `one_step_off_policy`) keeps sampling from the
initial policy on every step, silently.

## 2. Mechanism verification — the claim is TRUE (code reading + CPU reproduction + a user report)

Trainer side, non-naive branch (`verl/workers/engine_workers.py`):
- `:812` `effective_mode = mode if mode != "auto" else config.rollout.checkpoint_engine.backend`;
  `:815` `if effective_mode != "naive":` → `:821` `per_tensor_param, _ = self.actor.engine.get_per_tensor_param()`
  — no kwargs, so `layered_summon=False, base_sync_done=False` (`verl/workers/engine/fsdp/transformer_impl.py:981`),
  and the second return value (the `peft_config` dict, `:1064-1065`) is bound to `_`;
  `:822` `send_weights(per_tensor_param, global_steps=...)`; `:825` return. Nothing else runs.
- The naive branch is the only one that drives the two-phase protocol: `:845-847`
  `get_per_tensor_param(layered_summon=..., base_sync_done=True)`, `:850-852` sets
  `rollout.sleep_level = 1` and `do_lora_base_sync = not self.base_sync_done`, `:855-865` base
  push (once) then adapter push with `peft_config=peft_config, base_sync_done=True`, `:885` flips the flag.
  `self.base_sync_done` is initialised at `:717` from `"dummy" not in rollout.load_format`.

FSDP engine export (`verl/workers/engine/fsdp/transformer_impl.py`):
- `:1000` `merge_lora = self.model_config.lora.get("merge", False)` (default False); `:1003-1012`
  LoRA + not merged → `peft_config = peft_model.peft_config["default"]`, `collect_lora_params(module,
  layered_summon, base_sync_done)`, and when `base_sync_done` is False the keys are renamed with
  `replace_lora_wrapper` (`:1011-1012`) to plain HF base names; `:1029-1039` because `base_sync_done`
  is False the generator all-gathers each base tensor (`full_tensor()`); `:1064-1065` returns
  `(per_tensor_param, peft_config.to_dict())` — the dict the caller throws away.
- `verl/utils/fsdp_utils.py:706-780 collect_lora_params`: FSDP branch `:743-765` and plain branch
  `:767-779` apply the same rule — `base_sync_done=True` → `get_peft_model_state_dict(peft_model)`
  (adapter tensors only, `:745`, `:769`); `base_sync_done=False` → iterate
  `peft_model.base_model.model.state_dict()`, **skip every name containing `_flat_param` or `lora_`**,
  strip `.base_layer` (`:756-759`, `:774-777`) → base weights only.
- The merged path is correct on every backend: `:1013-1018` → `_merged_lora_per_tensor_param`
  (`:1067-1096`) streams base+adapter merged full tensors (bf16 cast at `:1086`) — but that is the
  full 16.4 GB export per sync for an 8B, i.e. the thing LoRA-only sync exists to avoid.

Rollout side:
- `verl/checkpoint_engine/base.py:353-360` `CheckpointEngineWorker.update_weights(global_steps)`:
  `weights = self.checkpoint_engine.receive_weights(...)`; `server_adapter.update_weights(weights,
  global_steps=..., wire_format=...)` — the signature has no `peft_config`/`base_sync_done` to forward.
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:229-233` passes `{**kwargs, "use_shm"}` to
  `update_weights_from_ipc`, whose defaults are `peft_config=None, base_sync_done=False`
  (`verl/workers/rollout/vllm_rollout/utils.py:248`). `:266` `if peft_config and base_sync_done:` is
  False → `:274-282` plain base-weight preparation; `:293` `lora_weights = None`; `:326-330` →
  `_update_weights(weights, None, False)` → `:395-399` `model.load_weights(model_updates)`: the
  arriving names are valid HF base names, so the load **succeeds silently**.
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py:449-467`: standalone replicas do start with
  `enable_lora=True, max_loras=1` when `lora_rank > 0` and `merge` is False, so `lora_as_adapter` is
  on; `:705-713` attaches a `LoRARequest(123)` **only if** `VLLM_LORA_INT_ID in engine.list_loras()`,
  which is never populated (no `add_lora` ever runs, `utils.py:363-377` is only reached from the
  `peft_config and base_sync_done` branch). Every request therefore samples the **base model**,
  while `set_global_steps` still stamps the samples with the new step (`vllm_rollout.py:249-250`).

Which trainers hit the branch:
- V1 `separate_async` (the supported successor per `docs/advance/v1_async_trainer.md`):
  `verl/trainer/ppo/v1/trainer_separate_async.py:69-71` asserts `backend != "naive"`; `:394-397`
  `standalone_checkpoint_manager.update_weights` → `CheckpointEngineManager.update_weights`
  (`base.py:510-541`) → `actor_wg.update_weights(mode=self.backend)` → the branch above.
  With `enable_switch=True` the hybrid replicas are refreshed through the naive path
  (`trainer_separate_async.py:417-422`) and do get the adapter, while the standalone pool stays on the
  base: two policies serving one step (code reading, not run).
- V0 `fully_async_policy` (`verl/experimental/fully_async_policy/fully_async_trainer.py:734`) and
  `one_step_off_policy` go through the same manager; `delta_sharded` drives
  `get_per_tensor_param_shard` (`transformer_impl.py:922-952`, docstring "Non-LoRA base path only")
  with no LoRA guard anywhere (`grep -n "lora\|peft" verl/checkpoint_engine/delta_checkpoint_engine.py`: 0 hits).
- Megatron has the same shape: `verl/workers/engine/megatron/transformer_impl.py:993-1013`
  `adapter_only = base_sync_done and non_merge_lora_sync` → with the default it exports
  `export_hf_weights(..., merge_adapter_weights=False)` = base without adapters (code reading).

CPU reproduction (`research/vet/K4/k4b_lora_disagg_repro_cpu.py`, output `k4b_lora_disagg_repro_cpu.out`):
drives the REAL `ActorRolloutRefWorker.update_weights.__wrapped__` and the REAL
`CheckpointEngineWorker.update_weights.__wrapped__` with a real `peft` model (r=4, q/v targets,
22 params, 8 adapter tensors) through the REAL `collect_lora_params`/`replace_lora_wrapper`
(plain-module branch, same rule as the FSDP branch), `load_format=safetensors`, `merge=False`:

```
collect_lora_params  base_sync_done=False: 14 tensors, 0 contain 'lora_'
                     base_sync_done=True :  8 tensors, 8 contain 'lora_'
backend=nccl      calls=[False, False]  exported=14 tensors, 0 adapter  -> server_adapter kwargs={'wire_format': 'named_tensors'}
backend=nixl      calls=[False, False]  exported=14 tensors, 0 adapter  -> same
backend=mooncake  calls=[False, False]  exported=14 tensors, 0 adapter  -> same
backend=naive     calls=[True, True]    exported= 8 tensors, 8 adapter  -> kwargs={'peft_config': True, 'base_sync_done': True}
RESULT: every non-naive backend exports the frozen base only and forwards no peft_config.
```

Run: `cd research/vet/K4 && PYTHONPATH=$PWD/../../../verl ../../../.venv-k4/bin/python k4b_lora_disagg_repro_cpu.py`
(`.venv-k4` = `.venv-a9` + `pyzmq peft uvicorn fastapi`, layered through a `.pth`; the clone is untouched).

Tests, CI, docs on the clone:
- No test combines LoRA with a non-naive backend: `tests/checkpoint_engine/` holds
  `test_fsdp_lora_only_checkpoint_on_cpu.py` (checkpoint save, not sync), `test_correctness_on_gpu.py`
  (nccl/nixl/kimi with a full-weight model, `check_allclose`), `test_global_steps_on_cpu.py` (plumbing);
  `grep -rn lora .github/workflows/*.yml | grep -iE "nccl|nixl|separate_async|hybrid_engine=False|disagg"`: 0 hits;
  no e2e script under `tests/special_e2e/ examples/ recipe/` sets `lora_rank` together with a non-naive
  backend or `hybrid_engine=False` (0 hits).
- `docs/advance/ppo_lora.rst` never mentions async/disaggregated modes; `docs/advance/v1_async_trainer.md`
  never mentions LoRA (0 hits). Nothing documents the limitation, nothing asserts it.

User report (the decisive evidence): issue **#7495** (2026-08-20, ChaoyuWang04) "[BUG][LoRA][Async]
Disaggregated weight sync never delivers the LoRA adapter — rollout keeps sampling from the ...":
exactly this mechanism, found on a `fully_async` + NCCL run. The three comments:
wuxibin89 (collaborator): "@HollowMan6 Does LoRA really need async training?"; HollowMan6
(collaborator, LoRA sync owner): "Never tried to test async training for LoRA, since if we deploy LoRA
training as a service, we don't really need async ones as vllm supports multi-LoRA ..."; the reporter:
would not push the project to own async LoRA, but "At the moment it is accepted silently, training
proceeds and the metrics look plausible while the rollout policy never actually changes ... A warning
or an assert would be more than enough, and I would be glad to rework the PR into that". No maintainer
reply after that (as of 2026-10-29).

## 3. Duplicate / ownership check

- **PR #7496** "[rollout, fully_async] fix: sync LoRA adapters to disaggregated rollout engines"
  (ChaoyuWang04, created 2026-08-20, OPEN, not draft, `mergeable: CONFLICTING`, `reviewDecision:
  REVIEW_REQUIRED`, 2 commits, last commit 2026-08-21, +258/−4, CI `SUCCESS`, 0 reviews, 0 PR
  comments; `gh pr list --search "7495 in:body"` returns only this PR). Files: `verl/workers/engine_workers.py`
  (the non-naive branch now keeps a `ckpt_base_sync_done` flag initialised from `load_format` and
  calls `get_per_tensor_param(base_sync_done=...)`, flipping it once a `peft_config` comes back),
  `verl/checkpoint_engine/base.py` (`CheckpointEngineWorker.update_weights` peeks at the first tensor
  name; an all-`lora_` stream is an adapter push, for which it rebuilds a minimal `peft_config` from
  `model_config` via `_infer_lora_peft_config` and forwards `peft_config` + `base_sync_done=True`),
  `tests/checkpoint_engine/test_disaggregated_lora_sync_on_cpu.py` (7 CPU tests: 3 fail on main,
  7 pass with the PR), `test_global_steps_on_cpu.py` (signature update). Author-reported GPU evidence:
  Qwen3-4B + LoRA r32, 3 trainer + 1 rollout GPU, `fully_async`, NCCL: payload 8,414 MiB → 252 MiB
  (504/504 `lora_` tensors), `param_sync` 6.25 s → 0.97 s, `rollout_corr/kl` 6.4e-05 vs 3e-3-and-rising.
  Its mechanism matches what this vet derived independently (same two call sites, same rule).
- Adjacent owned work (none touches the disaggregated branch): #8032 HollowMan6 (collaborator)
  `lora.resync_base` — releases LoRA base weights on sleep, colocated path; #8105 / #8106 aoshen02
  (collaborator) drafts — adapter hooks during tensor sync / duplicate LoRA tensor copy after IPC
  receipt; #8246 aoshen02 draft — FP8 GRPO with LoRA on FSDP2 (touches `bucketed_weight_transfer.py`);
  #8307 Levius-Fubuki — VeOmni native LoRA; #8309 kahlun — critic LoRA; merged: #8352 aoshen02 (LoRA
  host block), #7327 HollowMan6 (receiver `.base_layer`), #7179 / #8287 wuxibin89 (weight-sync clean-up,
  unified quantizer). Issue #7909 (multi-LoRA) is the other open LoRA RFC.
- Owners (git + CODEOWNERS): `verl/workers/rollout/vllm_rollout` → wuxibin89, PeterSH6, chenhaiq,
  ArronHZG, HollowMan6; `verl/checkpoint_engine` → no CODEOWNER, last touched by ChangyiYang (delta),
  Parinay Chauhan (nccl), wuxibin89; `collect_lora_params` → Qing An (#6512), Fang Liu (#5115),
  wuxibin89 (#3285). Practically every merge in this area is wuxibin89's.
- Maintainer statements: the two #7495 comments quoted in §2 (wuxibin89: "Does LoRA really need async
  training?"; HollowMan6: "we don't really need async ones as vllm supports multi-LoRA"). Nobody has
  refused a fail-fast; nobody has reviewed #7496 in 69 days.
- Local dupcheck: `python3 -I research/scripts-a7/dupcheck.py --kw lora --kw peft` → 75 open PRs,
  only #7496 addresses this branch. Roadmap #8186 has no LoRA-sync item.

## 4. Minimum valuable PR

As a campaign contribution: **none** — the fix exists (#7496) and AGENTS.md forbids a second PR for an
open fix. What is legitimately ours:
1. A review of #7496 with the V1 angle (the issue and PR are framed around `fully_async`, which
   #8186 deprecates to verl-recipe; the supported V1 `separate_async` trainer takes the identical
   branch, `trainer_separate_async.py:69-71, :394-397` → `engine_workers.py:815-825`), plus the
   `enable_switch` mixed-policy observation — posted on #7495, not as a new PR.
2. Only if #7496 is abandoned (author silent since 2026-08-21, branch CONFLICTING): the fail-fast the
   reporter offered — `engine_workers.py:821` bind the second return value and
   `if peft_config is not None and not self.peft_merge: raise NotImplementedError("LoRA adapters are
   not delivered by checkpoint_engine.backend=<x>; use model.lora.merge=True or the colocated (naive)
   path; see #7495")` (~8 lines), a CPU test in the `test_global_steps_on_cpu.py` style (~40 lines),
   one sentence in `docs/advance/ppo_lora.rst`. Title: `[rollout, ckpt] fix: fail fast when LoRA
   adapters cannot reach a disaggregated rollout`. Default behaviour changes from silent to loud, which
   is the point; no config key.
3. The full fix (~260 lines, #7496's shape). The debatable part is the receiver inferring "adapter
   push" from the first tensor name; a cleaner design carries `peft_config` + version in each
   backend's metadata side channel (nccl ZMQ PUB, nixl side channel: ~40 lines per backend) — that is
   E1 (lora-edge-refresh) territory and must be built on top of #7496, not beside it.

## 5. Proof

- CPU: `research/vet/K4/k4b_lora_disagg_repro_cpu.py` (above) — runs in seconds, no GPU, no Ray;
  #7496's `test_disaggregated_lora_sync_on_cpu.py` is the regression test that would pin a fix.
- GPU (only if a fix is pursued): Modal, one container with 2×H100 (or `H100:4` as 3+1): Qwen3-0.6B,
  `model.lora_rank=8, lora.merge=False, rollout.load_format=safetensors`, V1 `separate_async`,
  `checkpoint_engine.backend=nccl`, 10 steps. Asserts: `engine.list_loras()` is `[123]` after the first
  sync (today: `[]`); `training/rollout_probs_diff_mean` (already in BASELINE-A's metrics.jsonl) stays at
  the floor (~5e-3 at step 1 on 1.7B colocated) instead of growing step over step; same-seed
  comparison against the colocated `naive` run. Expected (from #7496): payload 2-3 orders of
  magnitude smaller, `update_weights` several × faster, KL at the floor.

## 6. Adversarial answers

- A1 (why it will not merge): it is a duplicate of #7496, and the two maintainers who own LoRA sync
  said on record that async LoRA is not something they need; a second PR invites the AGENTS.md ban.
- A2 (what makes the gain disappear): `model.lora.merge=True` already works everywhere (at the cost of
  the full export); if #8032 / #8105 (HollowMan6 / aoshen02) restructure the LoRA sync they can fold
  the two-phase protocol into the shared branch and make #7496 moot.
- A3 (correctness risk): for the fix — the receiver-side name peek, base-first ordering with
  `load_format=dummy`, and mixed hybrid/standalone policies under `enable_switch`; caught by #7496's
  CPU tests plus a `list_loras()` assertion on GPU. For a fail-fast — none (it only raises).
  Existing tests that would catch a regression today: none (§2).
- A4 (who else): ChaoyuWang04 (#7496), HollowMan6 (#8032; LoRA owner), aoshen02 (#8105, #8106, #8246),
  Levius-Fubuki (#8307, VeOmni LoRA).
- A5 (half scope): the 8-line fail-fast + CPU test, or simply the #7495 comment and a review of #7496.

## 7. Gates, scores, verdict

| gate | result | evidence |
|---|---|---|
| G1 not a duplicate | **FAIL** | #7496 implements the same fix; head commit 69 d old (stale by the 60-d rule) but the author is responsive and offered the assert variant himself |
| G2 not reserved | pass (with a caveat) | nobody claimed it; the owners' recorded view is that the mode is unneeded, which is worse than reserved |
| G3 self-contained | pass | 2 source files, CPU-testable |
| G4 provable | pass | CPU regression test; GPU `list_loras()` + KL floor |
| G5 default-path impact | weak | only LoRA + `merge=False` + non-naive backend; off the reference user's path |

Scores: S1 **2** (a silent correctness failure of one configuration; no speed value for the
8×H100 / 8B / GRPO reference user; high value only for disaggregated/edge LoRA, i.e. E1), S2 **2**
(one issue, lukewarm maintainers), S3 **5** (reproduced on CPU through the real functions and
independently observed on GPU by the reporter), S4 **1** (duplicate; owners' stance), S5 **1**.

**Verdict: NO-GO as a top-3 contribution. Bug claim: CONFIRMED.** Strongest reason: the fix is an
open PR (#7496) whose author offered the fail-fast too, and the maintainers' recorded position is that
async LoRA is not needed — a second PR cannot be justified under AGENTS.md.

Implications for the rest of the campaign: (1) E1 / E2 (edge LoRA refresh, multi-adapter) presuppose
exactly the path wuxibin89 and HollowMan6 called unnecessary — frame them as "adapter-only refresh
for disaggregated rollouts", start by helping #7496 land (rebase, review, GPU evidence on
`separate_async`), and cite #7495; (2) any disaggregated LoRA measurement on Modal against main
silently measures the base policy — rebase on #7496 first; (3) `merge=True` is the only correct
disaggregated LoRA mode on main today and it pays the full export every sync.
