# P2 round-2 re-review verdict (AI reviewer, 2026-10-29)

Branch contrib/p2-oldlogprob-skip @ ba60cf8e (13 commits on main @ 2ab35292), worktree wt/p2, read-only; diff base main.
Probes re-run from a scratch copy with PYTHONPATH=wt/p2 (HF_HUB_OFFLINE=1, .venv-cpu). Nothing on the branch or in the
package was modified other than review-tests/round2/ and this file; wt/p2 `git status --porcelain` is empty before and after.

## the submitter's findings (research/review-bundle/REVIEW.md, "P2 accepts actors whose training forward differs ...")

| finding | status | evidence |
|---|---|---|
| override_config dropout accepted (gap 0.153) | FIXED | `p2_dropout_override.py` now prints `Validator sees dropout: [('attention_dropout', 0.5)]` and dies in `_validate_skip_old_log_prob_config` with `ValueError: ... requires a strictly on-policy update, but: the effective actor model (HF config after override_config, LoRA adapter) has train-mode-only settings (attention_dropout=0.5): ...`. Spellings checked against the real `HFModelConfig`: `model_config` sub-key, nested `text_config.attention_dropout` (both find `("attention_dropout", 0.5)`; `hf_config.text_config.attention_dropout == 0.5` on the real config), `hf_config_path` != `path` (the hf_config_path one is read, as the engine does). Bool/str rates never reach a model: transformers 5.16's strict config validation raises inside `HFModelConfig.__post_init__`. |
| loaded LoRA adapter lora_dropout=0.3 accepted (gap 0.065) | FIXED for LoRA-type adapters | `p2_train_eval_gaps.py` (its own `assert dropouts == []` replaced by a print so the case reaches the validator): `detected: [('lora_dropout', 0.3)]` then `ValueError: ... (lora_dropout=0.3)`; the walk finds `Dropout(p=0.3)` on the module built by the real `_build_lora_module`. The `lora_rank > 0` gate matches the engine's `_is_lora`. Other peft adapter types: new defect D1. |
| Mixtral router_jitter_noise=0.1 accepted (gap 0.00083) | FIXED | verbatim probe exits at `AssertionError: [('router_jitter_noise', 0.1)]`; through the validator: `ValueError: ... (router_jitter_noise=0.1)`. Systematic check: every `self.training` site in the modeling files of 40 transformers families verl can train (qwen2/3/moe/vl, llama/4, mistral/mixtral, gemma2/3, deepseek v2/v3, glm4/v/moe, gpt_oss, phi3/phimoe, minimax, dbrx, olmoe, granitemoe, jamba, seed_oss, siglip/clip/internvl, ...) is gated by a rate whose name carries one of the scanned fragments. PhiMoE's `sparsemixer` (Gumbel expert sampling in train mode) is exact at zero jitter (max gap 0.0 over 3 seeds) and rejected at `router_jitter_noise=0.01`. Mamba2/Zamba2 pick a different (deterministic) CUDA kernel path in train mode: not stochastic, no verl FSDP path, noted only. |
| warning-only when the config cannot be loaded vs the rejection guarantee | FIXED (design), wording D2 | `_check_actor_for_on_policy_skip` runs in `init_model` (ONE_TO_ALL, every rank) right after `self.actor.reset()` and before `set_loss_fn`; every v1 mode maps the actor role to `ActorRolloutRefWorker` (trainer_base.py:815) and `DetachActorWorker` inherits `init_model` unchanged; `separate_async` and non-FSDP strategies are rejected earlier. On CPU the walk sees through FSDP2 `fully_shard` (class swap to `FSDPQwen3ForCausalLM`) and `checkpoint_wrapper` (`_checkpoint_wrapped_module.self_attn`). The walk is structural, not a guarantee: D2. |
| "adding names to the dropout list alone does not fix (1)/(2)" | FIXED | two layers as described; the warning names the worker enforcement; review-tests' `_hf_config_dropouts` contract kept (39 passed / 1 skipped re-run). |
| bench wording: stale final-sha paragraph, "paired same-node" | FIXED | RESULTS.md: pooled-means section, "exploratory: similar windows do not prove the same physical node", round-2 commits stated as not run on GPU; PR-BODY quotes -5.3 % / -6.6 % pooled with pairwise readings labelled exploratory. |

## New defects

- D1 MEDIUM (guarantee gap): a peft **LoHa/LoKr** adapter at `lora_adapter_path` with `lora_rank > 0`. `_build_lora_module` loads
  any `peft_type` via `PeftModel.from_pretrained`; its train-mode rates are `rank_dropout` / `module_dropout`, which the validator
  does not read (only `lora_dropout`) and which peft stores as a dict `{adapter: rate}` on the layer, which the walk's
  `_is_positive_rate` ignores. Measured through the real loader: `_hf_config_dropouts == []`, walk `== []`, train/eval gap 1.9e-4.
  Reachability bound: the rollout LoRA sync (vLLM `TensorLoRARequest`, SGLang's `peft_type` check) expects LoRA tensors, so an
  end-to-end run most likely fails at the first weight sync rather than training inexactly. Fix: reject `peft_type != "LORA"` in
  `_lora_adapter_dropout` and let the walk flag dict/list-valued rates > 0 (or peft `LycorisLayer`).
- D2 LOW (wording): PR-BODY.md:38 and perf_tuning.rst "the module walk is the guarantee" overclaims. The walk sees named rates
  and dropout-family modules; code-level dependence (hard-coded `F.dropout(p)`, a rate read from `self.config` in forward,
  BatchNorm running stats) is invisible to it, as the `documents_` test shows. Say: the walk enforces the rule for the mechanisms
  it names; a custom (trust_remote_code) forward with its own train-mode branch is outside it.
- D3 LOW: `_BatchNorm` / `InstanceNorm` with `track_running_stats` are not flagged (train and eval outputs differ by
  construction). No LM verl trains carries one (Emu3's BatchNorm is in its VQ-VAE tokenizer); one `isinstance` line closes it.
- D4 LOW (stale text): CHECKLIST.md:13 and bench/RESULTS.md:103 still call 2b259b2d the "final HEAD"; HEAD is ba60cf8e
  (PR-BODY.md:172 says it correctly).
- Noted, out of round-2 scope: `main_ppo_v0.py` run directly with the default `trainer.use_v1=true` passes the validator while the
  legacy RayPPOTrainer ignores the flag (silent no-op, not inexact).

Checked, no defect: `use_shm` (copy_to_local fills the same local cache before the shm copy, no second download on the
driver); the ValueError names each probe's setting exactly; engine `_is_lora` == validator's `lora_rank > 0` gate.

## Tests added (packages/p2/review-tests/round2/, run from wt/p2 with the CPU harness)

`test_r2_effective_model_on_cpu.py`: 7 tests, **3 failed, 4 passed**. Failing = findings: `test_loha_adapter_rank_dropout_is_rejected_by_the_validator`,
`test_loha_adapter_rank_dropout_is_found_by_the_module_walk` (D1), `test_batchnorm_with_running_stats_is_found` (D3). Passing:
hf_config_path + nested text_config override, FSDP2 + checkpoint_wrapper walk, PhiMoE zero-jitter exact / 0.01 rejected,
`test_documents_that_code_level_train_mode_dependence_is_outside_both_layers` (pins D2's limit).

## Hygiene

`scripts/package-audit.sh`: `p2 contrib/p2-oldlogprob-skip ba60cf8e 13 0 fresh ok none`; worktree clean. tests.log reproduced:
author's two round-2 files + review-tests = 52 passed / 1 skipped (13 + 39/1); research/logs/p2r2-reg-*.log summaries sum to
1556 passed / 64 skipped as logged. PR-BODY: AI statement verbatim per the rules, 503-suite labelled as stock regression
coverage, no "already reviewed" / upstream-contact claim, conservative pooled numbers. CHECKLIST truthful except D4.

## Call: NEEDS-FIXES (small)

The submitter's three cases are rejected by name on every v1 path and the deferred worker check exists, but the claimed "guarantee" has a
measured hole for non-LoRA peft adapters (D1) and the wording overstates what a structural walk can see (D2); both are a few
lines plus D3/D4 text. After those, MERGE-READY on the validation side (GPU evidence for the round-2 commits is still none).

## Lead closure (2026-10-29, after ba0dccf5)
D1-D4 applied by the author in c7031289 + ba0dccf5: non-LoRA peft adapters rejected by `peft_type` and every rate
key of adapter_config.json scanned; dict/list-valued rate attributes expanded in the module walk; `_NormBase`
modules with `track_running_stats=True` flagged; the "guarantee" wording replaced by "structurally visible"; stale
HEAD text fixed. The reviewer's three failing round-2 tests now pass (review-tests rounds 1+2 + round2/: 46 passed /
1 skipped against wt/p2 @ ba0dccf5). Verdict moves to MERGE-READY on the validation side; the stock GPU suite on
ba0dccf5 is queued (regression coverage only).
Addendum: 7f0c53f5 closes the reviewer's out-of-scope note (the legacy RayPPOTrainer now refuses the flag with a
ValueError naming trainer.use_v1; test tests/trainer/ppo/test_skip_old_log_prob_legacy_trainer_on_cpu.py). Final P2
HEAD for this round: 7f0c53f5 (16 commits).
