# P1 build notes: `[fsdp, perf] feat: project the LM head only for loss rows`

Branch `contrib/p1-lmhead-rows` in worktree `<repo>/wt/p1`, base main @ 2ab35292.
Author: Claude Fable 5.1 (AI); the submitter reviews every line before anything leaves this machine.
All file:line references are the worktree at the base commit unless marked "(new)".

## Session log

- 23:20Z worktree created; `PYTHONPATH=wt/p1 .venv-cpu/bin/python -c "import verl"` prints the worktree path (the
  editable install in `.venv-cpu` points at the main clone; PYTHONPATH wins, no second venv needed).
- 23:25Z read the engine: `prepare_model_inputs` :1155-1342, `prepare_model_outputs` :1344-1568, `forward_step`
  :1570-1614, value head :1616-1660, build-time probe :592-600, `_gather_and_unpad_packed` :684-693.

## Design decisions (with evidence)

1. **Where the index is built.** `prepare_model_inputs` rmpad branch, after the static pad (:1214-1227) and the SP
   block (:1229-1256), where `model_inputs` is assembled (:1268-1272). The static pad is a suffix of the global
   packed sequence (:1217-1219 comment), so loss-row indices computed on the unpadded layout stay valid; SP is
   guarded off at build time, so no rank offset is needed in v1.
2. **Which kwarg.** `signature(module.forward)` probe at build time next to `pass_packed_cu_seqlens` (:596-600),
   on the raw HF module before PEFT/FSDP wrap. Probed both pins: transformers 5.16.1 (`.venv-cpu`) and 5.21.0
   (`.venv-a9`) expose `logits_to_keep: int | torch.Tensor` on Qwen3/Qwen2/Llama `forward` (5.21 routes it
   through `modeling_utils.slice_logits_to_keep`, modeling_qwen3.py:480). The probe accepts `logits_to_keep`
   or `slice_logits_to_keep` [superseded by review item D4: the alias was dropped in bc0e081f; only
   `logits_to_keep` is probed]; a bare `**kwargs` does not count (it would swallow the index and project every
   row, then the K-row output path would see T rows). Neither name -> one warning, full projection.
3. **Build-time guards (one warning each, flag resolved to False):** value-head engine (class attribute; the critic
   reads `output.logits` of a `num_labels=1` head, :1629-1636: nothing to gain), `use_remove_padding=False`
   (the padded branch :1289-1330 has no packed rows), Ulysses SP > 1 (`self.use_ulysses_sp`, :256; out of v1
   scope per the vet §4), `strategy == "fsdp_turbo"` (TP may shard the head as a DTensor, :1401-1403; untested),
   vision-language models (`config.vision_config`; their rmpad forward `forward_with_normal_backend` projects every
   row: qwen2_vl.py:482, qwen3_vl.py:327-336, and the CPU proof covers text Qwen3 only), no accepted kwarg.
4. **Per-micro-batch guards (silent no-op, same batch may differ per call):** `use_fused_kernels` (fused forwards
   ignore `logits_to_keep`, dense_common.py:84,152; #7370's territory), `distillation_use_topk` / `distillation_only`
   / `score_centering` (the logits processor reads all rows: :1430-1439 `assert v.shape == (logits_rmpad.shape[0],)`),
   no `responses` key (SFT batches read `loss_mask` rows, losses.py:28-44; the SFT dataset never emits
   `responses`: grep of verl/utils/dataset/ and sft_trainer.py), empty index (0-row GEMM/CE untested).
5. **`sum_pi_squared` stays eligible** (deviation from the brief's guard list, which conditioned the guard on
   "paths that read all rows"): `calculate_sum_pi_squared_from_logits` is a per-row reduction over the vocab
   (torch_functional.py:687-699) and its only consumer slices response rows with `no_padding_2_padding`
   (ray_trainer.py:1309-1322). It is scattered back exactly like entropy and covered by the identity test.
6. **Output path.** In the eager rmpad branch (:1400-1449): index `input_ids_rmpad_rolled` / `temperature_rmpad`
   by the kept rows, compute Σπ² / entropy / log-probs on K rows, then `index_copy` each into zeros of the local
   packed length before `_gather_and_unpad_packed` (:1452-1458), so every downstream consumer is unchanged
   (`no_padding_2_padding` padding.py:138, `response_from_nested` padding.py:209, trainer_base.py:1668-1739).
   `index_copy` with unique indices is deterministic and differentiable (gradient = gather).
7. **Row count check.** If `output.logits` has T rows while an index was passed (a wrapper swallowed the kwarg), raise
   a RuntimeError naming the model class rather than silently mis-scoring rows.
8. **Config.** `response_only_lm_head: bool = False` on `FSDPEngineConfig` (engine.py:233-300), key + comment in
   `engine/fsdp.yaml` next to `pad_to_length` (:63-73), and `ref/dp_ref.yaml` follows the actor through
   `${oc.select:actor_rollout_ref.actor.fsdp_config.response_only_lm_head,False}` (pattern of
   `pad_to_length_bucket`, dp_ref.yaml:35). Same key name as the Megatron twin #7686.
9. **Host syncs.** `loss_row_index` does `int(response_lens.sum())` (one sync) to size the index; the engine already
   syncs in this function (`int(model_cu_seqlens[-1].item())` :1276 under pad, `max_seq_len` :1296).
10. **Legacy `dp_actor.py` worker** is out of scope: the key lives on the engine config it also receives but it never
    reads it (no behaviour change there); documented in the docs paragraph as "engine-based FSDP workers".

## Deviations from the vet / brief

- D1: `sum_pi_squared` supported on the reduced rows (see decision 5) instead of guarded off.
- D2: `distillation_only` added to the per-batch guard set (conservative; distillation needs teacher-aligned layouts).
- D3: no `tests/models/test_fsdp_no_padding_on_gpu.py` case: cannot be run here; left for the benchmark phase.

## Open questions for the reviewer

- Q1: bitwise identity on x86 CI for the forward quantities: inner-dim reductions are identical for any M; the test
  asserts `torch.equal` on log-probs/entropy/Σπ² and `assert_close(atol=1e-6)` on gradients (row-reduction
  blocking may differ with M). Observed max diff on this Mac: 0.0 everywhere.
- Q2: default-on once the GPU A/B shows max |dlogp| within bf16 noise (vet §4), or keep opt-in.

## Session log (continued)

- 23:30Z config commit cc782aa9: dataclass + fsdp.yaml + dp_ref.yaml (follows the actor) + regenerated
  `_generated_ppo_trainer.yaml` (+3 lines: actor :48, ref :257 `oc.select`, critic :583). Hydra check: actor
  override propagates to the ref; ref can be set alone; defaults False.
- 23:34Z engine smoke run (temporary directory `smoke_p1.py`, real `prepare_model_inputs`/`outputs`, flag on vs off):
  rows 105 -> 32 and 128 -> 32 (pad bucket 64); log-probs, entropy, Σπ², nested consumer, head and
  embedding gradients all `torch.equal`; max |grad diff| 0.0.
- 23:38Z engine commit 3d611158 (transformer_impl.py +115/-1, padding.py +31). ruff + ruff-format clean.
- 23:45Z tests commit 4775e3f9 (`tests/workers/test_response_only_lm_head_on_cpu.py`, 23 tests, 11 s);
  docs commit 4bcbbeb9 (`docs/perf/perf_tuning.rst` item 11 + section, Last updated 10/28/2026).
- Local sanity hooks (`pre-commit run --all-files <id>`): check-license, check-docstrings, validate-structure,
  check-device-api-usage, check-naming-conventions, check-docs-time-info, autogen-trainer-cfg: all Passed.
  PR title check: valid (modules fsdp, perf; type feat).
- `docs/examples/config.rst` does not list the individual `fsdp_config` keys (only example blocks), so the
  key is documented in `perf_tuning.rst` only.

## Transformers compatibility note (for the PR body)

Both pins were probed in situ: 5.16.1 (`.venv-cpu`, verl's pin) and 5.21.0 (`.venv-a9`) keep
`logits_to_keep: int | torch.Tensor` on `Qwen3ForCausalLM.forward` / Llama / Qwen2; 5.21 implements the
slice through `modeling_utils.slice_logits_to_keep(hidden_states, logits_to_keep)` (modeling_qwen3.py:480),
so the kwarg name did not change for these models. [Superseded by D4: the probe no longer guesses other
spellings; a renamed kwarg falls back to the full projection with one warning, and the row-count RuntimeError
catches a wrapper that swallows it.]

## Duplicate check refresh (gh, read-only, 2026-10-28 ~23:55Z)

- #7370 CLOSED (2026-09-20, +159/-36, REVIEW_REQUIRED, never reviewed): fused-torch path only.
- #7686 OPEN, CONFLICTING, REVIEW_REQUIRED, last update 2026-10-12, +533/-43: Megatron only, same key name.
- #7712 is an issue (RFC), not a PR (`gh pr view 7712` cannot resolve it): orthogonal, more outputs not fewer rows.
- #8300 OPEN, MERGEABLE, REVIEW_REQUIRED, last update 2026-10-22, +1968/-75: prefix packing, composes.
- `gh pr list --search logits_to_keep` open: none; merged: none. Unchanged from the vet (§3).

## CPU regression against the worktree (2026-10-28 23:50Z, see tests.log)

p1-new 23 passed; reg-small 200; reg-workers 301 (= 278 baseline + 23 new), 40 skipped, 1 deselected; reg-utils
473, 23 skipped; reg-trainer 438, 1 skipped. All exit 0, counts identical to the main baseline. Raw logs + junit in
packages/p1/logs/.

## Package state

PR-TITLE.txt, PR-BODY.md (template filled; `### Benchmark` left as TBD placeholders for the benchmark phase),
branch.txt, diff.patch (`git diff main...HEAD`, 8 files, +529/-2; product code 170 lines, test 330, docs 31),
tests.log, CHECKLIST.md, logs/. Not produced in this phase: bench/ (benchmark phase) and REVIEW.md (review phase).

## Not verified here

- Anything on a GPU: bf16 rounding of the K-row GEMM (expected <= 1e-3 max |dlogp|), flash-attn Triton CE on K
  rows with `inplace_backward`, FSDP1/FSDP2 wrapping passing the kwarg through (PEFT does, peft_model.py:1958-2001;
  FSDP wraps `forward(*args, **kwargs)`), the `tests/models/test_fsdp_no_padding_on_gpu.py` pattern, timing and
  peak memory. The build-time resolver's wiring inside `_build_model_optimizer` runs only with a process group
  (not exercised by the bare-instance tests; the method itself is tested for every branch).

## Review fixes (2026-10-29 00:10-00:35Z, REVIEW.md verdict NEEDS-FIXES, no product defect)

Commits added (history kept, same trailer): 6f6fc85a (D1), bc0e081f (D2, D4, D6, D7, D10), 80aaef48 (D5),
4d36e7d2 (D2 wording, D9), b32c777e (ruff format). Mapping with evidence: REVIEW.md §7.

- D2 design: the probe alone cannot tell the fused forwards apart (`dense_common.py:84,152` list `logits_to_keep`
  and ignore it), so the resolver now reads `self.model_config.use_fused_kernels` (the same field `_build_module`
  passes to `apply_monkey_patch`, transformer_impl.py:355) and resolves False with the warning. The per-batch
  `use_fused_kernels` check stays as a backstop (the batch flag is the same config value, engine_workers.py:148).
- D10 ordering: value head first (keeps ignoring the flag with a warning, whatever its padding setting), then the
  `use_remove_padding=False` `ValueError`, then the warned layouts. Warnings gated on `getattr(self, "rank", 0) == 0`
  (`self.rank` is set in `__init__` :164; bare test instances default to rank 0).
- D5: `data["responses"].values().shape[0]` is the sum of the response lengths by construction of a jagged nested
  tensor; the empty-prompt assert moved to the consumer side only (`no_padding_2_padding`, padding.py:126), which
  is what the update path runs; the forward-only path (`response_from_nested`) never asserted it on main either.
- D4: kept the attribute name `logits_to_keep_kwarg` (value `"logits_to_keep"` or `None`) so the reviewer's
  fixtures and the probe-order evidence stay valid; only the alias and its wording went.
- D8: commit 4bcbbeb9 `[doc] docs:` uses a type outside feat/fix/refactor/chore/test. Not rewritten (lead's
  instruction); the squash-merge title `[fsdp, perf] feat: ...` is what `check_pr_title.py` enforces. All later
  commits use fix/chore.
- D3 (PR body): tensor-valued `logits_to_keep` arrived in transformers 4.49 (4.45 had `num_logits_to_keep: int`);
  verl pins `transformers>=5.5.3,!=5.6.0,<5.17` (requirements.txt:20) and 5.16.1 project-wide (pyproject.toml:100).

Disagreements: none on substance. Two of the reviewer's 20 tests still fail after the fixes because they encode the
pre-fix behaviour (bitwise-gradient assertion with rtol=0; the dropped empty-prompt assert); left untouched and
reported in REVIEW.md §7 and tests.log §5.

Verified after the fixes: new test 23 passed; workers 301 / utils 473 unchanged; `pre-commit run --all-files`
end to end 14/14 Passed on the second run (the first run's only failures were two long lines in the new test
file, fixed by the hook and committed). diff.patch regenerated (8 files, +546/-2; product code 171 lines).

## Re-verification nits (2026-10-29 ~00:50Z, REVIEW.md §8: MERGE-READY-PENDING-GPU)

Commit a11ca00b `[fsdp] fix: name the empty-prompt error in loss_row_index` (R1 + R2, one commit as asked):
- `loss_row_index` raises `ValueError("... sequence {i} has prompt length {p} and response length {r} (sequence
  lengths [...], response lengths [...])")` when any prompt is empty. Trade-off, stated in the docstring and the
  commit: the named check costs one device-to-host transfer per micro-batch when the flag is on (`int(prompt_lens
  .min())`; the lists are fetched only on failure). The reviewer's R1 said no sync-free form exists; the
  lead chose the named error over the device-side assert. The engine already syncs here under
  `pad_to_length` (transformer_impl.py `int(model_cu_seqlens[-1].item())`).
- Resolver docstring: the value-head engine ignores the flag with a warning before the `use_remove_padding` check.
- New CPU test `test_an_empty_prompt_is_rejected_by_name` (sequence 1 with prompt 0 / response 2).

Re-runs @ a11ca00b: new test file 24 passed; workers 302 passed / 40 skipped / 1 deselected (301 + 1 new); utils 473
passed / 23 skipped; ruff, ruff-format, mypy on the three changed files Passed. Reviewer's set (33 tests): 28 passed,
5 failed, all by construction of the tests: `test_no_device_to_host_sync_in_the_index_path` (the named check needs
one transfer), `test_empty_prompt_in_the_first_row` and `test_empty_prompt_in_a_middle_row_reads_the_same_wrong_row_
as_the_full_path` (characterise the pre-nit IndexError / silent read; both inputs now raise the ValueError),
`test_prompt_length_zero_is_rejected_not_misread` (expects AssertionError; it is now rejected with ValueError),
`test_single_sequence_single_response_token` (rtol=0 gradient assertion, unchanged since §7).
HEAD a11ca00b; diff.patch +564/-2 over 8 files; branch.txt records the head.

## Benchmark section filled (2026-10-29, no commit: PR-BODY.md / CHECKLIST.md live outside the worktree)

Sources: `bench/CROSSCHECK.md` (independent cross-check from `bench/results/*/summary.json`) and
`packages/p1/bench/RESULTS.md` (benchmark agent). Every number in the PR body comes from both; reconciliations:
- Training-side gain on `long-prompt`: 40.52 -> 35.68 s = -11.9 % to one decimal (RESULTS.md); CROSSCHECK.md writes
  -12.0 % for the same means. The body states -11.9 % and names the rounding.
- GPU suite: 503 passed / 0 failed / 8 skipped (pytest's console line over 511 test cases) + torchrun `2 passed on 2
  GPUs!` x2, identical on main and p1 (`bench/results/gpu-tests-{main-2,p1-1,p1-2}/pytest.log`). Lead's
  correction: CROSSCHECK.md's "521" came from the junit suite attribute (529 incl. 18 subtests) and is superseded; the
  body and CHECKLIST use 503/0/8. No run at a11ca00b (latest is b32c777e; a11ca00b adds one commit).
- GPU type: `meta.json` of every long-prompt arm and of baseline-a-p1-on-2 says `"gpu": "H100:8"`; the README's preset
  table row for `long-prompt` still says "H200 x8" (the H200 launches never got a GPU and were stopped in the queue,
  RESULTS.md app list). The body says 8x H100.
- K/T: the README preset row says ~0.14 (nominal 1024/7168); RESULTS.md measured 0.096 (prompt 5999, response ~636
  under the 1024 cap). The body uses the measured value.
- Peak memory comes from RESULTS.md only (the cross-check has no memory column): -4.65 GB/rank on the 2-run means,
  -4.67 GB ON vs OFF on the same tree.
Not measured and said so in the body and CHECKLIST.md: bf16 max |dlogp| identity on GPU, models > 1.7B, MoE /
remote-code, Ulysses SP, torch.compile recompiles under pad_to_length, NPU; no dedicated GPU unit test.

## Review round 2 (the submitter's independent review, 2026-10-29 ~13:00-14:30Z; author: Claude Fable 5.1)

the AI review (research/review-bundle/REVIEW.md "P1 performance is promising but numerical equivalence still needs a
direct check", FIX-CHECKLIST.md "P1 loss row LM head"): no correctness defect in the final source; the 24 tests pass
independently; ~12 % training-phase time and -4.65 GB/rank are credible for the long-prompt workload. Asked for:
(1) a frozen-batch, same-weights bf16 GPU flag-on/off comparison of response log-probs, entropy, Σπ², gradients and
one optimizer update, with entropy-loss, PEFT and checkpointing cases and justified tolerances; (2) the "no
device-to-host sync" claim in PR-BODY.md is wrong (`int(prompt_lens.min())` once per micro-batch); (3) measured
response length / K:T for steps 2-6 are ~657 / 0.099, not 636 / 0.096; (4) training-phase time and memory reported
separately from whole-step deltas; (5) keep opt-in.

Commits added (history kept, same trailer): 724d3d5e, 2634019d, d7d820aa.

- PEFT investigation. `_build_model_optimizer` (transformer_impl.py:654-664) probes the module returned by
  `_build_module()` BEFORE `_build_lora_module`, so under LoRA the probe sees the raw HF model and the flag survives
  build time. The index reaches the base model: PEFT 0.19.1 `PeftModelForCausalLM.forward(input_ids=..., ...,
  task_ids=None, **kwargs)` calls `self.base_model(..., **kwargs)` and `LoraModel.forward(*args, **kwargs)` calls the
  HF forward (probed in .venv-cpu). Had the probe run on the wrapper it would have returned None (no named
  `logits_to_keep` there) and silently turned the flag off, so 724d3d5e makes `_probe_logits_to_keep_kwarg` unwrap a
  `PeftModel` through `get_base_model()`: the answer no longer depends on the build order. CPU tests:
  `test_probe_and_resolver_see_the_base_model_through_a_peft_wrapper`,
  `test_the_index_passes_through_a_peft_wrapper_to_the_base_model` (32 of 105 rows projected, log-probs bitwise,
  LoRA gradients within 1e-5 / 1e-6). The file has 26 tests now.
- GPU identity test, 2634019d + d7d820aa: `tests/workers/test_response_only_lm_head_on_gpu.py` (278 lines; the ~250
  budget is spent on the docstring that justifies the tolerances). Pattern of tests/models/test_engine.py: `mp.spawn`
  one worker, `dist.init_process_group` over a rendezvous file, `EngineRegistry.new(backend=strategy)` with
  `HFModelConfig(path=~/models/Qwen/Qwen2.5-0.5B, use_remove_padding=True, load_tokenizer=False)` and
  `FSDPEngineConfig(strategy, fsdp_size=1, response_only_lm_head=flag)`, default mixed precision (fp32 master, bf16
  compute), AdamW lr 1e-3, clip 1.0. Frozen batch: six packed sequences, prompts 1/300/77/520/33/160 (the 1-token
  prompt is the guard's edge), responses 40/128/9/64/1/256 (T=1589, K=498), seeded ids and advantages,
  `old_log_probs` = the first flag-off forward-only log-probs (ratio 1 in the PPO loss), `calculate_entropy` +
  `calculate_sum_pi_squared` on the batch, `ActorConfig(entropy_coeff=0.01)` through `partial(ppo_loss, config=...)`.
  Per arm: forward-only pass, training pass with `return_model_output=True`, snapshot of every gradient and of every
  trainable parameter before and after `optimizer_step` (DTensor -> `to_local()`; FSDP1 flat params compare as flat
  params). Determinism baseline: a second flag-off engine must equal the first bitwise; `FLASH_ATTENTION_DETERMINISTIC=1`
  is set at import (the spawned child inherits it) so flash-attn's backward avoids atomics. Asserts per the brief:
  `rtol=atol=1e-2` on log-probs / entropy / Σπ² / loss; rel-Frobenius < 1e-2 plus element-wise
  `assert_close(rtol=1e-2, atol=1e-2 * max|ref|)` on gradients and parameters; the parameter *update* is printed
  only (AdamW's first step is ~lr * sign(g), so a sign flip at the noise floor is a 2 * lr delta, not an identity
  failure). Prints `[response_only_lm_head][strategy/variant] key: max_abs=... max_rel=...` plus per-group pooled
  rel_fro / worst tensor, so the Modal log records the deltas. Cases: {plain, gradient-checkpointing, lora rank 8} x
  {fsdp (FSDP1: verl's default and the bench's strategy), fsdp2} = 6 tests, one GPU each. On this Mac: 6 skipped (no
  CUDA). CPU smoke of the helpers: the batch, `loss_row_index` on it, the old_log_probs freeze, `_check` passing on
  identical dicts and failing on a 5 % gradient perturbation. Not under any harness ignore glob (`on_gpu`, under
  tests/workers); gpu_unit_tests.yml collects it too.
- Bench command (bench agent; Modal, verl CI image, models linked at /root/models by `_link_models`):
  `python3 bench/verl_bench.py gpu-tests --src <repo>/wt/p1 --run-name gpu-identity-p1-1
  --tests tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs" --no-torchrun` (default
  `--gpu H100:8`; one GPU suffices: `--gpu H100:1`). Expected: 6 passed; the deltas are the `[response_only_lm_head]`
  lines in results/gpu-identity-p1-1/pytest.log. Follow-up after the run: one commit tightening the tolerances to ~4x
  the measured deltas, and the deltas pasted into PR-BODY.md "Test".
- Documentation (PR-BODY.md only; the docs page never made the corrected claims): sync claim corrected (the index is
  sync-free; the empty-prompt guard is one `.item()` per micro-batch, negligible next to the forward); 657 / 0.099
  recomputed from bench/results/long-prompt-{main-1,main-2,p1-on-2,p1-on-3}/metrics.jsonl, steps 2..6,
  `data["response_length/mean"]`: 660.10 / 652.88 / 651.94 / 664.02 -> pooled 657.24, `prompt_length/mean` 5998.78,
  K/T = 657.24 / 6656.02 = 0.0987 (the OFF run 656.12 / 0.0986; all five runs 657.01 / 0.0987); training phases and
  peak memory are the claim, `gen` and step listed separately and not claimed (generation varies ~10 % between
  identical runs); opt-in kept with the reason (identity test unrun; the gain scales with the prompt fraction); GPU
  unit suite labelled per the rules; AI statement per the rules; BASELINE-A labelled exploratory; narrative cut,
  tables kept. Duplicate check refreshed 2026-10-29 13:19Z (read-only gh): nothing new on the eager FSDP path; #7686
  OPEN / CONFLICTING (2026-10-12), #8300 OPEN / MERGEABLE (2026-10-22); the new LM-head PRs #8441 (10-29), #8312,
  #8096, #8097, #8004 are fused-kernel / FSDP2-hook / FP32-head work on other paths; named in the body.
- Not changed, and why: the one host sync in `loss_row_index` stays (round-1 R1 chose the named ValueError over a
  device-side assert; one `.item()` per micro-batch next to a forward is negligible, and the engine already syncs
  there under `pad_to_length`): described correctly rather than removed. docs/perf/perf_tuning.rst untouched (it
  never claimed sync-free or quoted 636 / 0.096). No FSDP-wrapping change for PEFT: the kwarg already passes through.
  scripts/run-cpu-regression.sh takes group names, not a worktree path, so its six groups were run through
  scripts/run-cpu-tests-wt.sh with VERL_ROOT=wt/p1 (logs/p1r2-reg-*.log), same ignores and deselect.

Verification at 2634019d: new file 26 passed; regression 1510 passed / 64 skipped / 1 deselected / 0 failed (small
200, utils 473+23, trainer 438+1, workers 304+40, experimental 74, single_controller 21; main baseline 1484 / 64 plus
the 26 new); `pre-commit run --all-files` 14 hooks Passed. d7d820aa changes only the GPU test file (6 skipped on CPU;
`pre-commit run --files` Passed).

Open for the submitter: the bench run of the GPU identity test and the tolerance-tightening follow-up; whether to run the
identity on Qwen3-1.7B as well (`VERL_RESPONSE_ONLY_LM_HEAD_MODEL=~/models/Qwen/Qwen3-1.7B`, same file).

## Round-2 re-review fixes (verdict NEEDS-FIXES on the GPU test design only; 2026-10-29 ~14:30-15:15Z)

Source: packages/p1/REVIEW-ROUND2-VERDICT.md (product code, PEFT probe and the doc corrections confirmed FIXED;
D1-D3 on `tests/workers/test_response_only_lm_head_on_gpu.py`), measured by the reviewer on CPU through my own
`_check` (review-tests/round2/test_r2_gpu_check_mechanics_on_cpu.py: 2 pass / 4 fail by design at d7d820aa).

One commit, 407082ec `[fsdp] test: linear optimizer step and ulp-justified tolerances in the GPU identity test`:
- D1 (param/ after AdamW): AdamW's first step is ~lr * sign(g); `lora_B` starts at zero, so its post-step value IS the
  update and one noise-floor sign flip is a 2 * lr = 2e-3 delta against atol 1e-5 (same for FSDP2 weights with
  max|w| < 0.2). Fix: the engine's optimizer is configurable (`build_optimizer` imports any `torch.optim` class; betas
  only for Adam-family), so the step is plain SGD: `FSDPOptimizerConfig(optimizer="SGD", lr=1e-3, weight_decay=0.0,
  clip_grad=1e9)`: the update is exactly -lr * grad (no momentum, no decay, no clipping: `optimizer_step` asserts
  `clip_grad is not None`, 1e9 never clips). `update/` gets the gradient rule (rel-Frobenius < 1e-2, element-wise
  rtol 1e-2 / atol 1e-2 * max|ref|) plus sign agreement >= 99 % over non-zero elements (printed as "min sign
  agreement"); `param/` is bounded element-wise by 2 * lr * 1.01 + 1e-2 * |ref| (two opposite first-step updates; no
  rel-Frobenius on `param/`, which is ill-posed for `lora_B`). Stated in the docstring.
- D2 (entropy / Σπ² atol 1e-2 < one bf16 ulp of a top logit; reviewer: dH 0.029, dΣπ² 0.0196 for logit 20 one ulp up):
  per response row rtol 2e-2 / atol 5e-2 on log-probs, entropy and Σπ²; the batch mean (over positions that are
  non-zero in either arm, i.e. excluding the padding) must agree within (atol + rtol * mean|ref|) / sqrt(rows):
  full-row deltas on at most sqrt(rows) rows (~2.6e-3 for entropy at 498 rows); the PPO loss (a mean) rtol 1e-2 /
  atol 5e-3. The random labels keep |log-prob| large, so the rtol term also covers a one-ulp flip of the label's logit.
- D3 (`gpu_unit_tests.yml` runs `pytest -s -x`): a tolerance failure aborts verl's whole GPU job, so the tolerances stay
  conservative and every delta is printed; noted in the docstring and the PR body. Model path: `MODEL_CANDIDATES` =
  [`$VERL_RESPONSE_ONLY_LM_HEAD_MODEL`, `~/models/Qwen/Qwen2.5-0.5B`, `/root/models/Qwen/Qwen2.5-0.5B`], the first
  existing directory wins (in the harness HOME=/root so both spellings are the same link that `_link_models` makes);
  the skipif reason names every candidate and the test prints the resolved path, strategy and variant; run with
  `-s -rs` and require `skipped == 0` in the harness summary.json.

Verification: reviewer's round2 tests 13/13 (mechanics 6/6 incl. the four former failures; PEFT probe 7/7); CPU smoke
of the new `_check`: identical dicts pass, 5 % gradient perturbation fails (rel-Frobenius), 2 % update sign flips
fail, a +0.03 entropy shift on 128 of 498 rows fails the mean rule, two one-ulp entropy rows pass, an AdamW-style
2 * lr flip in `param/` passes, 3 * lr fails; `pre-commit run --files` Passed (ruff, ruff-format, mypy, license);
6 skipped on this Mac ("needs a CUDA device"; the model resolves at ~/models/Qwen/Qwen2.5-0.5B here). File 317 lines.

Bench command unchanged (file path unchanged): `python3 bench/verl_bench.py gpu-tests --src
<repo>/wt/p1 --run-name gpu-identity-p1-2 --tests
tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs" --no-torchrun` (`--gpu H100:1` suffices).
The queued run of d7d820aa (gpu-identity-p1-1) still measures the deltas, but its `param/` rule is the AdamW one:
expect the LoRA cases red there for D1, not for a defect; the 407082ec run is the one to judge.

## GPU run 1 and the fp32-reference redesign (2026-10-29 ~15:15-16:15Z)

Runs (Modal, 1x H100, verl CI image): `bench/results/gpu-identity-p1-2` @ d7d820aa and `-p1-3` @ 407082ec, 6 failed / 0
passed / 0 skipped each (~240 s). What they showed, identical on both trees: flag-off twice bitwise (82 / 877 / 1015
tensors on fsdp / fsdp2 / lora); forward **log-probs and entropy bitwise identical flag on vs off in all six cases**;
Σπ² max_abs 1.08e-7 (mean 2.8e-10): the same bf16 logits, a different fp32 reduction grouping; the gradient check failed
at its first key (sorted order: layer 0): `fsdp` flat param 1.694e-2, `fsdp2` `layers.0.input_layernorm.weight` 2.201e-2,
`lora` `layers.0.mlp.down_proj.lora_B` 1.676e-2 relative Frobenius vs the 1e-2 bound; later layers unreported (the
check raised at the first failure); `train/`, `loss/`, `update/`, `param/` unreported for the same reason.

Reading: the forward trunk is shared, so the only backward difference is the head: dH = dlogits @ W reduces over the
vocabulary (151936) with M = 498 vs 1589 rows (cuBLAS split-K differs by shape), and dW reduces over K vs T rows; both
round differently at the bf16 ulp in some elements. 24 layers of bf16 backward GEMMs then re-round everything they
touch, so the relative difference grows from the head toward layer 0: bf16 chaos, the same mechanism as run-to-run
nondeterminism in bf16 training, not a different gradient. A tolerance alone cannot prove that; an fp32 reference can.

Commit 363470c1 `[fsdp] test: fp32 reference arm and per-layer report in the GPU identity test` (file 409 lines):
- Five engines per case: bf16 off, bf16 off again (bitwise baseline), bf16 on, fp32 off, fp32 on. fp32 =
  `FSDPEngineConfig(mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"})` (the
  engine then skips autocast: `_autocast_dtype == fp32`, asserted) + `HFModelConfig(override_config={"attn_implementation":
  "sdpa"})` because flash-attn has no fp32; transformers 5.16 builds the packed block-diagonal mask from `position_ids`
  when `attention_mask is None` (masking_utils `find_packed_sequence_indices`; the engine's rmpad inputs pass
  `attention_mask=None`), and verl's monkey patch only replaces `_flash_attention_forward`, so sdpa is untouched.
  `_force_fp32_math()` after every engine build: TF32 off (cuBLAS, cuDNN, `set_float32_matmul_precision("highest")`;
  verl sets none of these) and only torch's math sdpa kernel (deterministic; the fp32 arms are not re-run twice).
- Sanity of the reference: fp32 flag-off log-probs vs bf16 flag-off log-probs mean |delta| < 0.5 nat (bf16 noise is
  ~1e-2-1e-1; a broken packed mask would be nats on most rows); printed with the max.
- Bounds (`TOLERANCES`): bf16 rows exact (log-probs, entropy, train forward, loss) and Σπ² atol 1e-6 (measured 1.1e-7);
  bf16 grad / update rel-Frobenius < 1e-1 per tensor (~4x the largest measured) and element-wise 1e-1 of the tensor's
  scale; fp32 rows atol 1e-4 (fp32 accumulation-order noise on a logit of 20 is ~1e-5; three orders under a bf16 ulp),
  fp32 grad / update 1e-4; `before/` (weights before the step) bitwise; `param/` two first-step updates + 1e-2 relative;
  update sign agreement printed only (dominated by noise-floor elements under 2 % chaos). The mean rule stays
  ((atol + rtol * mean|ref|) / sqrt(rows)).
- Reporting: `_compare` prints every key (forward / train / loss rows; a per-layer gradient table: n, max and pooled
  rel-Frobenius for layer 00..23, embed, norm, lm_head or FSDP1's root flat param; group summaries with the worst
  tensor) and returns the violations; `_check` (the reviewer-facing API) raises on them; the worker aggregates bf16 and
  fp32 violations and raises once at the end, so a failing run still prints both profiles.
- LoRA coverage: `_run_arm` asserts that the gradient keys equal the trainable set (every trainable tensor, nothing
  else) and, for LoRA, that every trainable name contains `lora_`; prints the count (fsdp2/lora: 168 adapter tensors).
- Verification here: `pre-commit run --files` Passed (ruff, ruff-format, mypy, license); 6 skipped (no CUDA); CPU smoke
  of `_compare` on synthetic dicts with layer names: identical pass; Σπ² +2e-6 fail, +5e-7 on every row fails the mean
  rule (a systematic shift; the measured mean is 2.8e-10 vs the 4.5e-8 bound); one log-prob +1e-3 fail (bf16 exact);
  grad +5 % bf16 pass, +20 % fail; grad +5e-5 fp32 pass, +5e-4 fail; `before/` 1e-6 mismatch fail; AdamW-style 2*lr
  `param/` flip pass, 3*lr fail; the per-layer table prints. Reviewer round2 tests: 12/13; the one failure,
  `test_entropy_tolerance_covers_one_bf16_ulp_of_a_top_logit`, encodes the superseded ulp rule (the bf16 forward is now
  asserted exact, as measured); the other three former failures still pass.
- Bench command (file path unchanged): `python3 bench/verl_bench.py gpu-tests --src
  <repo>/wt/p1 --run-name gpu-identity-p1-4 --tests
  tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs" --no-torchrun` (`--gpu H100:1`; ~7 min:
  five engines per case). Read the `[response_only_lm_head]` lines: the `grad layer NN` table for bf16 (expect ~1e-3 at
  layer 23 growing to ~2e-2 at layer 0) and the fp32 table (expect <= 1e-5 everywhere).

## GPU run 2 (gpu-identity-p1-4 @ 363470c1) and the final criterion (2026-10-29 ~16:30-17:30Z)

Run: 1x H100, 219 s, 2 passed (lora/fsdp, lora/fsdp2) / 4 failed / 0 skipped; `bench/results/gpu-identity-p1-4/pytest.log`,
tables in packages/p1/bench/RESULTS.md. Decisive facts, identical across the six cases: flag-off twice bitwise (107 / 902 /
1040 tensors); forward and training-forward log-probs, entropy and the loss max_abs 0 in bf16 AND in fp32; Σπ² 1.08e-7
(bf16) / 9.31e-8 (fp32); **fp32 arms agree on every gradient tensor**: pooled per layer 1.236e-5 (layer 0) .. 1.415e-5
(layer 12) .. 1.059e-5 (layer 23), worst tensor 5.663e-5 (`layers.20.input_layernorm.weight`); bf16 arm-vs-arm pooled
1.694e-2 (layer 0) falling to 2.199e-3 (layer 23), worst tensors `layers.4.input_layernorm.weight` 3.85e-2 and the q/k biases
up to 1.39e-1; `fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01`. So the bf16 gradient deltas are rounding
amplification (the fp32 arms, same math for K and T rows, agree to 1e-5), and the bf16 arm's own distance from fp32 is three
orders above that. The four failures were rule artefacts: (a) `fsdp` x2: the fp32 across-arm UPDATE rule at 1.002e-4 ..
1.290e-4 > 1e-4 on flat params: `param_after - param_before` cancels catastrophically (update ~lr*g against a parameter 1e2-1e4
larger), so the measured update carries eps_fp32*|param|/|update| of noise; (b) `fsdp2` x2: the bf16 relative update rule on
q/k biases and small layer-norm weights (up to 1.390e-1, sign agreement 0.89): small-norm tensors where ulp-level bf16 noise
dominates a relative measure. Also seen: the 407082ec stock GPU suite run shows 503 passed / 6 failed / 8 skipped, the 6
being this file: the stock selection collects it (no ignore glob matches), so the final suite run includes it; the skip
condition stays so hosts without ~/models/Qwen/Qwen2.5-0.5B skip rather than fail (verl's CI has the model).

Commit cf539906 `[fsdp] test: judge the bf16 gradients by their distance to the fp32 truth` (file 452 lines):
1. Kept: bitwise forward assertions (bf16 exact; Σπ² 1e-6; fp32 1e-4) and the fp32 arm-vs-arm gradient agreement at 1e-4
   per tensor (the correctness proof; measured worst 5.7e-5).
2. bf16 gradients: per tensor D_off = |g_off_bf16 - g_fp32|, D_on = |g_on_bf16 - g_fp32| with g_fp32 the fp32 flag-off
   gradient; assert D_on <= 2 * D_off + 1e-6 * |g_fp32| (independent noise: sqrt(2); 2 is margin; the floor covers tensors
   whose errors vanish). Scale-free: both sides are distances of the same tensor to the same truth. Printed per layer: n,
   max on-vs-off rel_fro, pooled D_off / |g32|, pooled D_on / |g32|, max ratio. No bf16 element-wise or rel-Frobenius
   arm-vs-arm assertion remains.
3. Update: no across-arm check. Within every arm, `sgd/residual_ulps` = max over tensors of |(after - before) + lr*grad| /
   (eps_fp32 * (|before| + |lr*grad|)) (Frobenius), asserted <= 4 for the on and the off arm (SGD with weight decay 0,
   clip 1e9 => clip coefficient exactly 1.0 => the step is one rounding of `before - lr*grad`; the synthetic smoke gives
   0.22 ulp). `before/` stays bitwise across arms; `param/` keeps the two-first-step-updates bound (trivially true for SGD;
   the reviewer's round-2 tests use it). The across-arm update rel_fro and the sign agreement are printed only.
4. Worker order: bf16 off (freeze old_log_probs), bf16 off again (bitwise), bf16 on, fp32 off (the truth; cross-precision
   sanity line), bf16 compare with truth, free, fp32 on, fp32 compare; violations of both precisions are collected and
   raised once.
5. Docstring: the measured profile (bitwise forward; fp32 1.1e-5 .. 1.4e-5, worst 5.7e-5; bf16 1.7e-2 -> 2.2e-3 by layer, up to
   1.4e-1 on q/k biases; 2.9e-2 nat cross-precision) and why the criterion is scale-free; the stock-suite / -x note.
Verification here: `pre-commit run --files` Passed; 6 skipped (no CUDA); CPU smoke of `_compare` with synthetic arms:
identical pass; on-arm with the same (independent) 2 % error as the off-arm pass (ratios 0.95-1.04); on-arm with 3x the
error fails (ratios 2.9-3.1); fp32 arms 5e-5 pass / 5e-4 fail; SGD residual 9 ulp fails; a 1e-7 `before/` mismatch fails.
Reviewer round2 tests: 12/13 (the one-ulp entropy test encodes the superseded ulp rule; bf16 forward is asserted exact).
Bench: `python3 bench/verl_bench.py gpu-tests --src <repo>/wt/p1 --run-name gpu-identity-p1-5
--tests tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs" --no-torchrun` (~4 min); expected 6
passed; read the `grad layer NN` lines (D_off ~ D_on, max ratio ~1-1.5) and the `SGD residual` lines (< 1 ulp).
Package: bench/RESULTS.md headline corrected to 657 tokens / K/T 0.0987 / 9.9 % of the rows (was 636 / 0.096 / 9.6 %);
no other 636 / 0.096 / 9.6 % left in RESULTS.md or CHECKLIST.md (the remaining mentions describe the correction).

## GPU run 3 and the final stock suite (2026-10-29, package files only, no branch change)

`gpu-identity-p1-5` @ cf539906 (1x H100, 196.6 s): 6 passed / 0 failed / 0 skipped. Forward and training-forward log-probs,
entropy and loss bitwise identical in both arms; fp32 on-vs-off gradient agreement max 1.792e-5 (fsdp, layer 16), 5.663e-5
(fsdp2, layer 20 input_layernorm), 3.834e-5 (lora, layer 20 k_proj); bf16 distance-to-truth max ratio D_on/D_off 1.01
(fsdp), 1.22 (fsdp2, layer 15), 1.10 (lora), e.g. layer 0 D_off 1.024e-1 / D_on 9.974e-2 (0.97); SGD residual 0.213-0.292 ulp
in every arm (0.000 for lora); `fp32 vs bf16 flag-off log-probs: mean=2.903e-02 max=4.309e-01`. Stock suite
`gpu-tests-p1-r2-final` @ cf539906 (H100:8): console 509 passed / 0 failed / 8 skipped (503 stock, identical to main, + the 6
identity cases), junit 535 / 527 / 0 / 8, torchrun 2/2. Logs: bench/results/gpu-identity-p1-5, bench/results/gpu-tests-p1-r2-final;
tables in packages/p1/bench/RESULTS.md (bench agent). PR-BODY.md, CHECKLIST.md and REVIEW.md §10.3 (MERGE-READY) updated.

## Round 3: the AI follow-up review (R2-03) (2026-10-29 ~18:00-19:30Z)

Source: research/review-bundle/follow-up-review/REVIEW.md R2-03 + evidence/p1/report.md (oracle test: 11 failed / 1 passed
against the cf539906 helper). No product defect; the GPU test passed NaN (every check was `delta > tol`, false for NaN,
incl. the engine's skipped step on non-finite gradients: transformer_impl.py ~895-899), the docstring / PR body / verdict
conflated the subtracted-update error with the gradient error (0.1390 is `update/...layers.2.self_attn.k_proj.bias`,
gpu-identity-p1-5/pytest.log:272; the worst actual bf16 gradient rel-Frobenius is 0.03851, fsdp2 layer-4
input_layernorm, log:270, and 0.03171 for LoRA; the run-4 failures named update/ keys), the bitwise forward was asserted
as if portable, and the scope was understated (one rank: FSDP1 NO_SHARD at world size 1; SGD; fresh zero-B LoRA with
zero lora_A gradients at the first backward; checkpointing and LoRA separate).

Commit e7c0d616 (GPU test 538 lines; oracle 131 lines):
- Finite checks: `_finite` requires `torch.isfinite(...).all()` on every inspected tensor of the enabled arm, the
  reference and the fp32 truth before any comparison; non-finite keys are reported and skipped. `_run_arm` records the
  optimizer step's returned grad norm (`sgd/grad_norm`); `_compare` requires it finite and > 0 when any reference gradient
  is non-zero, and requires a non-zero update in both arms wherever the reference gradient is non-zero (the skipped-step
  signature: NaN grads, unchanged params, zero update).
- Direct bf16 gradient bound retained: `|g_on - g_off| <= 0.1 |g_off| + 1e-6 G_layer` per tensor (0.1 ~ 2.5x the worst
  measured 0.0385; G_layer = the largest reference gradient norm of the tensor's layer, floors near-zero tensors), next
  to the distance-to-fp32 guard `D_on <= 2 D_off + 1e-6 |g32|`, now described as a relative-error guard (truth 1, off 1.1,
  on 0.8 passes the guard with a 27 % on/off difference; the direct bound rejects it). fp32: 1e-4 per tensor as before.
- Forward: bf16 asserted per row at rtol 2e-2 / atol 5e-2 (log-probs, entropy, Σπ²), batch mean within that over
  sqrt(rows), loss rtol 1e-2 / atol 5e-3; fp32 1e-4 absolute; the measured maxima are printed; the bitwise result is an
  observation on one H100 and shape (upstream CI may run L20).
- Scope stated in the docstring, PR body and here. `lora_B` seeded with randn * 1e-3 after `engine.initialize()`
  (sorted parameter names, one generator per arm, so both arms and the baseline get the same values; writes through
  `to_local()` for FSDP2 DTensors and into the flat param for FSDP1) so `lora_A` gets a first-backward gradient. New
  variant `lora-ckpt` (LoRA + gradient checkpointing). Two-rank plain cases for fsdp and fsdp2: `mp.spawn(nprocs=2)`,
  `fsdp_size=2` (FULL_SHARD), rank r takes sequences r::2 of the same frozen batch (global_batch_size 6, local 3), each
  rank compares its own shard of every gradient / parameter after the all-reduce (DTensor `to_local()`, FSDP1 local flat
  param shards), rank 0 prints; the cases skip when `torch.cuda.device_count() < 2`. Feasible without redesign because
  every rank runs the same five-engine flow on its own shard; the only shared state is the gradient all-reduce.
  Ten cases: {plain, gradient-checkpointing, lora, lora-ckpt} x {fsdp, fsdp2} x 1 rank + plain x {fsdp, fsdp2} x 2 ranks.
- CPU oracle `tests/workers/test_response_only_lm_head_oracle_on_cpu.py` (collected by cpu_unit_tests.yml): imports the
  helper from the GPU module the way the reviewer does; 19 tests: NaN in each of 6 inspected keys x 2 precisions, NaN in
  the reference / the truth, the skipped step on non-finite gradients, the skipped step with finite gradients (zero
  update, 840-ulp residual, zero grad norm), the direct bound rejecting the 27 % case the guard accepts and accepting a
  4.5 % case, the absolute floor for near-zero tensors, the ulp-sized forward delta passing and a row shift failing.
Verification here: pre-commit on both files Passed; oracle 19 passed; reviewer round2 13/13; reviewer round3 oracle 11/12
(`test_distance_rule_is_not_identity` expects the 27 % case to pass: by construction); shipped CPU file 26 passed; GPU
file on this Mac: 10 skipped. Bench (two devices so the two-rank cases run; eight is fine):
`python3 bench/verl_bench.py gpu-tests --src <repo>/wt/p1 --run-name gpu-identity-p1-6
--tests tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs" --no-torchrun --gpu H100:2`
(~8 min: ten cases, five engines each); expect 10 passed, then the stock suite on the same sha (513 = 503 + 10 if all run).

## Round-3 re-review D1-D5 (packages/p1/REVIEW-ROUND3-VERDICT.md; 2026-10-29 ~19:45-20:30Z)

Reviewer tests (not edited): review-tests/round3/test_r3_two_rank_split_on_cpu.py (3 pass / 1 fail = D1) and
test_r3_finite_gate_and_bounds_on_cpu.py (4 pass / 4 fail = D2-D5) at e7c0d616. Commit 1dbc0c07:
- D1 (HIGH): `_make_batch(vocab, rank, world)` now sets `max_response_len` / `max_response_length` to the rank-local
  maximum (rank 0 of the two-rank cases keeps responses 40/9/1): `ppo_loss` pads `old_log_probs` / `advantages` /
  `response_mask` with `to_padded_tensor()` (local max) and `log_prob` with `no_padding_2_padding` (`max_response_len`),
  so both were 256 vs 40 on rank 0 and the first arm crashed in `forward_backward_batch`. `_consumer_view` pads its
  output to the width over all ranks (256) so the comparison shape is the same on every rank (the reviewer's
  consumer-view test asserts (3, 256)); `_response_mask` gives the matching bool mask, carried per arm as
  `mask/response` (compared bitwise across arms). The loss normalisation itself was right (the reviewer's halves test).
- D2: `_finite` reports a key that the fp32 truth lacks and excludes it from the comparison.
- D3: the direct bound is `0.1 |g_off| + 1e-12` (tensor-relative; the absolute term only matters for a tensor with no
  gradient at all), so a negated tensor can never pass; tensors below 1e-6 of their layer's largest gradient norm are
  listed with the table. At cf539906 no tensor needed the old layer floor (the worst relative difference of any tensor was
  0.0385), so the change costs nothing on this model.
- D4: an empty tensor (an FSDP2 shard of a leading dimension below the world size) is skipped with a printed note.
- D5: the row checks take their valid mask from `mask/response` (falling back to all positions when the dict has no
  mask); a reference whose compared rows are all zero is a failure ("nothing was compared"); the mask is boolean, so
  it is compared with `torch.equal` before any arithmetic, and the finiteness gate applies to floating tensors only.
Branch oracle extended to 22 tests (negated tensor rejected / 5 % on a tiny tensor accepted / no gradient at all
accepted; missing truth key; empty shard; all-zero rows with and without a mask; the ulp forward case with the mask).
Verification: reviewer round3 12/12, round2 13/13, the submitter's oracle 11/12 (`test_distance_rule_is_not_identity` by
construction), shipped CPU file 26, GPU file 10 skipped here; pre-commit on both files Passed; tests/workers regression
re-run (tests.log §14). Bench unchanged: `python3 bench/verl_bench.py gpu-tests --src <repo>/wt/p1
--run-name gpu-identity-p1-6 --tests tests/workers/test_response_only_lm_head_on_gpu.py --pytest-extra "-s -rs"
--no-torchrun --gpu H100:2` (expect 10 passed), then the stock suite on the same sha.

## GPU runs 6-7 and the boolean-mask crash (2026-10-29 ~20:30-21:15Z)

- `gpu-identity-p1-6` @ e7c0d616 (H100:2, 326 s): **8 passed / 2 failed**: the eight one-rank cases passed, including the
  seeded-LoRA `lora` and `lora-ckpt` variants on both strategies (e.g. fsdp/lora-ckpt: fp32 vs bf16 flag-off log-probs mean
  3.197e-2, SGD residual 0.228-0.229 ulp, gradients on 336 adapter tensors; fsdp/plain bf16 grad pooled 1.190e-2 and fp32
  1.214e-5, as at cf539906); the two two-rank cases failed exactly as the round-3 reviewer predicted (D1): `RuntimeError:
  The size of tensor a (256) must match the size of tensor b (40) at non-singleton dimension 1` in `ppo_loss` on rank 0.
- `gpu-identity-p1-7` @ 1dbc0c07 (H100:2, 258 s): **10 failed**, every case at the same place, `_worker` line 537: the
  determinism-baseline loop computed `(baseline[key] - reference[key]).abs()` before `torch.equal` for every key, and the
  D1-D5 commit had added the boolean `mask/response` key: `RuntimeError: Subtraction, the '-' operator, with two bool
  tensors is not supported`. `_compare` had been taught to compare the mask before any arithmetic; the baseline loop had
  not, and the CPU oracle only exercised `_compare`. Diagnosed from the log; the lead's reading agrees.
- Commit 8c340e47 `[fsdp] test: compare the baseline without subtracting boolean tensors`: the loop is now
  `_assert_bitwise(what, baseline, reference)`: `torch.equal` on every key, a `max_abs` detail only for non-empty floating
  tensors. Oracle: `test_the_determinism_baseline_handles_boolean_and_empty_tensors` (bool, empty and float keys; a flipped
  mask and a shifted float each raise naming the key) and `test_the_response_mask_matches_the_rank_local_batch` (ranks
  (0,1), (0,2), (1,2): shape (local bsz, 256), row sums = the local response lengths, `max_response_len` = the local max).
  Re-check of the D1-D5 logic against the real-engine flow: every other subtraction in the worker and in `_compare` touches
  floating tensors only (cross-precision log-probs, rows, gradients, updates, params); the mask group returns before
  `delta`; the key sets are identical across arms and precisions by construction (same parameter names, same extra keys),
  so the missing-key rule cannot fire on a legitimate difference; the non-zero-update rule held on every p1-6 case
  (seeded LoRA included); the empty-shard skip is not reached by this model at two ranks.
Verification: branch oracle 26 passed; reviewer round3 12/12; round2 13/13; the submitter's oracle 11/12 (by construction); shipped
CPU file 26; GPU file 10 skipped here; tests/workers regression in tests.log §15; pre-commit Passed. Bench: the same command
with `--run-name gpu-identity-p1-8 --gpu H100:2`; expect 10 passed, then the stock suite on 8c340e47.

## GPU run 8 and the final stock suite (2026-10-29, package files only, no branch change)

`gpu-identity-p1-8` @ 8c340e47 (H100:2, 320.5 s): 10 passed / 0 failed / 0 skipped. Forward rows exact in every case (two-rank
fsdp rank 0: `forward/log_probs: max_abs=0.000e+00 mean_delta=0.000e+00 rows=50`); flag-off twice bitwise (110 tensors per rank on
two-rank fsdp); fp32 arms within 1e-4 per tensor; bf16 max on-vs-off gradient rel_fro 1.68e-2 (fsdp), 3.59e-2 (fsdp2), 1.44e-2
(two-rank), all under the 0.1 direct bound; D_on/D_off max 1.02 (fsdp), 1.22 (fsdp2), 1.11 (lora, lora-ckpt), 1.01 / 1.13 (two-rank
fsdp / fsdp2), e.g. two-rank fsdp `grad layer 00: n=1 max on-vs-off rel_fro=1.117e-02 D_off=1.450e-01 D_on=1.448e-01 max ratio=1.00`,
`grad: pooled rel_fro=1.026e-02 ... worst layers.7 _flat_param rel_fro=1.441e-02`; `sgd/residual_ulps: off=0.217 on=0.217`;
`fp32 vs bf16 flag-off log-probs: mean=4.694e-02 max=2.309e-01` (two-rank fsdp). Stock suite `gpu-tests-p1-r3-final` @ 8c340e47
(H200:8): console 513 passed / 0 failed / 8 skipped (503 stock + the 10 identity cases), junit 539 / 531 / 0 / 8, torchrun 2/2.
Logs: bench/results/gpu-identity-p1-8, bench/results/gpu-tests-p1-r3-final; tables in packages/p1/bench/RESULTS.md (bench agent).
PR-BODY.md, CHECKLIST.md, REVIEW.md §11 (MERGE-READY) and REVIEW-ROUND3-VERDICT.md (addendum) updated.
