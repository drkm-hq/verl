# PR checklist (verl .github/PULL_REQUEST_TEMPLATE.md) with evidence

Branch `contrib/p1-lmhead-rows` @ 8c340e47 (base main @ 2ab35292). Review rounds 2-3 (+ re-review), GPU runs 1-3 + stock suite, 2026-10-29.

## Checklist Before Starting
- [x] Search for similar PRs. Read-only `gh pr list --search` on 2026-10-29 13:19Z: `logits_to_keep` (#7370 CLOSED
      only), `lm head` (#7686 OPEN / CONFLICTING, last update 2026-10-12; #8441, #8312, #8096, #8097, #8004 are
      fused-kernel, FSDP2-hook or FP32-head work on other paths), `response_only_lm_head` (#7686 only), `loss rows`
      (nothing related); #8300 OPEN / MERGEABLE 2026-10-22 (prefix packing, composes). No new overlapping PR since
      the vet (research/vet/K2-lmhead-rows.md §3). Named in the body: #7370, #7686, #7712, #8300 and the five
      adjacent LM-head PRs.
- [x] Title `[fsdp, perf] feat: project the LM head only for loss rows` passes tests/special_sanity/check_pr_title.py
      (tests.log §3). No `[BREAKING]`: new optional config key, default off, no signature change.

## Test
- [x] CPU unit test `tests/workers/test_response_only_lm_head_on_cpu.py`: 26 passed (24 from the earlier rounds + the
      PEFT probe / resolver test + the PEFT pass-through test, tests.log §8), collected by
      `.github/workflows/cpu_unit_tests.yml`. Forward reads stay bitwise; gradients `assert_close(rtol=1e-5, atol=1e-6)`.
- [x] Full CPU regression on the worktree @ 2634019d (`scripts/run-cpu-tests-wt.sh`, the groups of
      `scripts/run-cpu-regression.sh`; logs/p1r2-reg-*.log): small 200 / utils 473 + 23 skipped / trainer 438 + 1 /
      workers 304 + 40 skipped + 1 deselected / experimental 74 / single_controller 21 = 1510 passed, 64 skipped,
      0 failed (main baseline 1484 / 64; the +26 are the new tests).
- [x] GPU identity test written: `tests/workers/test_response_only_lm_head_on_gpu.py` (commits 2634019d, d7d820aa, 407082ec, 363470c1, cf539906, e7c0d616, 1dbc0c07, 8c340e47; plain,
      gradient-checkpointing and LoRA variants, each on `fsdp` and `fsdp2`, on the real engine with Qwen2.5-0.5B; frozen batch, same seed,
      bf16 mixed precision; flag-off twice bitwise; flag on vs off on log-probs, entropy, Σπ², the PPO loss with
      `entropy_coeff=0.01`, every gradient, the update and the parameters after one plain-SGD step; per-row tolerances
      one bf16 ulp of a top logit, batch means over sqrt(rows); 6 cases). Skips loudly without CUDA or the model
      (6 skipped on this Mac: "needs a CUDA device"; the model itself resolves at ~/models/Qwen/Qwen2.5-0.5B here).
- [x] Round-2 re-review D1-D3 (REVIEW-ROUND2-VERDICT.md: AdamW sign-flip deltas in `param/`, entropy / Σπ² tolerance
      below one bf16 ulp, `pytest -x` in the GPU job): fixed in 407082ec; the reviewer's round2 CPU tests
      (review-tests/round2/, 13 tests) pass 13/13 (were 2 pass / 4 fail-by-design + 7). Collected by `gpu_unit_tests.yml` and by `bench/modal_verl.py gpu_tests` (no
      `on_cpu` / `special` / `vllm` / `sglang` in the name, under tests/workers).
- [x] GPU identity test run 1 (Modal 1x H100, `bench/results/gpu-identity-p1-2` @ d7d820aa and `-p1-3` @ 407082ec):
      forward log-probs and entropy bitwise identical flag on vs off in all six cases, Σπ² 1.1e-7, but gradients 1.7e-2
      (fsdp) / 2.2e-2 (fsdp2) / 1.7e-2 (lora) relative Frobenius at layer 0 against the 1e-2 bound: bf16 re-rounding of
      the head's backward through 24 layers, not a defect by construction, so 363470c1 adds the fp32 reference arm
      (same math for K and T rows; must agree to 1e-4), per-layer reporting of every tensor, and measured-based bounds.
- [x] GPU identity test run 2 (`gpu-identity-p1-4` @ 363470c1, 1x H100): 2 passed (lora x2) / 4 failed. Decisive facts: forward
      log-probs, entropy and loss bitwise in bf16 and fp32 (all six); fp32 gradients agree on every tensor, 1.1e-5 .. 1.4e-5
      pooled per layer (worst tensor 5.7e-5); bf16 arm-vs-arm 1.694e-2 (layer 0) -> 2.199e-3 (layer 23); `fp32 vs bf16
      flag-off log-probs` mean 2.9e-2, max 0.43. The four failures were rule artefacts: the fp32 across-arm UPDATE rule
      (1.0e-4 .. 1.3e-4: catastrophic cancellation of `param_after - param_before`) and the bf16 relative update rule on
      q/k biases (up to 1.39e-1, ulp noise on small-norm tensors). cf539906 replaces the bf16 arm-vs-arm tolerances by the
      distance-to-truth criterion `|g_on - g_fp32| <= 2 |g_off - g_fp32| + 1e-6 |g_fp32|`, drops the across-arm update
      check for a within-arm SGD consistency check (4 fp32 ulp), keeps the bitwise forward and the fp32 1e-4 proof.
- [x] GPU identity test run 3 (`gpu-identity-p1-5` @ cf539906, 1x H100, 197 s): **6 passed / 0 failed / 0 skipped**; forward
      and training-forward log-probs, entropy and loss bitwise in both arms; fp32 on-vs-off gradient agreement max 1.792e-5
      (fsdp, layer 16), 5.663e-5 (fsdp2, layer 20 input_layernorm), 3.834e-5 (lora, layer 20 k_proj); bf16 D_on/D_off max
      1.01 (fsdp), 1.22 (fsdp2, layer 15), 1.10 (lora); SGD residual 0.213-0.292 ulp (0.000 for lora); cross-precision
      mean 2.903e-2 / max 4.309e-1. Stock GPU suite on cf539906 (`gpu-tests-p1-r2-final`, H100:8): 509 passed / 0 failed /
      8 skipped (503 stock + the 6 identity cases; junit 535 / 527 / 0 / 8), torchrun 2/2.
- [x] Round 3 (the independent follow-up, R2-03: the helper passed NaN; gradient/update numbers conflated; bitwise forward not
      portable; scope): fixed in e7c0d616: every inspected tensor must be finite, the step's grad norm finite and positive,
      the update non-zero wherever the gradient is; direct bf16 gradient bound 0.1 per tensor with a 1e-6 layer floor next
      to the distance-to-fp32 guard (described as a relative-error guard); forward asserted at the ulp tolerance (bitwise
      reported as an observation); corrected numbers (worst bf16 gradient 0.0385 / 0.0317 LoRA; 0.139 was an update);
      scope stated (one rank, SGD, fresh adapter); `lora_B` seeded; `lora-ckpt` variant; two-rank plain cases (skip
      without two devices). CPU oracle `tests/workers/test_response_only_lm_head_oracle_on_cpu.py`: 19 passed; reviewer
      round2 13/13; reviewer round3 oracle 11/12 (the 27 %-difference characterization now fails by construction: the
      direct bound rejects it).
- [x] Round-3 re-review D1-D5 (REVIEW-ROUND3-VERDICT.md): fixed in 1dbc0c07: rank-local `max_response_len` on the two-rank
      cases (D1: `ppo_loss` padded log_prob to 256 and the other fields to the local 40 on rank 0 and crashed), the
      comparison view padded to the width over all ranks; a key missing from the fp32 reference is a failure (D2); the
      direct bound is `0.1 |g_off| + 1e-12` with the near-zero tensors listed (D3: the layer floor could hide a negated
      tensor); empty shards skipped with a note (D4); row checks on the response mask, all-zero rows reported (D5).
      Reviewer round3 tests 12/12 (were 7/12); round2 13/13; the submitter's oracle 11/12 (by construction); branch oracle 22/22.
- [x] GPU identity run 4 (`gpu-identity-p1-6` @ e7c0d616, H100:2): 8 passed / 2 failed, the two-rank cases crashing in
      `ppo_loss` exactly as the round-3 reviewer predicted (D1); run 5 (`gpu-identity-p1-7` @ 1dbc0c07): 10 failed, the
      determinism-baseline loop subtracted the new boolean `mask/response` (fixed in 8c340e47, `_assert_bitwise`).
- [x] GPU identity run 6 (`gpu-identity-p1-8` @ 8c340e47, H100:2, 320.5 s): **10 passed / 0 failed / 0 skipped**; forward
      rows exact in every case; fp32 arms within 1e-4 per tensor; bf16 on-vs-off gradient max 1.68e-2 (fsdp), 3.59e-2
      (fsdp2), 1.44e-2 (two-rank) under the 0.1 bound; D_on/D_off max 1.02 / 1.22 / 1.11 / 1.01 / 1.13 (fsdp / fsdp2 / lora +
      lora-ckpt / two-rank fsdp / fsdp2); SGD residual 0.217 ulp on the two-rank arms; flag-off twice bitwise (110 tensors per
      rank on two-rank fsdp). Stock suite `gpu-tests-p1-r3-final` @ 8c340e47 (H200:8): 513 passed / 0 failed / 8 skipped
      (503 stock + 10 identity; junit 539 / 531 / 0 / 8), torchrun 2/2. Logs: bench/results/gpu-identity-p1-8,
      bench/results/gpu-tests-p1-r3-final; tables in packages/p1/bench/RESULTS.md.
- [x] Timing / peak memory on 8x H100 (Modal harness, `verlai/verl:uv.cu130.dev5`, Qwen3-1.7B GRPO n=8, pooled means
      of steps 2..6, two runs per arm): `long-prompt` (6144/1024; measured prompt 5999 / response 657 tokens, K/T 0.099,
      recomputed 2026-10-29 from `bench/results/long-prompt-{main-1,main-2,p1-on-2,p1-on-3}/metrics.jsonl`) training
      phases 40.52 -> 35.68 s (-12 %), peak memory 16.56 -> 11.91 GB/rank (-4.65 GB); flag OFF = main (40.61 s).
      `baseline-a` (83-token prompts, K/T 0.94) training phases 15.74 -> 14.56 s (-7.5 %, exploratory), no regression.
      `gen` and whole-step deltas listed separately and not claimed (generation varies ~10 % between identical runs).
      Sources: `bench/CROSSCHECK.md` and `packages/p1/bench/RESULTS.md`, which agree.
- [x] GPU unit suite: verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and
      `tests/experimental`, so it does not execute this PR's new tests), 8x H100 at 4bcbbeb9 and b32c777e: 503 passed /
      8 skipped / 0 failed + torchrun 2/2, identical to main; @ cf539906 (`gpu-tests-p1-r2-final`) 509 passed / 0 failed / 8 skipped, the 6 extra being the
      identity cases, torchrun 2/2.
- [ ] Not measured: models larger than 1.7B, MoE / `trust_remote_code`, Ulysses SP (guarded off), `torch.compile`
      entropy recompiles under `pad_to_length`, NPU.

## API and Usage Example
- [x] `actor_rollout_ref.actor.fsdp_config.response_only_lm_head=True` (ref follows the actor via `oc.select`;
      `actor_rollout_ref.ref.fsdp_config.response_only_lm_head` overrides). Example in PR-BODY.md.

## Design & Code Changes
- [x] Listed in PR-BODY.md; file:line evidence in NOTES.md. Round 2: the probe unwraps a PEFT wrapper
      (`get_base_model()`), 724d3d5e; the PR body states the one device-to-host transfer per micro-batch
      (`int(prompt_lens.min())`, the empty-prompt guard) instead of "no device-to-host sync".

## Checklist Before Submitting
- [x] Read the Contribute Guide (CONTRIBUTING.md; AGENTS.md AI-attribution trailer on every commit).
- [x] pre-commit: `pre-commit run --all-files --show-diff-on-failure` at 2634019d on this Mac: every hook Passed,
      0 Failed (logs/pre-commit-all-files-r4.log; hook envs are the pinned ones, ruff v0.12.2, mypy v1.17.0). The
      submitter should repeat `--all-files` once on Linux before filing.
- [x] Documentation: docs/perf/perf_tuning.rst item 11 + section "Project the LM head only for loss rows (FSDP)";
      config docstring in verl/workers/config/engine.py; yaml comment in engine/fsdp.yaml. Nothing in the docs page
      repeats the corrected PR-body claims (sync, 636 / 0.096), so it is unchanged in round 2.
- [x] Unit tests in the CI workflows: CPU job collects `tests/workers/*_on_cpu.py`; the GPU job collects
      `tests/workers/test_response_only_lm_head_on_gpu.py`.
- [ ] CI request in the `ci-request` Slack channel: submitter, after review.
- [x] Not a `recipe` change: no submodule update needed.
- [x] AI statement in the PR body: "Written with Claude (Anthropic) under the author's direction. Commits carry a
      Co-authored-by trailer."
