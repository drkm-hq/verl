# P2 old-log-prob skip: Modal GPU evidence (bench/README.md harness)

Preset `baseline-a-k1` = BASELINE-A (8x H100 80GB, Qwen3-1.7B, GRPO n=8, `train_batch_size=128`, prompt 1024 /
response 1024..2048, 6 steps, `use_kl_loss=True`, dynamic bsz 24576) with `actor.ppo_mini_batch_size=128`: one
optimizer step per training step (k=1, strictly on-policy), the regime the skip is exact in. The flag:
`actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=True`; the OFF arm sets it to `False` explicitly on the same
tree. Image, lock file and data identical in all arms (`pip-freeze.txt`, `manifest.txt`, `digest_match=True`).
Full commands in `bench/results/<run>/command.txt`; per-step metrics in `metrics.jsonl`.

## Headline: training-side timings, mean of steps 2..6 (s), one container of 8x H100 each

| run | tree (sha) | old_log_prob | ref | update_actor | train sum (olp+ref+upd) | step | gen (rollout noise, see note) | peak mem GB (actor max_memory_allocated) |
|-----|------------|--------------|-----|--------------|-------------------------|------|-------------------------------|------------------------------------------|
| baseline-a-k1-main-1 | main@2ab35292 | 3.719 | 2.880 | 9.112 | 15.71 | 31.40 | 12.79 | 15.07 |
| baseline-a-k1-p2-off-1 | contrib/p2-oldlogprob-skip@1e45a508 (5 commits, flag OFF) | 3.509 | 2.593 | 8.603 | 14.70 | 29.20 | 10.88 | 14.97 |
| baseline-a-k1-p2-on-1 | contrib/p2-oldlogprob-skip@1e45a508 (flag ON) | **0.000** | 3.094 | 9.262 | **12.36** | **26.39** | 10.46 | 14.88 |
| baseline-a-k1-main-2 | main@2ab35292 | 3.644 | 2.896 | 9.408 | 15.95 | 32.16 | 13.16 | 15.05 |
| baseline-a-k1-p2-off-2 | p2@1e45a508 OFF | 3.431 | 2.645 | 8.323 | 14.40 | 29.05 | 10.85 | 14.91 |
| baseline-a-k1-p2-on-3 | contrib/p2-oldlogprob-skip@**5cb89c52** (9 commits), ON | **0.000** | 4.271 | 11.672 | 15.94 | 34.97 | 14.60 | 14.92 |
| baseline-a-k1-p2-on-4 | p2@5cb89c52, ON | **0.000** | 4.383 | 11.992 | 16.37 | 35.37 | 14.62 | 14.89 |
| baseline-a-k1-p2-off-3 | p2@2b259b2d (10 commits, final HEAD), OFF - same window as on-5, both slow nodes (wall 801 / 711 s) | 4.702 | 3.418 | 9.623 | 17.74 | 39.11 | 14.83 | 15.07 |
| baseline-a-k1-p2-on-5 | p2@2b259b2d (final HEAD), ON | **0.000** | 4.213 | 11.390 | **15.60** | 33.78 | 13.75 | 15.00 |
| **same scheduling window ON vs OFF** (on-5 vs off-3; exploratory: similar windows do not prove the same physical node) | | **-4.70 s** | +0.80 | +1.77 (+18 %) | **-2.14 s (-12.1 %)** | -5.3 s (-13.6 %) | | |
| **same-tree ON vs OFF at 1e45a508** (on-1/2 vs off-1/2 means; exploratory subgroup) | | **-3.47 s** | +0.67 | +1.26 (+15 %) | **-1.53 s (-10.5 %)** | -1.3 s | | |

Equality columns on the final-HEAD pair: entropy 0.203 (ON) / 0.207 (OFF), pg_clipfrac and ppo_kl exactly 0 on ON,
kl_loss 0.001, rollout_probs_diff 0.006. Similar aggregate values do not prove unchanged training signals (the exactness
proof is the bitwise CPU test `tests/workers/test_ppo_loss_on_policy_skip_on_cpu.py`); they show nothing gross moved.

on-3 and on-4 (launched 04:45-04:52, both ~710 s wall vs ~500 s for every other k1 run) read as slower hosts: `ref`
(+50 %: a param-offloaded forward, the phase most sensitive to host PCIe/CPU contention) and `gen` (+15 %) are not
touched by the flag, so their inflation is attributed to the host, not the branch (inferred from wall time, not
verified); `old_log_prob` is still exactly 0 and the equality columns are unchanged. off-3/on-5 ran in one scheduling
window (exploratory; the pooled means below are the number to quote).

Shas: arms on-1/on-2/off-1/off-2 and gpu-tests-p2-1 ran at 1e45a508 (5 commits); on-3/on-4 and gpu-tests-p2-2 at
5cb89c52 (9 commits); off-3/on-5 at 2b259b2d (10 commits, the validation-only change: nested dropout scan, fsdp_turbo
allowed). The timing numbers were measured at those shas; the product code since 2b259b2d (effective-model validation,
the worker module walk, the legacy-trainer rejection; HEAD 7f0c53f5) is not on the measured path and was not re-timed. The
stock GPU unit-test selection ran on 7f0c53f5 (gpu-tests-p2-r2) and on the final HEAD e19e7949 (gpu-tests-p2-r3-final, 8x H200), both
503 / 0 / 8 + torchrun 2/2, identical to main (regression coverage; the perf numbers stay those measured at 2b259b2d and earlier).

The 5cb89c52 tree rejects the flag unless the sampled batch is exactly one mini-batch of rows
(`ppo_mini_batch_size x n`); the preset satisfies it (`train_batch_size=128`, `ppo_mini_batch_size=128`, `n=8`) and the
final-sha ON arms also pass `trainer.v1.separate_async.parameter_sync_step=1` explicitly (unused in sync mode, makes the
condition explicit in `command.txt`). The OFF control ran on 1e45a508 (semantics identical when disabled).
| baseline-a-k1-p2-on-2 | p2@1e45a508 ON | **0.000** | 3.490 | 10.180 | **13.67** | 29.20 | 12.77 | 15.07 |

Two-run means (+- half-range): main old_log_prob 3.68 +- 0.04, ref 2.89 +- 0.01, update_actor 9.26 +- 0.15, train sum
15.83 +- 0.12, step 31.78 +- 0.38; OFF (1e45a508) 3.47 +- 0.04 / 2.62 +- 0.03 / 8.46 +- 0.14 / 14.55 +- 0.15 / 29.13 +- 0.08;
ON (1e45a508) 0.000 / 3.29 +- 0.20 / 9.72 +- 0.46 / 13.02 +- 0.66 / 27.80 +- 1.40; ON (final 5cb89c52): see rows on-3/on-4.

- `timing_s/old_log_prob` 3.4-3.7 s -> **0.000 s** on every ON run (the pass is skipped; the trainer logs
  `actor.skip_old_log_prob_when_on_policy=True: the old-log-prob pass is skipped` at start-up).
- `timing_s/update_actor` ON 9.72 +- 0.46 vs OFF 8.46 +- 0.14 on the same tree (+1.26 s, +15 %; vs main 9.26 +- 0.15:
  +5 %): the update pass now also returns the log-probs and computes the (no-grad) entropy metric that old_log_prob used
  to compute from its own logits. The two ON runs spread more (9.26 / 10.18) than the two OFF runs (8.60 / 8.32).
- Net training side at 1e45a508 (exploratory subgroup; the pooled claim is -5.3 % / -6.6 %, see below): 15.83 (main) /
  14.55 (OFF) -> 13.02 +- 0.66 s (ON): -1.5 s, -10.5 % vs OFF on the same tree (-18 % vs main); step time 29.13 -> 27.80 s (-4.6 %) with gen 10.9 / 10.5 / 12.8 s on those runs (gen varies +-10 %
  between identical runs on this harness, so step-level deltas of that size are rollout noise; the training-side
  columns are the metric). The OFF tree is itself ~1.3 s faster than main on the training side (gen-side host variance
  bleeds into `update_actor` through the vLLM weight sync that is timed inside it) - which is why the claim is read
  ON vs OFF on the same tree.
- `ref` is untouched by the flag (2.6-3.5 s across arms, inside its own within-run spread of 2.65-4.26 s).

## Equality checks (mean of steps 2..6)

| run | actor/entropy | actor/pg_clipfrac | actor/ppo_kl | actor/kl_loss | training/rollout_probs_diff_mean | critic/score/mean |
|-----|---------------|-------------------|--------------|---------------|----------------------------------|-------------------|
| baseline-a-k1-main-1 | 0.204 | 0.000 | 0.000 | 0.001 | 0.006 | 0.005 |
| baseline-a-k1-p2-off-1 | 0.207 | 0.000 | 0.000 | 0.001 | 0.006 | 0.003 |
| baseline-a-k1-p2-on-1 | 0.206 | **0.000 (exact)** | **0.000 (exact)** | 0.001 | 0.006 | 0.004 |

`pg_clipfrac` and `ppo_kl` are exactly 0 on the ON arm (ratio == 1 by construction); on main/OFF they round to 0.000
at k=1 as well (bf16 kernel noise between the two passes). Entropy, kl_loss and the rollout-vs-actor probability gap
(`training/rollout_probs_diff_mean`) are equal within noise; similar aggregate values do not prove unchanged training
signals (the bitwise CPU proof does that for the single-mini-batch regime), they show nothing gross moved. The entropy
metric and the log-prob write-back to TransferQueue now run in the update pass, which is where `update_actor` grows. Reward is
~0 on all arms (6 steps of a base model on GSM8K: a timing run, not a quality run).

## Pooled means, steps 2..6 per run then across runs (the PR's claim)

| arm (runs) | old_log_prob | ref | update_actor | training side (olp+ref+upd) |
|---|---|---|---|---|
| main @2ab35292 (2) | 3.68 | 2.89 | 9.26 | 15.83 s |
| branch flag OFF (3: off-1, off-2, off-3) | 3.88 | 2.89 | 8.85 | 15.62 s |
| branch flag ON (5: on-1..on-5) | 0.00 | 3.89 | 10.90 | 14.79 s |

ON vs OFF **-5.3 %**, ON vs main **-6.6 %**: the old-log-prob forward (3.4-4.7 s) is removed while `update_actor` grows by
1.0-1.7 s (entropy metric + log-prob write-back now in the update pass) and `ref` read slower on the ON arms that ran
later. The -10.5 % / -12.1 % pairwise readings above are exploratory subgroups of the same data. An independent
recomputation from `bench/results/*/metrics.jsonl` (review round 2) gives 15.8295 / 15.6156 / 14.7895.

## GPU unit tests

This is verl's stock GPU unit-test selection (regression coverage: it excludes `*_on_cpu.py` and `tests/experimental`,
so it does not execute this PR's new tests). Enabled-feature GPU evidence is the timing runs above only.

`uv run --frozen --all-packages --extra sglang --extra megatron --extra ci pytest -s <gpu_unit_tests.yml ignores> tests/`
+ the two torchrun tests, 8x H100; main reference `gpu-tests-main-2`: 503 passed / 0 failed / 8 skipped, torchrun 2/2 rc=0.

| run | tree | pytest | torchrun | new failures |
|-----|------|--------|----------|--------------|
| gpu-tests-p2-1 | p2@1e45a508 (5 commits) | **503 passed / 0 failed / 8 skipped**, 25 min 23 s, rc=0 | (see bench/README.md status) | none |
| gpu-tests-p2-2 | p2@5cb89c52 (9 commits) | **503 passed / 0 failed / 8 skipped**, 25 min 18 s, rc=0 | 2/2 rc=0 | none |
| gpu-tests-p2-r2 | p2@7f0c53f5 (16 commits: 2b259b2d + the round-2 / re-review fixes + the legacy rejection, i.e. validation / init-time check / tests / docs, none on the measured path) | **503 passed / 0 failed / 8 skipped**, 30 min 26 s, rc=0 | 2/2 rc=0 | none |
| gpu-tests-p2-r3-final | p2@e19e7949 (18 commits, the final HEAD; the two round-3 commits are validation-only, off the measured path) | **503 passed / 0 failed / 8 skipped** (junit 529 / 521 / 0 / 8), 30 min 30 s, rc=0, 8x H200 | 2/2 rc=0 | none |

Apps (stopped by id, the launch registry (kept privately)): baseline-a-k1-main-1 <modal-app>, baseline-a-k1-p2-on-1
<modal-app>, baseline-a-k1-p2-off-1 <modal-app>.
