# K5 gae-vectorized — vet (2026-10-29, adversarial vetter)

Scripts and raw output: `research/vet/K5/parity_bench.py`, `research/vet/K5/parity_bench.out`
(run in `.venv-a3`, torch 2.15.0, arm64 macOS, 6 threads, `PYTHONPATH=verl` @ 2ab35292).

## 1. Claim

Source: `research/03-trainer-datapath-audit.md` finding F1 (§2) and candidate C1 (§5). The GAE and REINFORCE++
advantage estimators (`verl/trainer/ppo/core_algos.py:216-262` and `:694-732`) are O(L) Python loops over the
response axis, ~6 width-N tensor ops per step, executed on the driver inside `compute_advantage` in both trainers
(V0 `ray_trainer.py:1667`, V1 `v1/trainer_base.py:1780` via `compute_advantage_for_multi_trajectories`). Report 03
measured 1.30 s (N=4096, L=8k) / 5.22 s (L=32k) for GAE and 0.81 / 3.46 s for RF++, and prototyped an exact
blockwise linear-recurrence scan (`research/bench-a3/bench_gae_vec.py`) with the same observation-token carry,
max |Δ| ≤ 1e-5, 7-27× faster at γλ=1. The candidate PR replaces both loops with the scan, keeps the loops as
reference implementations for a randomized parity test, and supersedes stale PR #2132 (conv1d, June 2025).

## 2. Mechanism verification

### 2.1 What the loops compute (read, `core_algos.py`)

GAE (`:245-262`), per column t from L-1 down to 0, with m_t = response_mask[:, t]:
- `delta = r_t + gamma * nextvalues - v_t` (`:250`), `lastgaelam_ = delta + gamma*lam*lastgaelam` (`:251`)
- `nextvalues = v_t*m_t + (1-m_t)*nextvalues` (`:254`): the carried value is the value of the next *valid* token
  after t (0 if none); on an observation token the carry passes through unchanged.
- `lastgaelam = lastgaelam_*m_t + (1-m_t)*lastgaelam` (`:255`) and the *updated* carry is appended for every t
  (`:257`), so on a masked column `advantages[:, t] = g_{t+1}` (the carry, not 0). Then `returns = advantages +
  values` (`:260`) at every column, and `advantages = masked_whiten(advantages, response_mask)` (`:262`), which
  whitens *all* columns with mask statistics (`torch_functional.py:342-343`; no re-masking). Downstream the masked
  columns are dead (every loss multiplies by `response_mask`), but a parity test must still compare them, because
  the critic's `returns` are consumed with the same mask (`losses.py:222-236`).
- `values` is read at every column including masked ones; `gamma`/`lam` arrive as floats from
  `algorithm.gamma=1.0`, `algorithm.lam=1.0` (`ppo_trainer.yaml:72,75`), so the default is q = γλ = 1.

RF++ (`:718-732`, semantics from #7300, merged 2026-08-10): `new = r_t + gamma*running`; `returns[:, t] = new*m_t`;
`running = new*m_t + running*(1-m_t)`, i.e. rewards on masked columns are *ignored* and the running return is
carried through observation spans; `advantages = masked_whiten(returns, mask) * mask` (`:730-731`).

Both are first-order linear recurrences g_t = q^{m_t} g_{t+1} + m_t·b_t with b = δ (GAE) or r (RF++): exactly what
the prototype solves (`bench_gae_vec.py:30-52`). `nextvalues` is a gather at the next valid index (reverse
`cummin` of valid indices, `:60-66`); δ is one vectorized op; the scan is solved per 256-column block in closed
form q^{-S_t}·revcumsum(b_j q^{S_j}) with S = in-block exclusive prefix count of valid tokens, plus an L/256-step
carry loop (`:34-52`); q = 1 collapses to one reverse `cumsum` (`:30-31`). Masked columns get g_t = g_{t+1},
matching `:255`. Branches: holes in `response_mask` (carry), γ<1, λ<1 (q<1 path), L not divisible by the block
(zero-padded then sliced, `:38-41`), L=1 (`nxt` cat of an empty slice, `:65`). All exercised below.

### 2.2 My own parity run (`parity_bench.py`, 216 cases)

Shapes (N,L) ∈ {(2,1),(5,1),(8,2),(8,7),(5,255),(5,256),(5,257),(6,513),(16,300),(64,1000),(256,2048),(128,8192)},
seeds 0-2, (γ,λ) ∈ {(1,1),(1,.95),(.99,.95),(.9,.9),(.5,.5),(.3,1)}; masks = random lengths with three random
observation holes per row plus five special rows (all-masked, all-valid, only-first-valid, leading observation
span, trailing padding); `values` unmasked (the loop reads them on masked columns); diffs measured on **all**
columns, NaN==NaN:

| quantity | fp64 scan | fp32 scan |
|---|---|---|
| GAE advantages max abs Δ (whitened, O(1) scale) | 9.06e-06 | 9.06e-06 |
| GAE returns max abs Δ (at max abs returns 216) | 5.04e-04 (2.3e-06 relative) | 5.04e-04 |
| RF++ returns / advantages max abs Δ (fp32 scan) | – | 5.49e-04 / 9.54e-06 |

The fp64 and fp32 scans give the same worst case, i.e. the residual is the fp32 *loop's* own sequential rounding
over up to 8192 steps, not the scan's; an fp32 production path is at least as accurate as the loop. Tiny shapes
(L=1, L=2, L=7) are bitwise or 1-ulp identical. Degenerate batches match the loop's behaviour exactly: a batch
with a single valid token raises verl's `masked_var` guard in both (`torch_functional.py:323`), an all-masked batch
raises `:319` in both.

### 2.3 The "cleaner" alternative, checked (`parity_bench.out`, naive global weighted cumsum)

A single cumsum over q^{S_t}-weighted terms across the whole response (no blocking) is exact only while q^{L}
stays in range: at q=0.95 it is fine to L=2048, returns NaN at L=8192 in fp32 and at L=32768 even in fp64
(0.95^32768 = e^-1681). A log-depth Hillis-Steele scan would be exact but costs ~log2(L)=15 full passes over N×L
versus ~6 for the blockwise form; `torch.associative_scan` only exists under `torch.compile`. The blockwise scan
*is* the standard chunked-SSM scan (Mamba-2 style) and stays exact in fp32 when the block is capped so that
q^{block} ≥ 2^-60 (block ≤ 41.6/−ln q: 256 for q ≥ 0.85, 60 at q=0.5; `parity_bench.py:24-27`). Recommendation:
fp32 math, block 256 capped by that rule, q = 1 fast path; no float64.

### 2.4 Benchmark (same machine as report 03; realistic masks with holes; loops single run, scans median of 3)

| N × L | GAE loop | scan q=1 fp32 | scan q=.95 fp32 | RF++ loop | RF++ scan γ=1 | RF++ scan γ=.99 |
|---|---|---|---|---|---|---|
| 1024 × 8192 | 420 ms | 28 ms (15×) | 56 ms (7.6×) | 258 ms | 12 ms (21×) | 40 ms (6.5×) |
| 4096 × 8192 | 1420 ms | 105 ms (14×) | 197 ms (7.2×) | 925 ms | 43 ms (22×) | 136 ms (6.8×) |
| 4096 × 32768 | 5685 ms | 368 ms (15×) | 843 ms (6.7×) | 3491 ms | 164 ms (21×) | 634 ms (5.5×) |

So the mechanism and the numbers in report 03 reproduce: the loops cost 1.4-5.7 s per step at N=4096 and the
exact scan removes 85-95 % of it. Whether that is worth a PR is §3-§7.

### 2.5 Where it sits in the step

V1 (`trainer_base.py:596-611`): steps 6-9 run serially on the driver — `_compute_values` → `_compute_advantage`
(timer `adv`, metric `timing_s/adv`, `metric_utils.py:645`) → `_update_critic` → `_update_actor`. Nothing
overlaps it in any V1 mode (`trainer_sync.py`, `trainer_colocate_async.py`, `trainer_separate_async.py` override
no step method except `_compute_old_log_prob`). V0 is the same serial call (`ray_trainer.py:1667`). GAE needs a
critic: it is the yaml default (`ppo_trainer.yaml:78 adv_estimator: gae`) but every shipped script overrides it:
`adv_estimator=gae` in 4 example scripts (`examples/ppo_trainer/run_qwen3_8b_{fsdp,megatron}.sh`,
`examples/ascend_extras/ppo_trainer/…`, `examples/tutorial/slurm/ray_on_slurm.slurm`), 0 under `recipe/`;
`reinforce_plus_plus` in 0 scripts (README mention only); `grpo` in 93 files. The reference user (GRPO) never
executes either loop.

## 3. Duplicate / ownership check (07 §8 recipe: local dump + `gh` reads, 2026-10-29)

- `dupcheck.py --kw gae --kw vectoriz --kw advantage_return --kw reinforce_plus_plus --path core_algos.py --merged`:
  57 open PRs touch `core_algos.py` or match; **none vectorizes the time loops**. `gh pr list --search gae` (open and
  merged) and the issue dump agree. Merged titles matching gae/advantage/reinforce since 2026-07-16: #7300 only.
- **#2132 "[algo] feat: A more Efficient implementation of GAE"** (Clement25, opened 2025-06-21, last commit
  2025-06-21, CONFLICTING, REVIEW_REQUIRED, +11/−9 in `core_algos.py`, conv1d reformulation, no observation-token
  handling, no test, 7 comments). The reviews are the decisive evidence:
  - eric-haibin-lin (2025-06-22): asked for a unit test asserting old/new consistency.
  - vermouth1992 (collaborator): "We probably don't want to directly modify the original implementation because it
    is much more easier to modify. Also, the fraction of time gae computation contributed to the overall training time
    is basically negligible … makes this optimization less necessary."
  - **wuxibin89** inline review (2025-07-02, `core_algos.py:194`): "Since GAE computation time is trivial in a whole
    step, the readability is more important. Moreover, `torch.conv1d` is hard to skip observation tokens in
    multi-turn rollout."
  The conv1d objection is answered by the scan (exact, carries the mask); the "trivial in a whole step" objection is
  not (see §5). Per 07 §8's decision rule, a maintainer rejection in comments means do not re-open the approach.
- Adjacent, alive, not duplicates: #8368 / #8379 (put the agent-loop score on the last *generated* token — reward
  placement feeding these loops, issue #8367, 2026-10-25); #7473 (pass `non_tensor_batch` to estimators;
  `ray_trainer.py` dispatch only); #7957 (reject `use_pf_ppo` on the V1 GAE path). Merged #7300 (shotsan,
  2026-08-10) is the current RF++ loop semantics — the loop is the maintained reference, changed 2 months ago.
- Ownership: the GAE loop's observation-token semantics are wuxibin89's own (#2337, 2025-07-04, merged by
  vermouth1992); the file's last 12 commits are by 10 different contributors; the reviewer to expect is wuxibin89.

## 4. Minimum valuable PR (if pursued despite §7)

`verl/trainer/ppo/core_algos.py`: one shared helper `_masked_discounted_reverse_scan(mask, b, q, block=256)`
(~45 lines, fp32, q=1 fast path, block capped at 41.6/−ln q) plus `_next_valid_values(values, mask)` (~10 lines);
rewrite the bodies of `compute_gae_advantage_return` (`:245-262`) and
`compute_reinforce_plus_plus_outcome_advantage` (`:718-732`) on top of them (~20 lines); keep the two loops as
private `_…_reference` functions (~30 lines) used only by tests. Tests: `tests/trainer/ppo/test_core_algos_on_cpu.py`
gains `test_gae_scan_matches_reference` / `test_reinforce_pp_scan_matches_reference` parametrized over shapes
incl. L=1, L=257, 8192, seeds, (γ,λ) incl. 0.3, masks with observation holes and the five degenerate rows,
`assert_close(atol=5e-5, rtol=1e-5)` on all columns (~60 lines); `test_multi_turn_compute_gae_advantage_return`
and `tests/trainer/ppo/test_reinforce_pp_multiturn_on_cpu.py` (4 tests, #7278 regression) must pass unchanged.
No config key (a flag for an exact refactor is busywork). Docs: docstrings only. ≈ 170 lines, 2 files.
Title: `[algo, perf] refactor: vectorize the GAE and REINFORCE++ time scans (exact, keeps the observation-token carry)`.
The body must cite #2132, quote both maintainer verdicts and argue with the 32k numbers.

## 5. Proof

CPU: §2.4 table (the mechanism) and the parity test (exactness). Modal (`bench/README.md`): preset `baseline-a`
(8×H100, Qwen3-1.7B, bs 128 × n=8 → N=1024, 1024/2048, 6 steps) with
`--overrides "algorithm.adv_estimator=gae critic.enable=True …"` A/B main vs branch, metric `timing_s/adv`
(`metric_utils.py:645`). Expected: loop ≈ 420 ms × 2048/8192 ≈ **0.1 s → 0.01 s per step**, invisible in
`timing_s/step` (≈ 1-2 min). The only shape where the saving is visible is N=4096 × L=32k
(`--train-batch-size 512 --max-response-length 32768`): ≈ 5.7 s → 0.4 s per step out of ≥ 20 min of
generation+training (0.4 %). There is no configuration in which this moves the step-time metric maintainers use;
the honest proof is `timing_s/adv` alone, which is exactly what wuxibin89 called "trivial in a whole step".

## 6. Adversarial answers

- **A1** Strongest reason it will not merge: two core maintainers rejected the premise on #2132 — "GAE computation
  time is trivial in a whole step, the readability is more important" (wuxibin89, the perf-PR merger) and "basically
  negligible" (vermouth1992) — and the scan is harder to read than the loop. A first-time author in the `algo` lane
  (7 merged vs 52 open per 07) re-opening a maintainer-rejected premise is the worst possible opening PR.
- **A2** The gain cannot disappear (it removes a Python loop), but it is bounded by shape: ≤ 1 % of a step at the
  largest shapes, 0.1 s at the harness preset, and exactly zero for GRPO/RLOO/REMAX/OPO/GPG users (93+ of the
  shipped scripts). No flag, vLLM version or model size changes that.
- **A3** Correctness: differences ≤ 1e-5 on whitened advantages (the loop's own fp32 rounding; the scan is as good or
  better); masked columns must keep the carry value (they do), and the block cap must hold for small q (it does, to
  q=0.3). Caught by the new parity test plus the two existing multi-turn tests; a GPU e2e run would not notice
  anything, which is the point.
- **A4** Nobody is working on it: only #2132 (stale 16 months, CONFLICTING). People working *next to* it: #8368/#8379
  (reward placement), #7473 (estimator kwargs), shotsan (#7300 RF++ semantics).
- **A5** Half scope: GAE only (q=1 fast path + blockwise), ~90 lines, 1 function + 1 test. Still fails A1.

## 7. Gates, scores, verdict

| gate | result |
|---|---|
| G1 not a duplicate | pass — only #2132 (stale, conflicting, no mask handling); the difference is stated (exact scan, carry, tests) |
| G2 not reserved | **fail in substance** — not reserved, but explicitly deprioritized by wuxibin89 and vermouth1992 on the only prior PR; 07 §8: do not re-open |
| G3 self-contained | pass — 2 files, no deps |
| G4 provable | pass for `timing_s/adv` only; the accepted metric (`timing_s/step`) will not move |
| G5 default-path impact | **fail** — PPO-with-critic / RF++ only (4 + 0 shipped scripts; GAE is the yaml default but every script overrides it); zero for the reference GRPO user |

S1 speed/value **1** (≤ 1 % of a step, 0 for the reference user) · S2 demand **1** (no issue asks; maintainers say
negligible) · S3 certainty **5** (measured 7-22×, exact to fp32 rounding) · S4 merge-ability **2** (small diff with
tests, but the named reviewer is on record against) · S5 novelty **2** (the chunked scan is standard SSM technique).

**Verdict: NO-GO.** Strongest reason: the saving is real but bounded at ≤ 1 % of a step, and the maintainers who
would review it already wrote on #2132 that this time is trivial and readability wins; the campaign's merge pattern
(before/after step numbers, one sitting for wuxibin89) cannot be satisfied here. Do not bundle it with K12 either
(different area, and it would drag a clean fix into a rejected premise). The only way it ships: as a rider on a
*maintainer-requested* substantive change to these two estimators (e.g. if #8367's semantics are reworked inside the
loops), with the parity test as the proof — not as its own PR.
