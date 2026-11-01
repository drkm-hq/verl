# verl contribution package: index (after review rounds 2 and 3)

Status 2026-10-29: round 3 complete, every follow-up finding fixed and re-reviewed, every final commit GPU-proven.
Nothing has been filed with verl-project/verl yet; every package directory holds what the PR would carry. Submit only
after a line-by-line review: verl's AGENTS.md requires the human submitter to understand and defend every line and to state AI
assistance in the PR body (each PR-BODY.md does), and the PR template asks for a ci-request message in verl's Slack.

## Round 2: the AI review and what changed
The AI review found four defects that all reproduced here and
asked for stronger evidence and more careful wording. Every package has
REVIEW.md section 10 (finding -> fix commit -> test -> status), NOTES.md "review round 2", a REVIEW-ROUND2-VERDICT.md
written by a fresh adversarial re-review agent (with its breaking tests under review-tests/round2/), and refreshed
diff.patch / tests.log / PR-BODY.md / CHECKLIST.md.

| Finding (independent review) | Fix | Where proven |
|---|---|---|
| P2 validator accepted override_config dropout, loaded LoRA lora_dropout, Mixtral router jitter | effective-config validator (overrides, adapter_config.json incl. non-LoRA peft types, jitter/layerdrop/drop-path keys) + a worker-side module walk at init that raises (dropout modules and rates, DropPath, jitter, BatchNorm); the legacy trainer refuses the flag | the review's two probes now rejected; 77 tests; p2 REVIEW-ROUND2-VERDICT |
| P3b crashed on an empty recovered validation batch; counters discarded | `_validate` skips row-less batches (no colocated pass, no empty fetch), keeps the accuracy accounting, returns the sampling + relaunch counters; wandb generations table tolerates it | the review's probe passes the former crash point; 9 validate tests; p3b verdict |
| P3 no total recovery budget | `trainer.v1.sampler.max_inflight_reissues_per_step` (per step, keyed on global_steps; default max_inflight_reissues x train_batch_size); when spent the TTL is suspended for the step (degrades to waiting, never a livelock); failure and drain paths unaffected | repeated-slow-replacement tests sync + async; p3a verdict |
| P4 DAPO/GDPO reward managers lacked the metadata | one `RewardManagerBase.extra_info_for_reward_fn` helper for naive/dapo/gdpo/rate_limited/remote (copy, non-mapping passthrough) | the review's probe: all three managers 1.0; p4a verdict |
| P1 numerical equivalence unproven; wrong no-sync claim; 636/0.096 | GPU identity test (frozen batch, bf16 flag on/off, fp32 reference arm, gradients, SGD step; plain / checkpointing / LoRA x fsdp / fsdp2); PEFT probe unwrap; body corrected (one `.item()` per micro-batch; 657 / 0.0987) | identity runs: forward bitwise identical, see GPU evidence |
| P0 NaN/inf and zero/negative horizon accepted | finite positive explicit factor (ints bounded to int64), positive-int inferred horizon | 62 tests; p0 verdict |
| P4 math/API over-claims, 503-suite label, span validation, disabled-path cost, terminal observation | docs qualify unbiasedness (independent trajectories, gamma=1, no std-normalisation), "same policy advantages as rloo", strict span validation, extra_fields fetched only for span-aware estimators, budget boundary documented | enumeration tests reproduce the review's numbers; p4b verdict |
| Hygiene | AI statement exact, no "every line reviewed", co-authorship as an offer in the text, stock GPU suite labelled as regression coverage, conservative pooled numbers, duplicate checks re-run 2026-10-29 | every PR-BODY.md |

## Round 3: the AI follow-up review (research/review-bundle/follow-up-review/) and what changed
The follow-up reviewed exactly the round-2 final heads and found four medium items plus an experiment-design flaw; every
one was fixed, re-reviewed by the same re-review agents (REVIEW-ROUND3-VERDICT.md, review-tests/round3/) and the
re-reviews' own findings were closed in turn (REVIEW.md section 11 per package).

| Finding (follow-up review) | Fix | Where proven |
|---|---|---|
| R2-01 P2: W4A4 QAT activation observers update their scale in train mode only; both guards accepted it | config-level rejection of `qat.enable` + `mode=w4a4` (string or Enum; W4A16 accepted, documented untested) and the module walk flags a `QATLinear` with activation quantization | The submitter's probe rejected at the validator; 80 tests; reviewer round-3 11/11; p2 verdict |
| R2-02 P3: `max_inflight_reissues=0` + refill modes livelocked (exhaustion bypassed the step budget) | every TTL-induced discard (re-issue or exhaustion failure) charged to the per-step budget; a step never loses its last finishable group to the TTL (per-poll hold-back, no spend); dead-worker recovery rounds bounded per step by death events, failing loudly; `max_inflight_reissues_per_step=0` rejected; suspension flag reduced as max; validation row-less counts always reported | The submitter's full-loop test in the branch (4/4 both trees); reviewer rounds 2/3/3b; p3a/p3b verdicts |
| R2-03 P1: the GPU comparison passed NaNs; gradient/update conflated; no direct bf16 bound; bitwise forward as a requirement; scope gaps | finiteness gate on every tensor + optimizer-step assertions; direct bf16 bound `0.1 |g_off| + 1e-12` beside the distance-to-fp32 guard; forward by ulp tolerance with the bitwise result reported as an observation; seeded LoRA, `lora-ckpt` variant, two-rank sharded cases; 26-test CPU oracle; rank-local padding | The submitter's oracle 11/12 (the 27 % case now rejected by the direct bound); reviewer round-3 12/12; gpu-identity-p1-8 10/10 |
| R2-04 P4: the shared helper changed the flag-OFF callback contract for DAPO/GDPO/rate-limited | merge only when `turn_level_rewards` is on for those three (naive/remote keep main's always-merge); dataset object passed through untouched as on main; precedence documented | The submitter's probe 7.0 on main and on the branch; reviewer round-3 30 + 82 tests; p4a/p4b verdicts |
| P4 experiment rewarded repeated correct submissions (923/1085 of 1319 episodes) | disclosed in the PR bodies; v2 experiment: first-correct reward with `done_when_correct`, same outcome scorer, arms A outcome GRPO / B dense + REINFORCE++ / C dense + turn_rloo, two data seeds | v2 runs on e1052882: repeats <= 1.4 %; pooled acc at 30: A 0.832 / B 0.822 / C 0.829 (seed spread 1-2 points): no demonstrated gain |

## The contributions, in recommended submission order
| # | Package | Branch @ HEAD (commits) | PR title | Round-2 verdict |
|---|---|---|---|---|
| 1 | p0 | contrib/p0-loss-horizon @ 9c99205c (7) | [BREAKING][algo, trainer] fix: resolve the seq-mean-token-sum-norm horizon to response_length under the V1 trainer | MERGE-READY |
| 2 | p1 | contrib/p1-lmhead-rows @ 8c340e47 (19) | [fsdp, perf] feat: project the LM head only for loss rows | MERGE-READY (identity test 10/10 on GPU incl. two-rank; suite 513/0/8) |
| 3 | p2 | contrib/p2-oldlogprob-skip @ e19e7949 (18) | [trainer, perf] feat: skip the old-log-prob pass when strictly on-policy | MERGE-READY (round 3 closed) |
| 4 | p3a | contrib/p3a-inflight-ttl @ 2bd5c040 (6, on main) | [1/2] [trainer, cfg] feat: in-flight prompt TTL with fresh-uid re-issue (V1) | MERGE-READY (round 3 closed) |
| 5 | p3b | contrib/p3b-worker-relaunch @ 1b397dfd (9, on p3a) | [2/2] [rollout, trainer, cfg] feat: dead agent-loop worker detection and relaunch (V1) | MERGE-READY (round 3 closed; chaos 6/6 on this sha) |
| 6 | p4a | contrib/p4a-turn-signals @ 0d4cd8ca (9, on main) | [1/2] [tool, rollout, cfg] feat: ToolResponse.done, generated turn spans and a dense per-turn reward row | MERGE-READY (round 3 closed) |
| 7 | p4b | contrib/p4b-turn-rloo @ e1052882 (6, on p4a) | [2/2] [algo, trainer] feat: turn_rloo advantage estimator with gated turn-span plumbing | MERGE-READY (v2 experiment done; no performance claim) |
Combined reference branches (re-pointed to the [2/2] heads): contrib/p3-v1-never-hang @ 1b397dfd, contrib/p4-turn-rewards
@ e1052882 (packages/p3, packages/p4 keep the round-1 combined records; packages/p4/bench holds the enabled-run inputs).

CPU regression (scripts/run-cpu-tests-wt.sh, six groups, main baseline 1484 passed / 64 skipped): p0 1546/64/2 xfail,
p1 1510/64 (+ tests/workers 330/40 after round 3), p2 1556/64 (+ touched groups), p3b 1542/64 (round 3b), p4b 1538/64
(+ touched groups), 0 failed everywhere. Pre-commit clean on every commit's files.

## GPU evidence (Modal, verl CI image verlai/verl:uv.cu130.dev5; bench/README.md "Review round 2", raw runs in bench/results/)
- P1 identity (tests/workers/test_response_only_lm_head_on_gpu.py; final run gpu-identity-p1-8 on 8c340e47, H100:2, ten
  cases: {plain, gradient-checkpointing, lora, lora-ckpt} x {fsdp, fsdp2} + two-rank plain fsdp/fsdp2): 10 passed / 0
  failed / 0 skipped. Forward and training log-probs, entropy and loss identical in every case (asserted by an ulp
  tolerance, measured bitwise on H100); the fp32 arms agree on every gradient tensor to <= 5.7e-5; bf16 on-vs-off
  gradient differences at most 1.68e-2 (fsdp) / 3.59e-2 (fsdp2) / 1.44e-2 (two-rank) against the 0.1 direct bound, and
  the distance-to-fp32 ratio D_on/D_off at most 1.22; SGD residual < 0.3 ulp. Earlier runs: d7d820aa/407082ec 6/6 red on
  tolerance rules (forward already bitwise), 363470c1 2/6 (update cancellation, bias noise), cf539906 6/6, e7c0d616 8/10
  (the predicted two-rank padding crash), 1dbc0c07 0/10 (a bool-tensor subtraction in the baseline loop). Round-1
  numbers stand: long prompts training side -12 % (40.5 -> 35.7 s), peak memory -4.65 GB/rank, flag off = main;
  measured K/T 0.0987 (657 response tokens on 5999-token prompts).
- P2: stock suite on 7f0c53f5 503 / 0 / 8. Round-1 numbers stand (old-log-prob pass 3.7 s -> 0; pooled training side -5.3 % vs flag off, -6.6 % vs main;
  the same-window -10.5..-12.1 % comparisons are labelled exploratory; the round-2 commits touch validation, not the measured path).
- P3b chaos (worker SIGKILL after step 2; knobs ON): final 1b397dfd completed 6/6 (kill -> next already-generated step
  2.3 s; recovery landing in steps 6.2 + 11.5 s; dead_workers_relaunched 1; inflight_reissued 7, all dead-worker;
  exhausted 0; no budget or death-event bound engaged); every superseded round-2/3 arm also completed. Cumulative
  knobs-ON record 12/12 (plus 5/5 on f6846af0 in round 1); main 0/3; TTL-only 0/1 (expected: the last worker was dead).
- P4 enabled runs (gsm8k tool loop, Qwen3-1.7B, 8x H100). Round 2 (step_reward = correctness, 988a10bc): every stage
  proven (rows_with_turn_spans 512 on 6/6 steps, metadata delivered, turn_rloo returns 0.95-1.13, actor update ran) but
  the 30-step curves rewarded repeated correct submissions (923/1085 of 1319 episodes), so they cannot compare estimators.
  Round 3 v2 (e1052882; first-correct reward, done_when_correct, same outcome scorer, two data seeds, 30 steps): pooled
  val acc at 0/5/.../30: A outcome-only GRPO 0.668 0.731 0.791 0.811 0.815 0.825 0.832; B dense + reinforce_plus_plus
  0.663 0.752 0.795 0.806 0.812 0.818 0.822; C dense + turn_rloo 0.662 0.757 0.789 0.802 0.812 0.828 0.829; per-seed
  step-30 A 0.835/0.830, B 0.823/0.821, C 0.841/0.818; repeats <= 1.4 % (was 70-82 %); rows_with_turn_spans 512 x 30 on
  both C seeds; `done` exercised on GPU. All arms within the 1-2-point seed spread: no demonstrated accuracy gain for
  either estimator on GSM8K (a single-answer task where per-turn credit has nothing to assign), and turn_rloo is
  indistinguishable from reinforce_plus_plus on the same dense reward. The PRs make no algorithm-performance claim; the
  proven value is the data path (generation -> tool -> reward callback with metadata -> TransferQueue -> estimator ->
  actor update) and the capability for environments with genuine per-step rewards and termination.
- Stock GPU unit-test selection (regression coverage; excludes *_on_cpu.py and tests/experimental, so it does not run
  the new CPU tests): final p0 9c99205c, p2 e19e7949, p3b 1b397dfd, p4b e1052882: 503 / 0 / 8 + torchrun 2/2, identical
  to main; p1 8c340e47: 513 / 0 / 8 + torchrun 2/2 (the stock glob collects the new GPU identity test, so its ten cases
  run inside verl's GPU CI too).

## Decisions (2026-10-29)
| Item | Decision | Why |
|---|---|---|
| p0 `[BREAKING]` prefix; strict-int `response_length` | keep both | the fix changes the normalization of existing Dr. GRPO V1 jobs (both reviews said so); a float horizon is never valid elsewhere in verl |
| p1 opt-in vs default-on; the host sync; the identity criterion | opt-in (default off) in this PR, propose default-on in a follow-up once maintainers have the A/B; keep the one `.item()` per micro-batch for a named error; keep the criterion (bitwise-by-tolerance forward, fp32 agreement, direct bf16 bound + distance-to-fp32 ratio) | a first PR from a new contributor should not change defaults; the sync is negligible next to the forward and buys a clear error |
| p2 one PR or split; the broad name-fragment rule; FSDP-only scope | ONE PR; keep the rule; accept the scope | the validator, the worker walk and the legacy rejection are three halves of one safety property (an accepted flag must be enforced); a false rejection names the setting and costs nothing on the measured path; the entropy-metric speedup stays a follow-up |
| p3 budget default; all-failed sources report no `acc`; land before #8203 or re-target; TTL defaults | keep the default (a uniformly slow step waits; stragglers still hit the per-prompt cap); keep upstream's policy (no synthetic acc); FILE NOW against main with REBASE-PLAN.md attached and re-target only if #8203 merges first; keep `inflight_ttl_s=null`, `max_inflight_reissues=3` | #8203 is open with red checks and no human review; waiting could be weeks; the plan makes the re-target ~1-2 days |
| p4 strict spans; GDPO non-mapping `extra_info`; co-authorship; #8305; fail-fast with a colocated RM; no performance claim | keep all; co-authorship is offered in the PR text to the RFC #7244 author; the #8305 paragraph proposes converging on the field names and waits for the maintainers | strictness was proven safe against the real loop; GDPO's behaviour equals main; the v2 experiment showed no gain, so the PR sells the data path, not a number |
| reviewers to request | p0: tongyx361 + wuxibin89; p1: tardis-key + wuxibin89; p2: wuxibin89 + vermouth1992; p3a/p3b: wuxibin89 + Begunner (#8203); p4a/p4b: wuxibin89 + lijuliana (RFC #7244), cc luzy99 (#8305) | area owners named in each body; wuxibin89 merges ~60 % of verl's PRs |
| submission order and pacing | p0 alone first; after its first review round (or merge) p1, then p2; then p3a and p4a; each [2/2] (p3b, p4b) is filed only after its [1/2] merges, rebased onto main | 7 of verl's last 400 merges came from first-time authors: establish the account with the small certain fix before the perf PRs |
| the ci-request step; duplicate check | the human submitter posts in verl's Slack `ci-request` after each PR; the duplicate-work searches (links in every body) are re-run on the filing day | verl's PR template requires both |

## How to file (nothing below has been run)
1. `gh repo fork verl-project/verl --clone=false` (a public fork under the submitter's account; GitHub cannot fork a public repository privately).
2. For the PR being filed: `git -C verl push <your-fork-url> contrib/<branch>` (or the HTTPS URL).
3. `gh pr create --repo verl-project/verl --base main --head <your-account>:contrib/<branch> --title "$(cat packages/<id>/PR-TITLE.txt)" --body-file packages/<id>/PR-BODY.md --reviewer <logins from the table>`; the body already carries the AI-assistance statement and the duplicate-check links.
4. Post the ci-request message in verl's Slack; answer review comments from the package evidence (REVIEW.md, verdicts, bench/RESULTS.md).
5. Re-run `scripts/package-audit.sh` and the duplicate searches the same day; do not file a [2/2] before its [1/2] merged.

## Per-package contents
PR-TITLE.txt, PR-BODY.md, branch.txt, diff.patch (against the branch's base), tests.log, CHECKLIST.md, NOTES.md,
REVIEW.md (round 1 + section 10 round 2), REVIEW-ROUND2-VERDICT.md, review-tests/ (+ round2/), bench/RESULTS.md,
REBASE-PLAN.md (p3 family). Deferred and rejected candidates are summarized in research/DECISION.md.
