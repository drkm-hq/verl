# Independent review of the verl contributions

**Do not submit all seven contributions as merge-ready yet.** The work contains useful ideas, substantial tests, careful fix cycles, and real benchmark evidence. Three additional integration/correctness issues were reproduced. P0 looks closest to ready, and P1 is a strong next candidate after direct GPU numerical validation.

Reviewed source root: `<repo>`. Review date: October 29, 2026. Exact heads and bases are in `reviewed-heads.tsv`. References below use those source trees and their line numbers at review time.

## Confirmed defects to address before submission

### P2 accepts actors whose training forward differs from the old eval forward

Priority: medium, and a submission blocker for the current exactness guarantee.

Source: `wt/p2/verl/utils/config.py:103–138`, particularly the saved-configuration load at line 116 and the scan at line 138. The validation call is at line 168. The real LoRA adapter loader is `wt/p2/verl/workers/engine/fsdp/transformer_impl.py:356`.

The validator inspects selected fields from the saved base Hugging Face configuration. Training uses the effective model after configuration overrides and adapter loading. It can also contain architecture-specific training-only behavior that is not named dropout.

Three accepted configurations produced different train/eval log-probabilities at identical weights:

| Configuration accepted by the full skip validator | Maximum absolute log-probability difference |
|---|---:|
| Qwen3 with attention dropout enabled through `override_config` | 0.152963161 |
| Dropout-free Qwen3 base with a loaded nonzero LoRA adapter using `lora_dropout=0.3` | 0.065443993 |
| Mixtral with saved `router_jitter_noise=0.1` and attention dropout zero | 0.000833511 |

In each case `_hf_config_dropouts` returned an empty list. The override probe constructs the actual `HFModelConfig`; the LoRA probe uses the real engine's `_build_lora_module`, not a mocked loader. Tiny randomly initialized models suffice to expose the mismatch.

The old pass evaluates the actor in eval mode. The update pass uses train mode. Replacing the old denominator with the training result's detached value forces the ratio to one even when those forwards differ, changing the policy gradient. These cases invalidate the broad promise that the flag is rejected whenever the skip is inexact.

**Required change:** validate the effective actor, including overrides and loaded adapter settings, and reject unsupported training-dependent behavior. Adding more names to the base-config dropout list alone does not resolve all three cases. Add regression cases for the demonstrated configurations. Reconcile the warning-only behavior when inspection fails with the claimed rejection guarantee; a deferred worker-side check is one possible design.

Evidence: `evidence/reproductions/p2_dropout_override.py`, `evidence/reproductions/p2_train_eval_gaps.py`, and `evidence/logs/p2_train_eval_gaps.log`.

### P3b recovery can crash on an empty validation batch

Priority: medium, and a submission blocker for the stated validation recovery behavior.

Source: `wt/p3b/verl/trainer/ppo/v1/replay_buffer.py:427–431`; caller `wt/p3b/verl/trainer/ppo/v1/trainer_base.py:1156–1188`.

The new recovery branch marks dead-worker validation groups failed. If the worker owned every trajectory in the current batch and none was written, `ReplayBuffer.sample()` returns an empty `KVBatchMeta`. This is reachable with one worker or a small final validation batch.

The real `_validate()` then calls `tq.kv_batch_get(keys=[])` at line 1187, raising `ValueError: Received an empty list as keys.` A colocated reward model encounters the empty batch earlier, before that fetch.

The existing recovery test leaves one healthy trajectory and stops at `ReplayBuffer.sample()`. It does not exercise the failing caller boundary. The independent reproduction uses actual trainer/replay methods, an in-memory transport stand-in, and the installed TransferQueue client's real empty-key guard. It is a deterministic integration reproduction, not a live distributed chaos test.

**Required change:** handle empty recovered batches before reward computation and materialization. Preserve expected validation counts so failed sessions retain the intended accuracy accounting. Skip operations that require actual rows. Add a test through `_validate()` in which every group in a batch loses its worker.

Evidence: `evidence/reproductions/p3_empty_validation_on_cpu.py` and `evidence/logs/p3_empty_validation.log`.

### P3b validation recovery counters are discarded

Priority: low to medium; fix alongside the empty-batch issue.

`replay_buffer.py:430` produces `validation/inflight_dead_worker_failed`, but `_validate()` assigns `batch, _ = self.replay_buffer.sample(...)` at `trainer_base.py:1156`, discarding the metrics. Adding an aggregation rule elsewhere does not connect this path. Manager recovery metrics are also drained before validation, so validation relaunch counts can appear on a later training step or be lost after final validation.

**Required change:** accumulate sampling metrics across validation batches and return them with the validation metrics. Drain manager recovery counters after validation as appropriate. This matters because failed validation groups are intentionally scored as zero rather than regenerated.

### P4 metadata is missing from DAPO and GDPO custom reward callbacks

Priority: medium; a scoped integration gap rather than failure of dense turn rewards.

The new contract in `wt/p4b/docs/advance/agent_loop.rst:256–257` and `packages/p4a/PR-BODY.md:18–19` says custom reward functions receive `generated_turn_spans`, `turn_rewards`, and `episode_done` in `extra_info`.

The agent loop transports these as `tool_extra_fields`. Naive, limited, and remote managers merge that field. DAPO and GDPO do not: see `wt/p4b/verl/experimental/reward_loop/reward_manager/dapo.py:62` and `gdpo.py:45`.

A reproduction called the actual managers with identical metadata containing `episode_done=True`. A custom scorer returning `float(extra_info.get('episode_done', False))` produced:

| Manager | Metadata received | Reward |
|---|---|---:|
| Naive | Yes | 1.0 |
| DAPO | No | 0.0 |
| GDPO | No | 0.0 |

Direct access to `extra_info['episode_done']` would fail. Dense tool rewards themselves still flow.

**Required change:** consistently merge tool metadata before invoking custom reward functions, preferably through a common helper, and parameterize tests over supported managers. Alternatively restrict and document the supported manager contract explicitly.

Evidence: `evidence/reproductions/p4_reward_manager.py`.

## Performance evidence and GPU validation

### P1 performance is promising but numerical equivalence still needs a direct check

No confirmed correctness defect was found in P1's final source. The index/scatter path, consumers, configuration inheritance, fallback guards, and prior fixes were inspected. Its 24 shipped tests passed independently.

Raw timing arithmetic checks out. Long-prompt response means for the main runs were approximately 660.10 and 652.88 tokens; enabled runs were 651.94 and 664.02. Full-token counts were within roughly 0.2%. Shorter responses therefore do not explain the measured gain. Baseline enabled runs had slightly more tokens while running faster.

Approximately **12% lower measured training-phase time and 4.65 GB less allocated memory per rank** is a credible result for the tested long-prompt workload. Keep the feature opt-in. Treat observed whole-step changes separately from the optimization's isolated causal effect because generation and other phases vary across independent runs.

Before submission, add a frozen-batch, same-weights bf16 GPU enabled/disabled comparison of response log-probabilities, entropy, sum-pi-squared where supported, gradients, and one optimizer update. Include entropy-loss and PEFT/checkpointing coverage. Use justified numerical tolerances; shape changes need not be bitwise identical on GPU. Similar aggregate metrics from separate stochastic training runs are not an identity proof.

Documentation corrections:

- `packages/p1/PR-BODY.md:74` says there is no device-to-host synchronization. `wt/p1/verl/workers/utils/padding.py:167` converts `prompt_lens.min()` to a Python integer, synchronizing each microbatch. Fix the claim or move validation earlier where appropriate.
- The stated measured response length and K:T ratio should be approximately **657 tokens and 0.099**, rather than 636 and 0.096, for the reported steps 2–6.

### P2 pooled gains are modest and supported by the raw records

Independent recomputation from `bench/results/*/metrics.jsonl`, taking steps 2–6 and then the mean across runs:

| Arm | Mean training-phase sum in seconds |
|---|---:|
| Main, two runs | 15.8295 |
| Flag off, three runs | 15.6156 |
| Flag on, five runs | 14.7895 |

The observed gain is about **5.3% versus flag off** or **6.6% versus main**. The cautious 5–7% claim is supported. The old-log-prob pass disappears, while update-time entropy and output handling add cost elsewhere.

The index's “paired same-node” wording is stronger than the underlying report's same-window comparisons. Similar scheduling windows do not establish that runs used the same physical node. Different phases and separate runs have material variation. Treat stronger 10.5–12.1% subgroup comparisons as exploratory, and use controlled same-allocation/frozen-workload comparisons for a stronger causal claim.

Clean up stale paragraphs in `packages/p2/bench/RESULTS.md`: it says no arms were rerun at the final SHA even though its table includes final-head ON/OFF arms. Aggregate entropy or rounded KL values being similar does not prove unchanged training signals.

### P4's 503-pass GPU count does not exercise its new tests

`bench/results/gpu-tests-p4b-1/command.txt` explicitly ignores `*on_cpu.py` and all `tests/experimental`. None of P4's new tests executed in the advertised 503-pass GPU suite. This is stock regression evidence, not validation of enabled turn rewards, `turn_rloo`, or their full data path.

Require a small enabled run through actual generation, tool execution, reward callback, TransferQueue transport, advantage estimation, and actor update. Include a real tokenizer boundary/junction case. A learning curve is separately useful for demonstrating practical value; it is not a substitute for basic integration coverage.

## P4 mathematical and API qualifications

No new blocker was found in the default RFC construction under independent trajectories, `gamma=1`, and `norm_adv_by_std=False`. The earlier reviewer fixes are present, including generated-token reward placement, signature-gated estimator arguments, rejection of colocated reward models, padded fallback, and registry restoration.

The broader unbiasedness claims need explicit conditions:

- `norm_adv_by_std=True` scales by a standard deviation that includes the current sample's return. The action-independence argument no longer applies. Exact enumeration with rewards `[0,1,4]`, action probabilities `[0.2,0.3,0.5]`, and group size 3 gives unnormalized expected gradient `[-0.46,-0.39,0.85]` and normalized expected gradient approximately `[-0.24526257,-0.18470760,0.42997017]`, not even a common scaling.
- With `gamma<1`, the intended objective must be explicit. The implementation uses discounted return-to-go without an outer `gamma**t` weight for turn t. For two independent Bernoulli turns with rewards equal to their actions and gamma 0.5, the expected implemented score-gradient is `[0.25,0.25]`, while the gradient of `E[r1+0.5*r2]` is `[0.25,0.125]`. Discounting may be an intentional heuristic, but this is not proof of an unbiased gradient for that discounted episodic objective.
- Correlated outputs from one V1 session are outside the independent-trajectory argument, as already acknowledged in the package notes.

Ship the narrow RFC case first or clearly describe the other options and their limits. The exact-gradient test establishes the gamma-one, unnormalized case; checking other options against their own formula does not establish unbiasedness.

“Exactly RLOO” should say **“the same policy advantages as RLOO.”** For one-token rewards 1 and 3, existing RLOO returns advantages `[-2,2]` and returns `[-2,2]`; `turn_rloo` returns the same advantages but raw returns `[1,3]`. This generally changes return metrics rather than actor gradients because the estimator is critic-free. Keep raw returns if intended, but document and test the distinction.

## Additional improvements and scope limits

These items are separate from the reproduced blockers above.

- **P0 finite normalizer validation:** `wt/p0/verl/utils/config.py:112` accepts explicit NaN and positive infinity. Its inferred-value branch at line 126 accepts zero/negative response lengths. Require a finite positive explicit factor and a valid inferred horizon. These cases were independently checked. The main loss-horizon fix and its invariance tests look sound.
- **P0 launcher scope:** the fix relies on `validate_config`. Direct worker/dataclass construction still retains the old null fallback. This limitation is already documented; retain that qualification rather than presenting a universal loss-layer fix. Keep the breaking-behavior explanation for affected Dr. GRPO jobs.
- **P3 total recovery budget:** a per-prompt retry limit does not bound a whole stalled step when exhausted prompts are replaced by new prompts whose budgets reset. This matters in async refill and `sync_refill_failed_groups=True` when every generation exceeds TTL. The existing low-TTL test fabricates successful replacement trajectories and does not establish a general no-livelock guarantee. Consider a step-level time/recovery budget, or narrow “never hang” claims and add the repeated-slow-replacement scenario.
- **P3 evidence cleanup:** refresh pending-GPU wording, obsolete 4/4 survival counts, and the P3a statement that reissue counts stayed zero on every completed run. The final P3b proof records reissues. The proof commit f6846af0 and current d0da57de have the same production code; their difference is documentation.
- **P4 disabled-path cost:** `trainer_base.py:1747–1762` fetches/extracts the entire `extra_fields` column for every algorithm. Gate this work by estimator capability where feasible. Span bookkeeping also runs at merge sites when the feature is disabled.
- **P4 final observation:** the response-budget boundary at `tool_agent_loop.py:456–457` can exit before applying the context merge, after step reward and done have been recorded. The terminal observation may therefore be absent from decoded text used by an outcome scorer. Document this exception or expose terminal observation explicitly.
- **P4 external span validation:** validate custom spans as ordered, nonnegative, nonempty, and nonoverlapping. The built-in producer appears to satisfy those invariants, but malformed external spans can silently distort grouping.

## Contribution positioning and submission order

| Contribution | Recommendation |
|---|---|
| P0 | Best first submission; small, meaningful correctness fix with good invariance tests. Add defensive finite-value checks. |
| P1 | Strong next candidate after direct GPU numerical validation and documentation cleanup. Keep opt-in. |
| P2 | Fix effective-model validation before submission; keep measured gains conservative. |
| P3a | No new reproduced blocker; preserve recovery-scope limitations. |
| P3b | Fix empty validation batches and missing diagnostics before submission. |
| P4a | Fix callback coverage, complete enabled integration validation, and coordinate shared interfaces. |
| P4b | Qualify mathematical/API claims and complete enabled integration validation. |

The overlap with [PR 8305](https://github.com/verl-project/verl/pull/8305) is broader than span plumbing: it proposes direct turn rewards, input validation, transport, and actor-advantage shaping. The algorithms differ, but their shared interface deserves coordination. [RFC 7244](https://github.com/verl-project/verl/issues/7244) also describes the author's existing reference implementation. Do not describe co-authorship as already offered unless that communication occurred.

Refresh P3's integration plan against [PR 8203](https://github.com/verl-project/verl/pull/8203), which moves ownership of rollout dispatch, sampling, and recovery-related operations into the V1 dataloader actor. Previously recorded CI counts are not a durable basis for deciding whether the old integration target will remain viable.

Shorten PR bodies around the concrete behavior, compatibility conditions, strongest evidence, and actual limitations. Leave statements such as “every line reviewed by the author” only after the human submitter has completed that review. The repository's contribution policy also requires a fresh duplicate-work check before proposing a PR; this review is not a claim that every possible current duplicate was excluded.

## Verification performed

All seven packaged patches matched their Git diffs against the stated bases. The split pairs and relevant callers were inspected. No contribution source was modified by this review.

| Branch area | Independently rerun selected CPU tests |
|---|---|
| P0 | 69 passed, 2 expected failures |
| P1 | 24 passed |
| P2 | 98 passed, 1 skipped |
| P3 | 52 passed |
| P4 | 30 new tests plus 96 packaged review tests passed |
| Total | 369 passed, 1 skipped, 2 expected failures |

The independent P3 empty-validation probe fails as expected and is separate from that passing-test total. P2 and P4 diagnostic scripts successfully demonstrate their mismatches. These results show why passing existing tests does not close the newly found gaps.

The initial restricted P3 run had 23 passes and 29 setup errors because Ray process enumeration was denied. The identical suite passed after automatically approved local process/socket access; those initial errors were environmental, not contribution failures.

GPU commands, raw metrics, reported run SHAs, and recorded chaos results were audited. No new GPU or multi-node chaos jobs were launched. The saved logs included in this folder cover P0, P2, P3, and the logged defect probes; P1/P4 pass counts were reported by the AI reviewers, and no additional raw log file for those runs was retained in the shared workspace.
