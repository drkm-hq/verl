# Proposed fixes and acceptance checks

This is a handoff checklist, not a record of fixes already applied. Preserve the reviewed branch boundaries and keep source changes distinguishable from review evidence.

## P0 loss horizon

- [ ] Require finite positive explicit normalizers.
- [ ] Validate the inferred response-length horizon before assigning it.
- [ ] Cover NaN, infinity, zero, and negative inferred horizons in focused tests.
- [ ] Retain documentation that direct callers bypassing `validate_config` must supply the factor.
- [ ] Retain a clear explanation of the changed normalization for affected jobs.

## P1 loss row LM head

- [ ] Run enabled/disabled bf16 GPU comparisons with identical frozen batches and weights.
- [ ] Compare response log-probabilities, entropy, supported sum-pi-squared outputs, gradients, and one optimizer update.
- [ ] Include entropy-loss and PEFT/checkpointing cases; justify tolerances.
- [ ] Keep opt-in until this evidence supports broader use.
- [ ] Correct the no-device-to-host-sync claim or change the validation placement.
- [ ] Correct measured response-length/K:T wording to approximately 657 / 0.099 for the reported steps.
- [ ] Report training-phase and memory measurements separately from noisy whole-step deltas.

## P2 old log probability skip

- [ ] Validate the effective model after overrides and adapter configuration are accounted for.
- [ ] Reject enabled dropout through `override_config`.
- [ ] Reject loaded adapters with nonzero dropout, including the supplied nonzero-LoRA reproduction.
- [ ] Reject unsupported training-only behavior such as Mixtral router jitter.
- [ ] Resolve unknown/uninspectable model handling consistently with the exactness guarantee.
- [ ] Add all three demonstrated cases as regression tests; do not fix only the list of base-config field names.
- [ ] Keep the performance claim at the supported pooled 5–7% level unless controlled measurements justify more.
- [ ] Replace unsupported same-node wording and stale final-SHA benchmark paragraphs.

## P3 recovery pair

- [ ] Handle empty recovered validation batches before reward evaluation or TransferQueue fetches.
- [ ] Preserve failed-session accuracy denominators and skip row-dependent output operations.
- [ ] Test the real `_validate()` boundary with all groups failing, including a single-worker case.
- [ ] Cover colocated reward handling for an empty validation batch.
- [ ] Accumulate and return validation recovery metrics rather than discarding sample metrics.
- [ ] Drain manager recovery counters after validation so final-validation events are not lost.
- [ ] Test every replacement remaining slow/failing; define a total recovery budget or narrow no-hang claims.
- [ ] Refresh pending-GPU, survival-count, and zero-reissue statements.
- [ ] Recheck the integration plan against PR 8203 before submission.

## P4 turn rewards and estimator

- [ ] Merge tool metadata consistently for DAPO/GDPO custom reward callbacks, or explicitly restrict support.
- [ ] Parameterize metadata/scoring tests over supported reward managers.
- [ ] Run one enabled generation-to-actor-update integration case with actual TransferQueue transport.
- [ ] Include a real tokenizer boundary/junction test.
- [ ] Qualify unbiasedness to independent trajectories, gamma one, and no sample-dependent standard-deviation normalization.
- [ ] Define gamma-less-than-one objective semantics and label heuristic variants clearly.
- [ ] Replace exact-RLOO wording with same-policy-advantages wording; test the different return tensors.
- [ ] Label the 503-pass GPU suite as general regression coverage, since it excludes the new feature tests.
- [ ] Coordinate shared metadata/reward interfaces with PR 8305 and the RFC 7244 author.
- [ ] Consider gating extra-fields fetching and span bookkeeping on actual use.
- [ ] Document the terminal-observation response-budget exception.
- [ ] Validate malformed externally supplied spans.
- [ ] Add a learning-value comparison when making practical algorithm-performance claims.

## Package and submission hygiene

- [ ] Refresh all summaries from one consistent set of final results; avoid stale review verdicts and SHA descriptions.
- [ ] Ensure claims of human review and co-authorship communication are factual before filing.
- [ ] Run the repository-required current duplicate-work checks before proposing a PR.
- [ ] Keep test commands and measured evidence reproducible and distinguish stock regression from enabled-feature tests.
- [ ] Complete the human submitter's line-by-line review and relevant tests.
