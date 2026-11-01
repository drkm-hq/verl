# P4 second independent review

Reviewed p4a d8c4b925 and p4b 988a10bc against previously reviewed f4651857 / 97a43c92. Product trees were read-only. Repository AGENTS.md was read. No GPU run or external writes were performed.

## Verdict

The original P4 callback integration defect is closed. The intended enabled path is substantially better supported now: shared callback metadata, strict span validation, disabled-path gating, qualified math claims, real Qwen tokenizer tests, and actual generation-to-actor-update evidence all check out. No new blocker was found in that intended enabled path.

There is one default-path compatibility caveat from the new shared reward helper, and the new experiment demonstrates a reward-design failure rather than an algorithm-performance gain.

## Remaining scoped compatibility issue

`wt/p4b/verl/experimental/reward_loop/reward_manager/base.py:89-94` unconditionally merges loop metadata and overwrites `num_turns` / `rollout_reward_scores`. DAPO and GDPO now call this helper (`dapo.py:62`, `gdpo.py:45`) even when `turn_level_rewards=False`. Previously they passed the dataset's extra_info without this merge. Rate-limited also acquires the new standardized fields.

Actual DAPO reproduction, default feature OFF: dataset extra_info has num_turns=7, tool_rewards=[123], while the loop has __num_turns__=3 and tool_rewards=[2]. A callback returning extra_info['num_turns'] yields 7 on main and 3 on p4b. The script `flag_off_reward_repro.py` in this directory reproduces it with each tree on PYTHONPATH. This is a scoped P2 backward-compatibility issue for existing custom callbacks relying on such fields; the generic claim that flag-off tensors and metadata are bitwise main is too strong. Either gate/preserve the old callback contract when the feature is off, or explicitly document the unconditional contract change and collision precedence. The intended enabled metadata contract is correct.

## Verified closures

- Shared helper is used by naive, DAPO, GDPO, rate-limited and remote managers; original callback repro is covered by actual manager tests. Dicts and built-in turn metadata lists are copied; non-mapping values are passed through with a warning (GDPO's further mutation still requires a mapping, as on main).
- Both Continuous Token span-update calls are gated by turn_level_rewards. V1 extra_fields retrieval is gated by estimator capability.
- Integral offsets, row counts, bounds, order, nonempty spans and nonoverlap are validated. Built-in producer tests cover real tokenizer merges and response-budget sweeps.
- Documentation correctly qualifies unbiasedness to independent trajectories, gamma=1 and no std normalization; distinguishes actor advantages from raw returns; and documents dropped terminal observation at the preexisting budget boundary.
- New real Qwen tokenizer tests run successfully offline; real model/TransferQueue/actor-update proof is separate from the stock GPU unit suite.

## Enabled GPU evidence independently audited

Raw runs: `<repo>/bench/results/p4-enabled-r2-final`, `p4-control-r2-final`, `p4-curve-enabled-s1`, `p4-curve-control-s1`, `p4-curve-enabled-s2`, `p4-curve-control-s2`.

All six meta.json files identify src_head 988a10bcb6ce7699bd22cbb4d4b1c532843fef40. All report the same manifest digest, `6a7690ea483f3e6b926a831caf58c3730723c3e835b414133df11d289209c2b8`. I independently recomputed it from the current p4b tree plus the three declared benchmark extras; it matches. The dirty suffix is explained by bench_extras, not a hidden product change.

Raw metrics confirm 512 rows with turn spans at every enabled training step, metadata_present and spans_ok equal to 1 at every enabled validation, and no span metric / metadata equal to zero in controls. Curves have 30 training steps each and validations every five steps. The final 6-step arms also satisfy these checks. This closes the previous lack of enabled-path evidence for Qwen/V1/naive reward manager. done_when_correct is false in all these GPU runs, so they do not exercise GPU terminal-done behavior; that behavior is CPU tested. Other tokenizer families and other managers remain outside the GPU integration proof.

## The learning experiment exposes repeated-reward farming

The bench tool (`bench/gsm8k_tool/gsm8k_tool.py`) awards +1 for every correct submitted answer under step_reward=correctness, with done_when_correct=false. It does not make repeated correct submissions free. The enabled arm therefore optimizes a different reward objective from the outcome-only GRPO control.

At final validation (step 30), I parsed every output's `<tool_call>...</tool_call>` JSON payload, converted arguments.answer to a string, and counted a trajectory as repeated when the number of parsed answers exceeded the number of distinct answers. Malformed JSON was ignored. This counting rule and full audit are in `analyze_runs.py`; actual recorded turn_rewards_sum independently confirms multiple rewards were earned.

| Final run | Validation rows | Parsed tool calls | Episodes repeating an identical answer | Episodes earning >=2 step reward |
|---|---:|---:|---:|---:|
| enabled s1 | 1319 | 2227 | 923 | 901 |
| control s1 | 1319 | 1189 | 2 | 0 |
| enabled s2 | 1319 | 2418 | 1085 | 1049 |
| control s2 | 1319 | 1188 | 3 | 0 |

Final accuracies are 0.8264 vs 0.8324 for s1, and 0.8097 vs 0.8241 for s2 (enabled vs control), pooled 0.818 vs 0.828. Mean training step time excluding step 1 is 20.622 vs 16.109 seconds for s1 (+28.0%) and 17.549 vs 15.713 for s2 (+11.7%). These are observed differences, not statistical equivalence or a general verdict on the estimator. Only data.seed changes between the two named seeds; the report should name that precisely.

The current package is honest that it makes no performance claim, but 'same learning rate' / 'same curve within noise' is still too strong without an equivalence analysis. Say no demonstrated accuracy gain and disclose the repeated-submission behavior directly.

Recommended next experiment: award only first-correct or delta improvement, or terminate on correct submission and adapt the outcome scorer to terminal tool success. Compare dense rewards with a stock estimator against dense rewards with turn_rloo, keeping the reward objective fixed. The missing dense+stock-estimator arm means the current experiment cannot attribute behavior to the estimator versus reward shaping. This is a benchmark/design suggestion, not a product-code blocker.

## Tests independently run

165 passed in one session: six current product test files plus all p4a/p4b round2 reviewer tests, including offline real Qwen tokenizer cases and budget sweeps. Full output is `targeted.log`. Python bytecode and pytest caches were disabled to preserve the source trees. No full regression or GPU rerun was performed; saved GPU evidence was audited as above.

## Independent sanity check of root's P2 QAT concern

Source inspection agrees: built-in FSDP QAT applies QATLinear; W4A4 activation quantization updates its input_global_scale only in training mode (`wt/p2/verl/utils/qat/linear.py:356-357`). Every activation observer can change the scale. The P2 config and module guards do not reject QAT W4A4. This is a documented experimental built-in configuration, not arbitrary custom forward code. Recommend explicitly rejecting enabled W4A4 when the skip flag is on; W4A16 lacks this activation-mode branch. No extra QAT execution was performed in this subreview.
