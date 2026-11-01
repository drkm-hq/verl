# P2 round-3 re-review verdict (AI reviewer, 2026-10-29)

Branch contrib/p2-oldlogprob-skip @ 5532f3a4 (17 commits on main @ 2ab35292), wt/p2 read-only; the submitter's probe `reproduce_qat_skip.py` re-run from a scratch copy (PYTHONPATH=wt/p2). Only review-tests/round3/ and this file were added; wt/p2 `git status --porcelain` is empty before and after.

## Finding R2-01 (W4A4 QAT activation observers pass both guards): FIXED

- Probe: exits 1 at `_validate_skip_old_log_prob_config` with `ValueError: ... actor.fsdp_config.qat.mode=w4a4 with qat.enable=True (W4A4 QAT's
  activation observers update the activation quantization scale in train mode only, so the update pass would quantize with different scales than
  the eval-mode old-log-prob pass; W4A16 is weight-only and accepted, but untested with the skip)`, the message the author's test pins and the PR body quotes.
- Real key: the FSDP engine reads `engine_config.qat` (transformer_impl.py:173); `FSDPActorConfig.__post_init__` sets `engine = fsdp_config` for
  fsdp, fsdp2 and fsdp_turbo (verified by instantiating the composed config for each); dp_actor.yaml composes `fsdp_config` from engine/fsdp.yaml.
  `W4A4` upper case is rejected; `apply_qat` lowercases before `QATMode(...)`, which itself refuses `"W4A4"`.
- Observers: `static_minmax` (running max), `memoryless_minmax` (per batch), `minmax` (EMA) all sit under the single `self.training` read in
  linear.py (`_fake_quantize_activation`, line 356); W4A16 has no train-mode branch, nor do quantizer.py / compressed_tensors.py / modelopt.py
  (Megatron-only, a rejected strategy); a test pins the single site.
- Walk: a `QATLinear` subclass under `checkpoint_wrapper` is flagged with its path (by `mode`, not class name); W4A16 is not; QAT on the ref or
  critic only is accepted, correctly (not the actor's forward).

## New defects

- D1 LOW: an Enum-valued `mode` passes the validator. A programmatic DictConfig can carry `QATMode.W4A4` (YAML/CLI cannot):
  `str(QATMode.W4A4).lower()` is `"qatmode.w4a4"`, while `apply_qat` (`.lower()` on the str-enum) and the restore gate (`== "w4a4"`) apply W4A4.
  The walk still refuses it (`mode.value`). Fix: compare `getattr(mode, "value", mode)` in the validator as the walk does.
- Note, no defect: upstream has two QAT blocks; `actor.qat` (actor.yaml's "automatically applied" block) has no reader in the v1 FSDP engine,
  which takes `fsdp_config.qat`. The validator accepts the inert key and rejects the live one, consistent with the engine; one docs sentence would help.

## Tests added: packages/p2/review-tests/round3/test_r3_qat_on_cpu.py

11 tests (two parametrised), 10 passed, 1 failed: `test_enum_valued_mode_is_rejected` (D1). Passing: engine key path for three strategies,
upper-case mode, every observer rejected and flagged, subclass under checkpoint_wrapper plus the single `self.training` site, ref/critic-only and
inert `actor.qat` accepted, the documented message, W4A16 accepted.

## Hygiene and call

`scripts/package-audit.sh`: `p2 contrib/p2-oldlogprob-skip 5532f3a4 17 0 fresh ok none`. tests.log reproduced: eight author files 80 passed;
review-tests rounds 1+2 plus round2/ 46 passed / 1 skipped (round-2's D1-D5 closed by c7031289 / ba0dccf5 / 7f0c53f5). PR-BODY QAT bullet names
key and mode, W4A16 "accepted, but untested", 18 conditions; CHECKLIST row 26 and REVIEW.md section 11 match; "no GPU run of 5532f3a4" stated.

**MERGE-READY on the validation side**: R2-01 is closed at both layers on the key the engine reads, for every spelling YAML can deliver and every
observer; D1 is a one-line comparison behind a second guard that already refuses it. GPU evidence stays the earlier timing runs.

## Lead closure (2026-10-29, after e19e7949)
D1 LOW applied in e19e7949: the validator compares `str(getattr(mode, "value", mode)).lower()`, so an Enum-valued
`QATMode.W4A4` is rejected like the string; the branch test gained the enum case. Reviewer tests rounds 1-3: 57 passed /
1 skipped against wt/p2 @ e19e7949. Verdict: MERGE-READY (stock suite on e19e7949 queued; validation-only commits since
the measured 7f0c53f5).
