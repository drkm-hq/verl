# verl contribution review handoff

Adversarial review of the proposed verl contributions, produced by a second AI model under the submitter's direction, completed on October 29, 2026. This folder contains the original review and a follow-up review of the updated branches, with reproducible defect probes and saved test logs.

**Start with the latest review: [follow-up-review/REVIEW.md](follow-up-review/REVIEW.md).** The original reproduced failures are fixed. The follow-up identifies a P2 W4A4 compatibility hole, a P3 zero-retry refill loop, P1 test-oracle and evidence-description issues, and a P4 disabled-callback contract change. It also documents repeated-reward farming in the new P4 experiment and proposes a better comparison. No new P0 blocker was found.

The follow-up folder includes updated source hashes, new probes, logs, detailed P1/P3/P4 subreviews and a reproduction guide. Root-level review/checklist/reproduction documents below are **historical first-review records**; use the follow-up closure table to distinguish resolved issues from remaining work.

Original review files:

- [REVIEW.md](REVIEW.md): complete findings, evidence, priorities, mathematical qualifications, benchmark critique, and submission recommendations.
- [FIX-CHECKLIST.md](FIX-CHECKLIST.md): concrete proposed changes and acceptance checks for each contribution.
- [REPRODUCTIONS.md](REPRODUCTIONS.md): commands, prerequisites, and expected results for the included probes.
- `evidence/reproductions/`: four independent reproduction scripts.
- `evidence/logs/`: saved P0, P2, and P3 test logs, the P2 train/eval probe log, and the failing P3 validation probe log.
- `reviewed-heads.tsv`: exact reviewed branch heads and diff bases.
- `MANIFEST.sha256`: integrity hashes for the packaged files.

The original contribution directory is `<repo>`. This handoff does not contain a replacement clone or apply any fixes. All seven packaged contribution diffs matched their Git diffs against their stated bases, and their worktrees were clean at verification.

The reproduction scripts require the original source checkouts and CPU environment. Some paths are intentionally absolute and must be adjusted if the source directory moves. Model weights for the tiny LoRA/jitter probes are generated locally by the script; no pretrained model download or GPU is needed.

The first review's selected existing CPU tests totaled **369 passes, one skip, and two expected failures**. Its separate probes demonstrated the then-current defects. The follow-up has its own test table and logs, including successful closure of the original P3 failure and new failing regression expectations. GPU evidence was audited from supplied records; no new GPU job was launched in either review.

This is a review and proposed remediation plan. Nothing was pushed, posted, or sent to upstream maintainers as part of this review.
