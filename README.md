# verl contributions from Darkmatter Labs

Seven proposed pull requests for [verl](https://github.com/verl-project/verl) (base: upstream `main` at
`2ab35292`), with the research that chose them, the benchmark harness that measured them and the review
record behind each one. Darkmatter Labs, Inc. (drkm.ai) builds Darkmatter and its `drkm` command; verl is
the reinforcement-learning trainer this work targets.

**Status: none of these PRs has been filed with verl-project/verl yet.** Every branch has CPU tests, GPU
measurements and three rounds of adversarial review behind it, and will be submitted in the order below
after the submitter's own final line-by-line review. The code, tests and review rounds were produced with
AI assistance (Claude) under human direction; each PR body states this, as verl's contribution rules
require. The "AI review" the notes cite is an adversarial review of the packages produced by a second AI model
under the submitter's direction; its findings and reproductions are in `research/review-bundle/`.

## The contributions

| # | Package | Branch | PR title |
|---|---|---|---|
| 1 | `packages/p0` | `contrib/p0-loss-horizon` | [BREAKING][algo, trainer] fix: resolve the seq-mean-token-sum-norm horizon to response_length under the V1 trainer |
| 2 | `packages/p1` | `contrib/p1-lmhead-rows` | [fsdp, perf] feat: project the LM head only for loss rows |
| 3 | `packages/p2` | `contrib/p2-oldlogprob-skip` | [trainer, perf] feat: skip the old-log-prob pass when strictly on-policy |
| 4 | `packages/p3a` | `contrib/p3a-inflight-ttl` | [1/2] [trainer, cfg] feat: in-flight prompt TTL with fresh-uid re-issue (V1) |
| 5 | `packages/p3b` | `contrib/p3b-worker-relaunch` (on p3a) | [2/2] [rollout, trainer, cfg] feat: dead agent-loop worker detection and relaunch (V1) |
| 6 | `packages/p4a` | `contrib/p4a-turn-signals` | [1/2] [tool, rollout, cfg] feat: ToolResponse.done, generated turn spans and a dense per-turn reward row |
| 7 | `packages/p4b` | `contrib/p4b-turn-rloo` (on p4a) | [2/2] [algo, trainer] feat: turn_rloo advantage estimator with gated turn-span plumbing |

`contrib/p3-v1-never-hang` and `contrib/p4-turn-rewards` are combined reference branches equal to the [2/2]
heads; `packages/p3` and `packages/p4` keep the earlier combined records. Every flag is opt-in and default off
except the p0 fix, which changes the normalization of an existing Dr. GRPO setting and is marked `[BREAKING]`.

What the measurements say, with the file each number comes from:

- p1: on 6k-token prompts the training side of a step drops from 40.52 s to 35.68 s (-12 %) and peak memory
  per rank by 4.65 GB; flag off equals main; the GPU identity test passes 10 of 10 cases
  (`packages/p1/PR-BODY.md`, `bench/CROSSCHECK.md`).
- p2: the old-log-prob pass goes from 3.68 s to 0.00 s per step; the training side gains 5.3 % against the
  flag-off arm and 6.6 % against main, less than the pass itself because the entropy metric moves into the
  update pass (`packages/p2/PR-BODY.md`, `bench/CROSSCHECK.md`).
- p3a + p3b: after an agent-loop worker is killed, main crashed 3 of 3 runs; with both knobs on, 12 of 12
  runs completed, with the recovery landing in one or two slower steps (`packages/p3b/PR-BODY.md`).
- p4a + p4b: the dense-reward data path is proven end to end on GPU, but on GSM8K the three arms end within
  the seed spread (pooled validation accuracy at step 30: 0.832 / 0.822 / 0.829), so the PRs make no accuracy
  claim (`packages/p4b/PR-BODY.md`).
- p0: the new invariance test file passes 62 cases on the branch; the same file on main fails 29 and passes
  33 (`packages/p0/PR-BODY.md`).
- Regression: verl's stock GPU unit-test selection is 503 passed / 0 failed / 8 skipped on main and on every
  final branch head (513 on p1, which adds ten GPU cases); the CPU regression set is green on every branch
  (`bench/CROSSCHECK.md`, `packages/README.md`).

## How to read this repository

| Path | What it is |
|---|---|
| `packages/<id>/` | one directory per PR: `PR-TITLE.txt`, `PR-BODY.md` (the text to file), `diff.patch` (against the branch's base), `tests.log`, `CHECKLIST.md`, `NOTES.md`, `REVIEW.md` (adversarial review and fixes), `REVIEW-ROUND2-VERDICT.md` / `REVIEW-ROUND3-VERDICT.md` (AI re-reviews), `review-tests/` (the reviewers' breaking tests), `bench/RESULTS.md` (GPU tables) |
| `packages/README.md` | the index: submission order, final commits, verdicts, GPU evidence |
| `research/` | the research sweep over verl (reports 00 to 11), the vetting of the candidates (`research/vet/`, rubric in `VETTING-RUBRIC.md`), the decision (`DECISION.md`) and the AI review that drove rounds 2 and 3 (`research/review-bundle/`) |
| `bench/` | the Modal benchmark harness (verl's CI image on H100/H200 nodes) and every raw run under `bench/results/<run>/`: `summary.json`, `metrics.jsonl`, `train.log`, `meta.json`, commands and manifests (the per-step validation dumps are omitted for size) |
| `scripts/` | the CPU regression harness and the package audit |

To review a PR, read `packages/<id>/PR-BODY.md`, then compare `verl-main...contrib/<branch>` on GitHub.
`packages/<id>/diff.patch` is the same diff as a file. The branches are plain verl checkouts, so
`git worktree add wt/<id> contrib/<branch>` gives a tree to run.

## Running the tests

The CPU harness mirrors verl's `cpu_unit_tests.yml` on a Mac without a GPU (`research/00-cpu-env.md` has the
replayable install). `requirements-cpu-macos.txt` holds the exact pins, `pytest-cpu.ini` the CI pytest
settings. With a verl checkout of a branch:

```bash
VERL_ROOT=/path/to/checkout scripts/run-cpu-tests-wt.sh <label> 300 tests/trainer tests/workers tests/utils
```

Every new code path has a `*_on_cpu.py` test. The one GPU-only test,
`tests/workers/test_response_only_lm_head_on_gpu.py` (p1), runs with `pytest -s` on a CUDA machine.

## Running the benchmarks

`bench/README.md` documents the harness: `bench/modal_verl.py` is the Modal app (verl's `verlai/verl` CI image,
`uv sync --frozen`, the checkout shipped into the container), `bench/verl_bench.py` launches runs and pulls
results, `bench/summarize.py` prints per-step timings, `bench/analyze_runs.py` audits the p4 arms. You need a
Modal account and `MODAL_PROFILE` set to your own profile. Presets: `baseline-a` (8 GPUs, Qwen3-1.7B GRPO),
`baseline-a-k1` (p2), `long-prompt` (p1), `separate-async-2gpu --chaos` (p3) and `gsm8k-tool` (p4). The
`bench/results/` directories are the raw records those presets produced; `bench/CROSSCHECK.md` reads the A/B
numbers independently from `summary.json`.

## License

verl is licensed under Apache-2.0, and so are the patches, branches and tests here: see `LICENSE`. The
harness, scripts and notes in this repository are released under the same Apache-2.0 license.
`bench/gsm8k_tool/gsm8k_tool.py` is verl's own removed reference tool, kept with its original header.
