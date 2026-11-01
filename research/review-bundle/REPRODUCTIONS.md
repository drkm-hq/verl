# Reproduction instructions

These four probes reproduce review findings in the currently reviewed branches. They do not modify the contribution source. They require the existing source directory and CPU environment:

```text
<repo>
<repo>/.venv-cpu/bin/python
```

Python 3.12 and the existing CPU dependencies are used, including PyTorch, Transformers, PEFT, Hydra, OmegaConf, TensorDict, pytest, and TransferQueue. No GPU or model download is needed. Some scripts embed the original source paths; update those paths if the clone is moved. The bundle is a review handoff, not a self-contained replacement environment.

To keep generated fixtures out of the copied handoff, copy the scripts to a scratch directory first. Replace the `cd` path below if the handoff folder has been moved:

```bash
cd <workdir>
review_scratch=$(mktemp -d /tmp/verl-review-repro.XXXXXX)
cp evidence/reproductions/*.py "$review_scratch/"
cd "$review_scratch"
review_python=<repo>/.venv-cpu/bin/python
export PYTHONDONTWRITEBYTECODE=1
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export HF_HUB_OFFLINE=1
```

## P2 effective configuration and adapter checks

```bash
PYTHONPATH=<repo>/wt/p2 "$review_python" p2_dropout_override.py
PYTHONPATH=<repo>/wt/p2 "$review_python" p2_train_eval_gaps.py
```

Expected on reviewed head `2b259b2d`:

- Saved base configuration inspection reports `[]`, and the full skip validator accepts each case.
- The override probe's actual model has attention dropout 0.5 and train/eval maximum log-probability difference approximately 0.153.
- Mixtral router jitter produces a difference approximately 0.000834.
- The real engine loader loads a nonzero LoRA adapter with dropout 0.3, producing a difference approximately 0.0654.

These diagnostic scripts currently exit successfully when they reproduce the defect. A corrected validator should reject the configurations; adapt the tests to assert that rejection during remediation. Exact numerical values may vary with library/BLAS versions. The qualitative acceptance and train/eval mismatch are the findings.

The LoRA/jitter script generates tiny model/config fixtures beside itself. It does not need the model assets from the original scratch run.

## P3 empty validation batch

```bash
PYTHONPATH=<repo>/wt/p3b "$review_python" -m pytest \
  -q -p no:cacheprovider -c /dev/null p3_empty_validation_on_cpu.py
```

Expected on reviewed head `d0da57de`: the regression expectation fails with `ValueError: Received an empty list as keys.` The test uses real trainer/replay methods and the installed TransferQueue client's input guard, with in-memory transport stubs. It does not launch Ray workers or a GPU job.

During remediation, extend this into a passing integration regression that checks failed-session metrics/accounting, including the all-failed batch and colocated reward path.

## P4 custom reward metadata

```bash
PYTHONPATH=<repo>/wt/p4b "$review_python" p4_reward_manager.py
```

Expected on reviewed head `97a43c92`:

```text
NaiveRewardManager: episode_done is present; reward is 1.0
DAPORewardManager: episode_done is absent; reward is 0.0
GDPORewardManager: episode_done is absent; reward is 0.0
```

The script prints the actual dictionaries and scores. All managers receive the same input with `episode_done=True`; the callback reads that signal from `extra_info`. Fix acceptance is consistent documented metadata delivery and scoring across the supported manager set.

## Saved verification logs

`evidence/logs/p0-tests.log`, `p2-tests.log`, and `p3-tests.log` contain the successful selected existing-test runs. `p3_empty_validation.log` contains the independent failing regression expectation, not a failure from those passing selections. `p2_train_eval_gaps.log` records the router-jitter and loaded-adapter probes.

P1's 24 passes and P4's 30 plus 96 passes were reported by the AI reviewers. No additional raw log files for those runs were retained in the shared workspace; this bundle does not invent them. The override and reward-manager outputs are summarized in the review and can be rerun with the included scripts.
