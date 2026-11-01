# Reproducing the follow-up findings

Run these commands from this `follow-up-review` folder. The review scripts and logs are included; the source checkouts and existing CPU environment remain at `<repo>`. These are CPU checks and do not launch GPU jobs or download model weights. Some scripts embed the original source paths; adjust those if moving the checkouts.

The P1 and P3 tests intentionally assert the desired safe behavior, so their nonzero exit is the evidence of the current defect. They should pass after an appropriate fix. Run the probes separately from the shipped regression suites.

## P1 comparison must reject NaNs

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/wt/p1 \
  <repo>/.venv-cpu/bin/python -B -m pytest \
  -q -s -p no:cacheprovider evidence/p1/test_gpu_oracle_on_cpu.py
```

Reviewed-head result: **11 failed, 1 passed**. Ten NaN cases and a skipped-step scenario are incorrectly accepted by the real helper; the passing finite case demonstrates the allowed error region. Saved output: `evidence/p1/oracle-tests.log`.

## P2 W4A4 passes guards but changes observer state

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/wt/p2 \
  <repo>/.venv-cpu/bin/python -B evidence/p2/reproduce_qat_skip.py
```

Reviewed-head result: configuration and worker checks accept; module walker returns `[]`; quantizer input `global_amax` is `[1.0, 2.0]` for eval/train. The test substitutes a spy for CUDA quantization, preserving the actual QAT class/observer code; it does not claim a measured GPU log-probability difference. Saved output: `evidence/p2/qat-reproduction.log`.

## P3 zero-retry refill must eventually complete

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/wt/p3b \
  <repo>/.venv-cpu/bin/python -B -m pytest \
  -q -p no:cacheprovider evidence/p3/test_zero_retry_slow_refill_on_cpu.py
```

Repeat with `PYTHONPATH=<repo>/wt/p3a`. Each reviewed-head result: **2 failed, 2 passed**. Cap-zero fails in sync-refill and async; cap-three controls finish. A virtual-time bound prevents an actual hanging process. Saved outputs: `evidence/p3/zero-retry-p3b.log` and `evidence/p3/final-p3a-and-zero-retry.log` (the latter also contains the 49 existing passing tests).

Original P3b regression closure:

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/wt/p3b \
  <repo>/.venv-cpu/bin/python -B -m pytest \
  -q -p no:cacheprovider evidence/p3/test_original_empty_validation_fixed_on_cpu.py
```

Result: **1 passed**. This preserves the original queue-input failure check while binding the new real validation helpers.

## P4 disabled callback contract

Run once against main:

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/verl \
  <repo>/.venv-cpu/bin/python -B evidence/p4/flag_off_reward_repro.py
```

Then against P4b:

```bash
env PYTHONDONTWRITEBYTECODE=1 TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH=<repo>/wt/p4b \
  <repo>/.venv-cpu/bin/python -B evidence/p4/flag_off_reward_repro.py
```

Result: **7.0 on main, 3.0 on P4b**, despite `turn_level_rewards=False`. Saved outputs: `evidence/p4/flag-off-main.log`, `flag-off-p4b.log`.

Read-only P4 raw-run audit:

```bash
<repo>/.venv-cpu/bin/python -B evidence/p4/analyze_runs.py
```

This recomputes the shipped-source digest and counts repeated identical JSON `arguments.answer` values in final validation trajectories. Malformed tool-call JSON is ignored. It requires the original raw `bench/results` and benchmark extras. Saved result: `evidence/p4/audit.json`.

## Other saved tests and provenance

`REVIEW.md` lists the independent regression selections and counts. The P1/P3 detailed reports retain original execution commands and original workspace paths as provenance; the commands above refer to their packaged copies. P2's first targeted log contains a local socket access error; `socket-test.log` is the successful rerun of that one test with local loopback access. It is not a product failure. The broader P3 Ray tests also used local process/socket access.

No source fixes are included. To reproduce the exact review, use the full heads in `reviewed-heads.tsv`; changing the source naturally changes these results.
