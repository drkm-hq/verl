#!/usr/bin/env bash
# Run verl's CPU unit tests the way .github/workflows/cpu_unit_tests.yml does
# (pytest.ini python_files=*_on_cpu.py, -s --asyncio-mode=auto, TORCH_COMPILE_DISABLE=1,
# TORCHINDUCTOR_DISABLE=1, HF_HUB_ENABLE_HF_TRANSFER=0) but WITHOUT -x, with the ini,
# cache, log and junit kept outside the clone, and with a hard wall-clock cap.
# --continue-on-collection-errors is always on: the megatron-importing files cannot be
# collected without megatron-core (not installable on macOS) and would otherwise abort
# the whole session ("Interrupted: N errors during collection").
#
# Usage: run-cpu-tests.sh <label> <cap_seconds> <pytest target or options...>
#   e.g. run-cpu-tests.sh utils 300 tests/utils
#        run-cpu-tests.sh core 120 tests/trainer/ppo/test_core_algos_on_cpu.py -k gae
# Writes research/logs/<label>.log and research/junit/<label>.xml; exits with pytest's
# status (124 = cap hit, 5 = nothing collected).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLONE=$ROOT/verl
PY=$ROOT/.venv-cpu/bin/python
label=$1; cap=$2; shift 2
mkdir -p "$ROOT/research/logs" "$ROOT/research/junit"
cd "$CLONE" || exit 2
# CI runs under `uv run`, which puts the venv bin first on PATH; tests/utils/test_config_on_cpu.py
# shells out to bare `python3 scripts/print_cfg.py` and needs the same here.
export PATH="$ROOT/.venv-cpu/bin:$PATH"
# tests/utils/dataset fixtures write ./test_data (not gitignored) and test_config_on_cpu.py's
# print_cfg.py subprocess writes hydra run dirs under ./outputs (gitignored), both relative
# to cwd (= the clone, as on CI). Remove them afterwards unless they already existed.
had_test_data=0; [ -e "$CLONE/test_data" ] && had_test_data=1
had_outputs=0;   [ -e "$CLONE/outputs" ]   && had_outputs=1
env TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  "$PY" "$ROOT/scripts/timeout.py" "$cap" \
  "$PY" -m pytest -c "$ROOT/pytest-cpu.ini" --rootdir "$CLONE" -s --asyncio-mode=auto -rfEs \
    --continue-on-collection-errors --junitxml "$ROOT/research/junit/$label.xml" "$@" > "$ROOT/research/logs/$label.log" 2>&1
rc=$?
[ "$had_test_data" = 0 ] && rm -rf "$CLONE/test_data"
[ "$had_outputs" = 0 ]   && rm -rf "$CLONE/outputs"
summary=$(grep -E '^=+ .*(passed|failed|error|skipped|no tests ran).* =+$' "$ROOT/research/logs/$label.log" | tail -1)
echo "[$label] exit=$rc ${summary:-<no pytest summary line; see research/logs/$label.log>}"
exit $rc
