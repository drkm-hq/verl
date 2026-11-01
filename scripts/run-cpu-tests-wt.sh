#!/usr/bin/env bash
# run-cpu-tests.sh for a git worktree: identical, but the checkout is $VERL_ROOT (default: the clone)
# and PYTHONPATH points at it so `import verl` resolves to the worktree, not the clone's editable install.
# Usage: VERL_ROOT=/path/to/worktree run-cpu-tests-wt.sh <label> <cap_seconds> <pytest args...>
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLONE=${VERL_ROOT:-$ROOT/verl}
PY=$ROOT/.venv-cpu/bin/python
label=$1; cap=$2; shift 2
mkdir -p "$ROOT/research/logs" "$ROOT/research/junit"
cd "$CLONE" || exit 2
export PATH="$ROOT/.venv-cpu/bin:$PATH"
export PYTHONPATH="$CLONE${PYTHONPATH:+:$PYTHONPATH}"
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
