#!/usr/bin/env bash
# The CPU regression set that is green on this Mac (verl commit 2ab35292, 2026-10-28):
# everything .github/workflows/cpu_unit_tests.yml collects, minus
#   (a) the six files that import megatron (megatron-core/triton do not install on macOS;
#       they fail at collection here but run on CI),
#   (b) tests/utils/reward_score/test_sandbox_on_cpu.py (macOS multiprocessing "spawn":
#       prime_math cannot pickle are_equal_under_sympy and prime_code's per-case alarms
#       expire, so the scores are wrong; the file also takes ~4 min),
#   (c) one float32 parametrisation of test_chunked_topk_log_probs_on_cpu (5.72e-6 vs
#       atol 5e-6 on Apple CPU BLAS).
# Groups run one after another, each capped at 300 s by scripts/run-cpu-tests.sh.
#   Usage: run-cpu-regression.sh            # all groups, ~12 min
#          run-cpu-regression.sh utils workers   # a subset (groups below)
# Logs: research/logs/reg-<group>.log, junit: research/junit/reg-<group>.xml.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=$ROOT/scripts/run-cpu-tests.sh
ALL="small utils trainer workers experimental single_controller"
groups=${*:-$ALL}
rc_all=0
for g in $groups; do
  case $g in
    small)   "$RUN" reg-small 300 tests/test_base_config_on_cpu.py tests/test_protocol_on_cpu.py \
               tests/test_protocol_v2_on_cpu.py tests/checkpoint_engine tests/models tests/tools tests/model_merger ;;
    utils)   "$RUN" reg-utils 300 tests/utils \
               --ignore=tests/utils/ckpt/test_megatron_checkpoint_manager_on_cpu.py \
               --ignore=tests/utils/megatron/test_muon_layerwise_bridge_ddp_on_cpu.py \
               --ignore=tests/utils/reward_score/test_sandbox_on_cpu.py ;;
    trainer) "$RUN" reg-trainer 300 tests/trainer ;;
    workers) "$RUN" reg-workers 300 tests/workers \
               --ignore=tests/workers/test_megatron_ddp_config_on_cpu.py \
               --ignore=tests/workers/test_megatron_distillation_only_on_cpu.py \
               --ignore=tests/workers/test_megatron_liger_fused_kernels_on_cpu.py \
               --ignore=tests/workers/test_megatron_value_head_cp_layout_on_cpu.py \
               --deselect 'tests/workers/test_chunked_topk_log_probs_on_cpu.py::test_numerical_equivalence[2-32-152064-64-dtype2-5e-06]' ;;
    experimental)      "$RUN" reg-experimental 300 tests/experimental ;;
    single_controller) "$RUN" reg-single_controller 300 tests/single_controller ;;
    *) echo "unknown group: $g (known: $ALL)"; exit 2 ;;
  esac
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
done
echo "regression set exit=$rc_all (0 = every group green)"
exit $rc_all
